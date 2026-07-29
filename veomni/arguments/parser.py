# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import dataclasses
import os
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Callable, Dict, Literal, Sequence, Type, TypeVar, Union, get_type_hints

import yaml


try:
    from hdfs_io import copy, exists, makedirs  # for internal use only
except ImportError:
    from ..utils.hdfs_io import copy, exists, makedirs

from ..utils import helper, logging


logger = logging.get_logger(__name__)

T = TypeVar("T")


def _string_to_bool(value: Union[bool, str]) -> bool:
    """Converts a string representation of truth to True (1) or False (0)."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _is_pasted_yaml_sequence(values: Sequence[str]) -> bool:
    """Whether a ``nargs="+"`` group is a YAML flow sequence copied from a config.

    ``[a, b]`` reaches argparse as ``['[a,', 'b]']``, keeping both the wrapping
    brackets and the separators. Requiring the separators too is what tells a
    paste apart from values that merely contain brackets, such as the glob
    ``data/part-[0-9]`` or the character class ``[qk]_proj``.
    """
    if not values or not values[0].startswith("[") or not values[-1].endswith("]"):
        return False
    return all(value.endswith(",") for value in values[:-1])


class _ListArgumentAction(argparse.Action):
    """Convert list elements, rejecting YAML list syntax on the command line.

    ``nargs="+"`` consumes space-separated values, so a flow sequence or a
    comma-joined value would land in a ``List[str]`` as literal element names --
    silently, since any string is a valid element.
    """

    def __init__(self, *args: Any, item_type: Callable[[str], Any] = str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._item_type = item_type

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        flag = option_string or f"--{self.dest}"
        if _is_pasted_yaml_sequence(values):
            raise argparse.ArgumentError(
                self,
                f"{' '.join(values)!r} is YAML list syntax, not a command-line value. Pass the elements "
                f"space-separated and unbracketed ({flag} first second); an empty list needs a config file",
            )
        converted = []
        for value in values:
            if "," in value:
                raise argparse.ArgumentError(
                    self, f"{value!r} contains a comma; list elements are separated by spaces, not commas"
                )
            try:
                converted.append(self._item_type(value))
            except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
                raise argparse.ArgumentError(self, f"invalid value {value!r}: {exc}") from exc
        setattr(namespace, self.dest, converted)


def _deep_update(source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update the source dictionary with the overrides dictionary.
    This ensures nested dictionaries are merged rather than overwritten.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = _deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


# --- Recursive Argument Generation ---
def _add_arguments_recursive(parser: argparse.ArgumentParser, cls: Type[Any], prefix: str = ""):
    """
    Recursively traverse the Dataclass fields and generate arguments in the format
    --prefix.field.subfield for argparse.
    """
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {}

    for field_info in dataclasses.fields(cls):
        field_name = field_info.name
        arg_name = f"{prefix}.{field_name}" if prefix else field_name

        field_type = type_hints.get(field_name, field_info.type)

        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            args = field_type.__args__
            if type(None) in args:
                field_type = args[0]

        if is_dataclass(field_type):
            _add_arguments_recursive(parser, field_type, prefix=arg_name)
        else:
            kwargs = {}
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                kwargs["choices"] = [e.value for e in field_type]
                kwargs["type"] = type(list(field_type)[0].value)
            elif field_type is bool:
                kwargs["type"] = _string_to_bool
                kwargs["nargs"] = "?"
                kwargs["const"] = True
            # Handle List (Simple handling, no deep recursion for lists of objects)
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                kwargs["nargs"] = "+"
                list_item_type = field_type.__args__[0]
                if list_item_type is bool:
                    list_item_type = _string_to_bool
                # The action converts elements itself, so ``type`` stays unset:
                # argparse applies ``type`` before the action can see the group.
                kwargs["action"] = _ListArgumentAction
                kwargs["item_type"] = list_item_type
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
                kwargs["choices"] = list(field_type.__args__)
                kwargs["type"] = type(field_type.__args__[0])
            else:
                kwargs["type"] = field_type

            if field_info.metadata and "help" in field_info.metadata:
                kwargs["help"] = field_info.metadata["help"]

            kwargs["default"] = argparse.SUPPRESS

            parser.add_argument(f"--{arg_name}", **kwargs)


def _instantiate_recursive(cls: Type[T], config_dict: Dict[str, Any]) -> T:
    """
    Recursively convert a dictionary into Dataclass instances.
    This triggers __post_init__ validation at every level.
    """
    if not is_dataclass(cls):
        return config_dict

    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {}

    field_values = {}
    for field_info in dataclasses.fields(cls):
        field_name = field_info.name

        # If the key is not in the config dict, skip it.
        # The dataclass will use its defined default_factory or default value.
        if field_name not in config_dict:
            continue

        raw_value = config_dict[field_name]

        # Prefer resolved type hint
        field_type = type_hints.get(field_name, field_info.type)

        # Unwrap Optional[T]
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            args = field_type.__args__
            if type(None) in args:
                field_type = args[0]

        # If the field expects a Dataclass and we have a dict, recurse
        if is_dataclass(field_type) and isinstance(raw_value, dict):
            field_values[field_name] = _instantiate_recursive(field_type, raw_value)
        else:
            field_values[field_name] = raw_value

    return cls(**field_values)


# --- Main Entry Point ---
def parse_args(root_class: Type[T]) -> T:
    """
    Parses arguments from both a YAML configuration file and Command Line Arguments.
    CLI arguments override YAML configurations.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("config_file", nargs="?", help="Path to YAML config file")
    _add_arguments_recursive(parser, root_class)
    args = parser.parse_args()

    final_config = {}

    if (
        hasattr(args, "config_file")
        and args.config_file
        and (args.config_file.endswith(".yaml") or args.config_file.endswith(".yml"))
    ):
        with open(args.config_file) as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                final_config = yaml_config

    cli_config = {}
    for key, value in vars(args).items():
        if key == "config_file":
            continue

        keys = key.split(".")
        current_level = cli_config
        for _i, k in enumerate(keys[:-1]):
            if k not in current_level:
                current_level[k] = {}
            current_level = current_level[k]
        current_level[keys[-1]] = value

    final_config = _deep_update(final_config, cli_config)

    return _instantiate_recursive(root_class, final_config)


def save_args(args: T, output_path: str) -> None:
    """
    Saves arguments to a yaml file.

    Args:
        args (dataclass): The arguments object.
        output_path (str): The destination path (supports HDFS if configured).
    """
    if output_path.startswith("hdfs://"):
        local_dir = helper.get_cache_dir()
        remote_dir = output_path
    else:
        logger.warning_once("Recommend to use hdfs path or hdfs_fuse path as the output path.")
        local_dir = output_path
        remote_dir = None

    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, "veomni_cli.yaml")

    # Save as YAML
    with open(local_path, "w") as f:
        f.write(yaml.safe_dump(asdict(args), default_flow_style=False))

    if remote_dir is not None:
        if not exists(remote_dir):
            makedirs(remote_dir)

        remote_path = os.path.join(remote_dir, "veomni_cli.yaml")
        copy(local_path, helper.convert_hdfs_fuse_path(remote_path))
