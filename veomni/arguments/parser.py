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
from typing import Any, Dict, Literal, Type, TypeVar, Union, get_type_hints

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
                    kwargs["type"] = _string_to_bool
                else:
                    kwargs["type"] = list_item_type
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
                kwargs["choices"] = list(field_type.__args__)
                kwargs["type"] = type(field_type.__args__[0])
            else:
                kwargs["type"] = field_type

            if field_info.metadata and "help" in field_info.metadata:
                kwargs["help"] = field_info.metadata["help"]

            kwargs["default"] = argparse.SUPPRESS

            parser.add_argument(f"--{arg_name}", **kwargs)


# Keys that used to live elsewhere. A config carrying one of these is stale rather
# than merely wrong, so the error names the new location instead of listing every
# valid key. There is no compatibility shim: the old key never takes effect.
_RELOCATED_KEYS = {
    "train.accelerator": "model.accelerator",
    "train.optimizer": "model.optimizer",
    "train.init_device": "model.accelerator.init_device",
    "train.broadcast_model_weights_from_rank0": "model.accelerator.broadcast_model_weights_from_rank0",
    "train.ep_sharded_stream_load": "model.accelerator.ep_sharded_stream_load",
    "train.gradient_checkpointing": "model.accelerator.gradient_checkpointing",
    "train.torch_compile": "model.accelerator.torch_compile",
    "train.chunk_mbs_config": "model.accelerator.chunk_mbs_config",
}


def _reject_unknown_keys(cls: Type[Any], config_dict: Dict[str, Any], path: str) -> None:
    """Fail on config keys the dataclass does not declare.

    Silently dropping them is how a moved knob keeps its old default while the
    config still looks like it sets something.
    """
    known = {field_info.name for field_info in dataclasses.fields(cls)}
    unknown = [key for key in config_dict if key not in known]
    if not unknown:
        return

    problems = []
    for key in unknown:
        dotted = f"{path}.{key}" if path else key
        moved_to = _RELOCATED_KEYS.get(dotted)
        if moved_to:
            problems.append(f"  {dotted} has moved to {moved_to}")
        else:
            problems.append(f"  {dotted} is not a field of {cls.__name__} (known: {', '.join(sorted(known))})")

    raise ValueError("Invalid configuration keys:\n" + "\n".join(problems))


def _instantiate_recursive(cls: Type[T], config_dict: Dict[str, Any], path: str = "") -> T:
    """
    Recursively convert a dictionary into Dataclass instances.
    This triggers __post_init__ validation at every level.
    """
    if not is_dataclass(cls):
        return config_dict

    _reject_unknown_keys(cls, config_dict, path)

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
            child_path = f"{path}.{field_name}" if path else field_name
            field_values[field_name] = _instantiate_recursive(field_type, raw_value, child_path)
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
