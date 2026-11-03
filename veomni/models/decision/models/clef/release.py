import json
import os
import shutil
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from ....checkpoint.weights import save_model_weights


HEAD_PREFIX = "head."
HEAD_CONFIG_FILENAME = "joint_head_config.json"
HEAD_WEIGHTS_FILENAME = "joint_head.safetensors"
RELEASE_SIDECARS = ("chat_template.jinja", "generation_config.json", "joint_schema_model.py")


@dataclass
class ClefModelAssets:
    head_config: dict
    source_dir: str

    def save_pretrained(self, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / HEAD_CONFIG_FILENAME).write_text(json.dumps(self.head_config, indent=2) + "\n")
        (output / "decision_config.json").write_text(json.dumps({"model": "clef"}) + "\n")
        source = Path(self.source_dir) / "joint_schema_model.py"
        destination = output / "joint_schema_model.py"
        if source.is_file() and source.resolve() != destination.resolve():
            shutil.copy2(source, destination)


def split_clef_release_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    backbone_state = OrderedDict()
    head_state = OrderedDict()
    for name, tensor in state_dict.items():
        if name.startswith(HEAD_PREFIX):
            head_name = name.removeprefix(HEAD_PREFIX)
            if not head_name:
                raise ValueError("Clef head state contains an empty parameter name")
            head_state[head_name] = tensor
        else:
            backbone_state[name] = tensor

    if not head_state:
        raise ValueError("Clef release export requires at least one head.* parameter")
    return backbone_state, head_state


def _resolve_head_config(
    head_config: Mapping[str, Any] | str | os.PathLike[str] | None,
    source_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    if isinstance(head_config, Mapping):
        if head_config:
            return dict(head_config)
        head_config = None
    if head_config is None:
        if source_dir is None:
            raise ValueError("Clef release export requires head_config or source_dir")
        head_config = Path(source_dir) / HEAD_CONFIG_FILENAME
    with open(head_config, encoding="utf-8") as handle:
        return json.load(handle)


def _materialize_head_state(
    state_dict: Mapping[str, torch.Tensor],
    *,
    is_writer: bool,
) -> OrderedDict[str, torch.Tensor]:
    materialized = OrderedDict()
    for name, tensor in state_dict.items():
        value = tensor.data
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        if is_writer:
            materialized[name] = value.detach().cpu().contiguous()
    return materialized


def save_clef_release(
    output_dir: str | os.PathLike[str],
    state_dict: Mapping[str, torch.Tensor],
    *,
    head_config: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    model_assets: Sequence[Any] | None = None,
    source_dir: str | os.PathLike[str] | None = None,
    global_rank: int = 0,
) -> None:
    output_dir = str(output_dir)
    backbone_state, head_state = split_clef_release_state_dict(state_dict)

    save_model_weights(
        output_dir,
        backbone_state,
        global_rank=global_rank,
        model_assets=model_assets,
        save_dtype=None,
    )

    is_writer = global_rank == 0
    materialized_head = _materialize_head_state(head_state, is_writer=is_writer)
    if is_writer:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_file(materialized_head, output_path / HEAD_WEIGHTS_FILENAME)
        (output_path / HEAD_CONFIG_FILENAME).write_text(
            json.dumps(_resolve_head_config(head_config, source_dir), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "decision_config.json").write_text(json.dumps({"model": "clef"}) + "\n", encoding="utf-8")
        if source_dir is not None:
            source_path = Path(source_dir)
            for filename in RELEASE_SIDECARS:
                source_file = source_path / filename
                destination = output_path / filename
                if source_file.is_file() and source_file.resolve() != destination.resolve():
                    shutil.copy2(source_file, destination)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
