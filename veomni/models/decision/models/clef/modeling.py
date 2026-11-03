import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as functional

from veomni.data.decision.collator import model_features

from ....auto import build_foundation_model
from ....checkpoint.convert import ConvertedCheckpointTensor
from ....checkpoint.weights import init_empty_weights
from ...base import ScoringOutput


HEAD_CONFIG_FILENAME = "joint_head_config.json"
HEAD_SOURCE_FILENAME = "joint_schema_model.py"
HEAD_WEIGHTS_FILENAME = "joint_head.safetensors"


def _load_head_module(model_path: str | os.PathLike[str]) -> ModuleType:
    source_path = Path(model_path) / HEAD_SOURCE_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(f"Clef release is missing {source_path}")

    module_name = f"_veomni_clef_{hashlib.sha1(str(source_path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Clef head implementation from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_head_config(
    model_path: str | os.PathLike[str],
    head_config: Mapping[str, Any] | str | os.PathLike[str] | None,
) -> dict[str, Any]:
    if isinstance(head_config, Mapping):
        if head_config:
            return dict(head_config)
        head_config = None
    config_path = Path(head_config) if head_config is not None else Path(model_path) / HEAD_CONFIG_FILENAME
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _reshape_scalar_parameters(module: torch.nn.Module) -> tuple[str, ...]:
    """Make release-head scalar parameters shardable by composable FSDP2."""
    names = []
    for module_name, child in module.named_modules():
        for local_name, parameter in tuple(child._parameters.items()):
            if parameter is None or parameter.ndim != 0:
                continue
            child._parameters[local_name] = torch.nn.Parameter(
                parameter.reshape(1), requires_grad=parameter.requires_grad
            )
            names.append(".".join(part for part in (module_name, local_name) if part))
    return tuple(names)


@dataclass
class _ClefScalarCheckpointConverter:
    names: frozenset[str]

    def can_handle(self, name: str) -> bool:
        return name in self.names

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor:
        return ConvertedCheckpointTensor(name, tensor.reshape(1))

    def finalize(self):
        return []


class ClefDecisionModel(torch.nn.Module):
    """CLEF's Qwen-style backbone adapter, preserving released parameter names."""

    def __init__(
        self,
        foundation: torch.nn.Module,
        head: torch.nn.Module,
        head_path: str | os.PathLike[str],
        *,
        head_config=None,
        source_dir=None,
        scalar_head_parameters=(),
    ):
        super().__init__()
        self.model = foundation.model
        self.lm_head = foundation.lm_head
        if getattr(foundation, "mtp", None) is not None:
            self.mtp = foundation.mtp
        self.config = foundation.config
        self.head = head
        self.head_path = str(head_path)
        self.head_config = dict(head_config or {})
        self.source_dir = str(source_dir or Path(head_path).parent)
        self._scalar_checkpoint_names = frozenset(f"head.{name}" for name in scalar_head_parameters)
        self._no_split_modules = list(getattr(foundation, "_no_split_modules", ()) or ())
        self.metadata_hook = foundation.get_metadata_collate_func()
        self.position_hook = foundation.get_position_id_func()

        native_keys = {key for key in self.state_dict() if not key.startswith("head.")}
        if native_keys != set(foundation.state_dict()):
            raise ValueError("Decision module must preserve native backbone parameter names")

    @staticmethod
    def _create_checkpoint_tensor_converter(model):
        return _ClefScalarCheckpointConverter(model._scalar_checkpoint_names)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def get_metadata_collate_func(self):
        return self.metadata_hook

    def get_position_id_func(self):
        return self.position_hook

    def get_assets(self):
        from .release import ClefModelAssets

        return [ClefModelAssets(self.head_config, self.source_dir)]

    def get_additional_weight_sources(self):
        return ((self.head_path, "head."),)

    def freeze_model(self):
        visual = getattr(self.model, "visual", None)
        if visual is not None:
            visual.requires_grad_(False)
        if hasattr(self, "mtp"):
            self.mtp.requires_grad_(False)

    def gradient_checkpointing_enable(self, **kwargs):
        backbone = getattr(self.model, "language_model", self.model)
        return backbone.gradient_checkpointing_enable(**kwargs)

    @staticmethod
    def _release_records(meta):
        records = []
        offset = 0
        for count in meta["counts"]:
            questions = []
            for index in range(offset, offset + count):
                questions.append(
                    SimpleNamespace(
                        question_id=str(index),
                        question_type=meta["kinds"][index],
                        question_span=tuple(meta["question_spans"][index]),
                        option_spans=tuple(tuple(span) for span in meta["option_spans"][index]),
                        option_ids=tuple(meta["option_ids"][index]),
                    )
                )
            records.append(SimpleNamespace(questions=tuple(questions)))
            offset += count
        if offset != len(meta["question_spans"]):
            raise ValueError("Decision metadata counts do not match the encoded questions")
        return records

    @staticmethod
    def _pack_release_output(per_record, meta):
        per_question = [logits for record in per_record for logits in record]
        max_options = max(logits.numel() for logits in per_question)
        logits = torch.stack(
            [functional.pad(values, (0, max_options - values.numel()), value=-torch.inf) for values in per_question]
        )
        return ScoringOutput(logits=logits)

    def forward(self, batch):
        def to_device(value, device):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {key: to_device(item, device) for key, item in value.items()}
            return value

        device = self.get_input_embeddings().weight.device
        features = to_device(model_features(batch), device)
        backbone = self.model
        media_keys = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
        if not media_keys.intersection(features) and hasattr(backbone, "language_model"):
            backbone = backbone.language_model
            if features.get("position_ids") is not None and features["position_ids"].ndim == 3:
                # Shared packing uses [batch, axes, tokens]; the text backbone
                # consumes HF's [axes, batch, tokens] MRoPE layout.
                features["position_ids"] = features["position_ids"].transpose(0, 1).contiguous()
        hidden = backbone(**features, use_cache=False, return_dict=True).last_hidden_state
        per_record = []
        offset = 0
        for length, meta in zip(batch["decision_request_lengths"], batch["decision_meta"], strict=True):
            end = offset + length
            # The backbone packs requests, but the head's context pooling and
            # schema attention must see only this request, with local spans.
            per_record.extend(
                self.head(
                    hidden[:, offset:end],
                    features["input_ids"][:, offset:end],
                    features["attention_mask"][:, offset:end],
                    self._release_records(meta),
                    self.get_output_embeddings().weight,
                )
            )
            offset = end
        return self._pack_release_output(per_record, batch["decision_meta"])


def load_local_model(
    model_path: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] | None = None,
    head_config: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    head_path: str | os.PathLike[str] | None = None,
    init_device: str = "meta",
    torch_dtype: str = "bfloat16",
    ops_implementation,
    model_config: Mapping[str, Any] | None = None,
    encoder_data_balance: bool | None = False,
    encoder_data_balance_sorting_algo: str | None = "post_mbs_balancing_greedy_without_pad",
    load_pretrained_weights: bool = True,
) -> ClefDecisionModel:
    model_path = str(model_path)
    foundation = build_foundation_model(
        config_path=str(config_path or model_path),
        weights_path=model_path if load_pretrained_weights else None,
        torch_dtype=torch_dtype,
        init_device=init_device,
        encoder_data_balance=encoder_data_balance,
        encoder_data_balance_sorting_algo=encoder_data_balance_sorting_algo,
        ops_implementation=ops_implementation,
        config_kwargs=dict(model_config or {}),
    )

    text_config = getattr(foundation.config, "text_config", foundation.config)
    if getattr(text_config, "num_experts", 0) or getattr(text_config, "num_local_experts", 0):
        raise ValueError("CLEF currently supports dense backbones only; MoE conversion/parallel hooks are not adapted")
    config = _load_head_config(model_path, head_config)
    hidden_size = text_config.hidden_size
    if config.get("hidden_size") != hidden_size:
        raise ValueError(
            f"Clef head hidden_size={config.get('hidden_size')} does not match backbone hidden_size={hidden_size}"
        )

    head_factory = partial(_load_head_module(model_path).JointSchemaHead, **config)
    if init_device == "meta":
        with init_empty_weights():
            head = head_factory()
        head = head.to(dtype=getattr(torch, torch_dtype))
    else:
        head = head_factory().to(device=init_device, dtype=getattr(torch, torch_dtype))
    scalar_head_parameters = _reshape_scalar_parameters(head)

    resolved_head_path = Path(head_path) if head_path is not None else Path(model_path) / HEAD_WEIGHTS_FILENAME
    if load_pretrained_weights and not resolved_head_path.is_file():
        raise FileNotFoundError(f"Clef release is missing {resolved_head_path}")
    if init_device != "meta" and load_pretrained_weights:
        from safetensors.torch import load_file

        state = load_file(str(resolved_head_path))
        for name in scalar_head_parameters:
            state[name] = state[name].reshape(1)
        head.load_state_dict(state, strict=True)
    return ClefDecisionModel(
        foundation,
        head,
        resolved_head_path,
        head_config=config,
        source_dir=model_path,
        scalar_head_parameters=scalar_head_parameters,
    )
