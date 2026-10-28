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

import copy
import hashlib
import inspect
import math
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..arguments import ModelArguments, VeOmniArguments
from ..data import build_data_transform
from ..data.data_collator import PostCollator
from ..data.data_transform import dpo_producer_identity
from ..data.dpo_collator import DPOProvenanceCollator
from ..distributed.parallel_state import get_parallel_state, use_parallel_state
from ..distributed.sequence_parallel import gather_outputs
from ..distributed.torch_compile import mark_compile_step_begin
from ..models.model_runtime import VeOmniModelRuntime
from ..ops.batch_invariant import set_batch_invariant_mode
from ..utils import helper, logging
from ..utils.constants import IGNORE_INDEX
from ..utils.device import get_device_type, synchronize
from ..utils.dpo_provenance import (
    DPO_APPROVED_KEY,
    DPO_CALLBACK_METADATA_KEYS,
    DPO_POSTCHECK_KEY,
    DPO_PROOF_KEY,
    DPOInvalidProof,
    DPOPackedProof,
)
from .base import BaseTrainer, VeOmniIter
from .dpo_provenance import next_dpo_microbatches


logger = logging.get_logger(__name__)

_NON_MODEL_KEYS = set()


def _build_dpo_labels_list(
    all_labels: torch.Tensor,
    seq_lens: List[int],
    sp_enabled: bool,
) -> List[torch.Tensor]:
    """Split the packed label tensor into per-segment mask targets.

    DPO packs each preference pair as two adjacent segments
    ``[chosen | rejected]``; multiple pairs are packed together. Aligning
    the mask to per-token log-probs depends on whether SP is active:

    * SP off — ``SequenceParallelCollator`` does not run; labels are
      *unshifted*. Apply the causal shift per segment and pad the trailing
      slot with ``IGNORE_INDEX``.
    * SP on — ``SequenceParallelCollator`` already applied a single global
      shift across the packed sequence, so naive slicing leaves each
      segment's tail label holding the *next* segment's head token
      (chosen tail = rejected head). That position would leak the wrong
      target into ``loss_mask`` and pull cross-segment content into the
      chosen / rejected log-prob sums. Force the trailing slot of every
      segment to ``IGNORE_INDEX`` to mask the boundary.

    Args:
        all_labels: flat ``[sum(seq_lens)]`` label tensor, already gathered
            back from SP ranks by the caller when ``sp_enabled``.
        seq_lens: per-segment token counts, in order.
        sp_enabled: whether the SP path applied a global shift upstream.

    Returns:
        A list of per-segment label tensors, each of length ``seq_lens[i]``,
        with the segment boundary masked to ``IGNORE_INDEX``.
    """
    labels_list: List[torch.Tensor] = []
    offset = 0
    if sp_enabled:
        for sl in seq_lens:
            seg = all_labels[offset : offset + sl].clone()
            seg[-1] = IGNORE_INDEX
            labels_list.append(seg)
            offset += sl
    else:
        for sl in seq_lens:
            seq_labels = all_labels[offset : offset + sl]
            labels_list.append(F.pad(seq_labels[1:], (0, 1), value=IGNORE_INDEX))
            offset += sl
    return labels_list


@dataclass(frozen=True)
class _DPOPairWeighting:
    """Pair counts include the duplicated loss on each sequence-parallel rank."""

    local_counts: tuple[int, ...]
    global_count: int
    group_size: int
    group: Any

    @property
    def backward_scale(self) -> float:
        return self.group_size / self.global_count


def _dpo_pair_weighting(
    micro_batches: list[dict[str, Any]] | None,
    post_forward: PostCollator,
    state: Any,
    provenance_enabled: bool,
    *,
    loader_error: Exception | None = None,
    backward_hook: Any = None,
    vocab_size: Any = None,
) -> _DPOPairWeighting:
    """Agree on pair counts before backward enters any model collective."""
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    group = getattr(state, "fsdp_group", None) if distributed else None
    group_size = dist.get_world_size(group) if distributed else 1
    packet: dict[str, Any] = {"rank": rank, "status": "ok", "error": None}
    local_counts: list[int] = []
    layouts: list[Any] = []
    try:
        if loader_error is not None:
            raise loader_error
        if backward_hook is not None:
            try:
                signature = inspect.signature(backward_hook)
                parameter = signature.parameters.get("pair_normalization")
                if parameter is None or parameter.kind not in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    raise ValueError("missing explicit pair_normalization parameter")
                signature.bind({}, pair_normalization=1.0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "DPO forward_backward_step overrides must accept pair_normalization and apply it to the pair-loss sum"
                ) from exc
        if not isinstance(micro_batches, list) or not micro_batches:
            raise ValueError("DPO optimizer step requires a nonempty list of microbatches")
        vocabulary = vocab_size() if vocab_size is not None else None
        if vocabulary is not None and (type(vocabulary) is not int or vocabulary <= 0):
            raise ValueError("DPO model vocabulary size must be a positive integer")
        for batch in micro_batches:
            if not isinstance(batch, dict):
                raise ValueError("DPO microbatch must be a dictionary")
            if batch.get("padding_flag", False) is not False:
                raise ValueError("DPO does not support duplicated padding microbatches in pair-mean training")
            input_ids, labels, cu_q = batch.get("input_ids"), batch.get("labels"), batch.get("cu_seq_lens_q")
            if (
                not isinstance(input_ids, torch.Tensor)
                or not isinstance(labels, torch.Tensor)
                or input_ids.ndim != 2
                or input_ids.shape[0] != 1
                or input_ids.numel() == 0
                or labels.shape != input_ids.shape
            ):
                raise ValueError("DPO packed input_ids and labels must have the same nonempty [1, local_tokens] shape")
            if input_ids.dtype not in (torch.int32, torch.int64) or labels.dtype not in (torch.int32, torch.int64):
                raise ValueError("DPO input_ids and labels must use integer token IDs")
            if not bool((input_ids >= 0).all()) or not bool(((labels >= 0) | (labels == IGNORE_INDEX)).all()):
                raise ValueError("DPO input or label contains an invalid negative token ID")
            if vocabulary is not None and (
                not bool((input_ids < vocabulary).all())
                or not bool(((labels < vocabulary) | (labels == IGNORE_INDEX)).all())
            ):
                raise ValueError("DPO input or label token ID is outside the model vocabulary")
            if (
                not isinstance(cu_q, torch.Tensor)
                or cu_q.ndim != 1
                or cu_q.dtype not in (torch.int16, torch.int32, torch.int64)
                or cu_q.numel() < 2
            ):
                raise ValueError("DPO cu_seq_lens_q must be a one-dimensional integer boundary tensor")
            boundaries = cu_q.tolist()
            global_tokens = input_ids.numel() * (state.sp_size if getattr(state, "sp_enabled", False) else 1)
            if (
                boundaries[0] != 0
                or boundaries[-1] != global_tokens
                or any(right <= left for left, right in zip(boundaries, boundaries[1:]))
            ):
                raise ValueError("DPO packed boundaries must be increasing and cover all global input tokens")
            tail = batch.get("tail_padding_length")
            if tail is not None:
                if isinstance(tail, torch.Tensor):
                    if tail.ndim != 0 or tail.dtype not in (torch.int16, torch.int32, torch.int64):
                        raise ValueError("DPO tail padding length must be a scalar integer")
                    tail = int(tail)
                if (
                    type(tail) is not int
                    or tail < 0
                    or tail >= global_tokens
                    or global_tokens - tail not in boundaries
                ):
                    raise ValueError("DPO tail padding length must mark a valid packed boundary")
            cu_k = batch.get("cu_seq_lens_k")
            if cu_k is not None and (
                not isinstance(cu_k, torch.Tensor)
                or cu_k.ndim != 1
                or cu_k.dtype not in (torch.int16, torch.int32, torch.int64)
                or cu_k.tolist() != boundaries
            ):
                raise ValueError("DPO self-attention query and key boundaries must match")
            linear_cu = batch.get("linear_attn_cu_seq_lens_q")
            if linear_cu is not None and (
                not isinstance(linear_cu, torch.Tensor)
                or linear_cu.ndim != 1
                or linear_cu.dtype not in (torch.int16, torch.int32, torch.int64)
                or linear_cu.tolist() != boundaries
            ):
                raise ValueError("DPO linear-attention and query boundaries must match")
            maximum = max(right - left for left, right in zip(boundaries, boundaries[1:]))
            for key in ("max_length_q", "max_length_k"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    value = int(value)
                if value is not None and (type(value) is not int or value != maximum):
                    raise ValueError(f"DPO {key} disagrees with the packed boundaries")
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None and (
                not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != (1, global_tokens)
            ):
                raise ValueError("DPO attention mask must cover the global packed tokens")
            if provenance_enabled:
                proof = batch.get(DPO_APPROVED_KEY)
                if not isinstance(proof, DPOPackedProof):
                    raise ValueError("DPO pair counting requires post-callback segment provenance")
                lengths = proof.segment_lengths
            else:
                if tail is None and boundaries[-1] - boundaries[-2] == 1:
                    raise ValueError(
                        "DPO trailing one-token segments are ambiguous without tail padding metadata; "
                        "enable explicit_segment_provenance or supply the exact tail_padding_length"
                    )
                lengths = post_forward.compute_seqlens_func(batch)
            if not lengths or len(lengths) % 2 or any(type(length) is not int or length <= 0 for length in lengths):
                raise ValueError("DPO microbatch must contain nonempty chosen/rejected segment pairs")
            local_counts.append(len(lengths) // 2)
            layouts.append(
                (
                    tuple(lengths),
                    tuple(boundaries),
                    tail,
                    tuple(input_ids.shape),
                    cu_k is not None,
                    linear_cu is not None,
                    attention_mask is not None,
                    batch.get("max_length_q") is not None,
                    batch.get("max_length_k") is not None,
                )
            )
        layout_digest = hashlib.sha256(repr(tuple(layouts)).encode()).hexdigest()
        if getattr(state, "sp_enabled", False):
            if not distributed or getattr(state, "sp_group", None) is None:
                raise ValueError("DPO sequence parallelism requires an initialized process group")
            members = tuple(dist.get_process_group_ranks(state.sp_group))
            if (
                len(members) != state.sp_size
                or not 0 <= state.sp_rank < len(members)
                or members[state.sp_rank] != rank
            ):
                raise ValueError("DPO sequence-parallel group metadata is inconsistent")
        else:
            members = (rank,)
        packet.update(
            {
                "microsteps": len(local_counts),
                "pair_count": sum(local_counts),
                "sp_members": members,
                "layout_digest": layout_digest,
            }
        )
    except StopIteration:
        packet["status"] = "eof"
    except Exception as exc:  # noqa: BLE001 - every rank must agree before forward
        packet["status"] = "error"
        packet["error"] = f"{type(exc).__name__}: {exc}"[:512]
    if distributed:
        packets: list[dict[str, Any]] = [None] * group_size
        dist.all_gather_object(packets, packet, group=group)
    else:
        packets = [packet]
    statuses = {item["status"] for item in packets}
    if statuses == {"eof"}:
        raise StopIteration
    failures = [item for item in packets if item["status"] == "error"]
    if failures:
        raise RuntimeError(f"DPO pair-count preflight failed on rank {failures[0]['rank']}: {failures[0]['error']}")
    if "eof" in statuses:
        raise RuntimeError("DPO ranks reached the end of the data at different times")
    if len({item["microsteps"] for item in packets}) != 1:
        raise RuntimeError("DPO ranks must execute the same number of microbatches per optimizer step")
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for item in packets:
        groups.setdefault(item["sp_members"], []).append(item)
    for members, peers in groups.items():
        if {item["rank"] for item in peers} != set(members) or len({item["layout_digest"] for item in peers}) != 1:
            raise RuntimeError("DPO sequence-parallel ranks disagree on per-microbatch segment boundaries")
    global_count = sum(item["pair_count"] for item in packets)
    if global_count <= 0:
        raise RuntimeError("DPO optimizer step contains no preference pairs")
    return _DPOPairWeighting(tuple(local_counts), global_count, group_size, group)


def _dpo_sync_step_begin_error(error: Exception | None, *, context: str = "step-begin callback") -> None:
    """Prevent a local pre-forward callback failure stranding peers at the next collective."""
    if not dist.is_available() or not dist.is_initialized():
        if error is not None:
            raise error
        return
    packet = None if error is None else f"rank {dist.get_rank()}: {type(error).__name__}: {error}"
    packets = [None] * dist.get_world_size()
    dist.all_gather_object(packets, packet)
    failures = [failure for failure in packets if failure is not None]
    if failures:
        raise RuntimeError(f"DPO {context} failed: " + "; ".join(failures)) from error


def _dpo_check_consistent_across_ranks(value: Any, *, context: str) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    if any(item != values[0] for item in values):
        raise RuntimeError(f"DPO {context} does not match across ranks")


def _dpo_recovery_geometry(args: Any) -> dict[str, Any]:
    train, data, accelerator = args.train, args.data, args.model.accelerator
    loader = data.dataloader
    if loader.num_workers > 0 and loader.in_order is not True:
        raise ValueError("DPO exact recovery requires in-order multi-worker data loading")
    if train.dyn_bsz and train.dyn_bsz_runtime != "main":
        raise ValueError("DPO exact recovery requires main-process dynamic batching")
    if train.dyn_bsz and loader.use_background_prefetcher and not loader.drop_last:
        raise ValueError("DPO exact recovery cannot combine background prefetch with dynamic flush padding")
    fields = {
        "train": (
            train,
            (
                "world_size",
                "micro_batch_size",
                "global_batch_size",
                "gradient_accumulation_steps",
                "dataloader_batch_size",
                "dyn_bsz",
                "dyn_bsz_runtime",
                "dyn_bsz_count_mode",
                "dyn_bsz_physical_overflow_ratio",
                "pad_to_length",
                "bsz_warmup_ratio",
                "bsz_warmup_init_mbtoken",
                "seed",
                "num_train_epochs",
                "max_steps",
            ),
        ),
        "data": (
            data,
            (
                "train_path",
                "train_size",
                "train_sample",
                "dyn_bsz_buffer_size",
                "max_seq_len",
                "data_type",
                "datasets_type",
                "dataset_repeat",
                "multisource_datasets_type",
                "dataset_name",
                "enable_multisource",
                "text_keys",
                "silent_exception",
            ),
        ),
        "loader": (
            loader,
            (
                "num_workers",
                "type",
                "worker_num_threads",
                "prefetch_factor",
                "persistent_workers",
                "in_order",
                "drop_last",
                "shuffle",
                "pin_memory",
                "use_background_prefetcher",
            ),
        ),
        "parallel": (
            accelerator,
            (
                "dp_size",
                "dp_replicate_size",
                "dp_shard_size",
                "ep_size",
                "ep_outside",
                "extra_parallel_sizes",
                "extra_parallel_names",
                "extra_parallel_placement_innermost",
                "tp_size",
                "pp_size",
                "ulysses_size",
                "cp_size",
            ),
        ),
    }
    return {name: {key: copy.deepcopy(getattr(obj, key)) for key in keys} for name, (obj, keys) in fields.items()}


def _dpo_objective_parameters(config: Any) -> dict[str, Any]:
    result = {
        name: getattr(config, name)
        for name in (
            "beta",
            "label_smoothing",
            "reference_free",
            "loss_type",
            "average_log_prob",
            "refer_model_precision",
        )
    }
    result["audited_unpadded_tokenizer_sha256"] = getattr(config, "audited_unpadded_tokenizer_sha256", None)
    for name in ("beta", "label_smoothing"):
        if type(result[name]) not in (int, float) or not math.isfinite(result[name]):
            raise ValueError(f"DPO {name} must be a finite number")
    if result["beta"] <= 0 or not 0 <= result["label_smoothing"] <= 1:
        raise ValueError("DPO beta must be positive and label_smoothing must be between 0 and 1")
    for name in ("reference_free", "average_log_prob"):
        if type(result[name]) is not bool:
            raise ValueError(f"DPO {name} must be a boolean")
    if result["loss_type"] not in ("sigmoid", "ipo"):
        raise ValueError("DPO loss_type must be 'sigmoid' or 'ipo'")
    if result["loss_type"] == "ipo" and result["label_smoothing"] != 0:
        raise ValueError("DPO label_smoothing is unsupported for IPO")
    if result["refer_model_precision"] not in ("float32", "bfloat16"):
        raise ValueError("DPO refer_model_precision must be 'float32' or 'bfloat16'")
    fingerprint = result["audited_unpadded_tokenizer_sha256"]
    if fingerprint is not None and (
        type(fingerprint) is not str
        or len(fingerprint) != 64
        or any(ch not in "0123456789abcdef" for ch in fingerprint)
    ):
        raise ValueError("DPO audited_unpadded_tokenizer_sha256 must be a lowercase SHA-256 digest")
    return result


def _reduce_dpo_pair_metrics(
    loss_sum: float, metric_sums: dict[str, float], weighting: _DPOPairWeighting
) -> tuple[float, dict[str, float]]:
    """Return the same global pair-weighted metrics on every FSDP rank."""
    names = sorted(metric_sums)
    values = [loss_sum, *(metric_sums[name] for name in names)]
    if dist.is_available() and dist.is_initialized():
        backend = str(dist.get_backend(weighting.group)).lower()
        device = "cpu" if backend == "gloo" else get_device_type()
        result = torch.tensor(values, dtype=torch.float32, device=device)
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=weighting.group)
        values = result.cpu().tolist()
    return values[0] / weighting.global_count, {
        name: value / weighting.global_count for name, value in zip(names, values[1:])
    }


# ================================ DPO Arguments ======================================


@dataclass
class DPOConfig:
    """dpo.* — Direct Preference Optimization hyperparameters."""

    beta: float = field(
        default=0.1,
        metadata={"help": "Temperature parameter for the DPO loss. Controls deviation from the reference model."},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Label smoothing for DPO loss. Non-zero values assume noisy preference labels."},
    )
    reference_free: bool = field(
        default=False,
        metadata={"help": "If True, ignore the reference model and use an implicit uniform reference."},
    )
    loss_type: Literal["sigmoid", "ipo"] = field(
        default="sigmoid",
        metadata={"help": "DPO loss variant: 'sigmoid' for standard DPO, 'ipo' for Identity Preference Optimization."},
    )
    average_log_prob: bool = field(
        default=False,
        metadata={"help": "If True, average log probs per token instead of summing."},
    )
    refer_model_precision: Literal["float32", "bfloat16"] = field(
        default="bfloat16",
        metadata={"help": "Precision of the reference model."},
    )
    explicit_segment_provenance: bool = field(
        default=True,
        metadata={"help": "DPO packed-segment proof and rank agreement; required for pair-weighted training."},
    )
    allow_legacy_data_resume: bool = field(
        default=False,
        metadata={"help": "Legacy compatibility field; required provenance cannot be bypassed."},
    )
    allow_legacy_objective_resume: bool = field(
        default=False,
        metadata={"help": "Legacy compatibility field; it cannot bypass the producer-contract checkpoint gate."},
    )
    audited_unpadded_tokenizer_sha256: Optional[str] = field(
        default=None,
        metadata={"help": "Fingerprint of a fast tokenizer explicitly audited to emit unpadded DPO sequences."},
    )


@dataclass
class VeOmniDPOArguments(VeOmniArguments):
    """Root config for DPO training — extends VeOmniArguments with DPO hyperparameters.

    The frozen reference always copies ``model``. A custom reference-model
    config is not supported.
    """

    dpo_config: DPOConfig = field(default_factory=DPOConfig)


class DPOReferenceModelRuntime(VeOmniModelRuntime):
    """Frozen DPO reference: same model build as the policy, then eval.

    Always copies the policy's ``model`` args. A custom reference-model
    config is not supported. Frozen-eval knobs (no LoRA / AMP / recompute /
    compile / activation offload) are applied here, not by rewriting the
    caller's config.

    Init only builds the module. Optimizer, assets, and checkpoint stay
    uncalled — callbacks only ever fan out to the policy.
    """

    def __init__(
        self,
        args: ModelArguments,
        model_name: str = "reference",
        *,
        train,
        torch_dtype: str = "bfloat16",
    ):
        args = copy.deepcopy(args)
        args.lora_config = {}
        args.accelerator.fsdp_config.mixed_precision.enable = False
        args.accelerator.gradient_checkpointing.enable = False
        args.accelerator.torch_compile.enable = False
        # A second PinnedBufferPool on the frozen copy is wasted host RAM.
        # Load knobs (ep_sharded_stream_load / muon_expert_zero_comm /
        # fqn_to_index_mapping) stay: the reference materializes the same HF
        # checkpoint as the policy.
        args.accelerator.offload_config.enable_activation = False
        args.accelerator.offload_config.enable_async_activation = False
        self.args = args
        self.model_name = model_name
        self.train_args = train
        self.model_assets = []
        self._torch_dtype = torch_dtype
        self.setup()
        with use_parallel_state(self.model_name):
            self._build_model()
            self.model.requires_grad_(False)
            self._build_parallelized_model()
            self.model.eval()

    @property
    def skip_hf_weight_load(self) -> bool:
        """A policy resume does not carry reference weights; always materialize HF."""
        return False

    def _build_model(self) -> None:
        from ..models.auto import build_foundation_model

        args = self.args
        logger.info_rank0("Build DPO reference model")
        self.model = build_foundation_model(
            config_path=args.config_path,
            weights_path=args.model_path,
            torch_dtype=self._torch_dtype,
            init_device=args.accelerator.init_device,
            ops_implementation=args.ops_implementation,
            config_kwargs=args.model_config,
        )
        self.model_config = self.model.config


class TextDPOTrainer:
    """Text DPO trainer that composes BaseTrainer with DPO-specific logic."""

    base: BaseTrainer
    policy_model: VeOmniModelRuntime
    reference_model: DPOReferenceModelRuntime

    def __init__(self, args: VeOmniDPOArguments):
        self._dpo_provenance_required = True
        self._dpo_epoch_step = 0
        self._dpo_epoch_complete = False
        self._dpo_pending_epoch_end = None
        self._dpo_pending_loader_state = None
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base.device = self.base._setup(args)  # registers ParallelState("base") before seed
        config_error = None
        try:
            from .callbacks.channel_loss_callback import ChannelLossComputer

            if args.dpo_config.explicit_segment_provenance is not True:
                raise ValueError("DPO pair-weighted training requires explicit_segment_provenance=true")
            if args.train.channel_loss.enable and not hasattr(ChannelLossComputer, "observe_dpo_log_probs"):
                raise ValueError("DPO channel-loss metrics require the per-source channel-loss integration")
            if args.model.accelerator.tp_size != 1 or args.model.accelerator.pp_size != 1:
                raise ValueError("DPO pair-weighted training currently requires tp_size=pp_size=1")
            self._dpo_objective_parameters = _dpo_objective_parameters(args.dpo_config)
            self._dpo_recovery_geometry = _dpo_recovery_geometry(args)
        except Exception as exc:  # noqa: BLE001 - synchronize malformed rank-local configuration
            config_error = exc
        _dpo_sync_step_begin_error(config_error, context="configuration")
        _dpo_check_consistent_across_ranks(self._dpo_objective_parameters, context="objective parameters")
        _dpo_check_consistent_across_ranks(self._dpo_recovery_geometry, context="recovery geometry")
        self.policy_model = self._build_policy_model_runtime()

        with use_parallel_state(self.policy_model.parallel_state):
            self._build_data_transform()
            self.base._build_dataset()
            self.base._build_collate_fn()
            if args.dpo_config.explicit_segment_provenance:
                self.base.collate_fn = DPOProvenanceCollator(
                    self.base.collate_fn, require_ds_idx=args.data.enable_multisource
                )
            self.base._build_dataloader()
            from ..data.dynamic_batching import DynamicBatchSizeDataLoader

            if isinstance(self.base.train_dataloader, DynamicBatchSizeDataLoader):
                self.base.train_dataloader._dpo_disallow_implicit_restart = True
        self._build_postforward()
        self.policy_model._build_lr_scheduler(args.train_steps * args.train.num_train_epochs)
        self.base._build_training_context(self.policy_model)
        self.base._init_callbacks(self)

        self.reference_model = self._build_reference_model_runtime()

    @property
    def model(self):
        """Callbacks bind to this trainer and read ``.model`` — that is the policy."""
        return self.policy_model

    def __getattr__(self, name):
        return getattr(self.base, name)

    def load(self) -> None:
        self.policy_model.load()

    def save_dcp(self, state) -> None:
        self.policy_model.save_dcp(state)

    def save_hf_or_lora(self, state, stage: str = "step_end") -> None:
        self.policy_model.save_hf_or_lora(state, stage=stage)

    def wait_for_pending_save(self) -> None:
        self.policy_model.wait_for_pending_save()

    def save_model_assets(self) -> None:
        self.policy_model.save_model_assets()

    def _build_data_transform(self):
        args: VeOmniDPOArguments = self.base.args
        self._dpo_transform_max_seq_len = args.data.max_seq_len
        provenance_options = {"record_segment_provenance": True} if args.dpo_config.explicit_segment_provenance else {}
        tokenizer_fingerprint = getattr(args.dpo_config, "audited_unpadded_tokenizer_sha256", None)
        if getattr(self, "_dpo_provenance_required", False):
            identity_error = None
            try:
                self._dpo_producer_identity = dpo_producer_identity(
                    self.policy_model.tokenizer, self.policy_model.chat_template, tokenizer_fingerprint
                )
            except Exception as exc:  # noqa: BLE001 - report producer failures consistently across ranks
                identity_error = exc
            _dpo_sync_step_begin_error(identity_error, context="producer attestation")
            _dpo_check_consistent_across_ranks(self._dpo_producer_identity, context="producer identity")
            provenance_options["expected_dpo_producer_identity"] = self._dpo_producer_identity
        if tokenizer_fingerprint is not None:
            provenance_options["audited_unpadded_tokenizer_sha256"] = tokenizer_fingerprint
        self.base.data_transform = build_data_transform(
            "dpo",
            tokenizer=self.policy_model.tokenizer,
            chat_template=self.policy_model.chat_template,
            max_seq_len=args.data.max_seq_len,
            **provenance_options,
        )

    def _build_postforward(self):
        self.post_forward = PostCollator()

    def _dpo_vocabulary_size(self) -> int:
        config = self.policy_model.model_config
        get_text_config = getattr(config, "get_text_config", None)
        text_config = get_text_config() if callable(get_text_config) else getattr(config, "text_config", config)
        vocab_size = getattr(text_config, "vocab_size", None)
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("DPO requires a valid model vocabulary size")
        return vocab_size

    def dpo_provenance_checkpoint_state(self) -> dict[str, Any]:
        return {
            "version": 7,
            "enabled": self.base.args.dpo_config.explicit_segment_provenance,
            "plaintext_transform_version": 1,
            "conversation_transform_version": 1,
            "objective_normalization": "global_pair_mean_v1",
            "producer_contract_version": 1,
            "audited_unpadded_tokenizer_sha256": self._dpo_objective_parameters["audited_unpadded_tokenizer_sha256"],
            "producer_identity": self._dpo_producer_identity,
            "transform_max_seq_len": self._dpo_transform_max_seq_len,
            "objective_parameters": self._dpo_objective_parameters,
            "recovery_geometry": self._dpo_recovery_geometry,
            "steps_per_epoch": self.base.args.train_steps,
        }

    def validate_dpo_provenance_checkpoint(self, saved: Any) -> None:
        current = self.dpo_provenance_checkpoint_state()
        if _dpo_recovery_geometry(self.base.args) != current["recovery_geometry"]:
            raise ValueError("DPO recovery geometry changed after initialization")
        if current["recovery_geometry"]["data"]["dataset_name"] != "mapping":
            raise ValueError(
                "DPO trainer-state recovery requires a mapping dataset with a verified cursor; "
                "iterable and multisource replay are not yet supported"
            )
        if current["enabled"] is not True:
            raise ValueError("DPO checkpoint recovery requires explicit segment provenance")
        actual_identity = dpo_producer_identity(
            self.policy_model.tokenizer,
            self.policy_model.chat_template,
            getattr(self.base.args.dpo_config, "audited_unpadded_tokenizer_sha256", None),
        )
        if actual_identity != current["producer_identity"]:
            raise ValueError("DPO producer identity changed before checkpoint recovery")
        if saved is None:
            raise ValueError(
                "legacy DPO trainer state has no data-semantics version; start from weights only "
                "or perform an independently verified migration to required provenance"
            )
        if (
            type(saved) is not dict
            or set(saved)
            != {
                "version",
                "enabled",
                "plaintext_transform_version",
                "conversation_transform_version",
                "objective_normalization",
                "producer_contract_version",
                "audited_unpadded_tokenizer_sha256",
                "producer_identity",
                "transform_max_seq_len",
                "objective_parameters",
                "recovery_geometry",
                "steps_per_epoch",
            }
            or type(saved["version"]) is not int
            or saved["version"] != 7
            or type(saved["enabled"]) is not bool
            or type(saved["plaintext_transform_version"]) is not int
            or saved["plaintext_transform_version"] != 1
            or type(saved["conversation_transform_version"]) is not int
            or saved["conversation_transform_version"] != 1
            or saved["objective_normalization"] != "global_pair_mean_v1"
            or type(saved["producer_contract_version"]) is not int
            or saved["producer_contract_version"] != 1
            or (
                saved["audited_unpadded_tokenizer_sha256"] is not None
                and type(saved["audited_unpadded_tokenizer_sha256"]) is not str
            )
            or type(saved["producer_identity"]) is not dict
            or type(saved["transform_max_seq_len"]) is not int
            or saved["transform_max_seq_len"] <= 0
            or type(saved["objective_parameters"]) is not dict
            or type(saved["recovery_geometry"]) is not dict
            or type(saved["steps_per_epoch"]) is not int
            or saved["steps_per_epoch"] <= 0
        ):
            raise ValueError("Unsupported DPO data-semantics checkpoint protocol")
        if current != saved:
            raise ValueError(
                "DPO provenance mode must match the saved trainer state; "
                "resume with the same mode or start with weights only"
            )

    def _build_policy_model_runtime(self) -> VeOmniModelRuntime:
        """Build the trainable policy under ParallelState ``"policy"``."""
        return self.base._build_model_runtime(model_name="policy")

    def _build_reference_model_runtime(self) -> DPOReferenceModelRuntime:
        """Build the frozen reference as its own runtime.

        Always copies the policy's ``model``. Custom reference-model config
        is not supported.
        """
        args: VeOmniDPOArguments = self.base.args
        return DPOReferenceModelRuntime(
            args.model,
            "reference",
            train=args.train,
            torch_dtype=args.dpo_config.refer_model_precision,
        )

    def on_step_begin(self, micro_batches=None):
        # Each DPO preference pair is packed as two consecutive causal-LM
        # segments (chosen, rejected) but carries one source metadata entry.
        callback = getattr(self.base, "channel_loss_callback", None)
        repeat_key = (
            "channel_loss_source_repeat"
            if callback is not None and hasattr(callback.computer, "observe_dpo_log_probs")
            else "source_repeat"
        )
        self.base.on_step_begin(micro_batches=micro_batches, **{repeat_key: 2})

    @staticmethod
    def dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        beta: float,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        reference_free: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the DPO/IPO loss for a batch of policy and reference model log probabilities.

        Returns:
            (losses, chosen_rewards, rejected_rewards) -- each of shape (batch_size,).
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if loss_type == "ipo":
            losses = (logits - 1 / (2 * beta)) ** 2
        else:
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
            )

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(self, model: nn.Module, micro_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single forward pass on the packed batch containing chosen+rejected pairs.

        Each DPO sample contributes two consecutive sequences (chosen
        then rejected) to the packed sequence. Activates VeOmni's
        chunked CE log-probs path by passing ``return_log_probs=True``
        to the model — the wrapper installed by
        ``build_foundation_model`` promotes per-token NLL into
        ``output.fused_linear_aux.log_probs`` (actual log-probabilities,
        non-positive) without materializing the ``[B, L, V]`` logits
        tensor that the previous gather-on-logits path OOMed on at long
        context.
        This is the same entry point external integrators (verl) and
        the future PPO trainer use. Even-indexed sequences are
        chosen; odd are rejected.

        Caller must enter ``use_parallel_state`` for *this* model's
        mesh so forward and SP gathers resolve the correct groups.

        Returns:
            (chosen_logps, rejected_logps) each of shape ``(B,)``.
        """
        provenance_enabled = getattr(self, "_dpo_provenance_required", False) or getattr(
            self.base.args.dpo_config, "explicit_segment_provenance", False
        )
        approved = micro_batch.get(DPO_APPROVED_KEY) if provenance_enabled else None
        if provenance_enabled and (
            not isinstance(approved, DPOPackedProof) or micro_batch.get(DPO_POSTCHECK_KEY) != approved
        ):
            raise RuntimeError("DPO provenance must be approved before and after callbacks before model forward")
        model_inputs = {
            k: v
            for k, v in micro_batch.items()
            if k not in _NON_MODEL_KEYS
            and not (provenance_enabled and k in DPO_CALLBACK_METADATA_KEYS | set(approved.source_keys))
            and not (
                provenance_enabled
                and k in (DPO_APPROVED_KEY, DPO_POSTCHECK_KEY, DPO_PROOF_KEY)
                and isinstance(v, (DPOPackedProof, DPOInvalidProof))
            )
        }
        outputs = model(**model_inputs, return_log_probs=True, use_cache=False)
        channel_loss_callback = getattr(self, "channel_loss_callback", None)
        observe = getattr(getattr(channel_loss_callback, "computer", None), "observe_dpo_log_probs", None)
        if observe is not None:
            observe(outputs.fused_linear_aux.log_probs, micro_batch["labels"], micro_batch.get("cu_seq_lens_q"))

        # ``outputs.fused_linear_aux.log_probs`` is shape [1, packed_L]
        # (actual log-probabilities; sign already flipped). PostCollator
        # only knows about ``outputs.logits``, so we replicate its
        # SP-gather + per-seq split inline against the log_probs field.
        # Caller must enter this model's ``use_parallel_state`` so forward and
        # SP gathers resolve the correct groups.
        log_probs_packed = outputs.fused_linear_aux.log_probs.squeeze(0)  # [packed_L]
        seq_lens = (
            list(approved.segment_lengths)
            if isinstance(approved, DPOPackedProof)
            else self.post_forward.compute_seqlens_func(micro_batch)
        )
        sp_enabled = get_parallel_state().sp_enabled
        if sp_enabled:
            sp_group = get_parallel_state().sp_group
            log_probs_packed = gather_outputs(log_probs_packed, gather_dim=0, group=sp_group)
            log_probs_packed = log_probs_packed[: sum(seq_lens)]
        elif isinstance(approved, DPOPackedProof) or micro_batch.get("tail_padding_length") is not None:
            valid_length = sum(seq_lens)
            labels = micro_batch["labels"].reshape(-1)
            tail = (
                approved.collator_tail
                if isinstance(approved, DPOPackedProof)
                else int(micro_batch["tail_padding_length"])
            )
            if log_probs_packed.numel() != valid_length + tail or labels.numel() != log_probs_packed.numel():
                raise ValueError("DPO segments disagree with model log-probs or collator padding")
            log_probs_packed = log_probs_packed[:valid_length]
        log_probs_list = list(log_probs_packed.split(seq_lens, dim=0))

        # Build per-segment label targets aligned to the kernel's per-token
        # log-probs. ``_build_dpo_labels_list`` handles both SP-on (segment
        # boundary masking against the global shift) and SP-off (per-segment
        # causal shift with IGNORE_INDEX trailing pad) — see helper docstring.
        if sp_enabled:
            all_labels = gather_outputs(micro_batch["labels"], gather_dim=-1, group=sp_group)
            all_labels = all_labels.view(-1)[: sum(seq_lens)]
        else:
            all_labels = micro_batch["labels"].view(-1)
        labels_list = _build_dpo_labels_list(all_labels, seq_lens, sp_enabled)

        average_log_prob = getattr(self.base.args, "dpo_config", None) and self.base.args.dpo_config.average_log_prob
        all_logps: List[torch.Tensor] = []
        for seq_log_probs, seq_labels in zip(log_probs_list, labels_list):
            loss_mask = seq_labels != IGNORE_INDEX
            per_token_logps = seq_log_probs.float()  # already true log p; no negation
            if average_log_prob:
                logp = (per_token_logps * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            else:
                logp = (per_token_logps * loss_mask).sum()
            all_logps.append(logp)

        all_logps_t = torch.stack(all_logps)
        return all_logps_t[0::2], all_logps_t[1::2]

    def forward_backward_step(
        self, micro_batch: Dict[str, torch.Tensor], *, pair_normalization: float | None = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        get_registered = getattr(BaseTrainer, "_get_registered_channel_loss_callback", None)
        channel_loss_callback = (
            get_registered(self.base)
            if get_registered is not None
            else getattr(self.base, "channel_loss_callback", None)
        )
        micro_step_context = (
            channel_loss_callback.micro_step_context(self.state, micro_batch)
            if channel_loss_callback is not None
            else nullcontext()
        )
        with micro_step_context:
            args: VeOmniDPOArguments = self.base.args
            dpo_config = args.dpo_config

            micro_batch = self.base.preforward(micro_batch)
            if channel_loss_callback is not None:
                channel_loss_callback.strip_model_inputs(micro_batch)

            with torch.no_grad(), use_parallel_state(self.reference_model.parallel_state):
                ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(self.reference_model, micro_batch)

            channel_forward_context = (
                channel_loss_callback.model_forward_context() if channel_loss_callback is not None else nullcontext()
            )
            with (
                use_parallel_state(self.policy_model.parallel_state),
                self.base.model_fwd_context,
                set_batch_invariant_mode(args.train.enable_batch_invariant_mode),
                channel_forward_context,
            ):
                policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy_model, micro_batch)

            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=dpo_config.beta,
                label_smoothing=dpo_config.label_smoothing,
                loss_type=dpo_config.loss_type,
                reference_free=dpo_config.reference_free,
            )

            loss = losses.mean()

            reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
            loss_dict: Dict[str, torch.Tensor] = {
                "dpo_loss": loss.detach(),
                "chosen_rewards": chosen_rewards.mean().detach(),
                "rejected_rewards": rejected_rewards.mean().detach(),
                "reward_accuracy": reward_accuracies.detach(),
                "reward_margin": (chosen_rewards - rejected_rewards).mean().detach(),
            }

            with (
                use_parallel_state(self.policy_model.parallel_state),
                self.base.model_bwd_context,
                set_batch_invariant_mode(args.train.enable_batch_invariant_mode),
            ):
                backward_loss = loss if pair_normalization is None else losses.sum() * pair_normalization
                backward_loss.backward()

            del micro_batch
            return loss, loss_dict

    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        required = getattr(self, "_dpo_provenance_required", False)
        if required:
            config_error = None
            try:
                if self.base.args.dpo_config.explicit_segment_provenance is not True:
                    raise ValueError("DPO segment provenance cannot be disabled after trainer initialization")
                if _dpo_objective_parameters(self.base.args.dpo_config) != self._dpo_objective_parameters:
                    raise ValueError("DPO objective parameters cannot change after trainer initialization")
            except Exception as exc:  # noqa: BLE001 - synchronize malformed rank-local configuration
                config_error = exc
            _dpo_sync_step_begin_error(config_error, context="configuration")
        provenance_enabled = required or self.base.args.dpo_config.explicit_segment_provenance
        next_step = self.state.global_step + 1

        loader_error = None
        if provenance_enabled:
            micro_batches = next_dpo_microbatches(
                data_iterator,
                self.policy_model.parallel_state,
                self.reference_model.parallel_state,
                next_step,
                self.state.epoch,
                source_catalog=lambda: (
                    tuple(self.environ_meter.multisource_tracker.names)
                    if self.base.args.data.enable_multisource
                    else None
                ),
                vocab_size=self._dpo_vocabulary_size,
            )
        else:
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except Exception as exc:  # noqa: BLE001 - synchronize before callbacks/model collectives
                micro_batches, loader_error = None, exc

        weighting = _dpo_pair_weighting(
            micro_batches,
            self.post_forward,
            self.policy_model.parallel_state,
            provenance_enabled,
            loader_error=loader_error,
            backward_hook=self.forward_backward_step,
            vocab_size=self._dpo_vocabulary_size,
        )
        # An exhausted iterable does not perform an optimizer update. Advance
        # only after all ranks have obtained and validated this step's data.
        self.state.global_step = next_step

        expected_proofs = tuple(batch[DPO_APPROVED_KEY] for batch in micro_batches) if provenance_enabled else None
        step_begin_error = None
        try:
            self.base._reset_async_activation_offload_if_enabled(self.policy_model)
            self.on_step_begin(micro_batches=micro_batches)
            if required and self.base.args.dpo_config.explicit_segment_provenance is not True:
                raise ValueError("DPO segment provenance cannot be disabled during a callback")
            if required and _dpo_objective_parameters(self.base.args.dpo_config) != self._dpo_objective_parameters:
                raise ValueError("DPO objective parameters cannot change during a callback")
        except Exception as exc:  # noqa: BLE001 - report a local callback error to every rank before forwarding
            step_begin_error = exc
        _dpo_sync_step_begin_error(step_begin_error)
        if provenance_enabled:
            next_dpo_microbatches(
                iter([micro_batches]),
                self.policy_model.parallel_state,
                self.reference_model.parallel_state,
                self.state.global_step,
                self.state.epoch,
                after_callbacks=True,
                expected_proofs=expected_proofs,
            )
        else:
            weighting = _dpo_pair_weighting(
                micro_batches,
                self.post_forward,
                self.policy_model.parallel_state,
                False,
                vocab_size=self._dpo_vocabulary_size,
            )
        self.base.sync_before_train_step()

        total_loss = 0.0
        total_loss_dict: Dict[str, float] = defaultdict(float)

        num_micro_steps = len(micro_batches)
        for micro_step, (micro_batch, pair_count) in enumerate(zip(micro_batches, weighting.local_counts)):
            mark_compile_step_begin(getattr(self.policy_model, "_veomni_compile_uses_cuda_graphs", False))
            self.base.model_reshard(micro_step, num_micro_steps, self.policy_model)
            self.base._configure_hsdp_allreduce(micro_step, num_micro_steps, self.policy_model)
            loss, loss_dict = self.forward_backward_step(micro_batch, pair_normalization=weighting.backward_scale)

            total_loss += loss.item() * pair_count
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item() * pair_count

        total_loss, total_loss_dict = _reduce_dpo_pair_metrics(total_loss, total_loss_dict, weighting)

        grad_norm = self.policy_model.clip_grad_norm()

        self.policy_model.optimizer.step()
        self.policy_model.lr_scheduler.step()
        self.policy_model.optimizer.zero_grad()

        self._dpo_epoch_step = getattr(self, "_dpo_epoch_step", 0) + 1
        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    def train(self):
        args: VeOmniDPOArguments = self.base.args
        self.on_train_begin()
        logger.info(
            f"Rank{args.train.local_rank} Start DPO training. "
            f"Start step: {self.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {self.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        if getattr(self, "_dpo_pending_epoch_end", None) is not None:
            self.state.epoch, self._dpo_epoch_step = self._dpo_pending_epoch_end
            self._dpo_epoch_complete = True
            self.on_epoch_end()
            self._dpo_pending_epoch_end = None
            self._dpo_pending_loader_state = None

        for epoch in range(self.start_epoch, args.train.num_train_epochs):
            if hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.state.epoch = epoch
            self._dpo_epoch_step = self.start_step
            self._dpo_epoch_complete = False

            self.on_epoch_begin()

            # Create a batch generator
            self.base.data_iterator = VeOmniIter(
                self.base.train_dataloader, use_background_prefetcher=args.data.dataloader.use_background_prefetcher
            )

            for _ in range(self.start_step, args.train_steps):
                try:
                    self.train_step(self.base.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self._dpo_epoch_complete = True
            self.on_epoch_end()

            self.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
            if args.data.dataloader.use_background_prefetcher:
                self.base.data_iterator.stop()

        self.on_train_end()

        if args.data.dataloader.use_background_prefetcher and hasattr(self.base, "data_iterator"):
            self.base.data_iterator.stop()

        synchronize()

        self.base.destroy_distributed()
