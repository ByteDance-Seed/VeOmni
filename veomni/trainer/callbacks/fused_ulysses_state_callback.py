"""Lifecycle callback for the ``ws_push`` fused QKV-proj + Ulysses a2a kernel.

Constructs the symmetric-memory state collectively at ``on_train_begin`` and tears
it down at ``on_train_end`` (``cuda.sync → dist.barrier(sp_group) → state.close()``).
The fused kernel reads ``qkv_weight`` directly via ``WsPushDispatch.try_resolve_fused``
each forward, so no per-step cache invalidation is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ...utils.logging import get_logger
from .base import Callback, TrainerState


logger = get_logger(__name__)


if TYPE_CHECKING:
    from ...ops.kernels.fused_ulysses_projection import WsPushStateManager
    from ..base import BaseTrainer


def _resolve_param_dtype(args: Any) -> torch.dtype | None:
    try:
        mp = args.train.accelerator.fsdp_config.mixed_precision
        if getattr(mp, "enable", False):
            return _dtype_from_str(mp.param_dtype)
    except AttributeError:
        pass
    return None


def _dtype_from_str(name: str) -> torch.dtype:
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype string {name!r}")
    return table[name]


def _resolve_attention_shape(model_config: Any) -> dict[str, int]:
    # VLM composite configs (Qwen3-VL / Qwen3-VL-MoE / Qwen2.5-VL / ...) carry LM
    # attention metadata under ``text_config``; top-level vision heads are the wrong source.
    cfg = getattr(model_config, "text_config", None) or model_config

    nheads_q = getattr(cfg, "num_attention_heads", None)
    if nheads_q is None:
        raise ValueError("model.config[.text_config].num_attention_heads is required for ws_push state init")
    nheads_kv = getattr(cfg, "num_key_value_heads", None) or nheads_q

    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(cfg, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot derive head_dim: config has neither head_dim nor hidden_size")
        if hidden_size % nheads_q != 0:
            raise ValueError(f"hidden_size ({hidden_size}) is not divisible by num_attention_heads ({nheads_q})")
        head_dim = hidden_size // nheads_q

    return {
        "nheads_q": int(nheads_q),
        "nheads_k": int(nheads_kv),
        "nheads_v": int(nheads_kv),
        "head_dim": int(head_dim),
    }


class FusedUlyssesStateCallback(Callback):
    """Initialize / teardown the WsPushStateManager across the training run.

    No-op unless ``ops_implementation.ulysses_qkv_projection_implementation == 'ws_push'``,
    ``ulysses_size > 1``, and attention shape is extractable. On failure the callback
    stays inactive and ``async_ulysses_qkv_projection`` keeps falling through to eager.
    """

    def __init__(self, trainer: BaseTrainer) -> None:
        super().__init__(trainer)
        self._manager: WsPushStateManager | None = None
        self._disabled_reason: str | None = None
        self._init_args: dict[str, Any] | None = None

        try:
            impl_name = trainer.args.model.ops_implementation.ulysses_qkv_projection_implementation
        except AttributeError:
            impl_name = "eager"
        if impl_name != "ws_push":
            self._disabled_reason = f"ulysses_qkv_projection_implementation={impl_name!r} (not 'ws_push')"
            return

        try:
            self._init_args = self._prepare_init_kwargs(trainer)
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._init_args = None
            logger.warning_rank0(
                "FusedUlyssesStateCallback disabled at construction: %s. Falling back to eager async-ulysses path.",
                exc,
                exc_info=True,
            )

    @staticmethod
    def _prepare_init_kwargs(trainer: BaseTrainer) -> dict[str, Any]:
        from ...distributed.parallel_state import get_parallel_state

        parallel_state = get_parallel_state()
        if not getattr(parallel_state, "ulysses_enabled", False):
            raise ValueError(
                "ulysses_size must be > 1 to use ws_push; got ulysses_size="
                f"{getattr(parallel_state, 'ulysses_size', 1)}"
            )

        sp_group = parallel_state.sp_group
        if sp_group is None:
            raise ValueError("parallel_state.sp_group is None")

        args = trainer.args
        shape = _resolve_attention_shape(trainer.model_config)
        bs = int(args.train.micro_batch_size)

        max_seq_len = getattr(args.data, "max_seq_len", None)
        if max_seq_len is None:
            raise ValueError("args.data.max_seq_len is required for ws_push init")
        sp_size = int(parallel_state.sp_size)
        if max_seq_len % sp_size != 0:
            raise ValueError(f"data.max_seq_len ({max_seq_len}) is not divisible by sp_size ({sp_size})")
        local_seq = max_seq_len // sp_size

        # Silent-fallback warning: if batch shape != (bs, local_seq, ...), the dispatcher
        # in async_ulysses_qkv_projection drops back to eager — correct numerically, but
        # the symm-mem buffer is unused HBM. Loudly surface the two known violations.
        dyn_bsz = getattr(args.train, "dyn_bsz", False)
        pad_to_length = getattr(args.train, "pad_to_length", None)
        if dyn_bsz:
            logger.warning_rank0(
                "ws_push fused kernel + args.train.dyn_bsz=True: dynamic-batched packs "
                "typically produce ``[1, packed_seq, ...]`` tensors, which do not match "
                "the (bs=%d, local_seq=%d) shape the symm-mem buffer is sized for. "
                "Most or all batches will silently fall back to the eager path. "
                "Set ``dyn_bsz=False`` for the fused kernel to actually fire.",
                bs,
                local_seq,
            )
        elif pad_to_length and pad_to_length != max_seq_len:
            logger.warning_rank0(
                "ws_push: args.train.pad_to_length=%s does not equal args.data.max_seq_len=%d. "
                "The fused kernel will silently fall back to eager whenever batch shape != "
                "(bs=%d, local_seq=%d).",
                pad_to_length,
                max_seq_len,
                bs,
                local_seq,
            )

        return {
            "sp_group": sp_group,
            "device": trainer.device,
            "bs": bs,
            "local_seq": local_seq,
            "dtype": _resolve_param_dtype(args) or torch.bfloat16,
            **shape,
        }

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        import torch.distributed as dist

        from ...distributed.parallel_state import get_parallel_state
        from ...ops.kernels.fused_ulysses_projection import (
            WsPushStateManager,
            set_active_manager,
        )

        # Pre-rendezvous handshake: a disabled rank that early-returned would leave peers
        # waiting forever inside _symm_mem.rendezvous. All ranks must agree before any
        # collective allocation; if any rank is not ready, every rank bails.
        parallel_state = get_parallel_state()
        sp_group = getattr(parallel_state, "sp_group", None)
        if sp_group is None:
            return

        device = self.trainer.device
        local_ready = torch.ones(1, dtype=torch.int32, device=device)
        if self._init_args is None:
            local_ready.zero_()
        dist.all_reduce(local_ready, op=dist.ReduceOp.MIN, group=sp_group)
        if local_ready.item() == 0:
            if self._init_args is not None:
                self._init_args = None
                if self._disabled_reason is None:
                    self._disabled_reason = "peer rank failed _prepare_init_kwargs"
            logger.warning_rank0(
                "FusedUlyssesStateCallback: pre-init handshake failed on at least "
                "one SP rank; all ranks falling back to eager."
            )
            return

        local_ok = torch.ones(1, dtype=torch.int32, device=device)
        local_exc: Exception | None = None
        try:
            self._manager = WsPushStateManager(**self._init_args)
        except Exception as exc:
            local_ok.zero_()
            local_exc = exc

        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN, group=sp_group)
        if local_ok.item() == 0:
            if self._manager is not None:
                try:
                    self._manager.teardown()
                except Exception:
                    pass
                self._manager = None
            self._disabled_reason = str(local_exc) if local_exc is not None else "peer rank failed"
            self._init_args = None
            logger.warning_rank0(
                "FusedUlyssesStateCallback: ws_push init failed on at least one "
                "SP rank (local exc=%r); all ranks falling back to eager.",
                local_exc,
                exc_info=local_exc is not None,
            )
            return

        set_active_manager(self._manager)
        logger.info_rank0(
            "FusedUlyssesStateCallback: WsPushStateManager active (shape=%s)",
            self._manager.shape,
        )

        from ...ops.kernels.fused_ulysses_projection.ws_push import _compute_symm_layout

        world_size = dist.get_world_size(sp_group)
        pack_hidden_local = (
            (self._manager.nheads_q + self._manager.nheads_k + self._manager.nheads_v) // world_size
        ) * self._manager.head_dim
        symm_total_b, _sync_off, peer_out_b = _compute_symm_layout(
            bs=self._manager.bs,
            seq_global=self._manager.local_seq * world_size,
            pack_hidden_local=pack_hidden_local,
            world_size=world_size,
            dtype=self._manager.dtype,
        )
        ptr_buf_b = world_size * 8

        current_alloc_mb = torch.cuda.memory_allocated(device) / 2**20 if torch.cuda.is_available() else float("nan")
        logger.info_rank0(
            "FusedUlyssesStateCallback: symm-mem footprint per rank = %.2f MB "
            "(peer_out=%.2f MB, sync+ptrs=%.4f MB). Current total GPU alloc=%.2f MB.",
            (symm_total_b + ptr_buf_b) / 2**20,
            peer_out_b / 2**20,
            (symm_total_b - peer_out_b + ptr_buf_b) / 2**20,
            current_alloc_mb,
        )

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        from ...ops.kernels.fused_ulysses_projection import clear_active_manager

        if self._manager is not None:
            try:
                self._manager.teardown()
            except Exception as exc:
                logger.warning_rank0(
                    "FusedUlyssesStateCallback: teardown raised %s; releasing references regardless.",
                    exc,
                )
            self._manager = None
        clear_active_manager()

    @property
    def manager(self) -> WsPushStateManager | None:
        return self._manager

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason
