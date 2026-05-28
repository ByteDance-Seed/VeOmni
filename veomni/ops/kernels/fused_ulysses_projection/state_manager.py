"""Process-wide manager for the ``ws_push`` fused QKV-proj + Ulysses a2a path.

The fused kernel needs a long-lived symmetric-memory buffer (``WsPushState``)
sized at one fixed ``(bs, local_seq, nheads_*, head_dim, dtype)`` tuple. The
``KERNEL_REGISTRY`` mechanism assumes stateless callables, so this op opts
into the ``OpScope.GLOBAL`` pattern (see ``_config.py``) and exposes its
state through the module-level handles below instead of a function pointer.

Lifecycle
---------
* ``apply_global_ops`` — writes ``_active_impl_name`` ("eager" | "ws_push"); no
  buffers touched.
* ``FusedUlyssesStateCallback.on_train_begin`` — constructs the manager
  (collective ``init_ws_push_state``) and ``set_active_manager``.
* ``async_ulysses_qkv_projection`` — when no explicit dispatch is passed,
  resolves one via ``WsPushDispatch.try_resolve_fused(get_active_manager(),
  qkv_weight)``.
* ``FusedUlyssesStateCallback.on_train_end`` — ``manager.teardown()`` (see
  ``WsPushState`` for the teardown contract) then ``clear_active_manager``.

Invariant
---------
``_active_impl_name == "eager"`` ⇒ ``_active_manager is None`` (enforced by
``_eager_side_effect`` → ``clear_active_manager``). The reverse does *not*
hold: between ``apply_global_ops("ws_push")`` and ``on_train_begin`` the
manager is intentionally ``None`` and dispatch falls back to eager.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional, Tuple

import torch


if TYPE_CHECKING:
    from .ws_push import WsPushState


# ---------------------------------------------------------------------------
# Module-level holders
# ---------------------------------------------------------------------------
# ``_active_impl_name``: the GLOBAL OpSpec slot, written by ``apply_global_ops``
# via the ``global_slot`` pointer declared in ``_config.py``. Records the user-
# selected backend name only — no resources attached.
#
# ``_active_manager``: the runtime holder of the symm-mem buffer, published by
# ``FusedUlyssesStateCallback`` and read by transparent dispatch in
# ``async_ulysses.py``. Decoupled from the name above because symm-mem
# allocation requires a live process group, which is built much later than
# config resolution.

_active_impl_name: str = "eager"
_active_manager: Optional["WsPushStateManager"] = None
_lock = threading.Lock()


def get_active_impl_name() -> str:
    """Currently selected dispatch backend (``"eager"`` or ``"ws_push"``)."""
    return _active_impl_name


def set_active_manager(manager: "WsPushStateManager") -> None:
    """Publish a manager so transparent dispatch can pick it up."""
    global _active_manager
    with _lock:
        if _active_manager is not None and _active_manager is not manager:
            raise RuntimeError(
                "Active WsPushStateManager already published; call "
                "clear_active_manager() (or the previous manager's teardown()) "
                "before publishing a new one."
            )
        _active_manager = manager


def get_active_manager() -> Optional["WsPushStateManager"]:
    """Return the currently published manager, or ``None`` if not initialized.

    Lock-free read: reference assignment is atomic under the GIL, so a
    concurrent publish/clear can only flip the result between ``None`` (eager
    fallback) and a fully-constructed manager (``_check_alive`` raises if it
    has been torn down). Write paths still take ``_lock`` so publish and
    clear pair correctly under a future free-threading build.
    """
    return _active_manager


def clear_active_manager() -> None:
    """Reset both the manager handle and the impl-name slot. Idempotent.

    Runs the manager's documented ``cuda.sync → dist.barrier → state.close()``
    teardown contract before dropping the handle so a backend switch (e.g.
    ``_eager_side_effect`` after ``ws_push`` ran) cannot leak symm-mem or
    leave a pending kernel un-drained.
    """
    global _active_manager, _active_impl_name
    with _lock:
        manager = _active_manager
        _active_manager = None
        _active_impl_name = "eager"
    if manager is not None and getattr(manager, "_state", None) is not None:
        try:
            manager.teardown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WsPushStateManager
# ---------------------------------------------------------------------------


class WsPushStateManager:
    """Owns one ``WsPushState`` — the fixed-shape symm-mem buffer holder.

    Single-shape by design: the symm-mem buffer is sized at
    ``init_ws_push_state`` time and cannot accommodate different
    ``(bs, local_seq, heads, dtype)`` tuples. Callers must consult
    :py:meth:`is_compatible` and fall back to eager on mismatch; truly
    dynamic-shape models are out of scope.

    The manager holds no per-layer weight cache. ``WsPushDispatch.try_resolve_fused``
    detaches the caller's ``FusedQKVLinear.weight`` directly
    (one fused ``nn.Parameter`` → one FSDP2 shard, one optimizer state)
    so there is no value snapshot to invalidate at ``optimizer.step`` boundaries — the
    ``id(q_weight)``/data_ptr fingerprint cache that earlier three-Linear
    versions kept is structurally unnecessary on the fused path.
    """

    def __init__(
        self,
        *,
        sp_group,
        device: torch.device,
        bs: int,
        local_seq: int,
        nheads_q: int,
        nheads_k: int,
        nheads_v: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        # Local imports keep NPU / non-symm-mem hosts importable.
        import torch.distributed as dist

        from .ws_push import choose_tile_config, init_ws_push_state

        self.sp_group = sp_group
        self.device = device
        self.bs = bs
        self.local_seq = local_seq
        self.nheads_q = nheads_q
        self.nheads_k = nheads_k
        self.nheads_v = nheads_v
        self.head_dim = head_dim
        self.dtype = dtype

        # tile config chosen by global sequence length — see choose_tile_config.
        world_size = dist.get_world_size(sp_group)
        tile_n, pingpong = choose_tile_config(local_seq, world_size)
        self._state: Optional["WsPushState"] = init_ws_push_state(
            sp_group=sp_group,
            device=device,
            bs=bs,
            local_seq=local_seq,
            nheads_q=nheads_q,
            nheads_k=nheads_k,
            nheads_v=nheads_v,
            head_dim=head_dim,
            dtype=dtype,
            tile_n=tile_n,
            pingpong=pingpong,
        )

    @property
    def state(self) -> "WsPushState":
        if self._state is None:
            raise RuntimeError("WsPushStateManager has been torn down; cannot access state.")
        return self._state

    @property
    def shape(self) -> Tuple[int, int, int, int, int, int, torch.dtype]:
        return (
            self.bs,
            self.local_seq,
            self.nheads_q,
            self.nheads_k,
            self.nheads_v,
            self.head_dim,
            self.dtype,
        )

    def is_compatible(
        self,
        *,
        bs: int,
        local_seq: int,
        dtype: torch.dtype,
    ) -> bool:
        """Pre-flight check before binding a ``WsPushDispatch``.

        Only the input-tensor dimensions (``bs / local_seq / dtype``) need
        runtime checks; ``nheads_*`` and ``head_dim`` are model-intrinsic and
        would have raised at ``init_ws_push_state`` time.
        """
        return self._state is not None and self.bs == bs and self.local_seq == local_seq and self.dtype == dtype

    def teardown(self) -> None:
        """Release the symm-mem buffer following ``WsPushState``'s contract.

        Idempotent. Subsequent ``state`` access raises ``RuntimeError``.
        """
        if self._state is None:
            return

        import torch.distributed as dist

        # Device-agnostic equivalent of torch.cuda.synchronize() — keeps the
        # teardown path importable on NPU hosts.
        from veomni.utils.device import synchronize

        synchronize()
        if self.sp_group is not None:
            dist.barrier(self.sp_group)
        self._state.close()
        self._state = None
