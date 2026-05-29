"""Register ``ulysses_qkv_projection`` as a GLOBAL op.

``ws_push`` owns runtime state and teardown, so the registry stores only the
active implementation name here. The callback initializes ``WsPushState`` once
process groups exist.
"""

from __future__ import annotations

from ...config.registry import BackendSpec, OpScope, OpSpec, register_op


# Backend entries written to ``state_manager._active_impl_name``.
EAGER_IMPL_NAME: str = "eager"
WS_PUSH_IMPL_NAME: str = "ws_push"


def _preflight_ws_push() -> None:
    """Fail during backend resolution if this host cannot run ``ws_push``."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ulysses_qkv_projection_implementation='ws_push' requires CUDA; "
            "no CUDA device is visible. Set it to 'eager' or use a CUDA host."
        )
    cap = torch.cuda.get_device_capability()
    if cap < (9, 0):
        raise RuntimeError(
            f"ulysses_qkv_projection_implementation='ws_push' requires SM90+ "
            f"(Hopper). Detected sm_{cap[0]}{cap[1]}. Set it to 'eager'."
        )
    try:
        import torch.distributed._symmetric_memory  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError(
            "ulysses_qkv_projection_implementation='ws_push' requires a PyTorch "
            f"build that exposes torch.distributed._symmetric_memory ({exc}). "
            "Set it to 'eager'."
        ) from exc


def _ws_push_side_effect() -> None:
    """Validate ``ws_push`` before the callback initializes collective state."""
    _preflight_ws_push()


def _eager_side_effect() -> None:
    """Clear stale ``ws_push`` state when switching back to eager."""
    from .state_manager import clear_active_manager

    clear_active_manager()


register_op(
    OpSpec(
        name="ulysses_qkv_projection",
        config_field="ulysses_qkv_projection_implementation",
        label="UlyssesQKVProjection",
        scope=OpScope.GLOBAL,
        default="eager",
        global_slot=("veomni.ops.kernels.fused_ulysses_projection.state_manager:_active_impl_name"),
        backends={
            "eager": BackendSpec(
                entry=("veomni.ops.kernels.fused_ulysses_projection._config:EAGER_IMPL_NAME"),
                side_effect=("veomni.ops.kernels.fused_ulysses_projection._config:_eager_side_effect"),
            ),
            "ws_push": BackendSpec(
                entry=("veomni.ops.kernels.fused_ulysses_projection._config:WS_PUSH_IMPL_NAME"),
                side_effect=("veomni.ops.kernels.fused_ulysses_projection._config:_ws_push_side_effect"),
            ),
        },
    )
)
