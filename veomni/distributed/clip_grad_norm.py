from ..utils.import_utils import is_torch_npu_available
from .fsdp import clip_grad_norm_ as fsdp1_clip_grad_norm
from .fsdp2 import clip_grad_norm as fsdp2_clip_grad_norm
from .fsdp2 import npu_fsdp2_clip_grad_norm
from .parallel_state import get_parallel_state


def veomni_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
):
    parallel_state = get_parallel_state()
    dp_mode = parallel_state.dp_mode
    if dp_mode == "fsdp1":
        grad_norm = fsdp1_clip_grad_norm(model, max_norm, norm_type)
    elif dp_mode == "fsdp2" and not is_torch_npu_available():
        # fsdp2 on GPU
        grad_norm = fsdp2_clip_grad_norm(model, max_norm, norm_type, error_if_nonfinite, foreach)
    elif dp_mode == "fsdp2" and is_torch_npu_available():
        # fsdp2 on NPU, where we have to manually reduce gradients
        # context: https://github.com/ByteDance-Seed/VeOmni/issues/241
        grad_norm = npu_fsdp2_clip_grad_norm(model, max_norm, norm_type, error_if_nonfinite, foreach)
    else:
        raise RuntimeError(f"Unknown dp mode {dp_mode}")

    grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    return grad_norm
