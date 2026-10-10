"""Qwen4-Exp QSA indexer kernel registration."""

from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec


def _triton_qsa_indexer_factory():
    from .triton import qsa_indexer_forward_triton

    return qsa_indexer_forward_triton


KERNEL_REGISTRY.register(
    KernelSpec(
        name="triton",
        op_name="qsa_indexer",
        variant="standard",
        factory=_triton_qsa_indexer_factory,
        hardware=HardwareRequirement(device_type="npu"),
        description="Triton Ascend Qwen4-Exp QSA indexer with packed-varlen routing",
    )
)
