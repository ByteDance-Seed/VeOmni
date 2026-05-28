# GPU cross-rank group barrier — cross-rank synchronization.
#
# Self-contained: bundles `_compile_gpu_barrier` with every PTX inline-asm

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from quack.cache_utils import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor


# ============================================================================
# PTX inline-asm primitives
# ============================================================================


@dsl_user_op
def fence_acq_rel_sys(*, loc=None, ip=None) -> None:
    """System-scope acquire-release fence."""
    llvm.inline_asm(
        None,
        [],
        "fence.acq_rel.sys;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def atom_cas_sys_notify_s32(gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """System-scope CAS notify: swap 0 -> 1 and ignore the previous value."""
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr],
        "{\n\t.reg .b32 old;\n\tatom.cas.sys.global.b32 old, [$0], 0, 1;\n\t}\n",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def spin_cas_wait_clear_sys_s32(gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Spin until value == 1, then atomically clear it back to 0.

    Unbounded by design: a peer that never sends its CAS-notify (crash,
    driver fault) is killed host-side by PyTorch's NCCL watchdog at the
    next collective; no GPU-side timeout.
    """
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr],
        "{\n\t"
        ".reg .b32 old;\n\t"
        ".reg .pred p;\n\t"
        "LOOP_${:uid}:\n\t"
        "atom.cas.sys.global.b32 old, [$0], 1, 0;\n\t"
        "setp.ne.s32 p, old, 1;\n\t"
        "@p bra LOOP_${:uid};\n\t"
        "}\n",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def load_i64_as_gmem_ptr_s32(
    ptrs_i64: cute.Tensor,
    idx: Int32,
    *,
    loc=None,
    ip=None,
) -> cute.Pointer:
    """Load an int64 address from a tensor and reinterpret it as global int32*."""
    return cute.make_ptr(
        Int32,
        Int64(ptrs_i64[idx]),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )


# ============================================================================
# Kernel builder + cached compile entry point
# ============================================================================


def _make_gpu_barrier_kernel(world_size: int):
    @cute.kernel
    def gpu_barrier_all(barrier_ptrs_i64: cute.Tensor, rank: Int32):
        local_base_ptr = load_i64_as_gmem_ptr_s32(barrier_ptrs_i64, rank)

        for peer_rank in cutlass.range_constexpr(world_size):
            if rank != peer_rank:
                peer_base_ptr = load_i64_as_gmem_ptr_s32(barrier_ptrs_i64, Int32(peer_rank))
                peer_flag_ptr = peer_base_ptr + rank
                fence_acq_rel_sys()
                atom_cas_sys_notify_s32(peer_flag_ptr)

        for peer_rank in cutlass.range_constexpr(world_size):
            if rank != peer_rank:
                local_flag_ptr = local_base_ptr + Int32(peer_rank)
                spin_cas_wait_clear_sys_s32(local_flag_ptr)

        fence_acq_rel_sys()

    return gpu_barrier_all


@jit_cache
def _compile_gpu_barrier(world_size: int, device_capacity) -> object:
    """Compile the WS post-GEMM cross-rank GPU barrier."""
    del device_capacity

    fake_barrier_ptrs = fake_tensor(Int64, (world_size,), divisibility=1, leading_dim=0)
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    gpu_barrier_kernel = _make_gpu_barrier_kernel(world_size)

    @cute.jit
    def launch_gpu_barrier(
        barrier_ptrs_i64: cute.Tensor,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        gpu_barrier_kernel(barrier_ptrs_i64, rank).launch(
            grid=[1, 1, 1],
            block=[1, 1, 1],
            stream=stream,
        )

    return cute.compile(
        launch_gpu_barrier,
        fake_barrier_ptrs,
        Int32(0),
        fake_stream,
        options="--enable-tvm-ffi",
    )


compile_gpu_barrier = _compile_gpu_barrier
