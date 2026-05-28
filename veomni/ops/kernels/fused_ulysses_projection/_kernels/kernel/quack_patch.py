"""Runtime monkey-patch extending ``quack.gemm_sm90.GemmSm90`` with the four
epilogue hooks needed by the Ulysses pre-attention PackQKV mixin.

``install()`` adds hooks and replaces methods on ``GemmSm90`` idempotently:

* Hooks added (defaults are no-ops): ``epi_get_output_tensor``,
  ``epi_has_output_stage``, ``epi_make_peer_gmem_copy_fns``, ``epi_store_subtile``.
  Inert defaults for ``epi_setup_postact`` / ``epi_convert_postact`` /
  ``rounding_mode`` are also installed so the patched ``epilogue()`` runs on
  PackQKV (which does not opt into activation fusion).
* Methods replaced: ``__call__`` / ``kernel`` / ``epilogue`` — upstream body
  preserved, surgical diffs documented at each call site.

Neither ``install()`` nor ``_install_packqkv_scheduler_overrides()`` runs at
import time; the first real user (``ws_push._compile_ws_gemm_packqkv``) calls
them explicitly. Both are idempotent.

See ``dev_md/ulysses/PR_doc/patch_gemm_code_design.md`` for the full design.
"""

import os
import warnings
from functools import partial
from typing import Callable, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import quack.copy_utils as copy_utils
import quack.gemm_sm90 as _qgemm
import quack.sm90_utils as quack_sm90_utils
from cutlass import Boolean, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum
from quack.gemm_sm90 import GemmSm90 as _GemmSm90
from quack.pipeline import make_pipeline_state
from quack.rounding import RoundingMode
from quack.varlen_utils import VarlenArguments, VarlenManager


_PATCH_FLAG = "_ulysses_quack_patched"

# Pinned to fail fast on quack drift — patch depends on specific 0.3.4 internals.
# See ``.agents/skills/quack-v0.3.4-port/`` for failure modes on mismatch.
_KNOWN_GOOD_QUACK_VERSION = "0.3.4"

# Methods the patched bodies reach via ``self.<name>``. Missing-method probe in
# ``install()`` reports them all up front instead of crashing inside JIT compile.
# Scope: production path with pingpong=True and gather_A=False (PackQKV is dense);
# gather_A=True branches are intentionally excluded from the probe.
_REQUIRED_GEMMSM90_METHODS: Tuple[str, ...] = (
    # Core invocation
    "_setup_attributes",
    "_make_tma_atoms_and_tensors",
    "_make_tma_epi_atoms_and_tensors",
    "get_scheduler_class",
    "get_scheduler_arguments",
    "make_ab_pipeline",
    "make_epi_pipeline",
    "make_sched_pipeline",
    "make_epi_store_pipeline",
    "mma",
    "load_AB",
    "kernel",
    "epilogue",
    # Epilogue
    "epi_to_underlying_arguments",
    "epi_get_smem_struct",
    "epi_get_smem_tensors",
    "epi_get_tma_atoms",
    "epi_load_acc_subtile",
    "epi_visit_acc",
    "epi_visit_subtile",
    "epi_begin",
    "epi_begin_loop",
    "epi_end",
    "epilog_gmem_copy_and_partition",
    "epilog_smem_store_and_partition",
    "epilog_smem_load_and_partition",
    # Pingpong barriers
    "pingpong_barrier_arrive",
    "pingpong_barrier_sync",
)


# --- 1. New epilogue hooks (default implementations) ---


def _epi_get_output_tensor(self, args, *, loc=None, ip=None):
    """Default: signal that mD (passed to ``__call__``) is authoritative."""
    del args
    return None


def _epi_has_output_stage(self, params, *, loc=None, ip=None):
    """Default: no epilogue-driven output stage."""
    del params
    return False


def _epi_make_peer_gmem_copy_fns(
    self,
    params,
    sD,
    tile_coord_mnkl=None,
    *,
    loc=None,
    ip=None,
):
    """Default: no peer copies. Implementations must return a tuple (see assert in ``_patched_epilogue``)."""
    del params, sD, tile_coord_mnkl
    return ()


@cute.jit
def _epi_store_subtile(
    self,
    params,
    epi_loop_tensors,
    gmem_coord,
    copy_D,
    peer_copy_Ds,
    epi_buffer,
    epi_idx=Int32(0),  # noqa: B008  # mirrors upstream quack signature
):
    """Default: forward to ``copy_D`` (matches upstream inline store)."""
    del params, epi_loop_tensors, peer_copy_Ds, epi_idx
    if const_expr(copy_D is not None):
        copy_D(src_idx=epi_buffer, dst_idx=gmem_coord)


# --- 2. Inert defaults so the patched epilogue runs on subclasses without activation fusion ---


def _epi_setup_postact(self, *args, **kwargs):
    return None


def _epi_convert_postact(self, *args, **kwargs):
    return None


# --- 3. Patched ``__call__`` — adds dtype/layout fallback via epi_get_output_tensor and gates epi SMEM on has_output_stage ---


@cute.jit
def _patched_call(
    self,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mD: Optional[cute.Tensor],
    mC: Optional[cute.Tensor],
    epilogue_args: tuple,
    scheduler_args,
    varlen_args: Optional[VarlenArguments],
    stream: cuda.CUstream,
):
    epi_output_tensor = self.epi_get_output_tensor(epilogue_args)
    self.a_dtype = mA.element_type
    self.b_dtype = mB.element_type
    self.d_dtype = (
        mD.element_type
        if mD is not None
        else epi_output_tensor.element_type
        if epi_output_tensor is not None
        else None
    )
    self.c_dtype = mC.element_type if mC is not None else None
    self.a_layout = LayoutEnum.from_tensor(mA)
    self.b_layout = LayoutEnum.from_tensor(mB)
    self.d_layout = (
        LayoutEnum.from_tensor(mD)
        if mD is not None
        else LayoutEnum.from_tensor(epi_output_tensor)
        if epi_output_tensor is not None
        else None
    )
    self.c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None

    if const_expr(self.a_dtype.width == 16 and self.a_dtype != self.b_dtype):
        raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
    if const_expr(self.a_dtype.width != self.b_dtype.width):
        raise TypeError(f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}")
    if const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
        raise TypeError("a_dtype should be float16 or float8")

    if const_expr(varlen_args is None):
        varlen_args = VarlenArguments()
    assert (varlen_args.mAIdx is not None) == self.gather_A
    varlen_m = varlen_args.mCuSeqlensM is not None
    varlen_k = varlen_args.mCuSeqlensK is not None

    self._setup_attributes(epilogue_args)

    a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
    b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
    tma_atom_a, tma_tensor_a = None, None
    if const_expr(not self.gather_A):
        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            copy_utils.create_ragged_tensor_for_tma(mA, ragged_dim=1) if varlen_k and not self.gather_A else mA,
            a_smem_layout,
            (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
            self.cluster_shape_mnk[1],
        )
    tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
        copy_utils.create_ragged_tensor_for_tma(mB, ragged_dim=1) if varlen_k else mB,
        b_smem_layout,
        (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
        self.cluster_shape_mnk[0],
    )

    self.num_tma_load_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
    if const_expr(not self.gather_A):
        self.num_tma_load_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)

    tma_atom_d, tma_tensor_d = None, None
    if const_expr(mD is not None):
        tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
            copy_utils.create_ragged_tensor_for_tma(
                mD,
                ragged_dim=0,
                ptr_shift=True,
            )
            if varlen_m
            else mD,
            self.epi_smem_layout_staged,
            self.epi_tile,
            op_type="store"
            if not (hasattr(epilogue_args, "add_to_output") and epilogue_args.add_to_output)
            else "add",
        )
    tma_atom_c, tma_tensor_c = None, None
    if const_expr(mC is not None):
        tma_atom_c, tma_tensor_c = self._make_tma_epi_atoms_and_tensors(
            mC, self.epi_c_smem_layout_staged, self.epi_tile, op_type="load"
        )

    epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
    varlen_params = VarlenManager.to_underlying_arguments(varlen_args)

    has_output_stage = mD is not None or self.epi_has_output_stage(epilogue_params)

    TileSchedulerCls = self.get_scheduler_class(varlen_m=varlen_m)
    tile_sched_args = self.get_scheduler_arguments(mA, mB, mD, scheduler_args, varlen_args, epilogue_args)
    tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
    grid = TileSchedulerCls.get_grid_shape(tile_sched_params, scheduler_args.max_active_clusters)

    epi_smem_size = cute.cosize(self.epi_smem_layout_staged) if has_output_stage else 0
    epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0

    @cute.struct
    class SharedStorage:
        ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
        epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
        sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
        sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
        sD: cute.struct.Align[
            cute.struct.MemRange[self.d_dtype if self.d_dtype is not None else Int32, epi_smem_size],
            self.buffer_align_bytes,
        ]
        sC: cute.struct.Align[
            cute.struct.MemRange[self.c_dtype if self.c_dtype is not None else Int32, epi_c_smem_size],
            self.buffer_align_bytes,
        ]
        epi: self.epi_get_smem_struct(epilogue_params)
        sA: cute.struct.Align[
            cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
            self.buffer_align_bytes,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
            self.buffer_align_bytes,
        ]

    self.shared_storage = SharedStorage

    self.kernel(
        self.tiled_mma,
        tma_atom_a,
        tma_tensor_a if const_expr(not self.gather_A) else mA,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_d,
        tma_tensor_d,
        tma_atom_c,
        tma_tensor_c,
        epilogue_params,
        varlen_params,
        self.cluster_layout_mnk,
        self.a_smem_layout_staged,
        self.b_smem_layout_staged,
        self.epi_smem_layout_staged,
        self.epi_c_smem_layout_staged,
        tile_sched_params,
        TileSchedulerCls,
    ).launch(
        grid=grid,
        block=[self.threads_per_cta, 1, 1],
        cluster=self.cluster_shape_mnk,
        stream=stream,
        min_blocks_per_mp=1,
    )
    return


# --- 4. Patched ``kernel`` — widens has_D, prefetches epi TMA atoms, builds per-tile peer_copy_Ds ---


@cute.kernel
def _patched_kernel(
    self,
    tiled_mma: cute.TiledMma,
    tma_atom_a: Optional[cute.CopyAtom],
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_d: Optional[cute.CopyAtom],
    mD_mnl: Optional[cute.Tensor],
    tma_atom_c: Optional[cute.CopyAtom],
    mC_mnl: Optional[cute.Tensor],
    epilogue_params,
    varlen_params: VarlenManager.Params,
    cluster_layout_mnk: cute.Layout,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    epi_smem_layout: cute.ComposedLayout,
    epi_c_smem_layout: cute.ComposedLayout,
    tile_sched_params,
    TileSchedulerCls: cutlass.Constexpr[Callable],
):
    varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
    varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
    assert not (varlen_m and varlen_k)
    # Fail fast: gather_A path calls ``self._make_gather_A_copy`` which is
    # absent on quack 0.3.4 (also omitted from ``_REQUIRED_GEMMSM90_METHODS``);
    # without this assert the failure would surface deep inside MLIR.
    assert not self.gather_A, "patched _patched_kernel does not support gather_A on quack 0.3.4"
    if const_expr(self.gather_A):
        assert varlen_m or varlen_k
    has_D = const_expr(mD_mnl is not None or self.epi_has_output_stage(epilogue_params))
    has_C = const_expr(mC_mnl is not None)

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # Prefetch Tma desc
    if warp_idx == self.ab_load_warp_id:
        for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
            if const_expr(tma_atom is not None):
                cpasync.prefetch_descriptor(tma_atom)
        for tma_atom in self.epi_get_tma_atoms(epilogue_params):
            if const_expr(tma_atom is not None):
                cpasync.prefetch_descriptor(tma_atom)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    ab_pipeline = self.make_ab_pipeline(
        tiled_mma=tiled_mma,
        cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
        ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
    )
    epi_pipeline = None
    if const_expr(has_C):
        epi_pipeline = self.make_epi_pipeline(
            c_smem_layout=cute.slice_(epi_c_smem_layout, (None, None, 0)),
            epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
        )
    sched_pipeline = None
    sched_data = None
    if const_expr(self.is_persistent):
        sched_pipeline = self.make_sched_pipeline(
            cluster_layout_mnk,
            sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
            varlen_k=varlen_k,
        )
        sched_data = storage.sched_data.get_tensor((4, self.sched_stage))

    pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

    sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
    sD = None
    if const_expr(has_D):
        sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
    sC = None
    if const_expr(has_C):
        sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
    epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

    varlen_manager = VarlenManager.create(
        varlen_params,
        len_m_static=Int32(
            mA_mkl.shape[0] if varlen_k or varlen_params.mAIdx is None else varlen_params.mAIdx.shape[0]
        ),
        len_k_static=Int32(mA_mkl.shape[1]),
    )

    TileSchedulerCls = partial(TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline)

    pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

    if warp_idx >= self.ab_load_warp_id:
        cute.arch.setmaxregister_decrease(self.num_regs_load)
        if warp_idx >= self.ab_load_warp_id and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps:
            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
            a_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1)
            b_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0)
            a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
            b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

            is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
            if const_expr(cute.size(cluster_layout_mnk) > 1):
                is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                copy_A, prefetch_A = None, None
                if const_expr(not self.gather_A):
                    mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                    gA_mk = cute.local_tile(
                        mA_mk,
                        cute.select(self.cta_tile_shape_mnk, [0, 2]),
                        (tile_coord_mnkl[0], None),
                    )
                    copy_A, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_a,
                        cta_coord=block_in_cluster_coord_mnk[1],
                        cta_layout=cute.make_layout(cute.slice_(cluster_layout_mnk, (0, None, 0)).shape),
                        src_tensor=gA_mk,
                        dst_tensor=sA,
                        mcast_mask=a_mcast_mask,
                    )
                else:
                    copy_A, prefetch_A = self._make_gather_A_copy(
                        mA_mkl, sA, varlen_manager, tile_coord_mnkl, batch_idx
                    )
                gB_nk = cute.local_tile(
                    varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                    cute.select(self.cta_tile_shape_mnk, [1, 2]),
                    (tile_coord_mnkl[1], None),
                )
                copy_B, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_b,
                    cta_coord=block_in_cluster_coord_mnk[0],
                    cta_layout=cute.make_layout(cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape),
                    src_tensor=gB_nk,
                    dst_tensor=sB,
                    mcast_mask=b_mcast_mask,
                )
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                if const_expr(not self.gather_A):
                    ab_producer_state = self.load_AB(ab_pipeline, ab_producer_state, copy_A, copy_B, k_tile_cnt)
                else:
                    ab_producer_state = self.load_AB_gather_A(
                        ab_pipeline,
                        ab_producer_state,
                        copy_A,
                        prefetch_A,
                        copy_B,
                        k_tile_cnt,
                        varlen_m=varlen_m,
                    )
                tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.get_current_work()
            if const_expr(self.pingpong and not varlen_k):
                if is_scheduler_warp:
                    tile_scheduler.write_work_tile_to_smem(work_tile)
                work_tile = tile_scheduler.get_current_work()
            if warp_idx == self.ab_load_warp_id:
                ab_pipeline.producer_tail(ab_producer_state)
            if is_scheduler_warp:
                tile_scheduler.producer_tail()

    if warp_idx < self.ab_load_warp_id:
        cute.arch.setmaxregister_increase(self.num_regs_mma)
        is_tma_warp = Boolean(
            (not self.pingpong and warp_idx == 0) or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
        )
        tidx, _, _ = cute.arch.thread_idx()
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        if const_expr(self.pingpong):
            tidx = tidx % self.num_threads_per_warp_group
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups if const_expr(not self.pingpong) else 1,
            stride=self.num_threads_per_warp_group,
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx if not self.pingpong else 0))

        acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(thr_mma, self.cta_tile_shape_mnk, sA, sB)
        acc_slow = None
        if const_expr(self.fp8_slow_accum):
            acc_slow = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
        mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)

        if const_expr(self.pingpong):
            if warp_group_idx == 0:
                self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

        k_tile_cnt_static = cute.ceil_div(mA_mkl.shape[1], self.cta_tile_shape_mnk[2])
        c_tile_cnt = cute.size(cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile))

        ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
        epi_store_pipeline = self.make_epi_store_pipeline()
        epi_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.epi_c_stage)
        epi_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.epi_c_stage)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        if const_expr(self.pingpong):
            if warp_idx >= 4:
                epi_read_state.advance_iters(c_tile_cnt)
                epi_producer_state.advance_iters(c_tile_cnt)
                if const_expr(not varlen_k):
                    ab_read_state.advance_iters(k_tile_cnt_static)
                else:
                    len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                    k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    ab_read_state.advance_iters(k_tile_cnt)
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
        while work_tile.is_valid_tile:
            tile_coord_mnkl = work_tile.tile_idx
            batch_idx = tile_coord_mnkl[3]
            len_k = varlen_manager.len_k(batch_idx)
            k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
            # quack 0.3.4's mma() does its own pingpong_barrier_sync(stage="mma");
            # an external sync here would double-arrive the MmaWG0 barrier and deadlock.
            ab_read_state = self.mma(ab_pipeline, ab_read_state, mma_fn, acc, acc_slow, k_tile_cnt, warp_group_idx)
            if const_expr(varlen_k):
                if k_tile_cnt == 0:
                    acc.fill(0.0)

            if const_expr(self.pingpong):
                self.pingpong_barrier_sync(warp_group_idx, "epi")

            # In quack 0.3.4 ``epilogue_barrier`` is a per-tile local, not ``self.*``;
            # recreate it here to match the 0.3.4 contract.
            epilogue_barrier = pipeline.NamedBarrier(
                barrier_id=int(_qgemm.NamedBarrierGemm.Epilogue),
                num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
            )

            copy_D = None
            if const_expr(has_D):
                if const_expr(mD_mnl is not None):
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        varlen_manager.offset_batch_epi(mD_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sD,
                        tile_coord_mnkl,
                    )
            peer_copy_Ds = self.epi_make_peer_gmem_copy_fns(epilogue_params, sD, tile_coord_mnkl)
            copy_C = None
            if const_expr(has_C):
                copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                    tma_atom_c,
                    varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
                    self.cta_tile_shape_mnk[:2],
                    self.epi_tile,
                    sC,
                    tile_coord_mnkl,
                )
                copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)

            d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
            )
            # Inline of newer-quack ``epi_retile_acc`` (absent in 0.3.4).
            tRS_rAcc = cute.flat_divide(acc, tRS_rD.layout)
            load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
            if const_expr(has_C):
                tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                    tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                )
            else:
                tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None

            self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)

            epi_read_state, epi_producer_state = self.epilogue(
                epilogue_params,
                epi_smem_tensors,
                epi_pipeline,
                epi_store_pipeline,
                epi_read_state,
                epi_producer_state,
                self.epi_tile,
                load_acc_subtile,
                tRS_rD,
                tRS_rC,
                None,
                tiled_copy_r2s,
                tRS_sD,
                tiled_copy_s2r,
                tSR_rC,
                tSR_sC,
                copy_D,
                peer_copy_Ds,
                copy_C,
                tile_coord_mnkl,
                varlen_manager,
                epilogue_barrier,
                tile_scheduler,
                tidx,
                is_tma_warp,
            )

            if const_expr(self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()
                self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")

            if const_expr(not self.pingpong):
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            else:
                epi_read_state.advance_iters(c_tile_cnt)
                epi_producer_state.advance_iters(c_tile_cnt)
                if const_expr(not varlen_k):
                    ab_read_state.advance_iters(k_tile_cnt_static)
                    tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                    work_tile = tile_scheduler.get_current_work()
                else:
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                    if work_tile.is_valid_tile:
                        len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                        k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        ab_read_state.advance_iters(k_tile_cnt)
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.pingpong):
            if is_tma_warp:
                epi_store_pipeline.producer_tail()


# --- 5. Patched ``epilogue`` — accepts peer_copy_Ds, widens has_D, replaces inline copy_D with epi_store_subtile ---


@cute.jit
def _patched_epilogue(
    self,
    params,
    epi_smem_tensors: Tuple[cute.Tensor, ...],
    epi_pipeline: cutlass.pipeline.PipelineAsync,
    epi_store_pipeline: cutlass.pipeline.PipelineAsync,
    epi_read_state: cutlass.pipeline.PipelineState,
    epi_producer_state: Optional[cutlass.pipeline.PipelineState],
    epi_tile: cute.Tile,
    load_acc_subtile: Callable,
    tRS_rD: cute.Tensor,
    tRS_rC: Optional[cute.Tensor],
    tiled_copy_t2r: Optional[cute.TiledCopy],
    tiled_copy_r2s: cute.TiledCopy,
    tRS_sD: cute.Tensor,
    tiled_copy_s2r: Optional[cute.ThrCopy],
    tSR_rC: Optional[cute.Tensor],
    tSR_sC: Optional[cute.Tensor],
    copy_D: Optional[Callable],
    peer_copy_Ds,
    copy_C: Optional[Callable],
    tile_coord_mnkl: cute.Coord,
    varlen_manager: VarlenManager,
    epilogue_barrier: cutlass.pipeline.NamedBarrier,
    tile_scheduler,
    tidx: Int32,
    is_tma_warp: Boolean,
) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
    # Must be a tuple so ``len(...)`` works inside ``const_expr`` below.
    assert isinstance(peer_copy_Ds, tuple), (
        f"epi_make_peer_gmem_copy_fns must return a tuple; got {type(peer_copy_Ds).__name__}"
    )
    has_C = const_expr(tRS_rC is not None)
    has_D = const_expr(copy_D is not None or len(peer_copy_Ds) > 0)

    postact_ctx = self.epi_setup_postact(
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    )

    epi_tile_shape = cute.zipped_divide(cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile).shape[1]
    epi_tile_layout = cute.make_ordered_layout(epi_tile_shape, order=(1, 0))
    epi_tile_num = cute.size(epi_tile_shape)
    num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

    epi_tensors = self.epi_begin(
        params,
        epi_smem_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        epilogue_barrier,
        tidx,
    )

    if const_expr(copy_C is not None):
        for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
            gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx)
            if is_tma_warp:
                epi_pipeline.producer_acquire(epi_producer_state)
                copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                epi_pipeline.producer_commit(epi_producer_state)
            epi_producer_state.advance()

    for epi_idx in cutlass.range_constexpr(epi_tile_num):
        gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
        load_acc_subtile(tRS_rD, epi_idx)
        epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, gmem_coord)
        if const_expr(has_C):
            epi_pipeline.consumer_wait(epi_read_state)
            cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                epi_pipeline.consumer_release(epi_read_state)
            epi_read_state.advance()
        if const_expr(copy_C is not None and epi_idx + self.epi_c_stage < epi_tile_num):
            gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
            if is_tma_warp:
                epi_pipeline.producer_acquire(epi_producer_state)
                copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                epi_pipeline.producer_commit(epi_producer_state)
            epi_producer_state.advance()
        tRS_rPostAct = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, tRS_rC)
        if const_expr(postact_ctx is not None):
            tRS_rPostAct_out = self.epi_convert_postact(
                tRS_rPostAct,
                epi_loop_tensors["sr_seed"],
                tidx,
                tile_coord_mnkl,
                num_prev_subtiles,
                epi_idx,
            )
        if is_tma_warp:
            epi_store_pipeline.producer_acquire()
        epilogue_barrier.arrive_and_wait()
        epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
        if const_expr(copy_D is not None):
            # ``isinstance`` guard so a None default or mistakenly-set string
            # short-circuits to the safe plain-cast path.
            if const_expr(
                isinstance(self.rounding_mode, RoundingMode)
                and self.rounding_mode == RoundingMode.RS
                and self.acc_dtype == cutlass.Float32
                and self.d_dtype == cutlass.BFloat16
            ):
                seed = epi_loop_tensors["sr_seed"] + (
                    tile_coord_mnkl[0] * 65537
                    + tile_coord_mnkl[1] * 257
                    + tile_coord_mnkl[3] * 17
                    + (num_prev_subtiles + epi_idx) * 7
                )
                copy_utils.sr_cvt_copy(
                    tiled_copy_r2s,
                    tRS_rD,
                    tRS_sD[None, None, None, epi_buffer],
                    seed,
                    tidx,
                )
            else:
                copy_utils.cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer])
        elif const_expr(has_D):
            # Peer-store path: TMA src comes from sD, so still populate it.
            copy_utils.cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer])
        if const_expr(postact_ctx is not None):
            tiled_copy_postact_r2s, tRS_sPostAct, copy_postact = postact_ctx
            cute.copy(
                tiled_copy_postact_r2s,
                tiled_copy_postact_r2s.retile(tRS_rPostAct_out),
                tRS_sPostAct[None, None, None, epi_buffer],
            )
        cute.arch.fence_view_async_shared()
        epilogue_barrier.arrive_and_wait()
        if is_tma_warp:
            if const_expr(has_D):
                self.epi_store_subtile(
                    params,
                    epi_loop_tensors,
                    gmem_coord,
                    copy_D,
                    peer_copy_Ds,
                    epi_buffer,
                    epi_idx,
                )
            if const_expr(postact_ctx is not None):
                copy_postact(src_idx=epi_buffer, dst_idx=gmem_coord)
            epi_store_pipeline.producer_commit()

    self.epi_end(
        params,
        epi_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    )

    return epi_read_state, epi_producer_state


# --- 6. Install (idempotent) ---


_ALLOW_QUACK_DRIFT_ENV = "VEOMNI_FUSED_ALLOW_QUACK_DRIFT"


def _check_quack_compatibility() -> None:
    """Two-layer defence: version pin + missing-method probe.

    The version check hard-raises on drift (silent miscompile is the documented
    failure mode); set ``VEOMNI_FUSED_ALLOW_QUACK_DRIFT=1`` to downgrade to a
    warning for an investigative run. The structural probe hard-raises when any
    ``_REQUIRED_GEMMSM90_METHODS`` entry is absent.
    """
    import quack as _quack

    installed = getattr(_quack, "__version__", None)
    if installed is not None and installed != _KNOWN_GOOD_QUACK_VERSION:
        msg = (
            f"veomni.fused_ulysses_projection._kernels.kernel.quack_patch was "
            f"authored against quack=={_KNOWN_GOOD_QUACK_VERSION}; installed "
            f"version is {installed!r}. Documented drift failure modes: "
            f"bit-exact zero on alternating batches; non-deterministic forward "
            f"output; AttributeError inside cute.compile. Re-pin quack to "
            f"{_KNOWN_GOOD_QUACK_VERSION} or update the patch."
        )
        if os.environ.get(_ALLOW_QUACK_DRIFT_ENV) == "1":
            warnings.warn(
                msg + f" ({_ALLOW_QUACK_DRIFT_ENV}=1, downgrading to warning.)",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            raise RuntimeError(msg + f" Set {_ALLOW_QUACK_DRIFT_ENV}=1 to bypass for an investigative run.")

    missing = [m for m in _REQUIRED_GEMMSM90_METHODS if not hasattr(_GemmSm90, m)]
    if missing:
        raise RuntimeError(
            "quack.gemm_sm90.GemmSm90 is missing methods that veomni's "
            "fused-Ulysses patch depends on: "
            + ", ".join(missing)
            + f". Pin quack=={_KNOWN_GOOD_QUACK_VERSION} or update "
            "veomni/ops/kernels/fused_ulysses_projection/_kernels/kernel/quack_patch.py to "
            "match the new upstream API."
        )


def install() -> None:
    """Install the Ulysses extension hooks onto ``quack.gemm_sm90.GemmSm90``.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    if getattr(_GemmSm90, _PATCH_FLAG, False):
        return

    _check_quack_compatibility()

    _GemmSm90.epi_get_output_tensor = _epi_get_output_tensor
    _GemmSm90.epi_has_output_stage = _epi_has_output_stage
    _GemmSm90.epi_make_peer_gmem_copy_fns = _epi_make_peer_gmem_copy_fns
    _GemmSm90.epi_store_subtile = _epi_store_subtile

    if not hasattr(_GemmSm90, "epi_setup_postact"):
        _GemmSm90.epi_setup_postact = _epi_setup_postact
    if not hasattr(_GemmSm90, "epi_convert_postact"):
        _GemmSm90.epi_convert_postact = _epi_convert_postact
    if not hasattr(_GemmSm90, "rounding_mode"):
        _GemmSm90.rounding_mode = None

    _GemmSm90.__call__ = _patched_call
    _GemmSm90.kernel = _patched_kernel
    _GemmSm90.epilogue = _patched_epilogue

    setattr(_GemmSm90, _PATCH_FLAG, True)


# --- 7. PackQKV-only scheduler override (deferred call; see end-of-file note) ---
#
# Replaces quack's default TileScheduler with PackQKVTileScheduler on the
# ``GemmWsPreAttnPackQKVSm90`` subclass only. Lazy import below avoids a
# Python-level cycle with ``gemm_ws_packqkv.py``.


_PACKQKV_SCHED_FLAG = "_packqkv_scheduler_installed"


def _install_packqkv_scheduler_overrides() -> None:
    from .gemm_ws_packqkv import GemmWsPreAttnPackQKVSm90
    from .pack_qkv_scheduler import (
        PackQKVTileScheduler,
        PackQKVTileSchedulerArguments,
    )

    if getattr(GemmWsPreAttnPackQKVSm90, _PACKQKV_SCHED_FLAG, False):
        return

    _orig_get_scheduler_arguments = GemmWsPreAttnPackQKVSm90.get_scheduler_arguments

    def _packqkv_get_scheduler_class(self, varlen_m: bool = False):
        # PackQKV epilogue assumes dense M (see _compile_ws_gemm_packqkv invariants).
        assert not varlen_m, "PackQKVTileScheduler does not support varlen_m"
        return PackQKVTileScheduler

    def _packqkv_get_scheduler_arguments(self, mA, mB, mD, scheduler_args, varlen_args, epilogue_args):
        base_args = _orig_get_scheduler_arguments(self, mA, mB, mD, scheduler_args, varlen_args, epilogue_args)
        # Read head shapes from instance attrs (Python ints, set by
        # _compile_ws_gemm_packqkv) — they participate in the @jit_cache key,
        # so the cache never returns a binary built for a different head layout.
        try:
            nheads_q = self.packqkv_nheads_q
            nheads_k = self.packqkv_nheads_k
            nheads_v = self.packqkv_nheads_v
            head_dim = self.packqkv_head_dim
            world_size = self.packqkv_world_size
        except AttributeError as e:
            raise RuntimeError(
                "PackQKVTileScheduler requires gemm.packqkv_{nheads_q,k,v,head_dim,world_size} "
                "to be set before cute.compile(). Set them in _compile_ws_gemm_packqkv."
            ) from e
        cta_tile_n = int(self.cta_tile_shape_mnk[1])
        cluster_n = int(self.cluster_shape_mnk[1])
        unit = cta_tile_n * cluster_n
        n_q_cols = nheads_q * head_dim
        n_k_cols = nheads_k * head_dim
        n_v_cols = nheads_v * head_dim
        assert n_q_cols % unit == 0, f"Q segment cols={n_q_cols} not divisible by cta_tile_n*cluster_n={unit}"
        assert n_k_cols % unit == 0, f"K segment cols={n_k_cols} not divisible by cta_tile_n*cluster_n={unit}"
        assert n_v_cols % unit == 0, f"V segment cols={n_v_cols} not divisible by cta_tile_n*cluster_n={unit}"
        return PackQKVTileSchedulerArguments(
            problem_shape_ntile_mnl=base_args.problem_shape_ntile_mnl,
            raster_order=base_args.raster_order,
            group_size=base_args.group_size,
            cluster_shape_mnk=base_args.cluster_shape_mnk,
            tile_count_semaphore=base_args.tile_count_semaphore,
            batch_idx_permute=base_args.batch_idx_permute,
            persistence_mode=base_args.persistence_mode,
            n_clusters_q=n_q_cols // unit,
            n_clusters_k=n_k_cols // unit,
            n_clusters_v=n_v_cols // unit,
            world_size_const=world_size,
        )

    GemmWsPreAttnPackQKVSm90.get_scheduler_class = _packqkv_get_scheduler_class
    GemmWsPreAttnPackQKVSm90.get_scheduler_arguments = _packqkv_get_scheduler_arguments
    setattr(GemmWsPreAttnPackQKVSm90, _PACKQKV_SCHED_FLAG, True)


# Both ``install()`` and ``_install_packqkv_scheduler_overrides()`` are called
# explicitly from ``ws_push._compile_ws_gemm_packqkv``. Doing it at import time
# would mutate ``GemmSm90`` for every process that imports this file and
# re-introduce the import cycle with ``gemm_ws_packqkv.py``.
