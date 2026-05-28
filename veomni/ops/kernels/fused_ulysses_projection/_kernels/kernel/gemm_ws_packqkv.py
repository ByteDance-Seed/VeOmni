from typing import NamedTuple
from dataclasses import dataclass
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import quack.copy_utils as copy_utils
import quack.sm90_utils as quack_sm90_utils
from cutlass import Boolean, Int32, Int64, Float32, const_expr
from quack.cute_dsl_utils import ParamsBase, mlir_namedtuple

# Base class import. The Ulysses extension hooks and the PackQKV scheduler
from quack.gemm_sm90 import GemmSm90 as GemmWsPreAttnSm90


@cute.jit
def _rank_seq_offset(
    seq_offsets: cute.Tensor | None,
    use_seq_offsets: cutlass.Constexpr[bool],
    rank: Int32,
    local_seq: Int32,
) -> Int32:
    if const_expr(use_seq_offsets):
        assert seq_offsets is not None
        return seq_offsets[rank]
    return rank * local_seq


@cute.jit
def input_nh_to_dst_rank_and_pack_head(     # segment-blockwise: raw W_qkv layout, no host shuffle
    input_nh: Int32,
    nheads_q: Int32,
    nheads_k: Int32,
    nheads_v: Int32,
    nheads_q_per_rank: Int32,
    nheads_k_per_rank: Int32,
    nheads_v_per_rank: Int32,
    world_size: Int32,
) -> tuple[Int32, Int32]:
    # NVLink balance is achieved by PackQKVTileScheduler reordering tile_n
    # emission across ranks — see pack_qkv_scheduler.py for the three-piece
    # invariant. ``world_size`` arg kept for ABI stability; unused here.
    del world_size
    dst_rank = Int32(0)
    pack_head_local = Int32(0)

    if input_nh < nheads_q:
        dst_rank = input_nh // nheads_q_per_rank
        pack_head_local = input_nh % nheads_q_per_rank
    elif input_nh < nheads_q + nheads_k:
        k_idx = input_nh - nheads_q
        dst_rank = k_idx // nheads_k_per_rank
        pack_head_local = nheads_q_per_rank + (k_idx % nheads_k_per_rank)
    else:
        v_idx = input_nh - nheads_q - nheads_k
        dst_rank = v_idx // nheads_v_per_rank
        pack_head_local = nheads_q_per_rank + nheads_k_per_rank + (v_idx % nheads_v_per_rank)

    return dst_rank, pack_head_local


@mlir_namedtuple
class PackQKVEpilogueArguments(NamedTuple):
    mPeerOuts_mnl: tuple = ()
    # Optional per-N bias broadcast across M.  Shape [1, N_total] (or [bs, N_total]).
    # Layout must match the raw block-per-rank W_qkv N-axis order — no host
    # permute is applied. The epilogue's N-coordinate math indexes the bias
    # at the same physical column as the W_qkv weight it corresponds to.
    mBias: cute.Tensor | None = None
    rank: Int32 = Int32(0)
    world_size: Int32 = Int32(1)
    bs: Int32 = Int32(1)
    local_seq: Int32 = Int32(0)
    seq_global: Int32 = Int32(0)
    nheads_q: Int32 = Int32(0)
    nheads_k: Int32 = Int32(0)
    nheads_v: Int32 = Int32(0)
    nheads_q_per_rank: Int32 = Int32(0)
    nheads_k_per_rank: Int32 = Int32(0)
    nheads_v_per_rank: Int32 = Int32(0)
    head_dim: Int32 = Int32(0)
    pack_hidden_local: Int32 = Int32(0)
    seq_offsets: cute.Tensor | None = None
    use_seq_offsets: cutlass.Constexpr[bool] = False
    peer_store_enabled: cutlass.Constexpr[bool] = True


@dataclass
class PackQKVEpilogueParams(ParamsBase):
    mPeerOuts_mnl: tuple[cute.Tensor, ...] = ()
    peer_tma_atoms: tuple[cute.CopyAtom, ...] = ()
    peer_tma_tensors: tuple[cute.Tensor, ...] = ()
    mBias: cute.Tensor | None = None
    has_bias: cutlass.Constexpr[bool] = False
    rank: Int32 = Int32(0)
    world_size: Int32 = Int32(1)
    bs: Int32 = Int32(1)
    local_seq: Int32 = Int32(0)
    seq_global: Int32 = Int32(0)
    nheads_q: Int32 = Int32(0)
    nheads_k: Int32 = Int32(0)
    nheads_v: Int32 = Int32(0)
    nheads_q_per_rank: Int32 = Int32(0)
    nheads_k_per_rank: Int32 = Int32(0)
    nheads_v_per_rank: Int32 = Int32(0)
    head_dim: Int32 = Int32(0)
    pack_hidden_local: Int32 = Int32(0)
    seq_offsets: cute.Tensor | None = None
    use_seq_offsets: cutlass.Constexpr[bool] = False
    peer_store_enabled: cutlass.Constexpr[bool] = True


class PackQKVEpilogueMixin:
    EpilogueArguments = PackQKVEpilogueArguments
    EpilogueParams = PackQKVEpilogueParams

    @staticmethod
    def make_mnl_view_from_bsn(out_packed_bsn: cute.Tensor) -> cute.Tensor:
        assert cute.rank(out_packed_bsn) == 3, "expected [bs, seq_global, pack_hidden_local]"
        bs, seq_global, pack_hidden_local = out_packed_bsn.shape
        stride_b, stride_s, stride_n = out_packed_bsn.stride
        return cute.make_tensor(
            out_packed_bsn.iterator,
            cute.make_layout(
                (seq_global, pack_hidden_local, bs),
                stride=(stride_s, stride_n, stride_b),
            ),
        )

    def epi_get_output_tensor(self, args: PackQKVEpilogueArguments, *, loc=None, ip=None) -> cute.Tensor | None:
        del loc, ip
        return args.mPeerOuts_mnl[0] if len(args.mPeerOuts_mnl) > 0 else None

    def validate_packqkv_epilogue_args(self, args: PackQKVEpilogueArguments) -> None:
        assert len(args.mPeerOuts_mnl) > 0, "packQKV epilogue requires at least one peer output tensor"
        # Only validate Python-level properties; symbolic Int32 comparisons
        # fail during CuTe compile-time tracing (DSL Boolean cannot convert to bool).
        ref_dtype = args.mPeerOuts_mnl[0].element_type
        r = cute.rank(args.mPeerOuts_mnl[0])
        assert r in (2, 3), f"packQKV peer output must be rank-2 or rank-3, got {r}"
        for peer_out in args.mPeerOuts_mnl[1:]:
            assert peer_out.element_type == ref_dtype, "all peer output tensors must share the same element type"
            assert cute.rank(peer_out) == r, (
                f"all peer output tensors must have the same rank, got {cute.rank(peer_out)} vs {r}"
            )

    def epi_to_underlying_arguments(
        self, args: PackQKVEpilogueArguments, *, loc=None, ip=None
    ) -> PackQKVEpilogueParams:
        self.validate_packqkv_epilogue_args(args)

        def new_stride(t: cute.Tensor):
            return tuple(
                cute.assume(s, divby=128 // t.element_type.width) if not cute.is_static(s) else s for s in t.stride
            )

        mPeerOuts_mnl = tuple(  # mPeerOuts_mnl is a tuple; cannot be passed directly to make_tiled_tma_atom
            cute.make_tensor(
                peer_out.iterator,
                cute.make_layout(peer_out.shape, stride=new_stride(peer_out)),
            )
            for peer_out in args.mPeerOuts_mnl
        )
        peer_tma_pairs = tuple(
            self._make_tma_epi_atoms_and_tensors(
                peer_out,
                self.epi_smem_layout_staged,
                self.epi_tile,
                op_type="store",
            )
            for peer_out in mPeerOuts_mnl
        )
        peer_tma_atoms = tuple(pair[0] for pair in peer_tma_pairs)
        peer_tma_tensors = tuple(pair[1] for pair in peer_tma_pairs)

        # Bias param: assume stride divisibility so cp.async can vectorize.
        has_bias = args.mBias is not None
        mBias_param = None
        if has_bias:
            b = args.mBias
            mBias_param = cute.make_tensor(
                b.iterator,
                cute.make_layout(
                    b.shape,
                    stride=tuple(
                        cute.assume(s, divby=128 // b.element_type.width) if not cute.is_static(s) else s
                        for s in b.stride
                    ),
                ),
            )

        return self.EpilogueParams(
            mPeerOuts_mnl=mPeerOuts_mnl,
            peer_tma_atoms=peer_tma_atoms,
            peer_tma_tensors=peer_tma_tensors,
            mBias=mBias_param,
            has_bias=has_bias,
            rank=args.rank,
            world_size=args.world_size,
            bs=args.bs,
            local_seq=args.local_seq,
            seq_global=args.seq_global,
            nheads_q=args.nheads_q,
            nheads_k=args.nheads_k,
            nheads_v=args.nheads_v,
            nheads_q_per_rank=args.nheads_q_per_rank,
            nheads_k_per_rank=args.nheads_k_per_rank,
            nheads_v_per_rank=args.nheads_v_per_rank,
            head_dim=args.head_dim,
            pack_hidden_local=args.pack_hidden_local,
            seq_offsets=args.seq_offsets,
            use_seq_offsets=args.use_seq_offsets,
            peer_store_enabled=args.peer_store_enabled,
        )

    def epi_has_output_stage(self, params: PackQKVEpilogueParams, *, loc=None, ip=None) -> bool:
        del loc, ip
        return params.peer_store_enabled and len(params.peer_tma_atoms) > 0

    def epi_get_tma_atoms(self, params: PackQKVEpilogueParams, *, loc=None, ip=None) -> list[cute.CopyAtom]:
        del loc, ip
        return list(params.peer_tma_atoms)

    # --- Bias smem plumbing (cp.async + 1D smem stage, broadcast across M) --
    # Mirrors the canonical RowVecLoad pattern from quack.epi_ops:VecLoad.

    @classmethod
    def epi_smem_bytes_per_stage(cls, args, cta_tile_shape_mnk, epi_tile) -> int:
        del epi_tile
        if args is None or args.mBias is None:
            return 0
        return cta_tile_shape_mnk[1] * (args.mBias.element_type.width // 8)

    def epi_get_smem_struct(self, params: PackQKVEpilogueParams):
        has_bias = params.has_bias and params.mBias is not None
        bias_dtype = params.mBias.element_type if has_bias else Float32
        bias_size = self.cta_tile_shape_mnk[1] if has_bias else 0

        class EpiSharedStorage:
            __annotations__ = {
                "s_bias": cute.struct.Align[cute.struct.MemRange[bias_dtype, bias_size], 16],
            }

        return cute.struct(EpiSharedStorage)

    def epi_get_smem_tensors(self, params: PackQKVEpilogueParams, storage) -> tuple:
        if not (params.has_bias and params.mBias is not None):
            return (None,)
        sBias = storage.epi.s_bias.get_tensor(cute.make_layout(self.cta_tile_shape_mnk[1]))
        return (sBias,)

    def epilog_packqkv_peer_gmem_copy_and_partition(
        self,
        atom: cute.CopyAtom,
        peer_mOut_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
    ):
        tDgD_for_tma_partition = cute.zipped_divide(peer_mOut_mnl, epi_tile)
        return copy_utils.tma_get_copy_fn(
            atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=sD,
            dst_tensor=tDgD_for_tma_partition,
            filter_zeros=True,
        )

    def epi_make_peer_gmem_copy_fns(
        self,
        params: PackQKVEpilogueParams,
        sD: cute.Tensor,
        tile_coord_mnkl=None,
        *,
        loc=None,
        ip=None,
    ) -> tuple[Callable, ...]:
        del loc, ip, tile_coord_mnkl
        if not params.peer_store_enabled:
            return tuple()
        return tuple(
            self.epilog_packqkv_peer_gmem_copy_and_partition(
                atom,
                peer_tma_tensor,
                self.epi_tile,
                sD,
            )[0]
            for atom, peer_tma_tensor in zip(params.peer_tma_atoms, params.peer_tma_tensors, strict=True)
        )

    @cute.jit
    def epi_begin(
        self,
        params: PackQKVEpilogueParams,
        epi_smem_tensors: tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: cute.TiledCopy | None,
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        unused_varlen,
        epilogue_barrier,
        tidx: Int32,
    ):
        del unused_varlen
        src_rank = params.rank
        seq_base = _rank_seq_offset(  # for varlen batch
            params.seq_offsets, params.use_seq_offsets, src_rank, params.local_seq
        )

        # Bias staging: cp.async-load N-tile-sized slice of bias into sBias,
        # build a broadcast-strided (M, N) view, partition for the epi r2s copy.
        # No-op when bias is absent (const_expr-eliminated).
        tDsBias = None
        if const_expr(params.has_bias and params.mBias is not None):
            sBias = epi_smem_tensors[0]
            b_dtype = params.mBias.element_type
            num_copy_elems = const_expr(max(32, b_dtype.width)) // b_dtype.width
            num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
            thr_copy = copy_utils.tiled_copy_1d(b_dtype, num_epi_threads, num_copy_elems, is_async=True).get_slice(
                tidx
            )

            # mBias is rank-2 [batch_or_1, N_total]; bias is shared across the
            # batch dim so we always slice row 0.
            mVec = params.mBias[0, None]
            tile_n = self.cta_tile_shape_mnk[1]
            coord_n = tile_coord_mnkl[1]
            gVec = cute.local_tile(mVec, (tile_n,), (coord_n,))
            tVgV = thr_copy.partition_S(gVec)
            tVsV = thr_copy.partition_D(sBias)
            tVcV = thr_copy.partition_S(cute.make_identity_tensor(tile_n))
            limit = min(mVec.shape[0] - coord_n * tile_n, tile_n)
            pred = cute.make_rmem_tensor((1, cute.size(tVsV.shape[1])), Boolean)
            for i in cutlass.range(cute.size(tVsV.shape[1]), unroll_full=True):
                pred[0, i] = tVcV[0, i] < limit
            cute.copy(thr_copy, tVgV, tVsV, pred=pred)

            # Build (M, N) view with broadcast stride (0, 1) so each thread's
            # per-subtile fragment carries the N-coords needed for the bias add.
            bias_broadcast = cute.make_tensor(
                sBias.iterator,
                cute.make_layout(
                    (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]),
                    stride=(0, 1),
                ),
            )
            tDsBias = quack_sm90_utils.partition_for_epilogue(
                bias_broadcast,
                epi_tile=epi_tile,
                tiled_copy=tiled_copy_r2s,
                tidx=tidx,
                reference_src=True,
            )

            # Fence cp.async + cross-warp barrier so bias is visible before
            # subtile loop consumers read it.
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            epilogue_barrier.arrive_and_wait()
        else:
            del epi_smem_tensors, tiled_copy_t2r, tiled_copy_r2s, epilogue_barrier, tidx

        return (
            tile_coord_mnkl[0],
            tile_coord_mnkl[1],
            tile_coord_mnkl[3],
            seq_base,
            tDsBias,
        )

    @cute.jit
    def epi_begin_loop(
        self,
        params: PackQKVEpilogueParams,
        epi_tensors,
        epi_coord: cute.Coord,
    ):
        """Map GEMM epilogue subtile coord → (flat_idx, dst_tm, dst_tn, tDrBias).

        Contiguous layout: peer buffer = [Q region | K region | V region].
        flat_idx = dst_rank * 3 + section_id indexes into the 3*ws copy function array.
        dst_tm/dst_tn are tile coords within the section's 2D view.
        tDrBias is the per-thread bias register fragment for this subtile (or
        None when bias is absent — const_expr-eliminated downstream).
        """
        tile_m, tile_n, _, seq_base, tDsBias = epi_tensors
        epi_m, epi_n = epi_coord

        src_seq0 = tile_m * self.cta_tile_shape_mnk[0] + epi_m * self.epi_tile[0]
        src_n0 = tile_n * self.cta_tile_shape_mnk[1] + epi_n * self.epi_tile[1]

        input_nh = src_n0 // params.head_dim
        hd0 = src_n0 % params.head_dim
        dst_rank, pack_head_local = input_nh_to_dst_rank_and_pack_head(
            input_nh,
            params.nheads_q,
            params.nheads_k,
            params.nheads_v,
            params.nheads_q_per_rank,
            params.nheads_k_per_rank,
            params.nheads_v_per_rank,
            params.world_size,
        )

        # Decompose pack_head_local into section (Q/K/V) + section-local column.
        # pack_head_local ranges: [0, nq_pr) = Q, [nq_pr, nq_pr+nk_pr) = K, rest = V
        section_id = Int32(0)
        section_col0 = Int32(0)
        if pack_head_local < params.nheads_q_per_rank:
            section_id = Int32(0)  # Q
            section_col0 = pack_head_local * params.head_dim + hd0
        elif pack_head_local < params.nheads_q_per_rank + params.nheads_k_per_rank:
            section_id = Int32(1)  # K
            section_col0 = (pack_head_local - params.nheads_q_per_rank) * params.head_dim + hd0
        else:
            section_id = Int32(2)  # V
            section_col0 = (
                pack_head_local - params.nheads_q_per_rank - params.nheads_k_per_rank
            ) * params.head_dim + hd0

        # Decompose src_seq0 into batch index and local sequence offset,
        # since batch is folded into the GEMM M dimension (M = bs * local_seq).
        batch_from_m = src_seq0 // params.local_seq
        seq_in_batch = src_seq0 % params.local_seq
        dst_seq0 = seq_base + seq_in_batch
        # Fold batch into the row dimension for the 2D section output layout:
        # row = batch_from_m * seq_global + dst_seq0
        dst_row = batch_from_m * params.seq_global + dst_seq0
        dst_tm = dst_row // self.epi_tile[0]
        dst_tn = section_col0 // self.epi_tile[1]

        # Flat index into 3*world_size copy function array:
        # [q_r0, k_r0, v_r0, q_r1, k_r1, v_r1, ...]
        flat_idx = dst_rank * Int32(3) + section_id

        # Extract per-thread bias register fragment for this subtile, converted
        # to acc_dtype so the in-place add in epi_visit_subtile stays at fp32.
        tDrBias = None
        if const_expr(tDsBias is not None):
            tDsBias_cur = cute.group_modes(tDsBias, 3, cute.rank(tDsBias))[None, None, None, epi_coord]
            tDrBias_raw = cute.make_rmem_tensor(tDsBias_cur.layout, tDsBias_cur.element_type)
            cute.autovec_copy(cute.filter_zeros(tDsBias_cur), cute.filter_zeros(tDrBias_raw))
            tDrBias = cute.make_rmem_tensor_like(tDrBias_raw, self.acc_dtype)
            tDrBias.store(tDrBias_raw.load().to(self.acc_dtype))

        return flat_idx, dst_tm, dst_tn, tDrBias

    @cute.jit
    def epi_visit_subtile(
        self,
        params: PackQKVEpilogueParams,
        epi_loop_tensors,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None = None,
    ):
        """Add per-N-column bias to the accumulator register fragment in-place.

        Matches the canonical pattern from quack.gemm_default_epi.GemmDefaultEpiMixin
        ("if tDrRowVec is not None: tRS_rD[i] += tDrRowVec[i]").  Returns None —
        the postact path is unused on this mixin.
        """
        del params, tRS_rC
        _flat_idx, _dst_tm, _dst_tn, tDrBias = epi_loop_tensors
        if const_expr(tDrBias is not None):
            for i in cutlass.range(cute.size(tDrBias), unroll_full=True):
                tRS_rD[i] += tDrBias[i]
        return None

    @cute.jit
    def epi_store_subtile(
        self,
        params: PackQKVEpilogueParams,
        epi_loop_tensors,
        gmem_coord: cute.Coord,
        copy_D: Callable | None,
        peer_copy_Ds,
        epi_buffer: Int32,
        epi_idx: Int32 = Int32(0),
    ) -> None:
        del params, gmem_coord, epi_idx
        if const_expr(len(peer_copy_Ds) > 0):
            flat_idx, dst_tm, dst_tn, _tDrBias = epi_loop_tensors
            s2g_cache_hint = Int64(Int64(1).ir_value())  # CU_ACCESS_PROPERTY_STREAMING
            for i in cutlass.range_constexpr(len(peer_copy_Ds)):
                peer_copy_D = peer_copy_Ds[i]
                if const_expr(peer_copy_D is not None):
                    if flat_idx == i:
                        peer_copy_D(src_idx=epi_buffer, dst_idx=(dst_tm, dst_tn), cache_policy=s2g_cache_hint)
        elif const_expr(copy_D is not None):
            copy_D(src_idx=epi_buffer, dst_idx=gmem_coord)

    @cute.jit
    def epi_end(
        self,
        params: PackQKVEpilogueParams,
        epi_tensors,
        epi_tile: cute.Tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl: cute.Coord,
        unused_varlen,
        tidx,
    ) -> None:
        # Cross-GPU visibility is handled by post-GEMM GpuBarrierAll
        # (fence.acq_rel.sys + system-scope atomics), not per-tile signals.
        # TMA store drain is handled by producer_tail() in the base class.
        del (
            params,
            epi_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            unused_varlen,
            tidx,
        )


class GemmWsPreAttnPackQKVSm90(PackQKVEpilogueMixin, GemmWsPreAttnSm90):
    """WS pre-attn GEMM specialization that isolates packQKV epilogue routing.

    This specialization supplies the packQKV coordinate transform, constructs
    per-peer epilogue TMA descriptors during launch preparation, and dispatches
    peer TMA stores from the generic gemm_ws_pre_attn.py epilogue path.
    """

    pass


# NOTE: the PackQKVTileScheduler override is installed lazily by
# ``ws_push._compile_ws_gemm_packqkv`` (alongside ``quack_patch.install()``).
# Keeping the call out of this module's import-time code preserves the
# property that importing this file does not mutate global quack state.
