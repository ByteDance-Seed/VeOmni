"""Custom tile scheduler that round-robins N-axis cluster IDs across rank
segments, used by ``GemmWsPreAttnPackQKVSm90`` so that the device-side
``dst_rank`` formula in ``epi_begin_loop`` can be the trivial
``input_nh // nheads_X_per_rank`` (i.e. ``W_qkv`` keeps its raw
block-per-rank layout) while still keeping every NVLink lane balanced
during the post-GEMM PUSH.

Three-piece invariant — host W layout, device routing formula, scheduler emit
order must all agree:

    host W layout      device dst_rank        scheduler emit order
    ----------------   --------------------   -------------------------
    raw (no shuffle)   input_nh // n_pr       round-robin across ranks

If any one of these is changed in isolation the kernel produces garbage.

The scheduler composes its remap on top of quack's default serpentine
``_swizzle_cta`` (L2-aware) — the super call decides ``(cid_m, cid_n)`` for
locality, and the override only re-labels ``cid_n`` so consecutive emissions
hit different rank segments.

Per-segment edge case (n_clusters_X < world_size):
  Common in GQA K/V (e.g. Hk=8, head_dim=64, cta_tile_n=128, W=8 → only
  4 K-clusters total). Here a single CTA tile already spans 2 ranks' worth
  of heads, and the inner epi-subtile loop naturally distributes dst_rank
  across {0..W-1} within a wave (each cluster's 4 epi-subtiles ping 2 ranks;
  4 K-clusters × 2 = 8 = W). No scheduler remap helps. We detect this case
  at trace time and fall back to identity for the affected segment, leaving
  the natural epi-subtile diversity in charge of balance.
"""

from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from quack.tile_scheduler import (
    TileScheduler,
    TileSchedulerArguments,
)


# Sentinel values used as dataclass defaults below. Dataclass inheritance
# requires the new fields to carry defaults
_UNSET_N_CLUSTERS = -1
_UNSET_WORLD_SIZE = 0


def _segment_per_rank(n_clusters: int, world_size: int) -> int:
    """Sentinel-0 if segment is too small to round-robin; else clusters/rank.

    Single source of truth for the segment-divisibility rule shared by:
      * ``PackQKVTileScheduler.Params.create`` (device path),
      * ``_packqkv_cid_n_remap_cpu`` / ``_physical_cid_n_to_dst_rank_cpu``
        (CPU oracles used by unit tests).

    A return value of ``0`` signals "identity fallback" for that segment
    (n_clusters < world_size). Otherwise we require strict divisibility so
    that every rank receives the same number of clusters.
    """
    if n_clusters < world_size:
        return 0
    assert n_clusters % world_size == 0, f"n_clusters={n_clusters} >= world_size={world_size} but not divisible"
    return n_clusters // world_size


def _segment_remap(c_in_segment: int, n_pr: int, world_size: int) -> int:
    """Round-robin remap inside one segment."""
    round_idx, rank = divmod(c_in_segment, world_size)
    return rank * n_pr + round_idx


@dataclass
class PackQKVTileSchedulerArguments(TileSchedulerArguments):
    """Adds Q/K/V segment cluster counts on top of quack's base arguments.

    All four extra fields are ``Constexpr[int]`` because each is a pure
    function of ``(nheads_X, head_dim, cta_tile_n, cluster_n, world_size)``,
    all of which are baked into the compiled kernel — so they participate in
    the JIT cache key automatically and let the compiler strength-reduce the
    ``divmod(cid, W)`` inside ``_swizzle_cta`` to constant-divisor ops.

    Defaults are intentionally sentinel values rejected by ``Params.create``
    — see ``_UNSET_N_CLUSTERS`` / ``_UNSET_WORLD_SIZE`` above for the why.
    """

    n_clusters_q: cutlass.Constexpr[int] = _UNSET_N_CLUSTERS
    n_clusters_k: cutlass.Constexpr[int] = _UNSET_N_CLUSTERS
    n_clusters_v: cutlass.Constexpr[int] = _UNSET_N_CLUSTERS
    world_size_const: cutlass.Constexpr[int] = _UNSET_WORLD_SIZE


class PackQKVTileScheduler(TileScheduler):
    @dataclass
    class Params(TileScheduler.Params):
        n_clusters_q: cutlass.Constexpr[int] = _UNSET_N_CLUSTERS
        n_clusters_k: cutlass.Constexpr[int] = _UNSET_N_CLUSTERS
        n_clusters_qk: cutlass.Constexpr[int] = _UNSET_N_CLUSTERS
        n_clusters_q_per_rank: cutlass.Constexpr[int] = 0
        n_clusters_k_per_rank: cutlass.Constexpr[int] = 0
        n_clusters_v_per_rank: cutlass.Constexpr[int] = 0
        world_size_const: cutlass.Constexpr[int] = _UNSET_WORLD_SIZE

        @staticmethod
        @cute.jit
        def create(args: "PackQKVTileSchedulerArguments", *, loc=None, ip=None) -> "PackQKVTileScheduler.Params":
            # Reject the sentinel defaults: callers MUST set all four explicitly.
            assert args.world_size_const >= 1, (
                f"PackQKVTileSchedulerArguments.world_size_const must be set explicitly "
                f"(>=1); got {args.world_size_const}"
            )
            assert args.n_clusters_q >= 1, (
                f"PackQKVTileSchedulerArguments.n_clusters_q must be set explicitly (>=1); got {args.n_clusters_q}"
            )
            assert args.n_clusters_k >= 1, (
                f"PackQKVTileSchedulerArguments.n_clusters_k must be set explicitly (>=1); got {args.n_clusters_k}"
            )
            assert args.n_clusters_v >= 1, (
                f"PackQKVTileSchedulerArguments.n_clusters_v must be set explicitly (>=1); got {args.n_clusters_v}"
            )
            # Cluster N must be 1 for the cluster-level remap to coincide with
            # the per-CTA tile_n dst_rank routing in ``epi_begin_loop``.
            # ``_packqkv_get_scheduler_arguments`` already validates this earlier
            # with a richer error message; this is a defensive last-line check
            # in case ``Params.create`` is ever invoked from another code path.
            assert args.cluster_shape_mnk[1] == 1, (
                "PackQKVTileScheduler currently assumes cluster_shape_n == 1; "
                f"got cluster_shape_mnk={args.cluster_shape_mnk}."
            )
            # Per-segment: round-robin only when there are enough clusters to
            # cover all ranks AND they divide evenly; otherwise identity
            W_ = args.world_size_const
            n_pr_q = _segment_per_rank(args.n_clusters_q, W_)
            n_pr_k = _segment_per_rank(args.n_clusters_k, W_)
            n_pr_v = _segment_per_rank(args.n_clusters_v, W_)

            base = TileScheduler.Params.create(args, loc=loc, ip=ip)
            return PackQKVTileScheduler.Params(
                problem_shape_ncluster_mnl=base.problem_shape_ncluster_mnl,
                raster_order=base.raster_order,
                num_clusters_per_problem_fdd=base.num_clusters_per_problem_fdd,
                num_groups_regular=base.num_groups_regular,
                group_size_fdd=base.group_size_fdd,
                group_size_tail_fdd=base.group_size_tail_fdd,
                num_clusters_in_group_fdd=base.num_clusters_in_group_fdd,
                tile_count_semaphore=base.tile_count_semaphore,
                batch_idx_permute=base.batch_idx_permute,
                cluster_shape_mn=base.cluster_shape_mn,
                persistence_mode=base.persistence_mode,
                n_clusters_q=args.n_clusters_q,
                n_clusters_k=args.n_clusters_k,
                n_clusters_qk=args.n_clusters_q + args.n_clusters_k,
                n_clusters_q_per_rank=n_pr_q,
                n_clusters_k_per_rank=n_pr_k,
                n_clusters_v_per_rank=n_pr_v,
                world_size_const=args.world_size_const,
            )

    @staticmethod
    def to_underlying_arguments(
        args: PackQKVTileSchedulerArguments, *, loc=None, ip=None
    ) -> "PackQKVTileScheduler.Params":
        return PackQKVTileScheduler.Params.create(args, loc=loc, ip=ip)

    @cute.jit
    def _swizzle_cta(self, cluster_id_in_problem: Int32, *, loc=None, ip=None) -> Tuple[Int32, Int32]:
        # Step 1 — defer to quack's L2-aware serpentine swizzle for the
        # base (cid_m, cid_n) decision. The super call honours raster_order,
        # max_swizzle_size and serpentine inner ordering exactly as upstream.
        cid_m, cid_n_default = super()._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)

        # Step 2 — re-label cid_n so that consecutive scheduler emissions land
        params = self.params
        W = params.world_size_const
        if cid_n_default < params.n_clusters_q:
            if const_expr(params.n_clusters_q_per_rank == 0):
                cid_n_remapped = cid_n_default
            else:
                cid_n_remapped = _segment_remap(cid_n_default, params.n_clusters_q_per_rank, W)
        elif cid_n_default < params.n_clusters_qk:
            if const_expr(params.n_clusters_k_per_rank == 0):
                cid_n_remapped = cid_n_default
            else:
                cid_n_remapped = params.n_clusters_q + _segment_remap(
                    cid_n_default - params.n_clusters_q,
                    params.n_clusters_k_per_rank,
                    W,
                )
        else:
            if const_expr(params.n_clusters_v_per_rank == 0):
                cid_n_remapped = cid_n_default
            else:
                cid_n_remapped = params.n_clusters_qk + _segment_remap(
                    cid_n_default - params.n_clusters_qk,
                    params.n_clusters_v_per_rank,
                    W,
                )

        return cid_m, cid_n_remapped
