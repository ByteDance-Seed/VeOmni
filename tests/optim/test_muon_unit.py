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

"""Single-process unit tests for ``veomni.optim.muon``."""

from __future__ import annotations

import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Partial, Replicate, Shard

from tests.tools.training_utils import make_eager_ops_config
from veomni.models.configuration_base import as_veomni_config
from veomni.optim import muon as muon_impl
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_device_type,
    get_gpu_compute_capability,
    get_torch_device,
)


muon_module = pytest.importorskip("torch.optim._muon")
from torch.optim import Muon as UpstreamMuon  # noqa: E402
from torch.optim._muon import _adjust_lr as upstream_adjust_lr  # noqa: E402
from torch.optim._muon import _zeropower_via_newtonschulz as upstream_ns  # noqa: E402

from veomni.optim.muon import (  # noqa: E402
    DEFAULT_NS_COEFFICIENTS,
    DEFAULT_NS_STEPS,
    DistributedMuon,
    _fsdp_all2all_submesh,
    _shard_row_sizes,
    batched_gram_newton_schulz,
    batched_newton_schulz,
    infer_head_block_counts,
    run_newton_schulz,
    split_muon_adamw_params,
)


def _toy_model() -> nn.Module:
    """Tiny model mixing 2D linears with 1D and embedding params."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Embedding(8, 4),  # 2D weight, force-AdamW by name
        nn.LayerNorm(4),  # 1D weight + 1D bias
        nn.Linear(4, 8, bias=True),  # 2D weight (Muon), 1D bias (AdamW)
        nn.Linear(8, 4, bias=False),  # 2D weight (Muon), no bias
    )


def _sample_2d(M: int, K: int, dtype: torch.dtype = torch.float32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(M, K, generator=g, dtype=dtype)


_QKV_MODULES = ("q_proj", "k_proj", "v_proj")


class _FakeAttention(nn.Module):
    """GQA attention plus ``o_proj``/``q_a_proj``, which head splitting must not touch.

    ``hidden == num_heads * head_dim`` is deliberate: ``o_proj`` then resolves to a
    valid head layout, so it can only stay out of the split by name.
    """

    def __init__(self, hidden: int = 32, num_heads: int = 8, num_kv_heads: int = 2, head_dim: int = 4):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.q_a_proj = nn.Linear(hidden, 4 * head_dim, bias=False)


def _attention_model(**kwargs) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Module()
    model.embed_tokens = nn.Embedding(8, 32)
    model.self_attn = _FakeAttention(**kwargs)
    return model


_TOY_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "toy_config"


def _dsv4_csa_attention_model() -> nn.Module:
    """A real ``DeepseekV4Attention`` on a CSA layer, so the nesting is the real one.

    What the nested-name rule keys on is that the indexer sits at
    ``self_attn.compressor.indexer`` while naming a projection the enclosing MLA also
    names; a synthetic stand-in would pin a hand-built path instead of the shipped one.

    ``index_n_heads`` / ``index_head_dim`` are raised to the 4-layer checkpoint's
    64 x 128 (the toy config ships 8 x 128) so the indexer's head count differs from
    the main attention's 8 and one assertion can tell the two apart.
    """
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4 import modeling_deepseek_v4 as modeling

    config = AutoConfig.from_pretrained(str(_TOY_CONFIG_ROOT / "deepseek_v4_toy"))
    as_veomni_config(config, validate=False)._set_ops_implementation(make_eager_ops_config())
    config.index_n_heads = 64
    config.index_head_dim = 128
    torch.manual_seed(0)
    model = nn.Module()
    model.self_attn = modeling.DeepseekV4Attention(config, config.layer_types.index("compressed_sparse_attention"))
    return model


def _glm_dsa_attention_model() -> nn.Module:
    """A real ``GlmMoeDsaAttention``, whose indexer names ``wq_b`` and never collides."""
    from transformers import AutoConfig

    from veomni.models.transformers.glm_moe_dsa import modeling_glm_moe_dsa as modeling

    config = AutoConfig.from_pretrained(str(_TOY_CONFIG_ROOT / "glm_moe_dsa_toy"))
    as_veomni_config(config, validate=False)._set_ops_implementation(make_eager_ops_config())
    torch.manual_seed(0)
    model = nn.Module()
    model.self_attn = modeling.GlmMoeDsaAttention(config, 0)
    return model


def _head_split_muon(param: nn.Parameter, head_blocks: int, adjust_lr_fn=None, lr: float = 1.0) -> DistributedMuon:
    """Muon over one param with momentum disabled, so update == orthogonalized grad."""
    return DistributedMuon(
        [{"params": [param], "head_blocks": head_blocks}],
        lr=lr,
        weight_decay=0.0,
        momentum=0.0,
        nesterov=False,
        adjust_lr_fn=adjust_lr_fn,
        ns_implementation="std",
    )


class TestSplit:
    def test_split_partitions_correctly(self):
        model = _toy_model()
        muon, adamw, muon_names, adamw_names = split_muon_adamw_params(model)

        assert sorted(muon_names) == sorted(["2.weight", "3.weight"])

        assert "0.weight" in adamw_names  # embedding
        assert "1.weight" in adamw_names  # LN weight (1D)
        assert "1.bias" in adamw_names  # LN bias (1D)
        assert "2.bias" in adamw_names  # Linear bias (1D)

        assert set(muon_names).isdisjoint(set(adamw_names))
        assert len(muon) == len(muon_names)
        assert len(adamw) == len(adamw_names)

    def test_no_decay_modules_overrides_to_adamw(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        muon, adamw, muon_names, adamw_names = split_muon_adamw_params(
            model,
            no_decay_modules=["Linear"],
        )
        assert muon_names == []
        assert "0.weight" in adamw_names
        assert "0.bias" in adamw_names
        assert "1.weight" in adamw_names

    def test_frozen_params_excluded(self):
        model = nn.Linear(4, 4)
        for p in model.parameters():
            p.requires_grad_(False)
        muon, adamw, *_ = split_muon_adamw_params(model)
        assert muon == [] and adamw == []

    def test_routing_and_gating_matrices_go_to_adamw(self):
        """A matrix that selects or mixes is not a feature transform, so NS must not touch it.

        Covers the names the shipped models actually use: MoE router gates, scalar
        expert gates, hyper-connection stream mixing, the DSA indexer's per-head
        scores, and the compressors' softmax gates. The sibling feature projections
        in each of those modules must stay on Muon, which is what the scoped
        `compressor.gate_proj` / `indexer.gate_proj` entries are for.
        """
        model = nn.Module()
        layer = nn.Module()
        layer.attn_hc = nn.ParameterDict({"fn": nn.Parameter(torch.randn(24, 64))})
        layer.ffn_hc = nn.ParameterDict({"fn": nn.Parameter(torch.randn(24, 64))})
        layer.mlp = nn.Module()
        layer.mlp.gate = nn.Linear(16, 8, bias=False)
        layer.mlp.router = nn.Linear(16, 8, bias=False)
        layer.mlp.shared_expert_gate = nn.Linear(16, 1, bias=False)
        layer.mlp.experts = nn.ParameterDict({"gate_up_proj": nn.Parameter(torch.randn(4, 16, 16))})
        layer.mlp.shared_experts = nn.ModuleDict({"gate_proj": nn.Linear(16, 8, bias=False)})
        layer.self_attn = nn.Module()
        layer.self_attn.compressor = nn.ModuleDict(
            {
                "kv_proj": nn.Linear(16, 8, bias=False),
                "gate_proj": nn.Linear(16, 8, bias=False),
            }
        )
        layer.self_attn.compressor.indexer = nn.ModuleDict(
            {
                "q_b_proj": nn.Linear(16, 32, bias=False),
                "kv_proj": nn.Linear(16, 8, bias=False),
                "gate_proj": nn.Linear(16, 8, bias=False),
                "weights_proj": nn.Linear(16, 4, bias=False),
            }
        )
        model.layers = nn.ModuleList([layer])
        model.hc_head = nn.ParameterDict({"hc_fn": nn.Parameter(torch.randn(4, 64))})

        _, _, muon_names, adamw_names = split_muon_adamw_params(model)

        assert set(adamw_names) == {
            "layers.0.attn_hc.fn",
            "layers.0.ffn_hc.fn",
            "layers.0.mlp.gate.weight",
            "layers.0.mlp.router.weight",
            "layers.0.mlp.shared_expert_gate.weight",
            "layers.0.self_attn.compressor.gate_proj.weight",
            "layers.0.self_attn.compressor.indexer.gate_proj.weight",
            "layers.0.self_attn.compressor.indexer.weights_proj.weight",
            "hc_head.hc_fn",
        }
        assert set(muon_names) == {
            "layers.0.mlp.experts.gate_up_proj",
            "layers.0.mlp.shared_experts.gate_proj.weight",
            "layers.0.self_attn.compressor.kv_proj.weight",
            "layers.0.self_attn.compressor.indexer.q_b_proj.weight",
            "layers.0.self_attn.compressor.indexer.kv_proj.weight",
        }


class TestParamShapeEligibility:
    """``DistributedMuon`` must accept 2D/3D and reject 1D/4D+."""

    @pytest.mark.parametrize(
        "shape, accepted",
        [
            ((4,), False),  # 1D bias / norm
            ((4, 8), True),  # 2D dense linear
            ((4, 8, 16), True),  # 3D MoE expert stack
            ((2, 4, 8, 16), False),  # 4D conv weight
        ],
    )
    def test_eligibility(self, shape, accepted):
        p = nn.Parameter(torch.randn(*shape))
        if accepted:
            opt = DistributedMuon([p], lr=1e-3)
            p.grad = torch.randn_like(p)
            opt.step()
            assert torch.isfinite(p).all()
        else:
            with pytest.raises(ValueError, match="2D and 3D"):
                DistributedMuon([p], lr=1e-3)


class _FakeMesh:
    """DeviceMesh stand-in exposing only what submesh resolution touches."""

    def __init__(self, sizes, dim_names):
        self._sizes = sizes
        self.ndim = len(sizes)
        self.mesh_dim_names = dim_names
        self.sliced_with = None

    def size(self, dim):
        return self._sizes[dim]

    def __getitem__(self, name):
        self.sliced_with = name
        return _FakeMesh((self._sizes[self.mesh_dim_names.index(name)],), (name,))


class TestFsdpAllToAllEligibility:
    def test_empty_rank_shards_remain_all_to_all_eligible(self):
        world_size = 32
        param = SimpleNamespace(
            device_mesh=_FakeMesh((world_size,), ("dp_shard",)),
            placements=(Shard(0),),
            shape=(24, 16384),
        )

        assert _shard_row_sizes(param.shape[0], world_size) == [1] * 24 + [0] * 8
        assert _fsdp_all2all_submesh(param) is not None

    def test_hsdp_mesh_resolves_to_the_shard_dim(self):
        """HSDP params are (Replicate(), Shard(0)); the path runs on dp_shard_sp."""
        mesh = _FakeMesh((2, 8), ("dp_replicate", "dp_shard_sp"))
        param = SimpleNamespace(device_mesh=mesh, placements=(Replicate(), Shard(0)))

        submesh = _fsdp_all2all_submesh(param)

        assert mesh.sliced_with == "dp_shard_sp"
        assert submesh.ndim == 1
        assert submesh.size(0) == 8

    def test_pending_reduction_is_not_eligible(self):
        """A Partial() dim owes a reduction, so the gather cannot be skipped."""
        mesh = _FakeMesh((2, 8), ("dp_replicate", "dp_shard_sp"))
        param = SimpleNamespace(device_mesh=mesh, placements=(Partial(), Shard(0)))

        assert _fsdp_all2all_submesh(param) is None
        assert mesh.sliced_with is None

    def test_bucket_key_separates_local_dtypes(self):
        mesh = object()
        fp32 = SimpleNamespace(device_mesh=mesh, to_local=lambda: torch.empty(1, dtype=torch.float32))
        bf16 = SimpleNamespace(device_mesh=mesh, to_local=lambda: torch.empty(1, dtype=torch.bfloat16))
        bucket_key = getattr(muon_impl, "_fsdp_all2all_bucket_key", lambda update, mesh: mesh)

        assert bucket_key(fp32, mesh) != bucket_key(bf16, mesh)


class TestIntermediateLifetime:
    def test_non_all_to_all_orthos_are_released_per_parameter(self):
        params = [nn.Parameter(torch.randn(4, 4)) for _ in range(8)]
        opt = DistributedMuon(params, lr=1e-3, nesterov=False)
        live = 0
        peak = 0
        calls = 0

        def fake_compute_ortho(update, kind, **kwargs):
            nonlocal calls, live, peak

            ortho = torch.zeros_like(update)
            calls += 1
            live += 1
            peak = max(peak, live)

            def release():
                nonlocal live
                live -= 1

            weakref.finalize(ortho, release)
            return ortho

        opt._compute_ortho = fake_compute_ortho
        for p in params:
            p.grad = torch.randn_like(p)

        opt.step()

        assert calls == len(params)
        assert peak <= 2
        assert live == 0


class TestNumerics:
    """Plain-tensor parity with upstream ``torch.optim.Muon``."""

    def test_single_step_matches_upstream(self):
        torch.manual_seed(123)
        w = torch.randn(8, 16, dtype=torch.float32)
        a = nn.Parameter(w.clone())
        b = nn.Parameter(w.clone())

        opt_up = UpstreamMuon(
            [a],
            lr=1e-2,
            weight_decay=0.05,
            momentum=0.9,
            nesterov=True,
            adjust_lr_fn="match_rms_adamw",
        )
        opt_ours = DistributedMuon(
            [b],
            lr=1e-2,
            weight_decay=0.05,
            momentum=0.9,
            nesterov=True,
            adjust_lr_fn="match_rms_adamw",
            ns_implementation="std",  # match upstream torch.optim.Muon NS path
        )

        torch.manual_seed(7)
        grad = torch.randn_like(a)
        a.grad = grad.clone()
        b.grad = grad.clone()

        opt_up.step()
        opt_ours.step()

        torch.testing.assert_close(a.detach(), b.detach(), atol=0.0, rtol=0.0)

    def test_state_dict_roundtrip(self):
        """Step + serialize + restore preserves both Muon and AdamW state."""
        from veomni.optim import build_optimizer

        model_a = _toy_model()
        opt_a = build_optimizer(
            model_a,
            lr=1e-4,
            weight_decay=0.01,
            optimizer_type="muon",
            muon_kwargs={"lr": 1e-2, "adjust_lr_fn": "match_rms_adamw"},
        )
        for p in model_a.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        opt_a.step()
        sd_before = opt_a.state_dict()

        flat_keys = set(sd_before.keys())
        assert any("momentum_buffer" in k for k in flat_keys), (
            f"expected Muon momentum_buffer in state_dict; got {sorted(flat_keys)[:8]}"
        )
        assert any("exp_avg" in k for k in flat_keys), (
            f"expected AdamW exp_avg in state_dict; got {sorted(flat_keys)[:8]}"
        )

        model_b = _toy_model()
        model_b.load_state_dict(model_a.state_dict())
        opt_b = build_optimizer(
            model_b,
            lr=1e-4,
            weight_decay=0.01,
            optimizer_type="muon",
            muon_kwargs={"lr": 1e-2, "adjust_lr_fn": "match_rms_adamw"},
        )
        opt_b.load_state_dict(sd_before)

        sd_after = opt_b.state_dict()
        assert set(sd_before.keys()) == set(sd_after.keys())
        for k, v_before in sd_before.items():
            v_after = sd_after[k]
            if isinstance(v_before, torch.Tensor):
                torch.testing.assert_close(v_before, v_after, atol=0.0, rtol=0.0, msg=f"state_dict mismatch at {k}")


class TestBuildOptimizer:
    def test_build_returns_multi_optimizer(self):
        from veomni.optim import build_optimizer
        from veomni.optim.optimizer import MultiOptimizer

        model = _toy_model()
        opt = build_optimizer(
            model,
            lr=1e-4,
            weight_decay=0.01,
            optimizer_type="muon",
            muon_kwargs={"lr": 5e-3, "adjust_lr_fn": "match_rms_adamw"},
        )
        assert isinstance(opt, MultiOptimizer)
        assert "muon" in opt.optimizers_dict and "adamw" in opt.optimizers_dict
        assert isinstance(opt.optimizers_dict["muon"], DistributedMuon)
        assert isinstance(opt.optimizers_dict["adamw"], torch.optim.AdamW)

    def test_build_no_2d_params_raises(self):
        from veomni.optim import build_optimizer

        model = nn.LayerNorm(4)
        with pytest.raises(ValueError, match="no eligible 2D/3D parameters"):
            build_optimizer(model, optimizer_type="muon", muon_kwargs={"lr": 1e-3})


class TestBatchedNS:
    """Math primitive parity + contract checks."""

    def test_2d_byte_parity_with_upstream(self):
        """2D batched NS must equal upstream's 2D NS bit-for-bit."""
        x = _sample_2d(8, 16, dtype=torch.float32, seed=42)
        out_upstream = upstream_ns(x.clone(), DEFAULT_NS_COEFFICIENTS, DEFAULT_NS_STEPS, eps=1e-7)
        out_ours = batched_newton_schulz(
            x.clone(),
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=DEFAULT_NS_STEPS,
            eps=1e-7,
            compute_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(out_ours.to(torch.bfloat16), out_upstream, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(
        "shape",
        [
            (3, 8, 16),  # tall expert stack
            (4, 16, 8),  # wide expert stack (transpose path)
            (2, 32, 32),  # square expert stack
        ],
    )
    def test_3d_matches_per_slice(self, shape):
        """3D batched NS must match a Python loop calling 2D NS per slice."""
        N, M, K = shape
        g = torch.Generator(device="cpu").manual_seed(123)
        x = torch.randn(N, M, K, generator=g, dtype=torch.float32)

        per_slice = torch.stack(
            [upstream_ns(x[i].clone(), DEFAULT_NS_COEFFICIENTS, DEFAULT_NS_STEPS, eps=1e-7) for i in range(N)],
            dim=0,
        )
        batched = batched_newton_schulz(
            x.clone(),
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=DEFAULT_NS_STEPS,
            eps=1e-7,
            compute_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(batched.to(torch.bfloat16), per_slice, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize(
        "case",
        [
            {"ns_steps": 150, "match": "less than 100"},
            {"ns_coefficients": (1.0, 2.0), "match": "Per-step ns_coefficients|exactly 3|Coefficients"},
        ],
        ids=["too_many_steps", "bad_coefficients"],
    )
    def test_contracts_reject_invalid(self, case):
        kwargs = {k: v for k, v in case.items() if k != "match"}
        with pytest.raises(ValueError, match=case["match"]):
            batched_newton_schulz(_sample_2d(4, 8), **kwargs)

    @pytest.mark.parametrize("shape", [(4, 8), (3, 4, 8)], ids=["2d", "3d"])
    def test_zero_input_safe(self, shape):
        out = batched_newton_schulz(torch.zeros(*shape))
        assert torch.isfinite(out).all()
        assert out.shape == tuple(shape)


class TestGramNewtonSchulz:
    """Gram-NS pure torch path + optional Dao-AILab package kernels."""

    def test_rectangular_close_to_standard_same_coeffs(self):
        x = _sample_2d(32, 128, dtype=torch.float32, seed=7)
        std = batched_newton_schulz(
            x.clone(),
            DEFAULT_NS_COEFFICIENTS,
            DEFAULT_NS_STEPS,
            eps=1e-7,
            compute_dtype=torch.float32,
        )
        # No restart + same coeffs => Gram rearrangement should match standard NS.
        gram = batched_gram_newton_schulz(
            x.clone(),
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=DEFAULT_NS_STEPS,
            eps=1e-7,
            reset_iterations=(),
            compute_dtype=torch.float32,
        )
        torch.testing.assert_close(gram.float(), std.float(), atol=5e-3, rtol=5e-3)

    def test_3d_batch_shape(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 16, 64, generator=g, dtype=torch.float32)
        out = batched_gram_newton_schulz(x, reset_iterations=(2,), compute_dtype=torch.float32)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_run_dispatch_gram_torch(self):
        x = _sample_2d(16, 64, dtype=torch.float32, seed=1)
        out = run_newton_schulz(
            x,
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=5,
            ns_implementation="gram",
            gram_ns_reset_iterations=(2,),
            compute_dtype=torch.float32,
        )
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    @pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA required")
    def test_package_kernel_or_hardware_fallback(self, monkeypatch):
        import veomni.optim.muon as muon_mod

        pytest.importorskip("gram_newton_schulz")
        monkeypatch.setattr(muon_mod, "_GRAM_QUACK_FALLBACK_WARNED", False)
        x = torch.randn(2, 256, 1024, device=get_device_type(), dtype=torch.bfloat16)
        out = run_newton_schulz(
            x,
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=5,
            ns_implementation="gram_quack",
            gram_ns_reset_iterations=(2,),
        )
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert muon_mod._GRAM_QUACK_FALLBACK_WARNED is (get_gpu_compute_capability(x.device) < 90)

    def test_distributed_muon_accepts_gram_flags(self):
        p = nn.Parameter(torch.randn(8, 16))
        opt = DistributedMuon(
            [p],
            lr=1e-3,
            ns_implementation="gram",
            gram_ns_reset_iterations=(2,),
        )
        p.grad = torch.randn_like(p)
        opt.step()
        assert torch.isfinite(p).all()

    @pytest.mark.parametrize("use_dcp", [False, True], ids=["plain", "dcp"])
    def test_resume_keeps_the_configured_expert_ns_budget(self, use_dcp):
        """The chunking switch must not be resumable state, on either restore path.

        Both directions fail differently: a keyless checkpoint makes torch's DCP
        unflatten raise ``KeyError`` if the live group declares the key, and a
        checkpoint carrying a value would overwrite this run's, so a job lowering
        the budget to fit could resume with the old larger one and OOM.
        """
        import torch.nn as nn

        if use_dcp:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_optimizer_state_dict,
                set_optimizer_state_dict,
            )

        def make(budget):
            model = nn.Linear(4, 4, bias=False)
            kwargs = {} if budget is None else {"expert_ns_max_experts": budget}
            return model, DistributedMuon(model.parameters(), lr=1e-3, **kwargs)

        def snapshot(budget):
            model, opt = make(budget)
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            opt.step()
            if use_dcp:
                return get_optimizer_state_dict(
                    model, opt, options=StateDictOptions(flatten_optimizer_state_dict=True)
                )
            return opt.state_dict()

        def restore(budget, saved):
            model, opt = make(budget)
            if use_dcp:
                set_optimizer_state_dict(
                    model, opt, optim_state_dict=saved, options=StateDictOptions(flatten_optimizer_state_dict=True)
                )
            else:
                opt.load_state_dict(saved)
            return opt

        # Nothing about the switch is persisted, whichever value produced the checkpoint.
        for budget in (None, 8):
            saved = snapshot(budget)
            if use_dcp:
                assert not any("expert_ns_max_experts" in key for key in saved)
            else:
                assert "expert_ns_max_experts" not in saved["param_groups"][0]

        # Saved off (keyless) -> configured 3: no KeyError, the configuration wins.
        assert restore(3, snapshot(None))._expert_ns_max_experts == 3
        # Saved 8 -> configured 1: the configuration wins.
        assert restore(1, snapshot(8))._expert_ns_max_experts == 1

        # Validation still rejects a value below the "off" sentinel.
        with pytest.raises(ValueError, match="must be >= -1"):
            DistributedMuon([nn.Parameter(torch.randn(8, 16))], lr=1e-3, expert_ns_max_experts=-2)


class TestMuonLrResolution:
    def test_muon_lr_inherits_adamw_lr_under_match_rms(self):
        from veomni.optim import build_optimizer

        model = _toy_model()
        opt = build_optimizer(
            model,
            lr=1.5e-4,
            weight_decay=0.01,
            optimizer_type="muon",
            muon_kwargs={"adjust_lr_fn": "match_rms_adamw"},  # no explicit lr
        )
        muon_opt = opt.optimizers_dict["muon"]
        assert muon_opt.param_groups[0]["lr"] == pytest.approx(1.5e-4)

    def test_muon_lr_original_defaults_to_25x(self):
        from veomni.optim import build_optimizer

        model = _toy_model()
        opt = build_optimizer(
            model,
            lr=1e-4,
            weight_decay=0.01,
            optimizer_type="muon",
            muon_kwargs={"adjust_lr_fn": "original"},
        )
        muon_opt = opt.optimizers_dict["muon"]
        assert muon_opt.param_groups[0]["lr"] == pytest.approx(2.5e-3)

    def test_muon_lr_explicit_wins(self):
        from veomni.optim import build_optimizer

        model = _toy_model()
        opt = build_optimizer(
            model,
            lr=1e-4,
            weight_decay=0.01,
            optimizer_type="muon",
            muon_kwargs={"lr": 3e-3, "adjust_lr_fn": "match_rms_adamw"},
        )
        muon_opt = opt.optimizers_dict["muon"]
        assert muon_opt.param_groups[0]["lr"] == pytest.approx(3e-3)


class TestMuonKwargsFromOptimizerConfig:
    """``build_optimizer`` reads Muon knobs off ``OptimizerConfig`` itself.

    Trainers used to unpack the ``muon_*`` fields and pre-resolve the Muon LR
    before calling ``build_optimizer``. These tests pin the equivalent results
    now that the whole config is handed over instead.
    """

    def _muon_config(self, **overrides):
        from veomni.arguments import OptimizerConfig

        kwargs = {"type": "muon", "lr": 1e-4, "muon_ns_implementation": "std"}
        kwargs.update(overrides)
        return OptimizerConfig(**kwargs)

    def _build(self, optimizer_config, **kwargs):
        from veomni.optim import build_optimizer

        kwargs.setdefault("lr", optimizer_config.lr)
        kwargs.setdefault("weight_decay", optimizer_config.weight_decay)
        return build_optimizer(
            _toy_model(),
            optimizer_type=optimizer_config.type,
            optimizer_config=optimizer_config,
            **kwargs,
        )

    def test_unset_muon_lr_inherits_the_adamw_lr_under_match_rms(self):
        # match_rms_adamw is also the builder's fallback default, so pin a knob
        # that is not (ns_steps) to prove the config was actually consulted.
        config = self._muon_config(muon_adjust_lr_fn="match_rms_adamw", muon_ns_steps=3)
        group = self._build(config, lr=7e-4).optimizers_dict["muon"].param_groups[0]
        assert group["lr"] == pytest.approx(7e-4)
        assert group["ns_steps"] == 3

    def test_unset_muon_lr_is_25x_the_adamw_lr_under_original(self):
        config = self._muon_config(muon_adjust_lr_fn="original")
        opt = self._build(config)
        assert opt.optimizers_dict["muon"].param_groups[0]["lr"] == pytest.approx(2.5e-3)

    def test_explicit_muon_lr_wins_over_inheritance(self):
        config = self._muon_config(muon_lr=3e-3)
        opt = self._build(config)
        assert opt.optimizers_dict["muon"].param_groups[0]["lr"] == pytest.approx(3e-3)

    def test_muon_lr_survives_yaml_exponent_literals(self):
        # PyYAML resolves "3e-3" to a str, and OptimizerConfig does not coerce.
        config = self._muon_config(muon_lr="3e-3")
        group = self._build(config).optimizers_dict["muon"].param_groups[0]
        assert isinstance(group["lr"], float)
        assert group["lr"] == pytest.approx(3e-3)

    def test_non_lr_muon_knobs_reach_the_optimizer(self):
        config = self._muon_config(
            muon_momentum=0.8,
            muon_nesterov=False,
            muon_weight_decay=0.05,
            muon_ns_steps=3,
            muon_eps=1e-5,
            muon_ns_coefficients=[3.0, -4.0, 2.0],
        )
        group = self._build(config).optimizers_dict["muon"].param_groups[0]
        assert group["momentum"] == pytest.approx(0.8)
        assert group["nesterov"] is False
        assert group["weight_decay"] == pytest.approx(0.05)
        assert group["ns_steps"] == 3
        assert group["eps"] == pytest.approx(1e-5)
        assert tuple(group["ns_coefficients"]) == (3.0, -4.0, 2.0)

    def test_muon_and_adamw_groups_keep_separate_weight_decay(self):
        config = self._muon_config(muon_lr=3e-3, weight_decay=0.02, muon_weight_decay=0.05)
        opt = self._build(config)
        assert opt.optimizers_dict["muon"].param_groups[0]["weight_decay"] == pytest.approx(0.05)
        adamw_groups = opt.optimizers_dict["adamw"].param_groups
        assert max(group["weight_decay"] for group in adamw_groups) == pytest.approx(0.02)

    def test_explicit_muon_kwargs_override_the_config_key_by_key(self):
        config = self._muon_config(muon_lr=3e-3, muon_momentum=0.8)
        opt = self._build(config, muon_kwargs={"lr": 7e-3})
        group = opt.optimizers_dict["muon"].param_groups[0]
        assert group["lr"] == pytest.approx(7e-3)
        # Overriding the LR must not reset the knobs the config supplied.
        assert group["momentum"] == pytest.approx(0.8)

    def test_empty_muon_kwargs_does_not_discard_the_config(self):
        config = self._muon_config(muon_lr=3e-3, muon_momentum=0.8)
        group = self._build(config, muon_kwargs={}).optimizers_dict["muon"].param_groups[0]
        assert group["lr"] == pytest.approx(3e-3)
        assert group["momentum"] == pytest.approx(0.8)

    def test_head_group_size_reaches_the_builder(self):
        config = self._muon_config(muon_head_group_size=2)  # no muon_head_split_modules
        with pytest.raises(ValueError, match="muon_head_split_modules"):
            self._build(config)


class TestHeadSplitInference:
    """``infer_head_block_counts`` must key off declared head layouts, not shapes."""

    def test_only_listed_modules_are_eligible(self):
        blocks = infer_head_block_counts(_attention_model(), head_group_size=1, module_names=_QKV_MODULES)

        # o_proj / q_a_proj stay out despite row counts that are multiples of head_dim.
        assert blocks == {
            "self_attn.q_proj.weight": 8,
            "self_attn.k_proj.weight": 2,
            "self_attn.v_proj.weight": 2,
        }

    def test_module_list_is_required(self):
        with pytest.raises(ValueError, match="explicit list of attention projection module names"):
            infer_head_block_counts(_attention_model(), head_group_size=1, module_names=())

    @pytest.mark.parametrize(
        "head_group_size, expected",
        [
            (2, {"self_attn.q_proj.weight": 4, "self_attn.k_proj.weight": 1, "self_attn.v_proj.weight": 1}),
            (4, {"self_attn.q_proj.weight": 2}),  # 2 KV heads are not divisible by 4
            (8, {}),  # one block per matrix is just full-matrix Muon
        ],
    )
    def test_group_size_controls_block_count(self, head_group_size, expected):
        blocks = infer_head_block_counts(
            _attention_model(), head_group_size=head_group_size, module_names=_QKV_MODULES
        )
        assert blocks == {k: v for k, v in expected.items() if v > 1}

    @pytest.mark.parametrize("head_group_size", [0, -1])
    def test_non_positive_group_size_disables_splitting(self, head_group_size):
        assert infer_head_block_counts(_attention_model(), head_group_size, module_names=_QKV_MODULES) == {}

    def test_listed_modules_are_trusted_even_when_column_stacked(self):
        """No name is vetoed: ``o_proj`` splits by row if you ask for it."""
        blocks = infer_head_block_counts(_attention_model(), head_group_size=1, module_names=("o_proj",))
        assert blocks == {"self_attn.o_proj.weight": 8}

    def test_modules_without_head_dim_are_skipped(self):
        model = _attention_model()
        del model.self_attn.head_dim
        assert infer_head_block_counts(model, head_group_size=1, module_names=_QKV_MODULES) == {}

    def test_modules_without_head_count_are_skipped(self):
        model = _attention_model()
        del model.self_attn.num_heads
        del model.self_attn.num_key_value_heads
        assert infer_head_block_counts(model, head_group_size=1, module_names=_QKV_MODULES) == {}

    def test_mla_row_stride_is_qk_head_dim_not_head_dim(self):
        """MLA's ``head_dim`` is the rope part only; blocks must follow ``qk_head_dim``."""
        num_heads, qk_nope, qk_rope = 8, 32, 16
        attn = nn.Module()
        attn.num_heads = num_heads
        attn.qk_head_dim = qk_nope + qk_rope
        attn.config = SimpleNamespace(num_attention_heads=num_heads, head_dim=qk_rope)
        attn.q_b_proj = nn.Linear(64, num_heads * attn.qk_head_dim, bias=False)
        model = nn.Module()
        model.self_attn = attn

        blocks = infer_head_block_counts(model, head_group_size=1, module_names=("q_b_proj",))
        assert blocks == {"self_attn.q_b_proj.weight": num_heads}

    def test_unresolvable_row_count_is_skipped(self):
        """A declared stride that does not tile the rows must not be guessed around."""
        attn = nn.Module()
        attn.num_heads = 8
        attn.head_dim = 16
        attn.q_proj = nn.Linear(64, 8 * 48, bias=False)  # stride is really 48, not head_dim
        model = nn.Module()
        model.self_attn = attn

        assert infer_head_block_counts(model, head_group_size=1, module_names=_QKV_MODULES) == {}

    def test_module_attrs_win_over_config_attrs(self):
        """A DSA indexer's own smaller head layout must beat the shared config's."""
        indexer = nn.Module()
        indexer.n_heads = 4
        indexer.head_dim = 32
        # 4 * 32 and 8 * 16 both equal 128, so config-first resolution would pick 8.
        indexer.config = SimpleNamespace(num_attention_heads=8, head_dim=16)
        indexer.wq_b = nn.Linear(64, 128, bias=False)
        model = nn.Module()
        model.indexer = indexer

        blocks = infer_head_block_counts(model, head_group_size=1, module_names=("wq_b",))
        assert blocks == {"indexer.wq_b.weight": 4}

    @pytest.mark.parametrize("head_group_size", [1, 2, 8, 16])
    def test_bare_name_selecting_two_nested_projections_is_rejected(self, head_group_size):
        """DeepSeek-V4's MLA and the DSA indexer *inside* it both name their
        up-projection ``q_b_proj``, and the indexer's own ``num_heads``/``head_dim``
        resolve its 8192 rows to 64 x 128 -- so a bare ``[q_b_proj]`` would split both,
        at different block counts, with nothing in the config saying so. Picking a
        winner would silently decide the update math, so the entry is refused and the
        two qualified forms are named.

        Parametrized over ``head_group_size`` because the pair has to stay visible when
        one side stops splitting: at 8 the MLA's 8 heads collapse to a single block and
        at 16 they are not divisible at all, and in both cases the indexer alone would
        otherwise be split -- the inversion of what the entry asks for.
        """
        with pytest.raises(ValueError) as excinfo:
            infer_head_block_counts(
                _dsv4_csa_attention_model(), head_group_size=head_group_size, module_names=("q_b_proj",)
            )

        message = str(excinfo.value)
        assert "self_attn.q_b_proj (8 heads)" in message
        assert "self_attn.compressor.indexer.q_b_proj (64 heads)" in message
        # The fix has to be copy-pasteable, not just described.
        assert "'self_attn.q_b_proj'" in message and "'indexer.q_b_proj'" in message

    def test_qualified_entries_address_each_projection(self):
        """A dotted suffix picks exactly one of the two, and listing both splits both."""
        mla_only = infer_head_block_counts(
            _dsv4_csa_attention_model(), head_group_size=1, module_names=("self_attn.q_b_proj",)
        )
        assert mla_only == {"self_attn.q_b_proj.weight": 8}

        indexer_only = infer_head_block_counts(
            _dsv4_csa_attention_model(), head_group_size=1, module_names=("indexer.q_b_proj",)
        )
        assert indexer_only == {"self_attn.compressor.indexer.q_b_proj.weight": 64}

        # Any depth of suffix works, not just one segment.
        deeper = infer_head_block_counts(
            _dsv4_csa_attention_model(), head_group_size=1, module_names=("compressor.indexer.q_b_proj",)
        )
        assert deeper == {"self_attn.compressor.indexer.q_b_proj.weight": 64}

        both = infer_head_block_counts(
            _dsv4_csa_attention_model(),
            head_group_size=1,
            module_names=("self_attn.q_b_proj", "indexer.q_b_proj"),
        )
        assert both == {"self_attn.q_b_proj.weight": 8, "self_attn.compressor.indexer.q_b_proj.weight": 64}

    @pytest.mark.parametrize("entries", [("q_b_proj", "self_attn.q_b_proj"), ("q_b_proj", "indexer.q_b_proj")])
    def test_supplementing_a_bare_entry_does_not_make_it_unambiguous(self, entries):
        """Adding one of the suggested entries instead of replacing the bare one still fails.

        The check is per pair of selected sites rather than per entry precisely for this:
        the likely reaction to the error is to *add* the qualified name, and a per-entry
        rule would then find one site under each entry, form no pair, and split both in
        silence. Both orders of qualification behave the same, so there is no spelling
        that is accidentally safe.
        """
        with pytest.raises(ValueError, match="is ambiguous"):
            infer_head_block_counts(_dsv4_csa_attention_model(), head_group_size=1, module_names=entries)

    def test_repeated_collisions_are_counted_once(self):
        """Per-layer repeats collapse into one error that says how many sites match."""
        model = nn.Module()
        model.layers = nn.ModuleList([_dsv4_csa_attention_model().self_attn for _ in range(3)])

        with pytest.raises(ValueError) as excinfo:
            infer_head_block_counts(model, head_group_size=1, module_names=("q_b_proj",))

        message = str(excinfo.value)
        assert "layers.0.q_b_proj (8 heads)" in message
        assert "repeats in 2 other module(s)" in message

    def test_sibling_matches_stay_selected_together(self):
        """Only *nested* matches are refused; two towers sharing a name are not.

        Mirrors Qwen2.5-Omni, where ``Qwen2_5OmniAttention`` and
        ``Qwen2_5OmniAudioAttention`` both name a ``q_proj``: neither encloses the
        other, so ``[q_proj]`` unambiguously means both and must keep working.
        """
        model = nn.Module()
        model.thinker = _FakeAttention()
        model.audio_tower = _FakeAttention(hidden=32, num_heads=4, num_kv_heads=4, head_dim=8)

        blocks = infer_head_block_counts(model, head_group_size=1, module_names=("q_proj",))
        assert blocks == {"thinker.q_proj.weight": 8, "audio_tower.q_proj.weight": 4}

        # And a qualified entry narrows to one tower, which a leaf-name matcher cannot do.
        one_tower = infer_head_block_counts(model, head_group_size=1, module_names=("audio_tower.q_proj",))
        assert one_tower == {"audio_tower.q_proj.weight": 4}

    def test_entries_match_whole_path_segments_only(self):
        """``q_b_proj`` must not select ``xq_b_proj``: suffixes break on dots."""
        attn = nn.Module()
        attn.num_heads = 8
        attn.head_dim = 16
        attn.xq_b_proj = nn.Linear(64, 8 * 16, bias=False)
        model = nn.Module()
        model.self_attn = attn

        assert infer_head_block_counts(model, head_group_size=1, module_names=("q_b_proj",)) == {}

    def test_glm_dsa_needs_no_qualification(self):
        """GLM MoE DSA's names never collide, so both bare entries stay usable.

        Its indexer calls its up-projection ``wq_b`` while the attention around it uses
        ``q_b_proj``. Built from the real classes, because a synthetic stand-in would
        pass no matter what the shipped modules are named.
        """
        model = _glm_dsa_attention_model()

        assert infer_head_block_counts(model, head_group_size=1, module_names=("wq_b",)) == {
            "self_attn.indexer.wq_b.weight": 4
        }
        assert infer_head_block_counts(model, head_group_size=1, module_names=("q_b_proj",)) == {
            "self_attn.q_b_proj.weight": 8
        }

    def test_head_split_modules_thread_from_the_optimizer_config(self):
        """Qualified entries have to survive the production build path.

        Head-split inference runs at optimizer construction from ``OptimizerConfig``
        via ``build_optimizer(optimizer_config=...)``, so that is what decides the
        update; a test against ``infer_head_block_counts`` alone would pass with the
        field unwired.
        """
        from veomni.arguments.arguments_types import OptimizerConfig
        from veomni.optim import build_optimizer

        model = _dsv4_csa_attention_model()

        def _q_b_proj_blocks(*entries: str) -> dict:
            cfg = OptimizerConfig(
                type="muon",
                lr=1e-4,
                muon_head_group_size=1,
                muon_head_split_modules=list(entries),
                muon_ns_implementation="std",
            )
            opt = build_optimizer(model, lr=cfg.lr, optimizer_type=cfg.type, optimizer_config=cfg)
            names = {id(p): n for n, p in model.named_parameters()}
            return {
                names[id(p)]: int(group.get("head_blocks", 1))
                for group in opt.optimizers_dict["muon"].param_groups
                for p in group["params"]
                if names[id(p)].endswith("q_b_proj.weight")
            }

        assert _q_b_proj_blocks("self_attn.q_b_proj") == {
            "self_attn.q_b_proj.weight": 8,
            "self_attn.compressor.indexer.q_b_proj.weight": 1,
        }
        assert _q_b_proj_blocks("self_attn.q_b_proj", "indexer.q_b_proj") == {
            "self_attn.q_b_proj.weight": 8,
            "self_attn.compressor.indexer.q_b_proj.weight": 64,
        }
        with pytest.raises(ValueError, match="is ambiguous"):
            _q_b_proj_blocks("q_b_proj")


class TestHeadSplitUpdate:
    """Row-block orthogonalization math and its learning-rate scaling."""

    def _per_block_reference(self, grad: torch.Tensor, head_blocks: int) -> torch.Tensor:
        rows = grad.shape[0] // head_blocks
        return torch.cat(
            [run_newton_schulz(grad[i * rows : (i + 1) * rows], ns_implementation="std") for i in range(head_blocks)],
            dim=0,
        )

    def test_update_matches_per_block_newton_schulz(self):
        head_blocks, rows, cols = 8, 16, 32
        grad = _sample_2d(head_blocks * rows, cols, seed=11)
        p = nn.Parameter(torch.zeros_like(grad))
        p.grad = grad.clone()

        _head_split_muon(p, head_blocks).step()

        # rows <= cols keeps _adjust_lr at 1.0, so the update is the raw polar factor.
        torch.testing.assert_close(-p.detach(), self._per_block_reference(grad, head_blocks))

    @pytest.mark.parametrize("adjust_lr_fn", ["original", "match_rms_adamw"])
    def test_lr_adjust_follows_the_block_shape(self, adjust_lr_fn):
        """A tall stack must not inherit the full matrix's larger LR factor."""
        head_blocks, block_rows, cols = 8, 16, 32
        rows = head_blocks * block_rows
        grad = _sample_2d(rows, cols, seed=5)
        p = nn.Parameter(torch.zeros_like(grad))
        p.grad = grad.clone()

        _head_split_muon(p, head_blocks, adjust_lr_fn=adjust_lr_fn).step()

        block_factor = upstream_adjust_lr(1.0, adjust_lr_fn, (block_rows, cols))
        full_factor = upstream_adjust_lr(1.0, adjust_lr_fn, (rows, cols))
        assert block_factor < full_factor
        torch.testing.assert_close(-p.detach(), block_factor * self._per_block_reference(grad, head_blocks))

    def test_split_and_full_updates_differ(self):
        """Guard against the split silently degenerating into full-matrix Muon."""
        head_blocks, block_rows, cols = 4, 8, 16
        grad = _sample_2d(head_blocks * block_rows, cols, seed=3)
        split, full = nn.Parameter(torch.zeros_like(grad)), nn.Parameter(torch.zeros_like(grad))
        split.grad, full.grad = grad.clone(), grad.clone()

        _head_split_muon(split, head_blocks).step()
        _head_split_muon(full, 1).step()

        assert not torch.allclose(-split.detach(), -full.detach(), atol=1e-3)


class TestHeadSplitGuards:
    def test_head_blocks_rejects_3d_params(self):
        p = nn.Parameter(torch.randn(2, 4, 8))
        with pytest.raises(ValueError, match="only applies to 2D"):
            DistributedMuon([{"params": [p], "head_blocks": 2}], lr=1e-3)

    def test_head_blocks_must_divide_rows(self):
        p = nn.Parameter(torch.randn(6, 8))
        with pytest.raises(ValueError, match="does not evenly divide"):
            DistributedMuon([{"params": [p], "head_blocks": 4}], lr=1e-3)

    def test_head_blocks_must_be_positive(self):
        p = nn.Parameter(torch.randn(4, 8))
        with pytest.raises(ValueError, match="head_blocks must be >= 1"):
            DistributedMuon([p], lr=1e-3, head_blocks=0)

    def test_add_param_group_is_validated_too(self):
        opt = DistributedMuon([nn.Parameter(torch.randn(4, 8))], lr=1e-3)
        with pytest.raises(ValueError, match="does not evenly divide"):
            opt.add_param_group({"params": [nn.Parameter(torch.randn(6, 8))], "head_blocks": 4})

    def test_configured_head_blocks_survives_resume(self):
        """A checkpoint from a full-matrix run must not silently undo the split."""
        p = nn.Parameter(torch.randn(16, 8))
        opt = _head_split_muon(p, head_blocks=4)
        p.grad = torch.randn_like(p)
        opt.step()

        stale = opt.state_dict()
        stale["param_groups"][0]["head_blocks"] = 1
        opt.load_state_dict(stale)

        assert opt.param_groups[0]["head_blocks"] == 4


class TestHeadSplitBuild:
    def _muon_groups_by_blocks(self, model, **muon_kwargs):
        from veomni.optim import build_optimizer

        muon_kwargs.setdefault("lr", 1e-2)
        muon_kwargs.setdefault("ns_implementation", "std")
        if muon_kwargs.get("head_group_size"):
            muon_kwargs.setdefault("head_split_modules", _QKV_MODULES)
        opt = build_optimizer(model, lr=1e-4, optimizer_type="muon", muon_kwargs=muon_kwargs)
        names = {id(p): n for n, p in model.named_parameters()}
        return {
            int(group.get("head_blocks", 1)): sorted(names[id(p)] for p in group["params"])
            for group in opt.optimizers_dict["muon"].param_groups
        }

    def test_params_are_grouped_by_block_count(self):
        model = _attention_model()
        by_blocks = self._muon_groups_by_blocks(model, head_group_size=1)

        assert by_blocks[8] == ["self_attn.q_proj.weight"]
        assert by_blocks[2] == ["self_attn.k_proj.weight", "self_attn.v_proj.weight"]
        # o_proj has the same shape as q_proj but is not head-stacked by row.
        assert by_blocks[1] == ["self_attn.o_proj.weight", "self_attn.q_a_proj.weight"]

    def test_default_config_keeps_one_full_matrix_group(self):
        by_blocks = self._muon_groups_by_blocks(_attention_model())
        assert list(by_blocks) == [1]

    def test_negative_group_size_raises(self):
        with pytest.raises(ValueError, match="muon_head_group_size must be >= 0"):
            self._muon_groups_by_blocks(_attention_model(), head_group_size=-2)

    def test_group_size_without_module_list_raises(self):
        with pytest.raises(ValueError, match="requires muon_head_split_modules"):
            self._muon_groups_by_blocks(_attention_model(), head_group_size=1, head_split_modules=[])

    @pytest.mark.parametrize(
        "head_group_size, expected",
        [
            (0, []),
            (1, ["self_attn.k_proj.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight"]),
        ],
    )
    def test_head_blocks_only_reaches_the_state_dict_when_splitting(self, head_group_size, expected):
        """With the option off, the flattened DCP schema must match a pre-feature run."""
        from veomni.optim import build_optimizer

        model = _attention_model()
        opt = build_optimizer(
            model,
            lr=1e-4,
            optimizer_type="muon",
            muon_kwargs={
                "lr": 1e-2,
                "ns_implementation": "std",
                "head_group_size": head_group_size,
                "head_split_modules": _QKV_MODULES,
            },
        )
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        opt.step()

        tagged = sorted(k[len("param_groups.") : -len(".head_blocks")] for k in opt.state_dict() if "head_blocks" in k)
        assert tagged == expected

    def test_split_run_survives_a_state_dict_roundtrip(self):
        """Multi-group Muon must save and reload through the MultiOptimizer path."""
        from veomni.optim import build_optimizer

        def _build(model):
            return build_optimizer(
                model,
                lr=1e-4,
                optimizer_type="muon",
                muon_kwargs={
                    "lr": 1e-2,
                    "ns_implementation": "std",
                    "head_group_size": 2,
                    "head_split_modules": _QKV_MODULES,
                },
            )

        model_a = _attention_model()
        opt_a = _build(model_a)
        for p in model_a.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        opt_a.step()
        saved = opt_a.state_dict()

        model_b = _attention_model()
        model_b.load_state_dict(model_a.state_dict())
        opt_b = _build(model_b)
        opt_b.load_state_dict(saved)

        assert [g.get("head_blocks", 1) for g in opt_b.optimizers_dict["muon"].param_groups] == [1, 4]
        reloaded = opt_b.state_dict()
        assert set(saved) == set(reloaded)
        for key, before in saved.items():
            if isinstance(before, torch.Tensor):
                torch.testing.assert_close(before, reloaded[key], atol=0.0, rtol=0.0, msg=f"mismatch at {key}")


class TestGramQuackFallback:
    def test_gram_quack_falls_back_without_package(self, monkeypatch):
        import veomni.optim.muon as muon_mod

        monkeypatch.setattr(muon_mod, "_GRAM_QUACK_FALLBACK_WARNED", False)
        monkeypatch.setattr(muon_mod, "_gram_quack_unavailable_reason", lambda _grad: None)
        call_count = 0

        def _boom(*_a, **_k):
            nonlocal call_count
            call_count += 1
            raise ImportError("quack missing for test")

        monkeypatch.setattr(muon_mod, "_package_gram_newton_schulz", _boom)
        x = _sample_2d(8, 16, dtype=torch.float32, seed=0)
        out = run_newton_schulz(
            x,
            ns_implementation="gram_quack",
            gram_ns_reset_iterations=(2,),
            compute_dtype=torch.float32,
        )
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert muon_mod._GRAM_QUACK_FALLBACK_WARNED is True
        assert call_count == 1

    def test_gram_quack_falls_back_on_pre_hopper_cuda(self, monkeypatch):
        import veomni.optim.muon as muon_mod

        device_type = get_device_type()
        monkeypatch.setattr(muon_mod, "IS_CUDA_AVAILABLE", True)
        monkeypatch.setattr(muon_mod, "get_device_type", lambda: device_type)
        queried_devices = []
        monkeypatch.setattr(
            muon_mod, "get_gpu_compute_capability", lambda device: queried_devices.append(device) or 80
        )
        grad = type("FakeTensor", (), {"device": torch.device(f"{device_type}:1")})()
        assert "compute capability 8.0" in muon_mod._gram_quack_unavailable_reason(grad)
        assert queried_devices == [grad.device]

    def test_gram_quack_falls_back_without_cuda(self, monkeypatch):
        import veomni.optim.muon as muon_mod

        monkeypatch.setattr(muon_mod, "IS_CUDA_AVAILABLE", False)
        monkeypatch.setattr(muon_mod, "_GRAM_QUACK_FALLBACK_WARNED", False)

        def _unexpected(*_a, **_k):
            raise AssertionError("package backend should not be called for non-CUDA tensors")

        monkeypatch.setattr(muon_mod, "_package_gram_newton_schulz", _unexpected)
        x = _sample_2d(8, 16, dtype=torch.float32, seed=0)
        out = run_newton_schulz(
            x,
            ns_implementation="gram_quack",
            gram_ns_reset_iterations=(2,),
            compute_dtype=torch.float32,
        )
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert muon_mod._GRAM_QUACK_FALLBACK_WARNED is True

    def test_gram_quack_propagates_unknown_not_implemented(self, monkeypatch):
        import veomni.optim.muon as muon_mod

        monkeypatch.setattr(muon_mod, "_gram_quack_unavailable_reason", lambda _grad: None)

        def _boom(*_a, **_k):
            raise NotImplementedError("unexpected kernel limitation")

        monkeypatch.setattr(muon_mod, "_package_gram_newton_schulz", _boom)
        x = _sample_2d(8, 16, dtype=torch.float32, seed=0)
        with pytest.raises(NotImplementedError, match="unexpected kernel limitation"):
            run_newton_schulz(
                x,
                ns_implementation="gram_quack",
                gram_ns_reset_iterations=(2,),
                compute_dtype=torch.float32,
            )

    def test_gram_quack_uses_package_on_hopper(self, monkeypatch):
        import veomni.optim.muon as muon_mod

        device_type = get_device_type()
        monkeypatch.setattr(muon_mod, "IS_CUDA_AVAILABLE", True)
        monkeypatch.setattr(muon_mod, "get_device_type", lambda: device_type)
        monkeypatch.setattr(muon_mod, "get_gpu_compute_capability", lambda _device: 90)
        grad = type("FakeTensor", (), {"device": torch.device(device_type)})()
        assert muon_mod._gram_quack_unavailable_reason(grad) is None


def test_moe_expert_ns_chunk_sizes_schedule():
    """The slicer must produce a valid, balanced, minimal-call schedule for any budget.

    Invariants: every expert covered once, no call over budget, spread within one
    expert (so no single-expert ``bmm``), and the fewest calls the budget allows.
    ``-1`` (the default) means "off" and must reproduce the one-call shape.
    """
    from veomni.optim.muon import _DEFAULT_MOE_LOCAL_3D_NS_MAX_EXPERTS, _expert_ns_chunk_sizes

    # Chunking is off by default, whatever the expert count.
    assert _DEFAULT_MOE_LOCAL_3D_NS_MAX_EXPERTS == -1
    for experts in (1, 4, 6, 7, 32, 64):
        assert _expert_ns_chunk_sizes(experts) == [experts]

    # Enabled at 3: every remainder class across a wide range.
    for experts in range(1, 65):
        sizes = _expert_ns_chunk_sizes(experts, 3)
        assert sum(sizes) == experts
        assert max(sizes) <= 3
        # Spread, not adjacent pairs: ``[3, 3, 1]`` has adjacent diffs 0 and -2 but
        # still ends in the single-expert tail this scheme exists to avoid.
        assert max(sizes) - min(sizes) <= 1, (experts, sizes)
        # Fewest calls the budget allows: dropping one call must overflow it.
        if len(sizes) > 1:
            assert -(-experts // (len(sizes) - 1)) > 3, (experts, sizes)

    # A budget at or above the stack size is effectively off.
    assert _expert_ns_chunk_sizes(7, 0) == [7]
    assert _expert_ns_chunk_sizes(7, 7) == [7]
    assert _expert_ns_chunk_sizes(7, 64) == [7]

    # A larger budget trades workspace for fewer calls, and stays balanced.
    assert _expert_ns_chunk_sizes(32, 8) == [8, 8, 8, 8]
    assert _expert_ns_chunk_sizes(32, 1) == [1] * 32
    for budget in (1, 2, 3, 5, 8):
        for experts in (7, 10, 13, 31):
            sizes = _expert_ns_chunk_sizes(experts, budget)
            assert sum(sizes) == experts
            assert max(sizes) - min(sizes) <= 1, (budget, experts, sizes)


def test_moe_expert_ns_chunking_handles_an_empty_local_shard():
    """A non-owner rank's FSDP shard of a 3D param is empty and must not divide by zero.

    Wan's ``[1, 6, dim]`` modulation parameters leave non-owner ranks holding
    ``[0, 6, dim]``; those are Muon-eligible and reach this path. The DTensor is
    real (single-process gloo mesh), so classification and dispatch both run.
    """
    import tempfile

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    from veomni.optim.muon import _chunked_newton_schulz, _classify_param, _expert_ns_chunk_sizes

    assert _expert_ns_chunk_sizes(0) == []
    assert _expert_ns_chunk_sizes(0, 0) == []

    empty = torch.empty(0, 6, 4)
    out = _chunked_newton_schulz(empty, lambda x: torch.zeros_like(x))
    assert out.shape == empty.shape
    assert out.numel() == 0

    created = False
    if not (dist.is_available() and dist.is_initialized()):
        rendezvous = tempfile.mkdtemp()
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous}/rendezvous",
            rank=0,
            world_size=1,
        )
        created = True
    try:
        mesh = init_device_mesh("cpu", (1,))
        sharded = DTensor.from_local(empty, device_mesh=mesh, placements=(Shard(0),), run_check=False)
        assert _classify_param(sharded) == "moe_local_3d"
        chunked = _chunked_newton_schulz(sharded._local_tensor, lambda x: torch.zeros_like(x))
        assert chunked.shape == sharded._local_tensor.shape
    finally:
        if created:
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "experts,rows,cols",
    [
        # Wide (rows <= cols): Gram-NS iterates on ``X @ X.T``.
        pytest.param(7, 4, 6, id="wide-remainder"),
        # Tall (rows > cols): Gram-NS iterates on ``X.T @ X``, the other branch.
        pytest.param(4, 8, 3, id="tall"),
    ],
)
def test_moe_expert_ns_chunking_matches_a_whole_stack_call(experts, rows, cols):
    """Chunking must match one whole-stack call numerically, and must actually chunk.

    Both Gram-NS branches are covered: slicing dim 0 is branch-agnostic but the
    trailing-matrix math is not. The recorded shapes also fail a silent regression
    to a single whole-stack call, which equivalence alone would not catch.
    """
    from veomni.optim.muon import _chunked_newton_schulz, _expert_ns_chunk_sizes

    torch.manual_seed(0)
    local = torch.randn(experts, rows, cols)

    def _ns(x):
        return run_newton_schulz(
            x,
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=DEFAULT_NS_STEPS,
            eps=1e-7,
            ns_implementation="gram",
            gram_ns_reset_iterations=(2,),
        )

    shapes = []

    def _recording_ns(x):
        shapes.append(tuple(x.shape))
        return _ns(x)

    chunked = _chunked_newton_schulz(local, _recording_ns, max_experts=3)

    torch.testing.assert_close(chunked, _ns(local))
    assert shapes == [(size, rows, cols) for size in _expert_ns_chunk_sizes(experts, 3)]


def test_moe_expert_ns_chunking_preserves_dtype_and_short_circuits():
    """The preallocated output must keep the update's dtype and skip work when unchunked.

    Widening the output would silently promote the update and break
    ``_apply_ortho``; an unchunked call must not pay a useless ``empty_like`` copy.
    """
    from veomni.optim.muon import _chunked_newton_schulz

    for dtype in (torch.bfloat16, torch.float32):
        local = torch.randn(5, 4, 6, dtype=dtype)
        out = _chunked_newton_schulz(local, torch.zeros_like, max_experts=3)
        assert out.dtype == dtype
        assert out.shape == local.shape

    single_call = []

    def _recording(x):
        single_call.append(tuple(x.shape))
        return x

    # Chunking off (default): one call even for a stack far above the budget.
    big = torch.randn(32, 4, 4)
    _chunked_newton_schulz(big, _recording)
    assert single_call == [(32, 4, 4)]

    # Chunking on but the stack fits one slice: still one call.
    single_call.clear()
    small = torch.randn(3, 4, 4)
    _chunked_newton_schulz(small, _recording, max_experts=3)
    assert single_call == [(3, 4, 4)]


@pytest.mark.skipif(
    not ((IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) and get_torch_device().is_available()),
    reason="Visible CUDA or NPU device required",
)
def test_moe_expert_ns_chunking_matches_whole_stack_on_accelerator(monkeypatch):
    """The chunked NS path must agree with a whole-stack call on a real accelerator.

    The CPU tests prove the slicing arithmetic; this proves the device kernels
    tolerate it, and that ``gram_quack`` did not silently fall back on Hopper.
    """
    import importlib.util

    import veomni.optim.muon as muon_mod
    from veomni.optim.muon import _chunked_newton_schulz

    torch.manual_seed(0)
    experts, rows, cols = 7, 128, 256
    local = torch.randn(experts, rows, cols, device=get_device_type(), dtype=torch.bfloat16)

    backends = ["gram"]
    if IS_CUDA_AVAILABLE and importlib.util.find_spec("gram_newton_schulz") is not None:
        backends.append("gram_quack")

    for backend in backends:
        if backend == "gram_quack":
            monkeypatch.setattr(muon_mod, "_GRAM_QUACK_FALLBACK_WARNED", False)

        shapes = []

        def _ns(x, _backend=backend, _shapes=shapes):
            _shapes.append(tuple(x.shape))
            return run_newton_schulz(
                x,
                ns_coefficients=DEFAULT_NS_COEFFICIENTS,
                ns_steps=DEFAULT_NS_STEPS,
                eps=1e-7,
                ns_implementation=_backend,
                gram_ns_reset_iterations=(2,),
            )

        chunked = _chunked_newton_schulz(local, _ns, max_experts=3)
        # The CPU test pins the size policy; here it only has to happen on-device.
        assert shapes == [(3, rows, cols), (2, rows, cols), (2, rows, cols)]

        whole = run_newton_schulz(
            local,
            ns_coefficients=DEFAULT_NS_COEFFICIENTS,
            ns_steps=DEFAULT_NS_STEPS,
            eps=1e-7,
            ns_implementation=backend,
            gram_ns_reset_iterations=(2,),
        )
        assert chunked.dtype == local.dtype
        assert torch.isfinite(chunked).all()
        torch.testing.assert_close(chunked.float(), whole.float(), atol=5e-3, rtol=5e-3)
        if backend == "gram_quack":
            assert muon_mod._GRAM_QUACK_FALLBACK_WARNED is (get_gpu_compute_capability(local.device) < 90)
