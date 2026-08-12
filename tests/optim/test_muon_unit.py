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

from veomni.optim import muon as muon_impl
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


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


_DSV4_TOY_CONFIG_DIR = Path(__file__).resolve().parents[1] / "toy_config" / "deepseek_v4_toy"


def _dsv4_csa_attention_model() -> nn.Module:
    """A real ``DeepseekV4Attention`` on a CSA layer, so its indexer is the real class.

    The head-split opt-out is keyed on the indexer's class, so a synthetic stand-in
    would pass with a class name that matches nothing in the model.

    ``index_n_heads`` / ``index_head_dim`` are raised to the 4-layer checkpoint's
    64 x 128 (the toy config ships 8 x 128) so the indexer's head count differs from
    the main attention's 8 and one assertion can tell the two apart.
    """
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    config = AutoConfig.from_pretrained(str(_DSV4_TOY_CONFIG_DIR))
    config.index_n_heads = 64
    config.index_head_dim = 128
    torch.manual_seed(0)
    model = nn.Module()
    model.self_attn = modeling.DeepseekV4Attention(config, config.layer_types.index("compressed_sparse_attention"))
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

    def test_indexer_q_b_proj_excluded_from_head_split_by_default(self):
        """The DSA indexer and the attention it sits in both name their up-projection
        ``q_b_proj``, and the indexer's own ``num_heads``/``head_dim`` resolve its 8192
        rows to 64 x 128 -- so head-split Muon switches itself on for the indexer the
        moment the indexer loss populates a gradient, with no configuration change.
        Excluding by default keeps that from being a side effect of a flag aimed at MLA;
        the papers describe plain Muon over the whole matrix.

        Both halves in one assertion: an exclusion broad enough to also veto the
        attention's own ``q_b_proj`` has to fail here.
        """
        blocks = infer_head_block_counts(_dsv4_csa_attention_model(), head_group_size=1, module_names=("q_b_proj",))
        assert blocks == {"self_attn.q_b_proj.weight": 8}

    def test_indexer_head_split_can_be_opted_in(self):
        blocks = infer_head_block_counts(
            _dsv4_csa_attention_model(),
            head_group_size=1,
            module_names=("q_b_proj",),
            exclude_indexer=False,
        )
        assert blocks == {
            "self_attn.q_b_proj.weight": 8,
            "self_attn.compressor.indexer.q_b_proj.weight": 64,
        }

    def test_exclude_indexer_threads_from_the_optimizer_arguments(self):
        """The knob has to arrive from ``OptimizerConfig``, not only as a direct kwarg.

        Head-split inference runs at optimizer construction, so the whole path --
        argument field, ``_collect_muon_kwargs``, ``_build_muon_with_adamw`` -- is what
        decides the update; a test against ``infer_head_block_counts`` alone would pass
        with the field unwired.
        """
        from veomni.arguments.arguments_types import OptimizerConfig
        from veomni.optim import build_optimizer
        from veomni.optim.optimizer import _collect_muon_kwargs

        model = _dsv4_csa_attention_model()
        cfg = OptimizerConfig(
            type="muon",
            lr=1e-4,
            muon_head_group_size=1,
            muon_head_split_modules=["q_b_proj"],
            muon_ns_implementation="std",
        )
        assert cfg.muon_head_split_exclude_indexer is True  # the default under test

        def _q_b_proj_blocks() -> dict:
            opt = build_optimizer(model, lr=cfg.lr, optimizer_type="muon", muon_kwargs=_collect_muon_kwargs(cfg))
            names = {id(p): n for n, p in model.named_parameters()}
            return {
                names[id(p)]: int(group.get("head_blocks", 1))
                for group in opt.optimizers_dict["muon"].param_groups
                for p in group["params"]
                if names[id(p)].endswith("q_b_proj.weight")
            }

        assert _q_b_proj_blocks() == {
            "self_attn.q_b_proj.weight": 8,
            "self_attn.compressor.indexer.q_b_proj.weight": 1,
        }

        cfg.muon_head_split_exclude_indexer = False
        assert _q_b_proj_blocks() == {
            "self_attn.q_b_proj.weight": 8,
            "self_attn.compressor.indexer.q_b_proj.weight": 64,
        }


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
