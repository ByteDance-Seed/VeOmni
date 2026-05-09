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

import pytest
import torch
import torch.nn as nn


muon_module = pytest.importorskip("torch.optim._muon")
from torch.optim import Muon as UpstreamMuon  # noqa: E402
from torch.optim._muon import _zeropower_via_newtonschulz as upstream_ns  # noqa: E402

from veomni.optim.muon import (  # noqa: E402
    DEFAULT_NS_COEFFICIENTS,
    DEFAULT_NS_STEPS,
    DistributedMuon,
    batched_newton_schulz,
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
            {"ns_coefficients": (1.0, 2.0), "match": "exactly 3"},
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
