# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing limitations
# under the License.

"""Shared HF toy-model comparison helpers for models_kernel tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.kernels.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL


def eager_kernels_config() -> SimpleNamespace:
    return SimpleNamespace(
        attn_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
        rotary_pos_emb_vision_implementation="eager",
        swiglu_mlp_implementation="eager",
        load_balancing_loss_implementation="eager",
        moe_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
        dsa_indexer_implementation="eager",
        dsa_attention_implementation="eager",
        mhc_implementation="eager",
    )


def pin_eager_attn_implementation(model: torch.nn.Module) -> None:
    """Force every config on ``model`` onto HF eager attention.

    Composite VL/omni configs drop ``attn_implementation`` when nested
    configs go through ``to_dict()``, so HuggingFace defaults to ``sdpa``.
    models_kernel consume reads kernels ``attn_implementation`` (eager in
    these tests). Pin HF to the same impl before comparing.
    """
    configs: list[object] = []
    top = getattr(model, "config", None)
    if top is not None:
        configs.append(top)
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None:
            configs.append(cfg)
    seen: set[int] = set()
    stack = list(configs)
    while stack:
        cfg = stack.pop()
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        if hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
        for name in ("text_config", "vision_config", "audio_config", "thinker_config"):
            stack.append(getattr(cfg, name, None))


def named_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def assert_no_ops_or_old_models_import(*modules, require_loss_utils: bool = True) -> None:
    for module in modules:
        source = module.__file__
        assert source is not None
        text = open(source, encoding="utf-8").read()
        assert "use_non_eager_impl" not in text
        assert "OpSlot" not in text
        assert "veomni.ops" not in text
        assert "from veomni.models." not in text
        assert "from veomni.models import" not in text
        if require_loss_utils:
            assert "from veomni.models_kernel.utils.loss_utils import" in text


def assert_eager_matches_hf(
    hf: torch.nn.Module,
    ours: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    fwd_kwargs: dict | None = None,
    ours_fwd_kwargs: dict | None = None,
    atol: float = EAGER_ATOL,
    rtol: float = EAGER_RTOL,
    grad_atol: float = EAGER_GRAD_ATOL,
    grad_rtol: float = EAGER_GRAD_RTOL,
) -> None:
    """Compare unlabeled logits, labeled loss, and grads against HF."""
    pin_eager_attn_implementation(hf)
    pin_eager_attn_implementation(ours)

    hf_kwargs = {} if fwd_kwargs is None else dict(fwd_kwargs)
    ours_kwargs = dict(hf_kwargs)
    if ours_fwd_kwargs is not None:
        ours_kwargs.update(ours_fwd_kwargs)

    hf_logits = hf(input_ids=input_ids, use_cache=False, **hf_kwargs).logits
    ours_logits = ours(input_ids=input_ids, use_cache=False, **ours_kwargs).logits
    torch.testing.assert_close(ours_logits, hf_logits, atol=atol, rtol=rtol)

    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, **hf_kwargs)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False, **ours_kwargs)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=atol, rtol=rtol)
    assert ours_out.logits is None

    hf_out.loss.backward()
    ours_out.loss.backward()
    hf_grads = named_trainable(hf)
    ours_grads = named_trainable(ours)
    assert hf_grads.keys() == ours_grads.keys()
    for name, param in hf_grads.items():
        if param.grad is None:
            assert ours_grads[name].grad is None, name
            continue
        assert ours_grads[name].grad is not None, name
        torch.testing.assert_close(ours_grads[name].grad, param.grad, atol=grad_atol, rtol=grad_rtol, msg=name)


def _as_tensors(value: object) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [item for item in value if torch.is_tensor(item)]
    raise TypeError(f"unsupported output type: {type(value)!r}")


def assert_outputs_and_grads_match(
    official: torch.nn.Module,
    ours: torch.nn.Module,
    call,
    *,
    atol: float = EAGER_ATOL,
    rtol: float = EAGER_RTOL,
    grad_atol: float = EAGER_GRAD_ATOL,
    grad_rtol: float = EAGER_GRAD_RTOL,
) -> None:
    """Compare a test-local official snapshot against models_kernel eager."""
    official.train()
    ours.train()

    official_out = call(official)
    ours_out = call(ours)
    official_tensors = _as_tensors(official_out)
    ours_tensors = _as_tensors(ours_out)
    assert len(official_tensors) == len(ours_tensors)
    for left, right in zip(ours_tensors, official_tensors, strict=True):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)

    official.zero_grad(set_to_none=True)
    ours.zero_grad(set_to_none=True)
    official_loss = sum(tensor.float().sum() for tensor in official_tensors)
    ours_loss = sum(tensor.float().sum() for tensor in ours_tensors)
    official_loss.backward()
    ours_loss.backward()

    official_grads = named_trainable(official)
    ours_grads = named_trainable(ours)
    assert official_grads.keys() == ours_grads.keys()
    for name, param in official_grads.items():
        if param.grad is None:
            assert ours_grads[name].grad is None, name
            continue
        assert ours_grads[name].grad is not None, name
        torch.testing.assert_close(ours_grads[name].grad, param.grad, atol=grad_atol, rtol=grad_rtol, msg=name)
