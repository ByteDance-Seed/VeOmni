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

"""Shared HF toy-model comparison helpers for models tests."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from tests.ops.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.ops.config import get_ops_config, set_ops_config


@contextmanager
def ops_config_scope(config):
    """Temporarily install a config, restoring the previous object even on failure.

    Scopes nest; None is a valid installed state. This restores the binding,
    not in-place changes to attributes of a caller-owned configuration object.
    """
    previous = get_ops_config()
    set_ops_config(config)
    try:
        yield
    finally:
        set_ops_config(previous)


def eager_ops_config() -> SimpleNamespace:
    """Return an all-eager ops selection for model parity tests."""
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
    configs go through ``to_dict()``, so Hugging Face defaults to ``sdpa``.
    VeOmni model implementations read the ops config's ``attn_implementation`` (eager in
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


def qwen_image_inputs(config, input_ids: torch.Tensor) -> dict:
    """Pack three unequal images into two prompts, with text between image spans.

    The 2x3 merged grid exercises non-square spatial RoPE and non-identity
    window ordering; the other grids exercise image and batch boundaries.
    Callers supply two rows with at least 16 tokens. Only response text is
    supervised, and the shorter second sample is right-padded.
    """
    vision = config.vision_config
    merge = vision.spatial_merge_size
    grids = torch.tensor([[1, 2 * merge, 3 * merge], [1, merge, 2 * merge], [1, 3 * merge, merge]])
    ids = input_ids.clone()
    ids[0, 1:7] = config.image_token_id
    ids[0, 9:11] = config.image_token_id
    ids[1, 2:5] = config.image_token_id
    if hasattr(config, "audio_config"):
        # Omni locates visual spans by delimiters rather than mm_token_type_ids.
        ids[0, [0, 8]] = config.vision_start_token_id
        ids[0, [7, 11]] = config.vision_end_token_id
        ids[1, 1] = config.vision_start_token_id
        ids[1, 5] = config.vision_end_token_id
    attention_mask = torch.ones_like(ids)
    attention_mask[1, -3:] = 0
    ids[1, -3:] = config.text_config.pad_token_id
    image_mask = ids == config.image_token_id
    labels = ids.clone()
    labels[0, :12] = -100
    labels[1, :6] = -100
    labels[attention_mask == 0] = -100
    feat_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    result = {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": torch.randn(int(grids.prod(dim=-1).sum()), feat_dim),
        "image_grid_thw": grids,
        "mm_token_type_ids": image_mask.int(),
        "image_mask": image_mask,
        "video_mask": torch.zeros_like(image_mask),
    }
    if hasattr(config, "audio_config"):
        result["audio_mask"] = torch.zeros_like(image_mask)
    return result


def qwen_video_inputs(config, input_ids: torch.Tensor, *, split_frames: bool) -> dict:
    """Build two unequal multi-frame clips with either contiguous or separated frames.

    Qwen2.5-VL consumes contiguous video tokens; Qwen3-VL separates each frame
    with text (timestamps in processor output). Use at least 20 input tokens.
    """
    vision = config.vision_config
    merge = vision.spatial_merge_size
    grids = torch.tensor([[2, 2 * merge, 3 * merge], [3, merge, 2 * merge]])
    ids = input_ids.clone()
    if split_frames:
        ids[0, 1:7] = config.video_token_id
        ids[0, 9:15] = config.video_token_id
        for start in (2, 6, 10):
            ids[1, start : start + 2] = config.video_token_id
        response_starts = (16, 13)
    else:
        ids[0, 1:13] = config.video_token_id
        ids[1, 2:8] = config.video_token_id
        response_starts = (14, 9)
    attention_mask = torch.ones_like(ids)
    attention_mask[1, -3:] = 0
    ids[1, -3:] = config.text_config.pad_token_id
    video_mask = ids == config.video_token_id
    labels = ids.clone()
    for row, start in enumerate(response_starts):
        labels[row, :start] = -100
    labels[attention_mask == 0] = -100
    feat_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values_videos": torch.randn(int(grids.prod(dim=-1).sum()), feat_dim),
        "video_grid_thw": grids,
        "mm_token_type_ids": video_mask.int() * 2,
        "image_mask": torch.zeros_like(video_mask),
        "video_mask": video_mask,
    }


def assert_tensors_match(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    equal: bool,
    atol: float,
    rtol: float,
    what: str,
) -> None:
    """Compare tensors bitwise when the family is bit-identical, else ``atol``/``rtol``.

    Eager FP32 logits can match Hugging Face exactly when VeOmni runs the same
    arithmetic. Loss and gradients stay on ``1e-6`` / ``1e-5`` even then, because
    the fused CE path and mixed-dtype reductions are not bit-identical.
    """
    if equal:
        if torch.equal(actual, reference):
            return
        diff = (actual.float() - reference.float()).abs()
        mismatched = actual != reference
        raise AssertionError(
            f"{what} are not bitwise equal: "
            f"{int(mismatched.sum().item())}/{actual.numel()} mismatched, "
            f"max_abs_diff={float(diff.max().item()):.3e}"
        )
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol, msg=what)


def assert_eager_matches_hf(
    hf: torch.nn.Module,
    ours: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    fwd_kwargs: dict | None = None,
    ours_fwd_kwargs: dict | None = None,
    atol: float = EAGER_ATOL,
    rtol: float = EAGER_RTOL,
    grad_atol: float = EAGER_GRAD_ATOL,
    grad_rtol: float = EAGER_GRAD_RTOL,
    logits_equal: bool = False,
) -> None:
    """Compare logits, loss and parameter gradients, optionally with masked labels.

    ``logits_equal=True`` is the eager FP32 contract for dense text families
    whose arithmetic matches Hugging Face. Families that keep ``False`` (the
    default) stay on ``1e-6`` because MoE expert loops, DSA/MLA, or vision
    scatter accumulate about 1e-7 ULP; that is still inside the old close
    bound and is recorded at each call site. Loss and gradients always use
    ``atol`` / ``grad_atol``.
    """
    pin_eager_attn_implementation(hf)
    pin_eager_attn_implementation(ours)

    hf_kwargs = {} if fwd_kwargs is None else dict(fwd_kwargs)
    ours_kwargs = dict(hf_kwargs)
    if ours_fwd_kwargs is not None:
        ours_kwargs.update(ours_fwd_kwargs)

    hf_logits = hf(input_ids=input_ids, use_cache=False, **hf_kwargs).logits
    ours_logits = ours(input_ids=input_ids, use_cache=False, **ours_kwargs).logits
    assert_tensors_match(
        ours_logits,
        hf_logits,
        equal=logits_equal,
        atol=atol,
        rtol=rtol,
        what="logits",
    )

    if labels is None:
        labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, **hf_kwargs)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False, **ours_kwargs)
    assert_tensors_match(ours_out.loss, hf_out.loss, equal=False, atol=atol, rtol=rtol, what="loss")
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


def assert_sequence_classification_matches_hf(ours: torch.nn.Module, *, supervision: str) -> None:
    """Check token-aligned classification against HF features and explicit CE.

    HF pools the last non-padding token; VeOmni returns every token's scores
    and supervises the positions with labels != -100, including packed samples.
    """
    from copy import deepcopy

    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification

    hf = AutoModelForSequenceClassification.from_config(deepcopy(ours.config))
    hf.load_state_dict(ours.state_dict())
    hf.eval()
    ours.eval()
    pin_eager_attn_implementation(hf)
    pin_eager_attn_implementation(ours)
    input_ids = torch.randint(3, ours.config.vocab_size, (3, 7))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 5:] = 0
    attention_mask[1, :2] = 0
    input_ids[attention_mask == 0] = ours.config.pad_token_id
    labels = torch.full_like(input_ids, -100)
    if supervision == "last-valid":
        rows, positions, targets = [0, 1, 2], [4, 6, 6], [1, 2, 3]
    else:
        # Unequal numbers of targets distinguish token mean from per-sequence mean.
        rows, positions, targets = [0, 0, 1, 2, 2, 2], [1, 4, 3, 0, 2, 6], [2, 1, 0, 3, 1, 2]
    labels[rows, positions] = torch.tensor(targets)
    hidden = hf.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    reference_logits = F.linear(hidden, hf.score.weight)
    reference_loss = F.cross_entropy(reference_logits[rows, positions].float(), torch.tensor(targets))
    if supervision == "last-valid":
        pooled = hf(input_ids=input_ids, attention_mask=attention_mask, labels=torch.tensor(targets), use_cache=False)
        torch.testing.assert_close(pooled.logits, reference_logits[rows, positions], atol=EAGER_ATOL, rtol=EAGER_RTOL)
        torch.testing.assert_close(pooled.loss, reference_loss, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    output = ours(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    torch.testing.assert_close(output.logits, reference_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(output.loss, reference_loss, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    reference_loss.backward()
    output.loss.backward()
    reference_params, actual_params = named_trainable(hf), named_trainable(ours)
    assert actual_params.keys() == reference_params.keys()
    for name, param in reference_params.items():
        if param.grad is None:
            assert actual_params[name].grad is None, name
        else:
            assert actual_params[name].grad is not None, name
            torch.testing.assert_close(
                actual_params[name].grad, param.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL, msg=name
            )


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
    """Compare a test-local official snapshot against models eager."""
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


def assert_module_forward_and_grads_match(
    hf_module: torch.nn.Module,
    ours_module: torch.nn.Module,
    *args: object,
    atol: float = EAGER_ATOL,
    rtol: float = EAGER_RTOL,
    grad_atol: float = EAGER_GRAD_ATOL,
    grad_rtol: float = EAGER_GRAD_RTOL,
) -> None:
    """Compare module outputs plus input and weight gradients against Hugging Face."""
    hf_args: list[object] = []
    ours_args: list[object] = []
    for value in args:
        if torch.is_tensor(value) and value.is_floating_point():
            hf_args.append(value.detach().clone().requires_grad_(True))
            ours_args.append(value.detach().clone().requires_grad_(True))
        elif torch.is_tensor(value):
            hf_args.append(value.clone())
            ours_args.append(value.clone())
        else:
            hf_args.append(value)
            ours_args.append(value)

    hf_module.train()
    ours_module.train()
    hf_out = hf_module(*hf_args)
    ours_out = ours_module(*ours_args)
    torch.testing.assert_close(ours_out, hf_out, atol=atol, rtol=rtol)

    grad_out = torch.randn_like(hf_out)
    hf_out.backward(grad_out)
    ours_out.backward(grad_out.clone())
    for hf_value, ours_value in zip(hf_args, ours_args, strict=True):
        if not (torch.is_tensor(hf_value) and hf_value.requires_grad):
            continue
        assert ours_value.grad is not None
        torch.testing.assert_close(ours_value.grad, hf_value.grad, atol=grad_atol, rtol=grad_rtol)

    hf_grads = named_trainable(hf_module)
    ours_grads = named_trainable(ours_module)
    assert hf_grads.keys() == ours_grads.keys()
    for name, param in hf_grads.items():
        if param.grad is None:
            assert ours_grads[name].grad is None, name
            continue
        assert ours_grads[name].grad is not None, name
        torch.testing.assert_close(ours_grads[name].grad, param.grad, atol=grad_atol, rtol=grad_rtol, msg=name)
