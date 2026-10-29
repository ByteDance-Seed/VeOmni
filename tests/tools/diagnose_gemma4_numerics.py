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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded, single-CUDA diagnostic for the nine historical Gemma comparisons.

Run ``python -m tests.tools.diagnose_gemma4_numerics --output-dir PATH``.
This saves each original fixture and reports controlled interventions; it does
not relax or replace the historical whole-model acceptance thresholds.
"""

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM as HFTextLM

from tests.tools.gemma4_numerics import dense_masks, fp32_attention, precision_config, relative_l2, run_kernel
from tests.tools.training_utils import make_eager_ops_config
from veomni.models import build_foundation_model
from veomni.models.transformers.gemma4.generated import patched_modeling_gemma4_gpu as modeling


class _NativeForwardMathBackward(torch.autograd.Function):
    """Keep the real forward bitwise fixed and intervene only on attention VJPs."""

    @staticmethod
    def forward(ctx, query, key, value, native_output, mask, scaling):
        ctx.save_for_backward(query, key, value, mask)
        ctx.scaling = scaling
        return native_output

    @staticmethod
    def backward(ctx, upstream):
        query, key, value, mask = ctx.saved_tensors
        with torch.enable_grad():
            leaves = [tensor.detach().float().requires_grad_() for tensor in (query, key, value)]
            output = fp32_attention(None, *leaves, mask, scaling=ctx.scaling)[0]
            gradients = torch.autograd.grad(output, leaves, upstream.float())
        return (
            *(gradient.to(tensor.dtype) for gradient, tensor in zip(gradients, (query, key, value))),
            None,
            None,
            None,
        )


def _buffers(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()}


def _restore(model, fixture, dtype):
    model.load_state_dict(fixture["state"])
    for name, value in fixture["buffers"].items():
        owner, _, field = name.rpartition(".")
        target = model.get_submodule(owner) if owner else model
        buffer_dtype = torch.float32 if dtype == torch.float32 and value.is_floating_point() else value.dtype
        setattr(target, field, value.clone().to(dtype=buffer_dtype))
    model.cuda().train()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    for name, value in model.named_buffers():
        expected = fixture["buffers"][name]
        if dtype == torch.float32 and expected.is_floating_point():
            expected = expected.float()
        torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
    return model


def _new_model(fixture, backend, *, native=False, dtype=None):
    dtype = dtype or fixture["dtype"]
    cls = HFTextLM if native else modeling.Gemma4ForCausalLM
    model = cls._from_config(deepcopy(fixture["config"]), attn_implementation="eager", dtype=dtype)
    model = _restore(model, fixture, dtype)
    model.config._attn_implementation = backend if native else "veomni_" + backend
    return model


def _fixture(dtype, length, bidirectional, backend):
    """Reproduce original initialization/RNG consumption, then freeze all inputs."""
    torch.manual_seed(1242)
    config = precision_config(bidirectional)
    with tempfile.TemporaryDirectory() as path:
        config.save_pretrained(path)
        models = [
            build_foundation_model(
                path,
                torch_dtype=str(dtype).removeprefix("torch."),
                init_device="cpu",
                ops_implementation=make_eager_ops_config(attn_implementation=impl),
            )
            .cuda()
            .train()
            for impl in ("sdpa", backend)
        ]
    models[1].load_state_dict(models[0].state_dict())
    buffers = _buffers(models[0])
    for name, value in models[1].named_buffers():
        torch.testing.assert_close(value.cpu(), buffers[name], rtol=0, atol=0)
    ids = torch.randint(3, 100, (1, length), device="cuda")
    cu = torch.tensor([0, length // 2, length], device="cuda", dtype=torch.int32)
    fixture = dict(
        config=deepcopy(models[0].config),
        dtype=dtype,
        state={name: value.detach().cpu().clone() for name, value in models[0].state_dict().items()},
        buffers=buffers,
        inputs=dict(
            input_ids=ids,
            labels=ids.clone(),
            position_ids=torch.arange(length // 2, device="cuda").repeat(2).unsqueeze(0),
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=length // 2,
            max_length_k=length // 2,
            use_cache=False,
        ),
    )
    return fixture, models


def _run(model, fixture, *, backend, mode="actual", native=False, capture=False, flash_semantics=False):
    inputs = dict(fixture["inputs"])
    length = inputs["input_ids"].shape[-1]
    masks = dense_masks(model.config, length, inputs["cu_seq_lens_q"], device="cuda", flash=flash_semantics)
    if native:
        if backend == "flash_attention_2":
            inputs["attention_mask"] = dict.fromkeys(masks)
        elif backend == "flex_attention":
            inputs["attention_mask"] = {
                name: modeling._gemma4_create_mask(
                    config=model.config,
                    inputs_embeds=torch.zeros(1, length, 64, device="cuda", dtype=fixture["dtype"]),
                    attention_mask=None,
                    past_key_values=None,
                    cu_seq_lens_q=inputs["cu_seq_lens_q"],
                    sliding=name == "sliding_attention",
                )
                for name in masks
            }
    if mode == "math" or (backend == "sdpa" and flash_semantics):
        inputs["attention_mask"] = masks
    inputs["kernel_options"] = {"BACKEND": "TRITON", "num_stages": 1}
    # FA2's default atomic backward is nondeterministic; use the same
    # deterministic reduction in the candidate and native-HF control.
    inputs["deterministic"] = True
    key = model.config._attn_implementation
    original = ALL_ATTENTION_FUNCTIONS.get_interface(key, modeling.eager_attention_forward)
    records = []

    def forward(module, query, key_states, value, attention_mask, **kwargs):
        semantic_mask = masks[module.layer_type]
        if mode == "math":
            result = fp32_attention(module, query, key_states, value, semantic_mask, **kwargs)
        else:
            result = original(module, query, key_states, value, attention_mask, **kwargs)
        if mode == "hybrid":
            result = (
                _NativeForwardMathBackward.apply(
                    query, key_states, value, result[0].detach(), semantic_mask, kwargs.get("scaling", 1.0)
                ),
                None,
            )
        if capture:
            entry = dict(
                layer=module.layer_idx,
                layer_type=module.layer_type,
                qkv=[tensor.detach().cpu().clone() for tensor in (query, key_states, value)],
                kv_identity=(key_states.data_ptr(), value.data_ptr()),
            )
            result[0].register_hook(lambda grad, entry=entry: entry.update(upstream=grad.detach().cpu().clone()))
            records.append(entry)
        return result

    with patch.dict(ALL_ATTENTION_FUNCTIONS._global_mapping, {key: forward}):
        output = model(**inputs)
        output.loss.backward()
    result = dict(
        logits=output.logits.detach().cpu(),
        loss=output.loss.detach().cpu(),
        gradients={
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
        records=records,
    )
    assert torch.isfinite(result["loss"])
    assert all(torch.isfinite(value).all() for value in result["gradients"].values())
    return result


def _compare(actual, expected):
    assert actual["gradients"].keys() == expected["gradients"].keys()
    errors = {name: relative_l2(value, expected["gradients"][name]) for name, value in actual["gradients"].items()}
    failures = [
        name
        for name, value in actual["gradients"].items()
        if errors[name] >= 0.08 and (value.float() - expected["gradients"][name].float()).abs().max() >= 1e-4
    ]
    delta = actual["logits"].float() - expected["logits"].float()
    return dict(
        logit_relative_l2=relative_l2(actual["logits"], expected["logits"]),
        logit_max_abs=delta.abs().max().item(),
        original_logit_mismatches=int((delta.abs() > 0.05 + 0.05 * expected["logits"].float().abs()).sum()),
        loss_abs=(actual["loss"] - expected["loss"]).abs().item(),
        max_parameter_gradient_relative_l2=max(errors.values()),
        original_gradient_failures=failures,
        worst_gradients=sorted(errors.items(), key=lambda item: item[1], reverse=True)[:5],
    )


def _joint_replay(result, fixture, backend):
    """Retain shared K/V leaves across calls 0/2 and 1/3, including their VJPs."""
    records = result["records"]
    assert [entry["layer"] for entry in records] == list(range(4))
    for source, shared in ((0, 2), (1, 3)):
        assert records[source]["kv_identity"] == records[shared]["kv_identity"]
        for slot in (1, 2):
            assert torch.equal(records[source]["qkv"][slot], records[shared]["qkv"][slot])
    tensors = [entry["qkv"][0] for entry in records] + [records[i]["qkv"][slot] for slot in (1, 2) for i in (0, 1)]
    actual_leaves = [tensor.cuda().requires_grad_() for tensor in tensors]
    reference_leaves = [tensor.cuda().float().requires_grad_() for tensor in tensors]
    inputs = fixture["inputs"]
    length = inputs["input_ids"].shape[-1]
    config = deepcopy(fixture["config"])
    config._attn_implementation = "veomni_" + backend
    masks = dense_masks(config, length, inputs["cu_seq_lens_q"], device="cuda", flash=backend == "flash_attention_2")
    actual_outputs, reference_outputs, upstream = [], [], []
    output_errors = []
    for layer, entry in enumerate(records):
        pair = layer % 2
        indices = (layer, 4 + pair, 6 + pair)
        module = torch.nn.Module()
        module.config = config
        module.is_causal = getattr(config, "is_causal", True)
        module.sliding_window = config.sliding_window if pair == 0 else None
        mask = None
        if backend == "flex_attention":
            mask = modeling._gemma4_create_mask(
                config,
                torch.zeros(1, length, 64, device="cuda"),
                None,
                None,
                cu_seq_lens_q=inputs["cu_seq_lens_q"],
                sliding=pair == 0,
            )
        actual = run_kernel(
            module, [actual_leaves[i] for i in indices], mask, inputs["cu_seq_lens_q"], length, backend
        )
        expected = fp32_attention(
            None, *[reference_leaves[i] for i in indices], masks[entry["layer_type"]], scaling=1.0
        )[0]
        actual_outputs.append(actual)
        reference_outputs.append(expected)
        upstream.append(entry["upstream"].cuda())
        output_errors.append(relative_l2(actual, expected))
    actual_grads = torch.autograd.grad(actual_outputs, actual_leaves, upstream)
    reference_grads = torch.autograd.grad(reference_outputs, reference_leaves, [value.float() for value in upstream])
    return dict(
        output_relative_l2=output_errors,
        gradient_relative_l2=[relative_l2(a, b) for a, b in zip(actual_grads, reference_grads, strict=True)],
    )


def diagnose(dtype, backend, bidirectional, length, output_dir):
    torch.compiler.reset()
    name = f"{dtype}-{backend}-{'bidirectional' if bidirectional else 'causal'}-{length}"
    dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    fixture, (sdpa, actual_model) = _fixture(dtype, length, bidirectional, backend)
    saved = dict(fixture)
    saved["inputs"] = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in fixture["inputs"].items()
    }
    torch.save(saved, output_dir / f"{name}.pt")
    baseline = _run(sdpa, fixture, backend="sdpa", capture=True)
    actual = _run(actual_model, fixture, backend=backend, capture=True, flash_semantics=backend == "flash_attention_2")
    native = _run(_new_model(fixture, backend, native=True), fixture, backend=backend, native=True, capture=True)
    for a, b in zip(actual["records"], native["records"], strict=True):
        for x, y in zip(a["qkv"], b["qkv"], strict=True):
            torch.testing.assert_close(x, y)
    torch.testing.assert_close(actual["logits"], native["logits"])
    torch.testing.assert_close(actual["loss"], native["loss"])
    for key, value in actual["gradients"].items():
        torch.testing.assert_close(value, native["gradients"][key], msg=lambda msg, key=key: f"{key}: {msg}")
    report = dict(case=name, historical=_compare(actual, baseline), native_hf=_compare(actual, native))
    report["joint_qkv_replay"] = _joint_replay(actual, fixture, backend)
    hybrid = _run(
        _new_model(fixture, backend),
        fixture,
        backend=backend,
        mode="hybrid",
        flash_semantics=backend == "flash_attention_2",
    )
    assert torch.equal(hybrid["logits"], actual["logits"]), "Backward intervention changed forward"
    report["native_forward_fp32_vjp"] = _compare(actual, hybrid)
    oracle = _run(
        _new_model(fixture, "sdpa"),
        fixture,
        backend="sdpa",
        mode="math",
        flash_semantics=backend == "flash_attention_2",
    )
    report["fp32_attention_oracle"] = _compare(actual, oracle)
    full_fp32 = _run(
        _new_model(fixture, "sdpa", dtype=torch.float32),
        fixture,
        backend="sdpa",
        mode="math",
        flash_semantics=backend == "flash_attention_2",
    )
    report["fp32_model_oracle"] = _compare(actual, full_fp32)
    report["layer_qkv_drift_from_sdpa"] = [
        [relative_l2(a, b) for a, b in zip(x["qkv"], y["qkv"], strict=True)]
        for x, y in zip(actual["records"], baseline["records"], strict=True)
    ]
    if backend == "flash_attention_2" and bidirectional:
        corrected = _run(_new_model(fixture, "sdpa"), fixture, backend="sdpa", flash_semantics=True)
        report["native_window_sdpa_control"] = _compare(actual, corrected)
    (output_dir / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", help="Run one case name instead of all nine")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU with FA2 and Triton is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("bf16", backend, bidirectional, length)
        for backend in ("flex_attention", "flash_attention_2")
        for bidirectional in (False, True)
        for length in (128, 512)
    ]
    cases.append(("fp16", "flash_attention_2", False, 512))
    names = {
        f"{dtype}-{backend}-{'bidirectional' if bidirectional else 'causal'}-{length}"
        for dtype, backend, bidirectional, length in cases
    }
    if args.case is not None and args.case not in names:
        parser.error(f"Unknown case {args.case!r}; choose from {', '.join(sorted(names))}")
    for case in cases:
        name = f"{case[0]}-{case[1]}-{'bidirectional' if case[2] else 'causal'}-{case[3]}"
        if args.case is None or args.case == name:
            diagnose(*case, args.output_dir)


if __name__ == "__main__":
    main()
