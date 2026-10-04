"""Shared-prefix training on an accelerator: the full Qwen3.5 model, flag on vs off.

A packed micro-batch holds rollouts that share a prompt, as GRPO produces. With
``text_config.shared_prefix_training`` the backbone computes the prompt once.
Log-probs and every parameter gradient must stay within the noise floor of
packed training, measured here as the difference between running the same
sequences one per call and packed together (different kernel shapes).
"""

import pytest
import torch

from veomni.utils.device import get_device_type


DEVICE = get_device_type()
pytestmark = pytest.mark.skipif(
    DEVICE not in ("cuda", "npu"), reason="shared-prefix training needs the GPU or NPU modeling path"
)

TOY_CONFIG = "tests/toy_config/qwen3_5_toy"
PREFIX, SUFFIX, GROUP = 1990, 96, 3  # prefix deliberately not a multiple of the 64-token chunk

# The GDN ops the two modeling paths bind: vendored Triton on NPU, flash-linear-attention on GPU.
GDN_BACKEND = {"npu": "npu", "cuda": "fla"}[DEVICE] if DEVICE in ("cuda", "npu") else "eager"


@pytest.fixture(scope="module")
def model():
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model

    torch.manual_seed(0)
    model = build_foundation_model(
        config_path=TOY_CONFIG,
        weights_path=None,
        torch_dtype="bfloat16",
        init_device=DEVICE,
        ops_implementation=OpsImplementationConfig(
            attn_implementation="veomni_flash_attention_2_with_sp",
            chunk_gated_delta_rule_implementation=GDN_BACKEND,
            causal_conv1d_implementation=GDN_BACKEND,
            rms_norm_gated_implementation=GDN_BACKEND,
        ),
    )
    model.train()
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model


def _sequences():
    torch.manual_seed(1)
    prefix = torch.randint(1000, 60000, (PREFIX,))
    shared = [torch.cat([prefix, torch.randint(1000, 60000, (SUFFIX,))]) for _ in range(GROUP)]
    return shared + [torch.randint(1000, 60000, (300,))]  # plus one unrelated sequence


def _run(model, seqs, weights):
    """Packed forward + backward; returns per-sequence log-probs and the gradients."""
    ids = torch.cat(seqs)[None].to(DEVICE)
    lens = [len(s) for s in seqs]
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).to(DEVICE)
    pos = torch.cat([torch.arange(n) for n in lens]).to(DEVICE).view(1, 1, -1).expand(4, 1, -1).contiguous()
    out = model(
        input_ids=ids,
        position_ids=pos,
        labels=ids,
        shift_labels=torch.roll(ids, -1, 1),
        return_log_probs=True,
        temperature=1.0,
        use_cache=False,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=max(lens),
        max_length_k=max(lens),
    )
    logp = out.log_probs.reshape(-1)
    per_seq = [logp[a : b - 1].float() for a, b in zip(cu[:-1].tolist(), cu[1:].tolist())]
    sum((lp * w).sum() for lp, w in zip(per_seq, weights)).backward()
    grads = {n: p.grad.float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return [lp.detach().cpu() for lp in per_seq], grads


def _distance(a, b):
    logp = max((x - y).abs().max().item() for x, y in zip(a[0], b[0]))
    grad = max(((a[1][n] - b[1][n]).norm() / b[1][n].norm().clamp_min(1e-12)).item() for n in b[1])
    return logp, grad


def test_shared_prefix_matches_packed_training(model, monkeypatch):
    from veomni.models.transformers.qwen3_5 import shared_prefix

    seqs = _sequences()
    weights = [torch.randn(len(s) - 1).to(DEVICE) for s in seqs]

    model.config.text_config.shared_prefix_training = False
    packed = _run(model, seqs, weights)
    separate = [_run(model, [s], [w]) for s, w in zip(seqs, weights)]
    separate = ([lp for r in separate for lp in r[0]], {n: sum(r[1][n] for r in separate) for n in packed[1]})

    calls = []
    original = shared_prefix.SharedPrefixPlan.gated_delta_rule
    monkeypatch.setattr(
        shared_prefix.SharedPrefixPlan,
        "gated_delta_rule",
        lambda self, *a, **k: calls.append(1) or original(self, *a, **k),
    )
    model.config.text_config.shared_prefix_training = True
    shared = _run(model, seqs, weights)

    linear_layers = model.config.text_config.layer_types.count("linear_attention")
    assert len(calls) >= linear_layers, "the shared-prefix path was not taken"
    floor_logp, floor_grad = _distance(separate, packed)
    logp, grad = _distance(shared, packed)
    assert logp <= 2 * floor_logp + 1e-3, (logp, floor_logp)
    assert grad <= 2 * floor_grad + 1e-3, (grad, floor_grad)
