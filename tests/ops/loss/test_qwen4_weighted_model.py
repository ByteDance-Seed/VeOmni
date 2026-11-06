"""Weighted-loss parity through the actual patched Qwen4-Exp model."""

import torch
import torch.nn.functional as F
from transformers import Qwen4ExpConfig

from tests.tools.training_utils import make_eager_ops_config
from veomni.models.configuration_base import as_veomni_config
from veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu import (
    Qwen4ExpForConditionalGeneration,
)


def test_qwen4_toy_weighted_loss_and_parameter_gradients_match_dense():
    """Compare weighted model loss and parameter gradients with dense CE."""
    torch.manual_seed(123)
    config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json")
    as_veomni_config(config, validate=False)._set_ops_implementation(
        make_eager_ops_config(cross_entropy_loss_implementation="chunk_loss")
    )
    config.use_cache = False
    model = Qwen4ExpForConditionalGeneration(config).float().train()
    ids = torch.tensor([[1, 5, 8, 9, 11, 14, 6, 2]])
    labels = torch.tensor([[-100, -100, 8, 9, 11, 14, 6, 2]])
    weights = torch.tensor([[0.0, 0.75, 0.25, 0.5, 1.0, 0.25, 0.5, 1.0]])
    positions = torch.arange(ids.size(1)).view(1, 1, -1).expand(4, -1, -1)
    kwargs = {
        "input_ids": ids,
        "position_ids": positions,
        "attention_mask": None,
        "use_cache": False,
        "cu_seq_lens_q": torch.tensor([0, ids.size(1)], dtype=torch.int32),
    }
    actual = model(**kwargs, labels=labels, loss_weights=weights).loss
    actual.backward()
    actual_grads = {
        name: param.grad.clone() if param.grad is not None else None for name, param in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)

    logits = model(**kwargs).logits
    target = labels[..., 1:]
    shifted_weights = weights[..., 1:]
    per_token = F.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.size(-1)).float(),
        target.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    expected = (per_token * shifted_weights.reshape(-1)).sum() / shifted_weights[target != -100].sum()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert sum(grad is not None for grad in actual_grads.values()) > 0
    for name, param in model.named_parameters():
        actual_grad = actual_grads[name]
        assert (actual_grad is None) == (param.grad is None), name
        if actual_grad is not None:
            assert torch.isfinite(actual_grad).all(), name
            torch.testing.assert_close(actual_grad, param.grad, atol=2e-5, rtol=2e-4, msg=name)
