"""Optimizer checkpoint keys must preserve their full parameter namespace."""

import pytest
import torch
from torch.distributed.tensor import Replicate

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint import dcp_checkpointer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("action", ["drop", "restore"])
@pytest.mark.parametrize(
    "key, expected, scalar",
    [
        ("state.embed_tokens.weight.exp_avg", "embed_tokens.weight", False),
        ("state.decoder.embed_tokens.weight.exp_avg", "decoder.embed_tokens.weight", False),
        ("state.decoder2.embed_tokens.weight.exp_avg_sq", "decoder2.embed_tokens.weight", False),
        ("state.decoder.embed_tokens.weight.step", None, True),
        ("state.decoder.embed_tokens.weight_extra.exp_avg", None, False),
        ("state.unrelated.decoder.embed_tokens.weight.exp_avg", None, False),
        ("param_groups.decoder.embed_tokens.weight.lr", None, False),
        ("decoder.embed_tokens.weight.exp_avg", None, False),
    ],
)
def test_optimizer_spec_matching_uses_the_full_namespace(monkeypatch, action, key, expected, scalar):
    selected = []

    class Spec:
        def __init__(self, name):
            self.name = name

        @property
        def placement(self):
            selected.append(self.name)
            return Replicate()

    specs = {
        name: Spec(name)
        for name in ("embed_tokens.weight", "decoder.embed_tokens.weight", "decoder2.embed_tokens.weight")
    }
    value = torch.ones(()) if scalar else torch.ones(2, 3)
    state = {key: value, "param_groups.decoder.embed_tokens.weight.amsgrad": False}
    monkeypatch.setattr(dcp_checkpointer, "_validate_extra_parallel_meshes", lambda _: None)
    result = dcp_checkpointer._apply_extra_parallel_dim(state, specs, None, action, key_match="substring")
    assert selected == ([] if expected is None else [expected])
    assert result is state and result[key] is value
    assert result["param_groups.decoder.embed_tokens.weight.amsgrad"] is False


@pytest.mark.parametrize("action", ["drop", "restore"])
def test_optimizer_spec_matching_still_rejects_ambiguous_specs(monkeypatch, action):
    monkeypatch.setattr(dcp_checkpointer, "_validate_extra_parallel_meshes", lambda _: None)
    with pytest.raises(AssertionError, match="Ambiguous ExtraParallel spec match"):
        dcp_checkpointer._apply_extra_parallel_dim(
            {"state.embedding.weight.exp_avg": torch.ones(2, 3)},
            dict.fromkeys(("embedding", "embedding.weight")),
            None,
            action,
            key_match="substring",
        )
