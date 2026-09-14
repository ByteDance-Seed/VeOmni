"""SeedVR2 VAE checkpoint validation must survive convolution inflation."""

import pytest
import torch

from veomni.models.diffusers.seedvr2.core.vae.causal_inflation_lib import InflatedCausalConv3d


def make_model(mode):
    return torch.nn.ModuleDict({"conv": InflatedCausalConv3d(2, 2, 3, inflation_mode=mode)})


@pytest.mark.parametrize("mode", ["none", "pad", "tail", "replicate"])
@pytest.mark.parametrize("key", ["conv.weight", "conv.bias"])
def test_seedvr2_vae_rejects_missing_keys(mode, key):
    model = make_model(mode)
    state = model.state_dict()
    del state[key]
    with pytest.raises(RuntimeError, match=key):
        model.load_state_dict(state, strict=True)
    result = model.load_state_dict(state, strict=False)
    assert result.missing_keys == [key]


@pytest.mark.parametrize("mode", ["none", "pad", "tail", "replicate"])
def test_seedvr2_vae_rejects_unexpected_keys(mode):
    model = make_model(mode)
    state = model.state_dict()
    state["conv.extra"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="conv.extra"):
        model.load_state_dict(state, strict=True)
    result = model.load_state_dict(state, strict=False)
    assert result.unexpected_keys == ["conv.extra"]


@pytest.mark.parametrize("mode", ["none", "pad", "tail", "replicate"])
def test_seedvr2_vae_complete_state_round_trip(mode):
    source, target = make_model(mode), make_model(mode)
    result = target.load_state_dict(source.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["tail", "replicate"])
def test_seedvr2_vae_2d_inflation_preserves_strict_loading(mode):
    model = make_model(mode)
    weight = torch.arange(36, dtype=torch.float32).reshape(2, 2, 3, 3)
    bias = torch.tensor([1.0, 2.0])
    result = model.load_state_dict({"conv.weight": weight, "conv.bias": bias}, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    expected = torch.zeros(2, 2, 3, 3, 3)
    if mode == "tail":
        expected[:, :, -1] = weight
    else:
        expected[:] = weight.unsqueeze(2) / 3
    torch.testing.assert_close(model["conv"].weight, expected, rtol=0, atol=0)
    torch.testing.assert_close(model["conv"].bias, bias, rtol=0, atol=0)
