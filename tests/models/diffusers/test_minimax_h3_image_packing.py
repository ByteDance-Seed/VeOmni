"""CPU equivalence checks for packed H3-Image training."""

import copy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config
from tests.models.tiny_configs import tiny_minimax_h3_config
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
    MiniMaxH3DiTModel,
)
from veomni.models.diffusers.minimax_h3_image.layout import pack_image_samples
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.configuration_minimax_h3_image_condition import (
    MiniMaxH3ImageConditionModelConfig,
)
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.modeling_minimax_h3_image_condition import (
    MiniMaxH3ImageConditionModel,
)
from veomni.trainer.dit_trainer import DiTDataCollator


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _model(attn_implementation="eager"):
    config = tiny_minimax_h3_config(
        audio_enabled=False, num_pdd_heads=6, image_training=True, patch_size=(1, 2, 2), num_layers=2
    )
    ops = eager_ops_config()
    ops.attn_implementation = attn_implementation
    config._set_ops_implementation(ops)
    return MiniMaxH3DiTModel(config)


def _samples():
    config = MiniMaxH3ImageConditionModelConfig(skip_encoder_load=True, seed=42)
    config._set_ops_implementation(eager_ops_config())
    condition = MiniMaxH3ImageConditionModel(config)
    rows = [
        {
            "clean_latents": torch.randn(1, 2, 2, 4),
            "prompt_embeds": torch.randn(1, 3, 16),
            "text_token_tags": torch.ones(3, dtype=torch.long),
        },
        {
            "clean_latents": torch.randn(1, 2, 4, 6),
            "prompt_embeds": torch.randn(1, 5, 16),
            "text_token_tags": torch.tensor([1, 0, 1, 1, 1]),
            "reference_latents": [torch.randn(1, 2, 2, 4), torch.randn(1, 2, 4, 2)],
        },
    ]
    samples = [condition.process_condition(**DiTDataCollator()([row])) for row in rows]
    for sample, head, weight in zip(samples, (0, 5), (0.25, 1.7)):
        sample["image_inputs"]["pdd_step_index"] = head
        sample["image_loss_weight"] = torch.tensor(weight)
        for key in ("x", "prompt_embeds", "unique_timesteps"):
            sample["image_inputs"][key] = sample["image_inputs"][key].detach().requires_grad_()
    return samples


def _batch(samples):
    return {key: [sample[key] for sample in samples] for key in samples[0]}


@pytest.mark.parametrize("checkpointing", [False, True])
def test_image_packing_matches_serial_outputs_gradients_and_adamw(checkpointing):
    torch.manual_seed(17)
    serial_model = _model()
    packed_model = copy.deepcopy(serial_model)
    serial_samples = _samples()
    packed_samples = copy.deepcopy(serial_samples)
    if checkpointing:
        packed_model.gradient_checkpointing_enable({"use_reentrant": False})
    block_calls = []
    handles = [
        block.register_forward_pre_hook(lambda module, args: block_calls.append(module))
        for block in packed_model.dit.blocks
    ]
    serial_outputs = [serial_model(**sample) for sample in serial_samples]
    packed_output = packed_model(**_batch(packed_samples))
    assert len(block_calls) == len(packed_model.dit.blocks)
    assert [prediction.shape[0] for prediction in packed_output.predictions] == [2, 6]
    for prediction, expected in zip(packed_output.predictions, serial_outputs):
        torch.testing.assert_close(prediction, expected.predictions[0], rtol=2e-5, atol=2e-6)
    serial_loss = torch.stack([output.loss["mse_image"] for output in serial_outputs]).mean()
    torch.testing.assert_close(packed_output.loss["mse_image"], serial_loss, rtol=2e-5, atol=2e-6)
    serial_loss.backward()
    packed_output.loss["mse_image"].backward()
    assert len(block_calls) == len(packed_model.dit.blocks) * (2 if checkpointing else 1)
    for handle in handles:
        handle.remove()
    for serial_sample, packed_sample in zip(serial_samples, packed_samples):
        for key in ("x", "prompt_embeds", "unique_timesteps"):
            torch.testing.assert_close(
                packed_sample["image_inputs"][key].grad,
                serial_sample["image_inputs"][key].grad,
                rtol=2e-4,
                atol=2e-6,
                msg=key,
            )
    for (name, serial_param), (_, packed_param) in zip(
        serial_model.named_parameters(), packed_model.named_parameters()
    ):
        if serial_param.requires_grad:
            assert serial_param.grad is not None, name
            torch.testing.assert_close(packed_param.grad, serial_param.grad, rtol=2e-4, atol=2e-6, msg=name)
        else:
            assert serial_param.grad is None and packed_param.grad is None, name
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in packed_model.named_parameters()
        if not parameter.requires_grad
    }
    for model in (serial_model, packed_model):
        torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3).step()
    for (name, serial_param), (_, packed_param) in zip(
        serial_model.named_parameters(), packed_model.named_parameters()
    ):
        torch.testing.assert_close(packed_param, serial_param, rtol=2e-5, atol=2e-6, msg=name)
        if name in frozen_before:
            torch.testing.assert_close(packed_param, frozen_before[name], rtol=0, atol=0, msg=name)


def test_image_packing_keeps_samples_isolated():
    torch.manual_seed(23)
    model = _model()
    samples = _samples()
    changed = copy.deepcopy(samples)
    for key in ("x", "prompt_embeds", "unique_timesteps"):
        changed[0]["image_inputs"][key] = changed[0]["image_inputs"][key].detach() + 0.7
    original = model(**_batch(samples))
    modified = model(**_batch(changed))
    torch.testing.assert_close(original.predictions[1], modified.predictions[1], rtol=0, atol=0)
    assert not torch.allclose(original.predictions[0], modified.predictions[0])
    original.predictions[1].sum().backward()
    for key in ("x", "prompt_embeds", "unique_timesteps"):
        assert torch.count_nonzero(samples[0]["image_inputs"][key].grad) == 0, key


def test_image_packing_preserves_local_positions_and_offsets_indices():
    torch.manual_seed(31)
    samples = [sample["image_inputs"] for sample in _samples()]
    packed, row_counts = pack_image_samples(samples)
    lengths = [sample["position_ids"].shape[0] for sample in samples]
    assert row_counts == [2, 10]
    assert packed["pdd_step_index"] == [(0, lengths[0]), (5, lengths[1])]
    assert packed["packed_seq_params"]["cu_seqlens_host"] == (0, lengths[0], sum(lengths))
    assert packed["refiner_packed_seq_params"]["cu_seqlens_host"] == (0, 3, 8)
    for key in ("packed_seq_params", "refiner_packed_seq_params"):
        torch.testing.assert_close(
            packed[key]["cu_seqlens_q"], torch.tensor(packed[key]["cu_seqlens_host"], dtype=torch.int32)
        )
    for key in ("x", "unique_timesteps", "token_tags", "position_ids"):
        torch.testing.assert_close(packed[key], torch.cat([sample[key] for sample in samples]), rtol=0, atol=0)
    torch.testing.assert_close(
        packed["prompt_embeds"], torch.cat([sample["prompt_embeds"].squeeze(0) for sample in samples]), rtol=0, atol=0
    )
    for key in ("image_indices", "text_indices"):
        torch.testing.assert_close(packed[key], torch.cat([samples[0][key], samples[1][key] + lengths[0]]))
    torch.testing.assert_close(
        packed["timestep_indices"],
        torch.cat([samples[0]["timestep_indices"], samples[1]["timestep_indices"] + 1]),
    )
    assert packed["position_ids"].dtype == packed["unique_timesteps"].dtype == torch.float32


def test_image_packing_requires_refiner_boundaries():
    packed, _ = pack_image_samples([sample["image_inputs"] for sample in _samples()])
    packed.pop("refiner_packed_seq_params")
    with pytest.raises(ValueError, match="text-refiner sequence boundaries"):
        _model().forward_image(**packed)


@pytest.mark.parametrize("key", ["position_ids", "unique_timesteps"])
def test_image_packing_rejects_lower_precision_metadata(key):
    samples = [sample["image_inputs"] for sample in _samples()]
    samples[0][key] = samples[0][key].to(torch.bfloat16)
    with pytest.raises(ValueError, match="precision|float32|cast_forward_inputs"):
        pack_image_samples(samples)


def test_image_packing_dispatches_flash_segments_and_matches_eager(monkeypatch):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import minimax_h3_dit
    from veomni.models.diffusers.minimax_h3.minimax_h3_transformer import (
        modeling_minimax_h3_transformer as h3_modeling,
    )

    torch.manual_seed(37)
    eager = _model()
    samples = _samples()
    expected = eager(**_batch(samples))
    expected.loss["mse_image"].backward()
    calls = []

    def bind_name(module, *, is_causal, impl=None):
        module.is_causal = is_causal
        module.config._attn_implementation = impl

    def packed_attention(module, query, key, value, *, cu_seqlens, max_seqlen, valid_seqlen):
        bounds = tuple(cu_seqlens.tolist())
        calls.append((module, bounds))
        assert max_seqlen == max(stop - start for start, stop in zip(bounds, bounds[1:]))
        assert valid_seqlen == bounds[-1]
        return minimax_h3_dit._sdpa_varlen_attention(query, key, value, bounds, module.softmax_scale, True)

    original_attention = minimax_h3_dit.MiniMaxH3Attention._attention

    def inspect_bounds(module, query, key, value, *, cu_seqlens, max_seqlen, valid_seqlen):
        assert isinstance(cu_seqlens, minimax_h3_dit._PackedBounds)
        return original_attention(
            module,
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            valid_seqlen=valid_seqlen,
        )

    monkeypatch.setattr(h3_modeling, "bind_minimax_attention", bind_name)
    monkeypatch.setattr(minimax_h3_dit.MiniMaxH3Attention, "_run_packed_attention", packed_attention)
    monkeypatch.setattr(minimax_h3_dit.MiniMaxH3Attention, "_attention", inspect_bounds)
    fused = _model("flash_attention_2")
    fused.load_state_dict(eager.state_dict())
    actual = fused(**_batch(copy.deepcopy(samples)))
    assert calls == [
        (fused.dit.token_refiner.blocks[0].attn, (0, 3)),
        (fused.dit.token_refiner.blocks[0].attn, (0, 5)),
        (fused.dit.blocks[0].attn, (0, 5, 20)),
        (fused.dit.blocks[1].attn, (0, 5, 20)),
    ]
    torch.testing.assert_close(actual.predictions, expected.predictions, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-5, atol=2e-6)
    actual.loss["mse_image"].backward()
    for (name, eager_parameter), (_, fused_parameter) in zip(eager.named_parameters(), fused.named_parameters()):
        torch.testing.assert_close(fused_parameter.grad, eager_parameter.grad, rtol=2e-4, atol=2e-6, msg=name)


def test_image_packing_uses_host_bounds_without_tensor_scalar_reads(monkeypatch):
    torch.manual_seed(41)
    model = _model()
    samples = _samples()
    reads = []

    def forbidden_tolist(tensor):
        raise AssertionError(f"Packed forward must use host bounds, got tolist on {tensor.shape}.")

    class RecordScalarReads(TorchDispatchMode):
        def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
            if operation == torch.ops.aten._local_scalar_dense.default:
                reads.append(operation)
            return operation(*args, **(kwargs or {}))

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", forbidden_tolist)
        with RecordScalarReads():
            output = model(**_batch(samples))
            output.loss["mse_image"].backward()
    assert reads == []
