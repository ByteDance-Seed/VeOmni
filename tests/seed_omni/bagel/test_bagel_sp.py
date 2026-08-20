"""Distributed numerical parity for BAGEL sequence-parallel modules."""

from __future__ import annotations

from contextlib import nullcontext
from types import MethodType

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from tests.seed_omni.bagel.helpers import config_cls, model_cls, tiny_bagel_qwen2_cfg
from tests.tools.launch_utils import torchrun
from veomni.distributed.parallel_state import init_parallel_state, use_parallel_state
from veomni.models.seed_omni.accelerator.dispatch import call_graph_endpoint
from veomni.models.seed_omni.modules.bagel.qwen2_mot import accelerated as qwen2_mot_accelerated
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
    _OMNI_POSITION_IDS,
    _OMNI_TOKEN_LEN,
)
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_FLOW_HIDDEN, BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.vae.accelerated import BAGEL_VAE_PIXEL_SHAPE
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem
from veomni.ops.kernels.cross_entropy import install_loss_mapping
from veomni.utils.device import get_device_type, get_torch_device
from veomni.utils.save_safetensor_utils import get_model_save_state


def _sample_items(
    sample_index: int,
    device: torch.device,
    hidden_size: int,
) -> tuple[list[ConversationItem], list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(7300 + sample_index)

    def make_tensor(length: int) -> torch.Tensor:
        return torch.randn(
            length,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    if sample_index == 0:
        values = [make_tensor(2), make_tensor(3)]
        items = [
            ConversationItem(type="text", value=values[0], role="user"),
            ConversationItem(
                type="image",
                value=values[1],
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
            ),
        ]
    elif sample_index == 1:
        values = [make_tensor(2), make_tensor(3), make_tensor(1)]
        items = [
            ConversationItem(type="text", value=values[0], role="user"),
            ConversationItem(
                type="image",
                value=values[1],
                role="assistant",
                source=BAGEL_VAE_CONTEXT,
                meta={_IMG_TAG_KEY: "gen"},
            ),
            ConversationItem(type="text", value=values[2], role="assistant"),
        ]
    elif sample_index == 2:
        values = [make_tensor(2), make_tensor(5)]
        items = [
            ConversationItem(
                type="image",
                value=values[0],
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
            ),
            ConversationItem(type="text", value=values[1], role="assistant"),
        ]
    else:
        values = [make_tensor(1), make_tensor(2), make_tensor(2), make_tensor(3)]
        items = [
            ConversationItem(type="text", value=values[0], role="user"),
            ConversationItem(
                type="image",
                value=values[1],
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
            ),
            ConversationItem(
                type="image",
                value=values[2],
                role="assistant",
                source=BAGEL_VAE_CONTEXT,
                meta={_IMG_TAG_KEY: "gen"},
            ),
            ConversationItem(
                type="image",
                value=values[3],
                role="assistant",
                source=BAGEL_VAE_CONTEXT,
                meta={_IMG_TAG_KEY: "gen"},
            ),
        ]

    dummy = make_tensor(1)
    values.append(dummy)
    items.append(
        ConversationItem(
            type="image",
            value=dummy,
            role="dummy",
            source=BAGEL_SIGLIP_CONTEXT,
        )
    )
    return items, values


def _replicated_batch(
    device: torch.device,
    hidden_size: int,
) -> tuple[list[list[ConversationItem]], list[torch.Tensor]]:
    conversation: list[list[ConversationItem]] = []
    inputs: list[torch.Tensor] = []
    for sample_index in range(4):
        items, sample_inputs = _sample_items(sample_index, device, hidden_size)
        conversation.append(items)
        inputs.extend(sample_inputs)
    return conversation, inputs


def _carrier_hidden(conversation: list[list[ConversationItem]]) -> torch.Tensor:
    real_hidden = [
        item.value
        for sample in conversation
        for item in sample
        if item.role != "dummy" and torch.is_tensor(item.value)
    ]
    assert real_hidden
    return torch.cat(real_hidden, dim=0)


def _forward_qwen_carrier(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
    input_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
) -> torch.Tensor:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(
            device_type=get_device_type(),
            dtype=torch.bfloat16,
        ),
    ):
        inputs = model.forward_pre(conversation_list=conversation)
        if input_shapes is not None:
            input_shapes.append(
                (
                    tuple(inputs["packed_sequence"].shape),
                    tuple(inputs["packed_attention_metadata"].shape),
                )
            )
        outputs = model(**inputs)
        result = model.forward_post(**outputs)
    return _carrier_hidden(result["conversation_list"])


def _enable_scoped_gradient_checkpointing(model: torch.nn.Module, parallel_state) -> None:
    def context_fn():
        return nullcontext(), use_parallel_state(parallel_state)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": context_fn,
        }
    )


def _qwen2_mot_sp_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4
    device = torch.device(f"{get_device_type()}:{rank}")
    qwen2_mot_accelerated.veomni_rms_norm.bind("liger_kernel")
    qwen2_mot_accelerated.veomni_apply_rotary_pos_emb.bind("liger_kernel")
    qwen2_mot_accelerated.veomni_swiglu_mlp.bind("liger_kernel")

    non_sp_state = init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
    )
    sp_state = init_parallel_state(
        ulysses_size=world_size,
        dp_mode="fsdp2",
    )

    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    config_kwargs = {
        **tiny_bagel_qwen2_cfg(),
        # FlexAttention's Triton kernel requires head_dim >= 16.
        "hidden_size": 448,
        "intermediate_size": 896,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "attn_implementation": "veomni_flex_attention_with_sp",
    }
    torch.manual_seed(9102)
    reference = (
        BagelQwen2MoT(BagelQwen2MoTConfig(**config_kwargs))
        .to(
            device=device,
            dtype=torch.bfloat16,
        )
        .train()
    )
    torch.manual_seed(9102)
    sequence_parallel = (
        BagelQwen2MoT(BagelQwen2MoTConfig(**config_kwargs))
        .to(
            device=device,
            dtype=torch.bfloat16,
        )
        .train()
    )
    sequence_parallel.load_state_dict(reference.state_dict())

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in sequence_parallel.model.layers:
        fully_shard(layer, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)
    fully_shard(sequence_parallel, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)
    _enable_scoped_gradient_checkpointing(sequence_parallel, sp_state)

    qkv_checkpoint_keys = {
        f"model.layers.0.self_attn.{projection}.{kind}"
        for projection in ("q_proj", "k_proj", "v_proj", "q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen")
        for kind in ("weight", "bias")
    }
    exported_qkv = get_model_save_state(
        sequence_parallel,
        fqn_to_index_mapping=dict.fromkeys(qkv_checkpoint_keys, 0),
        parallel_state=sp_state,
    )
    assert set(exported_qkv) == qkv_checkpoint_keys
    assert tuple(exported_qkv["model.layers.0.self_attn.q_proj.weight"].shape) == (448, 448)
    assert tuple(exported_qkv["model.layers.0.self_attn.k_proj.weight"].shape) == (64, 448)
    assert tuple(exported_qkv["model.layers.0.self_attn.v_proj.weight"].shape) == (64, 448)

    hidden_size = int(reference.config.hidden_size)
    reference_conversation, reference_inputs = _replicated_batch(device, hidden_size)
    sp_conversation, sp_inputs = _replicated_batch(device, hidden_size)
    expected_sample_lengths = [5, 6, 7, 8]
    input_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    reference_hidden = _forward_qwen_carrier(reference, reference_conversation, non_sp_state)
    sp_hidden = _forward_qwen_carrier(sequence_parallel, sp_conversation, sp_state, input_shapes)
    assert input_shapes == [((7, hidden_size), (3, 28))]
    assert sequence_parallel._metric_full_seqlens["forward"] == expected_sample_lengths
    assert torch.isfinite(reference_hidden).all()
    assert torch.isfinite(sp_hidden).all()
    torch.testing.assert_close(sp_hidden, reference_hidden, rtol=2e-2, atol=2e-2)

    reference_loss = reference_hidden.float().square().mean()
    sp_loss = sp_hidden.float().square().mean()
    reference_loss.backward()
    sp_loss.backward()

    for reference_input, sp_input in zip(reference_inputs, sp_inputs, strict=True):
        assert reference_input.grad is not None
        assert sp_input.grad is not None
        assert torch.isfinite(reference_input.grad).all()
        assert torch.isfinite(sp_input.grad).all()
        sp_grad = sp_input.grad.detach().clone()
        dist.all_reduce(sp_grad, op=dist.ReduceOp.SUM, group=sp_state.sp_group)
        sp_grad /= world_size
        torch.testing.assert_close(sp_grad, reference_input.grad, rtol=3e-2, atol=3e-2)

    reference_grads: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        assert parameter.grad is not None, f"reference parameter {name} did not participate in backward"
        assert torch.isfinite(parameter.grad).all()
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        reference_grads[name] = grad / world_size

    for name, parameter in sequence_parallel.named_parameters():
        assert parameter.grad is not None, f"SP parameter {name} did not participate in backward"
        local_grad = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        assert torch.isfinite(local_grad).all()
        sp_grad = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
        torch.testing.assert_close(
            sp_grad,
            reference_grads[name],
            rtol=5e-2,
            atol=5e-2,
            msg=f"SP gradient mismatch for {name}",
        )

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
def test_qwen2_mot_sp4_matches_non_sp_with_fsdp2_and_gradient_checkpointing() -> None:
    torchrun(_qwen2_mot_sp_worker, world_size=4)


def _text_encoder_sp_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    device = torch.device(f"{get_device_type()}:{rank}")
    module_state = init_parallel_state(ulysses_size=world_size, dp_mode="ddp")
    install_loss_mapping("eager")

    BagelTextEncoder = model_cls("bagel_text_encoder")
    BagelTextEncoderConfig = config_cls("bagel_text_encoder")
    with use_parallel_state(module_state):
        torch.manual_seed(7301)
        model = BagelTextEncoder(
            BagelTextEncoderConfig(
                vocab_size=32,
                hidden_size=8,
                tie_word_embeddings=False,
            )
        ).to(device=device, dtype=torch.float32)

    # Uniform SP replicates the same DP sample across every SP rank. Use an odd
    # sequence length so encode_pre must pad to six tokens before slicing it into
    # two equal local shards.
    input_ids = torch.arange(5, device=device, dtype=torch.long)
    item = ConversationItem(
        type="text",
        value=input_ids,
        role="assistant",
        meta={
            "_omni_tokenized": True,
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        },
    )
    conversation = [[item]]

    with use_parallel_state(module_state):
        encode_inputs = model.encode_pre(conversation_list=conversation)
        assert encode_inputs["input_ids"].shape == (3,)
        encoded = model.encode(**encode_inputs)
        model.encode_post(**encoded)

    expected_embeds = model.embed_tokens(input_ids)
    torch.testing.assert_close(item.value, expected_embeds)
    item.value.retain_grad()

    with use_parallel_state(module_state):
        decode_inputs = model.decode_pre(conversation_list=conversation)
        assert decode_inputs["hidden_states"].shape == (5, model.config.hidden_size)
        decoded = model.decode(**decode_inputs)
        outputs = model.decode_post(**decoded)

    loss = outputs["_loss"]
    assert torch.isfinite(loss)
    gathered_losses = [torch.empty_like(loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, loss.detach())
    for gathered_loss in gathered_losses:
        torch.testing.assert_close(gathered_loss, loss.detach())

    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
    assert item.value.grad is not None
    assert torch.isfinite(item.value.grad).all()

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
def test_bagel_text_encoder_sp2_handles_replicated_padded_sequence() -> None:
    torchrun(_text_encoder_sp_worker, world_size=2)


def _replicated_siglip_conversation(
    device: torch.device,
    patch_dim: int,
) -> tuple[list[list[ConversationItem]], list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(8124)
    conversation: list[list[ConversationItem]] = []
    inputs: list[torch.Tensor] = []
    for token_len in (2, 3, 5):
        value = torch.randn(
            token_len,
            patch_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        inputs.append(value)
        conversation.append(
            [
                ConversationItem(
                    type="image",
                    value=value,
                    role="user",
                    source=BAGEL_SIGLIP_CONTEXT,
                    meta={
                        _OMNI_POSITION_IDS: torch.arange(token_len, device=device),
                        _OMNI_TOKEN_LEN: token_len,
                    },
                )
            ]
        )
    return conversation, inputs


def _carrier_embeds(conversation: list[list[ConversationItem]]) -> torch.Tensor:
    return torch.cat([sample[0].value for sample in conversation], dim=0)


def _forward_siglip_carrier(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
) -> tuple[torch.Tensor, list[int]]:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(device_type=get_device_type(), dtype=torch.bfloat16),
    ):
        inputs = model.forward_pre(conversation_list=conversation)
        local_token_lens = inputs["token_lens"].detach().cpu().tolist()
        outputs = model(**inputs)
        result = model.forward_post(**outputs)
    return _carrier_embeds(result["conversation_list"]), local_token_lens


def _siglip_navit_sp_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4
    device = torch.device(f"{get_device_type()}:{rank}")

    non_sp_state = init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
    )
    sp_state = init_parallel_state(
        ulysses_size=world_size,
        dp_mode="fsdp2",
    )

    BagelSiglip = model_cls("bagel_siglip_navit")
    BagelSiglipConfig = config_cls("bagel_siglip_navit")
    config = BagelSiglipConfig(
        hidden_size=16,
        output_size=16,
        image_size=8,
        min_image_size=2,
        max_pixels=64,
        intermediate_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_channels=1,
        patch_size=2,
        vit_max_num_patch_per_side=4,
    )
    torch.manual_seed(4901)
    reference = BagelSiglip(config).to(device=device, dtype=torch.bfloat16).train()
    torch.manual_seed(4901)
    sequence_parallel = BagelSiglip(config).to(device=device, dtype=torch.bfloat16).train()
    sequence_parallel.load_state_dict(reference.state_dict())

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in sequence_parallel.vision_model.encoder.layers:
        fully_shard(layer, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)
    fully_shard(sequence_parallel, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)

    patch_dim = int(config.num_channels * config.patch_size * config.patch_size)
    reference_conversation, reference_inputs = _replicated_siglip_conversation(device, patch_dim)
    sp_conversation, sp_inputs = _replicated_siglip_conversation(device, patch_dim)

    reference_embeds, reference_token_lens = _forward_siglip_carrier(reference, reference_conversation, non_sp_state)
    sp_embeds, local_token_lens = _forward_siglip_carrier(sequence_parallel, sp_conversation, sp_state)

    assert reference_token_lens == [2, 3, 5]
    assert local_token_lens == [[2], [3], [5], [1]][rank]
    assert sequence_parallel._metric_full_seqlens["forward"] == [2, 3, 5]
    assert torch.isfinite(reference_embeds).all()
    assert torch.isfinite(sp_embeds).all()
    torch.testing.assert_close(sp_embeds, reference_embeds, rtol=2e-2, atol=2e-2)

    reference_embeds.float().square().mean().backward()
    sp_embeds.float().square().mean().backward()

    for reference_input, sp_input in zip(reference_inputs, sp_inputs, strict=True):
        assert reference_input.grad is not None
        assert sp_input.grad is not None
        assert torch.isfinite(reference_input.grad).all()
        assert torch.isfinite(sp_input.grad).all()
        sp_grad = sp_input.grad.detach().clone()
        dist.all_reduce(sp_grad, op=dist.ReduceOp.SUM, group=sp_state.sp_group)
        sp_grad /= world_size
        torch.testing.assert_close(sp_grad, reference_input.grad, rtol=3e-2, atol=3e-2)

    reference_grads: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"reference parameter {name} did not participate in backward"
        assert torch.isfinite(parameter.grad).all()
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        reference_grads[name] = grad / world_size

    for name, parameter in sequence_parallel.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"SP parameter {name} did not participate in backward"
        local_grad = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        assert torch.isfinite(local_grad).all()
        sp_grad = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
        torch.testing.assert_close(
            sp_grad,
            reference_grads[name],
            rtol=5e-2,
            atol=5e-2,
            msg=f"SP gradient mismatch for {name}",
        )

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
def test_siglip_navit_sp4_matches_non_sp_with_variable_image_lengths() -> None:
    torchrun(_siglip_navit_sp_worker, world_size=4)


def _replicated_vae_conversation(
    device: torch.device,
) -> tuple[list[list[ConversationItem]], list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(6117)
    conversation: list[list[ConversationItem]] = []
    inputs: list[torch.Tensor] = []
    for height, width in ((8, 8), (8, 4), (4, 8)):
        value = torch.zeros(3, 8, 8, device=device, dtype=torch.bfloat16)
        real_pixels = torch.randn(
            3,
            height,
            width,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        value[:, :height, :width] = real_pixels
        value.requires_grad_()
        inputs.append(value)
        conversation.append(
            [
                ConversationItem(
                    type="image",
                    value=value,
                    role="assistant",
                    source=BAGEL_VAE_CONTEXT,
                    meta={BAGEL_VAE_PIXEL_SHAPE: torch.tensor([height, width], device=device)},
                )
            ]
        )
    return conversation, inputs


def _deterministic_latents(self, posterior: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mean, _ = posterior
    return self.config.scale_factor * (mean - self.config.shift_factor)


def _forward_encode(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
) -> tuple[list[torch.Tensor], tuple[int, ...]]:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(device_type=get_device_type(), dtype=torch.bfloat16),
    ):
        inputs = model.encode_pre(conversation_list=conversation)
        local_shape = tuple(inputs["pixel_values"].shape)
        outputs = call_graph_endpoint(model, model, method="encode", kwargs=inputs)
        result = model.encode_post(**outputs)
    return [sample[0].value for sample in result["conversation_list"]], local_shape


def _forward_offline_encode(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
) -> list[torch.Tensor]:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(device_type=get_device_type(), dtype=torch.bfloat16),
    ):
        inputs = model.encode_pre(conversation_list=conversation)
        outputs = call_graph_endpoint(model, model, method="offline_encode", kwargs=inputs)
        result = model.offline_encode_post(**outputs)
    return [sample[0].value for sample in result["conversation_list"]]


def _vae_sp_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4
    device = torch.device(f"{get_device_type()}:{rank}")

    non_sp_state = init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
    )
    sp_state = init_parallel_state(
        ulysses_size=world_size,
        dp_mode="fsdp2",
    )

    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    config_kwargs = {
        "resolution": 8,
        "in_channels": 3,
        "downsample": 2,
        "ch": 32,
        "ch_mult": [1, 1],
        "num_res_blocks": 1,
        "z_channels": 2,
        "max_image_size": 8,
        "min_image_size": 4,
        "image_stride": 4,
        "max_pixels": 64,
        "freeze": False,
    }
    torch.manual_seed(2271)
    reference = BagelVAE(BagelVAEConfig(**config_kwargs)).to(device=device, dtype=torch.bfloat16).train()
    torch.manual_seed(2271)
    sequence_parallel = (
        BagelVAE(BagelVAEConfig(**config_kwargs))
        .to(
            device=device,
            dtype=torch.bfloat16,
        )
        .train()
    )
    sequence_parallel.load_state_dict(reference.state_dict())
    reference._sample_scaled_latents = MethodType(_deterministic_latents, reference)
    sequence_parallel._sample_scaled_latents = MethodType(_deterministic_latents, sequence_parallel)

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(sequence_parallel, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)

    reference_conversation, reference_inputs = _replicated_vae_conversation(device)
    sp_conversation, sp_inputs = _replicated_vae_conversation(device)
    reference_latents, reference_local_shape = _forward_encode(reference, reference_conversation, non_sp_state)
    sp_latents, sp_local_shape = _forward_encode(sequence_parallel, sp_conversation, sp_state)

    assert reference_local_shape == (3, 3, 8, 8)
    assert sp_local_shape == (1, 3, 8, 8)
    assert sequence_parallel._metric_full_seqlens["encode"] == [16, 8, 8]
    assert [tuple(latent.shape) for latent in sp_latents] == [(2, 4, 4), (2, 4, 2), (2, 2, 4)]
    for reference_latent, sp_latent in zip(reference_latents, sp_latents, strict=True):
        assert torch.isfinite(reference_latent).all()
        assert torch.isfinite(sp_latent).all()
        torch.testing.assert_close(sp_latent, reference_latent, rtol=2e-2, atol=2e-2)

    reference_loss = sum(latent.float().square().mean() for latent in reference_latents)
    sp_loss = sum(latent.float().square().mean() for latent in sp_latents)
    reference_loss.backward()
    sp_loss.backward()

    for reference_input, sp_input in zip(reference_inputs, sp_inputs, strict=True):
        assert reference_input.grad is not None
        assert sp_input.grad is not None
        assert torch.isfinite(reference_input.grad).all()
        assert torch.isfinite(sp_input.grad).all()
        sp_grad = sp_input.grad.detach().clone()
        dist.all_reduce(sp_grad, op=dist.ReduceOp.SUM, group=sp_state.sp_group)
        sp_grad /= world_size
        torch.testing.assert_close(sp_grad, reference_input.grad, rtol=3e-2, atol=3e-2)

    reference_grads: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all()
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        reference_grads[name] = grad / world_size

    for name, parameter in sequence_parallel.named_parameters():
        if name not in reference_grads:
            assert parameter.grad is None
            continue
        assert parameter.grad is not None, f"SP parameter {name} did not participate in backward"
        local_grad = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        assert torch.isfinite(local_grad).all()
        sp_grad = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
        torch.testing.assert_close(
            sp_grad,
            reference_grads[name],
            rtol=5e-2,
            atol=5e-2,
            msg=f"SP gradient mismatch for {name}",
        )

    reference_offline_conversation, _ = _replicated_vae_conversation(device)
    sp_offline_conversation, _ = _replicated_vae_conversation(device)
    reference_cache = _forward_offline_encode(reference, reference_offline_conversation, non_sp_state)
    sp_cache = _forward_offline_encode(sequence_parallel, sp_offline_conversation, sp_state)
    assert [tuple(cache.shape) for cache in sp_cache] == [(2, 2, 4, 4), (2, 2, 4, 2), (2, 2, 2, 4)]
    for reference_item_cache, sp_item_cache in zip(reference_cache, sp_cache, strict=True):
        assert torch.isfinite(reference_item_cache).all()
        assert torch.isfinite(sp_item_cache).all()
        torch.testing.assert_close(sp_item_cache, reference_item_cache, rtol=2e-2, atol=2e-2)

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
def test_vae_sp4_matches_non_sp_for_encode_and_offline_encode() -> None:
    torchrun(_vae_sp_worker, world_size=4)


def _replicated_embed_conversation(device: torch.device) -> list[list[ConversationItem]]:
    generator = torch.Generator(device=device).manual_seed(9182)
    conversation: list[list[ConversationItem]] = []
    for height, width in ((2, 2), (1, 2), (1, 3)):
        conversation.append(
            [
                ConversationItem(
                    type="image",
                    value=torch.randn(
                        1,
                        height,
                        width,
                        generator=generator,
                        device=device,
                        dtype=torch.bfloat16,
                    ),
                    role="assistant",
                    source=BAGEL_VAE_CONTEXT,
                    meta={_IMG_TAG_KEY: "edit"},
                )
            ]
        )
    return conversation


def _replicated_decode_conversation(
    device: torch.device,
    *,
    hidden_size: int,
    patch_latent_dim: int,
) -> tuple[list[list[ConversationItem]], list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(1743)
    conversation: list[list[ConversationItem]] = []
    inputs: list[torch.Tensor] = []
    for token_count in (4, 2, 3):
        hidden = torch.randn(
            token_count,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        target = torch.randn(
            token_count,
            patch_latent_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        inputs.append(hidden)
        conversation.append(
            [
                ConversationItem(
                    type="image",
                    value=hidden,
                    role="assistant",
                    source=BAGEL_FLOW_HIDDEN,
                    meta={"flow_velocity_target": target},
                )
            ]
        )
    return conversation, inputs


def _forward_embed(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(device_type=get_device_type(), dtype=torch.bfloat16),
    ):
        inputs = model.embed_latent_pre(conversation_list=conversation)
        local_shape = tuple(inputs["latents"].shape)
        outputs = call_graph_endpoint(model, model, method="embed_latent", kwargs=inputs)
        result = model.embed_latent_post(**outputs)
    embeds = torch.cat([sample[0].value for sample in result["conversation_list"]], dim=0)
    return embeds, local_shape


def _forward_decode(
    model: torch.nn.Module,
    conversation: list[list[ConversationItem]],
    parallel_state,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    with (
        use_parallel_state(parallel_state),
        torch.autocast(device_type=get_device_type(), dtype=torch.bfloat16),
    ):
        inputs = model.decode_velocity_pre(conversation_list=conversation)
        local_shape = tuple(inputs["hidden_states"].shape)
        outputs = call_graph_endpoint(model, model, method="decode_velocity", kwargs=inputs)
        result = model.decode_velocity_post(**outputs)
    velocity = torch.cat([sample[0].value for sample in result["conversation_list"]], dim=0)
    return velocity, result["_loss"], local_shape


def _flow_connector_sp_worker() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4
    device = torch.device(f"{get_device_type()}:{rank}")

    non_sp_state = init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
    )
    sp_state = init_parallel_state(
        ulysses_size=world_size,
        dp_mode="fsdp2",
    )

    BagelFlowConnector = model_cls("bagel_flow_connector")
    BagelFlowConnectorConfig = config_cls("bagel_flow_connector")
    config = BagelFlowConnectorConfig(
        hidden_size=8,
        z_channels=1,
        latent_patch_size=1,
        patch_latent_dim=1,
        max_latent_size=4,
        timestep_frequency_embedding_size=4,
        timestep_shift=1.0,
    )
    torch.manual_seed(6029)
    reference = BagelFlowConnector(config).to(device=device, dtype=torch.bfloat16).train()
    with torch.no_grad():
        reference.llm2vae.weight.normal_(mean=0.0, std=0.1)
        reference.llm2vae.bias.normal_(mean=0.0, std=0.1)
    torch.manual_seed(6029)
    sequence_parallel = BagelFlowConnector(config).to(device=device, dtype=torch.bfloat16).train()
    sequence_parallel.load_state_dict(reference.state_dict())

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(sequence_parallel, mesh=sp_state.fsdp_mesh, mp_policy=mp_policy)

    reference_embed_conversation = _replicated_embed_conversation(device)
    sp_embed_conversation = _replicated_embed_conversation(device)
    reference_embeds, reference_embed_shape = _forward_embed(
        reference,
        reference_embed_conversation,
        non_sp_state,
    )
    sp_embeds, sp_embed_shape = _forward_embed(sequence_parallel, sp_embed_conversation, sp_state)

    assert reference_embed_shape == (9, 1)
    assert sp_embed_shape == (3, 1)
    assert sequence_parallel._metric_full_seqlens["embed_latent"] == [4, 2, 3]
    assert torch.isfinite(reference_embeds).all()
    assert torch.isfinite(sp_embeds).all()
    torch.testing.assert_close(sp_embeds, reference_embeds, rtol=2e-2, atol=2e-2)

    reference_decode_conversation, reference_inputs = _replicated_decode_conversation(
        device,
        hidden_size=int(config.hidden_size),
        patch_latent_dim=int(config.patch_latent_dim),
    )
    sp_decode_conversation, sp_inputs = _replicated_decode_conversation(
        device,
        hidden_size=int(config.hidden_size),
        patch_latent_dim=int(config.patch_latent_dim),
    )
    reference_velocity, reference_decode_loss, reference_decode_shape = _forward_decode(
        reference,
        reference_decode_conversation,
        non_sp_state,
    )
    sp_velocity, sp_decode_loss, sp_decode_shape = _forward_decode(
        sequence_parallel,
        sp_decode_conversation,
        sp_state,
    )

    assert reference_decode_shape == (9, int(config.hidden_size))
    assert sp_decode_shape == (3, int(config.hidden_size))
    assert sequence_parallel._metric_full_seqlens["decode_velocity"] == [4, 2, 3]
    assert torch.isfinite(reference_velocity).all()
    assert torch.isfinite(sp_velocity).all()
    assert torch.isfinite(reference_decode_loss)
    assert torch.isfinite(sp_decode_loss)
    torch.testing.assert_close(sp_velocity, reference_velocity, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(sp_decode_loss, reference_decode_loss, rtol=2e-2, atol=2e-2)

    reference_loss = reference_embeds.float().square().mean() + reference_decode_loss.float()
    sp_loss = sp_embeds.float().square().mean() + sp_decode_loss.float()
    reference_loss.backward()
    sp_loss.backward()

    for reference_input, sp_input in zip(reference_inputs, sp_inputs, strict=True):
        assert reference_input.grad is not None
        assert sp_input.grad is not None
        assert torch.isfinite(reference_input.grad).all()
        assert torch.isfinite(sp_input.grad).all()
        sp_grad = sp_input.grad.detach().clone()
        dist.all_reduce(sp_grad, op=dist.ReduceOp.SUM, group=sp_state.sp_group)
        sp_grad /= world_size
        torch.testing.assert_close(sp_grad, reference_input.grad, rtol=3e-2, atol=3e-2)

    reference_grads: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"reference parameter {name} did not participate in backward"
        assert torch.isfinite(parameter.grad).all()
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        reference_grads[name] = grad / world_size

    for name, parameter in sequence_parallel.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"SP parameter {name} did not participate in backward"
        local_grad = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        assert torch.isfinite(local_grad).all()
        sp_grad = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
        torch.testing.assert_close(
            sp_grad,
            reference_grads[name],
            rtol=5e-2,
            atol=5e-2,
            msg=f"SP gradient mismatch for {name}",
        )

    dist.barrier()


@pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
def test_flow_connector_sp4_matches_non_sp_for_embed_and_decode() -> None:
    torchrun(_flow_connector_sp_worker, world_size=4)
