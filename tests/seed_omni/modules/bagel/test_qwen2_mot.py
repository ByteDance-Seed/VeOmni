"""BAGEL module registry, YAML graph configs, MoT attention masks, QKV checkpoint conversion, and ops."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
from torch.nn.attention.flex_attention import BlockMask
from transformers.utils import SAFE_WEIGHTS_NAME

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.conftest import available_nvidia_ops  # noqa: F401
from tests.seed_omni.modules.bagel.helpers import (
    bagel_cfg_dir,
    load_module_runtime_args,
    load_omni_config,
    tiny_bagel_qwen2_cfg,
)
from veomni.models.checkpoint.weights import load_model_weights
from veomni.models.seed_omni import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated import accelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated.accelerated import BagelQwen2MoTAttentionAccelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.configuration import BagelQwen2MoTConfig
from veomni.models.seed_omni.modules.bagel.qwen2_mot.masking import (
    build_mot_attention_metadata,
    build_mot_magi_mask,
    build_mot_sdpa_mask,
    pad_mot_attention_metadata,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT, BagelQwen2RotaryEmbedding
from veomni.models.seed_omni.modules.bagel.siglip_navit.accelerated.accelerated import BagelSiglipNavitAccelerated
from veomni.models.seed_omni.modules.bagel.siglip_navit.configuration import BagelSiglipNavitConfig
from veomni.models.seed_omni.modules.bagel.siglip_navit.modeling import BagelSiglipNavit
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import _OMNI_POSITION_IDS, _OMNI_TOKEN_LEN
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_CONTEXT_KEY, BAGEL_SIGLIP_CONTEXT
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.ops.mask import MagiAttentionMask
from veomni.utils.save_safetensor_utils import get_model_save_state


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _flex_config() -> BagelQwen2MoTConfig:
    return BagelQwen2MoTConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        ops_implementation={
            "attn_implementation": "veomni_flex_attention",
            "rms_norm_implementation": "eager",
            "swiglu_mlp_implementation": "eager",
            "rotary_pos_emb_implementation": "eager",
        },
    )


def _expand_magi_mask(mask: MagiAttentionMask, sequence_length: int) -> torch.Tensor:
    visible = torch.zeros(sequence_length, sequence_length, dtype=torch.bool, device=mask.q_ranges.device)
    attn_types = (
        mask.attn_type_map
        if mask.attn_type_map is not None
        else torch.zeros(mask.q_ranges.shape[0], dtype=torch.int32, device=mask.q_ranges.device)
    )
    for range_index in range(mask.q_ranges.shape[0]):
        q_start, q_end = int(mask.q_ranges[range_index, 0]), int(mask.q_ranges[range_index, 1])
        k_start, k_end = int(mask.k_ranges[range_index, 0]), int(mask.k_ranges[range_index, 1])
        if int(attn_types[range_index]) == 0:
            visible[q_start:q_end, k_start:k_end] = True
        else:
            q_idx = torch.arange(q_start, q_end, device=visible.device)
            k_idx = torch.arange(k_start, k_end, device=visible.device)
            visible[q_start:q_end, k_start:k_end] |= q_idx[:, None] >= k_idx[None, :]
    return visible


@pytest.mark.parametrize(
    "infer_graph",
    ["infer/graph_infer_und.yaml", "infer/graph_infer_gen.yaml", "infer/graph_infer_edit.yaml"],
)
def test_bagel_registry_and_yaml_graphs_load(infer_graph: str) -> None:
    """Verify all five BAGEL modules are registered and train + infer YAML configs resolve cleanly."""
    expected = {
        "bagel_text_encoder": ("BagelTextEncoder", "BagelTextEncoderAccelerated"),
        "bagel_siglip_navit": ("BagelSiglipNavit", "BagelSiglipNavitAccelerated"),
        "bagel_qwen2_mot": ("BagelQwen2MoT", "BagelQwen2MoTAccelerated"),
        "bagel_flow_connector": ("BagelFlowConnector", "BagelFlowConnectorAccelerated"),
        "bagel_vae": ("BagelVAE", "BagelVAEAccelerated"),
    }
    for model_type, (native_name, accel_name) in expected.items():
        assert model_type in OMNI_CONFIG_REGISTRY.valid_keys()
        assert OMNI_MODEL_REGISTRY[model_type]().__name__ == native_name
        assert OMNI_ACCELERATED_MODEL_REGISTRY[model_type]().__name__ == accel_name
    assert {"bagel_siglip_navit", "bagel_vae"}.issubset(set(OMNI_PROCESSOR_REGISTRY.valid_keys()))

    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "train/modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "train/graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / infer_graph,
    )
    assert set(cfg._module_entries) == set(expected)
    assert cfg.generation_graph["initial"] == "prompt_encode"
    assert any(
        t.get("next_state") == "done"
        for state in cfg.generation_graph["states"].values()
        for t in state.get("transitions", [])
    )
    runtime_modules = load_module_runtime_args(modules_path=bagel_cfg_dir() / "train/modules_train.yaml")
    assert runtime_modules["bagel_qwen2_mot"].ops_implementation.attn_implementation == "veomni_flex_attention"
    assert runtime_modules["bagel_qwen2_mot"].accelerator.enable_async is True


def test_mot_attention_masks_and_sp_padding() -> None:
    """Verify SDPA, Magi, and Flex masks agree on multi-sample causal/full/noise visibility and SP padding."""
    metadata = build_mot_attention_metadata(
        [[2, 3, 2], [4, 2]],
        [["full", "causal", "noise"], ["causal", "noise"]],
        device=torch.device("cpu"),
    )
    padded = pad_mot_attention_metadata(metadata, padded_length=16)
    sdpa = build_mot_sdpa_mask(padded)
    magi = build_mot_magi_mask(padded)
    visible = _expand_magi_mask(magi, 16)
    torch.testing.assert_close(visible[None, None], sdpa)
    assert not bool(visible[:13, 13:].any())
    assert bool(visible[13:, 13:].all())
    assert isinstance(
        BagelQwen2MoTAttentionAccelerated.build_attention_mask(padded, attn_implementation="veomni_flex_attention"),
        BlockMask,
    )
    assert isinstance(
        BagelQwen2MoTAttentionAccelerated.build_attention_mask(padded, attn_implementation="veomni_magi_attention"),
        MagiAttentionMask,
    )


def test_accelerated_qkv_preserves_hf_checkpoint_schema_and_loaders(tmp_path: Path) -> None:
    """Verify fused runtime QKV parameters keep fused keys in ``state_dict`` / DCP and split on HF safetensors export."""
    torch.manual_seed(3014)
    eager_cfg = BagelQwen2MoTConfig(**{**tiny_bagel_qwen2_cfg(), "num_key_value_heads": 1})
    eager_ref = BagelQwen2MoT(eager_cfg)
    eager_split_state = eager_ref.state_dict()

    reference = accelerated.BagelQwen2MoTAccelerated(_flex_config())
    reference.load_state_dict(eager_split_state)
    runtime_state = reference.state_dict()
    dcp_state = get_model_state_dict(reference)
    assert "model.layers.0.self_attn.qkv_proj_und.weight" in runtime_state
    assert "model.layers.0.self_attn.qkv_proj_gen.weight" in runtime_state
    assert "model.layers.0.self_attn.q_proj.weight" not in runtime_state
    assert set(dcp_state) == set(runtime_state)

    dcp_loaded = accelerated.BagelQwen2MoTAccelerated(_flex_config())
    set_model_state_dict(dcp_loaded, dcp_state)
    for name in ("qkv_proj_und", "qkv_proj_gen"):
        expected = getattr(reference.model.layers[0].self_attn, name)
        actual = getattr(dcp_loaded.model.layers[0].self_attn, name)
        torch.testing.assert_close(actual.weight, expected.weight)
        torch.testing.assert_close(actual.bias, expected.bias)

    reverted_hf_state = get_model_save_state(reference, None)
    for key in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.q_proj_moe_gen.weight",
    ):
        torch.testing.assert_close(reverted_hf_state[key], eager_split_state[key].to(torch.bfloat16))

    reference.save_pretrained(tmp_path)
    with safe_open(tmp_path / SAFE_WEIGHTS_NAME, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
    assert "model.layers.0.self_attn.q_proj.weight" in checkpoint_keys
    assert "model.layers.0.self_attn.qkv_proj_und.weight" not in checkpoint_keys

    hf_loaded = accelerated.BagelQwen2MoTAccelerated.from_pretrained(tmp_path, config=_flex_config())
    veomni_loaded = accelerated.BagelQwen2MoTAccelerated(_flex_config())
    load_model_weights(veomni_loaded, str(tmp_path), init_device="cpu")
    for name in ("qkv_proj_und", "qkv_proj_gen"):
        expected = getattr(reference.model.layers[0].self_attn, name)
        torch.testing.assert_close(getattr(hf_loaded.model.layers[0].self_attn, name).weight, expected.weight)
        torch.testing.assert_close(getattr(veomni_loaded.model.layers[0].self_attn, name).weight, expected.weight)


def test_bagel_modules_ops_dispatch_meta_rotary_and_flops(available_nvidia_ops) -> None:  # noqa: F811
    """Verify module-level ``VeomniOp`` binding, ``meta -> to_empty`` FP32 rotary table, and multi-image SigLIP FLOPs."""
    default_cfg = BagelQwen2MoTConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
    )
    assert default_cfg._attn_implementation == "veomni_flex_attention"

    with torch.device("meta"):
        meta_model = accelerated.BagelQwen2MoTAccelerated(_flex_config())
    materialized = meta_model.to_empty(device="cpu").to(dtype=torch.bfloat16)
    layer = materialized.model.layers[0]
    assert layer.self_attn.veomni_async_ulysses_qkv.variant == "bagel"
    assert layer.self_attn.veomni_async_ulysses_o.variant == "bagel"
    assert layer.self_attn.veomni_attn.impl == "veomni_flex_attention"
    assert layer.self_attn.veomni_denoise_attn.impl == "veomni_flash_attention_2"

    ref_rotary = BagelQwen2RotaryEmbedding(_flex_config())
    rotary = materialized.model.rotary_emb
    assert rotary.inv_freq.dtype == torch.float32
    assert rotary.original_inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, ref_rotary.inv_freq)

    # BagelFlowConnector meta -> to_empty -> init_weights preserves zero-initialized llm2vae velocity head
    flow_cfg = OMNI_CONFIG_REGISTRY["bagel_flow_connector"]()(
        hidden_size=16,
        z_channels=2,
        latent_patch_size=1,
        patch_latent_dim=2,
        max_latent_size=4,
        timestep_frequency_embedding_size=4,
    )
    eager_flow = OMNI_MODEL_REGISTRY["bagel_flow_connector"]()(flow_cfg)
    with torch.device("meta"):
        meta_flow = OMNI_MODEL_REGISTRY["bagel_flow_connector"]()(flow_cfg)
    materialized_flow = meta_flow.to_empty(device="cpu")
    materialized_flow.init_weights()
    assert torch.count_nonzero(materialized_flow.llm2vae.weight) == 0
    assert torch.count_nonzero(materialized_flow.llm2vae.bias) == 0
    assert torch.count_nonzero(materialized_flow.vae2llm.weight) > 0
    torch.testing.assert_close(materialized_flow.latent_pos_embed.pos_embed, eager_flow.latent_pos_embed.pos_embed)

    siglip_cfg = BagelSiglipNavitConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
        ops_implementation={"attn_implementation": "sdpa", "rotary_pos_emb_vision_implementation": "eager"},
    )
    siglip = BagelSiglipNavit(siglip_cfg).eval()
    assert siglip.vision_model.encoder.layers[0].self_attn.veomni_attn.impl == "veomni_sdpa"

    # Multi-image SigLIP FLOPs meter uses per-image lengths while keeping per-sample token counts
    acc_siglip = BagelSiglipNavitAccelerated(siglip_cfg)
    acc_siglip.config.num_hidden_layers = 26
    patch_dim = siglip_cfg.num_channels * siglip_cfg.patch_size * siglip_cfg.patch_size
    items = [
        ConversationItem(
            type="image",
            role="user",
            value=torch.zeros(1024, patch_dim, dtype=torch.bfloat16),
            meta={
                BAGEL_CONTEXT_KEY: BAGEL_SIGLIP_CONTEXT,
                _OMNI_TOKEN_LEN: 1024,
                _OMNI_POSITION_IDS: torch.zeros(1024, dtype=torch.long),
            },
        )
        for _ in range(2)
    ]
    acc_siglip.forward_pre(conversation_list=[items])
    acc_siglip.metric_meter_add("forward")
    flops, seqlens = acc_siglip.metric_meter_collect()
    assert seqlens == [2048]
    assert flops == pytest.approx(acc_siglip.estimate_flops([1024, 1024]))
    assert flops < acc_siglip.estimate_flops([2048])


def _two_rank_fsdp2_pos_embed_materialize_worker(rank: int, rendezvous: str) -> None:
    """Verify two-rank FSDP2 ``meta -> fully_shard -> to_empty -> init_weights`` materializes sharded ``PositionEmbedding``."""
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=2, rank=rank)
    try:
        ps = _init_parallel_state(dp_size=2, dp_shard_size=2, ulysses_size=1, dp_mode="fsdp2", device_type="cpu")

        flow_cfg = OMNI_CONFIG_REGISTRY["bagel_flow_connector"]()(
            hidden_size=16,
            z_channels=2,
            latent_patch_size=1,
            patch_latent_dim=2,
            max_latent_size=4,
            timestep_frequency_embedding_size=4,
        )
        eager_flow = OMNI_MODEL_REGISTRY["bagel_flow_connector"]()(flow_cfg)
        with torch.device("meta"):
            meta_flow = OMNI_MODEL_REGISTRY["bagel_flow_connector"]()(flow_cfg)
        fully_shard(meta_flow, mesh=ps.fsdp_mesh)
        meta_flow.to_empty(device="cpu")
        meta_flow.init_weights()

        assert isinstance(meta_flow.latent_pos_embed.pos_embed.data, DTensor)
        assert meta_flow.latent_pos_embed.pos_embed.to_local().shape == (8, 16)
        torch.testing.assert_close(
            meta_flow.latent_pos_embed.pos_embed.full_tensor(),
            eager_flow.latent_pos_embed.pos_embed,
        )
        assert torch.count_nonzero(meta_flow.llm2vae.weight.full_tensor()) == 0
        assert torch.count_nonzero(meta_flow.llm2vae.bias.full_tensor()) == 0
        assert torch.count_nonzero(meta_flow.vae2llm.weight.full_tensor()) > 0

        siglip_cfg = BagelSiglipNavitConfig(
            hidden_size=16,
            output_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
            vit_max_num_patch_per_side=4,
            ops_implementation={"attn_implementation": "eager", "rotary_pos_emb_vision_implementation": "eager"},
        )
        eager_siglip = BagelSiglipNavit(siglip_cfg)
        with torch.device("meta"):
            meta_siglip = BagelSiglipNavit(siglip_cfg)
        fully_shard(meta_siglip, mesh=ps.fsdp_mesh)
        meta_siglip.to_empty(device="cpu")
        meta_siglip.init_weights()

        assert isinstance(meta_siglip.vit_pos_embed.pos_embed.data, DTensor)
        assert meta_siglip.vit_pos_embed.pos_embed.to_local().shape == (8, 16)
        torch.testing.assert_close(
            meta_siglip.vit_pos_embed.pos_embed.full_tensor(),
            eager_siglip.vit_pos_embed.pos_embed,
        )
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_bagel_pos_embed_two_rank_fsdp2_meta_materialization(tmp_path: Path) -> None:
    """Verify two-rank FSDP2 config-only initialization distributes ``PositionEmbedding.pos_embed`` across shards."""
    import torch.multiprocessing as mp

    mp.spawn(
        _two_rank_fsdp2_pos_embed_materialize_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=True,
    )
