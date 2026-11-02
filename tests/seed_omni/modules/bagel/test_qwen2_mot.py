"""BAGEL Qwen2-MoT attention dispatch and QKV checkpoint schema."""

from __future__ import annotations

import pytest
import torch
from safetensors import safe_open
from transformers.utils import SAFE_WEIGHTS_NAME

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.conftest import available_nvidia_ops  # noqa: F401
from tests.seed_omni.modules.bagel.helpers import config_cls, tiny_bagel_qwen2_cfg
from veomni.models.checkpoint.weights import load_model_weights
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated import accelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.checkpoint_conversion import (
    BagelQwen2MoTCheckpointTensorConverter,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.configuration import BagelQwen2MoTConfig
from veomni.models.seed_omni.modules.bagel.qwen2_mot.masking import (
    build_mot_attention_metadata,
    build_mot_block_mask,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import _sdpa_packed_attention


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


def test_cached_sdpa_causal_attention_is_bottom_right_aligned() -> None:
    query = torch.zeros(1, 1, 1)
    key = torch.zeros(4, 1, 1)
    value = torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(4, 1, 1)

    output = _sdpa_packed_attention(
        query,
        key,
        value,
        is_causal=True,
        cu_seq_lens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 4], dtype=torch.int32),
        scale=1.0,
        enable_gqa=False,
    )

    # A single cached decode query is the final logical row, so it can attend
    # to all four historical keys. Upper-left causal alignment would return 1.
    torch.testing.assert_close(output, torch.tensor([[[3.75]]]))


def test_accelerated_training_attention_rejects_non_flex_backend() -> None:
    """Verify ``BagelQwen2MoTAttentionAccelerated.forward_packed_train`` rejects non-packed backends."""
    config = config_cls("bagel_qwen2_mot")(**tiny_bagel_qwen2_cfg(), attn_implementation="sdpa")
    attention = accelerated.BagelQwen2MoTAttentionAccelerated(config, layer_idx=0)
    sequence_length = 2
    metadata = build_mot_attention_metadata([[2]], [["causal"]], device=torch.device("cpu"))

    with pytest.raises(ValueError, match="requires packed fused attention"):
        attention.forward_packed_train(
            packed_sequence=torch.randn(sequence_length, config.hidden_size),
            attention_mask=build_mot_block_mask(metadata),
            packed_position_cos=torch.ones(sequence_length, attention.head_dim),
            packed_position_sin=torch.zeros(sequence_length, attention.head_dim),
            packed_und_token_indexes=torch.arange(sequence_length),
            packed_gen_token_indexes=torch.empty(0, dtype=torch.long),
        )


def test_native_gqa_rejects_invalid_global_head_ratio_at_config_init() -> None:
    """Verify ``BagelQwen2MoTConfig`` rejects query head counts not divisible by KV head counts."""
    with pytest.raises(ValueError, match="query heads must be divisible"):
        config_cls("bagel_qwen2_mot")(
            **{
                **tiny_bagel_qwen2_cfg(),
                "num_attention_heads": 6,
                "num_key_value_heads": 4,
            }
        )


def test_checkpoint_tensor_converter_combines_legacy_qkv_groups() -> None:
    """Verify ``BagelQwen2MoTCheckpointTensorConverter`` merges split Q/K/V weights and biases."""
    converter = BagelQwen2MoTCheckpointTensorConverter()
    prefix = "model.layers.3.self_attn."
    tensors = {
        f"{prefix}q_proj.weight": torch.full((64, 64), 1.0),
        f"{prefix}k_proj.weight": torch.full((16, 64), 2.0),
        f"{prefix}v_proj.weight": torch.full((16, 64), 3.0),
        f"{prefix}q_proj_moe_gen.bias": torch.full((64,), 4.0),
        f"{prefix}k_proj_moe_gen.bias": torch.full((16,), 5.0),
        f"{prefix}v_proj_moe_gen.bias": torch.full((16,), 6.0),
    }

    converted = []
    for name in reversed(tensors):
        result = converter.convert(name, tensors[name])
        if result is not None:
            converted.append(result)
    assert converter.finalize() == []
    assert {result.name for result in converted} == {
        f"{prefix}qkv_proj_und.weight",
        f"{prefix}qkv_proj_gen.bias",
    }

    by_name = {result.name: result.tensor for result in converted}
    und_weight = by_name[f"{prefix}qkv_proj_und.weight"]
    gen_bias = by_name[f"{prefix}qkv_proj_gen.bias"]
    torch.testing.assert_close(und_weight[:64], tensors[f"{prefix}q_proj.weight"])
    torch.testing.assert_close(und_weight[64:80], tensors[f"{prefix}k_proj.weight"])
    torch.testing.assert_close(und_weight[80:], tensors[f"{prefix}v_proj.weight"])
    torch.testing.assert_close(gen_bias[:64], tensors[f"{prefix}q_proj_moe_gen.bias"])
    torch.testing.assert_close(gen_bias[64:80], tensors[f"{prefix}k_proj_moe_gen.bias"])
    torch.testing.assert_close(gen_bias[80:], tensors[f"{prefix}v_proj_moe_gen.bias"])


def test_accelerated_qkv_preserves_hf_checkpoint_schema_and_loaders(tmp_path) -> None:
    """Verify fused runtime QKV parameters keep fused keys in ``state_dict`` / DCP and split on HF safetensors export."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

    from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT
    from veomni.utils.save_safetensor_utils import _revert_model_weight_conversions_for_hf

    torch.manual_seed(3014)
    eager_cfg = BagelQwen2MoTConfig(**{**tiny_bagel_qwen2_cfg(), "num_key_value_heads": 1})
    eager_ref = BagelQwen2MoT(eager_cfg)
    eager_split_state = eager_ref.state_dict()
    assert "model.layers.0.self_attn.q_proj.weight" in eager_split_state
    assert "model.layers.0.self_attn.q_proj_moe_gen.weight" in eager_split_state

    reference = accelerated.BagelQwen2MoTAccelerated(_flex_config())
    reference.load_state_dict(eager_split_state)
    internal_parameter_names = set(dict(reference.named_parameters()))
    assert "model.layers.0.self_attn.qkv_proj_und.weight" in internal_parameter_names
    assert "model.layers.0.self_attn.qkv_proj_gen.weight" in internal_parameter_names
    assert "model.layers.0.self_attn.q_proj.weight" not in internal_parameter_names
    assert "model.layers.0.self_attn.q_proj_moe_gen.weight" not in internal_parameter_names

    # PyTorch state_dict() and DCP get_model_state_dict() keep fused parameter keys so
    # FQN resolution works under DDP as well as FSDP2.
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

    # HF export reverts fused QKV back to split q_proj/k_proj/v_proj keys.
    reverted_hf_state = _revert_model_weight_conversions_for_hf(reference, dict(dcp_state))
    assert "model.layers.0.self_attn.q_proj.weight" in reverted_hf_state
    assert "model.layers.0.self_attn.k_proj.weight" in reverted_hf_state
    assert "model.layers.0.self_attn.v_proj.weight" in reverted_hf_state
    assert "model.layers.0.self_attn.q_proj_moe_gen.weight" in reverted_hf_state
    assert "model.layers.0.self_attn.qkv_proj_und.weight" not in reverted_hf_state
    assert "model.layers.0.self_attn.qkv_proj_gen.weight" not in reverted_hf_state

    reference.save_pretrained(tmp_path)
    with safe_open(tmp_path / SAFE_WEIGHTS_NAME, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
    assert "model.layers.0.self_attn.q_proj.weight" in checkpoint_keys
    assert "model.layers.0.self_attn.q_proj_moe_gen.weight" in checkpoint_keys
    assert "model.layers.0.self_attn.qkv_proj_und.weight" not in checkpoint_keys
    assert "model.layers.0.self_attn.qkv_proj_gen.weight" not in checkpoint_keys

    hf_loaded = accelerated.BagelQwen2MoTAccelerated.from_pretrained(tmp_path, config=_flex_config())
    veomni_loaded = accelerated.BagelQwen2MoTAccelerated(_flex_config())
    load_model_weights(veomni_loaded, str(tmp_path), init_device="cpu")
    for name in ("qkv_proj_und", "qkv_proj_gen"):
        expected = getattr(reference.model.layers[0].self_attn, name)
        torch.testing.assert_close(getattr(hf_loaded.model.layers[0].self_attn, name).weight, expected.weight)
        torch.testing.assert_close(getattr(hf_loaded.model.layers[0].self_attn, name).bias, expected.bias)
        torch.testing.assert_close(getattr(veomni_loaded.model.layers[0].self_attn, name).weight, expected.weight)
        torch.testing.assert_close(getattr(veomni_loaded.model.layers[0].self_attn, name).bias, expected.bias)


def test_bagel_qwen2_mot_selects_module_level_ops_without_global_ops_config(available_nvidia_ops) -> None:  # noqa: F811
    """Verify ``BagelQwen2MoTAccelerated`` binds ``VeomniOp`` handles from ``config.ops_config``."""
    cfg = BagelQwen2MoTConfig(
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
            "attn_implementation": "flex_attention",
            "rms_norm_implementation": "eager",
            "swiglu_mlp_implementation": "eager",
            "rotary_pos_emb_implementation": "eager",
        },
    )
    model = accelerated.BagelQwen2MoTAccelerated(cfg)

    layer = model.model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer.mlp.veomni_swiglu_mlp.impl == "eager"
    assert layer.self_attn.veomni_rms_norm.impl == "eager"
    assert layer.self_attn.veomni_rope.impl == "eager"
    assert layer.self_attn.veomni_async_ulysses_qkv.variant == "bagel"
    assert layer.self_attn.veomni_async_ulysses_o.variant == "bagel"
    assert layer.self_attn.veomni_attn is not None
    assert layer.self_attn.veomni_attn.impl == "veomni_flex_attention"
    assert layer.self_attn.veomni_denoise_attn is not None
    assert layer.self_attn.veomni_denoise_attn.impl == "veomni_flash_attention_2"


def test_bagel_siglip_navit_selects_module_level_ops_and_forwards() -> None:
    """Verify ``BagelSiglipNavit`` binds vision RoPE and attention ops from ``config.ops_config``."""
    from veomni.models.seed_omni.modules.bagel.siglip_navit.configuration import BagelSiglipNavitConfig
    from veomni.models.seed_omni.modules.bagel.siglip_navit.modeling import BagelSiglipNavit

    cfg = BagelSiglipNavitConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
        ops_implementation={
            "attn_implementation": "sdpa",
            "rotary_pos_emb_vision_implementation": "eager",
        },
    )
    model = BagelSiglipNavit(cfg).eval()
    attn = model.vision_model.encoder.layers[0].self_attn
    assert attn.veomni_attn.impl == "veomni_sdpa"
    assert attn.veomni_rope.impl == "eager"

    packed_patches = torch.randn(4, 3 * 14 * 14)
    position_ids = torch.arange(4, dtype=torch.long)
    for cu_seqlens, max_seqlen in (
        (torch.tensor([0, 4], dtype=torch.int32), 4),
        (torch.tensor([0, 2, 4], dtype=torch.int32), 2),
    ):
        out = model.vision_model(
            patchified_pixel_values=packed_patches,
            patchified_position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()


class _CallCounter:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    @property
    def impl(self):
        return self.inner.impl

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.inner(*args, **kwargs)


@pytest.mark.parametrize("hidden_act", ["silu", "swish", "gelu"])
def test_bagel_qwen2_mot_mlp_and_rms_norm_dispatch_through_veomni_ops(hidden_act: str) -> None:
    """Verify ``Qwen2RMSNormAccelerated`` and ``Qwen2MLPAccelerated`` route through their ``VeomniOp`` handles."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as HFQwen2MLP
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as HFQwen2RMSNorm

    from tests.models.compare import assert_module_forward_and_grads_match

    cfg = BagelQwen2MoTConfig(
        **{
            **tiny_bagel_qwen2_cfg(),
            "hidden_act": hidden_act,
            "ops_implementation": {
                "attn_implementation": "sdpa",
                "rms_norm_implementation": "eager",
                "swiglu_mlp_implementation": "eager",
                "rotary_pos_emb_implementation": "eager",
            },
        }
    )

    torch.manual_seed(0)
    hf_norm = HFQwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    ours_norm = accelerated.Qwen2RMSNormAccelerated(cfg.hidden_size, eps=cfg.rms_norm_eps, config=cfg)
    ours_norm.load_state_dict(hf_norm.state_dict())
    norm_counter = _CallCounter(ours_norm.veomni_rms_norm)
    ours_norm.veomni_rms_norm = norm_counter
    assert_module_forward_and_grads_match(hf_norm, ours_norm, torch.randn(3, cfg.hidden_size))
    assert norm_counter.calls == 1
    empty_norm_out = ours_norm(torch.empty(0, cfg.hidden_size, requires_grad=True))
    assert empty_norm_out.shape == (0, cfg.hidden_size)
    assert norm_counter.calls == 2

    hf_mlp = HFQwen2MLP(cfg)
    ours_mlp = accelerated.Qwen2MLPAccelerated(cfg)
    ours_mlp.load_state_dict(hf_mlp.state_dict())
    mlp_counter = _CallCounter(ours_mlp.veomni_swiglu_mlp)
    ours_mlp.veomni_swiglu_mlp = mlp_counter
    assert_module_forward_and_grads_match(hf_mlp, ours_mlp, torch.randn(3, cfg.hidden_size))
    if hidden_act in {"silu", "swish"}:
        assert mlp_counter.calls == 1
        empty_mlp_out = ours_mlp(torch.empty(0, cfg.hidden_size, requires_grad=True))
        assert empty_mlp_out.shape == (0, cfg.hidden_size)
        assert mlp_counter.calls == 2
    else:
        assert mlp_counter.calls == 0


def test_bagel_modules_keep_construction_ops_when_config_mutated() -> None:
    """Verify ``BagelQwen2MoTAccelerated``, ``BagelSiglipNavit``, and ``BagelTextEncoderAccelerated`` keep construction-time ops after ``config.ops_implementation`` is mutated."""
    from veomni.models.seed_omni.modules.bagel.siglip_navit.configuration import BagelSiglipNavitConfig
    from veomni.models.seed_omni.modules.bagel.siglip_navit.modeling import BagelSiglipNavit
    from veomni.models.seed_omni.modules.bagel.text_encoder.accelerated.accelerated import BagelTextEncoderAccelerated
    from veomni.models.seed_omni.modules.bagel.text_encoder.configuration import BagelTextEncoderConfig

    poison_ops = {
        "attn_implementation": "sdpa",
        "rms_norm_implementation": "liger_kernel",
        "swiglu_mlp_implementation": "liger_kernel",
        "rotary_pos_emb_implementation": "liger_kernel",
        "rotary_pos_emb_vision_implementation": "liger_kernel",
        "cross_entropy_loss_implementation": "liger_kernel",
    }

    # 1. BagelSiglipNavit forward isolation
    siglip_cfg = BagelSiglipNavitConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
        rope=True,
        ops_implementation={
            "attn_implementation": "eager",
            "rotary_pos_emb_vision_implementation": "eager",
        },
    )
    siglip = BagelSiglipNavit(siglip_cfg).eval()
    siglip_attn = siglip.vision_model.encoder.layers[0].self_attn
    assert siglip_attn.veomni_attn.impl == "eager"
    assert siglip_attn.veomni_rope.impl == "eager"

    patches = torch.randn(4, 3 * 14 * 14)
    pos_ids = torch.arange(4, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    with torch.no_grad():
        siglip_ref = siglip.vision_model(patches, pos_ids, cu_seqlens=cu_seqlens, max_seqlen=2)
        siglip_cfg.ops_implementation = dict(poison_ops)
        assert siglip_attn.veomni_attn.impl == "eager"
        assert siglip_attn.veomni_rope.impl == "eager"
        siglip_out = siglip.vision_model(patches, pos_ids, cu_seqlens=cu_seqlens, max_seqlen=2)
    torch.testing.assert_close(siglip_out, siglip_ref)

    # 2. BagelTextEncoderAccelerated decode loss isolation
    text_cfg = BagelTextEncoderConfig(
        vocab_size=32,
        hidden_size=16,
        ops_implementation={"cross_entropy_loss_implementation": "eager"},
    )
    text_enc = BagelTextEncoderAccelerated(text_cfg).eval()
    assert text_enc.veomni_ce.impl == "eager"
    hidden = torch.randn(4, 16)
    shift_labels = torch.tensor([1, 2, 3, -100], dtype=torch.long)
    with torch.no_grad():
        loss_ref = text_enc.decode(hidden_states=hidden, shift_labels=shift_labels)["loss"]
        text_cfg.ops_implementation = dict(poison_ops)
        assert text_enc.veomni_ce.impl == "eager"
        loss_out = text_enc.decode(hidden_states=hidden, shift_labels=shift_labels)["loss"]
    torch.testing.assert_close(loss_out, loss_ref)

    # 3. BagelQwen2MoTAccelerated RoPE / RMSNorm / SwiGLU / Attention isolation
    mot_cfg = _flex_config()
    mot = accelerated.BagelQwen2MoTAccelerated(mot_cfg).eval()
    layer = mot.model.layers[0]
    rope_counter = _CallCounter(layer.self_attn.veomni_rope)
    layer.self_attn.veomni_rope = rope_counter
    q = torch.randn(3, layer.self_attn.num_heads, layer.self_attn.head_dim)
    k = torch.randn(3, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim)
    cos = torch.ones(3, layer.self_attn.head_dim)
    sin = torch.zeros(3, layer.self_attn.head_dim)
    with torch.no_grad():
        q_ref, k_ref = layer.self_attn.apply_rotary_pos_emb(q, k, cos, sin)
        mlp_ref = layer.mlp(torch.randn(3, mot_cfg.hidden_size))
        mot_cfg.ops_implementation = dict(poison_ops)
        assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
        assert layer.mlp.veomni_swiglu_mlp.impl == "eager"
        assert layer.self_attn.veomni_rms_norm.impl == "eager"
        assert layer.self_attn.veomni_rope.impl == "eager"
        assert layer.self_attn.veomni_attn_impl == "veomni_flex_attention"
        assert mot.model._bound_attn_implementation() == "veomni_flex_attention"
        q_out, k_out = layer.self_attn.apply_rotary_pos_emb(q, k, cos, sin)
    torch.testing.assert_close(q_out, q_ref)
    torch.testing.assert_close(k_out, k_ref)
    assert rope_counter.calls == 2
    assert torch.isfinite(mlp_ref).all()


@pytest.mark.parametrize("target_proj", ("qkv_proj_und", "qkv_proj_gen", "o_proj", "o_proj_moe_gen"))
def test_bagel_qwen2_mot_async_ulysses_falls_back_to_sync_on_lora_projections(
    monkeypatch: pytest.MonkeyPatch,
    target_proj: str,
) -> None:
    """Verify ``BagelQwen2MoTAttentionAccelerated.forward_packed_train`` falls back to synchronous projections when LoRA wraps any projection layer."""
    from types import SimpleNamespace

    from veomni.lora.layers import LoraLinear

    cfg = _flex_config()
    attn = accelerated.BagelQwen2MoTAttentionAccelerated(cfg, layer_idx=0).eval()
    base_linear = getattr(attn, target_proj)
    lora_linear = LoraLinear(base_linear, adapter_name="default", r=2, lora_alpha=4)
    with torch.no_grad():
        lora_linear.lora_B["default"].weight.fill_(0.25)
    setattr(attn, target_proj, lora_linear)

    def _fake_attn_op(module, q, k, v, mask, **kwargs):
        del module, k, mask, kwargs
        v_expanded = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        return (q + v_expanded).transpose(1, 2).contiguous(), None

    attn.veomni_attn = _fake_attn_op
    async_qkv_counter = _CallCounter(lambda **kwargs: (_ for _ in ()).throw(AssertionError("async qkv called")))
    async_o_counter = _CallCounter(lambda **kwargs: (_ for _ in ()).throw(AssertionError("async o called")))
    attn.veomni_async_ulysses_qkv = async_qkv_counter
    attn.veomni_async_ulysses_o = async_o_counter

    packed_sequence = torch.randn(4, cfg.hidden_size)
    cos = torch.ones(4, attn.head_dim)
    sin = torch.zeros(4, attn.head_dim)
    und_idx = torch.tensor([0, 2], dtype=torch.long)
    gen_idx = torch.tensor([1, 3], dtype=torch.long)

    monkeypatch.setattr(
        accelerated,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=False, async_enabled=False, ulysses_size=1, ulysses_group=None),
    )
    with torch.no_grad():
        sync_ref = attn.forward_packed_train(packed_sequence, None, cos, sin, und_idx, gen_idx)

    monkeypatch.setattr(
        accelerated,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=True, async_enabled=True, ulysses_size=2, ulysses_group=None),
    )
    with torch.no_grad():
        async_with_lora = attn.forward_packed_train(packed_sequence, None, cos, sin, und_idx, gen_idx)

    assert async_qkv_counter.calls == 0
    assert async_o_counter.calls == 0
    torch.testing.assert_close(async_with_lora, sync_ref)


def test_bagel_siglip_navit_multi_image_sample_flops_and_seqlens() -> None:
    """Verify multi-image samples preserve per-sample token counts while using per-image lengths for attention FLOPs."""
    from veomni.models.seed_omni.modules.bagel.siglip_navit.accelerated.accelerated import BagelSiglipNavitAccelerated
    from veomni.models.seed_omni.modules.bagel.siglip_navit.configuration import BagelSiglipNavitConfig
    from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import _OMNI_POSITION_IDS, _OMNI_TOKEN_LEN
    from veomni.models.seed_omni.modules.bagel.sources import BAGEL_CONTEXT_KEY, BAGEL_SIGLIP_CONTEXT
    from veomni.models.seed_omni.utils.conversation import ConversationItem

    cfg = BagelSiglipNavitConfig(
        num_hidden_layers=1,
        ops_implementation={
            "attn_implementation": "eager",
            "rotary_pos_emb_vision_implementation": "eager",
        },
    )
    model = BagelSiglipNavitAccelerated(cfg)
    model.config.num_hidden_layers = 26

    patch_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size
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
    model.forward_pre(conversation_list=[items])
    model.metric_meter_add("forward")
    flops, seqlens = model.metric_meter_collect()

    assert seqlens == [2048]
    expected_per_image_flops = model.estimate_flops([1024, 1024])
    overcounted_sample_flops = model.estimate_flops([2048])
    assert flops == pytest.approx(expected_per_image_flops)
    assert flops == pytest.approx(5.834821, rel=1e-4)
    assert overcounted_sample_flops == pytest.approx(6.588592, rel=1e-4)
    assert flops < overcounted_sample_flops


def test_bagel_flow_connector_and_siglip_navit_support_lora_wrapped_projections() -> None:
    """Verify ``BagelFlowConnector`` and ``BagelSiglipNavit`` run with ``LoraLinear``-wrapped projections."""
    from veomni.lora.layers import LoraLinear
    from veomni.models.seed_omni.modules.bagel.flow_connector.configuration import BagelFlowConnectorConfig
    from veomni.models.seed_omni.modules.bagel.flow_connector.modeling import BagelFlowConnector
    from veomni.models.seed_omni.modules.bagel.siglip_navit.configuration import BagelSiglipNavitConfig
    from veomni.models.seed_omni.modules.bagel.siglip_navit.modeling import BagelSiglipNavit

    flow_cfg = BagelFlowConnectorConfig(
        hidden_size=16,
        z_channels=4,
        latent_patch_size=2,
        max_latent_size=8,
        timestep_frequency_embedding_size=8,
    )
    flow = BagelFlowConnector(flow_cfg).eval()
    flow.vae2llm = LoraLinear(flow.vae2llm, adapter_name="default", r=2, lora_alpha=4)
    flow.llm2vae = LoraLinear(flow.llm2vae, adapter_name="default", r=2, lora_alpha=4)
    flow.time_embedder.mlp[0] = LoraLinear(flow.time_embedder.mlp[0], adapter_name="default", r=2, lora_alpha=4)
    flow.time_embedder.mlp[2] = LoraLinear(flow.time_embedder.mlp[2], adapter_name="default", r=2, lora_alpha=4)
    with torch.no_grad():
        flow.vae2llm.lora_B["default"].weight.fill_(0.1)
        flow.llm2vae.lora_B["default"].weight.fill_(0.1)
        latent_out = flow.embed_latent(
            latents=torch.randn(3, flow_cfg.patch_latent_dim),
            position_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            timesteps=torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32),
        )["latent_embeds"]
        vel_out = flow.decode_velocity(torch.randn(3, flow_cfg.hidden_size))["velocity"]
    assert latent_out.shape == (3, flow_cfg.hidden_size)
    assert vel_out.shape == (3, flow_cfg.patch_latent_dim)
    assert torch.isfinite(latent_out).all()
    assert torch.isfinite(vel_out).all()

    siglip_cfg = BagelSiglipNavitConfig(
        hidden_size=16,
        output_size=24,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
        rope=False,
        ops_implementation={
            "attn_implementation": "eager",
            "rotary_pos_emb_vision_implementation": "eager",
        },
    )
    siglip = BagelSiglipNavit(siglip_cfg).eval()
    siglip.vision_model.embeddings.patch_embedding = LoraLinear(
        siglip.vision_model.embeddings.patch_embedding, adapter_name="default", r=2, lora_alpha=4
    )
    siglip.connector.fc1 = LoraLinear(siglip.connector.fc1, adapter_name="default", r=2, lora_alpha=4)
    siglip.connector.fc2 = LoraLinear(siglip.connector.fc2, adapter_name="default", r=2, lora_alpha=4)
    patch_dim = siglip_cfg.num_channels * siglip_cfg.patch_size * siglip_cfg.patch_size
    with torch.no_grad():
        siglip_out = siglip(
            patchified_pixel_values=torch.randn(4, patch_dim),
            patchified_position_ids=torch.arange(4, dtype=torch.long),
            cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
            max_seqlen=2,
            token_lens=torch.tensor([2, 2], dtype=torch.int32),
        )
    assert siglip_out["image_embeds"].shape == (4, siglip_cfg.output_size)
    assert torch.isfinite(siglip_out["image_embeds"]).all()
