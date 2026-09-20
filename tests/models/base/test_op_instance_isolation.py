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

"""Forward isolation for instance-local attention and rope handles.

Handle snapshots in ``test_auto_registry.py`` stay, but they do not execute
``forward``. These cases construct two models under different ops configs,
poison the global config, then run text / vision / condition / DiT forwards.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import torch

from tests.models.compare import eager_ops_config, ops_config_scope, qwen_image_inputs, stamp_attn_implementation
from tests.models.tiny_configs import (
    tiny_deepseek_v3_config,
    tiny_deepseek_v4_config,
    tiny_flux_config,
    tiny_gemma3_text_config,
    tiny_glm_moe_dsa_config,
    tiny_ltx2_3_config,
    tiny_minimax_h3_config,
    tiny_qwen2_config,
    tiny_qwen2_vl_config,
    tiny_qwen3_moe_config,
    tiny_qwen_image_config,
    tiny_wan_config,
)
from veomni.ops import VeomniOp


def _output_tensor(result):
    if torch.is_tensor(result):
        return result
    if hasattr(result, "logits") and result.logits is not None:
        return result.logits
    if hasattr(result, "last_hidden_state"):
        return result.last_hidden_state
    if isinstance(result, (tuple, list)):
        return _output_tensor(result[0])
    raise TypeError(f"unsupported forward result {type(result)!r}")


def _handle(model, path: str) -> VeomniOp:
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    assert isinstance(module, VeomniOp), path
    return module


def _attn_cfg(impl: str) -> SimpleNamespace:
    cfg = eager_ops_config()
    cfg.attn_implementation = impl
    return cfg


def _poison_cfg() -> SimpleNamespace:
    cfg = eager_ops_config()
    cfg.attn_implementation = "flex_attention"
    cfg.rotary_pos_emb_implementation = "liger_kernel"
    cfg.rotary_pos_emb_vision_implementation = "npu"
    cfg.rms_norm_implementation = "liger_kernel"
    cfg.moe_implementation = "fused_triton"
    cfg.dsa_indexer_implementation = "cudnn"
    cfg.dsa_attention_implementation = "flashmla_cudnn"
    return cfg


def _assert_isolated(
    *,
    build: Callable[[], torch.nn.Module],
    run: Callable[[torch.nn.Module], object],
    attn_paths: Sequence[str],
    sticky_paths: Sequence[str] = (),
    eager_cfg: SimpleNamespace,
    alt_cfg: SimpleNamespace,
    poison_cfg: SimpleNamespace,
    hf_config: object | None = None,
) -> None:
    stamp_attn_implementation(hf_config, eager_cfg.attn_implementation)
    with ops_config_scope(eager_cfg):
        seed = build()
        state = {key: value.detach().clone() for key, value in seed.state_dict().items()}

    def construct(cfg: SimpleNamespace) -> torch.nn.Module:
        stamp_attn_implementation(hf_config, cfg.attn_implementation)
        with ops_config_scope(cfg):
            model = build()
        model.load_state_dict(state)
        model.eval()
        return model

    def check(model: torch.nn.Module, attn_impl: str) -> None:
        for path in attn_paths:
            assert _handle(model, path).impl == attn_impl, path
        for path in sticky_paths:
            assert _handle(model, path).impl == "eager", path

    eager = construct(eager_cfg)
    alternate = construct(alt_cfg)
    check(eager, eager_cfg.attn_implementation)
    check(alternate, alt_cfg.attn_implementation)

    with torch.no_grad():
        with ops_config_scope(eager_cfg):
            eager_ref = _output_tensor(run(eager))
        with ops_config_scope(alt_cfg):
            alt_ref = _output_tensor(run(alternate))
        with ops_config_scope(poison_cfg):
            eager_out = _output_tensor(run(eager))
            alt_out = _output_tensor(run(alternate))
            check(eager, eager_cfg.attn_implementation)
            check(alternate, alt_cfg.attn_implementation)

    torch.testing.assert_close(eager_out, eager_ref)
    torch.testing.assert_close(alt_out, alt_ref)


def test_qwen2_text_forward_keeps_construction_impls():
    from veomni.models.transformers.qwen2.generated.patched_modeling_qwen2_gpu import Qwen2ForCausalLM

    config = tiny_qwen2_config()
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    _assert_isolated(
        build=lambda: Qwen2ForCausalLM(config),
        run=lambda model: model(input_ids=input_ids, use_cache=False),
        attn_paths=("model.layers.0.self_attn.veomni_attn",),
        sticky_paths=("model.layers.0.self_attn.veomni_rope",),
        eager_cfg=_attn_cfg("eager"),
        alt_cfg=_attn_cfg("sdpa"),
        poison_cfg=_poison_cfg(),
        hf_config=config,
    )


def test_qwen2_vl_vision_forward_keeps_construction_impls():
    from veomni.models.transformers.qwen2_vl.generated.patched_modeling_qwen2_vl_gpu import (
        Qwen2VLForConditionalGeneration,
    )

    config = tiny_qwen2_vl_config()
    input_ids = torch.randint(3, 100, (2, 20))
    image = qwen_image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    image.pop("labels")
    _assert_isolated(
        build=lambda: Qwen2VLForConditionalGeneration(config),
        run=lambda model: model(input_ids=ids, use_cache=False, **image),
        attn_paths=(
            "model.language_model.layers.0.self_attn.veomni_attn",
            "model.visual.blocks.0.attn.veomni_attn",
        ),
        eager_cfg=_attn_cfg("eager"),
        alt_cfg=_attn_cfg("sdpa"),
        poison_cfg=_poison_cfg(),
        hf_config=config,
    )


def test_wan_condition_forward_keeps_construction_impls():
    from veomni.models.transformers.wan.modeling_wan import WanModel

    config = tiny_wan_config()
    inputs = {
        "x": torch.randn(2, config.in_dim, 2, 8, 8),
        "timestep": torch.rand(2),
        "context": torch.randn(2, config.text_len, config.text_dim),
    }
    _assert_isolated(
        build=lambda: WanModel(config),
        run=lambda model: model(**inputs),
        attn_paths=("blocks.0.self_attn.attn.veomni_attn",),
        sticky_paths=("blocks.0.self_attn.veomni_rope",),
        eager_cfg=_attn_cfg("eager"),
        alt_cfg=_attn_cfg("sdpa"),
        poison_cfg=_poison_cfg(),
    )


def test_flux_forward_keeps_construction_impls():
    from veomni.models.transformers.flux.modeling_flux import FluxModel

    # These families have no local eager_attention_forward. Isolate the two
    # CPU-runnable SDPA rows instead of eager vs sdpa.
    config = tiny_flux_config()
    inputs = {
        "hidden_states": torch.randn(2, 4, 4, 6),
        "timestep": torch.rand(2),
        "prompt_emb": torch.randn(2, 3, 48),
        "pooled_prompt_emb": torch.randn(2, 32),
        "guidance": torch.rand(2),
        "text_ids": torch.zeros(2, 3, 3),
    }
    _assert_isolated(
        build=lambda: FluxModel(config),
        run=lambda model: model(**inputs),
        attn_paths=("blocks.0.attn.veomni_attn", "single_blocks.0.veomni_attn"),
        eager_cfg=_attn_cfg("sdpa"),
        alt_cfg=_attn_cfg("veomni_sdpa"),
        poison_cfg=_poison_cfg(),
    )


def test_ltx_forward_keeps_construction_impls():
    from veomni.models.diffusers.ltx2_3.ltx_transformer.modeling_ltx2_3_transformer import LTXVideoTransformerModel

    config = tiny_ltx2_3_config()
    inputs = {
        "hidden_states": [torch.randn(1, 4, 1, 2, 2)],
        "timestep": [torch.tensor(0.5)],
        "encoder_hidden_states": [torch.randn(1, 3, 16)],
    }

    def build():
        model = LTXVideoTransformerModel(config)
        model.apply(model._init_weights)
        return model

    def run(model):
        call = {key: [value.detach().clone() for value in values] for key, values in inputs.items()}
        return model(**call).predictions[0]

    _assert_isolated(
        build=build,
        run=run,
        attn_paths=("transformer_blocks.0.attn1.attention_function.veomni_attn",),
        eager_cfg=_attn_cfg("sdpa"),
        alt_cfg=_attn_cfg("veomni_sdpa"),
        poison_cfg=_poison_cfg(),
    )


def test_minimax_h3_forward_keeps_construction_impls():
    from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    config = tiny_minimax_h3_config(latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2))
    video_rows = 4
    audio_rows = 4
    seq_len = video_rows + audio_rows + 1
    inputs = {
        "x": torch.randn(1, seq_len, 96),
        "audio_x": torch.randn(1, seq_len, 32),
        "img_position_ids": torch.arange(seq_len * 3).reshape(1, seq_len, 3),
        "unique_timesteps": torch.tensor([0.5]),
        "inverse_indices": torch.zeros(seq_len, dtype=torch.long),
        "update_mask": torch.arange(video_rows).remainder(2).float(),
        "token_tags": torch.tensor([0] * video_rows + [2] * audio_rows + [1]),
        "prompt_embeds": torch.randn(1, 16),
        "img_pos_info": {"position_ids": torch.arange(video_rows)},
        "audio_pos_info": {"position_ids": torch.arange(video_rows, video_rows + audio_rows)},
        "text_pos_info": {"position_ids": torch.tensor([seq_len - 1])},
        "img_pos_for_infer_output_info": {"position_ids": torch.arange(video_rows)},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, seq_len]), "max_seqlen_q": seq_len},
        "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 1]), "max_seqlen_q": 1},
    }
    _assert_isolated(
        build=lambda: MiniMaxH3DiTModel(config),
        run=lambda model: model(
            **inputs,
            cond_rows=0,
            video_latent_shape=(2, 1, 2),
            audio_latent_shape=(2, 2),
        ).predictions[0],
        attn_paths=("dit.blocks.0.attn.veomni_attn",),
        eager_cfg=_attn_cfg("sdpa"),
        alt_cfg=_attn_cfg("veomni_sdpa"),
        poison_cfg=_poison_cfg(),
    )


def test_qwen_image_forward_keeps_construction_impls():
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )

    config = tiny_qwen_image_config()
    inputs = {
        "hidden_states": torch.randn(1, 4, 16),
        "encoder_hidden_states": torch.randn(1, 3, 32),
        "encoder_hidden_states_mask": torch.ones(1, 3, dtype=torch.bool),
        "timestep": torch.tensor([0.25]),
        "img_shapes": [[(1, 2, 2)]],
    }
    _assert_isolated(
        build=lambda: QwenImageTransformer2DModel(config),
        run=lambda model: model(**inputs, return_dict=False)[0],
        attn_paths=("transformer_blocks.0.attn.processor.veomni_attn",),
        eager_cfg=_attn_cfg("sdpa"),
        alt_cfg=_attn_cfg("veomni_sdpa"),
        poison_cfg=_poison_cfg(),
    )


def test_qwen3_moe_text_forward_keeps_construction_impls():
    from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import Qwen3MoeForCausalLM

    config = tiny_qwen3_moe_config()
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    _assert_isolated(
        build=lambda: Qwen3MoeForCausalLM(config),
        run=lambda model: model(input_ids=input_ids, use_cache=False),
        attn_paths=("model.layers.0.self_attn.veomni_attn",),
        sticky_paths=(
            "model.layers.0.self_attn.veomni_rope",
            "model.layers.0.mlp.experts.veomni_moe",
        ),
        eager_cfg=_attn_cfg("eager"),
        alt_cfg=_attn_cfg("sdpa"),
        poison_cfg=_poison_cfg(),
        hf_config=config,
    )


def test_glm_moe_dsa_forward_keeps_construction_impls():
    from veomni.models.transformers.glm_moe_dsa.generated.patched_modeling_glm_moe_dsa_gpu import GlmMoeDsaForCausalLM

    config = tiny_glm_moe_dsa_config()
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    _assert_isolated(
        build=lambda: GlmMoeDsaForCausalLM(config),
        run=lambda model: model(input_ids=input_ids, use_cache=False),
        attn_paths=(),
        sticky_paths=(
            "model.layers.0.self_attn.veomni_dsa_attention",
            "model.layers.0.self_attn.indexer.veomni_dsa_indexer",
        ),
        eager_cfg=_attn_cfg("eager"),
        alt_cfg=_attn_cfg("sdpa"),
        poison_cfg=_poison_cfg(),
        hf_config=config,
    )


def test_gemma3_text_forward_keeps_construction_impls():
    from veomni.models.transformers.gemma3.generated.patched_modeling_gemma3_gpu import Gemma3ForCausalLM

    config = tiny_gemma3_text_config()
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    _assert_isolated(
        build=lambda: Gemma3ForCausalLM(config),
        run=lambda model: model(input_ids=input_ids, use_cache=False),
        attn_paths=("model.layers.0.self_attn.veomni_attn",),
        sticky_paths=("model.layers.0.self_attn.veomni_rope",),
        eager_cfg=_attn_cfg("eager"),
        alt_cfg=_attn_cfg("sdpa"),
        poison_cfg=_poison_cfg(),
        hf_config=config,
    )


def test_gemma3_eager_instance_keeps_first_token_causal_after_global_sdpa_switch():
    from veomni.models.transformers.gemma3.generated.patched_modeling_gemma3_gpu import Gemma3ForCausalLM

    config = tiny_gemma3_text_config()
    with ops_config_scope(_attn_cfg("eager")):
        model = Gemma3ForCausalLM(config).eval()
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    mutated = input_ids.clone()
    mutated[:, -1] = (mutated[:, -1] + 3) % config.vocab_size
    sdpa_cfg = _attn_cfg("sdpa")
    with torch.no_grad(), ops_config_scope(sdpa_cfg):
        logits = model(input_ids=input_ids, use_cache=False).logits
        mutated_logits = model(input_ids=mutated, use_cache=False).logits
    torch.testing.assert_close(logits[:, 0], mutated_logits[:, 0])


def test_deepseek_v3_rope_keeps_construction_impl():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.deepseek_v3.generated.patched_modeling_deepseek_v3_npu import (
            DeepseekV3ForCausalLM,
        )
    else:
        from veomni.models.transformers.deepseek_v3.generated.patched_modeling_deepseek_v3_gpu import (
            DeepseekV3ForCausalLM,
        )

    config = tiny_deepseek_v3_config()
    if hasattr(config, "rope_interleave"):
        config.rope_interleave = False
    eager_cfg = eager_ops_config()
    poison = eager_ops_config()
    poison.rotary_pos_emb_implementation = "triton"
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    with ops_config_scope(eager_cfg):
        model = DeepseekV3ForCausalLM(config).eval()
    assert model.model.layers[0].self_attn.veomni_rope.impl == "eager"
    assert model.model.rotary_emb.veomni_rope_use_triton is False
    with torch.no_grad(), ops_config_scope(eager_cfg):
        reference = model(input_ids=input_ids, use_cache=False).logits
    with torch.no_grad(), ops_config_scope(poison):
        isolated = model(input_ids=input_ids, use_cache=False).logits
        assert model.model.layers[0].self_attn.veomni_rope.impl == "eager"
        assert model.model.rotary_emb.veomni_rope_use_triton is False
    torch.testing.assert_close(isolated, reference)


def test_deepseek_v4_rope_keeps_construction_impl():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_npu import (
            DeepseekV4ForCausalLM,
        )
    else:
        from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
            DeepseekV4ForCausalLM,
        )

    config = tiny_deepseek_v4_config()
    eager_cfg = eager_ops_config()
    poison = eager_ops_config()
    poison.rotary_pos_emb_implementation = "triton"
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    with ops_config_scope(eager_cfg):
        model = DeepseekV4ForCausalLM(config).eval()
    assert model.model.layers[0].self_attn.veomni_rope.impl == "eager"
    with torch.no_grad(), ops_config_scope(eager_cfg):
        reference = model(input_ids=input_ids, use_cache=False).logits
    with torch.no_grad(), ops_config_scope(poison):
        isolated = model(input_ids=input_ids, use_cache=False).logits
        assert model.model.layers[0].self_attn.veomni_rope.impl == "eager"
    torch.testing.assert_close(isolated, reference)
