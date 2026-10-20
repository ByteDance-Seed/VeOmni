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

"""Qwen2.5-Omni registry, processor, and Hugging Face parity tests.

Direct-import the generated Thinker class. Compare a toy Thinker against Hugging Face.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig as HFQwen2_5OmniConfig,
)
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration as HFQwen2_5OmniThinker,
)
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor as HFQwen2_5OmniProcessor,
)

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    ops_config_scope,
    qwen_image_inputs,
)
from tests.models.tiny_configs import tiny_qwen2_5_omni_thinker_config as _tiny_thinker_config


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _build_ours(config: Qwen2_5OmniThinkerConfig, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.qwen2_5_omni.generated.patched_modeling_qwen2_5_omni_gpu import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return Qwen2_5OmniThinkerForConditionalGeneration(config)


def _mask_kwargs(input_ids: torch.Tensor) -> dict:
    zeros = torch.zeros_like(input_ids, dtype=torch.bool)
    return {
        "image_mask": zeros,
        "video_mask": zeros,
        "audio_mask": zeros,
    }


def test_qwen2_5_omni_registry_wires_kernel_and_upstream_talker_entries():
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniTalkerForConditionalGeneration,
        Qwen2_5OmniTalkerModel,
    )

    from veomni.models.transformers.qwen2_5_omni import (
        register_qwen2_5_omni_modeling,
        register_qwen2_5_omni_text_modeling,
        register_qwen2_5_omni_thinker_modeling,
    )

    assert register_qwen2_5_omni_modeling(None).__name__ == "Qwen2_5OmniForConditionalGeneration"
    assert register_qwen2_5_omni_thinker_modeling(None).__name__ == "Qwen2_5OmniThinkerForConditionalGeneration"
    assert register_qwen2_5_omni_text_modeling(None).__name__ == "Qwen2_5OmniThinkerTextModel"
    assert (
        register_qwen2_5_omni_modeling("Qwen2_5OmniTalkerForConditionalGeneration")
        is Qwen2_5OmniTalkerForConditionalGeneration
    )
    assert register_qwen2_5_omni_modeling("Qwen2_5OmniTalkerModel") is Qwen2_5OmniTalkerModel


def test_qwen2_5_omni_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_thinker_config()
    hf = HFQwen2_5OmniThinker(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=input_ids,
        ours_fwd_kwargs=_mask_kwargs(input_ids),
        logits_equal=True,
    )


def test_qwen2_5_omni_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_thinker_config()
    hf = HFQwen2_5OmniThinker(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 20))
    image = qwen_image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    labels = image.pop("labels")
    ours_masks = {
        "image_mask": image.pop("image_mask"),
        "video_mask": image.pop("video_mask"),
        "audio_mask": image.pop("audio_mask"),
    }
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=ids,
        labels=labels,
        fwd_kwargs=image,
        ours_fwd_kwargs=ours_masks,
        logits_equal=True,
    )


def test_qwen2_5_omni_vision_and_audio_attention_veomni_sdpa_matches_hf(available_nvidia_ops):
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioAttention as HFQwen2_5OmniAudioAttention,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniVisionAttention as HFQwen2_5OmniVisionAttention,
    )

    from veomni.models.transformers.qwen2_5_omni.generated.patched_modeling_qwen2_5_omni_gpu import (
        Qwen2_5OmniAudioAttention,
        Qwen2_5OmniVisionAttention,
    )

    torch.manual_seed(0)
    thinker_config = _tiny_thinker_config()
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    vision_cfg_hf = thinker_config.vision_config
    vision_cfg_hf._attn_implementation = "sdpa"
    hf_vision_attn = HFQwen2_5OmniVisionAttention(vision_cfg_hf).eval()

    vision_cfg_ours = _tiny_thinker_config().vision_config
    vision_cfg_ours._attn_implementation = "veomni_sdpa"
    with ops_config_scope(eager_ops_config()):
        ours_vision_attn = Qwen2_5OmniVisionAttention(vision_cfg_ours).eval()
    ours_vision_attn.load_state_dict(hf_vision_attn.state_dict())

    vision_hidden = torch.randn(7, vision_cfg_hf.hidden_size)
    head_dim = vision_cfg_hf.hidden_size // vision_cfg_hf.num_heads
    pos_emb = torch.randn(7, head_dim // 2)
    torch.testing.assert_close(
        ours_vision_attn(vision_hidden, cu_seqlens=cu_seqlens, position_embeddings=pos_emb),
        hf_vision_attn(vision_hidden, cu_seqlens=cu_seqlens, position_embeddings=pos_emb),
        rtol=0,
        atol=0,
    )

    audio_cfg_hf = thinker_config.audio_config
    audio_cfg_hf._attn_implementation = "sdpa"
    hf_audio_attn = HFQwen2_5OmniAudioAttention(audio_cfg_hf).eval()

    audio_cfg_ours = _tiny_thinker_config().audio_config
    audio_cfg_ours._attn_implementation = "veomni_sdpa"
    with ops_config_scope(eager_ops_config()):
        ours_audio_attn = Qwen2_5OmniAudioAttention(audio_cfg_ours).eval()
    ours_audio_attn.load_state_dict(hf_audio_attn.state_dict())

    audio_hidden = torch.randn(7, audio_cfg_hf.d_model)
    torch.testing.assert_close(
        ours_audio_attn(audio_hidden, cu_seqlens=cu_seqlens),
        hf_audio_attn(audio_hidden, cu_seqlens=cu_seqlens),
        rtol=0,
        atol=0,
    )

    # Also verify raw ``flex_attention`` and ``veomni_sage_attention`` preserve segment isolation.
    from veomni.ops.kernels.attention.standard import sage as sage_backend

    def fake_sage(query, key, value, tensor_layout="HND", is_causal=False, sm_scale=None):
        scale = (query.shape[-1] ** -0.5) if sm_scale is None else sm_scale
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(weights, value)

    orig_sageattn = sage_backend.sageattn
    sage_backend.sageattn = fake_sage
    try:
        for impl in ("flex_attention", "veomni_sage_attention"):
            cfg = _tiny_thinker_config().audio_config
            cfg._attn_implementation = impl
            with ops_config_scope(eager_ops_config()):
                attn = Qwen2_5OmniAudioAttention(cfg).eval()
            attn.load_state_dict(hf_audio_attn.state_dict())
            h1 = torch.randn(4, cfg.d_model)
            h2 = h1.clone()
            h2[2:] += 10.0
            with torch.no_grad():
                # Single-segment ([0, 4]) and multi-segment ([0, 2, 4]) both succeed and isolate segments.
                single_out = attn(h1, cu_seqlens=torch.tensor([0, 4], dtype=torch.int32))
                assert single_out.shape == (4, cfg.d_model)
                out1 = attn(h1, cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32))
                out2 = attn(h2, cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32))
                torch.testing.assert_close(out1[:2], out2[:2], rtol=1e-5, atol=1e-5)
    finally:
        sage_backend.sageattn = orig_sageattn


class _StubTokenizer:
    init_kwargs: dict = {}
    image_token = "<|image|>"
    audio_token = "<|audio|>"
    video_token = "<|video|>"
    vision_bos_token = "<|vision_bos|>"
    vision_eos_token = "<|vision_eos|>"
    audio_bos_token = "<|audio_bos|>"
    audio_eos_token = "<|audio_eos|>"

    def __call__(self, text, **kwargs):
        return {"input_ids": [[1, 2, 3]]}


def _stub_qwen2_5_omni_processor():
    from veomni.models.transformers.qwen2_5_omni.processing_qwen2_5_omni import (
        Qwen2_5OmniProcessor,
    )

    processor = object.__new__(Qwen2_5OmniProcessor)
    processor.tokenizer = _StubTokenizer()
    processor.image_processor = SimpleNamespace(merge_size=2)
    processor.video_processor = SimpleNamespace(merge_size=2, temporal_patch_size=2)
    processor.feature_extractor = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("empty VeOmni lists must not reach the feature extractor")
    )
    processor.image_token = processor.tokenizer.image_token
    processor.audio_token = processor.tokenizer.audio_token
    processor.video_token = processor.tokenizer.video_token
    processor.vision_bos_token = processor.tokenizer.vision_bos_token
    processor.vision_eos_token = processor.tokenizer.vision_eos_token
    processor.audio_bos_token = processor.tokenizer.audio_bos_token
    processor.audio_eos_token = processor.tokenizer.audio_eos_token
    return processor


def test_qwen2_5_omni_config_forces_untied_embeddings():
    from veomni.models.transformers.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniConfig,
    )

    assert not hasattr(HFQwen2_5OmniConfig(), "tie_word_embeddings")
    assert getattr(HFQwen2_5OmniConfig(), "tie_word_embeddings", True) is True
    assert Qwen2_5OmniConfig().tie_word_embeddings is False
    assert Qwen2_5OmniConfig(tie_word_embeddings=True).tie_word_embeddings is False


def test_qwen2_5_omni_processor_accepts_veomni_empty_lists():
    from veomni.models.transformers.qwen2_5_omni.processing_qwen2_5_omni import (
        Qwen2_5OmniProcessor,
    )

    assert "audios" in inspect.signature(Qwen2_5OmniProcessor.__call__).parameters
    assert "audios" not in inspect.signature(HFQwen2_5OmniProcessor.__call__).parameters

    processor = _stub_qwen2_5_omni_processor()
    batch = processor(text="hello", images=[], videos=[], audios=[])
    assert batch["input_ids"] == [[1, 2, 3]]


def test_qwen2_5_omni_processor_interleaves_video_and_audio():
    processor = _stub_qwen2_5_omni_processor()
    text = [f"{processor.vision_bos_token}{processor.video_token}{processor.vision_eos_token}"]
    grid = torch.tensor([2, 4, 4])
    processed = processor.replace_multimodal_special_tokens(
        text,
        iter([4]),
        iter([]),
        iter([grid]),
        video_second_per_grid=iter([1.0]),
        position_id_per_seconds=1,
        seconds_per_chunk=2,
    )
    sample = processed[0]
    assert processor.video_token in sample
    assert processor.audio_token in sample
    assert sample.count(processor.video_token) == int(grid.prod()) // (processor.video_processor.merge_size**2)
    assert sample.count(processor.audio_token) == 4
