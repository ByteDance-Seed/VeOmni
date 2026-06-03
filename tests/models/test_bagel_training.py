# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""End-to-end smoke test for unified Bagel training (understanding + generation).

Builds a tiny composed ``SeedOmniModel`` (bagel foundation MoT + SigLIP NaViT encoder +
FLUX-VAE / rectified-flow decoder), packs a heterogeneous batch (one image->text
understanding sample + one text->image generation sample) with ``BagelPackedCollator``,
and runs forward / backward / optimizer-step, asserting both the cross-entropy and the
rectified-flow MSE losses are finite and that parameters update.
"""

import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


def _skip_if_no_flash_attn():
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for the bagel packed forward (flash-attn varlen).")
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"flash-attn is not available: {exc}")


def _build_tiny_bagel_model(hidden_size: int = 64):
    from veomni.models.seed_omni.configuration_seed_omni import SeedOmniConfig
    from veomni.models.seed_omni.modeling_seed_omni import SeedOmniModel

    foundation = dict(
        model_type="bagel_foundation",
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_module="Qwen2MoTDecoderLayer",
        qk_norm=True,
        pad_token_id=0,
    )
    encoder = dict(
        image_config=dict(
            model_type="bagel_vision_encoder",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            patch_size=16,
            image_size=224,
            rope=True,
            output_size=hidden_size,
            vit_max_num_patch_per_side=70,
        ),
        text_config=dict(**foundation),
        encode_input=True,
        encode_output=False,
    )
    decoder = dict(
        image_config=dict(
            model_type="bagel_vision_decoder",
            downsample=8,
            ch=32,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=1,
            z_channels=16,
            latent_patch_size=2,
            max_latent_size=32,
            output_size=hidden_size,
        ),
        encode_input=False,
        encode_output=True,
    )
    cfg = SeedOmniConfig(encoder_config=encoder, foundation_config=foundation, decoder_config=decoder)
    return SeedOmniModel(cfg)


def test_bagel_unified_training_step():
    _skip_if_no_flash_attn()

    from veomni.data.multimodal.bagel_collator import BagelPackedCollator

    torch.manual_seed(0)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        device = get_device_type()
        model = _build_tiny_bagel_model().to(device).train()

        collator = BagelPackedCollator(
            vit_patch_size=16,
            vit_max_num_patch_per_side=70,
            latent_patch_size=2,
            latent_downsample=16,
            max_latent_size=32,
        )

        def rand_ids(n):
            return torch.randint(5, 128, (n,)).tolist()

        sample_und = {
            "segments": [
                {"kind": "text", "ids": rand_ids(3), "loss": False},
                {"kind": "vit", "image": torch.randn(3, 32, 32)},
                {"kind": "text", "ids": rand_ids(4), "loss": True},
            ]
        }
        sample_gen = {
            "segments": [
                {"kind": "text", "ids": rand_ids(3), "loss": False},
                {"kind": "vae", "image": torch.randn(3, 32, 32), "grid": (2, 2)},
            ]
        }

        batch = collator([sample_und, sample_gen])
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        ref = next(p for p in model.foundation.parameters() if p.requires_grad).detach().clone()

        out = model(**batch)
        assert "foundation_ce_loss" in out.losses
        assert "image_decoder_mse_loss" in out.losses
        assert torch.isfinite(out.loss)

        out.loss.backward()
        opt.step()

        updated = next(p for p in model.foundation.parameters() if p.requires_grad)
        assert (ref - updated).abs().sum() > 0, "foundation parameters did not update"
    finally:
        torch.set_default_dtype(prev_dtype)
