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
"""Tests for the vendored DiffusionGemma modeling and its training wiring.

The properties locked down here are the ones that are silent when broken: weight tying
across the encoder/decoder pair, the checkpoint key layout, gradient flow through the
encoder KV cache, and the fact that gradient checkpointing must not drop that cache.
"""

import json
import os
import warnings

import pytest
import torch
from safetensors.torch import save_file
from transformers.cache_utils import DynamicCache

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.data.data_collator import DiffusionGemmaCollator
from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY
from veomni.distributed import parallel_state
from veomni.models.transformers.diffusion_gemma import align_nemo, generation
from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
from veomni.models.transformers.diffusion_gemma.generation import (
    EntropyBoundedSamplerConfig,
    acceptance_mask,
    entropy_bounded_denoise,
)
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaDecoderTextLayer,
    DiffusionGemmaEncoderTextLayer,
    DiffusionGemmaForBlockDiffusion,
    SealableCache,
    chunked_self_conditioning_embeddings,
    chunked_weighted_diffusion_loss,
    expand_self_conditioning_mask,
)
from veomni.models.transformers.diffusion_gemma.parallel_plan import get_parallel_plan, retie_encoder_to_decoder
from veomni.utils.constants import IGNORE_INDEX


TOY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "toy_config", "diffusion_gemma_toy")


def build_toy_model(dtype=torch.float32, seed=0, attn_implementation="sdpa"):
    config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
    config._attn_implementation = attn_implementation
    torch.manual_seed(seed)
    return DiffusionGemmaForBlockDiffusion(config).to(dtype)


def toy_runtime(model, **assets):
    """Wrap a built model in the runtime handle a trainer reads it through, skipping the build.

    Bound to whatever parallel state is current, so no mesh has to be registered by name.
    """
    from veomni.distributed.clip_grad_norm import veomni_compute_grad_norm
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaModelRuntime

    class _ToyRuntime(DiffusionGemmaModelRuntime):
        @property
        def parallel_state(self):
            return get_parallel_state()

        def compute_grad_norm(self, norm_type=2.0):
            return veomni_compute_grad_norm(self.model, norm_type)

    runtime = _ToyRuntime.__new__(_ToyRuntime)
    runtime.model_name = "base"
    runtime.model = model
    runtime.model_config = model.config
    for name, value in assets.items():
        setattr(runtime, name, value)
    return runtime


def build_toy_batch(config, batch_size=2, prompt_len=24, seed=123):
    torch.manual_seed(seed)
    canvas_length = config.canvas_length
    vocab_size = config.text_config.vocab_size
    labels = torch.full((batch_size, canvas_length), IGNORE_INDEX, dtype=torch.long)
    labels[:, ::2] = torch.randint(3, vocab_size, (batch_size, canvas_length // 2))
    loss_weight = torch.zeros(batch_size, canvas_length)
    loss_weight[:, ::2] = 2.0 / canvas_length
    return {
        "input_ids": torch.randint(3, vocab_size, (batch_size, prompt_len)),
        "attention_mask": torch.ones(batch_size, prompt_len, dtype=torch.long),
        "decoder_input_ids": torch.randint(3, vocab_size, (batch_size, canvas_length)),
        "labels": labels,
        "loss_weight": loss_weight,
    }


class TestRegistration:
    def test_registered_under_model_type(self):
        from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY

        assert MODEL_CONFIG_REGISTRY["diffusion_gemma"]() is DiffusionGemmaConfig
        assert MODELING_REGISTRY["diffusion_gemma"]("DiffusionGemmaForBlockDiffusion") is (
            DiffusionGemmaForBlockDiffusion
        )

    def test_toy_config_loads_text_only(self):
        config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
        assert config.model_type == "diffusion_gemma"
        assert config.vision_config is None
        # Last layer is forced to full attention regardless of the 5:1 pattern.
        assert config.text_config.layer_types[-1] == "full_attention"

    @pytest.mark.parametrize(
        "implementation",
        [
            "flash_attention_2",
            "flash_attention_3",
            "veomni_flash_attention_2",
            "veomni_flash_attention_3",
        ],
    )
    def test_every_flash_attention_name_is_refused(self, implementation):
        """Flash reads the decoder's 4D mask as a padding mask and gathers out of bounds,
        which only shows up as a device-side assert. Fail at construction instead.

        Every name, not just the one a recipe writes: see the test below for why the
        two halves of this list are the same request.
        """
        with pytest.raises(ValueError, match="does not support FlashAttention"):
            build_toy_model(attn_implementation=implementation)

    @pytest.mark.parametrize("name", ["flash_attention_4", "veomni_flash_attention_4"])
    def test_flash_attention_4_is_closed_too(self, name):
        """Which error arrives first depends on the image, so this asserts only that one does.

        VeOmni checks `flash_attn.cute` is importable when the interface resolves, which is
        before this model's `__init__` gets to look at the name. Where FA4 *is* installed
        that check passes and the refusal above is what fires; here it is the ImportError.
        Either way the run stops, which is the property worth pinning.
        """
        with pytest.raises((ValueError, ImportError)):
            build_toy_model(attn_implementation=name)

    @pytest.mark.parametrize(
        "name",
        ["kernels-community/flash-attn2", "kernels-community/vllm-flash-attn3", "kernels-community/flash-attn4"],
    )
    @pytest.mark.parametrize("where", ["outer", "text"])
    def test_the_hub_kernel_names_are_refused_too(self, name, where, monkeypatch):
        """The second rewrite, and the one a substring test on `flash_attention` misses.

        `_check_and_adjust_attn_implementation` maps `flash_attention_N` onto these when
        flash-attn is not importable but `kernels` is, from inside
        `PreTrainedModel.__init__` -- i.e. before this model's own check.

        This image has flash-attn and no `kernels`, so that call raises `ImportError`
        while trying to fetch the kernel and the name never survives to be checked. An
        image with `kernels` installed is the one at risk, and it is emulated here by
        letting the name through untouched, which is what a successful kernel load
        leaves behind. Asserting `(ValueError, ImportError)` instead would have passed
        against the substring check this replaces, i.e. tested nothing.

        Parametrised over *where* the name lands because the rewrite writes only the
        outer config while the attention layers read `text_config`, so a check that
        consulted either one alone would pass half of these.
        """
        from transformers.modeling_utils import PreTrainedModel

        monkeypatch.setattr(
            PreTrainedModel,
            "_check_and_adjust_attn_implementation",
            lambda self, attn_implementation, **kwargs: attn_implementation,
        )
        config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
        target = config if where == "outer" else config.text_config
        target._attn_implementation_internal = name
        with pytest.raises(ValueError, match="does not support FlashAttention"):
            DiffusionGemmaForBlockDiffusion(config)

    def test_a_name_without_flash_in_it_still_builds(self):
        """The control for the test above: `flash` is the whole of the test, so a
        backend whose name merely looks unfamiliar must not be caught by it."""
        build_toy_model(attn_implementation="flex_attention")

    @pytest.mark.parametrize("version", [2, 3, 4])
    def test_the_name_a_recipe_writes_is_not_the_name_the_model_sees(self, version, monkeypatch):
        """Which is why the refusal above cannot compare against a literal.

        `MODELING_BACKEND` defaults to `veomni`, so asking for flash in a recipe
        reaches the model renamed. A check against `"flash_attention_2"` therefore
        never fired for a real run -- and never covered 3 or 4 under either name.

        The default is set explicitly rather than relied on: a developer with
        `MODELING_BACKEND=hf` exported would otherwise see this fail for a reason
        that has nothing to do with the property being pinned.
        """
        monkeypatch.setenv("MODELING_BACKEND", "veomni")
        resolved = OpsImplementationConfig(attn_implementation=f"flash_attention_{version}")
        assert resolved.attn_implementation == f"veomni_flash_attention_{version}"


class TestModelAssets:
    def _runtime(self, model_path="/model"):
        from types import SimpleNamespace

        from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaModelRuntime

        runtime = DiffusionGemmaModelRuntime.__new__(DiffusionGemmaModelRuntime)
        runtime.args = SimpleNamespace(model_path=model_path, tokenizer_path="/tokenizer")
        runtime.model_config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
        return runtime

    def test_text_checkpoint_serializes_source_generation_config(self, monkeypatch, tmp_path):
        from transformers import GenerationConfig

        from veomni.models import save_model_assets
        from veomni.trainer.dllm_trainer import DllmModelRuntime

        runtime = self._runtime()
        generation_config = GenerationConfig(max_new_tokens=17)

        def build_base_assets(self):
            self.tokenizer = None
            self.model_assets = [self.model_config]

        monkeypatch.setattr(DllmModelRuntime, "_build_model_assets", build_base_assets)
        monkeypatch.setattr(
            "veomni.trainer.diffusion_gemma_trainer.GenerationConfig.from_pretrained",
            lambda *args, **_: generation_config if args[-1] == "/model" else None,
        )

        runtime._build_model_assets()
        save_model_assets(str(tmp_path), [runtime.model_assets[-1]])

        saved = json.loads((tmp_path / "generation_config.json").read_text())
        assert saved["max_new_tokens"] == 17

    def test_missing_source_generation_config_falls_back_to_model_defaults(self, monkeypatch):
        from transformers import GenerationConfig

        from veomni.trainer.dllm_trainer import DllmModelRuntime

        runtime = self._runtime()
        derived = GenerationConfig(max_new_tokens=23)

        def build_base_assets(self):
            self.tokenizer = None
            self.model_assets = [self.model_config]

        def missing(*_args, **_kwargs):
            raise OSError("missing generation config")

        monkeypatch.setattr(DllmModelRuntime, "_build_model_assets", build_base_assets)
        monkeypatch.setattr(
            "veomni.trainer.diffusion_gemma_trainer.GenerationConfig.from_pretrained",
            missing,
        )
        monkeypatch.setattr(
            "veomni.trainer.diffusion_gemma_trainer.GenerationConfig.from_model_config",
            lambda *args, **_: derived,
        )

        runtime._build_model_assets()

        assert runtime.model_assets[-1] is derived

    def test_random_init_derives_generation_config_without_loading_a_checkpoint(self, monkeypatch):
        from transformers import GenerationConfig

        from veomni.trainer.dllm_trainer import DllmModelRuntime

        runtime = self._runtime(model_path=None)
        derived = GenerationConfig(max_new_tokens=31)

        def build_base_assets(self):
            self.tokenizer = None
            self.model_assets = [self.model_config]

        def unexpected_load(*_args, **_kwargs):
            raise AssertionError("random init must not load generation assets from a checkpoint")

        monkeypatch.setattr(DllmModelRuntime, "_build_model_assets", build_base_assets)
        monkeypatch.setattr(
            "veomni.trainer.diffusion_gemma_trainer.GenerationConfig.from_pretrained",
            unexpected_load,
        )
        monkeypatch.setattr(
            "veomni.trainer.diffusion_gemma_trainer.GenerationConfig.from_model_config",
            lambda *args, **_: derived,
        )

        runtime._build_model_assets()

        assert runtime.model_assets[-1] is derived


class TestWeightTying:
    """Encoder and decoder must share every transformer tensor, and only those."""

    def test_transformer_weights_are_shared(self):
        model = build_toy_model()
        encoder_layer = model.model.encoder.language_model.layers[0]
        decoder_layer = model.model.decoder.layers[0]

        assert encoder_layer.experts.gate_up_proj is decoder_layer.experts.gate_up_proj
        assert encoder_layer.experts.down_proj is decoder_layer.experts.down_proj
        assert encoder_layer.mlp.gate_proj.weight is decoder_layer.mlp.gate_proj.weight
        assert encoder_layer.self_attn.q_proj.weight is decoder_layer.self_attn.q_proj.weight
        assert encoder_layer.router.per_expert_scale is decoder_layer.router.per_expert_scale
        assert model.lm_head.weight is model.model.decoder.embed_tokens.weight

    def test_layer_scalar_buffers_stay_independent(self):
        model = build_toy_model()
        encoder_layer = model.model.encoder.language_model.layers[0]
        decoder_layer = model.model.decoder.layers[0]
        # Buffers are excluded from the tie on purpose: the checkpoint stores one per side.
        assert encoder_layer.layer_scalar is not decoder_layer.layer_scalar

    def test_only_self_conditioning_is_decoder_private(self):
        model = build_toy_model()
        encoder_param_ids = {id(p) for p in model.model.encoder.parameters()}
        private = {n for n, p in model.model.decoder.named_parameters() if id(p) not in encoder_param_ids}
        assert all(n.startswith("self_conditioning.") for n in private), private
        assert private

    def test_decoder_registered_before_encoder(self):
        """Deduplicated names must match the checkpoint, which stores the decoder side."""
        model = build_toy_model()
        names = [n for n, _ in model.named_parameters()]
        assert any(n.startswith("model.decoder.layers.") for n in names)
        assert not any(
            n.startswith("model.encoder.language_model.layers.") and n.endswith("q_proj.weight") for n in names
        )

    def test_freeze_router_matches_nemo_parameterization(self):
        config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
        config.freeze_router = True
        model = DiffusionGemmaForBlockDiffusion(config)
        encoder_router = model.model.encoder.language_model.layers[0].router
        decoder_router = model.model.decoder.layers[0].router

        assert not decoder_router.proj.weight.requires_grad
        assert not decoder_router.scale.requires_grad
        assert decoder_router.per_expert_scale.requires_grad
        assert encoder_router.proj.weight is decoder_router.proj.weight
        assert encoder_router.scale is decoder_router.scale
        assert encoder_router.per_expert_scale is decoder_router.per_expert_scale


class TestCheckpointLayout:
    """The published checkpoint stores shared tensors under `model.decoder.*`."""

    def _write_toy_checkpoint(self, path):
        model = build_toy_model(dtype=torch.float32, seed=7)
        for param in model.parameters():
            param.data.normal_(0, 0.02)
        for name, buffer in model.named_buffers():
            if name.endswith("layer_scalar"):
                buffer.data.normal_(1.0, 0.01)
        state_dict = model.state_dict()
        stored = {
            k: v.contiguous()
            for k, v in state_dict.items()
            if k.startswith("model.decoder.") or k.endswith("layer_scalar")
        }
        os.makedirs(path, exist_ok=True)
        save_file(stored, os.path.join(path, "model.safetensors"))
        with open(os.path.join(TOY_CONFIG_PATH, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
        return stored

    def test_every_parameter_is_covered_by_the_checkpoint(self, tmp_path):
        stored = self._write_toy_checkpoint(str(tmp_path))
        model = build_toy_model()
        params = {n for n, _ in model.named_parameters()}
        buffers = {n for n, _ in model.named_buffers()}
        # No parameter may be left uninitialized, and nothing stored may be dropped.
        assert not (params - set(stored))
        assert not (set(stored) - (params | buffers))

    def test_load_preserves_tying_and_values(self, tmp_path):
        from transformers.initialization import no_init_weights

        from veomni.models.checkpoint.weights import init_empty_weights, load_model_weights

        stored = self._write_toy_checkpoint(str(tmp_path))
        config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
        config._attn_implementation = "sdpa"
        # Same construction path as CustomizedModelingLoader: parameters on meta,
        # buffers materialized so non-persistent ones survive `to_empty`.
        with init_empty_weights(), no_init_weights():
            model = DiffusionGemmaForBlockDiffusion(config)
        load_model_weights(model, str(tmp_path), init_device="cpu")

        assert not any(p.is_meta for p in model.parameters())
        state_dict = model.state_dict()
        for name, expected in stored.items():
            assert torch.equal(state_dict[name], expected), name

        encoder_layer = model.model.encoder.language_model.layers[0]
        decoder_layer = model.model.decoder.layers[0]
        assert encoder_layer.experts.gate_up_proj is decoder_layer.experts.gate_up_proj
        assert model.lm_head.weight is model.model.decoder.embed_tokens.weight
        # Non-persistent buffers are recomputed, not loaded.
        inv_freq = model.model.decoder.rotary_emb.sliding_attention_inv_freq
        assert torch.isfinite(inv_freq).all()

        # `to_empty()` rebuilds every Parameter, so the retie hook has to cover *all* of
        # `DiffusionGemmaModel._tied_weights_keys`. Enumerating the sub-stacks by hand is
        # how `norm.weight` was missed: it stayed on uninitialized memory, never got a
        # gradient, and would have been written back out on save.
        decoder_params = dict(model.model.decoder.named_parameters())
        broken = [
            name
            for name, param in model.model.encoder.language_model.named_parameters()
            if name in decoder_params and decoder_params[name] is not param
        ]
        assert broken == []
        assert "norm.weight" in decoder_params

    def test_loads_a_raw_gemma4_layout_without_silent_missing_weights(self, tmp_path):
        from transformers.initialization import no_init_weights

        from veomni.models.checkpoint.weights import init_empty_weights, load_model_weights
        from veomni.models.transformers.diffusion_gemma import register_diffusion_gemma_modeling
        from veomni.models.transformers.diffusion_gemma.gemma4_initialization import (
            SELF_CONDITIONING_PARAMETER_NAMES,
        )

        register_diffusion_gemma_modeling("DiffusionGemmaForBlockDiffusion")
        source = build_toy_model(dtype=torch.float32, seed=11).state_dict()
        raw = {}
        for name, tensor in source.items():
            if name in SELF_CONDITIONING_PARAMETER_NAMES or name == "lm_head.weight":
                continue
            if name.startswith("model.decoder."):
                raw["model.language_model." + name[len("model.decoder.") :]] = tensor.contiguous()
            elif name.startswith("model.encoder.vision_tower."):
                raw["model.vision_tower." + name[len("model.encoder.vision_tower.") :]] = tensor.contiguous()
            elif name.startswith("model.encoder.embed_vision."):
                raw["model.embed_vision." + name[len("model.encoder.embed_vision.") :]] = tensor.contiguous()
        save_file(raw, os.path.join(tmp_path, "model.safetensors"))

        config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
        config._attn_implementation = "sdpa"
        with init_empty_weights(), no_init_weights():
            model = DiffusionGemmaForBlockDiffusion(config)
        load_model_weights(model, str(tmp_path), init_device="cpu")

        loaded = model.state_dict()
        for raw_name, expected in raw.items():
            if raw_name.startswith("model.language_model."):
                target = "model.decoder." + raw_name[len("model.language_model.") :]
            elif raw_name.startswith("model.vision_tower."):
                target = "model.encoder.vision_tower." + raw_name[len("model.vision_tower.") :]
            else:
                target = "model.encoder.embed_vision." + raw_name[len("model.embed_vision.") :]
            assert torch.equal(loaded[target], expected), target

        for layer in range(config.text_config.num_hidden_layers):
            decoder = loaded[f"model.decoder.layers.{layer}.layer_scalar"]
            encoder = loaded[f"model.encoder.language_model.layers.{layer}.layer_scalar"]
            assert torch.equal(decoder, encoder)
        assert all(torch.isfinite(loaded[name]).all() for name in SELF_CONDITIONING_PARAMETER_NAMES)


class TestForwardAndLoss:
    def test_forward_returns_canvas_logits_and_loss(self):
        model = build_toy_model().train()
        batch = build_toy_batch(model.config)
        outputs = model(**batch, use_cache=False)

        canvas_length = model.config.canvas_length
        assert outputs.logits.shape == (2, canvas_length, model.config.text_config.vocab_size)
        assert outputs.loss.isfinite()
        # Prompt is encoded separately, so the encoder state keeps the prompt length.
        assert outputs.encoder_last_hidden_state.shape[1] == batch["input_ids"].shape[1]

    def test_training_can_compute_the_same_loss_without_returning_logits(self):
        model = build_toy_model().eval()
        model.config.canvas_batching = "flat_seq"
        batch = build_toy_batch(model.config)
        # Toy vocab is 1024, so the default budget is one chunk for the whole canvas.
        # Force a multi-chunk path through the model so the split/accumulate boundary is
        # covered rather than only the unit-level helper.
        vocab = model.config.text_config.vocab_size
        tiny_budget = max(1, vocab * 2)
        with torch.no_grad():
            materialized = model(**batch, use_cache=False)
            chunked = model(
                **batch,
                use_cache=False,
                return_logits=False,
                logit_elements_per_chunk=tiny_budget,
            )

        assert chunked.logits is None
        torch.testing.assert_close(chunked.diffusion_loss, materialized.diffusion_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(chunked.encoder_ar_loss, materialized.encoder_ar_loss, rtol=1e-4, atol=1e-4)

    def test_self_conditioning_embeddings_match_materialized_logits(self):
        torch.manual_seed(7)
        hidden = torch.randn(2, 5, 7, dtype=torch.float64)
        weight = torch.randn(13, 7, dtype=torch.float64)
        scale = torch.tensor(7**0.5, dtype=torch.float64)
        logits = torch.nn.functional.linear(hidden, weight).float()
        logits = torch.tanh(logits / 30.0) * 30.0
        expected = torch.matmul(logits.softmax(dim=-1).to(weight.dtype), weight) * scale

        actual = chunked_self_conditioning_embeddings(
            hidden,
            weight,
            weight,
            scale,
            softcapping=30.0,
            logit_elements_per_chunk=26,
        )

        torch.testing.assert_close(actual, expected)

    def test_chunked_diffusion_loss_matches_value_and_gradients(self):
        torch.manual_seed(7)
        hidden = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
        head = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1, 2, IGNORE_INDEX], [4, 5, 6]])
        weights = torch.tensor([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float32)

        logits = torch.nn.functional.linear(hidden, head).float()
        logits = torch.tanh(logits / 30.0) * 30.0
        per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
        )
        reference = (per_token * weights.reshape(-1)).sum()
        reference_grads = torch.autograd.grad(reference, (hidden, head))

        actual = chunked_weighted_diffusion_loss(
            hidden,
            head,
            labels,
            weights,
            softcapping=30.0,
            logit_elements_per_chunk=22,
        )
        actual_grads = torch.autograd.grad(actual, (hidden, head))

        torch.testing.assert_close(actual, reference)
        for got, want in zip(actual_grads, reference_grads, strict=True):
            torch.testing.assert_close(got, want)

    def test_chunked_diffusion_loss_ignores_the_weight_at_an_ignored_position(self):
        """`dllm_masked_cross_entropy` never reads those weights, so neither may this."""
        torch.manual_seed(7)
        hidden = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
        head = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1, 2, IGNORE_INDEX], [4, 5, 6]])
        weights = torch.tensor([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float32)
        poisoned = weights.clone()
        poisoned[0, 2] = float("inf")

        kwargs = {"softcapping": 30.0, "logit_elements_per_chunk": 22}
        reference = chunked_weighted_diffusion_loss(hidden, head, labels, weights, **kwargs)
        actual = chunked_weighted_diffusion_loss(hidden, head, labels, poisoned, **kwargs)

        torch.testing.assert_close(actual, reference)

    def test_chunked_diffusion_loss_rejects_a_layout_that_only_matches_after_flatten(self):
        hidden = torch.randn(2, 3, 5)
        head = torch.randn(11, 5)
        labels = torch.arange(6).reshape(3, 2)
        weights = torch.ones(3, 2)
        with pytest.raises(ValueError, match="must describe the same tokens"):
            chunked_weighted_diffusion_loss(hidden, head, labels, weights)

    def test_use_cache_kwarg_is_accepted(self):
        """`BaseTrainer.forward_backward_step` always passes `use_cache=False`."""
        model = build_toy_model().train()
        batch = build_toy_batch(model.config)
        with_flag = model(**batch, use_cache=False).loss
        without_flag = model(**batch).loss
        assert torch.equal(with_flag, without_flag)

    def test_labels_require_loss_weight(self):
        model = build_toy_model().train()
        batch = build_toy_batch(model.config)
        batch.pop("loss_weight")
        with pytest.raises(ValueError, match="loss_weight"):
            model(**batch)

    def test_diffusion_loss_is_the_weighted_sum_over_labelled_positions(self):
        model = build_toy_model().eval()
        batch = build_toy_batch(model.config)
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        mask = batch["labels"] != IGNORE_INDEX
        per_token = torch.nn.functional.cross_entropy(logits[mask].float(), batch["labels"][mask], reduction="none")
        expected = (per_token * batch["loss_weight"][mask]).sum()
        torch.testing.assert_close(outputs.diffusion_loss, expected, rtol=1e-4, atol=1e-4)

    def test_loss_is_the_two_terms_of_eq_13_added(self):
        """No weight between them: what balances the two is that the encoder term
        divides by its whole sequence and the canvas term by one canvas."""
        model = build_toy_model().eval()
        with torch.no_grad():
            outputs = model(**build_toy_batch(model.config))
        torch.testing.assert_close(outputs.loss, outputs.diffusion_loss + outputs.encoder_ar_loss)
        assert outputs.encoder_ar_loss > 0

    def test_gradient_flows_through_the_encoder_kv_cache(self):
        """The encoder reaches the loss only via the cache the decoder reads."""
        model = build_toy_model().train()
        captured = {}

        def hook(module, args, kwargs, output):
            captured["cache"] = output.past_key_values
            return output

        model.model.encoder.register_forward_hook(hook, with_kwargs=True)
        outputs = model(**build_toy_batch(model.config))
        cache_keys = captured["cache"].layers[0].keys
        assert cache_keys.requires_grad
        grad = torch.autograd.grad(outputs.loss, cache_keys, retain_graph=True)[0]
        assert grad.abs().sum() > 0

    def test_all_parameters_receive_gradients(self):
        model = build_toy_model().train()
        model(**build_toy_batch(model.config)).loss.backward()
        without_grad = [n for n, p in model.named_parameters() if p.grad is None]
        assert not without_grad


class TestEncoderAutoregressiveLoss:
    """The encoder half of Eq. 13: causal next-token cross-entropy over the prompt.

    The canvas term already reaches the encoder through the KV cache, but only as far
    as the drawn canvas needs; this term is what keeps the autoregressive sampling the
    released weights ship with from drifting.
    """

    @staticmethod
    def _reference(model, batch):
        """The same loss with the logits materialized, which is what the chunked
        implementation exists to avoid at the real 262k vocabulary."""
        with torch.no_grad():
            hidden = model(**batch).encoder_last_hidden_state
        logits = model.lm_head(hidden).float()
        softcapping = model.config.text_config.final_logit_softcapping
        logits = torch.tanh(logits / softcapping) * softcapping
        input_ids, mask = batch["input_ids"], batch.get("attention_mask")
        total = torch.zeros(())
        for row in range(input_ids.shape[0]):
            real = mask[row].bool() if mask is not None else torch.ones_like(input_ids[row], dtype=torch.bool)
            positions = [
                t
                for t in range(1, input_ids.shape[1])
                if real[t]
                and real[t - 1]
                and (batch.get("mm_token_type_ids") is None or not batch["mm_token_type_ids"][row, t])
            ]
            per_token = torch.nn.functional.cross_entropy(
                logits[row, [t - 1 for t in positions]], input_ids[row, positions], reduction="none"
            )
            total = total + per_token.mean()
        return total

    def test_matches_the_loss_with_the_logits_materialized(self):
        model = build_toy_model().eval()
        batch = build_toy_batch(model.config)
        with torch.no_grad():
            outputs = model(**batch)
        torch.testing.assert_close(outputs.encoder_ar_loss, self._reference(model, batch), rtol=1e-4, atol=1e-4)

    def test_the_chunking_does_not_change_the_value(self):
        """The chunk is sized off the vocabulary, so a config change must not move the
        loss -- a bug in the split would show up as a silently different objective."""
        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import (
            chunked_weighted_causal_loss,
            encoder_ar_targets,
        )

        model = build_toy_model().eval()
        batch = build_toy_batch(model.config)
        with torch.no_grad():
            hidden = model(**batch).encoder_last_hidden_state
            labels, weights = encoder_ar_targets(batch["input_ids"], batch["attention_mask"])
            values = [
                chunked_weighted_causal_loss(
                    hidden,
                    model.lm_head.weight,
                    labels,
                    weights,
                    softcapping=model.config.text_config.final_logit_softcapping,
                    logit_elements_per_chunk=elements,
                )
                for elements in (2**26, 1024 * 3, 1024)
            ]
        for value in values[1:]:
            torch.testing.assert_close(value, values[0], rtol=1e-5, atol=1e-5)

    def test_padding_does_not_change_a_rows_contribution(self):
        """The collator left-pads, so a padded row's leading hidden states attend to
        nothing but themselves. Supervising them, or the first real token they precede,
        would make a sample's loss depend on who else was in the batch."""
        model = build_toy_model().eval()
        canvas_length = model.config.canvas_length
        torch.manual_seed(11)
        alone = {
            "input_ids": torch.randint(3, 1024, (1, 9)),
            "attention_mask": torch.ones(1, 9, dtype=torch.long),
            "decoder_input_ids": torch.randint(3, 1024, (1, canvas_length)),
            "labels": torch.full((1, canvas_length), IGNORE_INDEX),
            "loss_weight": torch.zeros(1, canvas_length),
        }
        padded = {key: value.clone() for key, value in alone.items()}
        padded["input_ids"] = torch.cat([torch.zeros(1, 4, dtype=torch.long), alone["input_ids"]], dim=1)
        padded["attention_mask"] = torch.cat([torch.zeros(1, 4, dtype=torch.long), alone["attention_mask"]], dim=1)
        with torch.no_grad():
            torch.testing.assert_close(
                model(**padded).encoder_ar_loss, model(**alone).encoder_ar_loss, rtol=2e-3, atol=2e-3
            )

    def test_image_spans_are_not_targets(self):
        """Image ids are one placeholder repeated per soft token and their embeddings
        come from the vision tower, so predicting them trains the head to emit the
        placeholder. The text token right after an image stays supervised."""
        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import encoder_ar_targets

        input_ids = torch.tensor([[5, 1023, 1023, 7, 8]])
        mm_token_type_ids = torch.tensor([[0, 1, 1, 0, 0]])
        labels, weights = encoder_ar_targets(input_ids, torch.ones_like(input_ids), mm_token_type_ids)
        assert labels.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 7, 8]]
        torch.testing.assert_close(weights, torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5]]))

    def test_a_prepared_mask_mapping_is_refused(self):
        """A dict hides where the padding is, and guessing "unpadded" would supervise
        pad positions -- a silent change of objective for exactly the batches that have
        padding."""
        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import encoder_ar_targets

        with pytest.raises(ValueError, match="attention_mask"):
            encoder_ar_targets(torch.tensor([[5, 6]]), {"full_attention": None})

    def test_a_row_with_nothing_to_supervise_contributes_zero(self):
        """A one-token prompt has no position with a predecessor. The weights must not
        divide by that zero -- the whole step's loss would go to NaN."""
        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import encoder_ar_targets

        labels, weights = encoder_ar_targets(torch.tensor([[5]]), torch.ones(1, 1, dtype=torch.long))
        assert (labels == IGNORE_INDEX).all()
        assert float(weights.sum()) == 0.0

    def test_masking_the_prompt_leaves_only_the_answer(self):
        """The prompt stays in the cache; only the clean canvases train the head.
        The first answer token is still predicted from the last prompt position."""
        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import encoder_ar_targets

        input_ids = torch.tensor([[5, 6, 7, 20, 21, 22]])
        prompt_mask = torch.tensor([[True, True, True, False, False, False]])
        off_labels, _ = encoder_ar_targets(input_ids, torch.ones_like(input_ids), encoder_prompt_mask=prompt_mask)
        on_labels, on_weights = encoder_ar_targets(
            input_ids, torch.ones_like(input_ids), encoder_prompt_mask=prompt_mask, mask_prompt_ar_loss=True
        )
        assert off_labels.tolist() == [[IGNORE_INDEX, 6, 7, 20, 21, 22]]
        assert on_labels.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21, 22]]
        torch.testing.assert_close(on_weights, torch.tensor([[0.0, 0.0, 0.0, 1 / 3, 1 / 3, 1 / 3]]))

    def test_masking_other_tail_eos_keeps_only_the_first(self):
        """The stop is the first EOS of the trailing fill; the rest is EOS-to-EOS."""
        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import encoder_ar_targets

        input_ids = torch.tensor([[5, 6, 7, 1, 1, 1]])
        attn = torch.ones_like(input_ids)
        on_labels, on_weights = encoder_ar_targets(input_ids, attn)
        off_labels, off_weights = encoder_ar_targets(input_ids, attn, mask_other_tail_eos=False)
        assert on_labels.tolist() == [[IGNORE_INDEX, 6, 7, 1, IGNORE_INDEX, IGNORE_INDEX]]
        torch.testing.assert_close(on_weights, torch.tensor([[0.0, 1 / 3, 1 / 3, 1 / 3, 0.0, 0.0]]))
        assert off_labels.tolist() == [[IGNORE_INDEX, 6, 7, 1, 1, 1]]
        torch.testing.assert_close(off_weights, torch.tensor([[0.0, 0.2, 0.2, 0.2, 0.2, 0.2]]))

    def test_a_single_trailing_eos_stays_supervised(self):
        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import encoder_ar_targets

        input_ids = torch.tensor([[5, 6, 1]])
        labels, _ = encoder_ar_targets(input_ids, torch.ones_like(input_ids))
        assert labels.tolist() == [[IGNORE_INDEX, 6, 1]]

    def test_an_eos_that_is_not_in_the_suffix_is_kept(self):
        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import encoder_ar_targets

        input_ids = torch.tensor([[5, 1, 6, 7]])
        labels, _ = encoder_ar_targets(input_ids, torch.ones_like(input_ids))
        assert labels.tolist() == [[IGNORE_INDEX, 1, 6, 7]]

    def test_masking_the_prompt_changes_the_logged_ar_loss(self):
        model = build_toy_model().eval()
        batch = build_toy_batch(model.config)
        mask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
        mask[:, :8] = True
        batch["encoder_prompt_mask"] = mask
        with torch.no_grad():
            full = model(**batch).encoder_ar_loss
            model.config.mask_prompt_ar_loss = True
            masked = model(**batch).encoder_ar_loss
        assert not torch.allclose(full, masked)
        assert float(masked) > 0

    def test_masking_other_tail_eos_changes_the_logged_ar_loss(self):
        model = build_toy_model().eval()
        assert model.config.mask_other_tail_eos is True
        assert model.config.compute_all_canvas_df_loss is False
        batch = build_toy_batch(model.config)
        batch["input_ids"][:, -4:] = 1
        with torch.no_grad():
            masked = model(**batch).encoder_ar_loss
            model.config.mask_other_tail_eos = False
            full = model(**batch).encoder_ar_loss
        assert not torch.allclose(full, masked)
        assert float(masked) > 0

    def test_encoder_ar_row_weight_drops_the_repeated_example_rows(self):
        """All-canvas expansion repeats the encoder; only the first row may carry AR."""
        model = build_toy_model().eval()
        batch = build_toy_batch(model.config, batch_size=2)
        batch["input_ids"][1] = batch["input_ids"][0]
        batch["attention_mask"][1] = batch["attention_mask"][0]
        with torch.no_grad():
            both = model(**batch).encoder_ar_loss
            batch["encoder_ar_row_weight"] = torch.tensor([1.0, 0.0])
            first = model(**batch).encoder_ar_loss
        assert float(both) > float(first) > 0

    def test_stacked_canvases_reuse_one_encoder_forward(self):
        """All-canvas training must not clone the encoder batch; that OOMs 26B."""
        model = build_toy_model().train()
        canvas_length = model.config.canvas_length
        vocab_size = model.config.text_config.vocab_size
        collator = DiffusionGemmaCollator(pad_token_id=0)
        feature = {
            "input_ids": torch.randint(3, vocab_size, (9,)),
            "decoder_context_mask": torch.stack(
                [
                    torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0]),
                    torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
                ]
            ),
            "decoder_input_ids": torch.randint(3, vocab_size, (2, canvas_length)),
            "labels": torch.randint(3, vocab_size, (2, canvas_length)),
            "loss_weight": torch.full((2, canvas_length), 0.5 / canvas_length),
        }
        batch = collator([feature])
        assert batch["input_ids"].shape[0] == 1
        assert batch["decoder_input_ids"].shape == (1, 2, canvas_length)
        outputs = model(**batch, use_cache=False)
        assert outputs.logits.shape == (1, 2, canvas_length, vocab_size)
        assert outputs.loss.isfinite()
        assert outputs.encoder_last_hidden_state.shape[0] == 1

    def test_the_encoder_is_supervised_outside_the_cache(self):
        """The decoder is nowhere in this term's graph, so a gradient reaching the
        encoder stack from it is supervision the cache read does not carry."""
        model = build_toy_model().train()
        outputs = model(**build_toy_batch(model.config))
        grads = torch.autograd.grad(
            outputs.encoder_ar_loss, model.model.encoder.language_model.layers[0].self_attn.q_proj.weight
        )
        assert grads[0].abs().sum() > 0

    def test_turning_it_off_leaves_the_canvas_term_alone(self):
        model = build_toy_model().eval()
        batch = build_toy_batch(model.config)
        with torch.no_grad():
            on = model(**batch)
            model.config.encoder_ar_loss = False
            off = model(**batch)
        assert float(off.encoder_ar_loss) == 0.0
        torch.testing.assert_close(off.loss, on.diffusion_loss)


class TestSlidingWindowMask:
    def test_window_is_materialized_even_without_padding(self):
        """`sdpa` ignores `sliding_window`, so the window only exists while the mask does."""
        from unittest import mock

        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaDecoderModel

        model = build_toy_model()
        text_config = model.config.text_config
        assert "sliding_attention" in text_config.layer_types, "toy config no longer covers this path"

        # No `decoder_attention_mask` and equal-length prompts: exactly the case the
        # upstream shortcut would hand to the backend unmaterialized.
        batch = build_toy_batch(model.config, batch_size=2, prompt_len=24)
        original = DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask
        captured = {}

        def spy(*args, **kwargs):
            captured.update(original(*args, **kwargs))
            return dict(captured)

        with mock.patch.object(
            DiffusionGemmaDecoderModel, "create_diffusion_decoder_attention_mask", staticmethod(spy)
        ):
            with torch.no_grad():
                model(**batch)

        assert captured["sliding_attention"] is not None
        assert captured["full_attention"] is not None
        # Training keeps the full encoder history, so the window is a mask on the same
        # key axis rather than a shorter tensor.
        assert captured["sliding_attention"].shape[-1] == captured["full_attention"].shape[-1]
        assert captured["sliding_attention"].sum() < captured["full_attention"].sum()


def build_padded_batch(config, batch_size=2, prompt_len=40, pad=7):
    """A batch with left padding on the prompt, plus the decoder mask the collator would emit.

    Padding is the only thing the decoder mask carries, so a batch without it cannot tell
    the attention backends apart.
    """
    batch = build_toy_batch(config, batch_size=batch_size, prompt_len=prompt_len)
    batch["attention_mask"][-1, :pad] = 0
    batch["input_ids"][-1, :pad] = 0
    decoder_attention_mask = torch.ones(batch_size, prompt_len + config.canvas_length, dtype=torch.long)
    decoder_attention_mask[-1, :pad] = 0
    batch["decoder_attention_mask"] = decoder_attention_mask
    return batch


class TestAttentionBackends:
    """The decoder's mask has to mean the same thing to every backend that is allowed."""

    def _logits(self, impl, device="cpu"):
        model = build_toy_model(attn_implementation=impl).to(device).eval()
        batch = {k: v.to(device) for k, v in build_padded_batch(model.config).items()}
        with torch.no_grad():
            return model(**batch, use_cache=False).logits.float().cpu()

    def test_eager_gets_an_additive_mask_not_a_boolean_one(self):
        """`eager_attention_forward` adds the mask to the scores.

        A boolean mask would therefore add +1.0 to every position it means to keep and 0.0 to
        the ones it means to drop -- padding stays visible and all scores shift. Upstream
        returns bool for every backend, so this is a VeOmni-side correction.
        """
        model = build_toy_model(attn_implementation="eager")
        masks = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=model.config.text_config,
            inputs_embeds=torch.zeros(2, model.config.canvas_length, model.config.text_config.hidden_size),
            past_key_values=_prefilled_cache(model, build_padded_batch(model.config)),
            decoder_attention_mask=build_padded_batch(model.config)["decoder_attention_mask"],
        )
        for mask in masks.values():
            assert mask.dtype is not torch.bool
            assert (mask <= 0).all(), "an additive mask may only subtract from the scores"
        # The prompt padding sits at the front of the cache, which the sliding window has
        # already scrolled past, so only the full-attention mask can show the -inf.
        assert masks["full_attention"].min() < -1e30, "masked-out positions must be effectively -inf"

    def test_eager_matches_sdpa(self):
        assert torch.allclose(self._logits("eager"), self._logits("sdpa"), atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs a GPU")
    def test_flex_attention_matches_sdpa(self):
        """flex sees the same mask as a `BlockMask` instead of a materialized tensor."""
        assert torch.allclose(self._logits("flex_attention", "cuda"), self._logits("sdpa", "cuda"), atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs a GPU")
    def test_flex_attention_matches_sdpa_in_the_backward(self):
        """The forward agreeing is not enough: flex once returned zero dK and dV here.

        ``flex_attention_mask`` builds its ``BlockMask`` through
        ``create_block_mask(_compile=True)``, and past the first shape in a process that
        builder disagrees with itself. The model gives it four shapes a step -- encoder and
        decoder, sliding and full -- and the decoder's, which come second, came back with the
        transposed sparsity flex's dK/dV kernel reads. Both gradients were exactly zero and
        the forward stayed exact, so ``test_flex_attention_matches_sdpa`` above passed
        throughout and only the grad norm moved.

        The padding matters as much as the two backends do: without it the decoder takes the
        shortcut in ``create_diffusion_decoder_attention_mask`` and gets no ``BlockMask`` at
        all, and this passes whether ``uncompiled_block_masks`` is in place or not.
        """
        gradients = {}
        for implementation in ("sdpa", "flex_attention"):
            model = build_toy_model(attn_implementation=implementation).to("cuda").train()
            batch = {k: v.to("cuda") for k, v in build_padded_batch(model.config).items()}
            model(**batch, use_cache=False).loss.backward()
            gradients[implementation] = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}

        # Compare the sets first. The fault being guarded here can also present as a parameter
        # dropping out of the graph entirely, and indexing one dict by the other's keys would
        # report that as a KeyError from this line rather than as the assertion below.
        assert gradients["sdpa"].keys() == gradients["flex_attention"].keys(), (
            "the two backends disagree about which parameters even receive a gradient: "
            f"sdpa-only {sorted(gradients['sdpa'].keys() - gradients['flex_attention'].keys())}, "
            f"flex-only {sorted(gradients['flex_attention'].keys() - gradients['sdpa'].keys())}"
        )

        worst_name, worst_error = "", 0.0
        for name, reference in gradients["sdpa"].items():
            error = (gradients["flex_attention"][name] - reference).norm() / reference.norm().clamp(min=1e-12)
            if error > worst_error:
                worst_name, worst_error = name, error.item()
        assert worst_error < 1e-3, (
            f"flex_attention's gradient for {worst_name} is {worst_error:.3f} away from sdpa's, so a run on "
            "flex trains a different model than the same config on sdpa"
        )

    def test_choosing_flex_buys_a_dynamo_budget_and_refuses_the_unfused_path(self):
        """Both are what keep flex from leaving without saying so.

        flex is compiled per call signature and this model needs more than Dynamo's default 8
        cache entries for the frame. Past the budget Dynamo stops compiling and flex answers by
        materializing the whole score matrix -- which under gradient checkpointing kills the step
        somewhere unrelated-looking, and without it just reports the dense implementation's cost
        as flex's. Neither the raised budget nor the promoted warning has anything to do with the
        toy model's own numbers, so nothing else here would notice their removal. No GPU needed:
        both are process state set at construction.
        """
        import torch._dynamo.config as dynamo_config

        build_toy_model(attn_implementation="flex_attention")

        assert dynamo_config.recompile_limit >= 64
        assert dynamo_config.accumulated_recompile_limit >= 512
        with pytest.raises(UserWarning, match="without torch.compile"):
            warnings.warn(
                "flex_attention called without torch.compile() - this will use an unfused implementation",
                UserWarning,
                stacklevel=1,
            )

    def test_asking_for_flex_and_dg_prefix_together_is_refused(self):
        """DG_PREFIX reads the decoder mask to learn which encoder keys to pack, and under flex
        that mask is a `BlockMask` with no key-validity tensor on it. Refuse at construction:
        the alternative found it 26B parameters into a run, and dropping either flag instead
        would have the run report the other's speed."""
        config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
        config._attn_implementation = "flex_attention"
        config.text_config.attn_dg_prefix = True
        with pytest.raises(ValueError, match="attn_dg_prefix and flex_attention"):
            DiffusionGemmaForBlockDiffusion(config)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs a GPU")
    def test_flex_attention_gets_a_block_mask(self):
        """A dense mask would be read as an *additive* score mask by flex, silently wrong."""
        from torch.nn.attention.flex_attention import BlockMask

        model = build_toy_model(attn_implementation="flex_attention").to("cuda")
        batch = {k: v.to("cuda") for k, v in build_padded_batch(model.config).items()}
        masks = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=model.config.text_config,
            inputs_embeds=torch.zeros(
                2, model.config.canvas_length, model.config.text_config.hidden_size, device="cuda"
            ),
            past_key_values=_prefilled_cache(model, batch),
            decoder_attention_mask=batch["decoder_attention_mask"],
        )
        assert isinstance(masks["full_attention"], BlockMask)
        assert isinstance(masks["sliding_attention"], BlockMask)
        # Training keeps the full encoder history; the window lives in the mask, not the length.
        assert masks["sliding_attention"].seq_lengths[1] == masks["full_attention"].seq_lengths[1]

    def test_flash_attention_2_is_still_rejected(self):
        with pytest.raises(ValueError, match="flash_attention_2"):
            build_toy_model(attn_implementation="flash_attention_2")

    def test_wide_global_heads_get_a_smaller_flex_tile(self):
        """The 512-wide global head does not fit any tile Inductor autotunes over on an A100.

        Without a cap the first global layer to compile kills the run with "No valid triton
        configs", so the model has to ask for a tile the device can hold, and for the main
        kernel rather than the short-query one that ignores the cap. Narrow heads are left to
        the autotuner.
        """
        from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import flex_tile_cap

        a100, h100 = 166912, 232448
        assert flex_tile_cap(head_dim=256, shared_memory_limit=a100) is None
        assert flex_tile_cap(head_dim=512, shared_memory_limit=a100) == {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_M1": 32,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 32,
            "FORCE_USE_FLEX_ATTENTION": True,
        }
        assert flex_tile_cap(head_dim=512, shared_memory_limit=h100) is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs a GPU")
    def test_flex_attention_runs_with_the_real_head_dims(self):
        """The toy config's heads are 32/64 wide, far too narrow to reach the SRAM limit.

        The prompt is deliberately under 128 tokens, which is where Inductor reaches for its
        short-query kernel and the tile cap alone is not enough.

        Inference only, since training on flex_attention is refused.
        """
        config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
        config.text_config.head_dim = 256
        config.text_config.global_head_dim = 512
        config._attn_implementation = "flex_attention"
        torch.manual_seed(0)
        model = DiffusionGemmaForBlockDiffusion(config).to(device="cuda", dtype=torch.bfloat16).eval()

        batch = {k: v.to("cuda") for k, v in build_toy_batch(config, batch_size=1, prompt_len=21).items()}
        with torch.no_grad():
            logits = model(**batch, use_cache=False).logits

        assert torch.isfinite(logits).all()


class TestFFPAAttention:
    """FFPA replaces sdpa on the 512-wide global heads, where sdpa has no fused kernel.

    Correctness is judged on gradients, not just outputs -- flex_attention was exact in the
    forward pass and silently wrong in the backward one.
    """

    # Long enough that FFPA does not hand the call straight back to sdpa (`Nq >= 512`), and
    # padded well past the 64-wide key tile whose boolean form used to come back NaN.
    SEQ_LEN = 640
    PAD = 128

    def _wide_config(self, attn_ffpa):
        config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
        config.text_config.head_dim = 256
        config.text_config.global_head_dim = 512
        config.text_config.attn_ffpa = attn_ffpa
        config._attn_implementation = "sdpa"
        return config

    def test_the_flag_defaults_to_off(self):
        assert build_toy_model().config.text_config.attn_ffpa is False

    def test_narrow_heads_stay_on_sdpa(self):
        """FFPA has no kernel at or below 256, so only the global layers may switch."""
        pytest.importorskip("ffpa_attn")
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        from veomni.models.transformers.diffusion_gemma.ffpa_attention import (
            ffpa_attention_forward,
            resolve_attention_interface,
        )

        config = self._wide_config(attn_ffpa=True).text_config
        assert resolve_attention_interface(config, config.head_dim, None) is sdpa_attention_forward
        assert resolve_attention_interface(config, config.global_head_dim, None) is ffpa_attention_forward

    def test_asking_for_ffpa_on_another_backend_is_refused(self):
        """The flag means "instead of sdpa", so pairing it with eager is a contradiction.

        Resolving it either way would be a guess about which half the caller meant.
        """
        from veomni.models.transformers.diffusion_gemma.ffpa_attention import resolve_attention_interface

        def sentinel(*args, **kwargs):
            raise AssertionError("not called")

        config = self._wide_config(attn_ffpa=True).text_config
        config._attn_implementation = "eager"
        with pytest.raises(ValueError, match="attn_ffpa"):
            resolve_attention_interface(config, config.global_head_dim, sentinel)

    def test_a_boolean_mask_is_handed_over_additively(self):
        """A boolean mask makes the Triton kernel return NaN when a whole key tile is dead."""
        from veomni.models.transformers.diffusion_gemma.ffpa_attention import _as_additive_mask

        valid_keys = torch.ones(2, 1, 4, 16, dtype=torch.bool)
        valid_keys[..., :8] = False
        additive = _as_additive_mask(valid_keys, torch.bfloat16)
        assert additive.dtype is torch.bfloat16
        # Shape too: a collapse that fired unconditionally would still satisfy the value
        # assertions below, and only the end-to-end test would notice.
        assert additive.shape == (2, 1, 4, 16)
        assert (additive[..., 8:] == 0).all()
        assert (additive[..., :8] == torch.finfo(torch.bfloat16).min).all()
        # A mask that is already additive has to pass through untouched.
        assert _as_additive_mask(additive, torch.bfloat16) is additive
        assert _as_additive_mask(None, torch.bfloat16) is None

    def test_a_broadcast_mask_stays_broadcast(self):
        """The model's mask is an `expand` over the query axis; materializing it would undo
        a good part of the memory this whole path is here to save, and FFPA accepts a mask
        whose query axis is 1."""
        from veomni.models.transformers.diffusion_gemma.ffpa_attention import _as_additive_mask

        valid_keys = torch.ones(2, 1, 1, 16, dtype=torch.bool).expand(2, 1, 512, 16)
        additive = _as_additive_mask(valid_keys, torch.bfloat16)
        assert additive.shape == (2, 1, 1, 16)

    def test_a_missing_package_fails_the_run_instead_of_downgrading(self, monkeypatch):
        """`ffpa-attn` is not in the lockfile, so any `uv sync` removes it under a live
        config. Downgrading to sdpa there is what let a trial report sdpa's step time and
        peak memory as FFPA's, so the mismatch has to stop the run."""
        from veomni.models.transformers.diffusion_gemma import ffpa_attention

        monkeypatch.setattr(ffpa_attention, "FFPA_AVAILABLE", False)
        monkeypatch.setattr(ffpa_attention, "FFPA_IMPORT_ERROR", "ModuleNotFoundError: ffpa_attn")
        config = self._wide_config(attn_ffpa=True).text_config
        with pytest.raises(ImportError, match="attn_ffpa"):
            ffpa_attention.ffpa_is_usable(config, config.global_head_dim)

    def test_the_refusal_happens_at_construction_not_at_the_first_step(self, monkeypatch):
        """Both conditions are static, so paying for a weight load and a step to learn about
        them is waste -- and under FSDP2 it raises mid-forward, between an all-gather and its
        reshard, rather than while the model is still a local object."""
        from veomni.models.transformers.diffusion_gemma import ffpa_attention

        monkeypatch.setattr(ffpa_attention, "FFPA_AVAILABLE", False)
        monkeypatch.setattr(ffpa_attention, "FFPA_IMPORT_ERROR", "ModuleNotFoundError: ffpa_attn")
        with pytest.raises(ImportError, match="attn_ffpa"):
            DiffusionGemmaForBlockDiffusion(self._wide_config(attn_ffpa=True))

    def test_the_package_is_only_required_when_the_flag_asks_for_it(self, monkeypatch):
        """Failing closed must not promote the optional dependency to a hard one for runs
        that never asked for it -- which is every DiffusionGemma run by default."""
        from veomni.models.transformers.diffusion_gemma import ffpa_attention

        monkeypatch.setattr(ffpa_attention, "FFPA_AVAILABLE", False)
        off = self._wide_config(attn_ffpa=False)
        assert not ffpa_attention.ffpa_is_usable(off.text_config, off.text_config.global_head_dim)
        ffpa_attention.ffpa_preflight(off.text_config)
        DiffusionGemmaForBlockDiffusion(off)

    def test_narrow_layers_of_an_ffpa_run_stay_on_sdpa(self):
        """The flag is honoured per layer: 25 of the 30 are 256 wide, below FFPA's range."""
        pytest.importorskip("ffpa_attn")
        from veomni.models.transformers.diffusion_gemma import ffpa_attention

        config = self._wide_config(attn_ffpa=True).text_config
        assert not ffpa_attention.ffpa_is_usable(config, config.head_dim)
        assert ffpa_attention.ffpa_is_usable(config, config.global_head_dim)

    def _attention_inputs(self, q_len, kv_len, padded):
        """Query/key/value and mask shaped like one of the model's global layers."""
        torch.manual_seed(0)
        heads, kv_heads, head_dim = 16, 2, 512
        shapes = [(1, heads, q_len, head_dim), (1, kv_heads, kv_len, head_dim), (1, kv_heads, kv_len, head_dim)]
        qkv = [
            (torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * head_dim**-0.25).requires_grad_(True)
            for shape in shapes
        ]
        # A random cotangent: `out.sum()` gives every query the same dO, which cancels enough
        # of the softmax Jacobian that a wrong dQ can still look right.
        cotangent = torch.randn(1, heads, q_len, head_dim, device="cuda", dtype=torch.bfloat16)
        mask = None
        if padded:
            valid_keys = torch.ones(1, kv_len, dtype=torch.bool, device="cuda")
            valid_keys[:, : self.PAD] = False  # prompts are left-padded
            mask = valid_keys[:, None, None, :].expand(1, 1, q_len, kv_len)
        return qkv, cotangent, mask

    def _out_and_grads(self, forward, qkv, mask, cotangent, is_causal=False, fp32=False):
        module = torch.nn.Module()
        module.is_causal = is_causal
        module.num_key_value_groups = qkv[0].shape[1] // qkv[1].shape[1]
        if fp32:
            qkv = [tensor.detach().float().requires_grad_(True) for tensor in qkv]
            cotangent = cotangent.float()
        out, _ = forward(module, *qkv, mask, dropout=0.0, scaling=1.0)
        return out, torch.autograd.grad((out * cotangent.transpose(1, 2)).sum(), qkv)

    def _assert_matches_fp32(self, actual, qkv, mask, cotangent, is_causal=False):
        """Both kernels are bf16, so fp32 sdpa is the only honest reference.

        Comparing the two bf16 kernels to each other would let a shared drift cancel out, and
        would also hide how much room there is: sdpa itself lands ~1.7e-3 from fp32 here.
        """
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        expected = self._out_and_grads(sdpa_attention_forward, qkv, mask, cotangent, is_causal, fp32=True)
        actual_tensors = [actual[0], *actual[1]]
        expected_tensors = [expected[0], *expected[1]]
        # bf16 keeps ~3 decimal digits. sdpa itself lands at 1.7e-3 against fp32 on all four
        # tensors; FFPA matches that on the output but runs 3-6x wider on the gradients,
        # which accumulate over every key tile through a non-deterministic atomic add.
        for name, got, want, tolerance in zip(
            ["out", "dQ", "dK", "dV"], actual_tensors, expected_tensors, [5e-3, 2e-2, 2e-2, 2e-2]
        ):
            assert torch.isfinite(got).all(), f"{name} is not finite"
            error = (got.float() - want.float()).norm() / want.float().norm()
            assert error < tolerance, f"{name} relative error {error:.2e}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FFPA needs a GPU")
    @pytest.mark.parametrize("padded", [False, True])
    def test_ffpa_matches_sdpa_forward_and_backward(self, padded):
        pytest.importorskip("ffpa_attn")
        from veomni.models.transformers.diffusion_gemma.ffpa_attention import ffpa_attention_forward

        qkv, cotangent, mask = self._attention_inputs(self.SEQ_LEN, self.SEQ_LEN, padded)
        actual = self._out_and_grads(ffpa_attention_forward, qkv, mask, cotangent)
        # A fully masked leading key tile used to come back NaN with a boolean mask.
        self._assert_matches_fp32(actual, qkv, mask, cotangent)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FFPA needs a GPU")
    def test_ffpa_applies_causality_when_there_is_no_mask(self):
        """An unpadded batch gets `attention_mask=None` and a bare `is_causal=True`.

        Nothing else carries the causal structure then, so dropping the flag would silently
        turn those batches bidirectional -- and the two libraries disagree on what the flag
        means unless the lengths match, which is why this is worth pinning down.
        """
        pytest.importorskip("ffpa_attn")
        from veomni.models.transformers.diffusion_gemma.ffpa_attention import ffpa_attention_forward

        qkv, cotangent, _ = self._attention_inputs(self.SEQ_LEN, self.SEQ_LEN, padded=False)
        actual = self._out_and_grads(ffpa_attention_forward, qkv, None, cotangent, is_causal=True)
        self._assert_matches_fp32(actual, qkv, None, cotangent, is_causal=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FFPA needs a GPU")
    def test_the_kernel_really_runs_on_the_wide_heads(self):
        """`ffpa_attn_func` short-circuits to sdpa internally, so calling it proves nothing."""
        pytest.importorskip("ffpa_attn")
        from unittest.mock import patch

        import ffpa_attn.functional as ffpa_functional

        from veomni.models.transformers.diffusion_gemma.ffpa_attention import ffpa_attention_forward

        qkv, cotangent, mask = self._attention_inputs(self.SEQ_LEN, self.SEQ_LEN, padded=True)
        with patch.object(
            ffpa_functional, "_ffpa_attn_forward_triton", wraps=ffpa_functional._ffpa_attn_forward_triton
        ) as triton_kernel:
            self._out_and_grads(ffpa_attention_forward, qkv, mask, cotangent)
        assert triton_kernel.called

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FFPA needs a GPU")
    @pytest.mark.parametrize(
        ("q_len", "kv_len", "is_causal", "reason"),
        [
            (256, SEQ_LEN, False, "the decoder's canvas is 256 and FFPA delegates below 512"),
            (SEQ_LEN, SEQ_LEN * 2, True, "sdpa aligns the causal triangle top-left, FFPA bottom-right"),
        ],
    )
    def test_unsupported_shapes_fall_back_to_sdpa_exactly(self, q_len, kv_len, is_causal, reason):
        pytest.importorskip("ffpa_attn")
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        from veomni.models.transformers.diffusion_gemma.ffpa_attention import ffpa_attention_forward

        qkv, cotangent, mask = self._attention_inputs(q_len, kv_len, padded=not is_causal)
        expected_out, expected_grads = self._out_and_grads(
            sdpa_attention_forward, qkv, mask, cotangent, is_causal=is_causal
        )
        actual_out, actual_grads = self._out_and_grads(
            ffpa_attention_forward, qkv, mask, cotangent, is_causal=is_causal
        )

        # Bit-identical output is what shows the call really went to sdpa. Its backward is
        # not reproducible run to run at this shape -- dQ moves by ~1e-3 between two
        # identical sdpa calls -- so the gradients only get a tolerance.
        assert torch.equal(actual_out, expected_out), reason
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FFPA needs a GPU")
    def test_the_whole_model_trains_the_same_with_ffpa(self):
        """End to end, including the encoder mask the model builds for a left-padded batch."""
        pytest.importorskip("ffpa_attn")
        from unittest.mock import patch

        import ffpa_attn.functional as ffpa_functional

        def loss_and_grad(attn_ffpa):
            torch.manual_seed(0)
            config = self._wide_config(attn_ffpa)
            model = DiffusionGemmaForBlockDiffusion(config).to(device="cuda", dtype=torch.bfloat16).train()
            batch = {
                k: v.to("cuda")
                for k, v in build_padded_batch(config, batch_size=1, prompt_len=self.SEQ_LEN, pad=self.PAD).items()
            }
            loss = model(**batch, use_cache=False).loss
            loss.backward()
            grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
            return loss.detach(), torch.cat(grads).norm()

        sdpa_loss, sdpa_grad_norm = loss_and_grad(attn_ffpa=False)
        # Counted, because a silent fallback would make the comparison below vacuous.
        with patch.object(
            ffpa_functional, "_ffpa_attn_forward_triton", wraps=ffpa_functional._ffpa_attn_forward_triton
        ) as triton_kernel:
            ffpa_loss, ffpa_grad_norm = loss_and_grad(attn_ffpa=True)
        # One full_attention layer, and only the encoder runs it -- the decoder's canvas is
        # shorter than FFPA's 512-query floor. Spying on the Triton entry rather than on
        # `ffpa_attn_func` matters: the latter also counts the calls it hands back to sdpa.
        assert triton_kernel.call_count == 1

        assert torch.isfinite(ffpa_loss) and torch.isfinite(ffpa_grad_norm)
        torch.testing.assert_close(ffpa_loss, sdpa_loss, rtol=2e-2, atol=1e-3)
        torch.testing.assert_close(ffpa_grad_norm, sdpa_grad_norm, rtol=5e-2, atol=1e-3)


def _prefilled_cache(model, batch):
    """Run the encoder so the decoder mask helper has a real cache to size itself against."""
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model.model.encoder(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            use_cache=True,
        )
    return outputs.past_key_values


class TestGradientCheckpointing:
    def test_both_stacks_are_recomputed(self):
        """The encoder holds the whole context, so recomputing it is where the memory is."""
        model = build_toy_model().train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        assert all(layer.gradient_checkpointing for layer in model.model.decoder.layers)
        assert all(layer.gradient_checkpointing for layer in model.model.encoder.language_model.layers)
        assert not isinstance(model.model.encoder.language_model.layers[0], DiffusionGemmaDecoderTextLayer)
        assert isinstance(model.model.encoder.language_model.layers[0], DiffusionGemmaEncoderTextLayer)

    def test_replay_does_not_append_to_the_cache_a_second_time(self):
        """Without the seal the replayed encoder would double the cache it attends over."""
        model = build_toy_model().train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        batch = build_toy_batch(model.config)
        prompt_len = batch["input_ids"].shape[1]

        outputs = model(**batch)
        cache = outputs.past_key_values
        assert cache.get_seq_length() == prompt_len

        outputs.loss.backward()
        assert cache.get_seq_length() == prompt_len

    def test_an_empty_cache_is_what_makes_the_seal_equivalent(self):
        """A sealed replay skips the concatenation, so a preloaded cache must be refused."""
        model = build_toy_model().train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        batch = build_toy_batch(model.config)
        cache = model(**batch).past_key_values

        with pytest.raises(ValueError, match="empty cache"):
            model.model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                past_key_values=cache,
            )

    def test_an_unsealable_cache_is_refused(self):
        model = build_toy_model().train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        batch = build_toy_batch(model.config)

        with pytest.raises(ValueError, match="SealableCache"):
            model.model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                past_key_values=DynamicCache(config=model.config.text_config),
            )

    def test_the_seal_is_lifted_for_the_next_canvas(self):
        """Block-autoregressive generation feeds one cache through many encoder calls."""
        model = build_toy_model().eval()
        batch = build_toy_batch(model.config)
        prompt_len = batch["input_ids"].shape[1]

        with torch.no_grad():
            cache = model.model.encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).past_key_values
            assert cache.sealed
            model.model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=torch.ones(batch["input_ids"].shape[0], 2 * prompt_len, dtype=torch.long),
                past_key_values=cache,
            )
        assert cache.get_seq_length() == 2 * prompt_len

    def test_a_sealed_update_returns_what_an_empty_cache_would_have(self):
        """Pins the upstream behaviour the seal stands in for: `update` on an empty cache
        hands back the states unchanged, so skipping it cannot change what attention reads.
        Both layer types matter -- the sliding one caches a truncated copy but still
        returns the full states."""
        config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH).text_config
        keys = torch.randn(1, 2, config.sliding_window + 3, 4)
        values = torch.randn_like(keys)

        for layer_idx in range(len(config.layer_types)):
            fresh_keys, fresh_values = SealableCache(config=config).update(keys, values, layer_idx)
            torch.testing.assert_close(fresh_keys, keys)
            torch.testing.assert_close(fresh_values, values)

            sealed = SealableCache(config=config)
            sealed.sealed = True
            sealed_keys, sealed_values = sealed.update(keys, values, layer_idx)
            assert sealed_keys is keys and sealed_values is values
            assert sealed.get_seq_length() == 0

    def test_recompute_matches_the_plain_forward(self):
        """Regression test: the stock HF layer blanks `past_key_values` under recompute,
        which would hide the prompt from the decoder entirely."""

        def run(enable):
            model = build_toy_model().train()
            if enable:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            outputs = model(**build_toy_batch(model.config))
            outputs.loss.backward()
            return outputs.loss, {n: p.grad.clone() for n, p in model.named_parameters()}

        baseline_loss, baseline_grads = run(False)
        recomputed_loss, recomputed_grads = run(True)
        torch.testing.assert_close(baseline_loss, recomputed_loss)
        for name, grad in baseline_grads.items():
            torch.testing.assert_close(grad, recomputed_grads[name], msg=lambda m, n=name: f"{n}: {m}")

    def test_reentrant_checkpointing_is_rejected(self):
        """Reentrant recompute would silently drop the encoder's only gradient path."""
        model = build_toy_model().train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        with pytest.raises(ValueError, match="non-reentrant"):
            model(**build_toy_batch(model.config))


class TestExpertParallelPlan:
    def test_plan_targets_the_deduplicated_expert_names(self):
        from veomni.distributed.utils import check_fqn_match

        model = build_toy_model()
        plan = get_parallel_plan().extra_parallel_plan["ep"]
        matched = [fqn for fqn, _ in model.named_parameters() for pattern in plan if check_fqn_match(pattern, fqn)]
        assert len(matched) == 2 * model.config.text_config.num_hidden_layers

    def test_share_expert_modules_collapses_the_pair(self):
        """Under EP the two expert modules must become one; see `share_expert_modules`."""
        from veomni.distributed.utils import set_module_from_path
        from veomni.models.transformers.diffusion_gemma.parallel_plan import share_expert_modules

        model = build_toy_model()
        num_layers = model.config.text_config.num_hidden_layers
        encoder_layer = model.model.encoder.language_model.layers[0]
        decoder_layer = model.model.decoder.layers[0]
        assert encoder_layer.experts is not decoder_layer.experts

        assert share_expert_modules(model) == num_layers
        assert encoder_layer.experts is decoder_layer.experts

        # With one module, EP's rebinding of the parameter is visible from both sides.
        fqn = "model.decoder.layers.0.experts.gate_up_proj"
        full = model.get_parameter(fqn)
        set_module_from_path(model, fqn, torch.nn.Parameter(full.data[:4].clone()))
        assert encoder_layer.experts.gate_up_proj is decoder_layer.experts.gate_up_proj
        assert encoder_layer.experts.gate_up_proj.shape[0] == 4

    def test_retie_restores_sharing_of_per_side_modules(self):
        """Attention and friends still live in separate modules, so they can still be untied."""
        model = build_toy_model()
        encoder_layer = model.model.encoder.language_model.layers[0]
        decoder_layer = model.model.decoder.layers[0]
        encoder_layer.self_attn.q_proj.weight = torch.nn.Parameter(decoder_layer.self_attn.q_proj.weight.data.clone())
        assert encoder_layer.self_attn.q_proj.weight is not decoder_layer.self_attn.q_proj.weight

        assert retie_encoder_to_decoder(model) == 1
        assert encoder_layer.self_attn.q_proj.weight is decoder_layer.self_attn.q_proj.weight

    def test_retie_also_covers_the_final_norm(self):
        """`norm.weight` is tied outside the layer stacks, so it needs the same walk."""
        model = build_toy_model()
        encoder = model.model.encoder.language_model
        encoder.norm.weight = torch.nn.Parameter(model.model.decoder.norm.weight.data.clone())
        assert encoder.norm.weight is not model.model.decoder.norm.weight

        assert retie_encoder_to_decoder(model) == 1
        assert encoder.norm.weight is model.model.decoder.norm.weight

    def test_encoder_layers_are_not_a_separate_fsdp_unit(self):
        """Both layer types in `_no_split_modules` makes FSDP2 backward fail; see the class comment."""
        assert "DiffusionGemmaEncoderTextLayer" not in DiffusionGemmaForBlockDiffusion._no_split_modules
        assert "DiffusionGemmaDecoderTextLayer" in DiffusionGemmaForBlockDiffusion._no_split_modules


class _StubChatTemplate:
    """Minimal stand-in: two prompt tokens, then the supervised answer."""

    def __init__(self, answer_len=6):
        self.answer_len = answer_len

    def encode_messages(self, messages, max_seq_len=None):
        prompt = [5, 6, 7, 8]
        answer = list(range(20, 20 + self.answer_len))
        return {
            "input_ids": prompt + answer,
            "labels": [IGNORE_INDEX] * len(prompt) + answer,
        }


class _StubMultiTurnChatTemplate:
    """Two assistant turns with a masked-out user turn between them."""

    def encode_messages(self, messages, max_seq_len=None):
        return {
            "input_ids": [5, 6] + [20, 21] + [7, 8] + [22, 23],
            "labels": [IGNORE_INDEX] * 2 + [20, 21] + [IGNORE_INDEX] * 2 + [22, 23],
        }


class _StubUnsupervisedChatTemplate:
    """A sample the template masked out entirely, e.g. a conversation with no answer."""

    def encode_messages(self, messages, max_seq_len=None):
        return {"input_ids": [5, 6, 7, 8], "labels": [IGNORE_INDEX] * 4}


class _SuffixTruncatingStubChatTemplate(_StubChatTemplate):
    """Match production templates, which keep the tail of an oversized chat."""

    def encode_messages(self, messages, max_seq_len=None):
        tokenized = super().encode_messages(messages, max_seq_len=max_seq_len)
        if max_seq_len is None:
            return tokenized
        return {key: value[-max_seq_len:] for key, value in tokenized.items()}


_EOS = 1


def _eos_pad(answer, canvas_length, eos=_EOS):
    """Official last-canvas fill: EOS out to a multiple of ``canvas_length``."""
    n = len(answer)
    if n == 0:
        return list(answer)
    target = -(-n // canvas_length) * canvas_length
    return list(answer) + [eos] * (target - n)


class TestDataTransform:
    transform = staticmethod(DATA_TRANSFORM_REGISTRY["diffusion_gemma"])

    def _run(self, answer_len=6, canvas_length=16, **kwargs):
        options = {
            "chat_template": _StubChatTemplate(answer_len),
            "max_seq_len": 32,
            "canvas_length": canvas_length,
            "vocab_size": 1024,
            "pad_token_id": 0,
        }
        options.update(kwargs)
        return self.transform({"messages": []}, **options)[0]

    def test_splits_prompt_from_canvas(self):
        sample = self._run(answer_len=6)
        # The one canvas still lives in the encoder (clean) so the AR term can cover it;
        # the decoder sees only the prompt. A short answer is EOS-filled to canvas_length.
        answer = _eos_pad(list(range(20, 26)), 16)
        assert sample["input_ids"].tolist() == [5, 6, 7, 8] + answer
        assert sample["decoder_context_mask"].tolist() == [1, 1, 1, 1] + [0] * 16
        assert sample["encoder_prompt_mask"].tolist() == [True] * 4 + [False] * 16
        assert sample["decoder_input_ids"].numel() == 16
        assert sample["labels"][:6].tolist() == list(range(20, 26))
        assert (sample["labels"][6:] == _EOS).all()

    def test_long_answer_is_capped_without_tail_truncating_away_the_prompt(self):
        sample = self._run(
            chat_template=_SuffixTruncatingStubChatTemplate(answer_len=100),
            answer_len=100,
            max_seq_len=32,
            max_canvases=1,
        )

        assert sample["input_ids"].tolist() == [5, 6, 7, 8] + list(range(20, 36))
        assert sample["encoder_prompt_mask"].tolist() == [True] * 4 + [False] * 16
        assert sample["decoder_context_mask"].tolist() == [1] * 4 + [0] * 16

    def test_canvas_start_zero_matches_the_default(self):
        # No reseeding between the two: the transform derives its own generator, so two
        # calls on the same sample and epoch are identical by construction.
        default = self._run(answer_len=6)
        explicit = self._run(answer_len=6, canvas_start=0)
        assert default["input_ids"].tolist() == explicit["input_ids"].tolist()
        assert default["labels"].tolist() == explicit["labels"].tolist()
        assert default["decoder_context_mask"].tolist() == explicit["decoder_context_mask"].tolist()

    def test_canvas_start_skips_leading_answer_tokens(self):
        """The first n answer tokens become encoder context and are never the noisy canvas."""
        sample = self._run(answer_len=6, canvas_start=2)
        remaining = _eos_pad(list(range(22, 26)), 16)
        assert sample["input_ids"].tolist() == [5, 6, 7, 8, 20, 21] + remaining
        assert sample["decoder_context_mask"].tolist() == [1] * 6 + [0] * 16
        assert sample["encoder_prompt_mask"].tolist() == [True] * 6 + [False] * 16
        assert sample["labels"][:4].tolist() == list(range(22, 26))
        assert (sample["labels"][4:] == _EOS).all()
        torch.testing.assert_close(sample["loss_weight"].sum(), torch.tensor(1.0))
        assert (sample["loss_weight"] > 0).all()

    def test_canvas_start_past_the_answer_is_zero_weight(self):
        sample = self._run(answer_len=6, canvas_start=100)
        assert float(sample["loss_weight"].sum()) == 0.0
        assert (sample["labels"] == IGNORE_INDEX).all()

    def test_negative_canvas_start_is_rejected(self):
        with pytest.raises(ValueError, match="canvas_start"):
            self._run(answer_len=6, canvas_start=-1)

    def test_loss_weight_averages_over_the_answer(self):
        sample = self._run(answer_len=8)
        torch.testing.assert_close(sample["loss_weight"].sum(), torch.tensor(1.0))
        # 8 answer tokens plus 8 supervised EOS; Eq. 13 is 1/C over the full canvas.
        torch.testing.assert_close(sample["loss_weight"], torch.full((16,), 1.0 / 16))

    def test_encoder_gets_every_block_and_the_mask_hides_the_draw(self):
        """The encoder always sees prompt + every canvas; whichever one is drawn, the
        decoder mask stops at that block so the canvas cannot read the clean answer.

        The draw is keyed on ``(seed, epoch, sample)``, so it is the epoch that moves
        it -- one sample re-processed inside an epoch is meant to be reproducible."""
        prompt, answer = [5, 6, 7, 8], list(range(20, 60))
        filled = _eos_pad(answer, 16)
        drawn = set()
        for epoch in range(40):
            sample = self._run(answer_len=len(answer), canvas_length=16, max_seq_len=64, corruption_epoch=epoch)
            assert sample["input_ids"].tolist() == prompt + filled
            block, remainder = divmod(int(sample["decoder_context_mask"].sum()) - len(prompt), 16)
            assert remainder == 0
            assert sample["decoder_context_mask"].tolist() == [1] * (len(prompt) + block * 16) + [0] * (
                len(filled) - block * 16
            )
            supervised = sample["labels"][sample["labels"] != IGNORE_INDEX].tolist()
            assert supervised == filled[block * 16 : (block + 1) * 16]
            drawn.add(block)
        assert drawn == {0, 1, 2}

    def test_a_single_block_answer_still_puts_the_answer_in_the_encoder(self):
        """Short answers used to leave the encoder as a bare prompt. The AR term needs
        the canvas tokens in the encoder even when there is only one of them."""
        filled = _eos_pad(list(range(20, 26)), 16)
        for epoch in range(10):
            sample = self._run(answer_len=6, canvas_length=16, corruption_epoch=epoch)
            assert sample["input_ids"].tolist() == [5, 6, 7, 8] + filled
            assert sample["decoder_context_mask"].tolist() == [1, 1, 1, 1] + [0] * 16

    def test_max_canvases_caps_how_far_into_an_answer_training_reaches(self):
        # Two canvases of 16 plus the 4-token prompt need 36 tokens of encoder budget.
        # Varying the epoch varies the block draw, which is what could move the budget.
        sizes = {
            self._run(answer_len=40, canvas_length=16, max_canvases=2, max_seq_len=64, corruption_epoch=epoch)[
                "input_ids"
            ].numel()
            for epoch in range(30)
        }
        assert sizes == {36}

    def test_the_prompt_budget_does_not_move_with_the_block_draw(self):
        """A prompt whose surviving length tracked the draw would make one sample teach a
        different instruction from epoch to epoch, and a deep draw could erase it."""
        answer = list(range(20, 68))
        drawn = set()
        for epoch in range(30):
            # 50 = 2 prompt tokens + 3 * 16 canvases. The prompt is truncated against
            # the full answer, not against whichever block came up.
            sample = self._run(
                answer_len=len(answer), canvas_length=16, max_canvases=3, max_seq_len=50, corruption_epoch=epoch
            )
            block, remainder = divmod(int(sample["decoder_context_mask"].sum()) - 2, 16)
            assert remainder == 0
            assert sample["input_ids"].tolist() == [7, 8] + answer
            assert sample["decoder_context_mask"].tolist() == [1] * (2 + block * 16) + [0] * (len(answer) - block * 16)
            drawn.add(block)
        assert drawn == {0, 1, 2}

    def test_a_short_answer_does_not_pay_for_canvases_it_lacks(self):
        """Reserving against `max_canvases` instead of the sample's own block count would
        truncate the prompt of every short answer to make room for nothing."""
        sample = self._run(answer_len=6, canvas_length=16, max_canvases=8, max_seq_len=32)
        assert sample["input_ids"].tolist() == [5, 6, 7, 8] + _eos_pad(list(range(20, 26)), 16)

    def test_the_weight_is_one_whatever_the_answer_spans(self):
        """Eq. 13 normalizes the drawn canvas by ``1/C`` alone. Scaling by the block
        count instead -- which this transform used to do, reading the multi-canvas
        objective of Eq. 11 into a single-canvas draw -- makes a long answer count for
        up to ``max_canvases`` short ones and drives the gradient norm into the
        hundreds, where clipping decides the step size."""
        drawn = set()
        for epoch in range(30):
            # 40 tokens over 16-wide canvases is 3 blocks; the last is 8 real + 8 EOS.
            sample = self._run(answer_len=40, canvas_length=16, max_seq_len=64, corruption_epoch=epoch)
            torch.testing.assert_close(sample["loss_weight"].sum(), torch.tensor(1.0))
            drawn.add((int(sample["labels"][0]) - 20) // 16)
        # Including the short trailing block, whose 16 positions (8 answer + 8 EOS) carry 1/16 each.
        assert drawn == {0, 1, 2}
        single = self._run(answer_len=8, canvas_length=16, max_seq_len=64)
        torch.testing.assert_close(single["loss_weight"].sum(), torch.tensor(1.0))

    def test_corruption_uses_random_tokens_not_a_mask_token(self):
        """DiffusionGemma renoises with uniform tokens; there is no absorbing mask state."""
        corrupted_values = set()
        for epoch in range(20):
            sample = self._run(answer_len=16, canvas_length=16, noise_eps=1.0, corruption_epoch=epoch)
            differs = sample["decoder_input_ids"] != sample["labels"]
            corrupted_values.update(sample["decoder_input_ids"][differs].tolist())
        # A single absorbing token would give exactly one distinct replacement value.
        assert len(corrupted_values) > 1

    def test_noise_eps_one_corrupts_everything(self):
        sample = self._run(answer_len=16, canvas_length=16, noise_eps=1.0)
        # With t == 1 every answer position is redrawn, so labels stay intact while the
        # canvas is (almost surely) different.
        assert (sample["labels"] != IGNORE_INDEX).all()

    def test_vocab_size_is_required(self):
        with pytest.raises(ValueError, match="vocab_size"):
            self.transform({"messages": []}, chat_template=_StubChatTemplate(), vocab_size=None)

    def test_canvas_tail_is_supervised_eos(self):
        """Official CanvasChunker fills the last canvas with EOS; those positions train."""
        sample = self._run(answer_len=4, canvas_length=16)
        assert sample["input_ids"].tolist() == [5, 6, 7, 8] + _eos_pad(list(range(20, 24)), 16)
        assert (sample["labels"][4:] == _EOS).all()
        assert (sample["labels"] != IGNORE_INDEX).all()
        torch.testing.assert_close(sample["loss_weight"], torch.full((16,), 1.0 / 16))

    def test_multi_turn_keeps_the_templates_label_mask(self):
        """A later user turn sits inside the canvas but must not become a training target."""
        sample = self.transform(
            {"messages": []},
            chat_template=_StubMultiTurnChatTemplate(),
            max_seq_len=32,
            canvas_length=8,
            vocab_size=1024,
        )[0]
        assert sample["input_ids"].tolist() == [5, 6, 20, 21, 7, 8, 22, 23, _EOS, _EOS]
        assert sample["decoder_context_mask"].tolist() == [1, 1] + [0] * 8
        # Canvas covers [answer1, user2, answer2, eos, eos]; only the assistant turns
        # (and the EOS fill) are supervised -- the user turn keeps IGNORE_INDEX.
        assert sample["labels"][:6].tolist() == [20, 21, IGNORE_INDEX, IGNORE_INDEX, 22, 23]
        assert sample["labels"][6:].tolist() == [_EOS, _EOS]
        assert (sample["loss_weight"][2:4] == 0).all()
        torch.testing.assert_close(sample["loss_weight"].sum(), torch.tensor(1.0))

    def test_compute_all_canvas_df_loss_averages_every_encoder_block(self):
        """Every canvas that fits is denoised; each 1/C mean is then divided by K."""
        sample = self._run(
            answer_len=40,
            canvas_length=16,
            max_seq_len=64,
            compute_all_canvas_df_loss=True,
        )
        filled = _eos_pad(list(range(20, 60)), 16)
        prompt = [5, 6, 7, 8]
        assert sample["input_ids"].tolist() == prompt + filled
        assert sample["decoder_input_ids"].shape == (3, 16)
        assert sample["labels"].shape == (3, 16)
        assert sample["loss_weight"].shape == (3, 16)
        assert sample["decoder_context_mask"].shape == (3, len(prompt) + len(filled))
        assert sample["encoder_prompt_mask"].tolist() == [True] * 4 + [False] * len(filled)
        for block in range(3):
            assert sample["labels"][block].tolist() == filled[block * 16 : (block + 1) * 16]
            assert sample["decoder_context_mask"][block].tolist() == [1] * (len(prompt) + block * 16) + [0] * (
                len(filled) - block * 16
            )
            torch.testing.assert_close(sample["loss_weight"][block], torch.full((16,), 1.0 / (16 * 3)))
        torch.testing.assert_close(sample["loss_weight"].sum(), torch.tensor(1.0))

    def test_compute_all_canvas_df_loss_does_not_draw_a_single_prefix(self):
        """A leftover Eq. 13 ``prefix_len`` from the encoder layout would make one
        canvas see the wrong cache (or leak its own clean tokens) depending on the draw."""
        masks = []
        for seed in range(8):
            torch.manual_seed(seed)
            sample = self._run(
                answer_len=40,
                canvas_length=16,
                max_seq_len=64,
                compute_all_canvas_df_loss=True,
            )
            masks.append(sample["decoder_context_mask"].clone())
        assert all(torch.equal(mask, masks[0]) for mask in masks[1:])

    def test_compute_all_canvas_df_loss_is_a_no_op_on_a_single_canvas(self):
        sample = self._run(answer_len=6, compute_all_canvas_df_loss=True)
        assert sample["decoder_input_ids"].ndim == 1
        assert sample["decoder_context_mask"].ndim == 1
        torch.testing.assert_close(sample["loss_weight"].sum(), torch.tensor(1.0))

    def test_nothing_supervised_degrades_to_a_zero_weight_sample(self):
        samples = self.transform(
            {"messages": []},
            chat_template=_StubUnsupervisedChatTemplate(),
            max_seq_len=32,
            canvas_length=8,
            vocab_size=1024,
        )
        assert len(samples) == 1
        assert (samples[0]["labels"] == IGNORE_INDEX).all()
        assert float(samples[0]["loss_weight"].sum()) == 0.0


class TestCollator:
    def test_pads_prompts_and_stacks_canvases(self):
        collator = DiffusionGemmaCollator(pad_token_id=0)
        features = [
            {
                "input_ids": torch.arange(1, 4),
                "decoder_context_mask": torch.ones(3, dtype=torch.long),
                "decoder_input_ids": torch.zeros(8, dtype=torch.long),
                "labels": torch.zeros(8, dtype=torch.long),
                "loss_weight": torch.full((8,), 0.125),
            },
            {
                "input_ids": torch.arange(1, 7),
                "decoder_context_mask": torch.ones(6, dtype=torch.long),
                "decoder_input_ids": torch.ones(8, dtype=torch.long),
                "labels": torch.ones(8, dtype=torch.long),
                "loss_weight": torch.full((8,), 0.125),
            },
        ]
        batch = collator(features)
        assert batch["input_ids"].shape == (2, 6)
        # Left padding: the last real token must sit right before the canvas.
        assert batch["input_ids"][0].tolist() == [0, 0, 0, 1, 2, 3]
        assert batch["attention_mask"][0].tolist() == [0, 0, 0, 1, 1, 1]
        assert batch["decoder_input_ids"].shape == (2, 8)
        # The decoder mask spans the encoder cache plus the canvas.
        assert batch["decoder_attention_mask"].shape == (2, 6 + 8)
        assert batch["decoder_attention_mask"][0].tolist() == [0, 0, 0, 1, 1, 1] + [1] * 8

    def test_decoder_context_mask_is_required(self):
        """An all-ones default would leak the clean canvas into the decoder."""
        collator = DiffusionGemmaCollator(pad_token_id=0)
        feature = {
            "input_ids": torch.arange(1, 4),
            "decoder_input_ids": torch.zeros(4, dtype=torch.long),
            "labels": torch.zeros(4, dtype=torch.long),
            "loss_weight": torch.ones(4),
        }
        with pytest.raises(KeyError, match="decoder_context_mask"):
            collator([feature])

    def test_encoder_prompt_mask_is_left_padded(self):
        collator = DiffusionGemmaCollator(pad_token_id=0)
        features = [
            {
                "input_ids": torch.arange(1, 4),
                "decoder_context_mask": torch.ones(3, dtype=torch.long),
                "encoder_prompt_mask": torch.tensor([True, True, False]),
                "decoder_input_ids": torch.zeros(4, dtype=torch.long),
                "labels": torch.zeros(4, dtype=torch.long),
                "loss_weight": torch.ones(4),
            },
            {
                "input_ids": torch.arange(1, 7),
                "decoder_context_mask": torch.ones(6, dtype=torch.long),
                "encoder_prompt_mask": torch.tensor([True, True, True, False, False, False]),
                "decoder_input_ids": torch.zeros(4, dtype=torch.long),
                "labels": torch.zeros(4, dtype=torch.long),
                "loss_weight": torch.ones(4),
            },
        ]
        batch = collator(features)
        assert batch["encoder_prompt_mask"][0].tolist() == [False, False, False, True, True, False]
        assert batch["encoder_prompt_mask"][1].tolist() == [True, True, True, False, False, False]

    def test_stacked_canvases_keep_one_encoder_row(self):
        """Repeating the encoder K times is what OOMed the 26B all-canvas run."""
        collator = DiffusionGemmaCollator(pad_token_id=0)
        feature = {
            "input_ids": torch.arange(1, 7),
            "decoder_context_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]]),
            "encoder_prompt_mask": torch.tensor([True, True, True, False, False, False]),
            "decoder_input_ids": torch.stack([torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)]),
            "labels": torch.stack([torch.full((4,), 7, dtype=torch.long), torch.full((4,), 8, dtype=torch.long)]),
            "loss_weight": torch.stack([torch.full((4,), 0.125), torch.full((4,), 0.125)]),
        }
        batch = collator([feature])
        assert batch["input_ids"].shape == (1, 6)
        assert batch["decoder_input_ids"].shape == (1, 2, 4)
        assert batch["decoder_input_ids"][0].tolist() == [[0, 0, 0, 0], [1, 1, 1, 1]]
        assert batch["decoder_attention_mask"].shape == (1, 2, 6 + 4)
        assert batch["decoder_attention_mask"][0, 0].tolist() == [1, 1, 1, 0, 0, 0] + [1] * 4
        assert batch["decoder_attention_mask"][0, 1].tolist() == [1, 1, 1, 1, 1, 1] + [1] * 4
        assert int(batch["num_supervised_examples"]) == 1

    def test_mixed_stacked_and_single_canvases_pad_k_without_repeating_the_encoder(self):
        collator = DiffusionGemmaCollator(pad_token_id=0)
        stacked = {
            "input_ids": torch.arange(1, 5),
            "decoder_context_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
            "decoder_input_ids": torch.zeros(2, 3, dtype=torch.long),
            "labels": torch.ones(2, 3, dtype=torch.long),
            "loss_weight": torch.full((2, 3), 1.0 / 6),
        }
        single = {
            "input_ids": torch.arange(1, 3),
            "decoder_context_mask": torch.tensor([1, 0]),
            "decoder_input_ids": torch.ones(3, dtype=torch.long),
            "labels": torch.full((3,), 9, dtype=torch.long),
            "loss_weight": torch.full((3,), 1.0 / 3),
        }
        batch = collator([stacked, single])
        assert batch["input_ids"].shape == (2, 4)
        assert batch["decoder_input_ids"].shape == (2, 2, 3)
        assert batch["labels"][1, 1].tolist() == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
        assert float(batch["loss_weight"][1, 1].sum()) == 0.0
        assert int(batch["num_supervised_examples"]) == 2

    def test_example_sample_count_uses_the_collator_tally(self):
        from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer

        trainer = DiffusionGemmaTrainer.__new__(DiffusionGemmaTrainer)
        stacked = {
            "loss_weight": torch.ones(1, 3, 4),
            "num_supervised_examples": torch.tensor(1),
        }
        assert trainer._example_sample_count(stacked) == 1
        assert trainer._example_sample_count({"loss_weight": torch.ones(1, 3, 4)}) == 1
        assert trainer._example_sample_count({"loss_weight": torch.ones(3, 4)}) == 3

    def test_pad_decoder_canvases_aligns_a_short_rank_to_the_step_max(self):
        from veomni.trainer.diffusion_gemma_trainer import pad_decoder_canvases

        single = {
            "decoder_input_ids": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.full((1, 4), 7, dtype=torch.long),
            "loss_weight": torch.full((1, 4), 0.25),
            "decoder_attention_mask": torch.ones(1, 6, dtype=torch.bool),
        }
        padded = pad_decoder_canvases(single, 3)
        assert padded["decoder_input_ids"].shape == (1, 3, 4)
        assert padded["labels"][0, 1].tolist() == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
        assert float(padded["loss_weight"][0, 1].sum()) == 0.0
        assert padded["decoder_attention_mask"].shape == (1, 3, 6)
        assert pad_decoder_canvases({"decoder_input_ids": torch.ones(1, 4, dtype=torch.long)}, 1)[
            "decoder_input_ids"
        ].shape == (1, 4)

    def test_decoder_context_mask_hides_future_canvases_after_left_padding(self):
        collator = DiffusionGemmaCollator(pad_token_id=0)
        features = [
            {
                "input_ids": torch.arange(1, 7),
                "decoder_context_mask": torch.tensor([1, 1, 1, 0, 0, 0]),
                "decoder_input_ids": torch.zeros(4, dtype=torch.long),
                "labels": torch.zeros(4, dtype=torch.long),
                "loss_weight": torch.ones(4),
            },
            {
                "input_ids": torch.arange(1, 4),
                "decoder_context_mask": torch.tensor([1, 1, 0]),
                "decoder_input_ids": torch.zeros(4, dtype=torch.long),
                "labels": torch.zeros(4, dtype=torch.long),
                "loss_weight": torch.ones(4),
            },
        ]
        batch = collator(features)
        # Left pad, then canvas ones. Future zeros in the encoder half must survive.
        assert batch["decoder_attention_mask"][0].tolist() == [1, 1, 1, 0, 0, 0] + [1] * 4
        assert batch["decoder_attention_mask"][1].tolist() == [0, 0, 0, 1, 1, 0] + [1] * 4

    def test_batch_feeds_the_model(self):
        model = build_toy_model().train()
        canvas_length = model.config.canvas_length
        collator = DiffusionGemmaCollator(pad_token_id=0)
        features = []
        for prompt_len in (5, 9):
            ids = torch.randint(3, 1024, (prompt_len,))
            features.append(
                {
                    "input_ids": ids,
                    "decoder_context_mask": torch.ones_like(ids),
                    "decoder_input_ids": torch.randint(3, 1024, (canvas_length,)),
                    "labels": torch.randint(3, 1024, (canvas_length,)),
                    "loss_weight": torch.full((canvas_length,), 1.0 / canvas_length),
                }
            )
        outputs = model(**collator(features), use_cache=False)
        assert outputs.loss.isfinite()

    def test_prompt_padding_does_not_change_the_answer(self):
        """Padding one sample must not perturb the other's logits."""
        model = build_toy_model().eval()
        canvas_length = model.config.canvas_length
        torch.manual_seed(3)
        short_ids = torch.randint(3, 1024, (5,))
        short = {
            "input_ids": short_ids,
            "decoder_context_mask": torch.ones_like(short_ids),
            "decoder_input_ids": torch.randint(3, 1024, (canvas_length,)),
            "labels": torch.full((canvas_length,), IGNORE_INDEX),
            "loss_weight": torch.zeros(canvas_length),
        }
        long_ids = torch.randint(3, 1024, (11,))
        long = {
            "input_ids": long_ids,
            "decoder_context_mask": torch.ones_like(long_ids),
            "decoder_input_ids": torch.randint(3, 1024, (canvas_length,)),
            "labels": torch.full((canvas_length,), IGNORE_INDEX),
            "loss_weight": torch.zeros(canvas_length),
        }
        collator = DiffusionGemmaCollator(pad_token_id=0)
        with torch.no_grad():
            padded = model(**collator([short, long])).logits[0]
            alone = model(**collator([short])).logits[0]
        torch.testing.assert_close(padded, alone, rtol=2e-4, atol=2e-4)


MM_TOY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "toy_config", "diffusion_gemma_mm_toy")
# The vision pooler folds `pooling_kernel_size ** 2` patches into one soft token, so the
# patch count is not free: 36 patches over a 3x3 kernel give the 4 soft tokens the toy
# config advertises (the real model pools 2520 patches into 280).
MM_TOY_PATCH_GRID = 6


def build_toy_mm_model(dtype=torch.float32, seed=0):
    config = DiffusionGemmaConfig.from_pretrained(MM_TOY_CONFIG_PATH)
    config._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    return DiffusionGemmaForBlockDiffusion(config).to(dtype)


def build_toy_image_inputs(config, num_images):
    grid = MM_TOY_PATCH_GRID
    num_patches = grid * grid
    patch_dim = config.vision_config.patch_size**2 * 3
    coords = torch.stack(torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij"), dim=-1)
    return {
        "pixel_values": torch.randn(num_images, num_patches, patch_dim),
        "image_position_ids": coords.reshape(1, num_patches, 2).expand(num_images, num_patches, 2).contiguous(),
    }


def build_toy_mm_sample(config, num_images=1, prompt_len=12):
    """A prompt whose leading tokens are image placeholders, plus a plain-text canvas."""
    n_image_tokens = config.vision_soft_tokens_per_image * num_images
    assert prompt_len > n_image_tokens
    input_ids = torch.randint(3, 1000, (prompt_len,))
    mm_token_type_ids = torch.zeros(prompt_len, dtype=torch.long)
    input_ids[:n_image_tokens] = config.image_token_id
    mm_token_type_ids[:n_image_tokens] = 1
    canvas_length = config.canvas_length
    sample = {
        "input_ids": input_ids,
        "decoder_context_mask": torch.ones(prompt_len, dtype=torch.long),
        "mm_token_type_ids": mm_token_type_ids,
        "decoder_input_ids": torch.randint(3, 1000, (canvas_length,)),
        "labels": torch.randint(3, 1000, (canvas_length,)),
        "loss_weight": torch.full((canvas_length,), 1.0 / canvas_length),
    }
    if num_images:
        sample.update(build_toy_image_inputs(config, num_images))
    return sample


class TestMultimodalModel:
    def test_vision_tower_is_built_only_with_a_vision_config(self):
        assert build_toy_model().model.encoder.vision_tower is None
        assert build_toy_mm_model().model.encoder.vision_tower is not None

    def test_forward_consumes_images(self):
        model = build_toy_mm_model().eval()
        torch.manual_seed(1)
        batch = DiffusionGemmaCollator(pad_token_id=0)([build_toy_mm_sample(model.config)])
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
        assert outputs.logits.shape == (1, model.config.canvas_length, model.config.text_config.vocab_size)
        assert outputs.loss.isfinite()

    def test_images_change_the_prediction(self):
        """Guards against the vision output being computed and then silently dropped."""
        model = build_toy_mm_model().eval()
        torch.manual_seed(2)
        sample = build_toy_mm_sample(model.config)
        batch = DiffusionGemmaCollator(pad_token_id=0)([sample])
        other = dict(batch)
        other["pixel_values"] = torch.randn_like(batch["pixel_values"])
        with torch.no_grad():
            first = model(**batch, use_cache=False).logits
            second = model(**other, use_cache=False).logits
        assert not torch.allclose(first, second)

    def test_vision_inputs_do_not_reach_the_decoder(self):
        """`kwargs` fans out to both halves upstream; the decoder has no vision path."""
        model = build_toy_mm_model().eval()
        seen = {}

        def hook(_module, args, kwargs):
            seen.update(kwargs)

        model.model.decoder.register_forward_pre_hook(hook, with_kwargs=True)
        torch.manual_seed(4)
        batch = DiffusionGemmaCollator(pad_token_id=0)([build_toy_mm_sample(model.config)])
        with torch.no_grad():
            model(**batch, use_cache=False)
        assert not {"pixel_values", "image_position_ids", "mm_token_type_ids"} & set(seen)

    def test_gradients_reach_the_vision_tower(self):
        model = build_toy_mm_model().train()
        torch.manual_seed(5)
        batch = DiffusionGemmaCollator(pad_token_id=0)([build_toy_mm_sample(model.config)])
        model(**batch, use_cache=False).loss.backward()
        grads = [p.grad for p in model.model.encoder.vision_tower.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)

    def test_fp32_standardization_buffers_do_not_break_the_projection(self):
        """Reproduces the dtype split FSDP2 mixed precision creates in the vision path.

        `Gemma4VisionModel` ends on `(hidden_states - std_bias) * std_scale`, and those are
        buffers. FSDP2 casts parameters to bf16 but leaves buffers in fp32, so the tower's
        output gets promoted back to fp32 while the projection weight stays bf16. Only the
        real 8-GPU run showed this, hence pinning it down here. Newer transformers casts
        the standardized features back to the working dtype itself; the projection still
        has to hold when a tower does not.
        """
        model = build_toy_mm_model(dtype=torch.bfloat16).eval()
        vision_tower = model.model.encoder.vision_tower
        assert vision_tower.config.standardize, "the toy config must standardize for this to bite"
        vision_tower.std_bias = vision_tower.std_bias.float()
        vision_tower.std_scale = vision_tower.std_scale.float()

        torch.manual_seed(6)
        batch = DiffusionGemmaCollator(pad_token_id=0)([build_toy_mm_sample(model.config)])
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            features = model.model.encoder.get_image_features(
                pixel_values=batch["pixel_values"],
                image_position_ids=batch["image_position_ids"],
            )
        assert features.pooler_output.dtype == torch.bfloat16

        # Before the fix this raised "expected mat1 and mat2 to have the same dtype".
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
        assert outputs.loss.isfinite()


class TestMultimodalCollator:
    def test_concatenates_images_and_left_pads_the_modality_tags(self):
        model_config = DiffusionGemmaConfig.from_pretrained(MM_TOY_CONFIG_PATH)
        torch.manual_seed(6)
        one = build_toy_mm_sample(model_config, num_images=1, prompt_len=12)
        two = build_toy_mm_sample(model_config, num_images=2, prompt_len=20)
        batch = DiffusionGemmaCollator(pad_token_id=0)([one, two])

        # Images are stacked along the image dimension, needing no padding: the processor
        # emits a fixed patch count per image whatever the source resolution.
        assert batch["pixel_values"].shape[0] == 3
        assert batch["image_position_ids"].shape[0] == 3
        # The shorter prompt is left-padded, and the padding counts as text.
        assert batch["mm_token_type_ids"].shape == (2, 20)
        assert batch["mm_token_type_ids"][0][:8].tolist() == [0] * 8
        assert batch["mm_token_type_ids"][0][8:12].tolist() == [1] * 4
        assert batch["attention_mask"][0][:8].tolist() == [0] * 8

    def test_rejects_a_partially_tagged_batch(self):
        model_config = DiffusionGemmaConfig.from_pretrained(MM_TOY_CONFIG_PATH)
        torch.manual_seed(7)
        with_image = build_toy_mm_sample(model_config, num_images=1)
        text_only = build_toy_mm_sample(model_config, num_images=1)
        del text_only["mm_token_type_ids"]
        with pytest.raises(ValueError, match="mm_token_type_ids"):
            DiffusionGemmaCollator(pad_token_id=0)([with_image, text_only])


class _StubProcessor:
    """Stands in for `Gemma4Processor`, whose real chat template needs the 47GB repo.

    Mimics the only property the transform relies on: encoding the conversation up to
    the last user turn yields a strict prefix of encoding the whole conversation.
    """

    prompt_ids = [5, 6, 7, 8]
    image_ids = [90, 91]

    def __init__(self, answer_len=6, append_only=True):
        self.answer_len = answer_len
        self.append_only = append_only

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        num_images = sum(part.get("type") == "image" for message in messages for part in message["content"])
        prompt = self.image_ids * num_images + self.prompt_ids
        if not self.append_only and not add_generation_prompt:
            # The full conversation reorders what the prompt-only pass produced, so the
            # answer can no longer be recovered as a suffix.
            prompt = prompt[::-1]
        out = {
            "input_ids": torch.tensor([prompt], dtype=torch.long),
            "attention_mask": torch.ones(1, len(prompt), dtype=torch.long),
            "mm_token_type_ids": torch.tensor(
                [[1] * len(self.image_ids) * num_images + [0] * len(self.prompt_ids)], dtype=torch.long
            ),
        }
        if num_images:
            out["pixel_values"] = torch.randn(num_images, 4, 3)
            out["image_position_ids"] = torch.zeros(num_images, 4, 2, dtype=torch.long)
        if not add_generation_prompt:
            answer = list(range(20, 20 + self.answer_len))
            out["input_ids"] = torch.tensor([prompt + answer], dtype=torch.long)
            out["attention_mask"] = torch.ones(1, len(prompt) + len(answer), dtype=torch.long)
            out["mm_token_type_ids"] = torch.cat(
                [out["mm_token_type_ids"], torch.zeros(1, len(answer), dtype=torch.long)], dim=-1
            )
        return out


class TestMultimodalDataTransform:
    transform = staticmethod(DATA_TRANSFORM_REGISTRY["diffusion_gemma_mm"])

    def _run(self, num_images=1, canvas_length=16, **kwargs):
        from PIL import Image

        images = [Image.new("RGB", (8, 8))] * num_images
        example = {
            "messages": [
                {"from": "human", "value": "<image>" * num_images + "describe"},
                {"from": "gpt", "value": "an answer"},
            ],
            "images": images,
        }
        options = {
            "processor": _StubProcessor(),
            "max_seq_len": 32,
            "canvas_length": canvas_length,
            "vocab_size": 1024,
            "pad_token_id": 0,
        }
        options.update(kwargs)
        return self.transform(example, **options)

    def test_prompt_keeps_images_and_canvas_stays_text(self):
        sample = self._run(num_images=1)[0]
        prompt = _StubProcessor.image_ids + _StubProcessor.prompt_ids
        answer = _eos_pad(list(range(20, 26)), 16)
        assert sample["input_ids"].tolist() == prompt + answer
        assert sample["mm_token_type_ids"].tolist() == [1, 1, 0, 0, 0, 0] + [0] * len(answer)
        assert sample["decoder_context_mask"].tolist() == [1] * len(prompt) + [0] * len(answer)
        assert sample["encoder_prompt_mask"].tolist() == [True] * len(prompt) + [False] * len(answer)
        assert sample["pixel_values"].shape[0] == 1
        # The canvas is the answer only -- images never enter it. Short answers are EOS-filled.
        assert sample["labels"][:6].tolist() == list(range(20, 26))
        assert (sample["labels"][6:] == _EOS).all()

    def test_canvas_start_commits_leading_answer_tokens_to_the_encoder(self):
        sample = self._run(num_images=1, canvas_start=2)[0]
        prompt = _StubProcessor.image_ids + _StubProcessor.prompt_ids
        remaining = _eos_pad(list(range(22, 26)), 16)
        assert sample["input_ids"].tolist() == prompt + list(range(20, 22)) + remaining
        assert sample["decoder_context_mask"].tolist() == [1] * (len(prompt) + 2) + [0] * 16
        assert sample["encoder_prompt_mask"].tolist() == [True] * (len(prompt) + 2) + [False] * 16
        assert sample["labels"][:4].tolist() == list(range(22, 26))
        assert (sample["labels"][4:] == _EOS).all()
        assert sample["mm_token_type_ids"].tolist() == [1, 1, 0, 0, 0, 0] + [0] * (2 + len(remaining))

    def test_text_only_sample_emits_no_image_tensors(self):
        sample = self._run(num_images=0)[0]
        assert "pixel_values" not in sample
        prompt = _StubProcessor.prompt_ids
        answer = _eos_pad(list(range(20, 26)), 16)
        assert sample["mm_token_type_ids"].tolist() == [0] * len(prompt) + [0] * len(answer)

    def test_multiple_images_are_all_kept(self):
        sample = self._run(num_images=3)[0]
        assert sample["pixel_values"].shape[0] == 3
        assert int((sample["mm_token_type_ids"] == 1).sum()) == 6

    def test_rejects_a_non_append_only_template(self):
        """The answer boundary is a prefix diff, so a reordering template must not pass silently."""
        with pytest.raises(ValueError, match="prefix"):
            self._run(processor=_StubProcessor(append_only=False))

    def test_more_placeholders_than_images_is_an_error(self):
        with pytest.raises(ValueError, match="placeholders"):
            self.transform(
                {
                    "messages": [{"from": "human", "value": "<image><image>x"}, {"from": "gpt", "value": "y"}],
                    "images": [],
                },
                processor=_StubProcessor(),
                canvas_length=16,
                vocab_size=1024,
            )

    def test_previous_blocks_extend_the_prompt_and_its_modality_tags(self):
        prompt = _StubProcessor.image_ids + _StubProcessor.prompt_ids
        answer = list(range(20, 26))
        drawn = set()
        for epoch in range(40):
            sample = self._run(num_images=1, canvas_length=2, corruption_epoch=epoch)[0]
            assert sample["input_ids"].tolist() == prompt + answer
            block, remainder = divmod(int(sample["decoder_context_mask"].sum()) - len(prompt), 2)
            assert remainder == 0
            assert sample["mm_token_type_ids"].tolist() == [1, 1, 0, 0, 0, 0] + [0] * len(answer)
            assert sample["decoder_context_mask"].tolist() == [1] * (len(prompt) + block * 2) + [0] * (
                len(answer) - block * 2
            )
            assert sample["labels"][:2].tolist() == answer[block * 2 : block * 2 + 2]
            drawn.add(block)
        assert drawn == {0, 1, 2}

    def test_the_context_budget_narrows_the_block_range_instead_of_truncating(self):
        """Left truncation here would cut image spans loose from the patches the
        processor emitted, so blocks that would not fit are simply never drawn."""
        # The 6-token prompt plus one 2-token canvas is all `max_seq_len=9` has room for.
        sizes = {
            self._run(num_images=1, canvas_length=2, max_seq_len=9, corruption_epoch=epoch)[0]["input_ids"].numel()
            for epoch in range(30)
        }
        assert sizes == {8}

    def test_compute_all_canvas_df_loss_stacks_every_text_canvas(self):
        sample = self._run(num_images=1, canvas_length=2, compute_all_canvas_df_loss=True)[0]
        answer = list(range(20, 26))
        assert sample["decoder_input_ids"].shape == (3, 2)
        assert sample["decoder_context_mask"].shape[0] == 3
        for block in range(3):
            assert sample["labels"][block].tolist() == answer[block * 2 : block * 2 + 2]
        torch.testing.assert_close(sample["loss_weight"].sum(), torch.tensor(1.0))
        torch.testing.assert_close(sample["loss_weight"][0], torch.full((2,), 1.0 / 6))

    def test_the_weight_is_one_however_many_blocks_the_budget_leaves(self):
        """Both paths carry the same ``1/C`` normalization, so neither the answer's
        length nor how much of it the image-span budget ruled out changes what a sample
        contributes."""
        # Six answer tokens over 2-wide canvases is three blocks with room to spare...
        roomy = {
            float(self._run(num_images=1, canvas_length=2, corruption_epoch=epoch)[0]["loss_weight"].sum())
            for epoch in range(30)
        }
        assert roomy == {1.0}
        # ...and `max_seq_len=9` fits only one canvas in the encoder, which changes nothing.
        cramped = {
            float(
                self._run(num_images=1, canvas_length=2, max_seq_len=9, corruption_epoch=epoch)[0]["loss_weight"].sum()
            )
            for epoch in range(30)
        }
        assert cramped == {1.0}

    def test_overlong_prompt_degrades_to_a_zero_weight_sample(self):
        """Never an empty list: `MakeMicroBatchCollator` indexes `features[i][0]` blindly."""
        samples = self._run(max_seq_len=2)
        assert len(samples) == 1
        sample = samples[0]
        assert (sample["labels"] == IGNORE_INDEX).all()
        assert float(sample["loss_weight"].sum()) == 0.0
        assert "pixel_values" not in sample
        assert sample["decoder_context_mask"].tolist() == [0]

    def test_an_unusable_sample_still_batches_alongside_a_usable_one(self):
        """The collator rejects a batch mixing samples with and without the modality tag,
        and every usable sample on this path carries one whether or not it has an image."""
        batch = DiffusionGemmaCollator(pad_token_id=0)([self._run(num_images=1)[0], self._run(max_seq_len=2)[0]])
        assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape
        assert float(batch["loss_weight"][1].sum()) == 0.0


class TestSelfConditioning:
    """Training has to show the model the signal the sampler feeds it at every step."""

    #: Samples for the tests below that assert two draws *agree*. Each sample is one
    #: Bernoulli, so at the default batch size of 2 a draw that really did read a
    #: polluted generator would still land on the same two bits one time in four.
    AGREEMENT_SAMPLES = 16

    def test_absent_self_conditioning_matches_explicit_zero_embeddings(self):
        model = build_toy_model().eval()
        batch = build_toy_batch(model.config, batch_size=1)
        batch.pop("labels")
        batch.pop("loss_weight")
        zero_embeddings = torch.zeros(
            1,
            model.config.canvas_length,
            model.config.text_config.hidden_size,
        )
        with torch.no_grad():
            implicit = model(**batch).logits
            explicit = model(**batch, self_conditioning_embeddings=zero_embeddings).logits
        torch.testing.assert_close(implicit, explicit, rtol=2e-5, atol=2e-5)

    def test_masked_off_self_conditioning_matches_absent_signal(self):
        model = build_toy_model().eval()
        batch = build_toy_batch(model.config, batch_size=1)
        batch.pop("labels")
        batch.pop("loss_weight")
        signal = torch.randn(1, model.config.canvas_length, model.config.text_config.hidden_size)
        with torch.no_grad():
            implicit = model(**batch).logits
            masked = model(
                **batch,
                self_conditioning_embeddings=signal,
                self_conditioning_mask=torch.tensor([False]),
            ).logits
        torch.testing.assert_close(implicit, masked, rtol=2e-5, atol=2e-5)

    def _trainer(self, model, probability, global_step=0):
        from types import SimpleNamespace

        from veomni.trainer.base import BaseTrainer
        from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer

        trainer = DiffusionGemmaTrainer.__new__(DiffusionGemmaTrainer)
        trainer.base = BaseTrainer.__new__(BaseTrainer)
        trainer.base.device = torch.device("cpu")
        trainer.base.LOG_SAMPLE = False
        trainer.base.args = SimpleNamespace(
            train=SimpleNamespace(local_rank=0, enable_batch_invariant_mode=False, seed=42)
        )
        trainer.base.state = SimpleNamespace(global_step=global_step)
        trainer.base.model = toy_runtime(model)
        model.config.self_conditioning_prob = probability
        return trainer

    def test_the_signal_is_attached_during_training(self):
        model = build_toy_model().train()
        batch = self._trainer(model, 1.0).preforward(build_toy_batch(model.config))

        assert batch["self_conditioning_embeddings"].shape == (
            2,
            model.config.canvas_length,
            model.config.text_config.hidden_size,
        )
        assert not batch["self_conditioning_embeddings"].requires_grad
        assert batch["self_conditioning_mask"].tolist() == [True, True]

    def test_probability_zero_skips_the_extra_pass_entirely(self):
        model = build_toy_model().train()
        batch = self._trainer(model, 0.0).preforward(build_toy_batch(model.config))
        assert "self_conditioning_logits" not in batch
        assert "self_conditioning_embeddings" not in batch
        assert "self_conditioning_mask" not in batch

    def test_nothing_is_attached_outside_training(self):
        model = build_toy_model().eval()
        batch = self._trainer(model, 1.0).preforward(build_toy_batch(model.config))
        assert "self_conditioning_logits" not in batch
        assert "self_conditioning_embeddings" not in batch

    def test_the_pass_runs_even_when_the_draw_selects_nothing(self):
        """Skipping it looks free and is not: the pass carries an FSDP2 all-gather and an
        expert-parallel all-to-all, so gating it on a random draw lets data-parallel ranks
        disagree on how many collectives to issue, which deadlocks."""
        model = build_toy_model().train()
        batch = self._trainer(model, 1e-9).preforward(build_toy_batch(model.config))
        assert batch["self_conditioning_mask"].tolist() == [False, False]
        assert batch["self_conditioning_embeddings"].shape[0] == 2

    def test_the_mask_is_drawn_per_example(self):
        model = build_toy_model().train()
        trainer = self._trainer(model, 0.5)
        batch = build_toy_batch(model.config)
        # Ten micro-steps, each deriving its own generator from a different key. The
        # default generator `build_toy_batch` reseeds is not in this stream.
        enabled = sum(
            int(trainer.preforward({k: v.clone() for k, v in batch.items()})["self_conditioning_mask"].sum())
            for _ in range(10)
        )
        assert 0 < enabled < 20

    def test_the_conditioning_draw_ignores_the_default_cpu_generator(self):
        """Triton autotune also burns the default CPU generator. A later `torch.rand`
        on CPU would still desync `attn_dg_prefix` vs sdpa after the first step."""
        model = build_toy_model().train()
        batch = build_toy_batch(model.config, batch_size=self.AGREEMENT_SAMPLES)

        def draw(pollute: bool):
            trainer = self._trainer(model, 0.5)
            if pollute:
                torch.randn(4096)
            return trainer.preforward({k: v.clone() for k, v in batch.items()})["self_conditioning_mask"]

        assert draw(False).tolist() == draw(True).tolist()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA generator to pollute")
    def test_the_conditioning_draw_ignores_the_cuda_generator(self):
        model = build_toy_model(dtype=torch.bfloat16).cuda().train()
        batch = {k: v.cuda() for k, v in build_toy_batch(model.config, batch_size=self.AGREEMENT_SAMPLES).items()}

        def draw(pollute: bool):
            trainer = self._trainer(model, 0.5)
            trainer.base.device = torch.device("cuda")
            if pollute:
                torch.randn(4096, device="cuda")
            return trainer.preforward({k: v.clone() for k, v in batch.items()})["self_conditioning_mask"].cpu()

        assert draw(False).tolist() == draw(True).tolist()

    def test_trainers_that_share_a_seed_draw_the_same_first_mask(self):
        """DP ranks seed from `train.seed`; the first step must agree even if something
        else has already consumed the default CPU generator."""
        model = build_toy_model().train()
        batch = build_toy_batch(model.config, batch_size=self.AGREEMENT_SAMPLES)
        left = self._trainer(model, 0.5)
        torch.randn(4096)
        right = self._trainer(model, 0.5)
        assert (
            left.preforward({k: v.clone() for k, v in batch.items()})["self_conditioning_mask"].tolist()
            == right.preforward({k: v.clone() for k, v in batch.items()})["self_conditioning_mask"].tolist()
        )

    def _step_mask(self, trainer, batch, global_step):
        """One step's worth of draws: advance the counter, then one micro-batch."""
        trainer.base.state.global_step = global_step
        trainer.on_micro_batches_ready([batch])
        return trainer.preforward({k: v.clone() for k, v in batch.items()})["self_conditioning_mask"].tolist()

    def test_a_resumed_step_draws_the_mask_that_step_would_have_drawn(self):
        """`_extra_state` persists only the default CPU RNG, so a generator carried
        across steps would rewind to `train.seed` on resume and replay step 0's masks."""
        model = build_toy_model().train()
        batch = self._packed_batch(model.config, n_samples=8)

        walked = self._trainer(model, 0.5)
        for step in range(1, 8):
            before_crash = self._step_mask(walked, batch, step)

        resumed = self._trainer(model, 0.5)
        after_resume = self._step_mask(resumed, batch, 7)

        assert before_crash == after_resume

    def test_consecutive_steps_do_not_repeat_the_mask(self):
        """Otherwise the resume check above would pass on a constant draw."""
        model = build_toy_model().train()
        batch = self._packed_batch(model.config, n_samples=8)
        trainer = self._trainer(model, 0.5)
        masks = [self._step_mask(trainer, batch, step) for step in range(1, 8)]
        assert len({tuple(mask) for mask in masks}) > 1

    def test_each_micro_batch_of_a_step_draws_its_own_mask(self):
        """Gradient accumulation would otherwise condition the same canvases in every
        micro-batch of the step, turning the Bernoulli into a per-step coin."""
        model = build_toy_model().train()
        batch = self._packed_batch(model.config, n_samples=8)
        trainer = self._trainer(model, 0.5)
        trainer.base.state.global_step = 3
        trainer.on_micro_batches_ready([batch, batch, batch])
        masks = [
            tuple(trainer.preforward({k: v.clone() for k, v in batch.items()})["self_conditioning_mask"].tolist())
            for _ in range(3)
        ]
        assert len(set(masks)) > 1

    def test_the_signal_actually_reaches_the_loss(self):
        """A conditioning tensor the model ignored would make the extra forward pointless."""
        model = build_toy_model().train()
        batch = build_toy_batch(model.config)
        without = model(**batch, use_cache=False).loss
        with_signal = model(**self._trainer(model, 1.0).preforward(batch), use_cache=False).loss
        assert not torch.allclose(without, with_signal)

    @staticmethod
    def _packed_batch(config, n_samples=2):
        """One padded row holding ``n_samples`` concatenated canvases."""
        unpacked = build_toy_batch(config, batch_size=n_samples)
        packed = {k: v[:1] for k, v in unpacked.items()}
        packed["decoder_input_ids"] = unpacked["decoder_input_ids"].reshape(1, -1)
        packed["labels"] = unpacked["labels"].reshape(1, -1)
        packed["loss_weight"] = unpacked["loss_weight"].reshape(1, -1)
        return packed

    def test_a_packed_row_draws_one_flag_per_canvas(self):
        model = build_toy_model().train()
        batch = self._trainer(model, 1.0).preforward(self._packed_batch(model.config, n_samples=3))
        assert batch["self_conditioning_mask"].tolist() == [True, True, True]
        assert batch["self_conditioning_embeddings"].shape == (
            1,
            3 * model.config.canvas_length,
            model.config.text_config.hidden_size,
        )

    def test_a_packed_row_is_not_all_or_nothing(self):
        """The old batch-dim draw would emit a single bit and zero every canvas together."""
        model = build_toy_model().train()
        trainer = self._trainer(model, 0.5)
        batch = self._packed_batch(model.config, n_samples=4)
        mixed = False
        for _ in range(20):
            mask = trainer.preforward({k: v.clone() for k, v in batch.items()})["self_conditioning_mask"]
            if bool(mask.any()) and not bool(mask.all()):
                mixed = True
                break
        assert mixed

    def test_packed_mask_gates_canvases_independently(self):
        model = build_toy_model().train()
        batch = self._packed_batch(model.config)
        logits = torch.randn(1, 2 * model.config.canvas_length, 1024)
        both = model(
            **batch,
            self_conditioning_logits=logits,
            self_conditioning_mask=torch.tensor([True, True]),
            use_cache=False,
        ).loss
        first = model(
            **batch,
            self_conditioning_logits=logits,
            self_conditioning_mask=torch.tensor([True, False]),
            use_cache=False,
        ).loss
        none = model(
            **batch,
            self_conditioning_logits=logits,
            self_conditioning_mask=torch.tensor([False, False]),
            use_cache=False,
        ).loss
        assert not torch.allclose(both, first)
        assert not torch.allclose(first, none)
        assert not torch.allclose(both, none)


class TestExpandSelfConditioningMask:
    def test_unpacked_broadcasts_over_the_canvas(self):
        expanded = expand_self_conditioning_mask(torch.tensor([True, False]), batch_size=2, sequence_length=16)
        assert expanded.shape == (2, 1, 1)
        assert expanded.tolist() == [[[True]], [[False]]]

    def test_packed_repeats_each_flag_across_its_canvas(self):
        expanded = expand_self_conditioning_mask(torch.tensor([1.0, 0.0, 1.0]), batch_size=1, sequence_length=12)
        assert expanded.shape == (1, 12, 1)
        assert expanded[0, :, 0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]

    def test_rejects_a_mask_that_does_not_split_the_sequence(self):
        with pytest.raises(ValueError, match="does not split evenly"):
            expand_self_conditioning_mask(torch.tensor([True, False, True]), batch_size=1, sequence_length=8)


class TestAcceptanceSet:
    """Rebuilt from the current entropies every step, and sized by an entropy budget.

    The bound applies to the ranks *before* each candidate, which is the accepted set's
    total entropy minus its largest -- the mutual-information bound the sampler enforces.
    """

    def test_accepts_by_increasing_entropy_while_the_bound_holds(self):
        # Ranked 0.0, 0.125, 0.25, 0.75 with running bounds 0, 0, 0.125, 0.375.
        accepted = acceptance_mask(torch.tensor([[0.25, 0.125, 0.75, 0.0]]), 0.3)
        assert accepted.tolist() == [[True, True, False, True]]

    def test_the_most_confident_position_is_always_accepted(self):
        """Nothing precedes the first rank, so no budget can starve a step into a no-op."""
        assert acceptance_mask(torch.tensor([[5.0, 6.0]]), 0.0).tolist() == [[True, False]]

    def test_ties_break_towards_the_lower_index(self):
        assert acceptance_mask(torch.tensor([[0.5, 0.5, 0.5]]), 0.5).tolist() == [[True, True, False]]

    def test_a_certain_position_costs_nothing(self):
        """Zero entropy consumes no budget, so it never blocks the position behind it."""
        assert acceptance_mask(torch.tensor([[0.0, 0.0, 4.0]]), 0.0).tolist() == [[True, True, True]]

    def test_rows_are_independent(self):
        entropies = torch.tensor([[1.0, 4.0], [4.0, 1.0]])
        assert acceptance_mask(entropies, 0.0).tolist() == [[True, False], [False, True]]


def _peaked_logits(canvas_length, vocab_size, token):
    logits = torch.zeros(1, canvas_length, vocab_size)
    logits[..., token] = 30.0
    return logits


class _ScriptedDenoiser:
    """Model stand-in replaying fixed logits, recording what it was asked to denoise."""

    def __init__(self, logits, canvas_length, vocab_size):
        from types import SimpleNamespace

        self.logits = logits
        self.config = SimpleNamespace(canvas_length=canvas_length, text_config=SimpleNamespace(vocab_size=vocab_size))
        self.canvases = []
        self.conditioning = []

    def __call__(
        self, past_key_values, decoder_input_ids, decoder_attention_mask, self_conditioning_logits, use_cache
    ):
        from types import SimpleNamespace

        self.canvases.append(decoder_input_ids.clone())
        self.conditioning.append(self_conditioning_logits)
        return SimpleNamespace(logits=self.logits[min(len(self.canvases) - 1, len(self.logits) - 1)])


class TestEntropyBoundedDenoise:
    canvas_length, vocab_size, num_steps = 4, 64, 6

    def _denoise(self, logits, **kwargs):
        model = _ScriptedDenoiser(logits, self.canvas_length, self.vocab_size)
        options = {
            "config": EntropyBoundedSamplerConfig(num_steps=self.num_steps),
            "generator": torch.Generator().manual_seed(0),
        }
        options.update(kwargs)
        prediction, steps = entropy_bounded_denoise(
            model,
            past_key_values=None,
            decoder_attention_mask=torch.ones(1, 8 + self.canvas_length, dtype=torch.bool),
            **options,
        )
        return model, prediction, steps

    def test_a_settled_canvas_stops_early(self):
        model, prediction, steps = self._denoise([_peaked_logits(self.canvas_length, self.vocab_size, token=7)])
        # Step 1 has nothing to compare against, so step 2 is the earliest possible stop.
        assert (steps, len(model.canvases)) == (2, 2)
        assert prediction.tolist() == [[7] * self.canvas_length]

    def test_confidence_alone_does_not_stop_it(self):
        """A canvas can be confidently wrong, so the prediction has to hold still too."""
        flipping = [
            _peaked_logits(self.canvas_length, self.vocab_size, token=7),
            _peaked_logits(self.canvas_length, self.vocab_size, token=9),
        ] * 3
        _, _, steps = self._denoise(flipping)
        assert steps == self.num_steps

    def test_an_uncertain_canvas_runs_to_the_step_ceiling(self):
        _, _, steps = self._denoise([torch.zeros(1, self.canvas_length, self.vocab_size)])
        assert steps == self.num_steps

    def test_the_first_step_is_unconditioned_and_later_ones_carry_the_raw_logits(self):
        model, _, _ = self._denoise([_peaked_logits(self.canvas_length, self.vocab_size, token=7)])
        assert model.conditioning[0] is None
        # Untempered, matching what the trainer injects -- training has no step index and
        # so no temperature. The decoder softmaxes this itself, hence logits not probabilities.
        torch.testing.assert_close(model.conditioning[1], model.logits[0])

    def test_uncommitted_positions_are_renoised_rather_than_carried_over(self):
        """Not committing a position must return it to a uniform prior, so the next
        forward pass keeps exploring it instead of re-reading its own last guess."""
        model, _, _ = self._denoise(
            [torch.zeros(1, self.canvas_length, self.vocab_size)],
            config=EntropyBoundedSamplerConfig(num_steps=self.num_steps, entropy_budget=0.0),
        )
        # A zero budget commits exactly one position, so carrying the rest over would
        # leave at most one position changed between consecutive canvases.
        assert int((model.canvases[1] != model.canvases[0]).sum()) >= 2

    def test_it_returns_the_prediction_not_the_last_noisy_canvas(self):
        model, prediction, _ = self._denoise([torch.zeros(1, self.canvas_length, self.vocab_size)])
        assert prediction.tolist() == [[0] * self.canvas_length]
        assert not torch.equal(prediction, model.canvases[-1])


class TestBlockAutoregressiveGeneration:
    prompt_len = 12

    def _prompt(self, model):
        torch.manual_seed(3)
        return {"input_ids": torch.randint(3, model.config.text_config.vocab_size, (1, self.prompt_len))}

    def test_each_committed_canvas_grows_the_cache(self):
        from unittest import mock

        model = build_toy_model().eval()
        canvas_length = model.config.canvas_length
        encoder_calls, widths = [], []
        model.model.encoder.register_forward_pre_hook(lambda module, args: encoder_calls.append(1))
        real = generation.entropy_bounded_denoise

        def spy(denoised_model, *, decoder_attention_mask, **kwargs):
            widths.append(decoder_attention_mask.shape[-1])
            return real(denoised_model, decoder_attention_mask=decoder_attention_mask, **kwargs)

        with mock.patch.object(generation, "entropy_bounded_denoise", spy):
            generated = generation.block_autoregressive_generate(
                model,
                self._prompt(model),
                eos_token_ids=[],
                config=EntropyBoundedSamplerConfig(num_steps=2),
                max_blocks=3,
            )

        # Equation (4): each canvas goes through the causal encoder before the next one
        # starts, so the decoder's key axis grows by exactly one canvas per block.
        assert widths == [self.prompt_len + (block + 1) * canvas_length for block in range(3)]
        assert generated.shape == (1, 3 * canvas_length)
        # One prefill plus one commit per canvas a later canvas depends on. The final
        # canvas is not committed: nothing reads it, so that pass would be thrown away.
        assert len(encoder_calls) == 3

    def test_an_end_of_turn_token_ends_generation_mid_canvas(self):
        model = build_toy_model().eval()
        prompt = self._prompt(model)
        options = {"config": EntropyBoundedSamplerConfig(num_steps=2), "max_blocks": 2}

        unbounded = generation.block_autoregressive_generate(
            model, prompt, eos_token_ids=[], generator=torch.Generator().manual_seed(11), **options
        )
        first_canvas = unbounded[0, : model.config.canvas_length]
        end_of_turn = int(first_canvas[model.config.canvas_length // 2])
        expected_end = int((first_canvas == end_of_turn).nonzero()[0])

        stopped = generation.block_autoregressive_generate(
            model, prompt, eos_token_ids=[end_of_turn], generator=torch.Generator().manual_seed(11), **options
        )
        assert torch.equal(stopped[0], first_canvas[: expected_end + 1])

    def test_it_refuses_a_batch(self):
        model = build_toy_model().eval()
        prompt = {"input_ids": torch.zeros(2, self.prompt_len, dtype=torch.long)}
        with pytest.raises(ValueError, match="one sequence at a time"):
            generation.block_autoregressive_generate(model, prompt, eos_token_ids=[])


class TestLossNormalization:
    def _trainer(self):
        from types import SimpleNamespace

        from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer

        trainer = DiffusionGemmaTrainer.__new__(DiffusionGemmaTrainer)
        trainer.base = SimpleNamespace(model=SimpleNamespace(model_config=SimpleNamespace()))
        return trainer

    def test_zero_weight_samples_stay_out_of_the_denominator(self):
        """Counting them would shrink the gradient by a data-dependent ratio nothing logs."""
        loss_weight = torch.zeros(3, 8)
        loss_weight[:2] = 1.0 / 8
        trainer = self._trainer()
        trainer.on_micro_batches_ready([{"loss_weight": loss_weight}] * 2)
        assert trainer.micro_batches_num_samples == 4

    def test_a_rank_whose_samples_all_degraded_reports_zero(self):
        """`dllm_mean_global_loss` all-reduces the count and handles a global zero, so
        this rank should report 0 rather than put a phantom sample in the denominator."""
        trainer = self._trainer()
        trainer.on_micro_batches_ready([{"loss_weight": torch.zeros(2, 8)}])
        assert trainer.micro_batches_num_samples == 0

    def test_a_packed_row_counts_each_canvas(self):
        from types import SimpleNamespace

        trainer = self._trainer()
        trainer.base.model = SimpleNamespace(model_config=SimpleNamespace(canvas_length=8))
        loss_weight = torch.zeros(1, 16)
        loss_weight[0, :8] = 1.0 / 8
        trainer.on_micro_batches_ready([{"loss_weight": loss_weight}])
        assert trainer.micro_batches_num_samples == 1
        loss_weight[0, 8:] = 1.0 / 8
        trainer.on_micro_batches_ready([{"loss_weight": loss_weight}])
        assert trainer.micro_batches_num_samples == 2


class TestComponentGradNorms:
    """``print_ar_df_norm`` splits Eq. 13 into two backwards and logs each param-grad L2."""

    def _trainer(self, model):
        from types import SimpleNamespace

        from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer

        trainer = DiffusionGemmaTrainer.__new__(DiffusionGemmaTrainer)
        trainer.base = SimpleNamespace(model=toy_runtime(model))
        return trainer

    def test_the_flag_defaults_to_off(self):
        model = build_toy_model()
        assert model.config.print_ar_df_norm is False
        assert self._trainer(model)._print_ar_df_norm_enabled() is False

    def test_each_term_gets_its_own_param_grad_norm(self):
        model = build_toy_model().train()
        trainer = self._trainer(model)
        batch = build_toy_batch(model.config)
        outputs = model(**batch, use_cache=False)
        loss_dict = {"encoder_ar_loss": outputs.encoder_ar_loss, "diffusion_loss": outputs.diffusion_loss}
        trainer._backward_and_print_component_grad_norms(outputs.loss, loss_dict)
        assert float(loss_dict["encoder_ar_grad_norm"]) > 0
        assert float(loss_dict["diffusion_grad_norm"]) > 0
        assert not torch.allclose(loss_dict["encoder_ar_grad_norm"], loss_dict["diffusion_grad_norm"])
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())

    def test_two_backwards_match_a_single_backward_of_the_sum(self):
        """Tolerances hold float32 accumulation noise, not a margin for a real deviation.

        `model.decoder.embed_tokens.weight` is tied across the encoder embedding, the
        decoder embedding and `lm_head`, so its gradient is a cancelling sum over the
        whole vocabulary and reaches ~11 in magnitude; splitting that sum into two
        batches moves the last bits by ~1.7e-5. Plain `ar.backward(retain_graph=True)`
        followed by `diffusion.backward()` -- no clone, no add -- deviates from one
        backward by exactly the same amount, and reversing the order does not change
        it, which is what rules out the trainer's grad bookkeeping. Note that `logits`
        and `loss` are cast to float32 even on a float64 model, so a wider dtype does
        not shrink this.
        """
        torch.manual_seed(0)
        split = build_toy_model().train()
        summed = build_toy_model().train()
        summed.load_state_dict(split.state_dict())
        batch = build_toy_batch(split.config)
        out_split = split(**batch, use_cache=False)
        out_sum = summed(**batch, use_cache=False)
        self._trainer(split)._backward_and_print_component_grad_norms(
            out_split.loss,
            {"encoder_ar_loss": out_split.encoder_ar_loss, "diffusion_loss": out_split.diffusion_loss},
        )
        out_sum.loss.backward()
        for split_param, sum_param in zip(split.parameters(), summed.parameters(), strict=True):
            if split_param.grad is None and sum_param.grad is None:
                continue
            torch.testing.assert_close(split_param.grad, sum_param.grad, rtol=1e-4, atol=1e-4)

    def test_more_than_one_micro_batch_is_refused(self):
        """The split backward has no way to tell an accumulation apart from its own work.

        It needs each micro-batch to start from no gradient, and under FSDP2 it cannot
        check that: an accumulation behind a parameter left unsharded -- the root group
        always, every parameter when `reshard_after_backward` is false -- shows up as
        `grad is None` and is missed by `zero_grad` too. The AR term then adds onto it
        and reports two micro-batches as one. So the refusal lives here, where the
        micro-batch count is known before the first forward and is the same on every
        rank, rather than in the backward where it would need a collective to agree.
        """
        from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer

        model = build_toy_model().train()
        model.config.print_ar_df_norm = True
        trainer = self._trainer(model)
        batch = build_toy_batch(model.config)

        with pytest.raises(ValueError, match="one micro-batch per step"):
            DiffusionGemmaTrainer.on_micro_batches_ready(trainer, [batch, batch])

    def test_one_micro_batch_is_accepted(self):
        from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer

        model = build_toy_model().train()
        model.config.print_ar_df_norm = True
        trainer = self._trainer(model)
        batch = build_toy_batch(model.config)

        DiffusionGemmaTrainer.on_micro_batches_ready(trainer, [batch])
        assert trainer._remaining_micro_steps == 1

    def test_a_disabled_ar_term_logs_zero_grad(self):
        model = build_toy_model().train()
        model.config.encoder_ar_loss = False
        trainer = self._trainer(model)
        batch = build_toy_batch(model.config)
        outputs = model(**batch, use_cache=False)
        loss_dict = {"encoder_ar_loss": outputs.encoder_ar_loss, "diffusion_loss": outputs.diffusion_loss}
        trainer._backward_and_print_component_grad_norms(outputs.loss, loss_dict)
        assert float(loss_dict["encoder_ar_grad_norm"]) == 0.0
        assert float(loss_dict["diffusion_grad_norm"]) > 0

    def test_works_with_non_reentrant_checkpointing(self):
        model = build_toy_model().train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        trainer = self._trainer(model)
        batch = build_toy_batch(model.config)
        outputs = model(**batch, use_cache=False)
        loss_dict = {"encoder_ar_loss": outputs.encoder_ar_loss, "diffusion_loss": outputs.diffusion_loss}
        trainer._backward_and_print_component_grad_norms(outputs.loss, loss_dict)
        assert float(loss_dict["encoder_ar_grad_norm"]) > 0
        assert float(loss_dict["diffusion_grad_norm"]) > 0
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())


class TestStrictLayout:
    """Eq. 13 puts every canvas in the encoder; the decoder must not see the draw."""

    def test_training_keeps_the_sliding_cache_untruncated(self):
        model = build_toy_model().train()
        batch = build_toy_batch(model.config, prompt_len=24)
        with torch.no_grad():
            cache = model.model.encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).past_key_values
        from transformers.cache_utils import DynamicSlidingWindowLayer

        assert isinstance(cache, SealableCache)
        assert not cache.truncate_sliding
        assert not any(isinstance(layer, DynamicSlidingWindowLayer) for layer in cache.layers)
        assert cache.get_seq_length() == 24

    def test_generation_still_truncates_the_sliding_cache(self):
        model = build_toy_model().eval()
        batch = build_toy_batch(model.config, prompt_len=24)
        with torch.no_grad():
            cache = model.model.encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).past_key_values
        from transformers.cache_utils import DynamicSlidingWindowLayer

        assert any(isinstance(layer, DynamicSlidingWindowLayer) for layer in cache.layers)
        sliding = next(layer for layer in cache.layers if isinstance(layer, DynamicSlidingWindowLayer))
        assert sliding.keys.shape[-2] < 24

    def test_a_truncated_cache_refuses_a_hidden_suffix(self):
        """The silent failure mode: sliding layers would store the tail, which is exactly
        the future the mask is trying to hide."""
        model = build_toy_model().eval()
        batch = build_padded_batch(model.config, prompt_len=24, pad=0)
        cache = _prefilled_cache(model, batch)
        decoder_attention_mask = batch["decoder_attention_mask"].clone()
        # Hide the last 4 encoder positions as if they were future canvases.
        decoder_attention_mask[:, 20:24] = 0
        with pytest.raises(ValueError, match="truncated the cache"):
            model.model.decoder.create_diffusion_decoder_attention_mask(
                config=model.config.text_config,
                inputs_embeds=torch.zeros(2, model.config.canvas_length, model.config.text_config.hidden_size),
                past_key_values=cache,
                decoder_attention_mask=decoder_attention_mask,
            )

    def test_the_decoder_mask_zeros_the_future_on_a_training_forward(self):
        model = build_toy_model().train()
        config = model.config
        prompt_len, future = 8, 6
        batch = build_toy_batch(config, batch_size=1, prompt_len=prompt_len + future)
        decoder_attention_mask = torch.ones(1, prompt_len + future + config.canvas_length, dtype=torch.bool)
        decoder_attention_mask[:, prompt_len : prompt_len + future] = 0
        batch["decoder_attention_mask"] = decoder_attention_mask
        masks = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=config.text_config,
            inputs_embeds=torch.zeros(1, config.canvas_length, config.text_config.hidden_size),
            past_key_values=_prefilled_cache(model, batch),
            decoder_attention_mask=decoder_attention_mask,
        )
        full = masks["full_attention"][0, 0, 0]
        # Additive sdpa mask: kept positions are 0 / True, dropped are -inf / False.
        dropped = full[prompt_len : prompt_len + future]
        kept_prefix = full[:prompt_len]
        canvas = full[prompt_len + future :]
        if full.dtype is torch.bool:
            assert not dropped.any()
            assert kept_prefix.all()
            assert canvas.all()
        else:
            assert (dropped < -1e30).all()
            assert (kept_prefix > -1e5).all()
            assert (canvas > -1e5).all()


class TestEncoderBudget:
    """`max_seq_len` has to hold every canvas and still leave room for a prompt."""

    def _warn_count(self, **options):
        from unittest import mock

        from veomni.trainer import diffusion_gemma_trainer

        with mock.patch.object(diffusion_gemma_trainer.logger, "warning") as warning:
            diffusion_gemma_trainer.DiffusionGemmaTrainer._warn_if_the_encoder_budget_is_tight(options)
        return warning.call_count

    def test_a_budget_the_canvases_would_swallow_is_flagged(self):
        # All 8 canvases occupy 2048 tokens, over a 1024 budget.
        assert self._warn_count(canvas_length=256, max_canvases=8, max_seq_len=1024) == 1

    def test_filling_the_budget_with_canvases_alone_is_flagged(self):
        assert self._warn_count(canvas_length=256, max_canvases=8, max_seq_len=2048) == 1

    def test_one_canvas_of_headroom_for_the_prompt_is_enough(self):
        assert self._warn_count(canvas_length=256, max_canvases=8, max_seq_len=2304) == 0

    def test_the_shipped_geometry_is_quiet(self):
        assert self._warn_count(canvas_length=256, max_canvases=8, max_seq_len=4096) == 0


class TestDgPrefixGather:
    """Encoder cache is packed by layer type; pad / current / future never enter the kernel."""

    def test_full_layers_drop_left_pad_and_future_canvases(self):
        from veomni.models.transformers.diffusion_gemma.dg_prefix import pack_prefix_kv

        # Two rows, cache 8: left pad, visible prefix, then a hidden future suffix.
        visible = torch.tensor(
            [
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        encoder_k = torch.arange(2 * 1 * 8 * 2, dtype=torch.float32).view(2, 1, 8, 2)
        encoder_v = encoder_k + 1000
        packed_k, packed_v, cu = pack_prefix_kv(encoder_k, encoder_v, visible)
        assert cu.tolist() == [0, 3, 5]
        expected_k = torch.stack(
            [
                encoder_k[0, 0, 2],
                encoder_k[0, 0, 3],
                encoder_k[0, 0, 4],
                encoder_k[1, 0, 1],
                encoder_k[1, 0, 2],
            ]
        )
        torch.testing.assert_close(packed_k[0, 0], expected_k)
        torch.testing.assert_close(packed_v[0, 0], expected_k + 1000)

    def test_none_mask_on_a_sliding_layer_keeps_only_the_last_w_minus_one_keys(self):
        from veomni.models.transformers.diffusion_gemma.dg_prefix import encoder_visible_from_decoder_mask

        visible = encoder_visible_from_decoder_mask(
            None, cache_len=8, batch_size=1, device=torch.device("cpu"), sliding_window=4
        )
        assert visible.tolist() == [[0, 0, 0, 0, 0, 1, 1, 1]]

    def test_poisoning_hidden_encoder_keys_does_not_change_the_packed_prefix(self):
        from veomni.models.transformers.diffusion_gemma.dg_prefix import pack_prefix_kv

        visible = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
        encoder_k = torch.randn(1, 2, 4, 3)
        encoder_v = torch.randn(1, 2, 4, 3)
        packed_before, _, _ = pack_prefix_kv(encoder_k, encoder_v, visible)
        poisoned = encoder_k.clone()
        poisoned[:, :, 0] = 1e6
        poisoned[:, :, 3] = -1e6
        packed_after, _, _ = pack_prefix_kv(poisoned, encoder_v, visible)
        torch.testing.assert_close(packed_before, packed_after)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="DG_PREFIX kernel needs a GPU")
    def test_decoder_logits_match_sdpa_when_the_cache_is_packed(self):
        sdpa = build_toy_model(dtype=torch.bfloat16).cuda().eval()
        packed = build_toy_model(dtype=torch.bfloat16).cuda().eval()
        packed.config.text_config.attn_dg_prefix = True
        batch = {k: v.cuda() for k, v in build_padded_batch(sdpa.config).items()}
        with torch.no_grad():
            sdpa_out = sdpa(**batch).logits
            packed_out = packed(**batch).logits
        torch.testing.assert_close(packed_out, sdpa_out, rtol=5e-2, atol=5e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="DG_PREFIX kernel needs a GPU")
    def test_decoder_gradients_match_sdpa_when_the_cache_is_packed(self):
        """flex_attention was exact in the forward and several times too large in the
        backward; this path has to be judged on grad norm, not just logits."""
        torch.manual_seed(0)
        sdpa = build_toy_model(dtype=torch.bfloat16).cuda().train()
        torch.manual_seed(0)
        packed = build_toy_model(dtype=torch.bfloat16).cuda().train()
        packed.config.text_config.attn_dg_prefix = True
        batch = {k: v.cuda() for k, v in build_padded_batch(sdpa.config).items()}
        sdpa(**batch, use_cache=False).loss.backward()
        packed(**batch, use_cache=False).loss.backward()
        sdpa_grads = torch.cat([p.grad.float().flatten() for p in sdpa.parameters() if p.grad is not None])
        packed_grads = torch.cat([p.grad.float().flatten() for p in packed.parameters() if p.grad is not None])
        torch.testing.assert_close(packed_grads.norm(), sdpa_grads.norm(), rtol=5e-2, atol=1e-3)


class TestAlignNemo:
    """The two opt-in switches in `align_nemo`, which adopt NeMo AutoModel's semantics.

    Both are places where the stacks compute the same function and hand the optimizer a
    different problem, so a forward comparison cannot see them and only the parameter
    list and the gradients can. Each switch is checked for being off by default, for
    leaving the forward alone, and for the one failure mode that would be silent.
    """

    @staticmethod
    def _scatter_expert_scale(model, seed=0):
        """Give every router a distinct non-unit scale, which is what makes a fold visible."""
        torch.manual_seed(seed)
        for layer in model.model.decoder.layers:
            with torch.no_grad():
                layer.router.per_expert_scale.copy_(torch.empty_like(layer.router.per_expert_scale).uniform_(0.5, 1.5))

    def test_fold_is_off_by_default(self):
        model = build_toy_model()
        self._scatter_expert_scale(model)
        before = model.model.decoder.layers[0].experts.down_proj.detach().clone()

        assert align_nemo.fold_per_expert_scale(model) == 0
        torch.testing.assert_close(model.model.decoder.layers[0].experts.down_proj, before)
        assert model.model.decoder.layers[0].router.per_expert_scale.requires_grad

    def test_fold_leaves_the_forward_alone(self, monkeypatch):
        """`s * (W @ h)` and `(s * W) @ h` are the same function; only the parameters move."""
        model = build_toy_model(dtype=torch.float64)
        self._scatter_expert_scale(model)
        batch = build_toy_batch(model.config)
        with torch.no_grad():
            before = model(**batch, use_cache=False).logits.clone()

        monkeypatch.setenv("VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD", "1")
        assert align_nemo.fold_per_expert_scale(model) == model.config.text_config.num_hidden_layers
        with torch.no_grad():
            after = model(**batch, use_cache=False).logits

        torch.testing.assert_close(after, before, rtol=1e-10, atol=1e-10)

    def test_fold_rewrites_the_parameters_and_freezes_the_scale(self, monkeypatch):
        model = build_toy_model()
        self._scatter_expert_scale(model)
        layer = model.model.decoder.layers[0]
        scale = layer.router.per_expert_scale.detach().clone()
        expected = layer.experts.down_proj.detach() * scale[:, None, None]

        monkeypatch.setenv("VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD", "1")
        align_nemo.fold_per_expert_scale(model)

        torch.testing.assert_close(layer.experts.down_proj.detach(), expected)
        assert torch.all(layer.router.per_expert_scale == 1.0)
        assert not layer.router.per_expert_scale.requires_grad

    def test_fold_counts_the_shared_encoder_stack_once(self, monkeypatch):
        """Encoder and decoder share one set of expert tensors, reachable from two layer
        modules. Folding per module would square the scale -- silently, since the forward
        stays self-consistent and only the value is wrong."""
        model = build_toy_model()
        self._scatter_expert_scale(model)
        encoder_layer = model.model.encoder.language_model.layers[0]
        decoder_layer = model.model.decoder.layers[0]
        assert encoder_layer.experts.down_proj is decoder_layer.experts.down_proj, "premise: the stack is tied"
        squared = decoder_layer.experts.down_proj.detach() * decoder_layer.router.per_expert_scale[:, None, None] ** 2

        monkeypatch.setenv("VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD", "1")
        folded = align_nemo.fold_per_expert_scale(model)

        assert folded == model.config.text_config.num_hidden_layers
        assert not torch.allclose(decoder_layer.experts.down_proj.detach(), squared)

    @staticmethod
    def _slice_experts_for_ep(model, ep_size):
        """Leave each layer holding only the leading `E/ep` experts, as EP does."""
        local = model.config.text_config.num_experts // ep_size
        for layer in model.model.decoder.layers:
            layer.experts.down_proj = torch.nn.Parameter(layer.experts.down_proj[:local].clone())
        return local

    @staticmethod
    def _pretend_expert_parallel(monkeypatch, ep_size, ep_rank):
        """`ParallelState` is frozen, so the reporter is replaced rather than edited.

        `_slice_shard_tensor` imports it inside the call, which is what makes patching
        the module attribute enough.
        """

        class Reporter:
            def extra_parallel_enabled(self, name):
                return ep_size > 1

            def extra_parallel_rank(self, name):
                return ep_rank

        monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: Reporter())

    def test_fold_takes_the_expert_block_the_loader_gave_this_rank(self, monkeypatch):
        """Under EP the experts arrive pre-sliced and the scale does not.

        The block must be the one `ParallelPlan._slice_shard_tensor` cut for this rank
        when it loaded `down_proj`, which is why the fold asks that same function
        rather than recomputing the offset.
        """
        model = build_toy_model()
        self._scatter_expert_scale(model)
        scale = model.model.decoder.layers[0].router.per_expert_scale.detach().clone()
        ep_size, ep_rank = 4, 2
        local = self._slice_experts_for_ep(model, ep_size)
        expected = (
            model.model.decoder.layers[0].experts.down_proj.detach()
            * scale[ep_rank * local : (ep_rank + 1) * local][:, None, None]
        )

        self._pretend_expert_parallel(monkeypatch, ep_size, ep_rank)
        monkeypatch.setenv("VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD", "1")
        align_nemo.fold_per_expert_scale(model)

        torch.testing.assert_close(model.model.decoder.layers[0].experts.down_proj.detach(), expected)

    def test_fold_refuses_sharded_experts_the_plan_cannot_place(self, monkeypatch):
        """`_slice_shard_tensor` returns the tensor untouched rather than raising when it
        cannot slice -- here because the experts do not divide the scale -- so the fold
        has to notice that for itself instead of broadcasting the wrong length."""
        model = build_toy_model()
        for layer in model.model.decoder.layers:
            layer.experts.down_proj = torch.nn.Parameter(layer.experts.down_proj[:3].clone())

        monkeypatch.setenv("VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD", "1")
        with pytest.raises(RuntimeError, match="did not reduce `per_expert_scale`"):
            align_nemo.fold_per_expert_scale(model)

    def test_fold_refuses_experts_shared_under_disagreeing_scales(self, monkeypatch):
        """The dedup is keyed on the expert tensor; two owners with different scales
        would make which one wins depend on module traversal order."""
        model = build_toy_model()
        encoder_layer = model.model.encoder.language_model.layers[0]
        encoder_layer.router.per_expert_scale = torch.nn.Parameter(
            encoder_layer.router.per_expert_scale.detach().clone()
        )

        monkeypatch.setenv("VEOMNI_ALIGN_NEMO_EXPERT_SCALE_FOLD", "1")
        with pytest.raises(RuntimeError, match="shares its experts with another layer"):
            align_nemo.fold_per_expert_scale(model)

    def test_self_conditioning_embeddings_match_the_materialized_path(self):
        """Same value and same `dE` as `softmax(logits) @ E`, which is the form NeMo runs
        and the one this replaces to avoid a `[tokens, vocab]` tensor."""
        torch.manual_seed(7)
        hidden = torch.randn(2, 5, 7, dtype=torch.float64)
        table = torch.randn(13, 7, dtype=torch.float64, requires_grad=True)
        embed_scale = torch.tensor(7**0.5, dtype=torch.float64)
        upstream = torch.randn(2, 5, 7, dtype=torch.float64)

        logits = torch.tanh(torch.nn.functional.linear(hidden, table.detach()).float() / 30.0) * 30.0
        reference = torch.matmul(logits.softmax(dim=-1).to(table.dtype), table) * embed_scale
        (reference * upstream).sum().backward()
        reference_grad, table.grad = table.grad.clone(), None

        # A chunk budget that splits the 10 positions, so the accumulation is exercised.
        actual = align_nemo.nemo_aligned_self_conditioning_embeddings(
            hidden, table.detach(), table, embed_scale, softcapping=30.0, logit_elements_per_chunk=3 * 13
        )
        (actual * upstream).sum().backward()

        torch.testing.assert_close(actual, reference.detach())
        torch.testing.assert_close(table.grad, reference_grad)
        assert table.grad.abs().max() > 0

    def test_self_conditioning_gradient_survives_the_callers_no_grad(self):
        """The first pass runs under `no_grad` on both stacks, and that is the whole
        difficulty: a view taken there keeps `requires_grad` but loses `grad_fn`, so the
        path reads as present and delivers exactly zero."""
        torch.manual_seed(7)
        hidden = torch.randn(1, 4, 5, dtype=torch.float64)
        table = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
        embed_scale = torch.tensor(5**0.5, dtype=torch.float64)

        with torch.no_grad():
            embeddings = align_nemo.nemo_aligned_self_conditioning_embeddings(
                hidden, table.detach(), table, embed_scale, softcapping=30.0
            )
        assert embeddings.grad_fn is not None
        embeddings.sum().backward()
        assert table.grad is not None and table.grad.abs().max() > 0

    def test_the_model_takes_the_switch_and_defaults_to_veomnis_own_path(self, monkeypatch):
        """End to end through `forward`: the value is the switch's business to preserve,
        the gradient to the embedding table is the switch's business to create."""
        model = build_toy_model(dtype=torch.float64)
        batch = build_toy_batch(model.config)
        conditioning = {k: v for k, v in batch.items() if k not in ("labels", "loss_weight")}

        def soft_embeddings():
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                output = model(
                    **conditioning, use_cache=False, return_logits=False, return_self_conditioning_embeddings=True
                )
            return output.self_conditioning_embeddings

        default = soft_embeddings()
        assert default.grad_fn is None

        monkeypatch.setenv("VEOMNI_ALIGN_NEMO_SELF_COND_EMBED_GRAD", "1")
        aligned = soft_embeddings()
        torch.testing.assert_close(aligned.detach(), default)

        aligned.sum().backward()
        assert model.model.decoder.embed_tokens.weight.grad.abs().max() > 0
