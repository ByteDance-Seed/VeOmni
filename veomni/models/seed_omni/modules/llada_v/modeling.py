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

"""LLaDA-V as a single SeedOmni module.

Same objective and same one-node graph as the dense :mod:`..llada` module: the
mask predictor supervises itself in place over one packed row. What this family
adds is the SigLIP tower and the projector, which run inside the same node -- the
image rows are scattered into the text embeddings before the backbone, so there is
still nothing to hand to a sibling module.

``_init_weights`` is inherited from ``LLaDAVPreTrainedModel`` and must stay that
way: ``to_empty()`` on the meta random-init path discards the constructor's zeros
for ``image_newline``, and nothing in the checkpoint restores it, so the hook
re-applies them after materialisation.
"""

from __future__ import annotations

from typing import Any

import torch

from ....transformers.llada_v.modeling_llada_v import LLaDAVForConditionalGeneration
from ..llada.modeling import LladaOmni
from ..module_modeling_base import PretrainedOmniModule
from .configuration import LladaVOmniConfig
from .processing import LladaVOmniPreprocessor


class LladaVOmni(LLaDAVForConditionalGeneration, PretrainedOmniModule):
    """The vendored LLaDA-V model, registered as an OmniModule.

    ``LLaDAVForConditionalGeneration`` comes first so its ``forward`` (vision
    scatter, bidirectional mask, ``dllm_lm_head_loss``) is the one the graph invokes;
    ``PretrainedOmniModule`` contributes the asset binding and the no-op inference
    defaults.

    ``preprocessor_class`` is what ``OmniProcessor.from_config`` reads to build the
    CPU worker; without it the sample reaches the graph unprocessed and
    ``batch_pre`` finds no tensors to stack.
    """

    config_class = LladaVOmniConfig
    preprocessor_class = LladaVOmniPreprocessor
    base_model_prefix = "llada_v_omni"
    _no_split_modules = ["LLaDADecoderLayer", "SiglipEncoderLayer"]

    # Borrowed from the dense module: the denoising loop itself reads only
    # `config.mask_token_id`, the tokenizer and its EOS. LLaDA-V differs in one place
    # -- the row it decodes is built from image-embedded inputs rather than plain
    # token embeddings -- so `_decode_forward` and the initialiser are stated below.
    SIGNAL_TEXT_DONE = LladaOmni.SIGNAL_TEXT_DONE
    tokenizer = LladaOmni.tokenizer
    _eos_token_id = LladaOmni._eos_token_id
    _reset_diffusion_state = LladaOmni._reset_diffusion_state
    _make_num_transfer_schedule = staticmethod(LladaOmni._make_num_transfer_schedule)
    _diffusion_step = LladaOmni._diffusion_step
    _finish_diffusion = LladaOmni._finish_diffusion
    generate = LladaOmni.generate
    finalize = LladaOmni.finalize

    def __init__(self, config: LladaVOmniConfig):
        super().__init__(config)
        self._reset_diffusion_state()
        # Per-request image inputs, captured by `_prompt_ids` on the first pass.
        self._infer_pixel_values: torch.Tensor | None = None
        self._infer_image_mask: torch.Tensor | None = None

    def reset_local_inference_state(self) -> None:
        """Clear the denoising row and the request's image inputs between requests.

        Stated here rather than borrowed: the dense module's zero-argument ``super()``
        is bound to that class, so borrowing it raises "obj must be an instance or
        subtype of type" when ``self`` is this one.
        """
        super().reset_local_inference_state()
        self._reset_diffusion_state()
        self._infer_pixel_values = None
        self._infer_image_mask = None

    def _prompt_ids(self, conversation_list: Any) -> torch.Tensor:
        """The prompt tokens plus the image tensors the decoder needs.

        The dense module reads ``input_ids`` and stops; here the same carrier also
        holds ``pixel_values`` / ``image_mask``, which the vendored forward scatters
        into the embeddings. They are captured into per-request state because the
        denoising passes carry only tokens.
        """
        items = conversation_list or []
        if items and not isinstance(items[0], (list, tuple)):
            items = [items]
        for sample in items:
            for item in sample:
                value = getattr(item, "value", None)
                if isinstance(value, dict) and "input_ids" in value:
                    # The carrier's tensors were built by the CPU preprocessor and
                    # were never moved; the vision tower runs on the module's device.
                    pixel_values = value.get("pixel_values")
                    image_mask = value.get("image_mask")
                    self._infer_pixel_values = (
                        pixel_values.to(self.device) if isinstance(pixel_values, torch.Tensor) else None
                    )
                    self._infer_image_mask = (
                        image_mask.to(self.device) if isinstance(image_mask, torch.Tensor) else None
                    )
                    return value["input_ids"].to(device=self.device, dtype=torch.long).reshape(-1)
        raise ValueError(
            "LladaVOmni.generate: no tokenized prompt in the carrier. The module's preprocessor must run "
            "before the FSM; check that `data_type: seedomni` and `processing.py` are both wired."
        )

    def _decode_forward(self) -> dict[str, Any]:
        """Run the mask predictor over the row, with the request's image slots filled.

        The image inputs go out on **every** pass, not just the first. A mask
        predictor is re-encoded whole each step -- ``self.model`` builds
        ``inputs_embeds`` from the row's token ids, and the image slots hold token 0
        rather than the features themselves -- so the pixels have to be scattered
        again or every pass after the first decodes against an empty image.
        """
        x = self._diff_x
        if x is None:
            raise RuntimeError("LladaVOmni: decode forward without an initialised row.")
        kwargs: dict[str, Any] = {
            "input_ids": x.unsqueeze(0),
            "attention_mask": torch.ones_like(x, dtype=torch.long).unsqueeze(0),
            "use_cache": False,
        }
        if self._infer_pixel_values is not None:
            kwargs["pixel_values"] = self._infer_pixel_values
            # `image_mask` covers the prompt only -- the masked answer span has no
            # image slots -- while the vendored forward requires it to line up with
            # the whole row it is about to embed. Padding with False says exactly
            # that: no image features go past the prompt.
            image_mask = self._infer_image_mask
            row_len = int(x.numel())
            if image_mask is not None and int(image_mask.numel()) < row_len:
                pad = torch.zeros(row_len - int(image_mask.numel()), dtype=image_mask.dtype, device=image_mask.device)
                image_mask = torch.cat([image_mask.reshape(-1), pad])
            kwargs["image_mask"] = image_mask.unsqueeze(0) if image_mask is not None else None
        outputs = self.model(**kwargs)
        return {"llada_logits": self.lm_head(outputs.last_hidden_state)}

    def _diffusion_init(
        self,
        conversation_list: Any,
        generation_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Same initial masked row as the dense module, plus the image inputs.

        Calls the dense implementation explicitly rather than through ``super()``: the
        loop lives on :class:`LladaOmni`, which is not in this class's MRO (only its
        methods were copied in), so ``super()._diffusion_init`` does not resolve.
        """
        return LladaOmni._diffusion_init(self, conversation_list, generation_kwargs)

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        # LLaDA-V owns attention on the backbone and validates its backend. The
        # vision tower is pinned to sdpa inside the vendored model, so it cannot
        # inherit a flash kernel that would all-to-all the patch sequence.
        return super()._check_and_adjust_attn_implementation(attn_implementation, **kwargs)

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Run the mask predictor and surface ``_loss``.

        The graph passes the collated micro-batch as keyword arguments. ``pixel_values``
        and ``image_mask`` travel with it when the batch has an image; without them the
        vendored forward still runs the tower once (a dummy image) whenever the world
        size is above one, so the FSDP all-gathers of ranks that do have images stay
        matched.
        """
        kwargs.pop("conversation_list", None)
        outputs = super().forward(use_cache=False, **kwargs)
        result: dict[str, Any] = {}
        if outputs.loss is not None:
            result["_loss"] = outputs.loss
        return result


__all__ = ["LladaVOmni"]
