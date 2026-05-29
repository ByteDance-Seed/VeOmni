"""
JanusVqvae — Janus' VQVAE + generation projection head as one OmniModule.

Mixin form: ``class JanusVqvae(OmniModule, PreTrainedModel)``.

Call-site methods (graph entry points)
--------------------------------------
This module exposes three call-site methods that map 1-to-1 to YAML
``nodes:`` entries of the form ``module: <name>.<method>``:

  ``encode``       — VQVAE encode (training + inference setup):
                     ``gen_image_patches`` → ``gen_embeds`` + ``vq_token_ids``.
  ``decode``       — Unified VQ head; one node covers training loss and
                     inference sampling+feedback:

                       * Training (``hidden_states`` + ``gt_token_ids``):
                         ``_loss`` (token-mean CE via ``generation_head``).
                       * Inference sample (``hidden_states`` only): project
                         the last position via ``generation_head``, sample
                         → ``vq_token_id``, codebook lookup → ``embed``
                         (next-step ``inputs_embeds``).
                       * Inference lookup (``token_id`` only): codebook
                         lookup → ``embed`` (when sampling lives outside
                         the FSM body).

  ``decode_pixels``— Image rendering (post-FSM): ``vq_token_ids`` →
                     pixels (raw ``[-1, 1]`` tensor — for callers that
                     want to do their own postprocess; the FSM
                     :meth:`generate` path uses :class:`JanusVqvaeProcessor`
                     to surface PIL images directly).

``forward`` aliases :meth:`encode` (the most common training default).

Single-loss protocol
--------------------
``decode`` returns a token-mean ``_loss`` scalar in training (the only
allowed loss key per the V2 ``OmniModel`` contract).  Inference paths
return ``vq_token_id`` / ``embed`` and never set ``_loss``.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedModel
from transformers.models.janus.configuration_janus import JanusVQVAEConfig
from transformers.models.janus.modeling_janus import (
    JanusVQVAE,
    JanusVQVAEAlignerMLP,
    JanusVQVAEHead,
)

from ....conversation import ConversationPart
from ....generation_graph import FSM_SIGNAL_KEY
from ....module import OmniModule
from .configuration import JanusVqvaeConfig
from .processing import JanusVqvaeProcessor


# Default Janus-1.3B grid: 24 x 24 = 576 VQ tokens per image.
_DEFAULT_NUM_IMAGE_TOKENS = 576


class JanusVqvae(OmniModule, PreTrainedModel):
    """VQVAE + generation head for Janus VQ image generation.

    The VQVAE encoder/decoder is frozen by default (matching the Janus
    paper).  Only the generation projection layers
    (``generation_embeddings``, ``generation_aligner``,
    ``generation_head``) are trainable.
    """

    config_class = JanusVqvaeConfig
    processor_class = JanusVqvaeProcessor
    base_model_prefix = "janus_vqvae"
    main_input_name = "gen_image_patches"
    _no_split_modules: list = []

    def __init__(self, config: JanusVqvaeConfig):
        super().__init__(config)

        vq_cfg = JanusVQVAEConfig(**config.vq_config) if config.vq_config else JanusVQVAEConfig()
        self._vq_cfg = vq_cfg
        self.vqmodel = JanusVQVAE._from_config(vq_cfg)
        self.generation_embeddings = nn.Embedding(vq_cfg.num_embeddings, vq_cfg.embed_dim)
        self.generation_aligner = JanusVQVAEAlignerMLP(vq_cfg)
        self.generation_head = JanusVQVAEHead(vq_cfg)

        if config.freeze_vqvae:
            self.vqmodel.requires_grad_(False)

        # Per-image VQ-token buffer used by :meth:`generate` to accumulate
        # sampled tokens between FSM iterations.  Reset on each
        # ``image_complete`` signal; finalize keeps the decoded PIL
        # images so the caller can collect them after the run.
        self._gen_buffer: List[int] = []
        self._collected_images: List[Image.Image] = []

        # Auto-populated by :meth:`OmniModule.from_pretrained` from
        # ``<weights_path>/preprocessor_config.json``.  Used by
        # :meth:`generate` to convert the VQVAE's ``[-1, 1]`` float output
        # back into a PIL image — see :class:`JanusVqvaeProcessor`.
        self._processor: Optional[Any] = None

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialiser for layers added on top of the upstream Janus
        VQ encoder/decoder modules (which carry their own ``_init_weights``).

        Dispatch through ``torch.nn.init.*`` rather than the tensor's
        own in-place methods so the
        :func:`transformers.initialization.guard_torch_init_functions`
        monkey-patch can skip already-loaded weights — see the matching
        comment in
        :meth:`veomni.models.seed_omni.modules.base.text_encoder.modeling.TextEncoder._init_weights`.
        """
        if hasattr(module, "_init_weights"):
            return
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(self, gen_image_patches: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """Flatten ``(B, N_images, 3, H, W)`` → ``(B*N_images, 3, H, W)``."""
        if gen_image_patches is not None and gen_image_patches.ndim == 5:
            b, n = gen_image_patches.shape[:2]
            gen_image_patches = gen_image_patches.reshape(b * n, *gen_image_patches.shape[2:])
            kwargs["_gpatch_batch_n_images"] = (b, n)
        return dict(gen_image_patches=gen_image_patches, **kwargs)

    def forward(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        """Default forward — alias for :meth:`encode`."""
        return self.encode(**kwargs)

    def encode(self, gen_image_patches: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """VQVAE encode pass: pixels → ground-truth tokens + teacher-forcing embeds.

        Returns ``{}`` for text-only batches.  When ``self.config.freeze_vqvae``
        is ``True`` the VQVAE is wrapped in ``torch.no_grad()`` to avoid
        tracking gradients on frozen parameters.
        """
        if gen_image_patches is None:
            return {}

        b_n = kwargs.pop("_gpatch_batch_n_images", None)

        with torch.no_grad() if self.config.freeze_vqvae else torch.enable_grad():
            vq_out = self.vqmodel.encode(gen_image_patches)
        vq_token_ids = vq_out.image_tokens

        gen_embeds_raw = self.generation_embeddings(vq_token_ids)
        gen_embeds = self.generation_aligner(gen_embeds_raw)

        if b_n is not None:
            b, n = b_n
            p = vq_token_ids.size(1)
            vq_token_ids = vq_token_ids.reshape(b, n * p)
            gen_embeds = gen_embeds.reshape(b, n * p, gen_embeds.size(2))

        return {"gen_embeds": gen_embeds, "vq_token_ids": vq_token_ids}

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        gt_token_ids: Optional[torch.Tensor] = None,
        token_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Unified VQ head — training loss + inference sample + lookup.

        Three input-driven dispatch paths:

          * **Training** (``hidden_states`` + ``gt_token_ids``):
              ``logits = generation_head(hidden_states)``, then token-mean
              CE w.r.t. ``gt_token_ids``.  Returns ``{"_loss"}``.

          * **Inference sample** (``hidden_states`` alone, no
            ``gt_token_ids``): project the last position, sample with
            multinomial, codebook lookup + ``generation_aligner`` for the
            next embed.  Returns ``{"vq_token_id", "embed"}``.

          * **Inference lookup** (``token_id``): codebook lookup +
            ``generation_aligner``.  Returns ``{"embed"}``.

        Branches are mutually exclusive at runtime — one set of inputs
        present means one path runs.
        """
        out: Dict[str, Any] = {}

        if hidden_states is not None and gt_token_ids is not None:
            logits = self.generation_head(hidden_states)
            out["_loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                gt_token_ids.reshape(-1).long(),
                ignore_index=-100,
            )
            return out

        if hidden_states is not None:
            last_logits = self.generation_head(hidden_states[:, -1:, :])
            probs = torch.softmax(last_logits.squeeze(1), dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            embed_raw = self.generation_embeddings(sampled)
            out["vq_token_id"] = sampled.squeeze(-1)
            out["embed"] = self.generation_aligner(embed_raw)
            return out

        if token_id is not None:
            if token_id.ndim == 1:
                token_id = token_id.unsqueeze(1)
            embed_raw = self.generation_embeddings(token_id)
            out["embed"] = self.generation_aligner(embed_raw)
            return out

        return out

    def decode_pixels(self, vq_token_ids: torch.Tensor) -> torch.Tensor:
        """Decode a sequence of VQ token IDs to pixel values ``(B, H, W, 3)``."""
        pixel_values = self.vqmodel.decode(vq_token_ids)
        return pixel_values.permute(0, 2, 3, 1)

    # ── Inference (conversation-list aware) ───────────────────────────────────

    def generate(
        self,
        *,
        hidden_states: Optional[torch.Tensor] = None,
        conversation_list: Optional[List[ConversationPart]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Sample one VQ token, buffer it, decode the image when full.

        Each FSM ``image_vq`` iteration calls this with the latest LLM
        hidden states.  We:

        1. Sample a VQ token id from ``generation_head`` (multinomial /
           greedy, controlled by ``generation_kwargs``).
        2. Append it to :attr:`_gen_buffer` and emit a ``token``-kind
           ``ConversationPart`` carrying the codebook-lookup embedding
           (the ``vaegen_to_llm`` edge feeds that embed back as the
           next iteration's ``inputs_embeds``).
        3. On the ``num_image_tokens``-th call (default 576) we decode
           the buffered tokens to pixels via ``vqmodel.decode``, clear
           the buffer, hand the ``[-1, 1]`` reconstruction to
           :meth:`JanusVqvaeProcessor.postprocess` to get a PIL image,
           stash that PIL image in :attr:`_collected_images` for
           :meth:`finalize`, expose it via ``ctx['generated_image']``
           and raise ``module_signal = 'image_complete'`` so the FSM
           transitions to ``image_vq_end``.

        Batch contract
        --------------
        Single-image generation (B == 1) is the only supported path
        today — the FSM driver and conversation-list shape both assume
        one image stream per request.  We assert it explicitly rather
        than silently mis-broadcast for B > 1 (the legacy ``decode()``
        below quietly assumed the same).
        """
        if hidden_states is None:
            return {"conversation_list": conversation_list} if conversation_list is not None else {}

        hidden_states = hidden_states.to(self.device)
        batch_size = hidden_states.size(0)
        sampling = self._extract_sampling_kwargs(generation_kwargs)
        cfg_w = float(sampling.pop("guidance_scale", 1.0) or 1.0)

        # Classifier-free guidance mix.
        #
        # When ``JanusLlama.generate`` expanded the KV cache to bs=2 on the
        # first ``image_vq`` step, every subsequent step delivers bs=2
        # hidden states here: row 0 = conditional, row 1 = unconditional.
        # We project both, mix via the standard ``uncond + w*(cond-uncond)``
        # formula (same as HF's ``ClassifierFreeGuidanceLogitsProcessor``
        # — see ``transformers/models/janus/modeling_janus.py:1250``), then
        # sample ONE token shared across both branches and broadcast it
        # back to bs=2 for the next AR step's ``inputs_embeds`` (matching
        # ``modeling_janus.py:1352``).  Single-branch bs=1 mode (no CFG)
        # keeps the legacy fast path.
        if batch_size == 2 and cfg_w > 1.0:
            cond_logits = self.generation_head(hidden_states[:1, -1:, :]).squeeze(1)
            uncond_logits = self.generation_head(hidden_states[1:, -1:, :]).squeeze(1)
            last_logits = uncond_logits + cfg_w * (cond_logits - uncond_logits)
        elif batch_size == 1:
            last_logits = self.generation_head(hidden_states[:, -1:, :]).squeeze(1)
        else:
            raise NotImplementedError(
                f"JanusVqvae.generate received hidden_states with B={batch_size}. "
                "Supported: B=1 (no CFG) or B=2 (CFG cond/uncond pair). "
                "Multi-image batched generation is not yet wired into the conversation-list FSM."
            )

        sampled = self._sample_vq_token(last_logits, **sampling).unsqueeze(-1)  # (1, 1)
        token_id_int = int(sampled[0, 0].item())
        self._gen_buffer.append(token_id_int)

        embed_raw = self.generation_embeddings(sampled)
        embed = self.generation_aligner(embed_raw)  # (1, 1, H)
        if batch_size == 2:
            # Feed the next AR step as bs=2 so the JanusLlama KV cache
            # (also bs=2) sees a matching tail.
            embed = embed.expand(2, *embed.shape[1:]).contiguous()

        out: Dict[str, Any] = {
            "vq_token_id": sampled.squeeze(-1),
            "embed": embed,
        }

        if conversation_list is not None:
            conversation_list.append(
                ConversationPart(
                    kind="token",
                    role="assistant",
                    token_id=token_id_int,
                    inputs_embeds=embed,
                    meta={"source": "vqvae"},
                )
            )
            out["conversation_list"] = conversation_list

        target = self._num_image_tokens()
        if len(self._gen_buffer) >= target:
            token_ids = torch.tensor([self._gen_buffer], dtype=torch.long, device=embed.device)
            with torch.inference_mode():
                decoded = self.vqmodel.decode(token_ids).permute(0, 2, 3, 1)  # (1, H, W, 3) in [-1, 1]
            self._gen_buffer.clear()

            # Postprocess into a directly-savable PIL image via the
            # processor (it owns the Janus-specific [-1, 1] → uint8
            # inverse-normalisation).  We require it here because the
            # convention is module-specific and shipped next to the
            # weights — see :class:`JanusVqvaeProcessor`.
            if self._processor is None:
                raise RuntimeError(
                    "JanusVqvae.generate: cannot postprocess VQVAE output — no processor was "
                    "loaded.  Ensure `preprocessor_config.json` ships next to the weights "
                    "checkpoint so OmniModule.from_pretrained can auto-load it."
                )
            image_pil = self._processor.postprocess(decoded)[0]
            self._collected_images.append(image_pil)
            out["generated_image"] = image_pil
            out[FSM_SIGNAL_KEY] = "image_complete"

        return out

    def reset_inference_state(self) -> None:
        """Wipe per-request buffers — called by :class:`OmniInferencer` between runs."""
        self._gen_buffer.clear()
        self._collected_images.clear()

    def finalize(self, *, ctx: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
        """Hand off accumulated images to the caller via the framework's finalize hook."""
        del request
        images = list(ctx.get("generated_images_collected", []) or self._collected_images)
        if not images:
            return {}
        return {"images": images}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _num_image_tokens(self) -> int:
        cfg = self.config.vq_config
        if isinstance(cfg, dict):
            return int(cfg.get("num_image_tokens", _DEFAULT_NUM_IMAGE_TOKENS))
        val = getattr(cfg, "num_image_tokens", None)
        return int(val) if val is not None else _DEFAULT_NUM_IMAGE_TOKENS

    @staticmethod
    def _extract_sampling_kwargs(generation_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # `guidance_scale` rides in this dict but is consumed by the CFG
        # mix above (not by `_sample_vq_token`), so we whitelist it here
        # and pop it before sampling.
        merged: Dict[str, Any] = {
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": True,
            "guidance_scale": 1.0,
        }
        if generation_kwargs:
            for k in ("temperature", "top_p", "do_sample", "guidance_scale"):
                if k in generation_kwargs:
                    merged[k] = generation_kwargs[k]
        return merged

    @staticmethod
    def _sample_vq_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Sample one VQ token id per row.  ``logits`` is ``(B, vocab)``;
        returns ``(B,)`` so the caller can ``unsqueeze(-1)`` for the
        codebook embedding lookup (matching the legacy ``decode()``
        ``(B, 1)`` shape contract)."""
        if not do_sample:
            return logits.argmax(dim=-1)
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            to_remove = cumulative - sorted_probs > top_p
            sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
            logits = logits.scatter(1, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ── Training-side dummy forward ────────────────────────────────────────────

    def dummy_inputs(self, *, batch_size: int, device: Any, dtype: Any) -> Dict[str, Any]:
        """Return zero ``gen_image_patches`` so the VQVAE encode path runs.

        Used by the trainer for micro-batches that don't carry any image
        for VQ-generation — keeps the FSDP graph aligned across DP/SP
        ranks.  See module-doc "Training vs. inference no input
        semantics" in :mod:`veomni.models.seed_omni.module`.
        """
        cfg = self.config.vq_config or {}
        h = cfg.get("resolution", 384) if isinstance(cfg, dict) else getattr(cfg, "resolution", 384)
        c = cfg.get("in_channels", 3) if isinstance(cfg, dict) else getattr(cfg, "in_channels", 3)
        return {
            "gen_image_patches": torch.zeros(batch_size, c, h, h, device=device, dtype=dtype),
        }
