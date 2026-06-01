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

from ....conversation import (
    ConversationPart,
    TrainConversation,
    assemble_labels,
    collect_modality_values,
    is_raw_training_conversation,
    is_train_conversation,
)
from ....generation_graph import FSM_SIGNAL_KEY
from ....image_inputs import build_pixel_values_batch
from ....module import OmniModule
from .configuration import JanusVqvaeConfig
from .processing import JanusVqvaeProcessor


# Default Janus-1.3B grid: 24 x 24 = 576 VQ tokens per image.
_DEFAULT_NUM_IMAGE_TOKENS = 576


def _raw_of(conversation: Any) -> Optional[List[List[dict]]]:
    """Raw ``list[list[dict]]`` view of a training conversation, else ``None``.

    Accepts either the :class:`TrainConversation` carrier (built upstream by
    ``JanusSiglip``) or the bare raw conversation (standalone / test path).
    """
    if is_train_conversation(conversation):
        return conversation.raw
    if is_raw_training_conversation(conversation):
        return conversation
    return None


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
    # The inner ``JanusVQVAE`` declares gradient-checkpointing support, so the
    # mixin advertises it too (keeps the wrapper's capability accurate and lets
    # the trainer's GC guard pass).  Note: the VQVAE is frozen by default
    # (``freeze_vqvae``) and runs under ``no_grad`` in training, so GC here is
    # effectively inert — only the trainable generation_* heads see grads.
    supports_gradient_checkpointing = True

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

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(
        self,
        gen_image_patches: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract generation images (encode node) + ``(B, N, ...)`` flatten.

        On the **encode** node during training (``hidden_states is None`` and
        a raw ``conversation_list`` is present), pull every ``vq_image``
        turn's raw ``(C, H, W)`` uint8 tensor and process it into the
        VQVAE-normalised ``(B, 3, H, W)`` batch (zero placeholder for
        gen-image-free samples → keeps FSDP aligned).  The **decode** node
        (``hidden_states`` present) skips extraction and passes its tensors
        through untouched.
        """
        if gen_image_patches is None and hidden_states is None:
            conversation = kwargs.get("conversation_list")
            raw = _raw_of(conversation)
            if raw is not None:
                gen_image_patches = self._extract_gen_patches(raw)
                # Keep ``conversation_list`` in kwargs so :meth:`encode` can fill
                # the carrier's ``gen_embeds`` / ``gen_token_ids``.

        if gen_image_patches is not None and gen_image_patches.ndim == 5:
            b, n = gen_image_patches.shape[:2]
            gen_image_patches = gen_image_patches.reshape(b * n, *gen_image_patches.shape[2:])
            kwargs["_gpatch_batch_n_images"] = (b, n)

        result = dict(gen_image_patches=gen_image_patches, **kwargs)
        if hidden_states is not None:
            result["hidden_states"] = hidden_states
        return result

    def _extract_gen_patches(self, conversation_list: List[List[dict]]) -> torch.Tensor:
        """Raw ``vq_image`` turns → VQVAE-normalised ``(B, 3, H, W)`` batch."""
        cfg = self.config.vq_config or {}
        image_size = cfg.get("resolution", 384) if isinstance(cfg, dict) else getattr(cfg, "resolution", 384)
        num_channels = cfg.get("in_channels", 3) if isinstance(cfg, dict) else getattr(cfg, "in_channels", 3)
        per_sample = collect_modality_values(conversation_list, ("vq_image",))
        param = next(self.parameters())
        return build_pixel_values_batch(
            per_sample,
            processor=self._processor,
            image_size=image_size,
            num_channels=num_channels,
            device=param.device,
            dtype=param.dtype,
        )

    def forward(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        """Default forward — alias for :meth:`encode`."""
        return self.encode(**kwargs)

    def encode(self, gen_image_patches: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """VQVAE encode pass: pixels → ground-truth tokens + teacher-forcing embeds.

        Training (``conversation_list`` is a :class:`TrainConversation`): fill
        the carrier's ``gen_embeds`` ``(B, P, D)`` + ``gen_token_ids`` ``(B, P)``
        and route the carrier on (``{"conversation_list": carrier}``).  The text
        encoder slices per-sample rows into ``vq_image`` segments; the backbone
        uses the whole ``gen_embeds`` tensor as an FSDP grad-sync anchor.

        Legacy / pre-tensorised path returns
        ``{"gen_embeds", "vq_token_ids"}`` unchanged.  Returns ``{}`` for
        text-only batches.  When ``self.config.freeze_vqvae`` is ``True`` the
        VQVAE is wrapped in ``torch.no_grad()``.
        """
        conversation = kwargs.get("conversation_list")
        if gen_image_patches is None:
            return {"conversation_list": conversation} if is_train_conversation(conversation) else {}

        b_n = kwargs.pop("_gpatch_batch_n_images", None)

        with torch.no_grad() if self.config.freeze_vqvae else torch.enable_grad():
            vq_out = self.vqmodel.encode(gen_image_patches)
        vq_token_ids = vq_out.image_tokens

        gen_embeds_raw = self.generation_embeddings(vq_token_ids)
        gen_embeds = self.generation_aligner(gen_embeds_raw)

        # HF ``JanusVQVAE.encode`` returns the quantizer ``indices`` flattened
        # to ``(B*P,)`` (P = patches/grid positions per image), so
        # ``gen_embeds`` is ``(B*P, D)``.  Restore the per-sample ``(B, P[, D])``
        # layout the backbone scatter (``_scatter_by_mask``) and the gen-loss
        # alignment expect.
        if vq_token_ids.dim() == 1:
            b = gen_image_patches.size(0)
            vq_token_ids = vq_token_ids.reshape(b, -1)
            gen_embeds = gen_embeds.reshape(b, vq_token_ids.size(1), gen_embeds.size(-1))

        if b_n is not None:
            b, n = b_n
            p = vq_token_ids.size(1)
            vq_token_ids = vq_token_ids.reshape(b, n * p)
            gen_embeds = gen_embeds.reshape(b, n * p, gen_embeds.size(2))

        if is_train_conversation(conversation):
            conversation.gen_embeds = gen_embeds
            conversation.gen_token_ids = vq_token_ids
            return {"conversation_list": conversation}
        if is_raw_training_conversation(conversation):
            return {
                "conversation_list": TrainConversation(
                    raw=conversation, gen_embeds=gen_embeds, gen_token_ids=vq_token_ids
                )
            }
        return {"gen_embeds": gen_embeds, "vq_token_ids": vq_token_ids}

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        gt_token_ids: Optional[torch.Tensor] = None,
        gen_image_mask: Optional[torch.Tensor] = None,
        token_id: Optional[torch.Tensor] = None,
        conversation_list: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Unified VQ head — training loss + inference sample + lookup.

        Three input-driven dispatch paths:

          * **Training** (``hidden_states`` + ``gt_token_ids`` +
            ``gen_image_mask``): next-token VQ CE.  ``hidden_states`` is the
            *full-sequence* backbone output ``(B, T, D)``; ``gen_image_mask``
            ``(B, T)`` marks the ``<image_k>`` generation-grid positions and
            ``gt_token_ids`` ``(B, num_image_tokens)`` are the teacher VQ ids
            from :meth:`encode`.  We scatter the VQ ids into a full-length
            ``gen_labels`` (``-100`` elsewhere) and run the *same* shifted CE
            the text head uses, so ``hidden[<image_{k-1}>]`` predicts
            ``vq[k]`` (and ``hidden[<boi>]`` predicts ``vq[0]``).  Returns
            ``{"_loss"}``.

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

        # V2 training: gen-labels are built from the carrier's per-sample
        # ``vq_image`` segments (``gen_ids``); the backbone already assembled
        # ``hidden_states`` from the *same* segment order + right-pad, so the
        # shifted CE lines up position-for-position with no mask plumbing.
        if hidden_states is not None and is_train_conversation(conversation_list):
            segments = conversation_list.segments
            if segments is None:
                raise ValueError("JanusVqvae.decode: TrainConversation has no segments (text encoder must run first).")
            out["_loss"] = self._gen_loss_from_segments(hidden_states, segments)
            return out

        if hidden_states is not None and gt_token_ids is not None:
            out["_loss"] = self._gen_loss(hidden_states, gt_token_ids, gen_image_mask)
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

    def _gen_loss(
        self,
        hidden_states: torch.Tensor,
        gt_token_ids: torch.Tensor,
        gen_image_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Next-token VQ cross-entropy over the generation-grid positions.

        ``gen_image_mask`` aligns the teacher VQ ids onto the full sequence;
        only **present rows** (``mask.any``) contribute teacher ids, so the
        per-sample dummy-forward of text-only / understanding micro-batches
        stays a no-op while keeping the FSDP graph aligned (the trailing
        ``+ logits.sum() * 0`` anchor guarantees a grad path even when the
        whole micro-batch has zero gen tokens).
        """
        logits = self.generation_head(hidden_states)  # (B, T, V)
        b, t, v = logits.shape
        gen_labels = torch.full((b, t), -100, dtype=torch.long, device=logits.device)
        if gen_image_mask is not None:
            gen_image_mask = gen_image_mask.bool()
            gen_rows = gen_image_mask.any(dim=1)
            if bool(gen_rows.any()):
                gen_labels[gen_image_mask] = gt_token_ids[gen_rows].reshape(-1).long()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = gen_labels[:, 1:].contiguous()
        ce_sum = nn.functional.cross_entropy(
            shift_logits.reshape(-1, v),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_valid = (shift_labels != -100).sum().clamp(min=1)
        return ce_sum / n_valid + logits.sum() * 0.0

    def _gen_loss_from_segments(self, hidden_states: torch.Tensor, segments: List[List[Any]]) -> torch.Tensor:
        """Next-token VQ CE from carrier segments (V2 segment-driven path).

        ``gen_ids`` is ``-100`` everywhere except the ``vq_image`` segment
        positions, where it holds the teacher VQ token ids.  The standard
        causal shift then makes ``hidden[<boi>]`` predict ``vq[0]`` and
        ``hidden[vq_{k-1}]`` predict ``vq[k]``.  The trailing
        ``+ logits.sum() * 0`` keeps a grad path through the generation head
        even on a micro-batch with zero gen tokens (FSDP DP alignment).
        """
        gen_labels = assemble_labels(segments, key="gen_ids").to(hidden_states.device)  # (B, T)
        # Trim any SP right-pad so logits/labels align (no-op when SP off).
        hidden_states = hidden_states[:, : gen_labels.size(1)]
        logits = self.generation_head(hidden_states)  # (B, T, V)
        v = logits.size(-1)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = gen_labels[:, 1:].contiguous()
        ce_sum = nn.functional.cross_entropy(
            shift_logits.reshape(-1, v),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_valid = (shift_labels != -100).sum().clamp(min=1)
        return ce_sum / n_valid + logits.sum() * 0.0

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
