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
"""Block-autoregressive sampling for DiffusionGemma.

Two pieces, matching the two halves of the model. `entropy_bounded_denoise` turns one
canvas of uniform noise into text against a fixed KV cache; `block_autoregressive_generate`
runs the causal encoder around it so that generation can pass a single canvas.

Both follow the reference recipe rather than a simplification of it, because the two
differ in ways that are visible in the output:

* The acceptance set is **recomputed from scratch every step** from the current
  predictive entropies. A position accepted at one step can be re-noised at the next if
  the model grows less certain about it -- that self-correction within a canvas is the
  point of the uniform-state formulation, and a monotonic freeze forecloses it.
* Its size is set by an **entropy budget**, not by a fixed schedule: positions are
  accepted in order of increasing entropy while their cumulative entropy stays under
  ``entropy_budget``. Easy canvases commit many positions per forward, hard ones few.
* Every step feeds the model its own previous prediction (**self-conditioning**), the
  same signal `DllmTrainer` shows the model during finetuning.

Only the canvas being denoised is in play at any moment: once committed, a canvas goes
through the causal encoder into the KV cache and its tokens are final. That is what makes
the cache append-only, and it is why the sliding-attention layers may keep only their
last ``sliding_window`` entries -- the canvas always sits at the very end of the sequence.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache


__all__ = [
    "EntropyBoundedSamplerConfig",
    "block_autoregressive_generate",
    "entropy_bounded_denoise",
]


@dataclass
class EntropyBoundedSamplerConfig:
    """Knobs of the entropy-bounded sampler, with the reference defaults.

    Attributes:
        num_steps: Denoising-step ceiling for a canvas. Adaptive stopping ends a canvas
            well before this in practice; see `stop_threshold` for measured counts.
        entropy_budget: Total predictive entropy, in nats, that one step may commit.
            Larger values accept more tokens per forward pass at the cost of quality.
        stop_threshold: Mean per-position entropy below which a canvas counts as settled.
            The mean covers the whole canvas, including the positions past the answer, but
            those settle too rather than staying noisy, so this fires readily: measured on
            the 26B checkpoint at a 256-wide canvas, every canvas of a four-canvas answer
            stopped after 6 to 14 of the 48 allowed steps. Treat `num_steps` as a ceiling
            that is not normally reached.
        temperature_max / temperature_min: Linear temperature anneal against the diffusion
            time, which runs from 1 down to `1 / num_steps` rather than to 0 -- so the last
            step lands slightly above `temperature_min`. Below 1 throughout, sharpening the
            distribution rather than flattening it.
    """

    num_steps: int = 48
    entropy_budget: float = 0.1
    stop_threshold: float = 0.005
    temperature_max: float = 0.8
    temperature_min: float = 0.4


#: Called after each denoising step with ``(step, prediction, entropies)``.
StepCallback = Callable[[int, torch.Tensor, torch.Tensor], None]


def acceptance_mask(entropies: torch.Tensor, entropy_budget: float) -> torch.Tensor:
    """Which canvas positions this step commits, given their predictive entropies.

    Positions are ranked by increasing entropy and accepted while the entropy already
    committed stays within ``entropy_budget``; the bound is on the *preceding* ranks, so
    the single most confident position is always accepted and a step can never stall.
    Ties resolve by position index, via a stable sort.

    Args:
        entropies: ``(batch, canvas_length)`` per-position entropy in nats.
        entropy_budget: Cumulative entropy a single step may commit.

    Returns:
        ``(batch, canvas_length)`` boolean mask over positions.
    """
    ordered, positions = torch.sort(entropies, dim=-1, stable=True)
    within_budget = (ordered.cumsum(dim=-1) - ordered) <= entropy_budget
    return torch.zeros_like(within_budget).scatter_(-1, positions, within_budget)


@torch.no_grad()
def entropy_bounded_denoise(
    model,
    past_key_values: Cache,
    decoder_attention_mask: torch.Tensor,
    config: EntropyBoundedSamplerConfig | None = None,
    generator: torch.Generator | None = None,
    on_step: StepCallback | None = None,
) -> tuple[torch.Tensor, int]:
    """Denoise one canvas against a fixed KV cache.

    Args:
        model: A `DiffusionGemmaForBlockDiffusion`.
        past_key_values: Encoder cache holding the prompt and any committed canvases. It
            is read, never written: the decoder does not extend it.
        decoder_attention_mask: ``(batch, cache_length + canvas_length)`` boolean mask
            marking which cache entries are real, with the canvas itself all-true.
        config: Sampler settings; the reference defaults when omitted.
        generator: Optional RNG for the initial canvas and the per-step draws. Must live
            on the same device as `decoder_attention_mask`.
        on_step: Optional progress hook.

    Returns:
        ``(prediction, steps_taken)`` where prediction is the ``(batch, canvas_length)``
        deterministic (argmax) reading of the final logits -- not the noisy canvas that
        produced them, which still carries re-noised positions.
    """
    config = config or EntropyBoundedSamplerConfig()
    canvas_length = model.config.canvas_length
    vocab_size = model.config.text_config.vocab_size
    device = decoder_attention_mask.device
    batch_size = decoder_attention_mask.shape[0]

    canvas = torch.randint(
        0, vocab_size, (batch_size, canvas_length), generator=generator, device=device, dtype=torch.long
    )
    self_conditioning_logits = None
    previous_prediction = None
    prediction = canvas

    for step in range(config.num_steps):
        logits = model(
            past_key_values=past_key_values,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_attention_mask,
            self_conditioning_logits=self_conditioning_logits,
            use_cache=False,
        ).logits

        # The deterministic reading is taken from the raw logits, so the anneal below
        # steers exploration without moving what the canvas would settle on.
        prediction = logits.argmax(dim=-1)
        remaining = 1.0 - step / config.num_steps
        temperature = (config.temperature_max - config.temperature_min) * remaining + config.temperature_min
        # Both the draw and the entropy accounting run on the annealed distribution, so an
        # early step -- where the temperature is highest -- also reports the highest
        # entropies and commits the fewest positions.
        probabilities = (logits / temperature).softmax(dim=-1, dtype=torch.float32)
        entropies = torch.special.entr(probabilities).sum(dim=-1)

        if on_step is not None:
            on_step(step, prediction, entropies)

        # Adaptive stopping: confident *and* unchanged. Either alone is too weak -- a
        # canvas can be confidently wrong early, and two steps can agree by luck while
        # the model is still moving. With a batch, every row has to agree, so stragglers
        # keep the settled rows in the loop; their predictions simply stop changing.
        settled = (entropies.mean(dim=-1) <= config.stop_threshold).all()
        if step > 0 and settled and torch.equal(prediction, previous_prediction):
            return prediction, step + 1
        previous_prediction = prediction

        if step == config.num_steps - 1:
            break

        # Untempered, and logits rather than probabilities because the decoder softmaxes
        # this itself. Training has no step index and so no temperature; feeding the
        # annealed logits here would hand the model a conditioning distribution up to
        # 2.5x sharper than any it was finetuned against. The anneal's job is to steer
        # which tokens get drawn and how the entropy budget is spent, not to redefine the
        # conditioning signal.
        self_conditioning_logits = logits
        sampled = torch.multinomial(probabilities.reshape(-1, vocab_size), num_samples=1, generator=generator).reshape(
            batch_size, canvas_length
        )
        accepted = acceptance_mask(entropies, config.entropy_budget)
        # Everything not accepted goes back to a uniform draw rather than keeping its
        # current value: an uncommitted position should re-enter the next forward pass as
        # a uniform prior, which is what forces the model to keep exploring it.
        renoised = torch.randint(
            0, vocab_size, (batch_size, canvas_length), generator=generator, device=device, dtype=torch.long
        )
        canvas = torch.where(accepted, sampled, renoised)

    return prediction, config.num_steps


@torch.no_grad()
def block_autoregressive_generate(
    model,
    prompt_inputs: dict[str, torch.Tensor],
    eos_token_ids: Sequence[int],
    config: EntropyBoundedSamplerConfig | None = None,
    max_blocks: int = 4,
    generator: torch.Generator | None = None,
    on_step: StepCallback | None = None,
) -> torch.Tensor:
    """Generate a response one canvas at a time, growing the KV cache as it goes.

    The prompt is encoded once into a cache; each canvas is then denoised against that
    cache, and once committed is pushed back through the *causal* encoder and appended
    (the encoder's causality is exactly what makes appending sound -- no earlier entry
    can depend on the canvas that follows it). Generation ends at the first end-of-turn
    token or after ``max_blocks`` canvases.

    Single-sequence only. Canvases finish at different steps and end at different
    tokens, so a batched loop would either have to keep finished rows spinning or
    special-case them at every boundary; neither is worth it for a path whose caller is
    a CLI. For the same reason this reaches into `model.model.encoder` directly, which
    bypasses the FSDP2 root module's pre-forward hook -- fine for a single-process CLI,
    but it would read unsharded parameters if this were ever called from a callback
    inside a distributed run.

    Args:
        model: A `DiffusionGemmaForBlockDiffusion`.
        prompt_inputs: Encoder inputs -- ``input_ids`` plus optionally ``attention_mask``
            and the image keys (``pixel_values``, ``mm_token_type_ids``,
            ``image_position_ids``).
        eos_token_ids: Token ids that end the model's turn.
        config: Sampler settings; the reference defaults when omitted.
        max_blocks: Ceiling on canvases, so a model that never emits end-of-turn stops.
        generator: Optional RNG, forwarded to the sampler.
        on_step: Optional progress hook, forwarded to the sampler.

    Returns:
        ``(1, generated_length)`` of generated token ids, truncated at (and including)
        the first end-of-turn token.
    """
    input_ids = prompt_inputs["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError(f"`block_autoregressive_generate` handles one sequence at a time, got {input_ids.shape[0]}.")

    canvas_length = model.config.canvas_length
    context_mask = prompt_inputs.get("attention_mask")
    if context_mask is None:
        context_mask = torch.ones_like(input_ids)
    context_mask = context_mask.bool()

    encoder_outputs = model.model.encoder(**{**prompt_inputs, "attention_mask": context_mask})
    past_key_values = encoder_outputs.past_key_values

    generated = []
    for block in range(max_blocks):
        decoder_attention_mask = torch.nn.functional.pad(context_mask, (0, canvas_length), value=True)
        canvas, _ = entropy_bounded_denoise(
            model,
            past_key_values=past_key_values,
            decoder_attention_mask=decoder_attention_mask,
            config=config,
            generator=generator,
            on_step=on_step,
        )

        end = _first_end_of_turn(canvas[0], eos_token_ids)
        generated.append(canvas[:, : end + 1] if end is not None else canvas)
        if end is not None or block == max_blocks - 1:
            break

        # Equation (4): commit the canvas by encoding it into the cache. The mask has to
        # span the cache plus the tokens being added, which is what the decoder mask
        # already is -- reuse it rather than rebuilding.
        model.model.encoder(input_ids=canvas, attention_mask=decoder_attention_mask, past_key_values=past_key_values)
        context_mask = decoder_attention_mask

    return torch.cat(generated, dim=-1)


def _first_end_of_turn(canvas: torch.Tensor, eos_token_ids: Sequence[int]) -> int | None:
    """Index of the earliest end-of-turn token in a canvas, or `None` if there is none."""
    if not len(eos_token_ids):
        return None
    is_end = torch.isin(canvas, torch.tensor(tuple(eos_token_ids), device=canvas.device, dtype=canvas.dtype))
    hits = is_end.nonzero(as_tuple=True)[0]
    return int(hits[0]) if hits.numel() else None
