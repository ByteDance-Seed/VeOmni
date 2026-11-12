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

import hashlib
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image

from veomni.utils import logging
from veomni.utils.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.registry import Registry


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from .chat_template import ChatTemplate


DATA_TRANSFORM_REGISTRY = Registry("DataTransform")
logger = logging.get_logger(__name__)


def _replace_gemma3_image_tokens(text: str, processor: "ProcessorMixin", num_crops: torch.Tensor, image_count: int):
    """Expand each original Gemma 3 image placeholder exactly once."""
    text_parts = text.split(processor.boi_token)
    if len(text_parts) != image_count + 1:
        raise ValueError(
            f"Gemma 3 chat template produced {len(text_parts) - 1} image placeholders for {image_count} images."
        )

    replacements = [
        processor.replace_image_token({"num_crops": num_crops}, image_idx) for image_idx in range(image_count)
    ]
    return text_parts[0] + "".join(
        replacement + suffix for replacement, suffix in zip(replacements, text_parts[1:], strict=True)
    )


def _get_exact_token_id(tokenizer: "PreTrainedTokenizer", token: str, fallback_token: str | None = None) -> int:
    candidates = (token,) if fallback_token is None else (token, fallback_token)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for candidate in candidates:
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        if token_id is not None and token_id != unk_token_id:
            return token_id

    token_names = " or ".join(candidates)
    raise ValueError(f"Cannot find token ({token_names}) in tokenizer vocab.")


def build_data_transform(transform_name: str, **kwargs) -> Callable:
    return partial(DATA_TRANSFORM_REGISTRY[transform_name], **kwargs)


def split_into_chunks(sequence: Sequence[int], chunk_size: int) -> List[List[int]]:
    """
    Splits a long sequence into chunks.
    """
    total_len = len(sequence)
    chunks = []
    for i in range(0, total_len, chunk_size):
        chunks.append(sequence[i : i + chunk_size])

    return chunks


@DATA_TRANSFORM_REGISTRY.register("plaintext")
def process_plaintext_example(
    example: Dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "content_split",
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    examples = []
    if isinstance(text_keys, str):
        text_example = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text_example = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    tokens = tokenizer.encode(text_example, add_special_tokens=False) + [tokenizer.eos_token_id]
    for input_ids in split_into_chunks(tokens, max_seq_len):
        examples.append(
            {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor([1] * len(input_ids)),
                "labels": torch.tensor(input_ids),
            }
        )

    return examples


@DATA_TRANSFORM_REGISTRY.register("conversation")
def process_conversation_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate",
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    if isinstance(text_keys, str):
        text_example = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text_example = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    tokenized_example = chat_template.encode_messages(text_example, max_seq_len=max_seq_len)
    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    return [tokenized_example]


def apply_dllm_masking(
    input_ids: "torch.Tensor",
    labels: "torch.Tensor",
    mask_token_id: int,
    mask_eps: float = 1e-3,
    generator: Optional["torch.Generator"] = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """LLaDA's absorbing-state forward process over one sample's answer tokens.

    Draws one mask ratio ``t ~ U(mask_eps, 1)`` for the sample and masks each answer
    token (``labels != IGNORE_INDEX``) with probability ``t``. Returns the corrupted
    ``input_ids``, ``labels`` kept only at masked positions, and ``loss_weight`` of
    ``1 / (t * N)`` there, ``N`` being the answer length, so that ``sum_i w_i * CE_i``
    is this sample's unbiased ELBO term and a batch averages it over samples.

    ``mask_eps`` bounds ``t`` away from 0, where the weight would explode. The training
    path passes a per-sample ``generator``; see `process_llada_example`.
    """
    answer = labels != IGNORE_INDEX
    num_answer = int(answer.sum())
    mask_ratio = (1.0 - mask_eps) * float(torch.rand((), generator=generator)) + mask_eps
    loss_weight = torch.zeros(labels.shape, dtype=torch.float32)
    if num_answer == 0:
        return input_ids, torch.full_like(labels, IGNORE_INDEX), loss_weight
    masked = (torch.rand(labels.shape, generator=generator) < mask_ratio) & answer
    input_ids = input_ids.masked_fill(masked, mask_token_id)
    labels = torch.where(masked, labels, torch.full_like(labels, IGNORE_INDEX))
    loss_weight = loss_weight.masked_fill(masked, 1.0 / (mask_ratio * num_answer))
    return input_ids, labels, loss_weight


@DATA_TRANSFORM_REGISTRY.register("llada")
def process_llada_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate",
    max_seq_len: int,
    mask_token_id: int = None,
    text_keys: Union[str, List[str]] = "messages",
    mask_eps: float = 1e-3,
    mask_seed: int = 0,
    mask_epoch: "int | DiffusionGemmaCorruptionEpoch" = 0,
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    """Chat-template a conversation and apply `apply_dllm_masking` to its answers.

    Used with the LLaDA-family templates (``chat_template: llada`` / ``llada_moe``);
    ``mask_token_id`` comes from the model config (`DllmTrainer` passes it).

    The mask is drawn from a generator keyed on ``(mask_seed, mask_epoch, tokens)``,
    for the reason `diffusion_gemma_noise_generator` gives: the process RNG is one
    stream shared by every rank. ``answer_mask`` marks the answer tokens, so that a
    sample whose draw masked nothing still counts toward the per-sample mean.
    """
    if mask_token_id is None:
        raise ValueError("`data_type: llada` needs the model's `mask_token_id`; the model config does not define one.")
    (tokenized,) = process_conversation_example(example, chat_template, max_seq_len, text_keys)
    generator = diffusion_gemma_noise_generator(
        diffusion_gemma_sample_key(mask_seed, mask_epoch, tokenized["input_ids"]), 0
    )
    input_ids, labels, loss_weight = apply_dllm_masking(
        tokenized["input_ids"], tokenized["labels"], mask_token_id, mask_eps, generator=generator
    )
    return [
        {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "loss_weight": loss_weight,
            "answer_mask": (tokenized["labels"] != IGNORE_INDEX).long(),
        }
    ]


def _llava_image_first(message: Dict[str, Any]) -> Dict[str, Any]:
    """LLaVA's ``preprocess_multimodal``: a lone ``<image>`` in a string turn moves to its front.

    LLaDA-V was fine-tuned on that normalization, ``<image>\\n{text.strip()}``, wherever the
    placeholder sat in the raw data. Several placeholders and typed content parts are kept
    in place.
    """
    key = "content" if "content" in message else "value"
    content = message.get(key)
    if not isinstance(content, str) or content.count(IMAGE_PLACEHOLDER) != 1:
        return message
    return {**message, key: IMAGE_PLACEHOLDER + "\n" + content.replace(IMAGE_PLACEHOLDER, "").strip()}


@DATA_TRANSFORM_REGISTRY.register("llada_v")
def process_llada_v_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate",
    max_seq_len: int,
    mask_token_id: int = None,
    image_processor: Any = None,
    image_seq_len: int = None,
    text_keys: Union[str, List[str]] = "messages",
    image_key: str = "images",
    mask_eps: float = 1e-3,
    mask_seed: int = 0,
    mask_epoch: "int | DiffusionGemmaCorruptionEpoch" = 0,
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    """`process_llada_example` for image-text conversations (``chat_template: llada``).

    Conversations are ShareGPT or OpenAI-style, with ``<image>`` placeholders in string
    content or typed parts in list content, and ``example[image_key]`` holds the images
    in order of appearance; see `_llava_image_first` for where a lone placeholder ends
    up. Each image is one ``image_processor`` tile of
    ``image_seq_len`` slots, marked by ``image_mask``; the slots carry token id 0 and are
    never masked or supervised.

    An over-long sample is cut from the right. When that would cut into an image, the
    sample is replaced by a one-token placeholder with no answer, so it neither trains
    nor counts toward the per-sample mean.
    """
    from .multimodal.image_utils import load_image

    if mask_token_id is None:
        raise ValueError(
            "`data_type: llada_v` needs the model's `mask_token_id`; the model config does not define one."
        )
    if image_processor is None or image_seq_len is None:
        raise ValueError("`data_type: llada_v` needs the image processor and the model's `image_seq_len`.")
    if isinstance(text_keys, str):
        text_keys = [text_keys]
    for key in text_keys:
        if key in example:
            messages = example[key]
            break
    else:
        raise ValueError(f"None of the keys {text_keys} are found in the example.")

    raw_images = example.get(image_key) or []
    if not isinstance(raw_images, (list, tuple)):
        raw_images = [raw_images]
    images = [image if isinstance(image, Image.Image) else load_image(image) for image in raw_images]
    messages = _to_processor_messages([_llava_image_first(message) for message in messages], images)
    num_image_parts = sum(
        part["type"] == "image" for message in messages for part in message["content"] if isinstance(part, dict)
    )
    if num_image_parts != len(images):
        raise ValueError(f"The conversation has {num_image_parts} image part(s) but {len(images)} image(s).")

    tokenized = {
        k: torch.tensor(v) for k, v in chat_template.encode_multimodal_messages(messages, image_seq_len).items()
    }
    if tokenized["input_ids"].numel() > max_seq_len:
        if bool((tokenized["input_ids"][max_seq_len:] == IMAGE_INPUT_INDEX).any()):
            logger.warning_once(
                f"Dropping LLaDA-V samples whose images do not fit in max_seq_len={max_seq_len} tokens."
            )
            return [_unsupervised_placeholder(tokenized["input_ids"][:1])]
        tokenized = {k: v[:max_seq_len] for k, v in tokenized.items()}

    image_mask = tokenized["input_ids"] == IMAGE_INPUT_INDEX
    generator = diffusion_gemma_noise_generator(
        diffusion_gemma_sample_key(mask_seed, mask_epoch, tokenized["input_ids"]), 0
    )
    input_ids, labels, loss_weight = apply_dllm_masking(
        tokenized["input_ids"], tokenized["labels"], mask_token_id, mask_eps, generator=generator
    )
    sample = {
        "input_ids": input_ids.masked_fill(image_mask, 0),
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
        "loss_weight": loss_weight,
        "answer_mask": (tokenized["labels"] != IGNORE_INDEX).long(),
        "image_mask": image_mask,
    }
    if images:
        sample["pixel_values"] = image_processor(
            images=[image.convert("RGB") for image in images], return_tensors="pt"
        )["pixel_values"]
    return [sample]


def _unsupervised_placeholder(first_token: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
    """A one-token sample with no answer: it neither trains nor counts toward the per-sample mean."""
    return {
        "input_ids": first_token,
        "attention_mask": torch.tensor([1]),
        "labels": torch.tensor([IGNORE_INDEX]),
        "loss_weight": torch.zeros(1),
        "answer_mask": torch.zeros(1, dtype=torch.long),
        "image_mask": torch.zeros(1, dtype=torch.bool),
    }


@DATA_TRANSFORM_REGISTRY.register("llada2_uni")
def process_llada2_uni_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate",
    max_seq_len: int,
    mask_token_id: int = None,
    image_processor: Any = None,
    text_keys: Union[str, List[str]] = "messages",
    image_key: str = "images",
    mask_eps: float = 1e-3,
    mask_seed: int = 0,
    mask_epoch: "int | DiffusionGemmaCorruptionEpoch" = 0,
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    """`process_llada_v_example` for LLaDA2.0-Uni (``chat_template: llada2_uni``).

    Each image becomes a ``(1, h, w)`` grid of GlmImage patches (see `_llada2_uni_patchify`)
    and takes ``h*w`` ``image_mask`` slots, which the model fills with the image's VQ
    token ids. The pixels go out as packed patches with their ``image_grid_thw``.

    A sample is replaced by a one-token unsupervised placeholder when an image cannot be
    patchified or does not fit in ``max_seq_len``; otherwise an over-long sample is cut
    from the right.
    """
    from .multimodal.image_utils import load_image

    if mask_token_id is None:
        raise ValueError(
            "`data_type: llada2_uni` needs the model's `mask_token_id`; the model config does not define one."
        )
    if image_processor is None:
        raise ValueError("`data_type: llada2_uni` needs the GlmImage image processor.")
    if isinstance(text_keys, str):
        text_keys = [text_keys]
    for key in text_keys:
        if key in example:
            messages = example[key]
            break
    else:
        raise ValueError(f"None of the keys {text_keys} are found in the example.")

    raw_images = example.get(image_key) or []
    if not isinstance(raw_images, (list, tuple)):
        raw_images = [raw_images]
    images = [image if isinstance(image, Image.Image) else load_image(image) for image in raw_images]
    messages = _to_processor_messages([_llava_image_first(message) for message in messages], images)
    num_image_parts = sum(
        part["type"] == "image" for message in messages for part in message["content"] if isinstance(part, dict)
    )
    if num_image_parts != len(images):
        raise ValueError(f"The conversation has {num_image_parts} image part(s) but {len(images)} image(s).")

    try:
        pixel_values, image_grid_thw = _llada2_uni_patchify(images, image_processor, chat_template)
    except ValueError as error:
        logger.warning_once(f"Dropping LLaDA2.0-Uni samples whose images cannot be tokenized: {error}")
        role_open = chat_template.tokenizer.convert_tokens_to_ids(chat_template.role_open)
        return [_unsupervised_placeholder(torch.tensor([role_open]))]
    grids = [(h, w) for _, h, w in image_grid_thw.tolist()]

    tokenized = {k: torch.tensor(v) for k, v in chat_template.encode_multimodal_messages(messages, grids).items()}
    if tokenized["input_ids"].numel() > max_seq_len:
        if bool((tokenized["input_ids"][max_seq_len:] == IMAGE_INPUT_INDEX).any()):
            logger.warning_once(
                f"Dropping LLaDA2.0-Uni samples whose images do not fit in max_seq_len={max_seq_len} tokens."
            )
            return [_unsupervised_placeholder(tokenized["input_ids"][:1])]
        tokenized = {k: v[:max_seq_len] for k, v in tokenized.items()}

    image_mask = tokenized["input_ids"] == IMAGE_INPUT_INDEX
    generator = diffusion_gemma_noise_generator(
        diffusion_gemma_sample_key(mask_seed, mask_epoch, tokenized["input_ids"]), 0
    )
    input_ids, labels, loss_weight = apply_dllm_masking(
        tokenized["input_ids"], tokenized["labels"], mask_token_id, mask_eps, generator=generator
    )
    sample = {
        "input_ids": input_ids.masked_fill(image_mask, 0),
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
        "loss_weight": loss_weight,
        "answer_mask": (tokenized["labels"] != IGNORE_INDEX).long(),
        "image_mask": image_mask,
    }
    if images:
        sample["pixel_values"] = pixel_values
        sample["image_grid_thw"] = image_grid_thw
    return [sample]


def _llada2_uni_patchify(
    images: List["Image.Image"], image_processor: Any, chat_template: "ChatTemplate"
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """GlmImage patches and ``(1, h, w)`` grids for each image, packed in order.

    The processor's resize gives each image a patch grid within its pixel budget, but
    the template can only spell a grid side that has a ``<|reserved_token_*|>``; the
    LLaDA2.0-Uni vocabulary has none for odd sides up to 35. Such an image is instead
    resized to the next spellable side. Raises ``ValueError`` for an image the
    processor refuses (aspect ratio above 4, or a side below one patch).
    """
    from transformers.models.glm_image.image_processing_glm_image import smart_resize

    if not images:
        return torch.zeros(0), torch.zeros(0, 3, dtype=torch.long)
    patch_size = image_processor.patch_size
    size = image_processor.size
    pixel_values, grids = [], []
    for image in images:
        image = image.convert("RGB")
        height, width = smart_resize(
            image.height,
            image.width,
            factor=patch_size * image_processor.merge_size,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        grid = (height // patch_size, width // patch_size)
        spellable = tuple(
            next((n for n in range(side, side + 8) if chat_template.grid_token_id(n) is not None), None)
            for side in grid
        )
        if None in spellable:
            raise ValueError(f"no reserved token spells an image grid near {grid[0]}x{grid[1]}")
        if spellable == grid:
            out = image_processor(images=[image], return_tensors="pt", do_resize=True)
        else:
            image = image.resize((spellable[1] * patch_size, spellable[0] * patch_size), Image.BICUBIC)
            out = image_processor(images=[image], return_tensors="pt", do_resize=False)
        if tuple(out["image_grid_thw"][0, 1:].tolist()) != spellable:
            raise ValueError(f"the processor patchified to {out['image_grid_thw'][0].tolist()}, expected {spellable}")
        pixel_values.append(out["pixel_values"])
        grids.append(out["image_grid_thw"])
    return torch.cat(pixel_values), torch.cat(grids)


_U64_MASK = 0xFFFFFFFFFFFFFFFF


class DiffusionGemmaCorruptionEpoch:
    """Epoch counter shared between the trainer and the data transform.

    A plain int cannot work: the dataset builders re-wrap the transform in a second
    `functools.partial`, which flattens the nested one by *copying* its keyword dict,
    so a write to the trainer's copy never lands. The copy still points at this object.
    DataLoader workers snapshot it on fork, so the trainer has to write the epoch
    before the epoch's iterator is created.
    """

    def __init__(self, epoch: int = 0):
        self.epoch = int(epoch)


def _corruption_epoch_value(corruption_epoch: "int | DiffusionGemmaCorruptionEpoch") -> int:
    """Unwrap the shared counter; tests pass a bare int.

    Matched by type rather than by an ``epoch`` attribute, which `TrainerState` has too.
    """
    if isinstance(corruption_epoch, DiffusionGemmaCorruptionEpoch):
        return corruption_epoch.epoch
    return int(corruption_epoch)


def diffusion_gemma_sample_key(
    corruption_seed: int,
    corruption_epoch: "int | DiffusionGemmaCorruptionEpoch",
    encoder_ids: "torch.Tensor",
) -> bytes:
    """Fold the run seed, the epoch and the sample's own tokens into one digest.

    Keying on what a sample *is* rather than on where it sits in the stream is what
    makes the draw independent of which rank, worker or pack it lands in, and identical
    across a resume. The epoch is what stops a sample from redrawing the same
    corruption -- and, in single-canvas mode, supervising the same block -- all run.

    Equal keys share a draw: token-for-token duplicate samples always, and a sample
    revisited inside one epoch, which needs `dyn_bsz` (false on both recipes). Empty
    ``encoder_ids`` collapses to ``(seed, epoch)``, and those are the zero-weight
    placeholders.

    A stable hash, not `hash()`: the key has to agree across DataLoader worker
    processes, which `PYTHONHASHSEED` randomization would break.
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update((int(corruption_seed) & _U64_MASK).to_bytes(8, "little"))
    digest.update((_corruption_epoch_value(corruption_epoch) & _U64_MASK).to_bytes(8, "little"))
    digest.update(encoder_ids.to("cpu", torch.int64).numpy().tobytes())
    return digest.digest()


def diffusion_gemma_noise_generator(sample_key: bytes, canvas_index: int) -> "torch.Generator":
    """CPU generator for one canvas' corruption draw.

    Every draw on this path must come from here. All ranks call `set_seed(train.seed)`
    with no rank offset and workers seed from that same base, so the process RNG is one
    shared stream; the draws have fixed shapes, so the n-th sample of the n-th worker
    would get a bit-identical ``t``, corruption mask and noise on *every* rank. Seeding
    per ``(sample_key, canvas)`` also drops the dependence on how many draws came
    before, which drifts between ranks as soon as two samples span a different K.

    The CPU generator keeps only the seed's low 32 bits, so distinct canvases collide
    at roughly the birthday rate over 2**32 -- about one pair per 100k canvases. A
    collision costs the noise diversity that pair would have added, the same price as
    a duplicate sample, so the digest's high half is left as headroom.
    """
    digest = hashlib.blake2b(sample_key + (int(canvas_index) & _U64_MASK).to_bytes(8, "little"), digest_size=8)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest.digest(), "little"))
    return generator


def _build_diffusion_gemma_canvas(
    answer_ids: "torch.Tensor",
    answer_labels: "torch.Tensor",
    canvas_length: int,
    vocab_size: int,
    noise_eps: float,
    generator: "torch.Generator",
    eos_token_id: int = 1,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Lay the answer into a fixed-length canvas and corrupt it.

    **Corruption is uniform-random, not mask-based.** DiffusionGemma is a uniform-state
    discrete diffusion model: at inference the canvas starts as uniformly random token
    ids and every non-accepted position is re-randomized each step (see
    ``EntropyBoundSampler.initialize_canvas`` / ``renoise_canvas`` upstream). There is
    no ``<mask>`` token anywhere in the sampling loop, so training with an absorbing
    mask token -- an absorbing-mask dLLM recipe -- would show the decoder inputs it never sees at
    inference. Each canvas position is therefore replaced by a uniformly random token
    with probability ``t ~ U(noise_eps, 1)``, drawn once per sample.

    Loss covers **every supervised** canvas position, not just the corrupted ones,
    because the sampler draws from the logits of all positions on every denoising step
    and only then decides which to accept. ``answer_labels`` is the label slice, so
    positions the chat template masked out (later user turns in a multi-turn sample)
    keep their ``IGNORE_INDEX`` and stay out of the loss.

    Weights are ``1/N`` over the ``N`` supervised tokens, so every canvas contributes
    its mean per-token cross-entropy and every sample counts once, whatever the length
    of its answer. That is the ``1/C`` of the paper's Eq. 13, which normalizes the
    drawn canvas alone and does not scale by how many canvases the answer spans. An
    earlier version scaled by the block count, on the argument that a ``B``-canvas
    answer otherwise contributes ``1/B`` of its per-block sum; that reading belongs to
    the multi-canvas objective of Eq. 11, and paying for it here bought nothing but
    gradient norms in the hundreds and a loss reading that tracked answer lengths.

    A short last canvas is filled with EOS, matching official Gemma4 ``CanvasChunker``:
    those positions stay in ``canvas_mask`` and are real diffusion (and encoder AR)
    targets. PAD is reserved for left-padding prompts in the collator; a run of pad
    ids inside a bidirectional canvas would leak a train/inference discrepancy for
    the same reason mask-based corruption is rejected above. Callers EOS-pad the
    answer before the encoder layout so both stacks see the same tail; the fill here
    is the fallback if a short slice still arrives. An empty slice is left as noise
    with ``IGNORE_INDEX`` labels so a zero-weight sample does not become a canvas of
    supervised EOS.

    Shared by the text-only and the text+image transforms so both train the exact same
    objective; only the prompt side differs between them.

    ``generator`` carries every draw here; see `diffusion_gemma_noise_generator` for why
    the process RNG is not an option.
    """
    n_answer = int(answer_ids.numel())
    canvas = torch.randint(0, vocab_size, (canvas_length,), dtype=torch.long, generator=generator)
    canvas_labels = torch.full((canvas_length,), IGNORE_INDEX, dtype=torch.long)
    if n_answer == 0:
        return canvas, canvas_labels, torch.zeros(canvas_length, dtype=torch.float32)

    n = min(n_answer, canvas_length)
    canvas[:n] = answer_ids[:n]
    canvas_labels[:n] = answer_labels[:n]
    if n < canvas_length:
        canvas[n:] = int(eos_token_id)
        canvas_labels[n:] = int(eos_token_id)
    supervised = canvas_labels != IGNORE_INDEX
    num_supervised = int(supervised.sum())
    loss_weight = torch.zeros(canvas_length, dtype=torch.float32)
    if num_supervised:
        loss_weight[supervised] = 1.0 / num_supervised

    noise_ratio = (1.0 - noise_eps) * float(torch.rand((), generator=generator)) + noise_eps
    corrupt = torch.rand(canvas_length, generator=generator) < noise_ratio
    noise = torch.randint(0, vocab_size, (canvas_length,), dtype=torch.long, generator=generator)
    canvas = torch.where(corrupt, noise, canvas)
    return canvas, canvas_labels, loss_weight


def _shift_answer_to_canvas_start(
    prompt_ids: "torch.Tensor",
    answer_ids: "torch.Tensor",
    answer_labels: "torch.Tensor",
    canvas_start: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Move the first ``canvas_start`` answer tokens into the prompt.

    ``canvas_start`` is a 0-based offset into the answer: 0 keeps the current layout
    (canvas 0 is the first token after the prompt), ``n`` starts the canvas grid at
    the n-th answer token. The skipped tokens stay in the encoder as already-committed
    context and are never the noisy canvas the decoder denoises.
    """
    if canvas_start < 0:
        raise ValueError(f"`canvas_start` must be >= 0, got {canvas_start}.")
    n = min(int(canvas_start), int(answer_ids.numel()))
    if n == 0:
        return prompt_ids, answer_ids, answer_labels
    return torch.cat([prompt_ids, answer_ids[:n]]), answer_ids[n:], answer_labels[n:]


def _diffusion_gemma_block_count(num_answer_tokens: int, canvas_length: int, num_blocks_allowed: int) -> int:
    """How many canvases of the answer this sample can draw from.

    The answer spans ``ceil(num_answer_tokens / canvas_length)`` canvases, capped by
    ``num_blocks_allowed``. Never returns 0, including for an empty answer, so callers
    never have to special-case the draw.
    """
    return max(1, min(-(-num_answer_tokens // canvas_length), num_blocks_allowed))


def _pad_answer_to_canvas_grid(
    answer_ids: "torch.Tensor",
    answer_labels: "torch.Tensor",
    canvas_length: int,
    eos_token_id: int,
    max_canvases: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Fill the last canvas with supervised EOS, matching official ``CanvasChunker``.

    Official Gemma4 training writes EOS from ``num_response_tokens`` to ``canvas_end``
    and keeps ``canvas_mask`` True on those positions, so the last block is a full
    bidirectional canvas whose tail is a real target for both the diffusion loss and
    the encoder AR term. PAD is for batching the prompt, not for filling a canvas.

    An empty answer is left empty: filling it would invent a canvas of EOS and turn a
    zero-weight sample into a supervised one. Answers longer than ``max_canvases``
    canvases are truncated to that grid; a kept last canvas that is already full is
    not padded.
    """
    n = int(answer_ids.numel())
    if n == 0:
        return answer_ids, answer_labels
    target = _diffusion_gemma_block_count(n, canvas_length, max_canvases) * canvas_length
    if n >= target:
        return answer_ids[:target], answer_labels[:target]
    pad = torch.full((target - n,), int(eos_token_id), dtype=answer_ids.dtype, device=answer_ids.device)
    return torch.cat([answer_ids, pad]), torch.cat([answer_labels, pad.clone()])


def _strict_encoder_layout(
    prompt_ids: "torch.Tensor",
    answer_ids: "torch.Tensor",
    canvas_length: int,
    max_canvases: int,
    max_seq_len: int,
    truncate_prompt: bool,
) -> tuple["torch.Tensor", int]:
    """Eq. 13's encoder sequence: prompt plus every canvas that fits.

    The encoder always sees the full ``P + K C`` prefix that fits in ``max_seq_len``.
    Shrinking ``K`` to fit the budget drops later blocks from both terms rather than
    putting them in the encoder and hoping a mask will hide them -- a mask bug would
    leak the answer into the canvas representations and the loss would only look better.

    This function does not draw a canvas or build ``decoder_context_mask``. A single
    ``prefix_len = prompt + k C`` would be wrong once ``compute_all_canvas_df_loss``
    denoises every encoder block: each canvas needs its own prefix. Both the Eq. 13
    draw and the per-block masks live in ``_build_diffusion_gemma_decoder_targets``.
    """
    n_prompt = int(prompt_ids.numel())
    n_answer = int(answer_ids.numel())
    k_max = _diffusion_gemma_block_count(n_answer, canvas_length, max_canvases)
    reserved_prompt = 1 if truncate_prompt else n_prompt
    blocks_in_encoder = k_max
    while blocks_in_encoder > 0 and reserved_prompt + blocks_in_encoder * canvas_length > max_seq_len:
        blocks_in_encoder -= 1

    answer_in_encoder = answer_ids[: blocks_in_encoder * canvas_length] if blocks_in_encoder > 0 else answer_ids[:0]

    if truncate_prompt:
        prompt_budget = max(1, max_seq_len - int(answer_in_encoder.numel()))
        prompt_ids = prompt_ids[-prompt_budget:]

    return torch.cat([prompt_ids, answer_in_encoder]), blocks_in_encoder


def _decoder_context_mask_for_block(
    encoder_len: int, prompt_len: int, block: int, canvas_length: int
) -> "torch.Tensor":
    """1 on the prompt and canvases ``0..block-1``, 0 on the drawn block and after."""
    mask = torch.zeros(int(encoder_len), dtype=torch.long)
    # ``prompt_len + k C``: ``k < K`` stops before the drawn canvas; ``K == 0`` is
    # the whole encoder (``prompt_len == encoder_len``).
    mask[: int(prompt_len) + int(block) * int(canvas_length)] = 1
    return mask


def _build_diffusion_gemma_decoder_targets(
    answer_ids: "torch.Tensor",
    answer_labels: "torch.Tensor",
    encoder_ids: "torch.Tensor",
    canvas_length: int,
    vocab_size: int,
    noise_eps: float,
    eos_token_id: int,
    blocks_in_encoder: int,
    compute_all_canvas_df_loss: bool,
    sample_key: bytes,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Build the decoder canvas(es) and the matching encoder-side masks.

    Default (``compute_all_canvas_df_loss=False``) is Eq. 13: one block drawn
    uniformly from those in the encoder, 1-D tensors of length ``canvas_length``.
    When the flag is on, every encoder block is denoised: the four decoder tensors
    are stacked as ``(K, C)`` / ``(K, L)`` and each canvas's ``1/C`` weights are
    divided by ``K``, so a later sum over those rows is ``(1/K) Σ_k mean_CE_k``.
    Each row's context mask stops at that block (``prefix_len = prompt + k C``).
    ``encoder_prompt_mask`` stays 1-D: the prompt span does not change with ``k``.

    Every draw runs on a generator derived from ``sample_key``, one per canvas, so two
    samples never share a corruption pattern and no rank's stream depends on another's.
    See `diffusion_gemma_sample_key` / `diffusion_gemma_noise_generator`.
    """
    encoder_len = int(encoder_ids.numel())
    n_answer = int(blocks_in_encoder) * int(canvas_length)
    prompt_len = encoder_len - n_answer if blocks_in_encoder > 0 else encoder_len
    if compute_all_canvas_df_loss and blocks_in_encoder > 0:
        block_ids = list(range(blocks_in_encoder))
    elif blocks_in_encoder > 0:
        # `-1`: the block draw is not one of the canvases, so it gets its own slot in
        # the key space rather than sharing canvas 0's generator.
        block_generator = diffusion_gemma_noise_generator(sample_key, -1)
        block_ids = [int(torch.randint(0, blocks_in_encoder, (), generator=block_generator))]
    else:
        block_ids = [0]

    scale = (1.0 / len(block_ids)) if compute_all_canvas_df_loss else 1.0
    canvases, labels, weights, masks = [], [], [], []
    for block in block_ids:
        start = block * canvas_length
        canvas, canvas_labels, loss_weight = _build_diffusion_gemma_canvas(
            answer_ids[start : start + canvas_length],
            answer_labels[start : start + canvas_length],
            canvas_length,
            vocab_size,
            noise_eps,
            diffusion_gemma_noise_generator(sample_key, block),
            eos_token_id,
        )
        canvases.append(canvas)
        labels.append(canvas_labels)
        weights.append(loss_weight * scale)
        masks.append(_decoder_context_mask_for_block(encoder_len, prompt_len, block, canvas_length))

    encoder_prompt_mask = torch.zeros(encoder_len, dtype=torch.bool)
    # ``prompt_len`` is ``encoder_len - K C`` (or ``encoder_len`` when ``K == 0``).
    encoder_prompt_mask[:prompt_len] = True
    if len(block_ids) == 1:
        return canvases[0], labels[0], weights[0], masks[0], encoder_prompt_mask
    return torch.stack(canvases), torch.stack(labels), torch.stack(weights), torch.stack(masks), encoder_prompt_mask


@DATA_TRANSFORM_REGISTRY.register("diffusion_gemma")
def process_diffusion_gemma_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate" = None,
    tokenizer: "PreTrainedTokenizer" = None,
    max_seq_len: int = 2048,
    canvas_length: int = 256,
    max_canvases: int = 8,
    vocab_size: int = None,
    noise_eps: float = 1e-3,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    text_keys: Union[str, List[str]] = "messages",
    canvas_start: int = 0,
    compute_all_canvas_df_loss: bool = False,
    corruption_seed: int = 0,
    corruption_epoch: Union[int, DiffusionGemmaCorruptionEpoch] = 0,
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    """Build one text-only block-diffusion training sample for DiffusionGemma.

    The answer is cut into ``canvas_length`` blocks and **one** of them is drawn to be
    the canvas this sample supervises. A last canvas shorter than ``canvas_length`` is
    filled with supervised EOS (not PAD or noise), matching official ``CanvasChunker``.
    The encoder is handed the prompt followed by *every* clean block that fits -- Eq.
    13's ``P + K C`` sequence -- so the autoregressive term covers the whole answer,
    including that EOS tail. The decoder still denoises only the drawn block;
    ``decoder_context_mask`` hides that block and everything after it inside the
    encoder cache, which is the conditioning block-autoregressive generation actually
    feeds from the second canvas onwards.

    Drawing one diffusion canvas per sample rather than supervising all of them in a
    single decoder pass (a block-diagonal mask over concatenated canvases) keeps the
    noisy canvas a single segment at the end of the decoder sequence. The 25
    sliding-attention layers would otherwise have to read several canvases out of a
    cache that, if truncated, only keeps the tail.

    ``compute_all_canvas_df_loss`` keeps that one-segment layout: it stacks a noisy
    canvas and a per-block context mask as ``(K, C)`` / ``(K, L)``, and scales each
    canvas's ``1/C`` weights by ``1/K``. The encoder sequence is still emitted once.

    See `_build_diffusion_gemma_canvas` for the corruption and loss weighting.

    Returns a single dict with ``input_ids``/``attention_mask``/``decoder_context_mask``
    for the encoder context and ``decoder_input_ids``/``labels``/``loss_weight`` for
    the canvas.
    """
    if vocab_size is None:
        raise ValueError("`vocab_size` is required to sample uniform corruption noise.")

    if isinstance(text_keys, str):
        text_example = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text_example = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    if chat_template is None:
        raise ValueError("`chat_template` is required for the diffusion_gemma data transform.")

    # Do not let the generic chat template tail-truncate the conversation here.  A
    # long assistant answer can otherwise consume that entire window and erase the
    # prompt before the diffusion-specific layout gets a chance to cap the answer.
    # `_pad_answer_to_canvas_grid` and `_strict_encoder_layout` below own those two
    # limits and preserve prompt context for canvas zero.
    tokenized = chat_template.encode_messages(text_example, max_seq_len=sys.maxsize)
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    labels = torch.tensor(tokenized["labels"], dtype=torch.long)

    # The prompt is the ignored prefix; everything from the first supervised token on is
    # the answer the canvases have to reconstruct. Degenerate samples (nothing supervised,
    # or an answer that truncation pushed out of reach) must not return an empty list:
    # `MakeMicroBatchCollator` unconditionally indexes `features[i][0]`, so that would
    # take down the whole dataloader. They degrade to a zero-weight sample instead, which
    # `dllm_masked_cross_entropy` handles while keeping the backward graph intact.
    supervised = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    answer_start = int(supervised[0]) if supervised.numel() else int(input_ids.numel())
    answer_ids = input_ids[answer_start:]
    answer_labels = labels[answer_start:]
    prompt_ids, answer_ids, answer_labels = _shift_answer_to_canvas_start(
        input_ids[:answer_start], answer_ids, answer_labels, canvas_start
    )
    answer_ids, answer_labels = _pad_answer_to_canvas_grid(
        answer_ids, answer_labels, canvas_length, eos_token_id, max_canvases
    )

    encoder_ids, blocks_in_encoder = _strict_encoder_layout(
        prompt_ids,
        answer_ids,
        canvas_length,
        max_canvases,
        max_seq_len,
        truncate_prompt=True,
    )
    if encoder_ids.numel() == 0:
        encoder_ids = torch.full((1,), pad_token_id, dtype=torch.long)
        blocks_in_encoder = 0

    canvas, canvas_labels, loss_weight, decoder_context_mask, encoder_prompt_mask = (
        _build_diffusion_gemma_decoder_targets(
            answer_ids,
            answer_labels,
            encoder_ids,
            canvas_length,
            vocab_size,
            noise_eps,
            eos_token_id,
            blocks_in_encoder,
            compute_all_canvas_df_loss,
            diffusion_gemma_sample_key(corruption_seed, corruption_epoch, encoder_ids),
        )
    )
    return [
        {
            "input_ids": encoder_ids,
            "attention_mask": torch.ones(encoder_ids.numel(), dtype=torch.long),
            "decoder_context_mask": decoder_context_mask,
            "encoder_prompt_mask": encoder_prompt_mask,
            "decoder_input_ids": canvas,
            "labels": canvas_labels,
            "loss_weight": loss_weight,
        }
    ]


# ShareGPT-style conversations mark image slots inline with this token, while the
# processor's chat template expects typed content parts instead.
IMAGE_PLACEHOLDER = "<image>"


def _to_processor_messages(messages: Sequence[Dict[str, Any]], images: List[Any]) -> List[Dict[str, Any]]:
    """Convert VeOmni-style conversations into the content-list form the processor wants.

    Accepts the two shapes found in this repo's datasets -- ShareGPT (``from``/``value``)
    and OpenAI-style (``role``/``content``) -- and splits each string content on the
    ``<image>`` placeholder, consuming `images` in order of appearance. Content that is
    already a list of typed parts is passed through untouched.
    """
    role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}
    image_iter = iter(images)
    converted = []
    for message in messages:
        role = role_map.get(str(message.get("role", message.get("from", ""))).lower())
        if role is None:
            raise ValueError(f"Unrecognized conversation role in message: {message}")
        content = message.get("content", message.get("value"))
        if isinstance(content, list):
            converted.append({"role": role, "content": content})
            continue
        parts = []
        for index, chunk in enumerate(str(content).split(IMAGE_PLACEHOLDER)):
            if index > 0:
                try:
                    parts.append({"type": "image", "image": next(image_iter)})
                except StopIteration:
                    raise ValueError(
                        f"Conversation has more `{IMAGE_PLACEHOLDER}` placeholders than the {len(images)} "
                        "image(s) supplied."
                    ) from None
            if chunk:
                parts.append({"type": "text", "text": chunk})
        converted.append({"role": role, "content": parts})
    return converted


def _unusable_diffusion_gemma_sample(
    canvas_length: int,
    vocab_size: int,
    pad_token_id: int,
    corruption_seed: int,
    corruption_epoch: Union[int, DiffusionGemmaCorruptionEpoch],
) -> List[Dict[str, "torch.Tensor"]]:
    """Zero-weight placeholder for a sample the multimodal transform cannot use.

    Returning ``[]`` is not an option: `MakeMicroBatchCollator` indexes ``features[i][0]``
    unconditionally, so a single dropped sample would take down the whole dataloader.
    A fully-``IGNORE_INDEX`` canvas contributes no gradient instead.

    The canvas is unsupervised, so its contents are arbitrary, but it still comes from a
    derived generator: the process RNG would be advanced only on whichever rank hit an
    unusable sample.

    ``mm_token_type_ids`` is emitted even though there is no image: every usable sample on
    this path carries the tag, image or not, and `DiffusionGemmaCollator` rejects a batch
    that mixes samples with and without it. ``decoder_context_mask`` is required by the
    collator (an all-ones default would leak a clean canvas). ``pixel_values`` and
    ``image_position_ids`` are left out, which the collator does tolerate.
    """
    placeholder_ids = torch.full((1,), pad_token_id, dtype=torch.long)
    generator = diffusion_gemma_noise_generator(
        diffusion_gemma_sample_key(corruption_seed, corruption_epoch, placeholder_ids), 0
    )
    return [
        {
            "input_ids": placeholder_ids,
            "attention_mask": torch.ones(1, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, dtype=torch.long),
            "encoder_prompt_mask": torch.ones(1, dtype=torch.bool),
            "decoder_context_mask": torch.zeros(1, dtype=torch.long),
            "decoder_input_ids": torch.randint(0, vocab_size, (canvas_length,), dtype=torch.long, generator=generator),
            "labels": torch.full((canvas_length,), IGNORE_INDEX, dtype=torch.long),
            "loss_weight": torch.zeros(canvas_length, dtype=torch.float32),
        }
    ]


@DATA_TRANSFORM_REGISTRY.register("diffusion_gemma_mm")
def process_diffusion_gemma_mm_example(
    example: Dict[str, Any],
    processor: "ProcessorMixin" = None,
    max_seq_len: int = 2048,
    canvas_length: int = 256,
    max_canvases: int = 8,
    vocab_size: int = None,
    noise_eps: float = 1e-3,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    text_keys: Union[str, List[str]] = "messages",
    image_key: str = "images",
    canvas_start: int = 0,
    compute_all_canvas_df_loss: bool = False,
    corruption_seed: int = 0,
    corruption_epoch: Union[int, DiffusionGemmaCorruptionEpoch] = 0,
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    """Build one text+image block-diffusion training sample for DiffusionGemma.

    Images are an **encoder-only** conditioning signal: they live in the prompt, and the
    canvas the decoder denoises stays pure text. So this differs from the text-only
    transform only in how the prompt is produced and in how the context budget is
    enforced -- the block draw, the full ``P + K C`` encoder sequence, and the
    corruption are shared.

    Unlike the Qwen-VL transforms, this one drives HuggingFace's own
    ``processor.apply_chat_template`` instead of a hand-written VeOmni
    `MultimodalChatTemplate`. DiffusionGemma ships an 18KB Jinja template whose image
    handling (``<|image|>`` runs sized per image by the pooler) would have to be
    reimplemented byte-for-byte to keep training and inference consistent; reusing it is
    both less code and less drift. The cost is that we cannot ask for label masks --
    the template has no ``{% generation %}`` block -- so the prompt/answer boundary is
    recovered by encoding twice: once through the last user turn (with a generation
    prompt) and once over the whole conversation. The former is a strict prefix of the
    latter, so the remainder is exactly the answer.

    Emits the text-only keys plus ``pixel_values``, ``image_position_ids`` and
    ``mm_token_type_ids``.
    """
    # Imported lazily: `veomni.data.multimodal` imports this module back for its own
    # transform registrations, so a module-level import would be circular.
    from .multimodal.image_utils import load_image

    if vocab_size is None:
        raise ValueError("`vocab_size` is required to sample uniform corruption noise.")
    if processor is None:
        raise ValueError("`processor` is required for the diffusion_gemma_mm data transform.")

    if isinstance(text_keys, str):
        text_keys = [text_keys]
    for key in text_keys:
        if key in example:
            messages = example[key]
            break
    else:
        raise ValueError(f"None of the keys {text_keys} are found in the example.")

    raw_images = example.get(image_key) or []
    if not isinstance(raw_images, (list, tuple)):
        raw_images = [raw_images]
    images = [image if isinstance(image, Image.Image) else load_image(image) for image in raw_images]

    messages = _to_processor_messages(messages, images)
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        # The answer is taken to be the final turn, so anything else is unusable.
        return _unusable_diffusion_gemma_sample(
            canvas_length, vocab_size, pad_token_id, corruption_seed, corruption_epoch
        )

    encode_kwargs = {"tokenize": True, "return_dict": True, "return_tensors": "pt"}
    prompt = processor.apply_chat_template(messages[:-1], add_generation_prompt=True, **encode_kwargs)
    full = processor.apply_chat_template(messages, **encode_kwargs)

    prompt_ids = prompt["input_ids"][0]
    full_ids = full["input_ids"][0]
    n_prompt = int(prompt_ids.numel())
    if not torch.equal(full_ids[:n_prompt], prompt_ids):
        raise ValueError(
            "The prompt encoding is not a prefix of the full conversation encoding, so the "
            "answer boundary cannot be recovered. This means the chat template is not "
            "append-only and this transform needs to be revisited."
        )
    answer_ids = full_ids[n_prompt:]
    processor_prompt_len = n_prompt
    prompt_ids, answer_ids, _ = _shift_answer_to_canvas_start(prompt_ids, answer_ids, answer_ids, canvas_start)
    # Truncating the prompt from the left would cut image spans loose from the patches the
    # processor emitted, so an over-long prompt makes the sample unusable rather than
    # being trimmed. The budget then shrinks ``K`` so every canvas that is drawn still
    # fits in the encoder; blocks past that point go untrained rather than corrupting
    # the image spans. ``canvas_start`` tokens count as prompt for the encoder but do
    # not change the processor tensors, so the over-long check stays on the image-bearing
    # prefix. Empty answers skip EOS-fill so they stay unusable rather than becoming a
    # supervised canvas of EOS.
    if answer_ids.numel() == 0 or processor_prompt_len > max_seq_len:
        return _unusable_diffusion_gemma_sample(
            canvas_length, vocab_size, pad_token_id, corruption_seed, corruption_epoch
        )
    answer_ids, _ = _pad_answer_to_canvas_grid(answer_ids, answer_ids, canvas_length, eos_token_id, max_canvases)

    encoder_ids, blocks_in_encoder = _strict_encoder_layout(
        prompt_ids, answer_ids, canvas_length, max_canvases, max_seq_len, truncate_prompt=False
    )
    # The whole final turn is the supervised answer here, unlike the text-only transform
    # which has to honour the chat template's own label mask for multi-turn samples.
    canvas, canvas_labels, loss_weight, decoder_context_mask, encoder_prompt_mask = (
        _build_diffusion_gemma_decoder_targets(
            answer_ids,
            answer_ids,
            encoder_ids,
            canvas_length,
            vocab_size,
            noise_eps,
            eos_token_id,
            blocks_in_encoder,
            compute_all_canvas_df_loss,
            diffusion_gemma_sample_key(corruption_seed, corruption_epoch, encoder_ids),
        )
    )
    prompt_attention_mask = prompt.get("attention_mask", torch.ones(1, processor_prompt_len, dtype=torch.long))[0]
    answer_in_encoder = encoder_ids[processor_prompt_len:]
    sample = {
        "input_ids": encoder_ids,
        "attention_mask": torch.cat([prompt_attention_mask, torch.ones_like(answer_in_encoder)]),
        "decoder_context_mask": decoder_context_mask,
        "encoder_prompt_mask": encoder_prompt_mask,
        # Answer tokens are text, so they extend the modality tag with 0s; only the prompt
        # can carry image spans.
        "mm_token_type_ids": torch.cat([prompt["mm_token_type_ids"][0], torch.zeros_like(answer_in_encoder)]),
        "decoder_input_ids": canvas,
        "labels": canvas_labels,
        "loss_weight": loss_weight,
    }
    if images:
        sample["pixel_values"] = prompt["pixel_values"]
        sample["image_position_ids"] = prompt["image_position_ids"]
    return [sample]


@DATA_TRANSFORM_REGISTRY.register("dpo")
def process_dpo_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate" = None,
    tokenizer: "PreTrainedTokenizer" = None,
    max_seq_len: int = 2048,
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    """Process a DPO preference pair into a single flat sample.

    Chosen and rejected sequences are concatenated into one 1-D tensor with
    ``position_ids`` that reset at the boundary so that flash-attention treats
    them as two independent sequences.  This format is directly compatible with
    ``MainCollator`` (packing + SP) — no DPO-specific collator is needed.

    Supported input formats:
      1. Conversation: {"chosen": [messages...], "rejected": [messages...]}
      2. Plaintext with prompt: {"prompt": str, "chosen": str, "rejected": str}

    Returns:
        A list with one dict.  Each value is a 1-D tensor of length
        ``len_chosen + len_rejected``.  Keys: ``input_ids``, ``attention_mask``,
        ``labels``, ``position_ids``.
    """
    chosen_raw = example["chosen"]
    rejected_raw = example["rejected"]

    if isinstance(chosen_raw, list):
        assert chat_template is not None, "chat_template is required for conversation-format DPO data"
        chosen_tok = chat_template.encode_messages(chosen_raw, max_seq_len=max_seq_len)
        rejected_tok = chat_template.encode_messages(rejected_raw, max_seq_len=max_seq_len)
    else:
        assert tokenizer is not None, "tokenizer is required for plaintext-format DPO data"
        prompt = example.get("prompt", "")
        chosen_text = prompt + chosen_raw
        rejected_text = prompt + rejected_raw

        chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=True)[:max_seq_len]
        rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=True)[:max_seq_len]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True) if prompt else []
        prompt_len = len(prompt_ids)

        chosen_tok = {
            "input_ids": chosen_ids,
            "attention_mask": [1] * len(chosen_ids),
            "labels": [IGNORE_INDEX] * prompt_len + chosen_ids[prompt_len:],
        }
        rejected_tok = {
            "input_ids": rejected_ids,
            "attention_mask": [1] * len(rejected_ids),
            "labels": [IGNORE_INDEX] * prompt_len + rejected_ids[prompt_len:],
        }

    def _to_tensor(v):
        return v if isinstance(v, torch.Tensor) else torch.tensor(v)

    c_ids = _to_tensor(chosen_tok["input_ids"])
    r_ids = _to_tensor(rejected_tok["input_ids"])
    c_len = c_ids.shape[-1]
    r_len = r_ids.shape[-1]

    result = {
        "input_ids": torch.cat([c_ids, r_ids]),
        "attention_mask": torch.cat(
            [_to_tensor(chosen_tok["attention_mask"]), _to_tensor(rejected_tok["attention_mask"])]
        ),
        "labels": torch.cat([_to_tensor(chosen_tok["labels"]), _to_tensor(rejected_tok["labels"])]),
        "position_ids": torch.cat([torch.arange(c_len, dtype=torch.int64), torch.arange(r_len, dtype=torch.int64)]),
    }
    return [result]


@DATA_TRANSFORM_REGISTRY.register("classification")
def process_classification_example(
    example: dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    max_seq_len: int,
    text_keys: Union[str, list[str]] = "text",
    label_key: str = "label",
    **kwargs,
) -> list[dict[str, "torch.Tensor"]]:
    """
    Convert a single raw example into one classification training sample.

    Args:
        example:
            A single record from the dataset. Expected format (minimal):
                {
                    "<text_key>":  str,   # e.g. news article / sentence
                    "<label_key>": int,   # e.g. 0..(num_labels-1)
                    ...                   # other fields are ignored
                }
            By default:
                text_key  = "text"
                label_key = "label"

        tokenizer:
            A HuggingFace tokenizer used to tokenize the input text.

        max_seq_len:
            Maximum sequence length (in tokens). Text longer than this
            will be truncated to the first `max_seq_len` tokens.

        text_keys:
            Keys in `example` that contains the raw input text. If a list, the first key found in `example` will be used.

        label_key:
            Key in `example` that contains the class id. The value should be int-like.

    Returns:
        A list with exactly one sample dict:
            {
                "input_ids":      LongTensor[L],
                "attention_mask": LongTensor[L],
                "labels":         LongTensor[L],
                "position_ids":   LongTensor[L]
            }
    """
    # 1) text
    if isinstance(text_keys, str):
        text = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    # 2) label
    if label_key not in example:
        raise ValueError(f"Missing label key '{label_key}' in example.")
    try:
        label_val = int(example[label_key])
    except Exception as e:
        raise ValueError(f"Label '{example[label_key]}' is not an int-like value.") from e

    # 3) tokenize
    tokens: list[int] = tokenizer.encode(text, add_special_tokens=True)

    # 4) build samples
    examples: list[dict[str, torch.Tensor]] = []

    def build_sample(seq: list[int]) -> dict[str, "torch.Tensor"]:
        L = len(seq)
        token_labels = torch.full((L,), IGNORE_INDEX, dtype=torch.long)
        token_labels[L - 1] = label_val

        sample: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "attention_mask": torch.ones(len(seq), dtype=torch.long),
            "labels": token_labels,
        }
        sample["position_ids"] = torch.arange(len(seq), dtype=torch.long)
        return sample

    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]

    examples.append(build_sample(tokens))
    return examples


def _process_sample_qwen_vl_base(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    from .multimodal import conv_preprocess
    from .multimodal.image_utils import fetch_images
    from .multimodal.video_utils import fetch_videos_metadata

    source = kwargs.get("source_name") or sample.get("source") or sample.get("source_name")

    if "conversations" in sample and sample["conversations"] is not None and len(sample["conversations"]) > 0:
        conversations = sample["conversations"]
    else:
        conversations = sample
    conversations = conv_preprocess(source, conversations, **kwargs)

    token_num_inputs, image_inputs, video_inputs = {}, {}, {}
    image_grid_thw, video_grid_thw = None, None
    video_metadata = None

    if "images" in sample and sample["images"]:
        images = fetch_images(sample["images"], **kwargs)
        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_token_num = image_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["image"] = image_token_num

    if "videos" in sample and sample["videos"]:
        videos, metadata, _, _ = fetch_videos_metadata(sample["videos"], **kwargs)
        # Fetch already sampled the frames. Preserve its source indices/FPS for timestamps.
        video_inputs = processor.video_processor(
            videos=videos,
            video_metadata=metadata,
            do_sample_frames=False,
            return_tensors="pt",
            return_metadata=True,
        )
        video_grid_thw = video_inputs["video_grid_thw"]
        video_metadata = video_inputs.pop("video_metadata", None)

        merge_length = processor.video_processor.merge_size**2
        video_token_num = video_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["video"] = video_token_num

    # Encoding
    encode_kwargs = {}
    if video_metadata is not None:
        encode_kwargs["video_metadata"] = video_metadata

    tokenized_example = chat_template.encode_messages(conversations, token_num_inputs, **encode_kwargs)

    tokenized_example = {
        k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in tokenized_example.items()
    }

    input_ids = tokenized_example["input_ids"]
    attention_mask = tokenized_example["attention_mask"]

    # Masks and Token Types
    tokenized_example["image_mask"] = input_ids == IMAGE_INPUT_INDEX
    tokenized_example["video_mask"] = input_ids == VIDEO_INPUT_INDEX

    # Position IDs
    position_id_func_kwargs = {
        "input_ids": input_ids.unsqueeze(0),
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
        "attention_mask": attention_mask.unsqueeze(0),
    }

    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[tokenized_example["image_mask"]] = 1
    mm_token_type_ids[tokenized_example["video_mask"]] = 2
    tokenized_example["mm_token_type_ids"] = mm_token_type_ids
    position_id_func_kwargs["mm_token_type_ids"] = mm_token_type_ids.unsqueeze(0)

    position_id_returns = position_id_func(**position_id_func_kwargs)
    # Squeeze position_ids to match the per-sample (no batch dim) convention
    # used everywhere else in this dict.
    position_id_returns["position_ids"] = position_id_returns["position_ids"].squeeze().clone()
    # Only position_ids is propagated into the training feature dict. The
    # rope_deltas position_id_func also returns is generation-only (KV-cache
    # decode); the training forward always receives a precomputed
    # position_ids and never derives or reads rope_deltas.
    tokenized_example["position_ids"] = position_id_returns["position_ids"]

    # Final cleanup
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
    tokenized_example.update(image_inputs)
    tokenized_example.update(video_inputs)
    # image_inputs / video_inputs carry the HF processor's CPU `image_grid_thw`
    # / `video_grid_thw` tensors; the collator packs them (DataCollateInfo
    # pack_dim=0) and the model's metadata_collate_func hook derives the ViT
    # metadata from them. No per-sample `.tolist()` sidecar needed here.

    return [tokenized_example]


@DATA_TRANSFORM_REGISTRY.register("qwen2_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen2_5_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen3_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen3_vl_moe")
@DATA_TRANSFORM_REGISTRY.register("qwen3_5")
@DATA_TRANSFORM_REGISTRY.register("qwen3_5_moe")
@DATA_TRANSFORM_REGISTRY.register("qwen4_exp")
def process_sample_qwen_vl(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    """
    Unified processing function for Qwen-VL series models.
    Automatically determines whether to use mm_token_type_ids based on transformers version.
    """
    return _process_sample_qwen_vl_base(
        sample,
        processor,
        chat_template,
        position_id_func,
        **kwargs,
    )


@DATA_TRANSFORM_REGISTRY.register("gemma3")
def process_sample_gemma3(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    position_id_func: "Callable",
    max_seq_len: int = 8192,
    **kwargs,
):
    """Build a Gemma 3 multimodal SFT sample with assistant-only labels."""
    from .multimodal import conv_preprocess
    from .multimodal.image_utils import fetch_images

    source = kwargs.get("source_name") or sample.get("source") or sample.get("source_name")
    conversations = sample.get("conversations")
    if conversations is None:
        conversations = sample
    conversations = conv_preprocess(source, conversations, **kwargs)

    images = sample.get("images")
    if images is None:
        images = []
    images = fetch_images(images, **kwargs)
    if images:
        image_inputs = processor.image_processor(images=images, return_tensors="pt", do_pan_and_scan=False)
        num_crops = image_inputs.pop("num_crops", torch.zeros(len(images), dtype=torch.long))
    else:
        image_inputs = {}
        num_crops = torch.empty(0, dtype=torch.long)

    messages = []
    for conversation in conversations:
        role = conversation[0]
        content = []
        for value_type, value in conversation[1:]:
            if value_type == "image":
                content.append({"type": "image"})
            elif value_type == "text":
                content.append({"type": "text", "text": value})
            else:
                raise ValueError(f"Unsupported Gemma 3 content type: {value_type}")
        messages.append({"role": role, "content": content})

    def tokenize_prefix(end: int) -> torch.Tensor:
        prefix_messages = messages[:end]
        text = processor.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=False)
        image_count = sum(item["type"] == "image" for msg in prefix_messages for item in msg["content"])
        text = _replace_gemma3_image_tokens(text, processor, num_crops, image_count)
        return processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    input_ids = tokenize_prefix(len(messages))
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    previous_length = 0
    for end, message in enumerate(messages, start=1):
        current_length = tokenize_prefix(end).numel()
        if message["role"] == "assistant":
            labels[previous_length:current_length] = input_ids[previous_length:current_length]
        previous_length = current_length

    if input_ids.numel() > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    image_token_id = processor.tokenizer.image_token_id
    expected_image_tokens = len(images) * processor.image_seq_length
    actual_image_tokens = int((input_ids == image_token_id).sum())
    if actual_image_tokens != expected_image_tokens:
        raise ValueError(
            "Gemma 3 max_seq_len truncated an image placeholder: "
            f"expected {expected_image_tokens} image tokens, found {actual_image_tokens}."
        )

    attention_mask = torch.ones_like(input_ids)
    position_ids = position_id_func(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))[
        "position_ids"
    ].squeeze(0)

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_type_ids": (input_ids == image_token_id).long(),
        "labels": labels,
    }
    model_inputs.update(image_inputs)
    return [model_inputs]


@DATA_TRANSFORM_REGISTRY.register("qwen2_5_omni")
@DATA_TRANSFORM_REGISTRY.register("qwen3_omni_moe")
def process_sample_qwen_omni(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    position_id_func: "Callable",
    **kwargs,
):
    from .multimodal import conv_preprocess
    from .multimodal.audio_utils import fetch_audios
    from .multimodal.image_utils import fetch_images
    from .multimodal.video_utils import fetch_videos_metadata

    QWEN_OMNI_SYSTEM_MESSAGE = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def get_omni_token_ids(processor: "ProcessorMixin") -> tuple[int, int, int]:
        tokenizer = getattr(processor, "tokenizer", processor)
        image_token_id = _get_exact_token_id(tokenizer, "<|image_pad|>", "<|IMAGE|>")
        video_token_id = _get_exact_token_id(tokenizer, "<|video_pad|>", "<|VIDEO|>")
        audio_token_id = _get_exact_token_id(tokenizer, "<|audio_pad|>", "<|AUDIO|>")
        return image_token_id, video_token_id, audio_token_id

    image_token_id, video_token_id, audio_token_id = get_omni_token_ids(processor)

    source = kwargs.get("source_name") or sample.get("source") or sample.get("source_name")
    conversations = (
        sample["conversations"] if ("conversations" in sample and len(sample["conversations"]) > 0) else sample
    )
    conversations = conv_preprocess(source, conversations, **kwargs)
    input_conversations = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": QWEN_OMNI_SYSTEM_MESSAGE,
                },
            ],
        },
    ]
    for conversation in conversations:
        contents = []
        for message in conversation[1:]:
            contents.append({"type": message[0], message[0]: message[1]})
        tmp_conv = {
            "role": conversation[0],
            "content": contents,
        }
        input_conversations.append(tmp_conv)
    text = processor.apply_chat_template(input_conversations, tokenize=False)

    images = sample.get("images", [])
    if images:
        images = fetch_images(images, **kwargs)
    else:
        images = []

    videos = sample.get("videos", [])
    if videos:
        videos, video_metadata, video_audios, _ = fetch_videos_metadata(videos, **kwargs)
    else:
        videos, video_metadata, video_audios = [], [], []

    audios = sample.get("audios", [])
    if audios:
        audio_audios = fetch_audios(audios, **kwargs)
    else:
        audio_audios = []

    video_audios_iter = iter(video_audios)
    audio_audios_iter = iter(audio_audios)
    audios = []
    for item in input_conversations:
        for content in item["content"]:
            if content["type"] == "video":
                audios.append(next(video_audios_iter))
            elif content["type"] == "audio":
                audios.append(next(audio_audios_iter))

    model_inputs = processor(
        text=text,
        audios=audios,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        do_sample_frames=False,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = model_inputs.data
    input_features = model_inputs.pop("input_features", None)
    feature_attention_mask = model_inputs.pop("feature_attention_mask", None)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        valid_mask = audio_feature_lengths != 0
        input_features = input_features[valid_mask].permute(0, 2, 1)[feature_attention_mask[valid_mask].bool()]

        model_inputs["input_features"] = input_features
        model_inputs["audio_feature_lengths"] = audio_feature_lengths
    else:
        audio_feature_lengths = None

    input_ids = model_inputs["input_ids"].squeeze(0)
    image_mask = input_ids == image_token_id
    video_mask = input_ids == video_token_id
    audio_mask = input_ids == audio_token_id
    input_ids[image_mask] = IMAGE_INPUT_INDEX
    input_ids[video_mask] = VIDEO_INPUT_INDEX
    input_ids[audio_mask] = AUDIO_INPUT_INDEX

    position_id_returns = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=model_inputs.get("image_grid_thw", None),
        video_grid_thw=model_inputs.get("video_grid_thw", None),
        attention_mask=model_inputs["attention_mask"],
        audio_seqlens=audio_feature_lengths,
        second_per_grids=model_inputs.pop("video_second_per_grid", None),
    )
    position_id_returns["position_ids"] = position_id_returns["position_ids"].clone()
    # Only position_ids is propagated — rope_deltas is generation-only; see
    # _process_sample_qwen_vl_base for the rationale. grid_thw tensors flow
    # through model_inputs and are packed by the collator.
    model_inputs["position_ids"] = position_id_returns["position_ids"]

    model_inputs["image_mask"] = image_mask
    model_inputs["video_mask"] = video_mask
    model_inputs["audio_mask"] = audio_mask
    input_ids[image_mask | video_mask | audio_mask] = 0
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)

    labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
    tokenizer = getattr(processor, "tokenizer", processor)
    user_token_id = _get_exact_token_id(tokenizer, "user")
    assistant_token_id = _get_exact_token_id(tokenizer, "assistant")
    user_start_index = torch.where(input_ids == user_token_id)[0].tolist()
    assistant_start_index = torch.where(input_ids == assistant_token_id)[0].tolist()
    user_start_index.append(len(input_ids) + 1)
    user_i = 0
    for assis_i in assistant_start_index:
        while user_start_index[user_i] < assis_i:
            user_i += 1
        labels[assis_i + 2 : user_start_index[user_i] - 1] = input_ids[assis_i + 2 : user_start_index[user_i] - 1]
    model_inputs["labels"] = labels
    return [model_inputs]
