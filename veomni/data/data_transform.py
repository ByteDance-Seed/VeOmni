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

import sys
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Union

import torch

from veomni.utils.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.dpo_provenance import (
    DPO_FEATURE_KEY,
    DPOFeatureSegments,
    dpo_declared_producer_identity,
    dpo_tokenizer_fingerprint,
    packed_batch_digest,
)
from veomni.utils.registry import Registry


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from .chat_template import ChatTemplate


DATA_TRANSFORM_REGISTRY = Registry("DataTransform")


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


def _dpo_plaintext_completion_start(tokenizer: Any, prompt: str, full_text: str, token_ids: list[int]) -> int:
    """Locate scoreable completion tokens without supervising prompt-only BPE merges."""
    if not prompt:
        return 0
    if full_text == prompt:
        raise ValueError("DPO prompt leaves no scoreable completion tokens; shorten it or increase max_seq_len")
    encoded = None
    if callable(tokenizer):
        try:
            encoded = tokenizer(
                full_text, add_special_tokens=True, return_offsets_mapping=True, return_special_tokens_mask=True
            )
        except TypeError:
            try:
                encoded = tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
            except (AttributeError, NotImplementedError, TypeError):
                pass
        except (AttributeError, NotImplementedError):
            pass
    if encoded is not None and "offset_mapping" in encoded and "input_ids" in encoded:
        encoded_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        special_mask = encoded.get("special_tokens_mask")
        if isinstance(encoded_ids, torch.Tensor):
            encoded_ids = encoded_ids.tolist()
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.tolist()
        if isinstance(special_mask, torch.Tensor):
            special_mask = special_mask.tolist()
        if list(encoded_ids[: len(token_ids)]) != list(token_ids) or len(offsets) < len(token_ids):
            raise ValueError("DPO tokenizer offsets do not match its encoded token IDs")
        if special_mask is not None and (
            len(special_mask) < len(token_ids) or any(flag not in (0, 1) for flag in special_mask[: len(token_ids)])
        ):
            raise ValueError("DPO tokenizer returned a malformed special-token mask")
        first, scoreable, previous_end = None, False, 0
        for index, pair in enumerate(offsets[: len(token_ids)]):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("DPO tokenizer returned malformed character offsets")
            start, end = pair
            if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(full_text):
                raise ValueError("DPO tokenizer returned invalid character offsets")
            if special_mask is not None and special_mask[index]:
                continue
            if start == end:
                if special_mask is None:
                    raise ValueError("DPO zero-width token requires a special-token mask")
                if start == len(prompt):
                    raise ValueError("DPO zero-width token lies at the prompt boundary; use untrimmed offsets")
            if end < previous_end:
                raise ValueError("DPO tokenizer returned unordered character offsets")
            previous_end = end
            if end > len(prompt):
                first = index if first is None else first
                scoreable = scoreable or index > 0
            elif first is not None:
                raise ValueError("DPO tokenizer returned prompt tokens after the completion boundary")
        if first is None or not scoreable:
            raise ValueError("DPO prompt leaves no scoreable completion tokens; shorten it or increase max_seq_len")
        return first

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prefix = list(prompt_ids)
    # Remove structurally appended suffix tokens before comparing prefixes. A
    # completion may itself begin with EOS, so even an apparent exact prefix
    # containing the automatically appended EOS would mask a real target.
    get_special_mask = getattr(tokenizer, "get_special_tokens_mask", None)
    special_mask = []
    if callable(get_special_mask):
        try:
            raw_prompt = tokenizer.encode(prompt, add_special_tokens=False)
            special_mask = list(get_special_mask(raw_prompt, already_has_special_tokens=False))
        except (TypeError, NotImplementedError):
            special_mask = []
    structural_mask_valid = (
        bool(special_mask)
        and len(special_mask) == len(prefix)
        and all(flag in (0, 1) for flag in special_mask)
        and [token for token, flag in zip(prefix, special_mask) if flag == 0] == list(raw_prompt)
    )
    if structural_mask_valid:
        if 0 not in special_mask:
            raise ValueError("DPO prompt boundary requires tokenizer offset mapping; use a fast tokenizer")
        while special_mask and special_mask[-1] == 1:
            special_mask.pop()
            prefix.pop()
    elif prefix and prefix[-1] in set(getattr(tokenizer, "all_special_ids", ()) or ()):
        # Without structural evidence, a trailing special token could be
        # literal prompt text or an automatically appended token.
        raise ValueError("DPO prompt boundary requires tokenizer offset mapping; use a fast tokenizer")
    if len(token_ids) <= len(prefix) and list(token_ids) == prefix[: len(token_ids)]:
        raise ValueError("DPO prompt leaves no scoreable completion tokens; shorten it or increase max_seq_len")
    if not prefix or prefix != list(token_ids[: len(prefix)]):
        raise ValueError("DPO prompt boundary requires tokenizer offset mapping; use a fast tokenizer")
    if structural_mask_valid:
        try:
            raw_full = tokenizer.encode(full_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=True)
            full_mask = list(get_special_mask(raw_full, already_has_special_tokens=False))
        except (TypeError, NotImplementedError) as exc:
            raise ValueError("DPO prompt boundary requires tokenizer offset mapping; use a fast tokenizer") from exc
        if (
            len(full_mask) != len(full_ids)
            or any(flag not in (0, 1) for flag in full_mask)
            or [token for token, flag in zip(full_ids, full_mask) if flag == 0] != list(raw_full)
            or list(full_ids[: len(token_ids)]) != list(token_ids)
        ):
            raise ValueError("DPO prompt boundary requires tokenizer offset mapping; use a fast tokenizer")
        if not any(full_mask[index] == 0 for index in range(max(len(prefix), 1), len(token_ids))):
            raise ValueError("DPO prompt leaves no scoreable completion tokens; shorten it or increase max_seq_len")
    if len(token_ids) <= max(len(prefix), 1):
        raise ValueError("DPO prompt leaves no scoreable completion tokens; shorten it or increase max_seq_len")
    return len(prefix)


def dpo_tokenizer_contract_identity(tokenizer: Any, audited_sha256: str | None) -> str:
    from transformers import PreTrainedTokenizerFast
    from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    if type(tokenizer).__dict__.get("dpo_unpadded_output_contract") is True:
        return dpo_declared_producer_identity(tokenizer, "tokenizer")
    if type(tokenizer) in (PreTrainedTokenizerFast, Qwen2Tokenizer, LlamaTokenizer, GPT2Tokenizer):
        actual = dpo_tokenizer_fingerprint(tokenizer)
        if type(audited_sha256) is str and actual == audited_sha256:
            return actual
    raise ValueError(
        "DPO tokenizer must declare dpo_unpadded_output_contract=True and a stable contract ID "
        "or match an audited_unpadded_tokenizer_sha256"
    )


def dpo_producer_identity(tokenizer: Any, chat_template: Any, audited_sha256: str | None) -> dict[str, str | None]:
    """Validate and identify the actual producers before loading buffered data."""
    tokenizer_identity = dpo_tokenizer_contract_identity(tokenizer, audited_sha256)
    template_identity = None
    template_tokenizer_identity = None
    if chat_template is not None:
        template_identity = dpo_declared_producer_identity(chat_template, "chat template")
        template_tokenizer = getattr(chat_template, "tokenizer", None)
        if template_tokenizer is not None:
            template_tokenizer_identity = (
                tokenizer_identity
                if template_tokenizer is tokenizer
                else dpo_tokenizer_contract_identity(template_tokenizer, audited_sha256)
            )
    return {
        "tokenizer": tokenizer_identity,
        "template": template_identity,
        "template_tokenizer": template_tokenizer_identity,
    }


def _dpo_completion_only_messages(chosen: list, rejected: list) -> tuple[list, list, list]:
    if not chosen or not rejected or any(not isinstance(message, dict) for message in chosen + rejected):
        raise ValueError("DPO conversations require nonempty chosen and rejected message lists")

    def content(message):
        return {key: value for key, value in message.items() if key != "loss_mask"}

    common = 0
    while common < min(len(chosen), len(rejected)) and content(chosen[common]) == content(rejected[common]):
        common += 1
    if common >= min(len(chosen), len(rejected)):
        raise ValueError("DPO chosen and rejected conversations require distinct assistant completions")
    if chosen[common].get("role") != "assistant" or rejected[common].get("role") != "assistant":
        raise ValueError("DPO chosen and rejected conversations do not share the same prompt history")

    def prepare(messages):
        result = []
        for index, message in enumerate(messages):
            copied = dict(message)
            if index < common or copied.get("role") != "assistant":
                copied["loss_mask"] = 0
            else:
                copied.setdefault("loss_mask", 1)
            result.append(copied)
        return result

    chosen_masked, rejected_masked = prepare(chosen), prepare(rejected)
    return chosen_masked, rejected_masked, chosen_masked[:common]


def _dpo_conversation_encodings(chat_template: Any, chosen: list, rejected: list, max_seq_len: int):
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError("DPO max_seq_len must be a positive integer")
    chosen_masked, rejected_masked, prefix = _dpo_completion_only_messages(chosen, rejected)

    def encode(messages, limit):
        result = chat_template.encode_messages(messages, max_seq_len=limit)
        return {
            key: result[key] if isinstance(result[key], torch.Tensor) else torch.tensor(result[key])
            for key in ("input_ids", "attention_mask", "labels")
        }

    prefix_ids = encode(prefix, sys.maxsize)["input_ids"] if prefix else torch.empty(0, dtype=torch.int64)
    prefix_len = prefix_ids.numel()
    contexts = []
    outputs = []
    for original, masked in ((chosen, chosen_masked), (rejected, rejected_masked)):
        original_with_masks = []
        for message in original:
            copied = dict(message)
            copied.setdefault("loss_mask", 1 if copied.get("role") == "assistant" else 0)
            original_with_masks.append(copied)
        raw_ids = encode(original_with_masks, sys.maxsize)["input_ids"]
        full = encode(masked, sys.maxsize)
        full_ids = full["input_ids"]
        truncated = encode(masked, max_seq_len)
        ids, labels = truncated["input_ids"], truncated["labels"]
        if (
            raw_ids.ndim != 1
            or full_ids.ndim != 1
            or ids.ndim != 1
            or labels.shape != ids.shape
            or full["labels"].shape != full_ids.shape
        ):
            raise ValueError("DPO conversation producer must emit flat token and label sequences")
        if not torch.equal(raw_ids, full_ids):
            raise ValueError("DPO conversation loss masks must not change tokenization")
        if prefix_ids.ndim != 1 or prefix_len > full_ids.numel() or not torch.equal(full_ids[:prefix_len], prefix_ids):
            raise ValueError("DPO conversation template must preserve the shared prompt prefix")
        for index in range(len(prefix), len(masked)):
            if masked[index].get("role") == "assistant":
                continue
            before = encode(masked[:index], sys.maxsize)["input_ids"]
            after = encode(masked[: index + 1], sys.maxsize)["input_ids"]
            if (
                before.ndim != 1
                or after.ndim != 1
                or before.numel() > after.numel()
                or after.numel() > full_ids.numel()
                or not torch.equal(full_ids[: before.numel()], before)
                or not torch.equal(full_ids[: after.numel()], after)
                or not bool((full["labels"][before.numel() : after.numel()] == IGNORE_INDEX).all())
            ):
                raise ValueError("DPO conversation template must mask non-assistant messages")
        offset = full_ids.numel() - ids.numel()
        if (
            offset < 0
            or offset > prefix_len
            or ids.numel() > max_seq_len
            or not torch.equal(full_ids[offset:], ids)
            or not torch.equal(full["labels"][offset:], labels)
            or not torch.equal(full["attention_mask"][offset:], truncated["attention_mask"])
        ):
            raise ValueError("DPO conversation truncation must preserve the completion and a shared prompt suffix")
        visible_prefix = prefix_len - offset
        if prefix_len and not visible_prefix:
            raise ValueError("DPO conversation truncation removed the entire prompt")
        if not bool((labels[:visible_prefix] == IGNORE_INDEX).all()):
            raise ValueError("DPO conversation template must mask the shared prompt history")
        # Even with no explicit shared history, causal prediction consumes the
        # first token as context; both branches must condition on the same one.
        contexts.append(ids[: max(visible_prefix, 1)])
        outputs.append(truncated)
    if not torch.equal(contexts[0], contexts[1]):
        raise ValueError("DPO chosen and rejected contexts differ after truncation")
    return outputs[0], outputs[1]


@DATA_TRANSFORM_REGISTRY.register("dpo")
def process_dpo_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate" = None,
    tokenizer: "PreTrainedTokenizer" = None,
    max_seq_len: int = 2048,
    record_segment_provenance: bool = False,
    audited_unpadded_tokenizer_sha256: str | None = None,
    expected_dpo_producer_identity: dict[str, str | None] | None = None,
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

    if record_segment_provenance:
        # An all-one mask and physical lengths do not establish that an opaque
        # producer has not already inserted padding. The producer must declare
        # the unpadded-output contract; this is an API promise, not an inference
        # from token values or supervision labels.
        if expected_dpo_producer_identity is not None:
            if dpo_producer_identity(tokenizer, chat_template, audited_unpadded_tokenizer_sha256) != (
                expected_dpo_producer_identity
            ):
                raise ValueError("DPO producer identity changed after initialization")
        elif isinstance(chosen_raw, list):
            dpo_declared_producer_identity(chat_template, "chat template")
            template_tokenizer = getattr(chat_template, "tokenizer", None)
            if template_tokenizer is not None:
                dpo_tokenizer_contract_identity(template_tokenizer, audited_unpadded_tokenizer_sha256)
        else:
            dpo_tokenizer_contract_identity(tokenizer, audited_unpadded_tokenizer_sha256)

    if isinstance(chosen_raw, list):
        assert chat_template is not None, "chat_template is required for conversation-format DPO data"
        if record_segment_provenance:
            if not isinstance(rejected_raw, list):
                raise ValueError("DPO chosen and rejected must use the same conversation format")
            chosen_tok, rejected_tok = _dpo_conversation_encodings(
                chat_template, chosen_raw, rejected_raw, max_seq_len
            )
        else:
            chosen_tok = chat_template.encode_messages(chosen_raw, max_seq_len=max_seq_len)
            rejected_tok = chat_template.encode_messages(rejected_raw, max_seq_len=max_seq_len)
    else:
        assert tokenizer is not None, "tokenizer is required for plaintext-format DPO data"
        prompt = example.get("prompt", "")
        chosen_text = prompt + chosen_raw
        rejected_text = prompt + rejected_raw

        chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=True)[:max_seq_len]
        rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=True)[:max_seq_len]
        chosen_prompt_len = _dpo_plaintext_completion_start(tokenizer, prompt, chosen_text, chosen_ids)
        rejected_prompt_len = _dpo_plaintext_completion_start(tokenizer, prompt, rejected_text, rejected_ids)
        if record_segment_provenance:
            if prompt and (chosen_prompt_len == 0 or rejected_prompt_len == 0):
                raise ValueError("DPO plaintext completion needs a shared token before the prompt boundary")
            chosen_context = chosen_ids[: max(chosen_prompt_len, 1)]
            rejected_context = rejected_ids[: max(rejected_prompt_len, 1)]
            if (
                not chosen_context
                or not rejected_context
                or not torch.equal(torch.as_tensor(chosen_context), torch.as_tensor(rejected_context))
            ):
                raise ValueError("DPO chosen and rejected plaintext contexts differ after tokenization or truncation")

        chosen_tok = {
            "input_ids": chosen_ids,
            "attention_mask": [1] * len(chosen_ids),
            "labels": [IGNORE_INDEX] * chosen_prompt_len + chosen_ids[chosen_prompt_len:],
        }
        rejected_tok = {
            "input_ids": rejected_ids,
            "attention_mask": [1] * len(rejected_ids),
            "labels": [IGNORE_INDEX] * rejected_prompt_len + rejected_ids[rejected_prompt_len:],
        }

    def _to_tensor(v):
        return v if isinstance(v, torch.Tensor) else torch.tensor(v)

    c_ids = _to_tensor(chosen_tok["input_ids"])
    r_ids = _to_tensor(rejected_tok["input_ids"])
    c_labels = _to_tensor(chosen_tok["labels"])
    r_labels = _to_tensor(rejected_tok["labels"])
    c_len = c_ids.shape[-1]
    r_len = r_ids.shape[-1]
    if record_segment_provenance:
        for ids, labels in ((c_ids, c_labels), (r_ids, r_labels)):
            if ids.shape != labels.shape or not bool(((labels == IGNORE_INDEX) | (labels == ids)).all()):
                raise ValueError("DPO labels must be unshifted token IDs or IGNORE_INDEX")
            if not bool((labels != IGNORE_INDEX).any()):
                raise ValueError("DPO chosen and rejected sequences must each contain supervised tokens")
            if labels.numel() > 1 and not bool((labels[..., 1:] != IGNORE_INDEX).any()):
                raise ValueError("DPO multi-token sequences must contain a causal prediction target")

    result = {
        "input_ids": torch.cat([c_ids, r_ids]),
        "attention_mask": torch.cat(
            [_to_tensor(chosen_tok["attention_mask"]), _to_tensor(rejected_tok["attention_mask"])]
        ),
        "labels": torch.cat([c_labels, r_labels]),
        "position_ids": torch.cat([torch.arange(c_len, dtype=torch.int64), torch.arange(r_len, dtype=torch.int64)]),
    }
    if record_segment_provenance:
        feature_digest = packed_batch_digest(
            result,
            feature=True,
            include_keys=("input_ids", "labels", "position_ids", "attention_mask"),
        )
        result[DPO_FEATURE_KEY] = DPOFeatureSegments(
            c_len, r_len, producer_contract_version=1, feature_digest=feature_digest
        )
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
