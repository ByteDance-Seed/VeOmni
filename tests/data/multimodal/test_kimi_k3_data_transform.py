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

"""Kimi K3 data transform: pre-expanded media placeholders.

``process_sample_kimi_k3`` materializes one ``<|media_pad|>`` per merged
vision token -- ``(h // mh) * (w // mw)`` per image, because the tower's
``tpool_patch_merger`` pools (averages) the temporal axis instead of
concatenating it -- inline in the text, so the in-model merge is
length-preserving; labels supervise only the caption tail, and a
``max_seq_len`` that would cut the media span must fail loudly (the merge
scatters exactly that many features onto the pad slots).
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from veomni.data.data_transform import process_sample_kimi_k3
from veomni.models.transformers.kimi_k3.modeling_kimi_k3 import tpool_patch_merger


MEDIA_END_ID = 9
MEDIA_PAD_ID = 5
EOS_ID = 3


class _FakeTokenizer:
    """Deterministic ids: [head] + [pad] * n + [media_end] + caption + eos."""

    eos_token_id = EOS_ID

    def encode(self, text):
        if text == "<|media_end|>":
            return [MEDIA_END_ID]
        n_pads = text.count("<|media_pad|>")
        head, _, rest = text.partition("<|media_content|>")
        caption = rest.split("<|media_end|>")[-1]
        head_tokens = [1] + [2] * len(head.split())
        caption_tokens = [10 + i for i, _ in enumerate(caption.split())]
        return head_tokens + [MEDIA_PAD_ID] * n_pads + [MEDIA_END_ID] + caption_tokens


class _FakeImageProcessor:
    merge_size = (2, 2)

    def __init__(self, grid_thws=((1, 2, 2),)):
        # One [t, h, w] row per image; the single-frame (1, 2, 2) default
        # merges to exactly 1 vision token under a 2x2 kernel.
        self.grid_thws = torch.tensor([list(g) for g in grid_thws])

    def preprocess(self, images, return_tensors="pt"):
        return {
            "grid_thws": self.grid_thws,
            "pixel_values": torch.zeros(len(self.grid_thws), 3, 8, 8),
        }


def _make_sample(tmp_path, caption="a cat sits here"):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(120, 140, 160)).save(image_path)
    return {"caption": caption, "image": str(image_path), "width": 8, "height": 8}


def _processor(grid_thws=((1, 2, 2),)):
    return SimpleNamespace(image_processor=_FakeImageProcessor(grid_thws), tokenizer=_FakeTokenizer())


def test_media_span_is_pre_expanded(tmp_path):
    sample = _make_sample(tmp_path)
    (model_inputs,) = process_sample_kimi_k3(sample, _processor(), SimpleNamespace(tokenizer=_FakeTokenizer()), None)

    input_ids = model_inputs["input_ids"]
    labels = model_inputs["labels"]
    attention_mask = model_inputs["attention_mask"]

    # 1-D per-sample tensors (MainCollator packs dim=-1 then unsqueezes).
    assert input_ids.dim() == 1 and labels.dim() == 1 and attention_mask.dim() == 1
    assert input_ids.numel() == labels.numel() == attention_mask.numel()
    # Exactly one <|media_pad|> per merged vision token: the [1,2,2] grid
    # under a 2x2 merge gives (2//2)*(2//2) = 1.
    assert int((input_ids == MEDIA_PAD_ID).sum()) == 1
    # Labels supervise only the caption tail: everything up to and including
    # <|media_end|> is -100 (covers the expanded media span).
    media_end_pos = int((input_ids == MEDIA_END_ID).nonzero()[0])
    assert labels[: media_end_pos + 1].eq(-100).all()
    assert not labels[media_end_pos + 1 :].eq(-100).any()
    # The caption tail (including the eos position) is supervised 1:1.
    assert labels[media_end_pos + 1 :].tolist() == input_ids[media_end_pos + 1 :].tolist()
    assert input_ids[-1].item() == EOS_ID
    # Vision payload passes through untouched.
    assert torch.equal(model_inputs["grid_thws"], torch.tensor([[1, 2, 2]]))
    assert model_inputs["pixel_values"].shape == (1, 3, 8, 8)


def test_empty_caption_returns_no_samples(tmp_path):
    sample = _make_sample(tmp_path, caption="   ")
    assert process_sample_kimi_k3(sample, _processor(), SimpleNamespace(tokenizer=_FakeTokenizer()), None) == []


def test_truncation_cuts_caption_tail_only(tmp_path):
    sample = _make_sample(tmp_path)
    model_inputs = process_sample_kimi_k3(
        sample, _processor(), SimpleNamespace(tokenizer=_FakeTokenizer()), None, max_seq_len=7
    )
    (model_inputs,) = model_inputs
    assert model_inputs["input_ids"].numel() == 7
    assert model_inputs["labels"].numel() == 7
    # The media span (through <|media_end|>) survived the truncation.
    assert (model_inputs["input_ids"] == MEDIA_END_ID).any()


def test_truncation_into_media_span_raises(tmp_path):
    sample = _make_sample(tmp_path)
    # The media span ends at position 4 (head tokens 0..2, pad 3, media_end 4);
    # a max_seq_len that cannot hold it must fail loudly instead of
    # desynchronizing input_ids from the published grid_thws.
    with pytest.raises(ValueError, match="truncated the media span"):
        process_sample_kimi_k3(sample, _processor(), SimpleNamespace(tokenizer=_FakeTokenizer()), None, max_seq_len=5)


def test_missing_tokenizer_raises(tmp_path):
    sample = _make_sample(tmp_path)
    processor = SimpleNamespace(image_processor=_FakeImageProcessor(), tokenizer=None)
    with pytest.raises(ValueError, match="needs a tokenizer"):
        process_sample_kimi_k3(sample, processor, SimpleNamespace(tokenizer=None), None)


def test_multi_frame_grid_pads_match_tpool_patch_merger(tmp_path):
    # tpool_patch_merger AVERAGES over the temporal axis (pooled, not
    # concatenated), so a [2, 4, 4] grid yields (4//2)*(4//2) = 4 merged
    # tokens -- a prod-based count would emit 8 and the in-model merge would
    # raise. The real merger is the single source of truth for the pad count.
    sample = _make_sample(tmp_path)
    (model_inputs,) = process_sample_kimi_k3(
        sample, _processor(grid_thws=((2, 4, 4),)), SimpleNamespace(tokenizer=_FakeTokenizer()), None
    )

    t, h, w = 2, 4, 4
    merged = tpool_patch_merger(torch.zeros(t * h * w, 8), torch.tensor([[t, h, w]]), merge_kernel_size=(2, 2))
    n_pads = int((model_inputs["input_ids"] == MEDIA_PAD_ID).sum())
    assert n_pads == merged[0].shape[0] == 4
    # The published grid passes through untouched.
    assert torch.equal(model_inputs["grid_thws"], torch.tensor([[t, h, w]]))
