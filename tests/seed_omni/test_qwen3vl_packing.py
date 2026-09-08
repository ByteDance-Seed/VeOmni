"""CPU unit tests for Qwen3-VL packed tokens / M-RoPE / visual masks."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from veomni.models.seed_omni.graphs.training_graph import TrainingGraph
from veomni.models.seed_omni.modules.qwen3vl.packing import (
    IMAGE_GRID_THW,
    PACKED_CU_SEQLENS,
    PACKED_INPUT_IDS,
    PACKED_LABELS,
    PACKED_POSITION_IDS,
    PIXEL_VALUES,
    VISUAL_NUM_REAL,
    VISUAL_POS_MASK,
    fold_dummy_anchor,
    masked_scatter_embeds,
    pack_qwen3vl_conversations,
    shift_packed_labels,
)
from veomni.models.seed_omni.modules.qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from veomni.models.seed_omni.modules.qwen3vl.text_encoder.processing import Qwen3VLTextEncoderPreprocessor
from veomni.models.seed_omni.modules.qwen3vl.vision.processing import _OMNI_GRID
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.constants import IGNORE_INDEX


def _cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "Qwen" / "qwen3vl_2b"


class _FakeTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 2
    pad_token_id = 0
    special_tokens = {"<|im_start|>": 10, "<|im_end|>": 2, "<|vision_start|>": 11, "<|vision_end|>": 12}

    def convert_tokens_to_ids(self, token):
        return self.special_tokens.get(token, -1)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        index = 0
        while index < len(text):
            matched = False
            for token, token_id in sorted(self.special_tokens.items(), key=lambda item: len(item[0]), reverse=True):
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    matched = True
                    break
            if not matched:
                ids.append(ord(text[index]) % 50 + 20)
                index += 1
        return {"input_ids": ids}


def test_packed_train_graph_is_a_dag_of_pack_nodes():
    spec = yaml.safe_load((_cfg_dir() / "graph_train_packed.yaml").read_text())
    graph = TrainingGraph(spec)
    names = [node.name for node in graph.iter_nodes()]
    assert names[0] == "qwen3vl_text_encoder.pack_encode"
    assert "qwen3vl_vision.pack_encode" in names
    assert "qwen3vl_llm.pack_forward" in names
    assert "qwen3vl_text_encoder.pack_decode" in names


def test_pack_qwen3vl_conversations_expands_variable_vision_and_skips_dummies():
    # merge=2 → 1*4*4 / 4 = 4 merged tokens
    patches = torch.ones(16, 8)
    dummy = torch.zeros(16, 8)
    grid = [1, 4, 4]
    sample = [
        ConversationItem(
            type="text",
            value=torch.tensor([1, 2, 3]),
            role="user",
            meta={"labels": torch.tensor([-100, -100, -100])},
        ),
        ConversationItem(
            type="image",
            value=patches,
            role="user",
            meta={_OMNI_GRID: grid},
        ),
        ConversationItem(
            type="text",
            value=torch.tensor([4, 5]),
            role="assistant",
            meta={"labels": torch.tensor([4, 5])},
        ),
        ConversationItem(
            type="image",
            value=dummy,
            role="dummy",
            source="qwen3vl_vision",
            meta={_OMNI_GRID: grid},
        ),
    ]
    packed = pack_qwen3vl_conversations([sample], pad_token_id=0, spatial_merge_size=2)
    # 3 text + 4 vision + 2 text = 9 (dummy is not in the sequence)
    assert packed[PACKED_INPUT_IDS].shape[-1] == 9
    assert int(packed[VISUAL_POS_MASK].sum()) == 4
    assert packed[VISUAL_NUM_REAL] == 1
    assert packed[PIXEL_VALUES].shape[0] == 32  # 16 real + 16 dummy patches
    assert packed[IMAGE_GRID_THW].tolist() == [[1, 4, 4], [1, 4, 4]]
    assert packed[PACKED_LABELS][0, 3:7].tolist() == [IGNORE_INDEX] * 4
    assert packed[PACKED_CU_SEQLENS].tolist() == [0, 9]
    assert packed[PACKED_POSITION_IDS].shape == (3, 1, 9)


def test_pack_qwen3vl_two_samples_cu_seqlens():
    patches = torch.ones(16, 8)
    grid = [1, 4, 4]
    samples = []
    for _ in range(2):
        samples.append(
            [
                ConversationItem(
                    type="text",
                    value=torch.tensor([1, 2]),
                    role="user",
                    meta={"labels": torch.tensor([-100, -100])},
                ),
                ConversationItem(type="image", value=patches, role="user", meta={_OMNI_GRID: grid}),
            ]
        )
    packed = pack_qwen3vl_conversations(samples, pad_token_id=0, spatial_merge_size=2)
    assert packed[PACKED_CU_SEQLENS].tolist() == [0, 6, 12]
    assert packed[VISUAL_NUM_REAL] == 2
    assert int(packed[VISUAL_POS_MASK].sum()) == 8


def test_masked_scatter_and_fold_dummy():
    packed = torch.zeros(1, 6, 2)
    mask = torch.tensor([[False, False, True, True, False, False]])
    embeds = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    out = masked_scatter_embeds(packed, mask, embeds)
    assert out[0, 2].tolist() == [1.0, 1.0]
    assert out[0, 3].tolist() == [2.0, 2.0]
    target = torch.ones(2, 3, requires_grad=True)
    dummy = torch.full((1, 3), 4.0, requires_grad=True)
    folded = fold_dummy_anchor(target, dummy)
    assert torch.equal(folded, target)
    folded.sum().backward()
    assert dummy.grad is not None
    assert torch.count_nonzero(dummy.grad) == 0


def test_shift_packed_labels_pads_ignore():
    labels = torch.tensor([[1, 2, 3, IGNORE_INDEX]])
    shifted = shift_packed_labels(labels)
    assert shifted.tolist() == [[2, 3, IGNORE_INDEX, IGNORE_INDEX]]


def test_qwen3vl_packed_preprocessor_writes_batch_keys():
    preprocessor = Qwen3VLTextEncoderPreprocessor(
        Qwen3VLChatTemplate(_FakeTokenizer()), packed_preprocess=True, spatial_merge_size=2
    )
    pixels = torch.ones(16, 8)
    conversation = [
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(type="image", value=pixels, role="user", meta={_OMNI_GRID: [1, 4, 4]}),
            ConversationItem(type="text", value="ok", role="assistant"),
            ConversationItem(
                type="image",
                value=torch.zeros(16, 8),
                role="dummy",
                source="qwen3vl_vision",
                meta={_OMNI_GRID: [1, 4, 4]},
            ),
        ]
    ]
    batch = {"conversation_list": conversation}
    preprocessor(batch, inference=False)
    assert PACKED_INPUT_IDS in batch
    assert VISUAL_POS_MASK in batch
    assert batch[VISUAL_NUM_REAL] == 1
    assert "conversation_list" not in batch


def test_qwen3vl_packed_preprocessor_keeps_conversation_list_on_inference():
    preprocessor = Qwen3VLTextEncoderPreprocessor(Qwen3VLChatTemplate(_FakeTokenizer()), packed_preprocess=True)
    conversation = [[ConversationItem(type="text", value="hi", role="user")]]
    batch = {"conversation_list": conversation}
    preprocessor(batch, inference=True)
    assert batch["conversation_list"] is conversation
    assert PACKED_INPUT_IDS not in batch
