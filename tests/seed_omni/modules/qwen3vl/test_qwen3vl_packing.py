"""CPU unit tests for Qwen3-VL packed tokens / M-RoPE / visual masks."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from veomni.models.seed_omni.graphs.training_graph import TrainingGraph
from veomni.models.seed_omni.modules.base.packing import PACKED_ATTENTION_MASK
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
    return Path(__file__).resolve().parents[4] / "configs" / "seed_omni" / "Qwen" / "qwen3vl"


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
    spec = yaml.safe_load((_cfg_dir() / "packed/graph_train.yaml").read_text())
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
            role="user",
            is_dummy=True,
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


def test_pack_qwen3vl_masks_sample_boundary_labels():
    """Each packed sample's first label is IGNORE, whatever the template emitted.

    The shift is global over the packed sequence, so an unmasked first label would
    make sample k-1's last position predict a token ``packed_cu_seqlens`` already
    cut attention at. These samples train every token (no masked leading marker),
    which is exactly the case the chat template would otherwise be covering up.
    """
    samples = [
        [
            ConversationItem(
                type="text",
                value=torch.tensor([1, 2, 3]),
                role="assistant",
                meta={"labels": torch.tensor([1, 2, 3])},
            ),
            # FSDP anchor only — dummies stay off the packed sequence.
            ConversationItem(
                type="image",
                value=torch.zeros(16, 8),
                role="user",
                is_dummy=True,
                meta={_OMNI_GRID: [1, 4, 4]},
            ),
        ],
        [
            ConversationItem(
                type="text",
                value=torch.tensor([4, 5]),
                role="assistant",
                meta={"labels": torch.tensor([4, 5])},
            )
        ],
    ]
    packed = pack_qwen3vl_conversations(samples, pad_token_id=0, spatial_merge_size=2)
    cu = packed[PACKED_CU_SEQLENS].tolist()
    assert cu == [0, 3, 5]
    # Sample 0 keeps its first label (the shift drops it anyway); sample 1's is masked.
    assert packed[PACKED_LABELS][0].tolist() == [1, 2, 3, IGNORE_INDEX, 5]
    # The carrier's own labels must not have been mutated in place.
    assert samples[1][0].meta["labels"].tolist() == [4, 5]

    shifted = shift_packed_labels(packed[PACKED_LABELS])[0]
    for boundary in cu[1:-1]:
        assert shifted[boundary - 1] == IGNORE_INDEX


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
                role="user",
                is_dummy=True,
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


def test_packed_samples_survive_omni_preprocess_and_collation():
    """The packed training path, end to end: ``OmniProcessor.preprocess`` keeps the
    keys the packed preprocessor wrote, and ``SeedOmniCollator`` batches them into
    one varlen packed sequence instead of list-grouping them."""
    from veomni.data.seed_omni.collator import SeedOmniCollator
    from veomni.models.seed_omni.processing_omni import OmniProcessor

    preprocessor = Qwen3VLTextEncoderPreprocessor(
        Qwen3VLChatTemplate(_FakeTokenizer()), packed_preprocess=True, spatial_merge_size=2
    )
    processor = OmniProcessor({"qwen3vl_text_encoder": preprocessor})

    def _sample() -> list[ConversationItem]:
        return [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(type="image", value=torch.ones(16, 8), role="user", meta={_OMNI_GRID: [1, 4, 4]}),
            ConversationItem(type="text", value="ok", role="assistant"),
        ]

    first = processor.preprocess(_sample())
    second = processor.preprocess(_sample())
    # The carrier is consumed, and everything the packed preprocessor wrote is kept.
    assert "conversation_list" not in first
    assert PACKED_INPUT_IDS in first and VISUAL_POS_MASK in first

    batch = SeedOmniCollator()([first, second])

    per_sample = int(first[PACKED_INPUT_IDS].shape[-1])
    assert tuple(batch[PACKED_INPUT_IDS].shape) == (1, 2 * per_sample)
    assert tuple(batch[PACKED_LABELS].shape) == (1, 2 * per_sample)
    assert tuple(batch[PACKED_POSITION_IDS].shape) == (3, 1, 2 * per_sample)
    assert tuple(batch[VISUAL_POS_MASK].shape) == (1, 2 * per_sample)
    # One varlen segment per sample, rebuilt by the collator.
    assert batch[PACKED_CU_SEQLENS].tolist() == [0, per_sample, 2 * per_sample]
    assert batch["packed_max_length"] == per_sample
    # Per-visual rows and the scalar both accumulate.
    assert batch[VISUAL_NUM_REAL] == 2
    assert tuple(batch[IMAGE_GRID_THW].shape) == (2, 3)
    assert tuple(batch[PIXEL_VALUES].shape) == (32, 8)


def test_packed_collation_batches_an_undeclared_key_like_the_base_collator():
    """An unknown key is not an error and not a per-sample list: the base
    ``PackingCollator`` runs it through ``default_collate``, and so does this one.
    Framework metadata (`ds_idx` appended after the transform) relies on that."""
    from veomni.data.seed_omni.collator import SeedOmniCollator

    def _packed(extra: dict) -> dict:
        return {
            PACKED_INPUT_IDS: torch.zeros(1, 4, dtype=torch.long),
            PACKED_LABELS: torch.zeros(1, 4, dtype=torch.long),
            PACKED_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.long),
            **extra,
        }

    batch = SeedOmniCollator()([_packed({"brand_new_key": 1}), _packed({"brand_new_key": 2})])
    # default_collate turns a list of ints into a tensor of ints.
    assert torch.equal(batch["brand_new_key"], torch.tensor([1, 2]))


def test_packed_collation_carries_the_multisource_metadata_the_meter_reads():
    """``WeightedMultiSourceDataset`` appends ``ds_idx`` / ``source_name`` to a sample
    *after* the transform, so a packed sample arrives carrying them. They must
    survive collation in the shape the trainer's multi-source meter pops them:
    a tensor of indices, a list of names."""
    from veomni.data.seed_omni.collator import SeedOmniCollator
    from veomni.utils.helper import _get_multisource_ds_idx

    def _packed(ds_idx: int, source_name: str) -> dict:
        return {
            PACKED_INPUT_IDS: torch.zeros(1, 4, dtype=torch.long),
            PACKED_LABELS: torch.zeros(1, 4, dtype=torch.long),
            PACKED_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.long),
            "ds_idx": ds_idx,
            "source_name": source_name,
        }

    batch = SeedOmniCollator()([_packed(0, "sharegpt4v"), _packed(2, "llava_video")])
    assert torch.equal(batch["ds_idx"], torch.tensor([0, 2]))
    assert batch["source_name"] == ["sharegpt4v", "llava_video"]

    # The meter reads them and pops them off, leaving only the packed tensors.
    assert _get_multisource_ds_idx(batch) == [0, 2]
    assert "ds_idx" not in batch and "source_name" not in batch
    assert PACKED_INPUT_IDS in batch


def test_packed_collation_carries_cur_token_num():
    """Dynamic batching writes ``cur_token_num`` per sample; it must survive collation
    so the meter's own pop finds it instead of leaking it into the graph."""
    from veomni.data.seed_omni.collator import SeedOmniCollator

    def _packed(cur_token_num: int) -> dict:
        return {
            PACKED_INPUT_IDS: torch.zeros(1, 4, dtype=torch.long),
            PACKED_LABELS: torch.zeros(1, 4, dtype=torch.long),
            PACKED_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.long),
            "cur_token_num": cur_token_num,
        }

    batch = SeedOmniCollator()([_packed(4), _packed(6)])
    assert torch.equal(batch["cur_token_num"], torch.tensor([4, 6]))


def test_packed_collation_puts_every_real_visual_before_every_dummy():
    """Each sample packs its visuals as ``[real..., dummy...]`` and reports how many
    are real. A plain concatenation would order the batch per sample while the
    summed ``visual_num_real`` counts all real rows, so the vision hook would read
    a dummy grid as real — scattering zeros into a real image's mask positions, or
    crashing when the grids differ in size."""
    from veomni.data.seed_omni.collator import SeedOmniCollator
    from veomni.models.seed_omni.modules.qwen3vl.packing import visual_token_count

    def _packed(real_grid: list[int], dummy_grid: list[int], n_real: int) -> dict:
        grids = torch.tensor([real_grid, dummy_grid])
        return {
            PACKED_INPUT_IDS: torch.zeros(1, 8, dtype=torch.long),
            PACKED_LABELS: torch.zeros(1, 8, dtype=torch.long),
            PACKED_ATTENTION_MASK: torch.ones(1, 8, dtype=torch.long),
            PIXEL_VALUES: torch.arange(int(grids.prod(dim=1).sum()), dtype=torch.float).unsqueeze(1),
            IMAGE_GRID_THW: grids,
            VISUAL_NUM_REAL: n_real,
        }

    # Asymmetric on purpose: a plain per-sample concat would interleave them as
    # [[1,4,4], [1,8,8], [1,4,4], [1,8,8]] and count sample 0's dummy as real.
    batch = SeedOmniCollator()([_packed([1, 4, 4], [1, 8, 8], 1), _packed([1, 4, 4], [1, 8, 8], 1)])
    assert batch[IMAGE_GRID_THW].tolist() == [[1, 4, 4], [1, 4, 4], [1, 8, 8], [1, 8, 8]]
    assert batch[VISUAL_NUM_REAL] == 2
    # The first `visual_num_real` rows are the real ones, so their merged-token
    # count is what the vision hook consumes.
    assert visual_token_count(batch[IMAGE_GRID_THW], 2, batch[VISUAL_NUM_REAL]) == (16 + 16) // 4
    # Pixels follow their grids: 16 + 16 + 64 + 64 patch rows.
    assert tuple(batch[PIXEL_VALUES].shape) == (160, 1)
