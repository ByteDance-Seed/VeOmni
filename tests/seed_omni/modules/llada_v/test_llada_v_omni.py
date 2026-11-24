"""Unit tests for the LLaDA-V Omni module.

The packing, sample counting and step-mean declaration are inherited from the
dense module and covered by
``tests/seed_omni/modules/llada/test_llada_omni_batching.py``. What is specific to
this family is the image half of the batch, and it has two failure modes worth
pinning:

* the carrier holds pixels as a ``(C, H, W)`` uint8 tensor while the shared
  transform speaks PIL, so a missing conversion raises ``NotImplementedError`` far
  from the cause;
* a batch whose samples carry no image must send **no** ``pixel_values`` at all,
  because the vendored forward then runs the tower once on a dummy tile to keep
  FSDP's all-gathers matched across ranks.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from veomni.models.seed_omni.modules.llada_v.accelerated.accelerated import TrainingMixin
from veomni.models.seed_omni.modules.llada_v.processing import _to_pil
from veomni.models.seed_omni.utils.conversation import ConversationItem


def _text_item(length: int, image_seq_len: int = 0) -> ConversationItem:
    """One processed carrier item, which is what the preprocessor emits.

    The preprocessor replaces a sample's whole turn list with a single item, so the
    image inputs live in the same ``value`` dict as the text tensors -- there is no
    separate image item once preprocessing has run.
    """
    value = {
        "input_ids": torch.arange(length),
        "attention_mask": torch.ones(length, dtype=torch.long),
        "labels": torch.full((length,), -100, dtype=torch.long),
        "loss_weight": torch.zeros(length, dtype=torch.float32),
        "answer_mask": torch.zeros(length, dtype=torch.long),
    }
    if image_seq_len:
        value["pixel_values"] = torch.zeros(1, 3, 4, 4, dtype=torch.uint8)
        mask = torch.zeros(length, dtype=torch.bool)
        mask[:image_seq_len] = True
        value["image_mask"] = mask
    return ConversationItem(type="text", value=value, role="assistant", meta={})


def _module() -> TrainingMixin:
    module = TrainingMixin()
    module.config = SimpleNamespace(pad_token_id=0, router_aux_loss_coef=0.0)
    module.get_input_embeddings = lambda: SimpleNamespace(weight=torch.zeros(1))
    return module


def test_the_carrier_tensor_becomes_the_pil_image_the_transform_reads():
    """`fetch_images` keeps pixels as uint8 to make the worker's IPC cheap, while
    `process_llada_v_example` speaks PIL. Without this conversion the failure is a
    bare `NotImplementedError` from the loader, naming neither the image nor this
    module."""
    pil = _to_pil(torch.zeros(3, 5, 7, dtype=torch.uint8))

    assert pil.mode == "RGB"
    assert pil.size == (7, 5)  # (W, H) after the (C, H, W) -> (H, W, C) move


def test_a_payload_the_transform_already_accepts_passes_through():
    """A carrier built by a different preprocessor may still hold a path."""
    assert _to_pil("some/path.png") == "some/path.png"


def test_the_images_of_a_micro_batch_are_concatenated_not_stacked():
    """`pixel_values` is `[num_images, 3, H, W]` for the whole micro-batch and the
    per-sample counts differ, so stacking would need equal counts."""
    batch = _module().batch_pre(conversation_list=[[_text_item(6, image_seq_len=4)]])

    assert batch["pixel_values"].shape == (1, 3, 4, 4)
    assert batch["image_mask"].shape == batch["input_ids"].shape
    assert bool(batch["image_mask"].any())


def test_a_mixed_batch_keeps_every_sample_s_image_mask():
    """Samples are laid end to end, so one that carries an image and one that does
    not must still produce a mask matching the packed row."""
    batch = _module().batch_pre(conversation_list=[[_text_item(6, image_seq_len=4)], [_text_item(5)]])

    assert batch["input_ids"].shape == (1, 11)
    assert batch["image_mask"].shape == (1, 11)
    assert int(batch["image_mask"].sum()) == 4
    assert batch["pixel_values"].shape[0] == 1


def test_a_batch_without_images_sends_no_pixel_values():
    """An image-free batch must omit the keys entirely: the vendored forward then
    runs the tower once on a dummy tile whenever the world size is above one, which
    is what keeps FSDP's all-gathers matched across ranks whose samples differ in
    whether they carry an image. An empty tensor would defeat that branch."""
    batch = _module().batch_pre(conversation_list=[[_text_item(5)]])

    assert "pixel_values" not in batch
    assert "image_mask" not in batch
    assert batch["input_ids"].shape == (1, 5)
