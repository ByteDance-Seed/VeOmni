"""Unit tests for the DiffusionGemma Omni module.

The loss conventions are inherited from the dense LLaDA module and covered by
``tests/seed_omni/modules/llada/test_llada_omni_batching.py``. What is specific to
this family is the batching: the batch is built by ``DiffusionGemmaCollator``, and
the module re-instantiates it per micro-batch with the live sequence-parallel size.
That indirection is what these tests pin, because a copy of the canvas geometry
would pass a shape check while packing differently.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from veomni.models.seed_omni.modules.diffusion_gemma.accelerated import DiffusionGemmaOmniAccelerated
from veomni.models.seed_omni.modules.diffusion_gemma.accelerated import accelerated as mod
from veomni.models.seed_omni.utils.conversation import ConversationItem


def _feature(canvas_length: int = 4, num_canvases: int = 1, prompt: int = 3) -> dict:
    """One processed sample, as ``process_diffusion_gemma_example`` emits it."""
    return {
        "input_ids": torch.arange(prompt),
        "encoder_prompt_mask": torch.ones(prompt, dtype=torch.bool),
        "decoder_input_ids": torch.zeros(num_canvases, canvas_length, dtype=torch.long),
        "labels": torch.zeros(num_canvases, canvas_length, dtype=torch.long),
        "loss_weight": torch.ones(num_canvases, canvas_length),
        "decoder_context_mask": torch.ones(num_canvases, prompt, dtype=torch.bool),
    }


def _item(feature: dict | None = None) -> ConversationItem:
    return ConversationItem(
        type="text", value=feature if feature is not None else _feature(), role="assistant", meta={}
    )


def _module(pack: bool = False) -> mod.TrainingMixin:
    module = mod.TrainingMixin()
    module.config = SimpleNamespace(
        canvas_batching="flat_seq_pack" if pack else "serial",
        pad_token_id=0,
    )
    return module


def test_the_batch_comes_from_the_real_collator(monkeypatch):
    """The canvas geometry, packing and the decoder masks all live in
    ``DiffusionGemmaCollator``. The module must call it rather than re-derive it: a
    copy would pass a shape check while packing differently."""
    seen = {}

    class FakeCollator:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def __call__(self, features):
            seen["features"] = features
            return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    monkeypatch.setattr(mod, "DiffusionGemmaCollator", FakeCollator)

    batch = _module().batch_pre(conversation_list=[[_item()]])

    assert len(seen["features"]) == 1
    assert "input_ids" in batch


def test_the_collator_is_built_with_the_live_parallel_size(monkeypatch):
    """``sp_size`` decides whether the row is padded to divide across the SP group,
    and the collator refuses ``sp_size > 1`` without packing. Reading it from the
    launcher instead of the live state would let the two disagree."""
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __call__(self, features):
            return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    monkeypatch.setattr(mod, "DiffusionGemmaCollator", Recorder)
    monkeypatch.setattr(mod, "get_parallel_state", lambda: SimpleNamespace(sp_size=4))

    _module(pack=True).batch_pre(conversation_list=[[_item()]])

    assert seen["sp_size"] == 4
    assert seen["pack_samples"] is True


def test_packing_is_read_off_the_canvas_batching_mode(monkeypatch):
    """Only the ``flat_seq*`` modes pack; ``serial`` and ``flat_batch`` do not."""
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __call__(self, features):
            return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    monkeypatch.setattr(mod, "DiffusionGemmaCollator", Recorder)
    monkeypatch.setattr(mod, "get_parallel_state", lambda: SimpleNamespace(sp_size=1))

    _module(pack=False).batch_pre(conversation_list=[[_item()]])

    assert seen["pack_samples"] is False


def test_a_sample_without_processed_tensors_is_refused():
    """The preprocessor must have run; failing here names the cause instead of
    handing an empty feature list to the collator."""
    module = _module()

    with pytest.raises(ValueError, match="no processed item"):
        module.batch_pre(conversation_list=[[ConversationItem(type="text", value="raw", role="user", meta={})]])


def test_the_module_declares_that_its_loss_spans_the_step():
    """``dllm_mean_global_loss`` divides by the whole accumulation window's sample
    count, so the trainer must not apply ``1 / num_micro_steps`` on top."""
    assert DiffusionGemmaOmniAccelerated.loss_is_step_mean is True
