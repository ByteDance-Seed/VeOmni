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
from veomni.models.seed_omni.modules.diffusion_gemma.modeling import DiffusionGemmaOmni
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion


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
    """The mixin alone, as it runs inside a `DiffusionGemmaOmni` subclass.

    `attach_self_conditioning` and `get_input_embeddings` are defined on the model below
    the mixin, so the bare mixin carries stand-ins here. The real wiring -- a graph
    module going through `attach_self_conditioning` and back -- is covered by
    `test_the_self_conditioning_pass_survives_the_graph_dict_return`.
    """
    module = mod.TrainingMixin()
    module.config = SimpleNamespace(
        canvas_batching="flat_seq_pack" if pack else "serial",
        pad_token_id=0,
        self_conditioning_prob=0.0,
    )
    module.attach_self_conditioning = lambda micro_batch, **kwargs: micro_batch
    module.get_input_embeddings = lambda: SimpleNamespace(weight=torch.zeros(1))
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

        def __call__(self, features, **kwargs):
            return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    monkeypatch.setattr(mod, "DiffusionGemmaCollator", Recorder)
    monkeypatch.setattr(mod, "get_parallel_state", lambda: SimpleNamespace(sp_size=1))

    _module(pack=False).batch_pre(conversation_list=[[_item()]])

    assert seen["pack_samples"] is False


def test_the_canvas_axis_is_aligned_across_ranks_when_canvases_are_serial(monkeypatch):
    """``serial`` issues one decoder pass per canvas, so an unequal `K` across ranks
    means an unequal number of FSDP/EP collectives -- a hang, not a wrong number.

    The collator only pads to the longest `K` inside one micro-batch, and ranks draw
    different samples, so the module has to reduce the max `K` itself.
    """
    captured = {}

    class Collator:
        def __init__(self, **kwargs):
            pass

        def __call__(self, features):
            return {
                "decoder_input_ids": torch.zeros(1, 2, 4, dtype=torch.long),
                "labels": torch.zeros(1, 2, 4, dtype=torch.long),
                "loss_weight": torch.ones(1, 2, 4),
            }

    monkeypatch.setattr(mod, "DiffusionGemmaCollator", Collator)
    monkeypatch.setattr(mod, "get_parallel_state", lambda: SimpleNamespace(sp_size=1))
    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mod, "all_reduce", lambda value, op: 5)
    monkeypatch.setattr(
        mod,
        "pad_decoder_canvases",
        lambda micro_batch, target_k: captured.update(k=micro_batch, target_k=target_k) or micro_batch,
    )

    _module(pack=False).batch_pre(conversation_list=[[_item()]])

    assert captured["target_k"] == 5
    assert captured["k"]["decoder_input_ids"].shape[1] == 2  # the move itself is covered by data_collator's test


def test_a_packed_batch_skips_the_canvas_alignment(monkeypatch):
    """A padded slot has no sample to belong to under packing, and the mode decodes in
    one pass, so no collective count depends on `K`."""
    called = {"all_reduce": False}

    class Collator:
        def __init__(self, **kwargs):
            pass

        def __call__(self, features):
            return {"decoder_input_ids": torch.zeros(1, 2, 4, dtype=torch.long)}

    monkeypatch.setattr(mod, "DiffusionGemmaCollator", Collator)
    monkeypatch.setattr(mod, "get_parallel_state", lambda: SimpleNamespace(sp_size=1))
    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mod, "all_reduce", lambda *args, **kwargs: called.update(all_reduce=True))

    _module(pack=True).batch_pre(conversation_list=[[_item()]])

    assert called["all_reduce"] is False


def test_the_draw_is_keyed_on_the_step_and_advances_per_micro_batch():
    """`(seed, global_step, micro_step)` is what a resume can restore, and the
    micro-batch counter is what stops gradient accumulation from turning the
    Bernoulli into a per-step coin."""
    keys = []
    module = _module(pack=True)
    module.attach_self_conditioning = (
        lambda micro_batch, **kwargs: keys.append((kwargs["seed"], kwargs["global_step"], kwargs["micro_step"]))
        or micro_batch
    )

    module.on_step_context(seed=42, global_step=7)
    for _ in range(2):
        module.batch_pre(conversation_list=[[_item()]])

    assert keys == [(42, 7, 0), (42, 7, 1)]


def test_a_new_step_resets_the_micro_batch_counter():
    """Without the reset, step N's draws would keep counting from step N-1 and the
    resume replay would not reproduce them."""
    keys = []
    module = _module(pack=True)
    module.attach_self_conditioning = (
        lambda micro_batch, **kwargs: keys.append((kwargs["global_step"], kwargs["micro_step"])) or micro_batch
    )

    module.on_step_context(seed=42, global_step=7)
    module.batch_pre(conversation_list=[[_item()]])
    module.on_step_context(seed=42, global_step=8)
    module.batch_pre(conversation_list=[[_item()]])

    assert keys == [(7, 0), (8, 0)]


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


def test_a_k_canvas_example_counts_once_not_k_times():
    """An all-canvas example's ``K`` rows already sum to one example's mean.

    ``_build_diffusion_gemma_decoder_targets`` divides each canvas's ``1/C`` weights by
    ``K``, so the whole stack sums to 1 for a fully supervised answer. Counting rows
    would divide that mean by ``K`` again, and the error would track the answer-length
    distribution instead of showing up as a constant factor.
    """
    stacked = _feature(num_canvases=3)
    stacked["loss_weight"] = torch.full((3, 4), 1.0 / 12.0)  # (1/K) * (1/C) per row
    assert float(stacked["loss_weight"].sum()) == pytest.approx(1.0)

    count = mod.TrainingMixin._micro_batch_sample_count({"conversation_list": [[_item(stacked)]]})

    assert count == 1


def test_an_example_whose_draw_masked_nothing_still_counts():
    """Zero weights but a supervised sample: leaving it out biases the mean up."""
    empty = _feature()
    empty["loss_weight"] = torch.zeros(1, 4)

    count = mod.TrainingMixin._micro_batch_sample_count({"conversation_list": [[_item(empty)], [_item(_feature())]]})

    assert count == 1


def test_the_training_forward_scores_the_canvas_without_materialising_logits(monkeypatch):
    """``(B, K*C, 262144)`` fp32 logits is the one allocation this recipe cannot afford.

    The vendored forward only takes the chunked path when ``return_logits=False``, so
    the module has to ask for it: nothing above this sends the flag, and the default is
    `True`.
    """
    seen = {}

    def fake_vendored_forward(self, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(loss=None, aux_loss=None)

    monkeypatch.setattr(DiffusionGemmaForBlockDiffusion, "forward", fake_vendored_forward)

    module = DiffusionGemmaOmni.__new__(DiffusionGemmaOmni)
    module.forward(decoder_input_ids=torch.zeros(1, 1, 4, dtype=torch.long), labels=torch.zeros(1, 1, 4))

    assert seen["return_logits"] is False


def test_the_self_conditioning_pass_survives_the_graph_dict_return(monkeypatch):
    """The graph module's ``forward`` returns a dict, not the vendored output object.

    ``attach_self_conditioning`` calls ``self(...)`` for the extra pass. On an Omni class
    that lands on ``DiffusionGemmaOmni.forward``, which collects a loss dict -- so reading
    ``.self_conditioning_embeddings`` off the result raises ``AttributeError`` on the
    first training step. The embeddings have to ride through that dict.
    """
    embeddings = torch.randn(1, 4, 2)

    def fake_vendored_forward(self, **kwargs):
        # `attach_self_conditioning` asks for embeddings and no logits; a real batch here
        # would come from the collator, so the toy stands in for the encoder/decoder pair.
        return SimpleNamespace(
            loss=None,
            aux_loss=None,
            self_conditioning_embeddings=embeddings,
        )

    monkeypatch.setattr(DiffusionGemmaForBlockDiffusion, "forward", fake_vendored_forward)

    module = DiffusionGemmaOmni.__new__(DiffusionGemmaOmni)
    torch.nn.Module.__init__(module)
    module.training = True
    module.config = SimpleNamespace(canvas_length=4, self_conditioning_prob=1.0)

    attached = DiffusionGemmaForBlockDiffusion.attach_self_conditioning(
        module,
        {"decoder_input_ids": torch.zeros(1, 1, 4, dtype=torch.long)},
        seed=1,
        global_step=1,
        micro_step=0,
    )

    assert attached["self_conditioning_embeddings"] is embeddings
    assert attached["self_conditioning_mask"].tolist() == [True]


def test_the_mixin_is_registered_as_the_forward_pre_hook():
    """`batch_pre` is what `execute_train_node` calls before the endpoint, and the hook
    registry resolves it by name from the MRO. Relying on an ancestor's decorator alone
    means a rename on either side stops the draw silently."""
    from veomni.models.seed_omni.mixins.base_mixin import hook_name

    assert hook_name(DiffusionGemmaOmniAccelerated, "_omni_pre_context", "forward") == "batch_pre"


def test_the_batch_is_moved_onto_the_modules_device(monkeypatch):
    """The carrier comes from the CPU preprocessor and was never moved, and the
    self-conditioning pass runs a real forward over it."""
    seen = {}

    class Collator:
        def __init__(self, **kwargs):
            pass

        def __call__(self, features):
            return {"decoder_input_ids": torch.zeros(1, 1, 4, dtype=torch.long)}

    device = torch.device("meta")
    monkeypatch.setattr(mod, "DiffusionGemmaCollator", Collator)
    monkeypatch.setattr(mod, "get_parallel_state", lambda: SimpleNamespace(sp_size=1))

    module = _module(pack=True)
    module.attach_self_conditioning = lambda micro_batch, **kwargs: seen.update(micro_batch) or micro_batch
    module.get_input_embeddings = lambda: SimpleNamespace(weight=torch.zeros(1, device=device))

    module.batch_pre(conversation_list=[[_item()]])

    assert seen["decoder_input_ids"].device == device
