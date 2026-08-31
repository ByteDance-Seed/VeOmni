"""Where a job's chat template comes from, and what it leaves behind.

It comes from the model runtime, built beside the preprocessor it needs, so a
trainer reads ``model.chat_template`` the way it reads ``model.tokenizer``
rather than assembling one of its own. It leaves nothing behind on the
tokenizer: training lays out tokens through :meth:`ChatTemplate.encode_messages`
itself, so an export still carries whatever jinja the checkpoint shipped with.
"""

from types import SimpleNamespace

import pytest

from veomni.data.chat_template import ChatTemplate, MultimodalChatTemplate
from veomni.models.model_runtime import VeOmniModelRuntime


NATIVE_TEMPLATE = "{{ native }}"


class _VisionTokenizer:
    def __init__(self, chat_template=NATIVE_TEMPLATE):
        self.chat_template = chat_template

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 1, "<|video_pad|>": 2, "<|vision_start|>": 3, "<|vision_end|>": 4}[token]


class _Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.video_processor = SimpleNamespace(temporal_patch_size=2)


def _stub_runtime(name, *, processor=None, tokenizer=None):
    """Enough of a runtime for ``build_model_assets``'s template step."""
    return SimpleNamespace(
        chat_template_name=name,
        processor=processor,
        tokenizer=tokenizer,
        chat_template=None,
        model_config=object(),
        model_assets=None,
        args=SimpleNamespace(tokenizer_path="tok", processor_config=None),
    )


def _build(runtime, monkeypatch):
    """Drive the merged assets step; skip the HF load so the stub's preprocessor stands."""

    def _fail(path, **kwargs):
        raise FileNotFoundError(path)

    monkeypatch.setattr("veomni.models.auto.build_processor", _fail)
    VeOmniModelRuntime.build_model_assets(runtime)


@pytest.mark.parametrize("template_name", ["qwen2vl", "qwen3vl"])
def test_a_multimodal_template_is_built_from_the_processor(template_name, monkeypatch):
    """Laying out image placeholders needs the grid parameters the processor
    used, so these templates take the processor rather than its tokenizer."""
    runtime = _stub_runtime(template_name, processor=_Processor(_VisionTokenizer()))

    _build(runtime, monkeypatch)

    assert isinstance(runtime.chat_template, MultimodalChatTemplate)


def test_a_text_template_is_built_from_the_tokenizer(monkeypatch):
    runtime = _stub_runtime("chatml", tokenizer=_VisionTokenizer())

    _build(runtime, monkeypatch)

    assert isinstance(runtime.chat_template, ChatTemplate)


@pytest.mark.parametrize("no_template", [None, ""])
def test_naming_no_template_leaves_the_job_without_one(no_template, monkeypatch):
    """Naming none is the default, and is how a job says its data needs no
    template: plaintext with no conversation to lay out, or a Qwen-Omni model
    that formats prompts through its own processor. Both used to be branches
    inside a trainer, keyed on ``data_type`` and on ``model_type``."""
    runtime = _stub_runtime(no_template, processor=_Processor(_VisionTokenizer()))

    _build(runtime, monkeypatch)

    assert runtime.chat_template is None


def test_a_model_with_no_preprocessor_warns_instead_of_failing_the_build(monkeypatch):
    """A DiT over latents loads neither tokenizer nor processor. Raising here
    would fail a build that has no use for a template anyway; the job fails at
    the data transform if it turns out it did."""
    runtime = _stub_runtime("chatml")

    _build(runtime, monkeypatch)

    assert runtime.chat_template is None


@pytest.mark.parametrize("template_name", ["qwen3vl", "chatml"])
@pytest.mark.parametrize("native", [NATIVE_TEMPLATE, None])
def test_building_a_template_never_writes_to_the_tokenizer(template_name, native, monkeypatch):
    """The exported preprocessor keeps describing the prompt format its own
    authors published, not the one this job happened to train with.

    A tokenizer carrying *no* template is the case that catches a regression:
    the stamping this replaced was guarded on the template having jinja to
    give, so only an empty tokenizer shows an unguarded write.
    """
    tokenizer = _VisionTokenizer(native)
    processor = _Processor(tokenizer)
    runtime = _stub_runtime(template_name, processor=processor, tokenizer=tokenizer)

    _build(runtime, monkeypatch)

    assert tokenizer.chat_template == native


def test_an_unregistered_name_fails_at_build_rather_than_silently(monkeypatch):
    """A typo'd template name is a config error. Answering it with ``None``
    would defer the failure to a NoneType error inside a dataloader worker."""
    runtime = _stub_runtime("chatmll", tokenizer=_VisionTokenizer())

    with pytest.raises(ValueError, match="chatmll"):
        _build(runtime, monkeypatch)
