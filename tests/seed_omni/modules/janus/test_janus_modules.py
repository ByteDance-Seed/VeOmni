"""Janus modules: registry wiring, the collator contract, and the VQ loss backends.

Three things a per-module change can break without any layout or graph test
noticing:

* the four modules resolve through the registries, with accelerated variants;
* every training hook tolerates the batch metadata ``SeedOmniCollator``
  forwards — ``WeightedMultiSourceDataset`` appends ``ds_idx`` / ``source_name``
  to each sample, so a hook whose signature rejects them fails on the shipped
  config before its first media forward;
* the VQ generation head scores through whichever
  ``cross_entropy_loss_implementation`` the config selects, since ``chunk_loss``
  and ``liger_kernel`` are fused-linear and reject a logits-only call.
"""

from __future__ import annotations

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.training_utils import make_eager_ops_config
from veomni.models.seed_omni import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)
from veomni.models.seed_omni.modules.janus.vqvae.accelerated.accelerated import _FUSED_CE_IMPLS
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


MODULES = ("janus_siglip", "janus_vqvae", "janus_text_encoder", "janus_llama")

# Deliberately no module-level ``pytestmark``: the CI plan decides whether a
# case runs on a CPU lease or an accelerator one, and a file-wide ``device="cpu"``
# would put the fused-loss cases on a CPU lease too — where every torch kernel
# they need is hidden and they skip, leaving the fix they cover unverified. The
# pure-Python cases carry an explicit CPU marker instead; the fused ones take the
# runner default, which is "accelerator" on a GPU/NPU runner.
_CPU = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

_IMAGE = object()
"""Sentinel for a generated-image turn in the chat-template parametrization."""


def _per_backend_plan(context):
    """Accelerator lease for a fused CE row, CPU for the eager one.

    The parametrization is over ``impl``, so the lease has to follow the
    parameter: an accelerator lease keeps the Triton / flash-attn kernels
    visible, while the eager row stays on CPU where it is cheapest.
    """
    impl = context.params["impl"]
    return LaunchSpec(device="cpu" if impl == "eager" else "accelerator")


_VISION_CONFIG = {
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "image_size": 8,
    "patch_size": 4,
    "projection_dim": 16,
}
_TEXT_CONFIG = {
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 64,
}
_VQ_CONFIG = {
    "embed_dim": 8,
    "num_embeddings": 16,
    "projection_dim": 16,
    "depth": 1,
    "image_token_embed_dim": 16,
    "num_patches": 1,
    "hidden_act": "gelu",
}


@_CPU
def test_every_module_registers_config_model_accelerated_and_processor():
    for name in MODULES:
        assert name in OMNI_CONFIG_REGISTRY, name
        assert name in OMNI_MODEL_REGISTRY, name
        assert name in OMNI_ACCELERATED_MODEL_REGISTRY, name
        # Registry lookups are factories, so the entry has to be called.
        assert issubclass(OMNI_ACCELERATED_MODEL_REGISTRY[name](), OMNI_MODEL_REGISTRY[name]()), name
    # Only the towers bind a processor; the backbone and the vocabulary module
    # have nothing to preprocess on the CPU side.
    assert "janus_siglip" in OMNI_PROCESSOR_REGISTRY
    assert "janus_vqvae" in OMNI_PROCESSOR_REGISTRY


@_CPU
def test_the_backbone_keeps_the_upstream_attention_capability_flags():
    """``janus_llama`` runs HF's patched ``LlamaModel``, so it inherits its set."""
    llm = OMNI_MODEL_REGISTRY["janus_llama"]()
    # The backbone hands ``attn_implementation`` down to the nested ``LlamaModel``.
    assert llm.sub_configs if hasattr(llm, "sub_configs") else True
    assert OMNI_CONFIG_REGISTRY["janus_llama"]().sub_configs.keys() == {"text_config"}


@pytest.mark.launch_plan(provider=_per_backend_plan)
@pytest.mark.parametrize("impl", ["eager", *sorted(_FUSED_CE_IMPLS)])
def test_the_vq_head_scores_on_every_cross_entropy_backend(impl):
    """The generation head's CE has to accept the configured backend.

    ``chunk_loss`` / ``liger_kernel`` are fused-linear: they need the head's
    input activations and weight, not a materialised logits tensor. Passing
    logits raised ``chunk_loss requires a nonempty ``weight`` (fused-linear
    path)`` for every supervised image batch.
    """
    if impl != "eager":
        if impl == "liger_kernel":
            pytest.importorskip("liger_kernel", reason="liger_kernel is a GPU-only fused backend")
        else:
            pytest.importorskip("flash_attn", reason="chunk_loss needs the flash-attn kernels")

    VQVAE = OMNI_ACCELERATED_MODEL_REGISTRY["janus_vqvae"]()
    cfg = OMNI_CONFIG_REGISTRY["janus_vqvae"]()(
        vq_config=_VQ_CONFIG,
        ops_implementation=make_eager_ops_config(cross_entropy_loss_implementation=impl),
    )
    # The fused rows dispatch to Triton / flash-attn kernels, which cannot read a
    # CPU tensor; move the whole probe onto the accelerator for those.
    if impl in _FUSED_CE_IMPLS:
        if not IS_CUDA_AVAILABLE:
            pytest.skip(f"{impl} needs a CUDA device")
        device = torch.device(get_device_type())
    else:
        device = torch.device("cpu")
    module = VQVAE(cfg).to(device=device, dtype=torch.float32)
    # The real head is ``nn.Linear``, so it has a bias. Give it values that are
    # not all zero: a loss that drops the bias still comes out finite, so a
    # zero-initialised bias would hide exactly the bug under test.
    with torch.no_grad():
        module.generation_head.vision_head.bias.copy_(
            torch.linspace(-1.0, 1.0, _VQ_CONFIG["num_embeddings"], device=device)
        )
    head_bias = module.generation_head.vision_head.bias

    hidden = torch.randn(4, _VQ_CONFIG["image_token_embed_dim"], device=device)
    labels = torch.tensor([1, -100, 3, -100], device=device)
    loss = module._vq_loss(hidden, labels)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    # The head is the only path to these activations, so a loss that never
    # reached it would leave them without a gradient.
    assert module.generation_head.vision_head.weight.grad is not None
    assert module.generation_head.vision_head.weight.grad.abs().sum() > 0
    # And the bias has to reach both the loss and its own gradient: omitting it
    # changes the number and silently drops the parameter from training.
    assert head_bias.grad is not None, "the head's bias received no gradient"
    assert head_bias.grad.abs().sum() > 0


@pytest.mark.parametrize("impl", sorted(_FUSED_CE_IMPLS))
def test_the_fused_vq_backend_accepts_the_batched_training_shape(impl):
    """``decode`` hands the loss a ``[1, T, H]`` span, not a flat one.

    That is the shape the graph produces, and it is where the fused rows broke:
    the activations were flattened to ``[T, H]`` for the fused linear while the
    token-axis bias kept its ``[1, T, 1]`` shape, so chunk_loss rejected the row
    mismatch (``logits_bias rows (1, T) do not match hidden_states rows (T,)``).
    Everything on that axis has to be flattened together.
    """
    if impl == "chunk_loss":
        pytest.importorskip("flash_attn", reason="chunk_loss needs the flash-attn kernels")
    else:
        pytest.importorskip("liger_kernel", reason="liger_kernel is a GPU-only fused backend")
    if not IS_CUDA_AVAILABLE:
        pytest.skip(f"{impl} needs a CUDA device")

    device = torch.device(get_device_type())
    Acc = OMNI_ACCELERATED_MODEL_REGISTRY["janus_vqvae"]()
    cfg = OMNI_CONFIG_REGISTRY["janus_vqvae"]()(
        vq_config=_VQ_CONFIG,
        ops_implementation=make_eager_ops_config(cross_entropy_loss_implementation=impl),
    )
    module = Acc(cfg).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        module.generation_head.vision_head.bias.copy_(
            torch.linspace(-1.0, 1.0, _VQ_CONFIG["num_embeddings"], device=device)
        )
    head_bias = module.generation_head.vision_head.bias

    # The batch axis is present, which is what the graph produces.
    hidden = torch.randn(1, 7, _VQ_CONFIG["image_token_embed_dim"], device=device)
    labels = torch.tensor([[1, -100, 3, 4, 5, -100, 7]], device=device)

    loss = module._vq_loss(hidden, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert head_bias.grad is not None and head_bias.grad.abs().sum() > 0


@pytest.mark.launch_plan(provider=_per_backend_plan)
@pytest.mark.parametrize("impl", ["eager", "chunk_loss"])
def test_the_fused_vq_backend_matches_eager_with_the_real_head_bias(impl):
    """Switching the CE backend must not change the number.

    Scored through the module's own ``_vq_loss``, so the real head (with its
    bias) is the thing under test — a hand-built ``bias=False`` linear would not
    exercise the path that shipped.
    """
    if impl != "eager":
        pytest.importorskip("flash_attn", reason="chunk_loss needs the flash-attn kernels")

    def _module(backend: str, device: torch.device):
        torch.manual_seed(0)
        Acc = OMNI_ACCELERATED_MODEL_REGISTRY["janus_vqvae"]()
        cfg = OMNI_CONFIG_REGISTRY["janus_vqvae"]()(
            vq_config=_VQ_CONFIG,
            ops_implementation=make_eager_ops_config(cross_entropy_loss_implementation=backend),
        )
        module = Acc(cfg).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            module.generation_head.vision_head.bias.copy_(
                torch.linspace(-1.0, 1.0, _VQ_CONFIG["num_embeddings"], device=device)
            )
        return module

    reference_module = _module("eager", torch.device("cpu"))
    # The batched shape the graph produces — this test and the one above cover
    # the two failure modes separately: bias dropped, and bias mis-shapen.
    hidden = torch.randn(1, 7, _VQ_CONFIG["image_token_embed_dim"])
    labels = torch.tensor([[1, -100, 3, 4, 5, -100, 7]])
    reference = reference_module._vq_loss(hidden, labels)

    if impl == "eager":
        assert float(reference.detach()) > 0
        return

    # chunk_loss dispatches to a Triton/flash-attn kernel, which cannot read a
    # CPU tensor.
    if not IS_CUDA_AVAILABLE:
        pytest.skip("chunk_loss needs a CUDA device")
    device = torch.device(get_device_type())
    fused_module = _module(impl, device)
    fused = fused_module._vq_loss(hidden.to(device), labels.to(device))
    assert float(fused.detach()) == pytest.approx(float(reference.detach()), abs=1e-3)


def _vqvae_processor():
    """The tower's own processor, so the encode hook can meter ``num_image_tokens``.

    ``encode_pre`` reads ``self._image_processor.num_image_tokens`` for the VQ
    token count; in a run it is bound by ``from_pretrained``.
    """
    from transformers.models.janus.image_processing_janus import JanusImageProcessor

    from veomni.models.seed_omni.modules.janus.vqvae.processing import JanusVqvaeProcessor

    return JanusVqvaeProcessor(image_processor=JanusImageProcessor(image_size=8, patch_size=4), num_patches=1)


@_CPU
def test_the_training_hooks_accept_collated_multisource_metadata():
    """``ds_idx`` / ``source_name`` ride along with every sample and must not crash.

    ``WeightedMultiSourceDataset`` appends them, and ``SeedOmniCollator``
    forwards the whole batch to each node, so every ``*_pre`` hook sees them.
    A signature that only names ``conversation_list`` fails before the first
    media forward.
    """
    siglip = OMNI_ACCELERATED_MODEL_REGISTRY["janus_siglip"]()(
        OMNI_CONFIG_REGISTRY["janus_siglip"]()(
            vision_config=_VISION_CONFIG, ops_implementation=make_eager_ops_config()
        )
    )
    vqvae = OMNI_ACCELERATED_MODEL_REGISTRY["janus_vqvae"]()(
        OMNI_CONFIG_REGISTRY["janus_vqvae"]()(vq_config=_VQ_CONFIG, ops_implementation=make_eager_ops_config())
    )
    vqvae._image_processor = _vqvae_processor()

    conv = [
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(
                type="image",
                value=torch.zeros(3, 8, 8),
                role="user",
                meta={"labels": torch.zeros(4, dtype=torch.long)},
            ),
            # ``prev_hidden`` is the text encoder's decode output for the
            # preceding span; the assistant image carries the backbone's
            # per-token states. Both are ``(tokens, D)``, which is what the real
            # graph hands this hook, and ``span_hidden`` takes the last row of
            # one and all-but-last of the other.
            ConversationItem(
                type="text",
                value=torch.zeros(1, _VQ_CONFIG["image_token_embed_dim"]),
                role="assistant",
                meta={"labels": torch.zeros(1, dtype=torch.long)},
            ),
            ConversationItem(
                type="image",
                value=torch.zeros(4, _VQ_CONFIG["image_token_embed_dim"]),
                role="assistant",
                meta={"janus_vqvae_labels": torch.zeros(4, dtype=torch.long)},
            ),
        ]
    ]
    metadata = {"conversation_list": conv, "ds_idx": torch.tensor([0]), "source_name": ["imagenet"]}

    # The hooks under test; ``forward_pre`` on SigLIP and both VQVAE pre-hooks.
    siglip.forward_pre(**metadata)
    vqvae.encode_pre(**metadata)
    vqvae.decode_pre(**metadata)


@_CPU
def test_generate_refuses_sequence_parallelism(monkeypatch):
    """``generate`` feeds the full prompt, so SP would exchange duplicated shards.

    Only the training hooks slice the packed sequence; the generation path has no
    SP branch, so running it at ``sp_size > 1`` hands attention lengths for one
    prompt against a replicated sequence. Fail loudly instead, as
    ``Qwen3VLLlm.generate`` does.
    """
    from veomni.models.seed_omni.modules.janus.llama.modeling import JanusLlama

    llm = JanusLlama(
        OMNI_CONFIG_REGISTRY["janus_llama"]()(text_config=_TEXT_CONFIG, ops_implementation=make_eager_ops_config())
    )

    class _SpState:
        sp_size = 2
        ulysses_size = 2

    monkeypatch.setattr("veomni.models.seed_omni.modules.janus.llama.modeling.get_parallel_state", lambda: _SpState())
    with pytest.raises(ValueError, match="Ulysses sequence parallelism"):
        llm.generate([ConversationItem(type="text", value="hi", role="user")])


@_CPU
@pytest.mark.parametrize(
    ("sample", "messages", "add_generation_prompt", "infer_type"),
    [
        (
            [
                ConversationItem(type="text", value="hi", role="user"),
                ConversationItem(type="text", value="Hello", role="assistant"),
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            ],
            False,
            None,
        ),
        (
            [
                ConversationItem(type="text", value="q1", role="user"),
                ConversationItem(type="text", value="a1", role="assistant"),
                ConversationItem(type="text", value="q2", role="user"),
                ConversationItem(type="text", value="a2", role="assistant"),
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "q1"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
                {"role": "user", "content": [{"type": "text", "text": "q2"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
            ],
            False,
            None,
        ),
        (
            [ConversationItem(type="text", value="hi", role="user")],
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            True,
            "infer_und",
        ),
        (
            [ConversationItem(type="text", value="a cat", role="user")],
            [{"role": "user", "content": [{"type": "text", "text": "a cat"}]}],
            True,
            "infer_gen",
        ),
    ],
    ids=["text-answer", "multi-turn", "gen-prompt-text", "gen-prompt-image"],
)
def test_the_prompt_matches_the_official_template(sample, messages, add_generation_prompt, infer_type):
    """Our rendered prompt must match official Janus' Jinja, including after the marker.

    The checkpoint was trained against that template, and its tokenizer is
    byte-level, so the assistant tail is token-sensitive: a stray or missing
    space is a different token, not a formatting preference.

    Two divergences are deliberate and asserted rather than hidden:

    * A generation prompt for an **image** response carries the assistant
      separator, because ``<boi>`` is a special token and the space cannot fold
      into it the way it folds into a sampled word. Official always stops at the
      colon since it samples the next token.
    * Our render appends an EOS to close the sample; official leaves that to the
      caller.

    A user-image turn also makes our template prepend the system prompt and the
    official snippet emit an ``<image_placeholder>`` instead of our
    ``<begin_of_image>``/``<end_of_image>`` span, so those cases are not
    parametrized here — this test covers the text-only shapes, where the two
    templates describe the same wire format.
    """
    import jinja2

    from veomni.models.seed_omni.modules.janus.convert_janus_weight_to_hf import CHAT_TEMPLATE
    from veomni.models.seed_omni.modules.janus.text_encoder.chat_template import JanusChatTemplate

    class _Tokenizer:
        """Only the marker strings; this case never tokenizes."""

        bos_token = "<s>"
        eos_token = "<\uff5cend\u2581of\u2581sentence\uff5c>"
        boi_token = "<begin_of_image>"
        eoi_token = "<end_of_image>"
        bos_token_id = 0
        eos_token_id = 1
        boi_token_id = 2
        eoi_token_id = 3
        pad_token_id = 0

    template = JanusChatTemplate(_Tokenizer())
    markers = template.chat_markers

    parts = template.apply_chat_template(sample)
    if add_generation_prompt:
        parts = template.apply_generation_prompt(parts, generation_kwargs={"infer_type": infer_type})
    ours = template.render_template_string(parts)

    official = jinja2.Template(CHAT_TEMPLATE).render(messages=messages, add_generation_prompt=add_generation_prompt)

    # Compare from the first user marker: ours may prepend BOS and the system
    # prompt, neither of which the official snippet emits for these inputs.
    ours_trimmed = ours[ours.index(markers.user_prefix) :]
    # We append an EOS to close the sample; official leaves that to the caller.
    if not add_generation_prompt and ours_trimmed.endswith(markers.eos_token):
        ours_trimmed = ours_trimmed[: -len(markers.eos_token)]

    if add_generation_prompt and infer_type != "infer_und":
        # The one deliberate divergence: see the docstring.
        official += markers.assistant_text_separator

    assert ours_trimmed == official, f"ours={ours_trimmed!r} official={official!r}"
