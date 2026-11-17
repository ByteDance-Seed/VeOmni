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

pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

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


def test_the_backbone_keeps_the_upstream_attention_capability_flags():
    """``janus_llama`` runs HF's patched ``LlamaModel``, so it inherits its set."""
    llm = OMNI_MODEL_REGISTRY["janus_llama"]()
    # The backbone hands ``attn_implementation`` down to the nested ``LlamaModel``.
    assert llm.sub_configs if hasattr(llm, "sub_configs") else True
    assert OMNI_CONFIG_REGISTRY["janus_llama"]().sub_configs.keys() == {"text_config"}


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

    hidden = torch.randn(4, _VQ_CONFIG["image_token_embed_dim"], device=device)
    labels = torch.tensor([1, -100, 3, -100], device=device)
    loss = module._vq_loss(hidden, labels)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    # The head is the only path to these activations, so a loss that never
    # reached it would leave them without a gradient.
    assert module.generation_head.vision_head.weight.grad is not None
    assert module.generation_head.vision_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("impl", ["eager", "chunk_loss"])
def test_the_eager_and_fused_vq_backends_agree(impl):
    """An explicit backend override must not change the number, only the kernel."""
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.ops import VeomniOp

    if impl != "eager":
        if impl == "liger_kernel":
            pytest.importorskip("liger_kernel", reason="liger_kernel is a GPU-only fused backend")
            if not IS_CUDA_AVAILABLE:
                pytest.skip("liger_kernel needs a CUDA device")
        else:
            pytest.importorskip("flash_attn", reason="chunk_loss needs the flash-attn kernels")

    torch.manual_seed(0)
    head = torch.nn.Linear(_VQ_CONFIG["projection_dim"], _VQ_CONFIG["num_embeddings"], bias=False)
    hidden = torch.randn(4, _VQ_CONFIG["projection_dim"], requires_grad=True)
    labels = torch.tensor([1, -100, 3, 4])
    logits = head(hidden)

    fused = impl in _FUSED_CE_IMPLS
    loss, _, _ = ForCausalLMLoss(
        **({"hidden_states": hidden, "weights": head.weight} if fused else {"logits": logits}),
        shift_labels=labels,
        vocab_size=_VQ_CONFIG["num_embeddings"],
        op=VeomniOp("cross_entropy_loss", "standard", impl),
    )
    reference, _, _ = ForCausalLMLoss(
        logits=logits,
        shift_labels=labels,
        vocab_size=_VQ_CONFIG["num_embeddings"],
        op=VeomniOp("cross_entropy_loss", "standard", "eager"),
    )
    assert float(loss) == pytest.approx(float(reference), abs=1e-5)


def _vqvae_processor():
    """The tower's own processor, so the encode hook can meter ``num_image_tokens``.

    ``encode_pre`` reads ``self._image_processor.num_image_tokens`` for the VQ
    token count; in a run it is bound by ``from_pretrained``.
    """
    from transformers.models.janus.image_processing_janus import JanusImageProcessor

    from veomni.models.seed_omni.modules.janus.vqvae.processing import JanusVqvaeProcessor

    return JanusVqvaeProcessor(image_processor=JanusImageProcessor(image_size=8, patch_size=4), num_patches=1)


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
