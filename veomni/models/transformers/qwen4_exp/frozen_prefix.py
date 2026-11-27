"""Avoid artificial input gradients for frozen-prefix, non-reentrant text SFT."""


def remove_frozen_prefix_input_grads(
    model, *, frozen_prefix_layers, text_sft, gradient_checkpointing, use_reentrant, lora_enabled
):
    """Remove only HF's embedding-output hook after checkpoint initialization.

    Never detach hidden states: trainable suffix parameters still need gradients.
    Reentrant checkpointing, LoRA, and multimodal training retain HF's policy.
    """
    count = frozen_prefix_layers
    if (
        getattr(model.config, "model_type", None) != "qwen4_exp"
        or not count
        or not text_sft
        or not gradient_checkpointing
        or use_reentrant
        or lora_enabled
    ):
        return False
    language_model = model.model.language_model
    if not 0 < count < len(language_model.layers):
        raise ValueError("Frozen prefix must leave a trainable suffix")
    frozen_modules = [language_model.embed_tokens, *language_model.layers[:count]]
    if any(p.requires_grad for module in frozen_modules for p in module.parameters()):
        raise ValueError("Cannot remove input gradient hook with trainable embedding/prefix parameters")
    hook = getattr(model, "_require_grads_hook", None)
    if hook is None:
        return False
    model.disable_input_require_grads()
    return True
