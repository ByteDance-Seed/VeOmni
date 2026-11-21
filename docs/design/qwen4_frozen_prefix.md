# Frozen-prefix checkpoint input gradients

`remove_frozen_prefix_input_grads` in `veomni.models.transformers.qwen4_exp.frozen_prefix` is an opt-in helper for text-only Qwen4-Exp training with a frozen embedding and leading decoder layers. Call it after gradient-checkpoint initialization. Pass the frozen layer count, whether this is text SFT, checkpoint enablement/reentrancy and LoRA status explicitly.

The helper removes only HuggingFace's embedding-output input-gradient hook. It never detaches decoder hidden states. Trainable suffix parameters still receive gradients; the frozen prefix no longer gets replayed solely because the artificial input gradient was enabled. Reentrant checkpointing, LoRA, multimodal and zero-prefix cases retain existing behavior.

Eight CPU regression tests verify trainable-gradient equality, replay counts and policy guards. This PR does not add a default freeze policy, change model trainability, or enable activation retention.
