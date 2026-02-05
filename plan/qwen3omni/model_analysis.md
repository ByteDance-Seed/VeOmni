# Qwen3OmniMoe Model Analysis

## 1. Architecture Overview

The `Qwen3OmniMoe` is a composite multimodal model designed for "Omni" capabilities (Text, Image, Video, Audio In/Out). The top-level class is `Qwen3OmniMoeForConditionalGeneration`, which orchestrates two main sub-models: the **Thinker** and the **Talker**.

### 1.1 Thinker (`Qwen3OmniMoeThinkerForConditionalGeneration`)
The "Brain" of the system, responsible for understanding multimodal inputs and generating text responses.

*   **Encoders:**
    *   **Audio Tower:** `Qwen3OmniMoeAudioEncoder` (CNN + Transformer) processes raw audio features.
    *   **Vision Tower:** `Qwen3OmniMoeVisionEncoder` (ViT-like) processes Image and Video inputs. It supports **DeepStack**, injecting visual features into early layers of the LLM.
*   **LLM Backbone:**
    *   `Qwen3OmniMoeThinkerTextModel`: A Sparse Mixture-of-Experts (MoE) Transformer.
    *   Uses **Rotary Positional Embeddings (RoPE)** with support for 3D (Temporal, Height, Width) positions for vision and 1D for text/audio.
*   **Key Components:**
    *   `Qwen3OmniMoeThinkerTextDecoderLayer`: Standard Transformer decoder block with MoE support.
    *   `Qwen3OmniMoeThinkerTextSparseMoeBlock`: The MoE layer with router and experts.

### 1.2 Talker (`Qwen3OmniMoeTalkerForConditionalGeneration`)
The "Voice" of the system, responsible for generating audio speech from the Thinker's hidden states and text output.

*   **Core Model:** A separate MoE Causal LM (`Qwen3OmniMoeTalkerModel`) specialized for audio code generation.
*   **Code Predictor:** `Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration` predicts residual codes for high-fidelity audio reconstruction.
*   **Vocoder:** `Qwen3OmniMoeCode2Wav` converts the generated discrete codes into raw audio waveforms using a Transformer + ConvNeXt-based decoder.

---

## 2. Parallelism (Distributed Training) Support

The model has built-in support for **Sequence Parallelism (SP)** and **FSDP**, but with varying degrees of maturity across modalities.

### 2.1 Supported Mechanisms
*   **Text (LLM):**
    *   Standard Sequence Parallelism is implemented.
    *   Loss calculation uses `reduce_sequence_parallel_loss`.
    *   Embeddings and RoPE are sliced/gathered as needed.
*   **Vision:**
    *   **DeepStack SP:** The code explicitly handles SP for visual features injected into the LLM.
    *   `gather_seq_scatter_heads` and `gather_heads_scatter_seq` are used to transition embeddings between "sequence parallel" and "tensor parallel" states (or just to gather for operations that need full context).
    *   Visual masks (`image_mask`, `video_mask`) are sliced per rank to ensure each GPU only processes its portion of the sequence.
*   **FSDP Safety:**
    *   `dummy_forward` methods exist in encoders to prevent FSDP hangs when some ranks receive no multimodal data (a common issue in sparse multimodal batches).

### 2.2 Missing Support (The "Audio" Gap)
*   **Audio Input SP:** Explicitly **NOT** implemented.
    *   In `Qwen3OmniMoeThinkerForConditionalGeneration.get_audio_features`:
        ```python
        # TODO audio sp support
        if get_parallel_state().sp_enabled:
            raise NotImplementedError("audio sp is not supported yet.")
        ```
*   **Audio Output (Talker):**
    *   The `generate` method enforces `input_ids.shape[0] == 1` (batch size 1) when `return_audio=True`.
    *   This suggests distributed inference for the Talker might not be fully optimized or tested.

---

## 3. How to Support Audio Data

To fully enable audio support, especially in distributed training environments (using Sequence Parallelism), the following changes are required in `veomni/models/transformers/qwen3_omni_moe/modeling_qwen3_omni_moe.py`:

### 3.1 Implement Audio Sequence Parallelism
In `Qwen3OmniMoeThinkerForConditionalGeneration.get_audio_features`, you need to replicate the SP logic used for Vision.

**Current Code:**
```python
if get_parallel_state().sp_enabled:
    raise NotImplementedError("audio sp is not supported yet.")
```

**Required Implementation:**
1.  **Scatter Audio Features:** After the audio tower processes the input, the resulting features need to be scattered across the sequence dimension if training with SP.
    ```python
    if self.training and get_parallel_state().sp_enabled:
        # (batch, seq, hidden) -> (batch, seq/sp_size, hidden) *conceptually*
        # Actually usually (seq/sp_size, hidden) if batch is 1 or handled by gather_seq_scatter_heads
        audio_features = gather_seq_scatter_heads(
            audio_features, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
        )
    ```
2.  **Handle Masks:** You likely need to pre-compute or adjust `audio_mask` so that each rank knows which part of the audio sequence it is responsible for, similar to how `rank_image_mask` is derived in the `forward` method for vision.

### 3.2 Update Forward Pass
In the main `forward` method of the Thinker:
1.  **Mask Handling:** Ensure `audio_mask` is compatible with SP slicing.
2.  **Embedding Injection:** When injecting `audio_features` into `inputs_embeds` (using `masked_scatter`), ensure that both tensors are correctly sliced for the current rank.

### 3.3 Talker Considerations
If you intend to train the Audio Output (Talker) part:
*   Ensure `Qwen3OmniMoeTalkerForConditionalGeneration`'s forward pass respects SP. It currently inherits from `Qwen3OmniMoeThinkerTextPreTrainedModel`, so standard text SP should work, but data loading and input preparation for the Talker need to ensure inputs are distributed correctly across ranks.

### 3.4 Summary Checklist for Audio Support
1.  [ ] Remove `NotImplementedError` in `get_audio_features`.
2.  [ ] Add SP scatter logic for `audio_features`.
3.  [ ] Verify `audio_mask` logic handles sliced sequences correctly.
4.  [ ] Test with `get_parallel_state().sp_enabled = True`.
