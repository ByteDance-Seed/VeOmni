# Qwen3OmniMoe Model Analysis & Implementation Plan

## 1. Architecture Overview (Existing)
[Kept as is from original file]

## 2. Parallelism (Distributed Training) Support
[Updated to reflect planned changes]

### 2.1 Supported Mechanisms
*   **Text (LLM):** Standard SP supported.
*   **Vision:** Specialized SP support with `gather_seq_scatter_heads`.
*   **Audio (Proposed):** Will align with Qwen2.5 Omni implementation to support SP.

### 2.2 Critical Missing Pieces (Addressed by this plan)
*   **`Qwen3OmniMoeAudioEncoder` SP Support:** Currently lacks the gather/scatter logic found in `Qwen2_5OmniAudioEncoder`.
*   **`Qwen3OmniMoeThinker` SP Integration:** `get_audio_features` raises error; `forward` pass lacks SP handling for audio feature injection.

---

## 3. Implementation Plan

### 3.1 `Qwen3OmniMoeAudioEncoder` Updates
**File:** `veomni/models/transformers/qwen3_omni_moe/modeling_qwen3_omni_moe.py`

1.  **Forward Pass with SP:**
    *   **Input Gathering:** If SP is enabled, `input_features` (which might be split across ranks) need to be gathered to perform convolution and processing on valid chunks.
        *   Use `gather_outputs` (or `gather_seq_scatter_heads` if input is already embedding-like, typically `input_features` is simpler so `gather_outputs` for full gather might be needed, but check `Qwen2_5Omni` uses `gather_outputs` for `input_features`).
    *   **Feature Processing:** Run convolution and encoder layers.
        *   Note: `Qwen2_5Omni` handles `cu_seqlens` padding when slicing for SP inside the encoder.
    *   **Output Slicing:** The final `token_audio` should be sliced back to SP format (sliced sequence) before returning.
        *   Use `slice_input_tensor`.

2.  **Dummy Forward:**
    *   Implement `dummy_forward` method to support FSDP when some ranks have no audio data (preventing hangs).

### 3.2 `Qwen3OmniMoeThinker` Updates
**File:** `veomni/models/transformers/qwen3_omni_moe/modeling_qwen3_omni_moe.py`

1.  **`get_audio_features`:**
    *   Remove `NotImplementedError`.
    *   Function should simply delegate to `self.audio_tower`.
    *   The `audio_tower` will handle the SP complexities.

2.  **`forward` Pass:**
    *   **Audio Feature Injection:**
        *   Retrieve `audio_features` using `get_audio_features`.
        *   **SP Alignment:** If SP is enabled, `audio_features` (returned as Sliced Seq) must be converted to **Sliced Head (Full Seq)** to match `inputs_embeds` during the injection phase.
            *   Use `gather_seq_scatter_heads(audio_features, seq_dim=0, head_dim=1, ...)`
        *   **Masking:** Ensure `audio_mask` is properly handled (it should be broadcastable or gathered if needed).
        *   **Injection:** Use `masked_scatter` to insert audio features into `inputs_embeds`.
    *   **FSDP Guard:**
        *   If no audio features are present but FSDP is enabled, call `self.audio_tower.dummy_forward()` and add a zero-weighted dummy tensor to `inputs_embeds` to maintain computation graph synchronization across ranks.

### 3.3 Verification Plan
1.  **Static Analysis:** Ensure code structure matches `Qwen2_5Omni` reference.
2.  **Mock Test:** Since we don't have a massive distributed cluster readily available for full training run in this environment, we will verify:
    *   The code imports `get_parallel_state` and related SP functions.
    *   The logic flow matches the plan.
    *   Basic single-GPU forward pass still works (regression test).
3.  **SP Logic Verification:** We can simulate `sp_enabled=True` with `sp_size=1` to ensure no shape mismatch errors occur in the added logic paths.

## 4. Checklist
1.  [ ] **AudioEncoder:** Add `gather_outputs` at start of `forward`.
2.  [ ] **AudioEncoder:** Add `slice_input_tensor` at end of `forward`.
3.  [ ] **AudioEncoder:** Add `dummy_forward`.
4.  [ ] **Thinker:** Update `get_audio_features`.
5.  [ ] **Thinker:** Update `forward` to use `gather_seq_scatter_heads` for audio features.
6.  [ ] **Thinker:** Add `dummy_forward` call in `forward`.
