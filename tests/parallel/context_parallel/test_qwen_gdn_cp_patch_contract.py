from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _function_block(source: str, function_name: str) -> str:
    marker = f"def {function_name}("
    start = source.index(marker)
    next_decorator = source.find("\n@", start)
    return source[start:] if next_decorator < 0 else source[start:next_decorator]


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/qwen3_5_gpu_patch_gen_config.py",
        "veomni/models/transformers/qwen3_5/qwen3_5_npu_patch_gen_config.py",
    ],
)
def test_gdn_cp_plan_uses_host_cu_before_cache_lookup(relative_path: str):
    source = (ROOT / relative_path).read_text()
    block = _function_block(source, "qwen3_5_gated_deltanet_forward_patched")

    assert "valid_points = [int(point) for point in cu_seqlens_list]" in block
    assert "cu_seq_lens_q.detach().cpu().tolist()" not in block
    assert block.index("valid_points =") < block.index("_gdn_lossless_plan_cache")
    assert "ulysses_local_cu_from_global" not in block
    assert "cu_seqlens_list=gdn_lossless_plan.owned_cu_seqlens" in block


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/qwen3_5_gpu_patch_gen_config.py",
        "veomni/models/transformers/qwen3_5/qwen3_5_npu_patch_gen_config.py",
    ],
)
def test_gdn_runtime_observer_is_wired_through_lossless_ownership(relative_path: str):
    source = (ROOT / relative_path).read_text()
    block = _function_block(source, "qwen3_5_gated_deltanet_forward_patched")
    assert "gdn_cp_runtime_evidence" in block
    assert "observer=gdn_cp_observer" in block


@pytest.mark.parametrize(
    ("relative_path", "function_name"),
    [
        (
            "veomni/models/transformers/qwen3_5/qwen3_5_gpu_patch_gen_config.py",
            "qwen3_5_decoder_layer_forward_patched",
        ),
        (
            "veomni/models/transformers/qwen3_5/qwen3_5_npu_patch_gen_config.py",
            "qwen3_5_decoder_layer_forward_patched",
        ),
        (
            "veomni/models/transformers/qwen3_5_moe/qwen3_5_moe_gpu_patch_gen_config.py",
            "qwen3_5_moe_decoder_layer_forward_patched",
        ),
        (
            "veomni/models/transformers/qwen3_5_moe/qwen3_5_moe_npu_patch_gen_config.py",
            "qwen3_5_moe_decoder_layer_forward_patched",
        ),
    ],
)
def test_decoder_keeps_physical_cu_for_ring_and_routes_global_cu_to_gdn(
    relative_path: str,
    function_name: str,
):
    source = (ROOT / relative_path).read_text()
    block = _function_block(source, function_name)

    assert 'kwargs.pop("linear_attn_cu_seqlens_list_q", None)' in block
    assert 'kwargs.pop("cu_seqlens_list_q"' not in block
    assert "cu_seqlens_list=linear_attn_cu_seqlens_list" in block


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_npu.py",
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
    ],
)
def test_generated_npu_model_separates_local_and_linear_cu_metadata(relative_path: str):
    source = (ROOT / relative_path).read_text()

    assert 'kwargs["cu_seqlens_list_q"] =' in source
    assert 'kwargs["linear_attn_cu_seqlens_list_q"] = cu_seqlens_list' in source
    assert "num_v_heads = ulysses_local_head_count(" in source
    assert "aligned_host_cu = aligned_gdn_cu_seqlens(" in source
    assert "cu_seqlens_list=aligned_cu_list" in source
    assert "chunk_indices=aligned_chunk_indices" in source
    assert "chunk_indices_list=aligned_chunk_indices_list" in source


@pytest.mark.parametrize(
    ("relative_path", "function_name"),
    [
        (
            "veomni/models/transformers/qwen3_5/qwen3_5_gpu_patch_gen_config.py",
            "qwen3_5_model_forward",
        ),
        (
            "veomni/models/transformers/qwen3_5_moe/qwen3_5_moe_gpu_patch_gen_config.py",
            "qwen3_5_moe_model_forward_patched",
        ),
    ],
)
def test_text_only_cp_skips_multimodal_full_sequence_transport(relative_path: str, function_name: str):
    source = (ROOT / relative_path).read_text()
    block = _function_block(source, function_name)

    assert "has_multimodal_inputs = pixel_values is not None or pixel_values_videos is not None" in block
    guarded = "if get_parallel_state().sp_enabled and has_multimodal_inputs:"
    assert block.count(guarded) == 3
    assert block.count("gather_outputs(inputs_embeds") == 1
    assert block.count("slice_input_tensor(inputs_embeds") == 1
