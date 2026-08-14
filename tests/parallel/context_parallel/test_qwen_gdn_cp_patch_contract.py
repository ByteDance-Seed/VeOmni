from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from veomni.models.transformers.qwen3_5 import qwen3_5_gpu_patch_gen_config as gpu_config
from veomni.models.transformers.qwen3_5.qwen3_5_gpu_patch_gen_config import (
    qwen3_5_vision_model_dummy_forward,
    qwen3_5_vision_model_forward,
)
from veomni.ops.kernels.attention._replicated_dummy import (
    is_replicated_dummy_sequence_parallel,
)


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


def test_gpu_gdn_runtime_rejects_ascend_only_kcp():
    source = (ROOT / "veomni/models/transformers/qwen3_5/qwen3_5_gpu_patch_gen_config.py").read_text()
    block = _function_block(source, "qwen3_5_gated_deltanet_forward_patched")
    assert 'self.gdn_context_parallel_implementation == "kcp"' in block
    assert "KCP ttx_bc8_m1 is currently supported on Ascend NPU only" in block


def test_npu_gdn_runtime_wires_kcp_through_lossless_ownership():
    source = (ROOT / "veomni/models/transformers/qwen3_5/qwen3_5_npu_patch_gen_config.py").read_text()
    block = _function_block(source, "qwen3_5_gated_deltanet_forward_patched")
    assert '("state_passing_lossless", "kcp")' in block
    assert "resolve_kcp_initial_state(" in block
    assert 'affine_impl="ttx_bc8_m1"' in block
    assert block.index("physical_to_owned(") < block.index("resolve_kcp_initial_state(")
    assert block.index('self.gdn_context_parallel_implementation == "kcp"') < block.index(
        "producer_dtype_l2norm(key_gdr)"
    )
    assert block.index("align_gdn_varlen_chunks(") < block.index("producer_dtype_l2norm(key_gdr)")
    assert block.index("producer_dtype_l2norm(key_gdr)") < block.index("resolve_kcp_initial_state(")
    assert "use_qk_l2norm=False" in block
    assert 'use_qk_l2norm_in_kernel=self.gdn_context_parallel_implementation != "kcp"' in block
    assert "extra_participation=make_state_participation(query_gdr)" in block
    readiness_guard = 'not getattr(\n                    self, "_gdn_kcp_affine_ready", False\n                )'
    assert readiness_guard in block
    assert block.index(readiness_guard) < block.index("and kcp_plan_requires_affine_scan(gdn_lossless_plan)")
    assert "coordinate_readiness=needs_affine_readiness" in block
    assert "self._gdn_kcp_affine_ready = True" in block
    assert block.count("attach_state_dependency(core_attn_out, initial_state)") == 1
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


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/qwen3_5_gpu_patch_gen_config.py",
        "veomni/models/transformers/qwen3_5/qwen3_5_npu_patch_gen_config.py",
    ],
)
def test_qwen35_vision_forward_reads_private_dummy_scope(relative_path: str):
    source = (ROOT / relative_path).read_text()
    block = _function_block(source, "qwen3_5_vision_model_forward")

    assert "reject_public_sequence_parallel_bypass(kwargs)" in block
    assert "is_replicated_dummy_sequence_parallel()" in block
    assert 'kwargs.pop("skip_sequence_parallel"' not in block
    assert "skip_sequence_parallel=" not in block
    assert block.count("if sequence_parallel_enabled:") == 2


def test_qwen35_dummy_vision_enters_private_scope_only_when_cp_is_enabled():
    source = (ROOT / "veomni/models/transformers/qwen3_5/qwen3_5_gpu_patch_gen_config.py").read_text()
    block = _function_block(source, "qwen3_5_vision_model_dummy_forward")

    assert "cp_dummy = bool(get_parallel_state().cp_enabled)" in block
    assert "if get_parallel_state().sp_enabled and not cp_dummy:" in block
    assert "with _replicated_dummy_sequence_parallel(_DUMMY_SP_TOKEN):" in block
    assert "skip_sequence_parallel=" not in block


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_gpu.py",
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_npu.py",
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_gpu.py",
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
    ],
)
def test_generated_qwen35_dummy_vision_preserves_private_scope_contract(relative_path: str):
    source = (ROOT / relative_path).read_text()

    assert "with _replicated_dummy_sequence_parallel(_DUMMY_SP_TOKEN):" in source
    assert "reject_public_sequence_parallel_bypass(kwargs)" in source
    assert "is_replicated_dummy_sequence_parallel()" in source
    assert 'kwargs.pop("skip_sequence_parallel"' not in source
    assert "skip_sequence_parallel=" not in source


def test_dummy_forward_rejects_forged_kwargs_and_restores_scope(monkeypatch):
    class _Recorder:
        dtype = torch.float32
        device = torch.device("cpu")

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(
                {
                    "active": is_replicated_dummy_sequence_parallel(),
                    "kwargs": kwargs,
                    "grid": kwargs["grid_thw"].tolist(),
                }
            )
            return "ok"

    monkeypatch.setattr(
        gpu_config,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, sp_enabled=True, sp_size=8),
    )
    recorder = _Recorder()
    assert qwen3_5_vision_model_dummy_forward(recorder) == "ok"
    assert is_replicated_dummy_sequence_parallel() is False
    assert recorder.calls[0]["active"] is True
    assert recorder.calls[0]["grid"] == [[1, 4, 4]]
    assert "skip_sequence_parallel" not in recorder.calls[0]["kwargs"]

    with pytest.raises(TypeError, match="not a public argument"):
        qwen3_5_vision_model_forward(
            recorder,
            hidden_states=torch.zeros(16, 4),
            grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.int32),
            skip_sequence_parallel=True,
        )


def test_ulysses_only_dummy_forward_does_not_enter_private_scope(monkeypatch):
    class _Recorder:
        dtype = torch.float32
        device = torch.device("cpu")
        calls = None

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(
                {
                    "active": is_replicated_dummy_sequence_parallel(),
                    "grid": kwargs["grid_thw"].tolist(),
                }
            )
            return "ok"

    monkeypatch.setattr(
        gpu_config,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=False, sp_enabled=True, sp_size=8),
    )
    recorder = _Recorder()
    assert qwen3_5_vision_model_dummy_forward(recorder) == "ok"
    assert recorder.calls[0]["active"] is False
    assert recorder.calls[0]["grid"] == [[1, 32, 4]]
    assert is_replicated_dummy_sequence_parallel() is False


def test_dummy_forward_restores_private_scope_after_exception(monkeypatch):
    class _Boom:
        dtype = torch.float32
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            assert is_replicated_dummy_sequence_parallel() is True
            raise RuntimeError("dummy boom")

    monkeypatch.setattr(
        gpu_config,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, sp_enabled=True, sp_size=2),
    )
    with pytest.raises(RuntimeError, match="dummy boom"):
        qwen3_5_vision_model_dummy_forward(_Boom())
    assert is_replicated_dummy_sequence_parallel() is False


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_gpu.py",
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
    ],
)
def test_qwen35_moe_aux_loss_consumes_rank_local_router_mask(relative_path: str):
    source = (ROOT / relative_path).read_text()
    assert source.count('router_attention_mask = kwargs.pop("router_attention_mask", attention_mask)') == 2
    assert source.count("                router_attention_mask,\n") == 4


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_gpu.py",
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_npu.py",
    ],
)
def test_dense_qwen35_consumes_moe_only_router_mask(relative_path: str):
    source = (ROOT / relative_path).read_text()
    assert source.count('kwargs.pop("router_attention_mask", None)') == 2
