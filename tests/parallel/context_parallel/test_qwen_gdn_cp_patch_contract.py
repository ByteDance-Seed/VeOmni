import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from veomni.models.transformers.qwen3_5 import qwen3_5_gpu_patch_gen_config as gpu_config
from veomni.models.transformers.qwen3_5.qwen3_5_gpu_patch_gen_config import (
    qwen3_5_vision_model_dummy_forward,
    qwen3_5_vision_model_forward,
)
from veomni.ops.kernels.attention._replicated_dummy import (
    _DUMMY_SP_TOKEN,
    _call_replicated_dummy_checkpointed_module,
    _replicated_dummy_sequence_parallel,
    is_replicated_dummy_sequence_parallel,
)


ROOT = Path(__file__).resolve().parents[3]


def _function_block(source: str, function_name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            if node.end_lineno is None:
                raise AssertionError(f"could not determine source span for {function_name}")
            return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"function {function_name} not found")


def _gdn_forward_block(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "qwen3_5_gated_deltanet_forward_patched":
            if node.end_lineno is None:
                raise AssertionError("could not determine patched GDN forward source span")
            return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
        if isinstance(node, ast.ClassDef) and node.name.endswith("GatedDeltaNet"):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "forward":
                    if child.end_lineno is None:
                        raise AssertionError("could not determine generated GDN forward source span")
                    return "\n".join(source.splitlines()[child.lineno - 1 : child.end_lineno])
    raise AssertionError("GDN forward function not found")


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
    selector_start = block.index("if cp_enabled and self.gdn_context_parallel_implementation not in (")
    selector_end = block.index("):", selector_start)
    selector_block = block[selector_start:selector_end]
    assert '"state_passing_lossless"' in selector_block
    assert '"kcp"' in selector_block
    assert '"headwise_lossless"' in selector_block
    assert "resolve_kcp_initial_state(" in block
    assert "cu_seqlens_list=aligned_host_cu" in block
    assert "kcp_affine_impl = resolve_kcp_affine_implementation(backend_impl)" in block
    assert "kcp_affine_backend = get_kcp_affine_backend_identity(kcp_affine_impl)" in block
    assert "affine_impl=kcp_affine_impl" in block
    assert 'affine_impl="ttx_bc8_m1"' not in block
    assert block.index("physical_to_owned_grouped(") < block.index("resolve_kcp_initial_state(")
    assert block.count("physical_to_owned_grouped(") == 1
    assert "mixed_qkv, b, a = physical_to_owned_grouped(" in block
    ownership_block = block[block.index("elif cp_enabled:") :]
    assert ownership_block.index("align_gdn_varlen_chunks(") < ownership_block.index("prepare_gated_delta_rule_qk(")
    assert ownership_block.index("prepare_gated_delta_rule_qk(") < ownership_block.index("resolve_kcp_initial_state(")
    empty_owner_guard = "if gdn_lossless_plan.local.owned_token_count == 0:"
    assert ownership_block.index(empty_owner_guard) < ownership_block.index("prepare_gated_delta_rule_qk(")
    empty_owner_block = ownership_block[
        ownership_block.index(empty_owner_guard) : ownership_block.index(
            "else:", ownership_block.index(empty_owner_guard)
        )
    ]
    assert "use_qk_l2norm_in_kernel = False" in empty_owner_block
    assert "prepare_gated_delta_rule_qk(" not in empty_owner_block
    assert 'force_external=self.gdn_context_parallel_implementation == "kcp"' in block
    assert "use_qk_l2norm=False" in block
    assert "use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel" in block
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
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/qwen3_5_npu_patch_gen_config.py",
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_npu.py",
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
    ],
)
def test_npu_kcp_groups_five_ulysses_inputs_before_ownership(relative_path: str):
    source = (ROOT / relative_path).read_text()
    block = _gdn_forward_block(source)
    grouped_start = block.index('if self.gdn_context_parallel_implementation == "kcp":')
    grouped_call = block.index("gather_seq_scatter_heads_grouped(", grouped_start)
    fallback_start = block.index("else:", grouped_call)
    reorder_start = block.index("reorder_ulysses_rank_major_to_sample_major(", fallback_start)
    ownership_start = block.index("physical_to_owned_grouped(", reorder_start)

    assert grouped_call < fallback_start < reorder_start < ownership_start
    grouped_block = block[grouped_start:fallback_start]
    assert "(q_proj, k_proj, v_proj, b, a)" in grouped_block
    assert "VEOMNI_GDN_KCP_RUNTIME grouped_ulysses_a2a=true" in grouped_block
    assert grouped_block.count("gather_seq_scatter_heads_grouped(") == 1


def test_moe_npu_patchgen_imports_grouped_ulysses_for_shared_gdn_forward():
    source = (ROOT / "veomni/models/transformers/qwen3_5_moe/qwen3_5_moe_npu_patch_gen_config.py").read_text()
    assert '"gather_seq_scatter_heads_grouped"' in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/qwen3_5_npu_patch_gen_config.py",
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_npu.py",
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
    ],
)
def test_npu_gdn_mojo_routes_share_external_qk_norm(relative_path: str):
    source = (ROOT / relative_path).read_text()
    assert source.count("prepare_gated_delta_rule_qk(") == 3
    headwise_start = source.index("elif headwise_enabled:")
    ownership_start = source.index("elif cp_enabled:", headwise_start)
    headwise_block = source[headwise_start:ownership_start]
    assert headwise_block.count("prepare_gated_delta_rule_qk(") == 1
    assert "producer_dtype_l2norm(query_gdr)" not in source
    assert 'use_qk_l2norm_in_kernel=self.gdn_context_parallel_implementation != "kcp"' not in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/qwen3_5_npu_patch_gen_config.py",
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_npu.py",
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
    ],
)
def test_npu_headwise_uses_cp_major_ulysses_inner_flat_rank(relative_path: str):
    source = (ROOT / relative_path).read_text()
    rank_contract_start = source.index("expected_sp_size = parallel_state.cp_size * parallel_state.ulysses_size")
    prepare_start = source.index("prepare_gdn_headwise_inputs(", rank_contract_start)
    rank_contract = source[rank_contract_start:prepare_start]
    compact_rank_contract = " ".join(rank_contract.split())

    assert (
        "expected_sp_rank = parallel_state.cp_rank * parallel_state.ulysses_size + "
        "( parallel_state.ulysses_rank if parallel_state.ulysses_enabled else 0 )"
    ) in compact_rank_contract
    assert "sp_size != expected_sp_size" in rank_contract
    assert "gdn_headwise_layout.world_size != expected_sp_size" in rank_contract
    assert "head_parallel_rank != expected_sp_rank" in rank_contract
    assert "gdn_headwise_layout.rank != expected_sp_rank" in rank_contract
    assert "headwise_lossless requires CP-major/Ulysses-inner flattened SP rank order" in rank_contract


def test_moe_npu_patchgen_imports_shared_external_qk_norm_contract():
    source = (ROOT / "veomni/models/transformers/qwen3_5_moe/qwen3_5_moe_npu_patch_gen_config.py").read_text()
    assert '"prepare_gated_delta_rule_qk"' in source
    assert '"resolve_kcp_affine_implementation"' in source
    assert '"prepare_kcp_affine_summary"' in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_npu.py",
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
    ],
)
def test_generated_npu_kcp_affine_tracks_the_selected_gdr_backend(relative_path: str):
    source = (ROOT / relative_path).read_text()
    assert "kcp_affine_impl = resolve_kcp_affine_implementation(backend_impl)" in source
    assert "affine_impl=kcp_affine_impl" in source
    assert "prepare_kcp_affine_summary(" in source
    assert 'affine_impl="ttx_bc8_m1"' not in source


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
    assert "_call_replicated_dummy_checkpointed_module(" in source
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


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_dummy_checkpoint_reenters_private_scope_during_backward(use_reentrant: bool):
    seen = []

    class _Block(torch.nn.Module):
        gradient_checkpointing = True

        def __init__(self):
            super().__init__()
            self._gradient_checkpointing_func = lambda function, *args: checkpoint(
                function,
                *args,
                use_reentrant=use_reentrant,
            )

        def forward(self, hidden_states, *, scale):
            seen.append(is_replicated_dummy_sequence_parallel())
            return hidden_states.sin() * hidden_states.cos() * scale

    block = _Block().train()
    hidden_states = torch.randn(8, requires_grad=True)
    oracle_input = hidden_states.detach().clone().requires_grad_(True)
    oracle = oracle_input.sin() * oracle_input.cos() * 3
    oracle.sum().backward()

    with _replicated_dummy_sequence_parallel(_DUMMY_SP_TOKEN):
        output = _call_replicated_dummy_checkpointed_module(
            _DUMMY_SP_TOKEN,
            block,
            hidden_states,
            scale=3,
        )
    assert is_replicated_dummy_sequence_parallel() is False
    output.sum().backward()

    assert seen == [True, True]
    torch.testing.assert_close(hidden_states.grad, oracle_input.grad)
    assert is_replicated_dummy_sequence_parallel() is False


def test_dummy_checkpoint_callable_keeps_module_binding_for_fsdp_shim():
    class _Block(torch.nn.Module):
        gradient_checkpointing = True

        def __init__(self):
            super().__init__()

            def checkpoint_fn(function, *args):
                assert function.__self__ is self
                return function(*args)

            self._gradient_checkpointing_func = checkpoint_fn

        def forward(self, hidden_states, *, offset):
            assert is_replicated_dummy_sequence_parallel() is True
            return hidden_states + offset

    block = _Block().train()
    with _replicated_dummy_sequence_parallel(_DUMMY_SP_TOKEN):
        output = _call_replicated_dummy_checkpointed_module(
            _DUMMY_SP_TOKEN,
            block,
            torch.ones(2),
            offset=2,
        )
    torch.testing.assert_close(output, torch.full((2,), 3.0))
    assert is_replicated_dummy_sequence_parallel() is False


def test_dummy_checkpoint_helper_fails_closed_and_restores_after_recompute_error():
    class _Block(torch.nn.Module):
        gradient_checkpointing = True

        def __init__(self):
            super().__init__()
            self.calls = 0
            self._gradient_checkpointing_func = lambda function, *args: checkpoint(
                function,
                *args,
                use_reentrant=True,
            )

        def forward(self, hidden_states):
            self.calls += 1
            assert is_replicated_dummy_sequence_parallel() is True
            if self.calls == 2:
                raise RuntimeError("recompute boom")
            return hidden_states.sin()

    with pytest.raises(RuntimeError, match="forged activation"):
        _call_replicated_dummy_checkpointed_module(object(), torch.nn.Identity(), torch.ones(1))
    with pytest.raises(RuntimeError, match="forged activation"):
        _call_replicated_dummy_checkpointed_module(_DUMMY_SP_TOKEN, torch.nn.Identity(), torch.ones(1))

    block = _Block().train()
    hidden_states = torch.randn(4, requires_grad=True)
    with _replicated_dummy_sequence_parallel(_DUMMY_SP_TOKEN):
        output = _call_replicated_dummy_checkpointed_module(_DUMMY_SP_TOKEN, block, hidden_states)
    with pytest.raises(RuntimeError, match="recompute boom"):
        output.sum().backward()
    assert is_replicated_dummy_sequence_parallel() is False


def test_dummy_checkpoint_helper_ac_off_uses_normal_module_call():
    class _Block(torch.nn.Module):
        gradient_checkpointing = False

        def forward(self, hidden_states):
            assert is_replicated_dummy_sequence_parallel() is True
            return hidden_states + 1

    block = _Block().train()
    block._gradient_checkpointing_func = lambda *_args: pytest.fail("checkpoint should not run")
    with _replicated_dummy_sequence_parallel(_DUMMY_SP_TOKEN):
        output = _call_replicated_dummy_checkpointed_module(_DUMMY_SP_TOKEN, block, torch.ones(2))
    torch.testing.assert_close(output, torch.full((2,), 2.0))
    assert is_replicated_dummy_sequence_parallel() is False


def test_dummy_checkpoint_helper_rejects_missing_checkpoint_function():
    class _Block(torch.nn.Module):
        gradient_checkpointing = True

        def forward(self, hidden_states):
            return hidden_states

    block = _Block().train()
    with _replicated_dummy_sequence_parallel(_DUMMY_SP_TOKEN):
        with pytest.raises(RuntimeError, match="without a checkpoint function"):
            _call_replicated_dummy_checkpointed_module(_DUMMY_SP_TOKEN, block, torch.ones(1))
    assert is_replicated_dummy_sequence_parallel() is False


def test_dummy_checkpoint_helper_eval_path_preserves_module_hooks():
    events = []

    class _Block(torch.nn.Module):
        gradient_checkpointing = True

        def forward(self, hidden_states):
            events.append("forward")
            return hidden_states + 1

    block = _Block().eval()
    block._gradient_checkpointing_func = lambda *_args: pytest.fail("checkpoint should not run in eval")
    block.register_forward_pre_hook(lambda *_args: events.append("pre"))
    block.register_forward_hook(lambda *_args: events.append("post"))
    with _replicated_dummy_sequence_parallel(_DUMMY_SP_TOKEN):
        output = _call_replicated_dummy_checkpointed_module(_DUMMY_SP_TOKEN, block, torch.ones(2))
    assert events == ["pre", "forward", "post"]
    torch.testing.assert_close(output, torch.full((2,), 2.0))
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
    # Each of the two LM forwards now has three correctness-preserving paths:
    # unified-SP global sufficient statistics, configured VeOmni backend, and
    # the single-rank Transformers fallback.  All must consume the rank-local
    # router mask produced before decoder dispatch.
    assert source.count("                router_attention_mask,\n") == 6
    assert source.count("                    router_attention_mask,\n                    group=sp_group,\n") == 2


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_moe/generated/patched_modeling_qwen3_moe_gpu.py",
        "veomni/models/transformers/qwen3_moe/generated/patched_modeling_qwen3_moe_npu.py",
    ],
)
def test_qwen3_moe_aux_loss_consumes_rank_local_router_mask(relative_path: str):
    source = (ROOT / relative_path).read_text()
    assert source.count('router_attention_mask = kwargs.pop("router_attention_mask", attention_mask)') == 1
    assert source.count("                router_attention_mask,\n") == 2


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
