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
        "veomni/models/transformers/qwen3_5_moe/qwen3_5_moe_gpu_patch_gen_config.py",
        "veomni/models/transformers/qwen3_5_moe/qwen3_5_moe_npu_patch_gen_config.py",
    ],
)
def test_removed_stateful_gdn_cp_paths_are_absent(relative_path: str):
    source = (ROOT / relative_path).read_text()
    for removed in (
        "state_passing_lossless",
        '"kcp"',
        "gdn_lossless",
        "gdn_kcp",
        "resolve_kcp_initial_state",
        "make_gdn_cp_runtime_observer",
    ):
        assert removed not in source


def test_gpu_gdn_context_parallel_remains_fail_closed():
    source = (ROOT / "veomni/models/transformers/qwen3_5/qwen3_5_gpu_patch_gen_config.py").read_text()
    block = _function_block(source, "qwen3_5_gated_deltanet_forward_patched")
    assert "Qwen3.5 GDN context parallelism is supported on Ascend NPU only" in block
    assert "prepare_gdn_headwise_inputs(" not in block


@pytest.mark.parametrize(
    "relative_path",
    [
        "veomni/models/transformers/qwen3_5/qwen3_5_npu_patch_gen_config.py",
        "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_npu.py",
        "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
    ],
)
def test_npu_gdn_exposes_only_headwise_context_parallel(relative_path: str):
    source = (ROOT / relative_path).read_text()
    block = _gdn_forward_block(source)
    assert 'gdn_context_parallel_implementation == "headwise_lossless"' in block
    assert "prepare_gdn_headwise_inputs(" in block
    assert "restore_gdn_headwise_output(" in block
    assert "physical_to_owned" not in block
    assert "receive_initial_state" not in block
    assert "resolve_kcp_initial_state" not in block


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
    assert source.count("prepare_gated_delta_rule_qk(") == 2
    assert "producer_dtype_l2norm(query_gdr)" not in source
    assert "force_external=" not in source


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


def test_moe_npu_patchgen_keeps_headwise_and_backend_adapter_contracts():
    source = (ROOT / "veomni/models/transformers/qwen3_5_moe/qwen3_5_moe_npu_patch_gen_config.py").read_text()
    assert '"prepare_gdn_headwise_inputs"' in source
    assert '"prepare_gated_delta_rule_qk"' in source
    assert '"requires_chunked_varlen_metadata"' in source
    assert "gdn_lossless" not in source
    assert "gdn_kcp" not in source


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
    assert "head_shard_size = (" in source
    assert "num_v_heads = ulysses_local_head_count(" in source
    assert "cu_seqlens_list=valid_points" in source
    assert "chunk_indices=chunk_indices" in source
    assert "chunk_indices_list=chunk_indices_list" in source
    assert "aligned_gdn_cu_seqlens" not in source


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
