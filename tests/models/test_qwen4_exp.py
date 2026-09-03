# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml

from tests.tools.training_utils import make_eager_ops_config
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models import build_foundation_model
from veomni.models.transformers.qwen4_exp import register_qwen4_exp_modeling, register_qwen4_exp_text_modeling
from veomni.models.transformers.qwen4_exp.checkpoint_tensor_converter import (
    Qwen4ExpCheckpointTensorConverter,
    convert_qwen4_exp_fqn_to_index_mapping,
    create_qwen4_exp_checkpoint_tensor_converter,
)


_TOY_CONFIG = "./tests/toy_config/qwen4_exp_toy/config.json"
_INTERNAL_TRAIN_CONFIG = Path("configs/text/qwen4exp.yaml")


def test_qwen4_exp_converter_pads_independent_ple_shards_and_ignores_mtp():
    converter = Qwen4ExpCheckpointTensorConverter(split_ngram_parts=2, shard_row_divisor=4)
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    shard_0 = torch.arange(6).reshape(2, 3)

    assert converter.convert("mtp.fc_embedding.weight", torch.ones(1)) is None
    result = converter.convert(f"{prefix}.shard_0.weight", shard_0)

    assert result is not None
    assert result.name == f"{prefix}.shard_0.weight"
    assert result.tensor.shape == (4, 3)
    assert torch.equal(result.tensor[:2], shard_0)
    assert torch.equal(result.tensor[2:], torch.zeros(2, 3, dtype=shard_0.dtype))
    assert converter.is_dim0_zero_pad(result.name)
    assert converter.should_skip_without_loading("mtp.fc_embedding.weight")
    assert converter.ignored_mtp_tensors == 1
    with patch(
        "veomni.models.transformers.qwen4_exp.checkpoint_tensor_converter.logger.warning_rank0"
    ) as warning_rank0:
        assert converter.finalize() == []
    warning_rank0.assert_called_once()
    assert warning_rank0.call_args.args[1] == 1


def test_qwen4_exp_converter_rejects_out_of_range_ple_shard():
    converter = Qwen4ExpCheckpointTensorConverter(split_ngram_parts=2, shard_row_divisor=4)
    with pytest.raises(RuntimeError, match="outside configured split_ngram_parts"):
        converter.convert(
            "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_2.weight",
            torch.zeros(2, 3),
        )


def test_qwen4_exp_converter_factory_supports_nested_and_flat_configs():
    flat = SimpleNamespace(split_ngram_parts=4, make_ngram_vocab_size_divisible_by=16)
    nested_model = SimpleNamespace(config=SimpleNamespace(text_config=flat))
    flat_model = SimpleNamespace(config=flat)

    assert create_qwen4_exp_checkpoint_tensor_converter(nested_model).split_ngram_parts == 4
    assert create_qwen4_exp_checkpoint_tensor_converter(flat_model).split_ngram_parts == 4
    assert create_qwen4_exp_checkpoint_tensor_converter(flat_model).shard_row_divisor == 16


def test_qwen4_exp_index_mapping_drops_mtp_and_merges_ple_shards():
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    mapping = {
        "model.language_model.embed_tokens.weight": 0,
        f"{prefix}.shard_0.weight": 3,
        f"{prefix}.shard_1.weight": 4,
        "mtp.fc_embedding.weight": 5,
    }

    converted = convert_qwen4_exp_fqn_to_index_mapping(mapping)

    assert converted == {
        "model.language_model.embed_tokens.weight": 0,
        f"{prefix}.shard_0.weight": 3,
        f"{prefix}.shard_1.weight": 4,
    }


def test_qwen4_exp_rejects_non_vlm_architectures():
    with pytest.raises(NotImplementedError, match="only Qwen4ExpForConditionalGeneration"):
        register_qwen4_exp_modeling("Qwen4ExpForCausalLM")
    with pytest.raises(NotImplementedError, match="does not support standalone text architectures"):
        register_qwen4_exp_text_modeling("Qwen4ExpForCausalLM")


def test_qwen4_exp_parallel_plan_declares_disjoint_ple_and_ep_parameters():
    model = build_foundation_model(
        config_path=_TOY_CONFIG,
        weights_path=None,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=make_eager_ops_config(),
    )

    plan = model.get_parallel_plan()

    assert set(plan.extra_parallel_plan) == {"ple", "ep"}
    assert set(plan.extra_parallel_persistent_modules) == {"ple"}
    assert set(plan.extra_parallel_plan["ep"]) == {
        "model.language_model.layers.*.mlp.experts.gate_up_proj",
        "model.language_model.layers.*.mlp.experts.down_proj",
    }
    assert not (set(plan.extra_parallel_plan["ple"]) & set(plan.extra_parallel_plan["ep"]))


def test_parallel_plan_rejects_parameter_shared_by_multiple_extra_parallel_dimensions():
    from torch.distributed.tensor import Shard

    from veomni.distributed.parallel_plan import ParallelPlan

    model = torch.nn.Linear(2, 2, bias=False)
    plan = ParallelPlan(
        extra_parallel_plan={
            "ple": {"weight": Shard(0)},
            "ep": {"weight": Shard(0)},
        }
    )
    fake_meshes = {
        "ple": {"ple": object()},
        "ep": {"ep": object()},
    }

    with pytest.raises(ValueError, match="matches multiple enabled dimensions"):
        plan.apply(model, fake_meshes)


@pytest.mark.parametrize(
    ("is_npu", "module_suffix"),
    [(False, "patched_modeling_qwen4_exp_gpu"), (True, "patched_modeling_qwen4_exp_npu")],
)
def test_qwen4_exp_registry_selects_device_modeling(is_npu, module_suffix):
    with patch("veomni.models.transformers.qwen4_exp.IS_NPU_AVAILABLE", is_npu):
        model_cls = register_qwen4_exp_modeling("Qwen4ExpForConditionalGeneration")

    assert model_cls.__module__.endswith(module_suffix)


def test_qwen4_exp_internal_config_resolves_default_moe_backend_on_npu():
    raw_ops_config = yaml.safe_load(_INTERNAL_TRAIN_CONFIG.read_text(encoding="utf-8"))["model"]["ops_implementation"]

    with (
        patch("veomni.utils.import_utils.is_torch_npu_available", return_value=True),
        patch("veomni.utils.import_utils.is_torch_mlu_available", return_value=False),
    ):
        ops_config = OpsImplementationConfig(**raw_ops_config)

    assert ops_config.attn_implementation == "sdpa"
    assert ops_config.moe_implementation == "fused_npu"


def test_qwen4_exp_vlm_sft_restores_modality_ids_for_ple():
    model = build_foundation_model(
        config_path=_TOY_CONFIG,
        weights_path=None,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=make_eager_ops_config(),
    )
    model.train()

    captured_ple_ids = []
    ple_embedding = model.model.language_model.layers[0].ple.ple_embedding
    assert list(ple_embedding.ngram_embedding) == [f"shard_{idx}" for idx in range(4)]
    assert all(
        embedding.num_embeddings == ple_embedding.padded_rows_per_shard
        for embedding in ple_embedding.ngram_embedding.values()
    )
    hook = ple_embedding.register_forward_pre_hook(
        lambda _module, args: captured_ple_ids.append(args[0].detach().clone())
    )

    input_ids = torch.tensor([[1, 7, 0, 8, 9, 2]], dtype=torch.long)
    image_mask = torch.tensor([[False, False, True, False, False, False]])
    video_mask = torch.zeros_like(image_mask)
    labels = input_ids.clone().masked_fill(image_mask, -100)
    position_ids = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(1, 3, -1).clone()
    base_loss_function = model.loss_function

    def loss_without_model_metadata(**kwargs):
        assert "qwen4_exp_position_ids_layout" not in kwargs
        return base_loss_function(**kwargs)

    model._loss_function = loss_without_model_metadata

    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=position_ids,
        labels=labels,
        pixel_values=torch.randn(4, 3 * 2 * 4 * 4),
        image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),
        image_mask=image_mask,
        video_mask=video_mask,
        qwen4_exp_position_ids_layout="batch_first",
    )
    hook.remove()

    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.lm_head.weight.grad is not None
    assert captured_ple_ids[0][0, 2].item() == model.config.image_token_id
    assert not hasattr(model, "mtp")


def test_qwen4_exp_ple_keeps_fp32_master_weights_with_bfloat16_model():
    model = build_foundation_model(
        config_path=_TOY_CONFIG,
        weights_path=None,
        torch_dtype="bfloat16",
        init_device="cpu",
        ops_implementation=make_eager_ops_config(),
    )
    ple = model.model.language_model.layers[0].ple

    assert ple.key_proj.weight.dtype == torch.bfloat16
    assert all(embedding.weight.dtype == torch.float32 for embedding in ple.ple_embedding.ngram_embedding.values())

    hidden_states = torch.randn(1, 5, 2 * model.config.text_config.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.arange(5, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = ple(hidden_states, input_ids, past_key_values=None)

    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("batch_size", [3, 4])
def test_qwen4_exp_preserves_canonical_position_ids_and_validates_image_count(batch_size):
    model = build_foundation_model(
        config_path=_TOY_CONFIG,
        weights_path=None,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=make_eager_ops_config(),
    )
    model.eval()

    captured_position_ids = []

    def capture_position_ids(_module, _args, kwargs):
        captured_position_ids.append(kwargs["position_ids"].detach().clone())

    hook = model.model.language_model.register_forward_pre_hook(capture_position_ids, with_kwargs=True)
    input_ids = torch.tensor([[1, 7, 8, 9, 2]] * batch_size, dtype=torch.long)
    canonical_position_ids = torch.arange(3 * batch_size * input_ids.shape[1]).reshape(
        3, batch_size, input_ids.shape[1]
    )
    empty_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=canonical_position_ids,
            image_mask=empty_mask,
            video_mask=empty_mask,
        )

    packed_position_ids = canonical_position_ids.transpose(0, 1).contiguous()
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=packed_position_ids,
            image_mask=empty_mask,
            video_mask=empty_mask,
            qwen4_exp_position_ids_layout="batch_first",
        )
    hook.remove()

    assert torch.equal(captured_position_ids[0], canonical_position_ids)
    assert torch.equal(captured_position_ids[1], canonical_position_ids)

    collated = {}
    model.get_metadata_collate_func()(collated, {})
    assert collated["qwen4_exp_position_ids_layout"] == "batch_first"

    with pytest.raises(ValueError, match="Image features and image placeholder tokens do not match"):
        model(
            input_ids=input_ids[:1],
            attention_mask=torch.ones_like(input_ids[:1]),
            position_ids=canonical_position_ids[:, :1],
            pixel_values=torch.randn(4, 3 * 2 * 4 * 4),
            image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),
            image_mask=empty_mask[:1],
            video_mask=empty_mask[:1],
        )


def _qwen4_exp_ple_parallel_lookup_worker(rank: int, world_size: int, port: int):
    import os

    import torch.distributed as dist

    from veomni.checkpoint.dcp_checkpointer import _apply_extra_parallel_dim
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state
    from veomni.optim.optimizer import _is_extra_parallel_param, build_optimizer

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        ple_size = 2
        parallel_state = init_parallel_state(
            dp_size=world_size,
            dp_shard_size=world_size,
            device_type="cpu",
            extra_parallel_names=("ple", "ep"),
            extra_parallel_sizes=(ple_size, 1),
            extra_parallel_placement_innermost=(False, False),
            name=None,
        )
        model = build_foundation_model(
            config_path=_TOY_CONFIG,
            weights_path=None,
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=make_eager_ops_config(),
        )
        ple_embedding = model.model.language_model.layers[0].ple.ple_embedding
        ple_fsdp_size = world_size // ple_size
        local_rows = ple_embedding.padded_rows_per_shard // ple_size

        full_weights = []
        for shard_idx, embedding in enumerate(ple_embedding.ngram_embedding.values()):
            full_weight = torch.arange(
                ple_embedding.padded_rows_per_shard * embedding.embedding_dim,
                dtype=torch.float32,
            ).view(ple_embedding.padded_rows_per_shard, embedding.embedding_dim)
            full_weight = full_weight + shard_idx * 10_000
            full_weights.append(full_weight)
            embedding.weight = torch.nn.Parameter(full_weight.clone())

        model.get_parallel_plan().apply(model, parallel_state.extra_parallel_fsdp_device_mesh)
        first_weight = ple_embedding.ngram_embedding["shard_0"].weight
        first_weight_fqn = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
        assert tuple(placement.dim for placement in first_weight.placements) == (1, 0)
        assert first_weight.shape == full_weights[0].shape
        assert first_weight.to_local().shape == (local_rows, first_weight.shape[1] // ple_fsdp_size)
        assert _is_extra_parallel_param(first_weight, ("ple", "ep")) == "ple"
        state_dict = {first_weight_fqn: first_weight.detach()}
        for action in ("restore", "drop"):
            processed = _apply_extra_parallel_dim(
                state_dict.copy(),
                model._fqn2spec_info,
                parallel_state,
                action,
                key_match="exact",
            )
            assert processed[first_weight_fqn] is state_dict[first_weight_fqn]

        request_count = rank + 1
        shard_ids = (torch.arange(request_count, dtype=torch.long) + rank) % len(full_weights)
        row_ids = (
            torch.arange(request_count, dtype=torch.long) * (rank + 3) + rank * local_rows
        ) % ple_embedding.padded_rows_per_shard

        output = ple_embedding._distributed_lookup(shard_ids, row_ids)
        expected = torch.stack([full_weights[s][r] for s, r in zip(shard_ids.tolist(), row_ids.tolist())])
        torch.testing.assert_close(output, expected)

        low_precision_output = ple_embedding._distributed_lookup(
            shard_ids,
            row_ids,
            output_dtype=torch.bfloat16,
        )
        assert low_precision_output.dtype == torch.bfloat16
        torch.testing.assert_close(low_precision_output, expected.to(torch.bfloat16))

        output.mul(rank + 1).sum().backward()
        all_requests = [None] * world_size
        dist.all_gather_object(all_requests, (shard_ids.tolist(), row_ids.tolist()))
        row_rank = parallel_state.extra_parallel_rank("ple")
        for shard_idx, embedding in enumerate(ple_embedding.ngram_embedding.values()):
            expected_grad = torch.zeros_like(embedding.weight.to_local())
            for source_rank, (source_shards, source_rows) in enumerate(all_requests):
                for source_shard, source_row in zip(source_shards, source_rows):
                    if source_shard == shard_idx and source_row // local_rows == row_rank:
                        expected_grad[source_row % local_rows] += (source_rank + 1) / world_size
            torch.testing.assert_close(embedding.weight.grad.to_local(), expected_grad)

        model._persistent_extra_parallel_param_ids = {
            id(param)
            for name, param in model.named_parameters()
            if model._fqn2spec_info[name].persistent_fsdp_shard_dim is not None
        }
        optimizer = build_optimizer(model, lr=0.1, weight_decay=0.0, fused=False)
        assert "ple" in optimizer.optimizers_dict

        import importlib

        clip_grad_norm_module = importlib.import_module("veomni.distributed.fsdp2.clip_grad_norm")
        clip_grad_norm_module.get_device_type = lambda: "cpu"

        expected_norm_squared = 0.0
        for shard_idx, full_weight in enumerate(full_weights):
            for row_id in range(ple_embedding.padded_rows_per_shard):
                grad_value = sum(
                    (source_rank + 1) / world_size
                    for source_rank, (source_shards, source_rows) in enumerate(all_requests)
                    for source_shard, source_row in zip(source_shards, source_rows)
                    if source_shard == shard_idx and source_row == row_id
                )
                expected_norm_squared += grad_value**2 * full_weight.shape[1]
        total_norm = clip_grad_norm_module.extra_parallel_fsdp2_clip_grad_norm(model, max_norm=1e9, foreach=False)
        torch.testing.assert_close(total_norm.cpu(), torch.tensor(expected_norm_squared).sqrt())

        before_step = first_weight.to_local().detach().clone()
        optimizer.step()
        assert not torch.equal(first_weight.to_local(), before_step)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


def test_qwen4_exp_ple_parallel_lookup_routes_different_rank_inputs_with_gradients():
    import torch.multiprocessing as mp

    from tests.tools.launch_utils import find_free_port

    world_size = 4
    mp.spawn(
        _qwen4_exp_ple_parallel_lookup_worker,
        args=(world_size, find_free_port()),
        nprocs=world_size,
        join=True,
    )


class _ToyStreamedPle(torch.nn.Module):
    _create_checkpoint_tensor_converter = staticmethod(
        lambda _model: Qwen4ExpCheckpointTensorConverter(split_ngram_parts=2, shard_row_divisor=4)
    )

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.ngram_embedding = torch.nn.ModuleDict(
            {
                "shard_0": torch.nn.Embedding(4, 2),
                "shard_1": torch.nn.Embedding(4, 2),
            }
        )
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def get_parallel_plan(self):
        from torch.distributed._tensor import Shard

        from veomni.distributed.parallel_plan import ParallelPlan

        return ParallelPlan(
            extra_parallel_plan={"ple": {"model.ngram_embedding.shard_*.weight": Shard(0)}},
            extra_parallel_persistent_modules={"ple": {"model.ngram_embedding": 1}},
        )


def _qwen4_exp_ple_stream_load_worker(rank: int, world_size: int, port: int, checkpoint_dir: str):
    import os

    import torch.distributed as dist

    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed import torch_parallelize
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        ple_size = 2
        parallel_state = init_parallel_state(
            dp_size=world_size,
            dp_shard_size=world_size,
            device_type="cpu",
            extra_parallel_names=("ple", "ep"),
            extra_parallel_sizes=(ple_size, 1),
            extra_parallel_placement_innermost=(False, False),
            name=None,
        )
        model = _ToyStreamedPle()
        torch_parallelize.get_device_type = lambda: "cpu"
        model = torch_parallelize.parallelize_model_fsdp2(
            model,
            weights_path=checkpoint_dir,
            mixed_precision=MixedPrecisionConfig(enable=False),
            init_device="meta",
            broadcast_model_weights_from_rank0=False,
            ep_sharded_stream_load=True,
        )

        row_rank = parallel_state.extra_parallel_rank("ple")
        col_rank = parallel_state.extra_parallel_fsdp_device_mesh["ple"].get_local_rank("ple_fsdp")
        padded = (
            torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [0.0, 0.0]]),
            torch.tensor([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [0.0, 0.0]]),
        )
        row_slice = slice(row_rank * 2, (row_rank + 1) * 2)
        col_slice = slice(col_rank, col_rank + 1)
        for shard_idx, expected_full in enumerate(padded):
            weight = model.model.ngram_embedding[f"shard_{shard_idx}"].weight
            assert tuple(placement.dim for placement in weight.placements) == (1, 0)
            assert id(weight) in model._persistent_extra_parallel_param_ids
            torch.testing.assert_close(weight.to_local(), expected_full[row_slice, col_slice])
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


def test_qwen4_exp_stream_loader_reads_only_local_padded_ple_rectangles(tmp_path):
    import torch.multiprocessing as mp
    from safetensors.torch import save_file

    from tests.tools.launch_utils import find_free_port

    save_file(
        {
            "model.ngram_embedding.shard_0.weight": torch.arange(6, dtype=torch.float32).view(3, 2),
            "model.ngram_embedding.shard_1.weight": torch.arange(10, 16, dtype=torch.float32).view(3, 2),
            "mtp.fc_embedding.weight": torch.ones(2, 2),
        },
        tmp_path / "model.safetensors",
    )
    world_size = 4
    mp.spawn(
        _qwen4_exp_ple_stream_load_worker,
        args=(world_size, find_free_port(), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )


def _qwen4_exp_ple_ep_parallelize_worker(rank: int, world_size: int, port: int, checkpoint_dir: str):
    import os

    import torch.distributed as dist
    from safetensors.torch import load_file
    from torch.distributed.tensor import DTensor

    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed import torch_parallelize
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state
    from veomni.optim import build_optimizer

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        parallel_state = init_parallel_state(
            dp_size=world_size,
            dp_shard_size=world_size,
            device_type="cpu",
            extra_parallel_names=("ple", "ep"),
            # Use different factorizations to exercise independent mesh views:
            # PLE is row-only here while EP retains a two-rank FSDP dimension.
            extra_parallel_sizes=(4, 2),
            extra_parallel_placement_innermost=(False, False),
            name=None,
        )
        model = build_foundation_model(
            config_path=checkpoint_dir,
            weights_path=checkpoint_dir,
            torch_dtype="float32",
            init_device="meta",
            ops_implementation=make_eager_ops_config(),
        )
        torch_parallelize.get_device_type = lambda: "cpu"
        torch_parallelize.IS_NPU_AVAILABLE = False
        model = torch_parallelize.parallelize_model_fsdp2(
            model,
            weights_path=checkpoint_dir,
            mixed_precision=MixedPrecisionConfig(enable=False),
            init_device="meta",
            broadcast_model_weights_from_rank0=False,
            ep_sharded_stream_load=True,
            enable_forward_prefetch=False,
        )

        parameters = dict(model.named_parameters())
        ple_name = next(name for name in parameters if ".ple.ple_embedding.ngram_embedding." in name)
        expert_name = next(name for name in parameters if name.endswith("mlp.experts.gate_up_proj"))
        ple_weight = parameters[ple_name]
        expert_weight = parameters[expert_name]

        assert isinstance(ple_weight, DTensor)
        assert tuple(placement.dim for placement in ple_weight.placements) == (1, 0)
        assert model._fqn2spec_info[ple_name].para_name == "ple"
        assert id(ple_weight) in model._persistent_extra_parallel_param_ids

        assert isinstance(expert_weight, DTensor)
        assert model._fqn2spec_info[expert_name].para_name == "ep"
        assert model._fqn2spec_info[expert_name].persistent_fsdp_shard_dim is None
        assert "ep_fsdp" in expert_weight.device_mesh.mesh_dim_names
        assert id(expert_weight) not in model._persistent_extra_parallel_param_ids

        source = load_file(str(Path(checkpoint_dir) / "model.safetensors"))[expert_name]
        local_experts = source.shape[0] // parallel_state.ep_size
        ep_rank = parallel_state.ep_rank
        torch.testing.assert_close(
            expert_weight.full_tensor(),
            source[ep_rank * local_experts : (ep_rank + 1) * local_experts],
        )

        optimizer = build_optimizer(model, lr=0.1, weight_decay=0.0, fused=False)
        assert {"ple", "ep", "non_extra_parallel"}.issubset(optimizer.optimizers_dict)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


def test_qwen4_exp_parallelize_supports_persistent_ple_with_ep(tmp_path):
    import torch.multiprocessing as mp

    from tests.tools.launch_utils import find_free_port

    model = build_foundation_model(
        config_path=_TOY_CONFIG,
        weights_path=None,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=make_eager_ops_config(),
    )
    model.save_pretrained(tmp_path, save_original_format=False)
    del model

    world_size = 4
    mp.spawn(
        _qwen4_exp_ple_ep_parallelize_worker,
        args=(world_size, find_free_port(), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )


def _qwen4_exp_ple_dcp_roundtrip_worker(rank: int, world_size: int, port: int, checkpoint_dir: str):
    import os

    import torch.distributed as dist

    import veomni.checkpoint.dcp_checkpointer as dcp_checkpointer
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state
    from veomni.optim.optimizer import build_optimizer

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        parallel_state = init_parallel_state(
            dp_size=world_size,
            dp_shard_size=world_size,
            device_type="cpu",
            extra_parallel_names=("ple",),
            extra_parallel_sizes=(2,),
            extra_parallel_placement_innermost=(False,),
            name=None,
        )
        model = _ToyStreamedPle()
        for shard_idx, embedding in enumerate(model.model.ngram_embedding.values()):
            full_weight = torch.arange(8, dtype=torch.float32).view(4, 2) + shard_idx * 10
            embedding.weight = torch.nn.Parameter(full_weight)
        model.get_parallel_plan().apply(model, parallel_state.extra_parallel_fsdp_device_mesh)

        optimizer = build_optimizer(model, lr=0.1, weight_decay=0.0, fused=False)
        assert set(optimizer.optimizers_dict) == {"ple"}
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, rank + 1.0)
        optimizer.step()
        optimizer.zero_grad()

        parameters = dict(model.named_parameters())
        expected_parameters = {name: parameter.to_local().clone() for name, parameter in parameters.items()}
        ple_optimizer = optimizer.optimizers_dict["ple"]
        expected_optimizer_state = {
            name: {
                state_name: state_value.to_local().clone() if hasattr(state_value, "to_local") else state_value.clone()
                for state_name, state_value in ple_optimizer.state[parameter].items()
            }
            for name, parameter in parameters.items()
        }

        # The CPU test environment may expose an unusable accelerator runtime;
        # DCP itself is device agnostic, so keep post-save cache cleanup a no-op.
        dcp_checkpointer.empty_cache = lambda: None
        dcp_checkpointer.synchronize = lambda: None
        DistributedCheckpointer.save(
            checkpoint_dir,
            {"model": model, "optimizer": optimizer},
            parallel_state=parallel_state,
        )

        with torch.no_grad():
            for parameter in parameters.values():
                parameter.to_local().zero_()
            for state in ple_optimizer.state.values():
                for state_value in state.values():
                    state_value.zero_()

        DistributedCheckpointer.load(
            checkpoint_dir,
            {"model": model, "optimizer": optimizer},
            parallel_state=parallel_state,
        )

        for name, parameter in parameters.items():
            assert tuple(placement.dim for placement in parameter.placements) == (1, 0)
            torch.testing.assert_close(parameter.to_local(), expected_parameters[name])
            for state_name, expected_state in expected_optimizer_state[name].items():
                state_value = ple_optimizer.state[parameter][state_name]
                local_state = state_value.to_local() if hasattr(state_value, "to_local") else state_value
                torch.testing.assert_close(local_state, expected_state)

        before_step = next(iter(parameters.values())).to_local().clone()
        for parameter in parameters.values():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert not torch.equal(next(iter(parameters.values())).to_local(), before_step)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


def test_qwen4_exp_ple_dcp_roundtrip_preserves_model_and_optimizer_state(tmp_path):
    import torch.multiprocessing as mp

    from tests.tools.launch_utils import find_free_port

    world_size = 4
    mp.spawn(
        _qwen4_exp_ple_dcp_roundtrip_worker,
        args=(world_size, find_free_port(), str(tmp_path / "dcp")),
        nprocs=world_size,
        join=True,
    )
