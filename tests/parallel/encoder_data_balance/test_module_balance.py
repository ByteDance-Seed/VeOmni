"""Shared scheduling and singleton transport tests (CPU, no model import)."""

import os
import subprocess
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from veomni.utils.data_balance.module_balance import ModuleDataBalancer, greedy_cost_assignment
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


@pytest.mark.parametrize("costs,size", [([], 4), ([9], 4), ([0, 0], 4), ([4, 4, 4, 4], 2), ([100, 9, 4, 1], 2)])
def test_cost_assignment(costs, size):
    assigned = greedy_cost_assignment(costs, size)
    assert len(assigned) == size
    assert sorted(i for bucket in assigned for i in bucket) == list(range(len(costs)))
    assert assigned == greedy_cost_assignment(costs, size)


@pytest.mark.parametrize("costs,size", [([float("nan")], 2), ([-1], 2), ([float("inf")], 2), ([], 0)])
def test_invalid_cost(costs, size):
    with pytest.raises(ValueError):
        greedy_cost_assignment(costs, size)


@pytest.mark.parametrize("lengths", [[], [1], [2, 5, 1]])
def test_singleton_roundtrip_and_gradient(lengths):
    original = torch.arange(sum(lengths) * 3, dtype=torch.float64).reshape(-1, 3).requires_grad_()
    metadata = torch.arange(len(lengths)).reshape(-1, 1)
    balanced, plan = ModuleDataBalancer(None, lambda splits: [n**2 for n in splits["pixels"]]).balance(
        {"pixels": original, "grid": metadata},
        {"pixels": lengths, "grid": [1] * len(lengths)},
        output_lengths=lengths,
    )
    recovered = plan.restore(balanced["pixels"])
    assert torch.equal(recovered, original)
    assert torch.equal(balanced["grid"], metadata)
    recovered.sum().backward()
    assert torch.equal(original.grad, torch.ones_like(original))


def test_invocation_plans_do_not_overwrite():
    balancer = ModuleDataBalancer(None)
    x, first = balancer.balance({"x": torch.ones(3, 2)}, {"x": [3]}, [1])
    y, second = balancer.balance({"x": torch.ones(7, 2)}, {"x": [7]}, [2])
    assert first.restore(x["x"][:1]).shape == (1, 2)
    assert second.restore(y["x"][:2]).shape == (2, 2)


def test_bad_splits():
    with pytest.raises(ValueError):
        ModuleDataBalancer(None).balance({"x": torch.ones(3, 2)}, {"x": [2]}, [1])


@pytest.mark.parametrize("invalid", ["fields", "input", "output", "scalar", "overflow"])
def test_invalid_metadata_is_shared_before_raising(monkeypatch, invalid):
    group = object()
    gathered_calls = []
    monkeypatch.setattr(dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(dist, "get_rank", lambda _: 0)

    def gather(result, local, group):
        gathered_calls.append(local)
        result[:] = [local, local]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    tensors, lengths, outputs = {"x": torch.ones(1, 2)}, {"x": [1]}, [1]
    if invalid == "fields":
        lengths = {"other": [1]}
    elif invalid == "input":
        lengths = {"x": [None]}
    elif invalid == "output":
        outputs = [None]
    elif invalid == "scalar":
        tensors = {"x": torch.tensor(1.0)}
    else:
        lengths = {"x": [float("inf")]}
    with pytest.raises(ValueError, match="metadata"):
        ModuleDataBalancer(group).balance(tensors, lengths, outputs)
    assert len(gathered_calls) == 1
    assert gathered_calls[0][4] is not None


@pytest.mark.parametrize("invalid", ["metric", "missing_metric", "missing_input", "missing_output", "missing_tensor"])
def test_mixin_metadata_errors_reach_shared_validation(monkeypatch, invalid):
    from types import SimpleNamespace

    import veomni.models.seed_omni.mixins.data_balance_mixin as balance

    class Module(balance.DataBalanceMixin):
        training = True
        config = SimpleNamespace()
        data_balance_specs = {
            "encode": balance.DataBalanceSpec(
                fields=(balance.ItemTensorField("x", "input"),),
                output_names=("out",),
                output_lengths_key="output",
                cost_lengths_key="metric",
                structure=balance.ModuleStructure(),
            )
        }

    group, calls = object(), []
    monkeypatch.setattr(balance, "get_parallel_state", lambda: SimpleNamespace(dp_size=2, sp_size=1, dp_group=group))
    monkeypatch.setattr(dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(dist, "get_rank", lambda _: 0)

    def gather(result, local, group):
        calls.append(local)
        result[:] = [local, local]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    tensors, metadata = {"x": torch.ones(1, 2)}, {"input": [1], "output": [1], "metric": [1]}
    if invalid == "metric":
        metadata["metric"] = [None]
    elif invalid == "missing_tensor":
        tensors = {}
    else:
        del metadata[invalid.removeprefix("missing_")]
    module = Module()
    with module.data_balance_scope(), pytest.raises(ValueError, match="metadata"):
        module.data_balance_pre("encode", tensors, metadata)
    assert len(calls) == 1 and calls[0][4] is not None


@pytest.mark.parametrize(
    "global_attn,patchify,items,dp,sp,override,expected,balance",
    [
        (True, True, True, 2, 2, None, "sp_slice", True),
        (True, True, True, 2, 2, "off", "sp_slice", False),
        (True, True, True, 1, 4, None, "sp_slice", False),
        (True, True, True, 4, 1, None, "dp_balance", True),
        (True, False, True, 2, 2, None, "dp_balance", True),
        (False, False, True, 1, 4, None, "replicate", False),
        (False, False, False, 2, 2, None, "replicate", False),
    ],
)
def test_structural_strategy(global_attn, patchify, items, dp, sp, override, expected, balance):
    from veomni.models.seed_omni.mixins.data_balance_mixin import ModuleStructure, resolve_balance_strategy

    plan = resolve_balance_strategy(
        ModuleStructure(global_attn, patchify, items), sp_size=sp, dp_size=dp, override=override
    )
    assert (plan.strategy, plan.balance_dp) == (expected, balance)


def test_invalid_strategy_override():
    from veomni.models.seed_omni.mixins.data_balance_mixin import ModuleStructure, resolve_balance_strategy

    with pytest.raises(ValueError, match="override"):
        resolve_balance_strategy(ModuleStructure(), sp_size=2, dp_size=2, override="conv_slice")


@pytest.mark.parametrize("exponent", [-1, float("nan"), float("inf")])
def test_invalid_spec_exponent(exponent):
    from veomni.models.seed_omni.mixins.data_balance_mixin import DataBalanceSpec, ItemTensorField, ModuleStructure

    with pytest.raises(ValueError, match="exponent"):
        DataBalanceSpec(
            fields=(ItemTensorField("x", "input"),),
            output_names=("out",),
            output_lengths_key="output",
            cost_lengths_key="metric",
            structure=ModuleStructure(),
            cost_exponent=exponent,
        )


def test_mixin_scope_cleanup_and_reentrancy():
    from veomni.models.seed_omni.mixins.data_balance_mixin import DataBalanceMixin

    module = DataBalanceMixin()
    with pytest.raises(RuntimeError, match="Reentrant"):
        with module.data_balance_scope(), module.data_balance_scope():
            pass
    assert module._data_balance_active is False
    assert module._data_balance_plans == {}


def test_declared_output_inverse_and_cost_metric_are_independent(monkeypatch):
    from types import SimpleNamespace

    import veomni.models.seed_omni.mixins.data_balance_mixin as balance

    class Module(balance.DataBalanceMixin):
        training = True
        config = SimpleNamespace()
        data_balance_specs = {
            "encode": balance.DataBalanceSpec(
                fields=(balance.ItemTensorField("x", "input"),),
                output_names=("out", "layers"),
                output_lengths_key="output",
                cost_lengths_key="metric",
                structure=balance.ModuleStructure(),
                cost_exponent=1,
            )
        }

    class FakeBalancer:
        def __init__(self, group):
            assert group is group_marker

        def balance(self, tensors, inputs, outputs, costs):
            assert inputs == {"x": [3]}
            assert outputs == [1]
            assert costs == [7]
            return tensors, SimpleNamespace(restore=lambda tensor: tensor + 1)

    group_marker = object()
    monkeypatch.setattr(balance, "ModuleDataBalancer", FakeBalancer)
    monkeypatch.setattr(
        balance, "get_parallel_state", lambda: SimpleNamespace(dp_size=2, sp_size=1, dp_group=group_marker)
    )
    module = Module()
    x = torch.zeros(3, 2)
    assert module.data_balance_pre("encode", {"x": x}, {}) == ({"x": x}, False)
    with module.data_balance_scope():
        _, moved = module.data_balance_pre("encode", {"x": x}, {"input": [3], "output": [1], "metric": [7]})
        assert moved
        out = module.data_balance_post("encode", {"out": x[:1], "layers": [x[:1]]})
        assert torch.equal(out["out"], torch.ones(1, 2))
        assert torch.equal(out["layers"][0], torch.ones(1, 2))


def test_qwen_video_cost_is_framewise_not_clip_squared():
    from types import SimpleNamespace

    from veomni.models.seed_omni.mixins import MetricMeterMixin
    from veomni.models.seed_omni.modules.qwen3vl.vision.accelerated.accelerated import BalanceMixin

    class Vision(BalanceMixin, MetricMeterMixin):
        config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))

        def data_balance_pre(self, method, tensors, metadata):
            assert metadata["merged_lengths"] == [4]
            assert metadata["attention_costs"] == [8]
            assert self.data_balance_specs[method].cost_exponent == 1
            return tensors, False

    vision = Vision()
    vision._balance_vision_inputs("forward", torch.ones(16, 3), torch.tensor([[2, 2, 4]]))
    assert vision.metric_meter_token_lengths("forward", {}) == [2, 2]


def distributed_worker():
    """One real DP/SP mesh, including empty owners, destinations, and global batch."""
    device_type = get_device_type()
    rank = int(os.environ["LOCAL_RANK"])
    if device_type != "cpu":
        get_torch_device().set_device(rank)
    device = torch.device(device_type if device_type == "cpu" else f"{device_type}:{rank}")
    dist.init_process_group(
        "gloo" if device_type == "cpu" else get_dist_comm_backend(),
        timeout=timedelta(seconds=120),
    )
    world = dist.get_world_size()
    for sp_size in (1, 2):
        dp_group = None
        for lane in range(sp_size):
            group = dist.new_group(list(range(lane, world, sp_size)))
            if rank % sp_size == lane:
                dp_group = group
        owner = rank // sp_size
        for scenario in ("uneven", "one_item", "all_empty"):
            lengths = (
                ([4, 16, 8, 12, 4, 8] if owner == 0 else [4] if owner == 1 else [])
                if scenario == "uneven"
                else ([4] if owner == 0 and scenario == "one_item" else [])
            )
            for merge in (1, 2):
                x = (
                    torch.arange(sum(lengths) * 3, device=device, dtype=torch.float64).reshape(-1, 3) + owner * 128
                ).requires_grad_()
                grids = torch.tensor(lengths, device=device, dtype=torch.long).reshape(-1, 1)
                balanced, plan = ModuleDataBalancer(dp_group).balance(
                    {"pixels": x, "grid": grids},
                    {"pixels": lengths, "grid": [1] * len(lengths)},
                    [n // merge for n in lengths],
                    costs=[n**2 for n in lengths],
                )
                expected_grid = torch.tensor([i.input_lengths[1] for i in plan.assigned], device=device).reshape(-1, 1)
                assert torch.equal(balanced["grid"], expected_grid)
                embeddings = balanced["pixels"].reshape(-1, merge, 3).sum(1)
                restored = plan.restore(embeddings)
                deepstack = plan.restore(embeddings * 2)
                expected = x.reshape(-1, merge, 3).sum(1)
                assert torch.equal(restored, expected)
                assert torch.equal(deepstack, expected * 2)
                (restored.sum() + deepstack.sum()).backward()
                assert torch.equal(x.grad, torch.full_like(x, 3))
                owned = torch.tensor(sum(lengths), device=device, dtype=torch.long)
                assigned = torch.tensor(
                    sum(i.input_lengths[1] for i in plan.assigned), device=device, dtype=torch.long
                )
                dist.all_reduce(owned, group=dp_group)
                dist.all_reduce(assigned, group=dp_group)
                assert torch.equal(owned, assigned)
        if rank == 0:
            print(
                f"DP{world // sp_size}/SP{sp_size}: roundtrip, deepstack, gradients, empty ranks, token totals passed",
                flush=True,
            )
        # One bad owner must fail *every* rank before a payload collective.
        with pytest.raises(ValueError, match="metadata"):
            ModuleDataBalancer(dp_group).balance(
                {"x": torch.ones(1, 2, device=device)}, {"x": [1]}, [1], costs=[-1 if owner == 0 else 1]
            )
        with pytest.raises(ValueError, match="schema"):
            ModuleDataBalancer(dp_group).balance(
                {"x": torch.ones(1, 3 if owner == 0 else 2, device=device)}, {"x": [1]}, [1], costs=[1]
            )
        with pytest.raises(ValueError, match="schema"):
            ModuleDataBalancer(dp_group).balance(
                {"x": torch.ones(1, 2, device=device, requires_grad=owner == 0)}, {"x": [1]}, [1], costs=[1]
            )
        for invalid in ("fields", "input", "output", "scalar", "overflow"):
            tensors, lengths, outputs = {"x": torch.ones(1, 2, device=device)}, {"x": [1]}, [1]
            if owner == 0:
                if invalid == "fields":
                    lengths = {"other": [1]}
                elif invalid == "input":
                    lengths = {"x": [None]}
                elif invalid == "output":
                    outputs = [None]
                elif invalid == "scalar":
                    tensors = {"x": torch.tensor(1.0, device=device)}
                else:
                    lengths = {"x": [float("inf")]}
            with pytest.raises(ValueError, match="metadata"):
                ModuleDataBalancer(dp_group).balance(tensors, lengths, outputs)
        from types import SimpleNamespace

        import veomni.models.seed_omni.mixins.data_balance_mixin as balance

        class DeclaredModule(balance.DataBalanceMixin):
            training = True
            config = SimpleNamespace()
            data_balance_specs = {
                "encode": balance.DataBalanceSpec(
                    fields=(balance.ItemTensorField("x", "input"),),
                    output_names=("out",),
                    output_lengths_key="output",
                    cost_lengths_key="metric",
                    structure=balance.ModuleStructure(),
                )
            }

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                balance,
                "get_parallel_state",
                lambda sp_size=sp_size, dp_group=dp_group: SimpleNamespace(
                    dp_size=world // sp_size, sp_size=sp_size, dp_group=dp_group
                ),
            )
            for invalid in ("metric", "missing_metric", "missing_input", "missing_output", "missing_tensor"):
                tensors = {"x": torch.ones(1, 2, device=device)}
                metadata = {"input": [1], "output": [1], "metric": [1]}
                if owner == 0:
                    if invalid == "metric":
                        metadata["metric"] = [None]
                    elif invalid == "missing_tensor":
                        tensors = {}
                    else:
                        del metadata[invalid.removeprefix("missing_")]
                module = DeclaredModule()
                with module.data_balance_scope(), pytest.raises(ValueError, match="metadata"):
                    module.data_balance_pre("encode", tensors, metadata)
    dist.destroy_process_group()


def test_distributed_roundtrip_and_backward():
    count = 0 if get_device_type() == "cpu" else get_torch_device().device_count()
    world = 8 if count >= 8 else 4
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world}",
            __file__,
            "--worker",
        ],
        check=True,
        timeout=240,
    )


def qwen_vision_worker():
    """Real FSDP2 toy ViT on/off values, globally weighted loss and token meter.

    This is a module training integration gate, not a full text+vision launcher
    e2e. Both ordinary carriers and the packed scatter path must retain dummy
    and deepstack gradient anchors on asymmetric ranks.
    """
    import torch.nn.functional as F
    from torch.distributed.fsdp import fully_shard

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
    from veomni.distributed.sequence_parallel import reduce_sequence_parallel_loss
    from veomni.models.seed_omni.accelerator.executor import execute_train_node
    from veomni.models.seed_omni.graphs.base import NodeDef
    from veomni.models.seed_omni.modules.qwen3vl.vision.accelerated.accelerated import Qwen3VLVisionEncoderAccelerated
    from veomni.models.seed_omni.modules.qwen3vl.vision.configuration import Qwen3VLVisionEncoderConfig
    from veomni.models.seed_omni.modules.qwen3vl.vision.processing import _OMNI_GRID, _SOURCE
    from veomni.models.seed_omni.utils.conversation import ConversationItem

    rank = int(os.environ["LOCAL_RANK"])
    get_torch_device().set_device(rank)
    device = torch.device(f"{get_device_type()}:{rank}")
    dist.init_process_group(get_dist_comm_backend(), timeout=timedelta(seconds=180))
    world = dist.get_world_size()
    for sp in (1, 2, 4):
        clear_parallel_state()
        state_name = f"vision-sp{sp}"
        ps = _init_parallel_state(
            dp_size=world // sp, dp_shard_size=world // sp, dp_mode="fsdp2", ulysses_size=sp, name=state_name
        )
        with use_parallel_state(state_name):
            torch.manual_seed(42)
            cfg = Qwen3VLVisionEncoderConfig(
                vision_config={
                    "hidden_size": 32,
                    "num_heads": 4,
                    "depth": 2,
                    "intermediate_size": 64,
                    "out_hidden_size": 32,
                    "deepstack_visual_indexes": [1],
                    "spatial_merge_size": 2,
                    "patch_size": 2,
                    "temporal_patch_size": 2,
                    "in_channels": 3,
                    "num_position_embeddings": 16,
                }
            )
            cfg._attn_implementation = "eager" if sp == 1 else "veomni_flash_attention_2_with_sp"
            model = Qwen3VLVisionEncoderAccelerated(cfg).to(device=device, dtype=torch.bfloat16).train()
            fully_shard(model, mesh=ps.fsdp_mesh)
            owner = rank // sp
            grids = [[1, 4, 4], [1, 6, 4], [2, 2, 4]] if owner == 0 else [[1, 2, 4]] if owner == 1 else [[1, 4, 4]]
            real_count = len(grids) if owner < 2 else 0
            for method in ("forward", "pack_encode"):
                results = []
                for override in ("off", "auto"):
                    model.config.data_balance_override = override
                    model.zero_grad(set_to_none=True)
                    generator = torch.Generator().manual_seed(100 + owner)
                    pixels = [torch.randn(t * h * w, 24, generator=generator) for t, h, w in grids]
                    real_tokens = sum(t * h * w // 4 for t, h, w in grids[:real_count])
                    if method == "forward":
                        items = [
                            ConversationItem(
                                type="video" if grid[0] > 1 else "image",
                                value=pixel,
                                role="user" if real_count else "dummy",
                                source=_SOURCE,
                                meta={_OMNI_GRID: grid},
                            )
                            for pixel, grid in zip(pixels, grids)
                        ]
                        batch = {"conversation_list": [items]}
                    else:
                        length = real_tokens + 4
                        mask = torch.arange(length, device=device) < real_tokens
                        batch = {
                            "pixel_values": torch.cat(pixels),
                            "image_grid_thw": torch.tensor(grids),
                            "packed_features": torch.zeros(1, length, 32, device=device, dtype=torch.bfloat16),
                            "visual_pos_mask": mask,
                            "visual_num_real": real_count,
                        }
                    execute_train_node(
                        model,
                        NodeDef(name=f"vision.{method}", module="vision", method=method),
                        batch,
                        scope_fn=lambda _, name=state_name: use_parallel_state(name),
                    )
                    local_sum = torch.zeros((), device=device)
                    snapshots = []
                    if method == "forward":
                        for item in batch["conversation_list"][0]:
                            snapshots.append(item.value.detach().clone())
                            if item.role == "dummy":
                                local_sum = local_sum + item.value.float().sum() * 0
                            else:
                                logits = item.value.float()[:, :8]
                                local_sum = local_sum + F.cross_entropy(
                                    logits,
                                    torch.zeros(logits.size(0), device=device, dtype=torch.long),
                                    reduction="sum",
                                )
                            for layer in item.meta["deepstack"]:
                                snapshots.append(layer.detach().clone())
                                local_sum = local_sum + layer.float().sum() * 0
                    else:
                        features = batch["packed_features"].float()
                        snapshots.append(features.detach().clone())
                        logits = features[0, :real_tokens, :8]
                        local_sum = (
                            local_sum
                            + F.cross_entropy(
                                logits, torch.zeros(real_tokens, device=device, dtype=torch.long), reduction="sum"
                            )
                            if real_tokens
                            else local_sum + features.sum() * 0
                        )
                        local_sum = local_sum + batch["deepstack_visual_embeds"].float().sum() * 0
                    count = torch.tensor(real_tokens, device=device)
                    loss = reduce_sequence_parallel_loss(
                        local_sum / count.clamp_min(1), count.clone(), group=ps.fsdp_group
                    )
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        assert parameter.grad is not None, f"Missing gradient anchor: {name}"
                        if parameter.grad is not None:
                            assert torch.isfinite(parameter.grad.to_local()).all(), name
                    _, lengths = model.metric_meter_collect()
                    consume_tokens = torch.tensor(sum(lengths), device=device)
                    if ps.dp_size > 1:
                        dist.all_reduce(consume_tokens, group=ps.dp_group)
                    assert consume_tokens.item() == {4: 24, 2: 16, 1: 14}[ps.dp_size], (
                        "Full pre-slice tokens miscounted"
                    )
                    results.append((loss.detach().clone(), consume_tokens, snapshots))
                off, on = results
                assert torch.equal(off[1], on[1]), f"{method} SP{sp} consume_tokens changed"
                assert torch.equal(off[0], on[0]), (
                    f"{method} SP{sp} loss differs: off={off[0].item()} on={on[0].item()}"
                )
                for index, (old, new) in enumerate(zip(off[2], on[2], strict=True)):
                    assert torch.equal(old, new), (
                        f"{method} SP{sp} output {index} differs by {(old - new).abs().max().item()}"
                    )
                if rank == 0:
                    print(
                        f"Qwen3-VL {method} DP{world // sp}/SP{sp}: exact on/off loss, tokens and outputs passed",
                        flush=True,
                    )
            del model
    clear_parallel_state()
    dist.destroy_process_group()


def test_qwen_vision_fsdp_on_off_parity():
    if get_device_type() == "cpu" or get_torch_device().device_count() < 4:
        pytest.skip("Actual toy vision FSDP2/SP parity requires four accelerators.")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            __file__,
            "--model-worker",
        ],
        check=True,
        timeout=480,
    )


if __name__ == "__main__" and "--worker" in sys.argv:
    distributed_worker()
elif __name__ == "__main__" and "--model-worker" in sys.argv:
    qwen_vision_worker()
