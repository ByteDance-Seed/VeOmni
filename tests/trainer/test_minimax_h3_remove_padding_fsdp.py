"""Two-GPU native H3 regression through the ordinary DiT condition/model interface.

No encoders or downloaded checkpoints are used. The reference runs each sample
through the legacy native H3 forward, averages samples (not tokens), and explicitly
averages FP32 gradients over DP. FSDP2 must produce the same SGD update through
the wrapped root and the production trainer loops.
"""

import importlib
import importlib.util
import os
from collections.abc import Mapping
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist

from tests.tools.launch_utils import torchrun
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device, is_nccl_backend


_WORLD_SIZE = 2
_MICROBATCHES = 2
_SAMPLES_PER_MICROBATCH = 2
_LOCAL_SAMPLES = _MICROBATCHES * _SAMPLES_PER_MICROBATCH
_LR = 0.05
# BF16 GEMMs see different row counts in packed and single-sample execution.
# Check updates, not just final weights: unchanged weights must not pass because
# their initial magnitude dwarfs the optimizer update.
_BF16_RTOL = 5e-2
_BF16_ATOL = 2e-2
_UPDATE_REL_L2 = 5e-2


def _tensor_leaves(value, path=()):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _tensor_leaves(item, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensor_leaves(item, (*path, index))


def _full_cpu(tensor):
    from torch.distributed.tensor import DTensor

    tensor = tensor.detach()
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.float().cpu().clone()


def _vector(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _assert_relative_l2(actual, expected, *, label):
    assert torch.isfinite(actual).all(), label
    assert torch.isfinite(expected).all(), label
    expected_norm = expected.norm().item()
    assert expected_norm > 1e-8, f"Vacuous {label}: reference is zero"
    relative_error = (actual - expected).norm().item() / expected_norm
    print(
        f"rank={dist.get_rank()} {label}: relative_l2={relative_error:.6g}, "
        f"tolerance={_UPDATE_REL_L2}, reference_norm={expected_norm:.6g}",
        flush=True,
    )
    assert relative_error < _UPDATE_REL_L2, f"{label}: relative L2 error {relative_error}"


def _raw_microbatches(rank, *, checkpointing, task):
    from tests.models.test_minimax_h3_remove_padding import raw_sample
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.packed_sequence import (
        build_packed_fl2va,
        build_packed_ref2va,
    )
    from veomni.trainer.dit_trainer import DiTDataCollator

    generator = torch.Generator().manual_seed(3000 + rank)
    batches = []
    for micro_step in range(_MICROBATCHES):
        raws = []
        for index in range(_SAMPLES_PER_MICROBATCH):
            text_len = 3 + 2 * micro_step + 4 * index + rank
            raw = raw_sample(text_len=text_len, task=task)
            # H3 requires fixed target geometry within a microbatch. Vary it
            # between microbatches to distinguish sample-count accumulation
            # from a global token/element-weighted objective.
            video_t = 2 + micro_step
            audio_t = 3 + micro_step
            offset = 0.75 * rank + 0.2 * (micro_step + index)
            raw["input_latents"] = (torch.randn(1, 24, video_t, 4, 6, generator=generator) + offset).bfloat16()
            raw["audio_input_latents"] = (torch.randn(2, 32, audio_t, generator=generator) - offset).bfloat16()
            raw["prompt_embeds"] = torch.randn(text_len, 32, generator=generator).bfloat16()
            if task == "fl2va":
                raw["packed"] = build_packed_fl2va(text_len, video_t, 4, 6, audio_t, keyframe_indices=[0])
                anchor_key = "keyframe_cond_anchor"
            else:
                refs = [{"kind": "image", "latent_t": 1, "latent_h": 2, "latent_w": 4}]
                if index:
                    refs.append({"kind": "video", "latent_t": 2, "latent_h": 4, "latent_w": 6})
                raw["packed"] = build_packed_ref2va(text_len, video_t, 4, 6, audio_t, refs)
                anchor_key = "ref_visual_anchor"
            raw[anchor_key] = torch.randn(raw["packed"]["cond_rows"], 96, generator=generator).bfloat16()
            raw["use_gradient_checkpointing"] = checkpointing
            raws.append(raw)
        batches.append(DiTDataCollator()(raws))
    return batches


class _StepRecorder:
    def __init__(self):
        self.begin_count = 0
        self.end_metrics = []

    def on_step_begin(self, state, *, micro_batches, **kwargs):
        assert state.global_step == 1
        assert len(micro_batches) == _MICROBATCHES
        self.begin_count += 1

    def on_step_end(self, state, **metrics):
        assert state.global_step == 1
        self.end_metrics.append(metrics)


def _run_fsdp_regression(*, step_driver, checkpointing, attention, task):
    # Keep model/helper imports behind the hardware gate, including under spawn.
    from accelerate import init_empty_weights
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor
    from torch.func import functional_call

    from tests.models.test_minimax_h3_remove_padding import condition_model, tiny_model
    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.parallel_state import (
        clear_parallel_state,
        init_parallel_state_from_config,
        use_parallel_state,
    )
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import core as h3_core
    from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
        MiniMaxH3DiTOutput,
    )
    from veomni.trainer.base import BaseTrainer
    from veomni.trainer.callbacks.base import TrainerState
    from veomni.trainer.dit_trainer import (
        DiTDataArguments,
        DiTModelArguments,
        DiTModelRuntime,
        DiTTrainer,
        DiTTrainingArguments,
        VeOmniDiTArguments,
    )

    assert IS_CUDA_AVAILABLE
    assert dist.get_world_size() == _WORLD_SIZE
    assert is_nccl_backend(dist.get_backend()), "Never substitute CPU/Gloo for this FSDP2 regression"
    rank = dist.get_rank()
    # The repository spawn helper sets RANK/WORLD_SIZE and the device, but not
    # LOCAL_RANK. AcceleratorConfig derives its real mesh from these variables.
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(_WORLD_SIZE)
    device = torch.device(get_device_type(), rank)
    assert get_torch_device().current_device() == rank
    torch.set_num_threads(1)

    args = VeOmniDiTArguments(
        model=DiTModelArguments(config_path="unused-native-h3"),
        data=DiTDataArguments(train_path="unused-offline-latents", log_sample=False),
        train=DiTTrainingArguments(
            training_task="offline_training",
            dyn_bsz=False,
            micro_batch_size=_SAMPLES_PER_MICROBATCH,
            global_batch_size=_WORLD_SIZE * _MICROBATCHES * _SAMPLES_PER_MICROBATCH,
        ),
    )
    args.model.accelerator.gradient_checkpointing.enable = checkpointing
    args.model.accelerator.fsdp_config.mixed_precision = MixedPrecisionConfig(
        enable=True, param_dtype="bfloat16", reduce_dtype="float32", cast_forward_inputs=False
    )
    args.model.optimizer.max_grad_norm = 1e6  # Exercise production clipping without clipping the oracle.
    assert args.train.gradient_accumulation_steps == _MICROBATCHES
    state = init_parallel_state_from_config(args.model.accelerator, name="base")
    assert state.dp_size == state.fsdp_size == _WORLD_SIZE
    assert state.fsdp_mesh.size() == _WORLD_SIZE
    assert not state.sp_enabled

    handles = []
    try:
        with use_parallel_state("base"):
            torch.manual_seed(202601)
            # Keep the oracle on unchanged legacy segmented SDPA, independently
            # of the new packed/varlen dispatch (including real FA2/FA3 cases).
            h3_core.ATTENTION_IMPLEMENTATION = "torch"
            reference = tiny_model().float().to(device).train()
            initial = {name: value.detach().cpu().clone() for name, value in reference.state_dict().items()}
            assert all(torch.isfinite(value).all() for value in initial.values())
            with init_empty_weights():
                model = type(reference)(deepcopy(reference.config))
            model._configure_packed_attention(attention)
            model = build_parallelize_model(
                model,
                init_device="meta",
                mixed_precision=args.model.accelerator.fsdp_config.mixed_precision,
                enable_gradient_checkpointing=checkpointing,
                # Materialize via the real resume path, then restore this tiny
                # in-memory state with the public distributed state-dict API.
                # This avoids both downloads and a test-patched weight loader.
                should_skip_hf_weight_load=True,
            )
            # The distributed state-dict API may replace mapping values with DTensors.
            set_model_state_dict(model, dict(initial), options=StateDictOptions(full_state_dict=True, strict=True))
            model.train()
            assert isinstance(model, FSDPModule)
            fsdp_modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
            assert len(fsdp_modules) > 1, "Must shard native H3 blocks as well as the root"
            assert all(isinstance(param, DTensor) and param.dtype == torch.float32 for param in model.parameters())
            for name, param in model.named_parameters():
                torch.testing.assert_close(_full_cpu(param), initial[name], rtol=0, atol=0)

            runtime = DiTModelRuntime.__new__(DiTModelRuntime)
            runtime.args = args.model
            runtime.model_name = "base"
            runtime.train_args = args.train
            runtime.model = model
            runtime.condition_model = condition_model()
            runtime.condition_model.requires_grad_(False)
            runtime.optimizer = torch.optim.SGD(model.parameters(), lr=_LR, foreach=False)
            runtime.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(runtime.optimizer, lambda _: 1.0)
            scheduler_epoch = runtime.lr_scheduler.last_epoch

            base = BaseTrainer.__new__(BaseTrainer)
            base.args, base.model, base.device = args, runtime, device
            base.state = TrainerState()
            base.LOG_SAMPLE = False
            base.num_micro_batches = _MICROBATCHES
            base._build_training_context(runtime)
            recorder = _StepRecorder()
            base._callbacks = [recorder]
            trainer = DiTTrainer.__new__(DiTTrainer)
            trainer.base = base
            trainer.training_task = "offline_training"

            batches = _raw_microbatches(rank, checkpointing=checkpointing, task=task)
            # Replay the production condition RNG, not hand-authored/noiseless
            # samples. Both microbatches must consume distinct noise/timesteps.
            noise_seed = 7000 + rank
            torch.manual_seed(noise_seed)
            with torch.no_grad():
                prepared = [
                    trainer.condition_model.process_condition(**trainer.preforward(batch)) for batch in batches
                ]
            for columns in prepared:
                assert all(len(values) == _SAMPLES_PER_MICROBATCH for values in columns.values())
                for index in range(_SAMPLES_PER_MICROBATCH):
                    inputs = {key: values[index] for key, values in columns.items()}
                    assert inputs["x"].dtype == inputs["audio_x"].dtype == torch.bfloat16
                    assert inputs["prompt_embeds"].dtype == torch.bfloat16
                    assert inputs["unique_timesteps"].dtype == torch.float32
                    assert inputs["img_position_ids"].dtype == torch.float64
                    assert inputs["packed_seq_params"]["cu_seqlens_q"].dtype == torch.int32
                    assert inputs["img_pos_info"]["position_ids"].dtype == torch.int64

            root_calls = []
            outputs_seen = []
            bf16_calls = {id(module): 0 for module in fsdp_modules}

            def check_bf16_parameters(module, positional, keywords):
                # Registered after fully_shard: its pre-hook has unsharded and
                # cast this group's parameters; descendant groups may still be
                # FP32 DTensors until their own pre-hooks execute.
                unsharded = [p for p in module.parameters() if not isinstance(p, DTensor)]
                assert unsharded and all(p.dtype == torch.bfloat16 for p in unsharded)
                bf16_calls[id(module)] += 1

            def check_root_inputs(module, positional, keywords):
                assert not positional and set(keywords) == set(prepared[0])
                micro_step = len(root_calls)
                expected = dict(_tensor_leaves(prepared[micro_step]))
                actual = dict(_tensor_leaves(keywords))
                assert actual.keys() == expected.keys()
                assert {t.dtype for t in actual.values() if t.is_floating_point()} >= {
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                }
                for path, tensor in actual.items():
                    assert tensor.device == device, path
                    # Exact dtype/value checks at every nested tensor leaf catch
                    # accidental casting of timesteps, positions or indices.
                    torch.testing.assert_close(tensor, expected[path], rtol=0, atol=0, msg=str(path))
                root_calls.append(micro_step)

            def check_root_output(module, positional, keywords, output):
                assert isinstance(output, MiniMaxH3DiTOutput)
                assert all(p.shape[0] == _SAMPLES_PER_MICROBATCH for p in output.predictions)
                losses = output.loss
                assert losses and all(v.ndim == 0 for v in losses.values())
                assert all(v.requires_grad and v.dtype == torch.float32 for v in losses.values())
                outputs_seen.append(
                    (
                        [p.detach().cpu().clone() for p in output.predictions],
                        {key: value.detach().cpu().clone() for key, value in losses.items()},
                    )
                )

            for module in fsdp_modules:
                handles.append(module.register_forward_pre_hook(check_bf16_parameters, with_kwargs=True))
            handles.append(model.register_forward_pre_hook(check_root_inputs, with_kwargs=True))
            handles.append(model.register_forward_hook(check_root_output, with_kwargs=True))
            gradients_seen = []

            def capture_gradients(optimizer, positional, keywords):
                gradients_seen.append(
                    {name: _full_cpu(param.grad) for name, param in model.named_parameters() if param.grad is not None}
                )

            handles.append(base.model.optimizer.register_step_pre_hook(capture_gradients))
            torch.manual_seed(noise_seed)
            if step_driver == "base":
                # BaseTrainer owns generic optimizer/accumulation mechanics but
                # expects auxiliary metrics and causal-LM label bookkeeping.
                # Adapt only that signature; the real DiT forward/backward and
                # root FSDP hooks still execute, with no parallel-state patches.
                def forward_backward_step(batch):
                    batch = {key: value for key, value in batch.items() if key != "labels"}
                    loss, losses = trainer.forward_backward_step(batch)
                    return loss, losses, {}

                base.forward_backward_step = forward_backward_step
                labelled = [{**batch, "labels": torch.zeros(1, dtype=torch.long)} for batch in batches]
                base.train_step(iter([labelled]))
            else:
                trainer.train_step(iter([batches]))

            assert root_calls == [0, 1], "Each microbatch must enter the wrapped root exactly once"
            for module in fsdp_modules:
                expected_calls = _MICROBATCHES * (2 if checkpointing and module is not model else 1)
                assert bf16_calls[id(module)] == expected_calls, "Must observe main-block checkpoint recomputation"
            assert len(outputs_seen) == _MICROBATCHES
            assert len(gradients_seen) == 1
            assert recorder.begin_count == len(recorder.end_metrics) == 1
            assert base.state.global_step == 1
            assert base.model.lr_scheduler.last_epoch == scheduler_epoch + 1
            assert all(param.grad is None for param in model.parameters())

            # FP32 master parameters with differentiable BF16 forward copies
            # match FSDP's param_dtype policy without BF16 optimizer rounding.
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=_LR, foreach=False)
            reference_losses = {}
            for micro_step, columns in enumerate(prepared):
                packed_predictions, packed_losses = outputs_seen[micro_step]
                micro_losses = dict.fromkeys(packed_losses, 0.0)
                for index in range(_SAMPLES_PER_MICROBATCH):
                    legacy_inputs = {key: values[index] for key, values in columns.items()}
                    # The oracle does not need checkpoint recomputation (which
                    # would run after functional_call restores FP32 parameters).
                    legacy_inputs["use_gradient_checkpointing"] = False
                    output = functional_call(
                        reference,
                        {name: param.to(torch.bfloat16) for name, param in reference.named_parameters()},
                        (),
                        legacy_inputs,
                    )
                    assert output.loss.keys() == packed_losses.keys()
                    # Prediction keys are model-owned; native video/audio shapes
                    # are distinct and identify the legacy [video, audio] outputs.
                    predictions = [packed_predictions[0][index : index + 1], packed_predictions[1][index]]
                    predictions_by_shape = {tuple(p.shape): p for p in predictions}
                    assert len(predictions_by_shape) == len(output.predictions) == 2
                    for prediction in output.predictions:
                        actual = predictions_by_shape[tuple(prediction.shape)]
                        torch.testing.assert_close(actual, prediction.detach().cpu(), rtol=_BF16_RTOL, atol=_BF16_ATOL)
                    for name, loss in output.loss.items():
                        micro_losses[name] += loss.detach().cpu() / _SAMPLES_PER_MICROBATCH
                        reference_losses[name] = reference_losses.get(name, 0.0) + loss.item() / _LOCAL_SAMPLES
                    # Two equally sized microbatches, two equally weighted
                    # samples each. Modality weights already belong to H3.
                    (torch.stack(list(output.loss.values())).sum() / _LOCAL_SAMPLES).backward()
                for name, loss in micro_losses.items():
                    torch.testing.assert_close(packed_losses[name], loss, rtol=_BF16_RTOL, atol=_BF16_ATOL)

            assert all(param.grad is not None for param in reference.parameters())
            local_grads = {name: _full_cpu(param.grad) for name, param in reference.named_parameters()}
            assert local_grads.keys() == gradients_seen[0].keys()
            for param in reference.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=state.dp_group)
                param.grad.div_(_WORLD_SIZE)
            global_grads = {
                name: _full_cpu(param.grad) for name, param in reference.named_parameters() if param.grad is not None
            }
            expected_grad = _vector(global_grads.values())
            # Ensure the fixture really distinguishes reduced from rank-local
            # gradients at a margin greater than our BF16 tolerance.
            dp_signal = (_vector(local_grads.values()) - expected_grad).norm() / expected_grad.norm()
            assert dp_signal.item() > 2 * _UPDATE_REL_L2, f"Insufficient DP signal: {dp_signal.item()}"
            _assert_relative_l2(_vector(gradients_seen[0].values()), expected_grad, label="DP gradients")
            reference_optimizer.step()

            actual_updates, expected_updates = [], []
            reference_params = dict(reference.named_parameters())
            for name, param in model.named_parameters():
                actual = _full_cpu(param)
                expected = _full_cpu(reference_params[name])
                actual_updates.append(actual - initial[name])
                expected_updates.append(expected - initial[name])
                torch.testing.assert_close(actual, expected, rtol=_BF16_RTOL, atol=2e-4, msg=name)
            _assert_relative_l2(_vector(actual_updates), _vector(expected_updates), label="SGD parameter update")
            metrics = recorder.end_metrics[0]
            assert metrics["loss_dict"].keys() == reference_losses.keys()
            for name, loss in reference_losses.items():
                assert metrics["loss_dict"][name] == pytest.approx(loss, rel=_BF16_RTOL, abs=_BF16_ATOL)
            assert metrics["loss"] == pytest.approx(sum(reference_losses.values()), rel=_BF16_RTOL, abs=_BF16_ATOL)
            assert float(metrics["grad_norm"]) == pytest.approx(expected_grad.norm().item(), rel=_BF16_RTOL)
    finally:
        for handle in handles:
            handle.remove()
        # The launch helper also tears down on error; clear the registry after
        # destroying this real process group so its cached mesh cannot survive.
        if dist.is_initialized():
            dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.parametrize(
    "step_driver,checkpointing,attention",
    [
        pytest.param("dit", False, "eager", id="dit-eager"),
        pytest.param("base", True, "sdpa", id="base-sdpa-checkpoint"),
        pytest.param("dit", True, "veomni_flash_attention_2_with_sp", id="dit-fa2"),
        pytest.param("dit", True, "veomni_flash_attention_3_with_sp", id="dit-fa3"),
    ],
)
@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_minimax_h3_remove_padding_fsdp2(step_driver, checkpointing, attention, task):
    if not IS_CUDA_AVAILABLE or get_torch_device().device_count() < _WORLD_SIZE:
        pytest.skip("Native H3 FSDP2 remove-padding regression requires two available CUDA GPUs")
    if not get_torch_device().is_bf16_supported():
        pytest.skip("Native H3 FSDP2 remove-padding regression requires BF16-capable GPUs")
    if attention in ("veomni_flash_attention_2_with_sp", "veomni_flash_attention_3_with_sp"):
        module_name = "flash_attn" if attention == "veomni_flash_attention_2_with_sp" else "flash_attn_interface"
        if module_name == "flash_attn_interface" and get_torch_device().get_device_capability()[0] != 9:
            pytest.skip("The local FA3 kernel requires SM90 hardware")
        if importlib.util.find_spec(module_name) is None:
            pytest.skip(f"Missing local kernel package: {module_name}")
        # An installed-but-broken kernel/ABI must fail, not be reported as a skip.
        importlib.import_module(module_name)
    torchrun(
        _run_fsdp_regression,
        world_size=_WORLD_SIZE,
        step_driver=step_driver,
        checkpointing=checkpointing,
        attention=attention,
        task=task,
    )
