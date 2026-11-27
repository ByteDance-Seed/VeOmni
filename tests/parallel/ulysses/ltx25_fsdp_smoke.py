"""Manual two-GPU FSDP2 LTX-2.5 smoke with same-weight dense references."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from tests.models.test_ltx25 import make_pack, tiny_config, tiny_model
from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
from veomni.models.diffusers.ltx2_3.ltx_condition.configuration_ltx2_3_condition import LTXVideoConditionModelConfig
from veomni.models.diffusers.ltx2_3.ltx_condition.ltx25 import prepare_training_directory
from veomni.models.diffusers.ltx2_3.ltx_condition.modeling_ltx2_3_condition import LTXVideoConditionModel
from veomni.trainer.dit_trainer import DiTDataCollator
from veomni.utils.device import empty_cache, get_device_type, get_dist_comm_backend, get_torch_device


def full(tensor):
    """Collectively materialize a DTensor or return an ordinary tensor unchanged."""
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def make_model():
    # Preserve the released model's head count; all other widths are reduced.
    """Build a reduced model retaining the released audio and video head counts."""
    return tiny_model(
        tiny_config(
            num_attention_heads=32,
            audio_num_attention_heads=32,
            num_layers=2,
            cross_attention_dim=512,
            audio_cross_attention_dim=512,
            caption_channels=512,
            connector_num_attention_heads=32,
            audio_connector_num_attention_heads=32,
        )
    )


def make_lora(model):
    """Attach rank-four native LoRA adapters to the attention projections."""
    return VeOmniLoraModel(
        model,
        VeOmniLoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
    )


def make_inputs(condition, device, step, odd):
    """Prepare deterministic mock conditions with divisible or odd token lengths."""
    torch.manual_seed(100 + step)
    h, w, audio_frames = (3, 3, 7) if odd else (2, 2, 8)
    row = dict(
        latents=torch.randn(1, 8, 3, h, w, device=device),
        context=torch.randn(1, 128, 512, device=device),
        context_mask=torch.cat([torch.zeros(1, 64), torch.ones(1, 64)], dim=1).to(device=device, dtype=torch.long),
        audio_context=torch.randn(1, 128, 512, device=device),
        audio_latents=torch.randn(1, 2, audio_frames, 4, device=device),
    )
    return condition.process_condition(**DiTDataCollator()([row]))


def main():
    """Compare two-rank FSDP2 updates with same-weight unsharded references."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device(get_device_type(), int(os.environ["LOCAL_RANK"]))
    get_torch_device().set_device(device)
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group(get_dist_comm_backend(), timeout=timedelta(minutes=5))
    assert dist.get_world_size() == 2
    _init_parallel_state(dp_size=2 // args.sp, ulysses_size=args.sp, dp_mode="fsdp2", device_type=get_device_type())
    _init_parallel_state(dp_size=2, ulysses_size=1, dp_mode="fsdp2", device_type=get_device_type(), name="reference")
    output_dir = Path(os.environ["LTX25_SMOKE_OUTPUT"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for first_frame in (False, True):
        for odd in (False, True):
            case = f"{'i2av' if first_frame else 't2av'}-{'odd' if odd else 'divisible'}"
            base = make_model()
            checkpoint = output_dir / f"base-sp{args.sp}-{case}"
            if rank == 0:
                pack = output_dir / f"base-sp{args.sp}-{case}.safetensors"
                make_pack(pack, base)
                prepare_training_directory(str(pack), str(checkpoint))
            dist.barrier()
            model = build_parallelize_model(
                make_lora(base),
                weights_path=str(checkpoint),
                is_peft_model=True,
                broadcast_model_weights_from_rank0=True,
                init_device="meta",
                enable_gradient_checkpointing=True,
                mixed_precision=MixedPrecisionConfig(enable=False),
            ).train()
            assert any(isinstance(p, DTensor) for p in model.parameters())
            reference = make_lora(make_model()).to(device).train()
            reference.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            condition_config = LTXVideoConditionModelConfig.from_pretrained(
                str(checkpoint),
                with_audio=True,
                first_frame_conditioning_p=float(first_frame),
            )
            condition = LTXVideoConditionModel(condition_config, meta_init=True).to(device).float().eval()
            trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            assert trainable and all("lora_" in n for n, _ in trainable)
            optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-3, weight_decay=0)
            for step in range(1, 4):
                prepared = make_inputs(condition, device, step, odd)
                # Refresh before each comparison so optimizer drift cannot hide scaling errors.
                reference.load_state_dict({n: full(v).detach() for n, v in model.state_dict().items()})
                optimizer.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                before = {n: full(p).detach().clone() for n, p in trainable}
                result = model(**prepared)
                loss = sum(result.loss.values())
                assert torch.isfinite(loss)
                loss.backward()
                with use_parallel_state("reference"):
                    expected = reference(**prepared)
                    reference_loss = sum(expected.loss.values())
                    reference_loss.backward()
                torch.testing.assert_close(loss, reference_loss, rtol=2e-4, atol=2e-5)
                prediction_errors = []
                for actuals, references in (
                    (result.predictions, expected.predictions),
                    (result.audio_predictions, expected.audio_predictions),
                ):
                    for actual, ref in zip(actuals, references):
                        torch.testing.assert_close(actual, ref, rtol=2e-3, atol=2e-5)
                        prediction_errors.append(float((actual - ref).norm() / ref.norm().clamp_min(1e-30)))
                reference_params = dict(reference.named_parameters())
                error_sq = reference_sq = actual_sq = max_error = 0.0
                for name, parameter in trainable:
                    assert parameter.grad is not None, name
                    gradient = full(parameter.grad)
                    ref_gradient = reference_params[name].grad
                    assert ref_gradient is not None and torch.isfinite(gradient).all(), name
                    torch.testing.assert_close(gradient, ref_gradient, rtol=2e-3, atol=2e-5, msg=name)
                    error_sq += float((gradient - ref_gradient).square().sum())
                    reference_sq += float(ref_gradient.square().sum())
                    actual_sq += float(gradient.square().sum())
                    max_error = max(max_error, float((gradient - ref_gradient).abs().max()))
                relative_error = (error_sq / max(reference_sq, 1e-30)) ** 0.5
                assert relative_error < 2e-3, (case, args.sp, relative_error)
                norm = model.clip_grad_norm_(1.0, error_if_nonfinite=True)
                assert float(norm) > 0
                optimizer.step()
                delta = sum(float((full(p).detach() - before[n]).abs().sum()) for n, p in trainable)
                assert delta > 0
                record = dict(
                    case=case,
                    sp=args.sp,
                    dp=2 // args.sp,
                    step=step,
                    loss=float(loss),
                    reference_loss=float(reference_loss),
                    loss_absolute_error=abs(float(loss) - float(reference_loss)),
                    prediction_relative_error=max(prediction_errors),
                    gradient_relative_error=relative_error,
                    gradient_max_absolute_error=max_error,
                    gradient_norm_ratio=(actual_sq / max(reference_sq, 1e-30)) ** 0.5,
                    gradient_norm=float(norm),
                    parameter_update_l1=delta,
                )
                if rank == 0:
                    records.append(record)
                    print("LTX25_SMOKE_STEP " + json.dumps(record), flush=True)
            del model, reference, condition, optimizer, base, result, expected
            empty_cache()
            dist.barrier()
    if rank == 0:
        (output_dir / f"results-sp{args.sp}.json").write_text(json.dumps(records, indent=2))
        print(f"LTX25_SMOKE_PASS sp={args.sp} cases=4 optimizer_updates={len(records)}", flush=True)
    dist.destroy_process_group()
    clear_parallel_state()


if __name__ == "__main__":
    main()
