"""Multi-process regression test for LTX-2.3 under Ulysses SP.

Covers two gaps in `LTXVideoModel_forward` / `LTXSPAttention_forward`:

1. A video or audio token count not divisible by the SP size crashed: `slice_input_tensor`
   pads `x`, but the pad was never masked in self-attention or stripped after the gathers.
2. The audio+video model crashed under SP for any length: the per-token AV cross-attention
   AdaLN inputs and cross RoPE were not sliced, and the audio<->video cross-attention used the
   other modality's local shard as its context instead of the full sequence.

Runs a tiny random-weight model with SP off and with Ulysses SP on, video-only and audio+video,
with and without gradient checkpointing, and compares predictions and SP-reduced grads.
"""

import pytest
import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    pytest.skip("c10d NCCL not available, skipping tests", allow_module_level=True)

from torch.testing._internal.common_utils import run_tests

import veomni.models.diffusers.ltx2_3.ltx_core  # noqa: F401
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.models.diffusers.ltx2_3.ltx_transformer.configuration_ltx2_3_transformer import (
    LTXVideoTransformerModelConfig,
)
from veomni.models.diffusers.ltx2_3.ltx_transformer.modeling_ltx2_3_transformer import (
    LTXVideoTransformerModel,
    apply_veomni_ltx_transformer_patch,
)

from .utils import SequenceParallelTest


# name: (video frames, height, width, audio frames or 0). patch_size=1 -> video tokens = F*H*W.
CASES = {
    "video_divisible": (2, 4, 4, 0),
    "video_odd": (3, 3, 3, 0),
    "av_divisible": (2, 4, 4, 8),
    "av_odd_audio": (2, 4, 4, 7),
    "av_odd_both": (3, 3, 3, 7),
}


def _build_model(device, with_audio, gradient_checkpointing):
    config = LTXVideoTransformerModelConfig(
        in_channels=8,
        out_channels=8,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=2,
        cross_attention_dim=32,
        caption_channels=32,
        with_audio=with_audio,
        audio_num_attention_heads=2,
        audio_attention_head_dim=16,
        audio_in_channels=8,
        audio_out_channels=8,
        audio_cross_attention_dim=32,
    )
    torch.manual_seed(0)
    model = LTXVideoTransformerModel(config).to(device).float()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    model.set_gradient_checkpointing(gradient_checkpointing)
    return model.train()


def _forward_backward(model, frames, height, width, audio_frames, device):
    gen = torch.Generator().manual_seed(1)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(device)

    kwargs = dict(
        hidden_states=[randn(1, 8, frames, height, width)],
        timestep=[torch.tensor(0.5, device=device)],
        encoder_hidden_states=[randn(1, 5, 32)],
        training_target=[randn(1, 8, frames, height, width)],
    )
    if audio_frames:
        kwargs.update(
            audio_hidden_states=[randn(1, 2, audio_frames, 4)],
            audio_timestep=[torch.tensor(0.5, device=device)],
            audio_training_target=[randn(1, 2, audio_frames, 4)],
        )
    model.zero_grad(set_to_none=True)
    out = model(**kwargs)
    out.loss["mse_loss"].backward()
    preds = [out.predictions[0].detach()] + ([out.audio_predictions[0].detach()] if audio_frames else [])
    return preds, {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


class LTX2UlyssesTest(SequenceParallelTest):
    @property
    def world_size(self):
        return 2

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_matches_non_sp_reference(self):
        apply_veomni_ltx_transformer_patch()
        group = self._get_process_group()
        device = torch.device(get_device_type(), self.rank)
        torch.backends.cuda.matmul.allow_tf32 = False

        failures = []
        for with_audio in (False, True):
            for gradient_checkpointing in (False, True):
                model = _build_model(device, with_audio, gradient_checkpointing)
                cases = {name: case for name, case in CASES.items() if bool(case[3]) == with_audio}
                try:
                    _init_parallel_state(
                        dp_size=self.world_size, ulysses_size=1, device_type=get_device_type(), name=None
                    )
                    refs = {name: _forward_backward(model, *case, device) for name, case in cases.items()}
                    clear_parallel_state()

                    _init_parallel_state(
                        dp_size=1, ulysses_size=self.world_size, device_type=get_device_type(), name=None
                    )
                    for name, case in cases.items():
                        tag = f"{name}{' (ckpt)' if gradient_checkpointing else ''}"
                        try:
                            preds, grads = _forward_backward(model, *case, device)
                        except RuntimeError as e:
                            failures.append(f"{tag}: {str(e).splitlines()[0]}")
                            continue
                        ref_preds, ref_grads = refs[name]
                        pred_err = max(((p - r).norm() / r.norm()).item() for p, r in zip(preds, ref_preds))
                        # Param grads cover only this rank's tokens and the loss is replicated on every
                        # SP rank, so sum over SP and divide by its size (FSDP's mean over the SP mesh).
                        diff_sq = ref_sq = 0.0
                        for n, g in grads.items():
                            c10d.all_reduce(g, group=group)
                            g /= self.world_size
                            diff_sq += (g - ref_grads[n]).pow(2).sum().item()
                            ref_sq += ref_grads[n].pow(2).sum().item()
                        grad_err = (diff_sq / ref_sq) ** 0.5
                        if pred_err > 1e-5 or grad_err > 1e-5:
                            failures.append(f"{tag}: prediction rel err {pred_err:.2e}, grad rel err {grad_err:.2e}")
                finally:
                    clear_parallel_state()
        assert not failures, "\n".join(failures)


if __name__ == "__main__":
    run_tests()
