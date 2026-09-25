"""Multi-process regression test for the diffusers Wan T2V model under Ulysses SP.

Covers three gaps in `WanTransformer3DModel_forward` / `WanSPAttnProcessor`:

1. A token count not divisible by the SP size crashed: `slice_input_tensor` pads the hidden
   states, but RoPE was sliced with a floor division, and the pad was never masked or stripped.
2. Per-token timesteps (Wan 2.2 style, `timestep.ndim == 2`) crashed: `timestep_proj` stayed
   at the full sequence length while the hidden states were sliced.
3. The `eager` backend never did the Ulysses all-to-all, so every rank attended only to its
   own shard.

Runs a tiny random-weight model with SP off and with Ulysses SP on and compares outputs and
SP-reduced parameter grads, with and without gradient checkpointing.
"""

import pytest
import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if get_device_type() == "cpu" or not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    pytest.skip("c10d NCCL not available, skipping tests", allow_module_level=True)

from torch.testing._internal.common_utils import run_tests

from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
    WanTransformer3DModelConfig,
)
from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
    WanTransformer3DModel,
    WanTransformer3DModel_forward,
)

from .utils import SequenceParallelTest


# name: (frames, height, width, per-token timestep). Patch (1, 2, 2) -> tokens = F * H/2 * W/2.
CASES = {
    "divisible": (2, 8, 8, False),
    "odd_tokens": (3, 6, 6, False),
    "per_token_timestep": (2, 8, 8, True),
    "odd_tokens_per_token_timestep": (3, 6, 6, True),
}


def _build_model(device, gradient_checkpointing):
    config = WanTransformer3DModelConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=32,
        freq_dim=32,
        ffn_dim=64,
        num_layers=2,
        rope_max_seq_len=64,
    )
    torch.manual_seed(0)
    model = WanTransformer3DModel(config, attn_implementation="eager").to(device)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model.train()


def _forward_backward(model, frames, height, width, per_token, device):
    gen = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 4, frames, height, width, generator=gen).to(device)
    num_tokens = frames * (height // 2) * (width // 2)
    if per_token:
        timestep = (torch.rand(1, num_tokens, generator=gen) * 1000).to(device)
    else:
        timestep = torch.tensor([500.0], device=device)
    context = torch.randn(1, 7, 32, generator=gen).to(device)

    model.zero_grad(set_to_none=True)
    out = WanTransformer3DModel_forward(model, latents, timestep, context)
    weight = torch.randn(out.shape, generator=gen).to(device)
    (out * weight).sum().backward()
    return out.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


class WanT2VUlyssesTest(SequenceParallelTest):
    @property
    def world_size(self):
        return 2

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_matches_non_sp_reference(self):
        group = self._get_process_group()
        device = torch.device(get_device_type(), self.rank)
        torch.backends.cuda.matmul.allow_tf32 = False

        failures = []
        for gradient_checkpointing in (False, True):
            model = _build_model(device, gradient_checkpointing)
            try:
                _init_parallel_state(dp_size=self.world_size, ulysses_size=1, device_type=get_device_type(), name=None)
                refs = {name: _forward_backward(model, *case, device) for name, case in CASES.items()}
                clear_parallel_state()

                _init_parallel_state(dp_size=1, ulysses_size=self.world_size, device_type=get_device_type(), name=None)
                for name, case in CASES.items():
                    tag = f"{name}{' (ckpt)' if gradient_checkpointing else ''}"
                    try:
                        out, grads = _forward_backward(model, *case, device)
                    except RuntimeError as e:
                        failures.append(f"{tag}: {str(e).splitlines()[0]}")
                        continue
                    ref_out, ref_grads = refs[name]
                    out_err = ((out - ref_out).norm() / ref_out.norm()).item()
                    # Param grads cover only this rank's tokens and the loss is replicated on every
                    # SP rank, so sum over SP and divide by its size (FSDP's mean over the SP mesh).
                    diff_sq = ref_sq = 0.0
                    for n, g in grads.items():
                        c10d.all_reduce(g, group=group)
                        g /= self.world_size
                        diff_sq += (g - ref_grads[n]).pow(2).sum().item()
                        ref_sq += ref_grads[n].pow(2).sum().item()
                    grad_err = (diff_sq / ref_sq) ** 0.5
                    if out_err > 1e-5 or grad_err > 1e-5:
                        failures.append(f"{tag}: output rel err {out_err:.2e}, grad rel err {grad_err:.2e}")
            finally:
                clear_parallel_state()
        assert not failures, "\n".join(failures)


if __name__ == "__main__":
    run_tests()
