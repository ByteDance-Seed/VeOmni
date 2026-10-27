"""Multi-process regression test for the diffusers Wan T2V model under Ulysses SP.

Covers `WanTransformer3DModel_forward` / `WanSPAttnProcessor` on the cases that used to break:

1. A token count not divisible by the SP size: the communication pad must not reach
   attention, and RoPE must line up with each token.
2. Per-token timesteps (Wan 2.2 style, `timestep.ndim == 2`): `timestep_proj` has to be
   sliced along with the hidden states.
3. The `eager` backend has to see the full sequence, not just the local shard.

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

from veomni.distributed.parallel_state import use_parallel_state
from veomni.models.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
    WanTransformer3DModelConfig,
)
from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
    WanTransformer3DModel,
    WanTransformer3DModel_forward,
    apply_veomni_wan_transformer_patch,
)

from .utils import SequenceParallelTest


# name: (frames, height, width, per-token timestep). Patch (1, 2, 2) -> tokens = F * H/2 * W/2.
CASES = {
    "divisible": (2, 8, 8, False),
    "odd_tokens": (3, 6, 6, False),
    "per_token_timestep": (2, 8, 8, True),
    "odd_tokens_per_token_timestep": (3, 6, 6, True),
}


# FSDP2 wraps the blocks, so the model forward has to reach them through `block(...)` (and its hooks).
apply_veomni_wan_transformer_patch()


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
        torch.set_float32_matmul_precision("highest")

        failures = []
        for gradient_checkpointing in (False, True):
            model = _build_model(device, gradient_checkpointing)
            # FSDP2 unshards a wrapped block in its forward pre-hook, so every block has to be
            # entered through `__call__` rather than by calling its forward helper directly.
            entered = set()
            for i, block in enumerate(model.blocks):
                block.register_forward_pre_hook(lambda mod, args, i=i, entered=entered: entered.add(i))
            with use_parallel_state(self.dp_state):
                refs = {name: _forward_backward(model, *case, device) for name, case in CASES.items()}

            for name, case in CASES.items():
                tag = f"{name}{' (ckpt)' if gradient_checkpointing else ''}"
                entered.clear()
                try:
                    with use_parallel_state(self.sp_state):
                        out, grads = _forward_backward(model, *case, device)
                except RuntimeError as e:
                    failures.append(f"{tag}: {str(e).splitlines()[0]}")
                    continue
                ref_out, ref_grads = refs[name]
                # Param grads cover only this rank's tokens and the loss is replicated on every
                # SP rank, so sum over SP and divide by its size (FSDP's mean over the SP mesh).
                for g in grads.values():
                    c10d.all_reduce(g, group=group)
                    g /= self.world_size
                if grads.keys() != ref_grads.keys():
                    failures.append(f"{tag}: grad params differ: {sorted(grads.keys() ^ ref_grads.keys())}")
                    continue
                pairs = [("output", out, ref_out)] + [(f"grad {n}", grads[n], ref_grads[n]) for n in ref_grads]
                for what, actual, expected in pairs:
                    # Matching infinities would pass assert_close, so require finite values first.
                    if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                        failures.append(f"{tag}: non-finite {what}")
                        continue
                    try:
                        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3, equal_nan=False)
                    except AssertionError as e:
                        failures.append(f"{tag}: {what}: {' '.join(str(e).split())}")
                if entered != set(range(len(model.blocks))):
                    failures.append(f"{tag}: blocks bypassed nn.Module.__call__: {entered}")
        assert not failures, "\n".join(failures)


if __name__ == "__main__":
    run_tests()
