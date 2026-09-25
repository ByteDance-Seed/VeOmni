"""Multi-process regression test for Flux Ulysses SP with non-divisible sequence lengths.

`FluxModel.forward` slices the image and text streams with `slice_input_tensor`, which pads
each stream up to a multiple of the SP size. The RoPE table and attention mask still covered
the unpadded `[text, image]` layout, so any padding crashed `apply_rope` with a shape
mismatch, and the pad tokens were never masked or stripped after the gathers.

This runs a small random-weight Flux (one joint block, one single block) with SP off and
with Ulysses SP on, for divisible and non-divisible lengths, and checks the outputs and SP-reduced parameter grads match.
"""

import pytest
import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    pytest.skip("c10d NCCL not available, skipping tests", allow_module_level=True)

from torch.testing._internal.common_utils import run_tests

from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.models.transformers.flux.config_flux import FluxConfig
from veomni.models.transformers.flux.modeling_flux import FluxModel

from .utils import SequenceParallelTest


# name: (latent H=W, prompt tokens). Flux patchifies 2x2, so image tokens = (H/2)^2.
CASES = {
    "divisible": (8, 4),
    "odd_image": (6, 4),
    "odd_image_and_text": (6, 5),
}


def _build_model(device):
    with torch.device("meta"):
        model = FluxModel(FluxConfig(num_blocks=1))
    model.single_blocks = model.single_blocks[:1]
    model = model.to_empty(device=device).float()
    gen = torch.Generator().manual_seed(0)
    for p in model.parameters():
        p.data.copy_(torch.randn(p.shape, generator=gen) * 0.02)
    return model


def _make_inputs(hw, txt, device):
    gen = torch.Generator().manual_seed(1)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(device)

    inputs = dict(
        hidden_states=randn(1, 16, hw, hw),
        timestep=torch.full((1,), 500.0, device=device),
        prompt_emb=randn(1, txt, 4096),
        pooled_prompt_emb=randn(1, 768),
        guidance=torch.full((1,), 3.5, device=device),
        text_ids=torch.zeros(1, txt, 3, device=device),
    )
    weight = torch.randn(1, 16, hw, hw, generator=gen).to(device)
    return inputs, weight


def _forward_backward(model, inputs, weight):
    model.zero_grad(set_to_none=True)
    out = model(**inputs)
    (out * weight).sum().backward()
    return out.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


class FluxUlyssesPaddingTest(SequenceParallelTest):
    @property
    def world_size(self):
        return 2

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_matches_non_sp_reference(self):
        group = self._get_process_group()
        device = torch.device(get_device_type(), self.rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = _build_model(device)

        try:
            _init_parallel_state(dp_size=self.world_size, ulysses_size=1, device_type=get_device_type(), name=None)
            refs = {}
            for name, (hw, txt) in CASES.items():
                inputs, weight = _make_inputs(hw, txt, device)
                refs[name] = (inputs, weight, *_forward_backward(model, inputs, weight))
            clear_parallel_state()

            _init_parallel_state(dp_size=1, ulysses_size=self.world_size, device_type=get_device_type(), name=None)
            failures = []
            for name, (inputs, weight, ref_out, ref_grads) in refs.items():
                try:
                    out, grads = _forward_backward(model, inputs, weight)
                except RuntimeError as e:
                    failures.append(f"{name}: {str(e).splitlines()[0]}")
                    continue
                out_err = ((out - ref_out).norm() / ref_out.norm()).item()

                sp_grad_names = [None] * self.world_size
                c10d.all_gather_object(sp_grad_names, set(grads), group=group)
                ref_grad_names = set(ref_grads)
                assert all(names == ref_grad_names for names in sp_grad_names), (
                    f"{name}: gradient-name mismatch; "
                    f"missing per rank: {[sorted(ref_grad_names - names) for names in sp_grad_names]}"
                )

                # Each rank's param grads cover only its own tokens, and the loss is replicated on
                # every SP rank, so sum over SP and divide by its size (FSDP's mean over the SP mesh).
                diff_sq = ref_sq = 0.0
                for n, g in grads.items():
                    c10d.all_reduce(g, group=group)
                    g /= self.world_size
                    diff_sq += (g - ref_grads[n]).pow(2).sum().item()
                    ref_sq += ref_grads[n].pow(2).sum().item()
                grad_err = (diff_sq / ref_sq) ** 0.5
                if out_err > 1e-3 or grad_err > 1e-3:
                    failures.append(f"{name}: output rel err {out_err:.2e}, grad rel err {grad_err:.2e}")
            assert not failures, "\n".join(failures)
        finally:
            clear_parallel_state()


if __name__ == "__main__":
    run_tests()
