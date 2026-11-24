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
"""Full determinism has to cover the attention backward, not just the flags around it.

``use_deterministic_algorithms`` cannot reach the backend a masked call actually uses.
Flash refuses a non-null ``attn_mask``, so SDPA falls to the memory-efficient kernel, whose
backward is nondeterministic and answers ``warn_only=True`` with a warning rather than a
failure. ``enable_full_determinism`` therefore turns that backend off -- and cuDNN with it,
which gates on ``warn_only`` the same way -- and these tests are what holds them off.

The shape is not decorative. Measured on one A800 by ``scripts/bench_sdpa_determinism.py``,
the backward disagreed with itself on 16 of 16 repeats at ``head_dim`` 256 and on none at
128 -- so a narrower head passes whether or not the fix is in place. DiffusionGemma runs
256, and on the 26B-A4B recipe this was the whole remaining gap: two backwards of one
micro-batch disagreed on up to 603 of 631 parameters by as much as 8.4%, and none once the
memory-efficient backend was off.

The drift is also a property of the device. Rather than assume it reproduces wherever CI
runs, the first test measures it and says so when it does not, since a gate that cannot
observe the fault it guards is worth knowing about.
"""

import os
import random
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from veomni.utils.device import IS_CUDA_AVAILABLE
from veomni.utils.helper import enable_full_determinism


HEADS, SEQ, HEAD_DIM, WINDOW = 8, 1024, 256, 512
REPEATS = 16

TOUCHED_ENVIRONMENT = (
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "NCCL_DETERMINISTIC",
    "FLASH_ATTENTION_DETERMINISTIC",
    "CLOSE_MATMUL_K_SHIFT",
)


@pytest.fixture
def restore_global_state():
    """``enable_full_determinism`` reaches process-wide state, so put all of it back.

    Anything missed here leaks into whatever runs next in the same process -- and the seeds
    it resets are as much of that state as the backend switches are.
    """
    saved_environment = {name: os.environ.get(name) for name in TOUCHED_ENVIRONMENT}
    saved = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if IS_CUDA_AVAILABLE:
        saved["math"] = torch.backends.cuda.math_sdp_enabled()
        saved["mem_efficient"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        saved["cudnn_sdp"] = torch.backends.cuda.cudnn_sdp_enabled()
        # Every device, because `enable_full_determinism` calls `manual_seed_all`.
        saved["cuda_rng"] = torch.cuda.get_rng_state_all()

    yield

    for name, value in saved_environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    torch.use_deterministic_algorithms(saved["deterministic_algorithms"], warn_only=saved["warn_only"])
    torch.backends.cudnn.deterministic = saved["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = saved["cudnn_benchmark"]
    torch.backends.cudnn.enabled = saved["cudnn_enabled"]
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if IS_CUDA_AVAILABLE:
        torch.backends.cuda.enable_math_sdp(saved["math"])
        torch.backends.cuda.enable_mem_efficient_sdp(saved["mem_efficient"])
        torch.backends.cuda.enable_cudnn_sdp(saved["cudnn_sdp"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def build_masked_attention():
    """A sliding causal mask, as an additive 4D tensor -- what the encoder's sliding layers pass."""
    torch.manual_seed(0)
    shape = (1, HEADS, SEQ, HEAD_DIM)
    tensors = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]

    positions = torch.arange(SEQ, device="cuda")
    visible = (positions[None, :] <= positions[:, None]) & (positions[None, :] > positions[:, None] - WINDOW)
    mask = torch.zeros(1, 1, SEQ, SEQ, device="cuda", dtype=torch.bfloat16)
    mask.masked_fill_(~visible, float("-inf"))
    return (*tensors, mask)


def masked_attention_gradients(query, key, value, upstream, mask, backend=None):
    leaves = [tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value)]
    with nullcontext() if backend is None else sdpa_kernel(backend):
        out = torch.nn.functional.scaled_dot_product_attention(*leaves, attn_mask=mask)
    out.backward(upstream)
    return [leaf.grad for leaf in leaves]


def backward_repeats(inputs, backend=None):
    first = masked_attention_gradients(*inputs, backend=backend)
    for _ in range(REPEATS):
        again = masked_attention_gradients(*inputs, backend=backend)
        if any(not torch.equal(a, b) for a, b in zip(again, first)):
            return False
    return True


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="the backward's accumulation order only varies on device")
def test_full_determinism_makes_the_masked_attention_backward_repeat(restore_global_state):
    """The gate, plus the control that says whether the gate can see anything.

    ``enable_full_determinism`` runs before the inputs are built, as it does in a real run:
    it writes ``CUBLAS_WORKSPACE_CONFIG``, which cuBLAS reads when it first creates a
    handle, so setting it after a matmul has already run is setting it too late.
    """
    enable_full_determinism(0)
    inputs = build_masked_attention()

    assert backward_repeats(inputs), (
        "two identical masked backwards disagreed under enable_full_determinism, so the run it promises to "
        "reproduce does not reproduce"
    )

    try:
        drifted = not backward_repeats(inputs, backend=SDPBackend.EFFICIENT_ATTENTION)
        why = "did not drift"
    except RuntimeError:  # the backend refuses these arguments here, so there is nothing to compare
        drifted, why = False, "could not be measured"
    if not drifted:
        pytest.skip(
            f"the memory-efficient backward {why} on this device at head_dim={HEAD_DIM}, so the assertion "
            "above holds with or without the fix -- it was measured drifting 16/16 on an A800"
        )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="the SDPA backend switches are CUDA-only")
def test_full_determinism_leaves_a_backend_that_can_serve_a_masked_call(restore_global_state):
    """Turning the nondeterministic backends off is only safe if something else answers.

    Flash cannot take an ``attn_mask`` at all, so with the other two off, math is the only
    backend left that can serve a masked call -- without it every masked attention in the
    model would raise instead of running.
    """
    enable_full_determinism(0)

    assert not torch.backends.cuda.mem_efficient_sdp_enabled()
    assert not torch.backends.cuda.cudnn_sdp_enabled()
    assert torch.backends.cuda.math_sdp_enabled()

    gradients = masked_attention_gradients(*build_masked_attention())
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
