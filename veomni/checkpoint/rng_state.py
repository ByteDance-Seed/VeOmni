# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Full RNG snapshot/restore for exact-resume determinism.

The checkpoint callback historically saved only the torch CPU generator state, so
Python's ``random``, NumPy, and the CUDA generators diverged across a resume. These
helpers capture and restore all four so a resumed run reproduces the continuous run's
sampling (data shuffling, dropout, flow/diffusion noise on non-derived paths, ...).

Each field is optional on restore: a snapshot taken without CUDA (or NumPy absent)
simply omits it, and an older checkpoint that stored only ``torch_rng_state`` is still
honored by the caller.
"""

import random
from typing import Any, Dict

import torch


def snapshot_rng_state() -> Dict[str, Any]:
    """Capture Python / NumPy / torch-CPU / torch-CUDA RNG states (per rank)."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:  # noqa: BLE001 - NumPy optional; omit rather than fail the save
        pass
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore whatever fields are present in ``state`` (missing ones are left as-is)."""
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(_as_byte_tensor(state["torch_cpu"]))
    if state.get("numpy") is not None:
        import numpy as np

        np.random.set_state(state["numpy"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        cuda_states = [_as_byte_tensor(s) for s in state["torch_cuda"]]
        # Only restore when the device count matches; a different GPU count is a
        # topology change the manifest already rejects, but guard defensively.
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)


def _as_byte_tensor(value: Any) -> torch.Tensor:
    """torch RNG states must be CPU ByteTensors; coerce defensively after (de)serialization."""
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.cpu().to(torch.uint8)


__all__ = ["snapshot_rng_state", "restore_rng_state"]
