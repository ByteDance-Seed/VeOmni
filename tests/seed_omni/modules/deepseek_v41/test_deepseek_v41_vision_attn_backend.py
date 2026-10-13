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

"""Which varlen attention backend the ViT picks, and whether the fallback agrees.

FA2 publishes no aarch64 wheel and ``pyproject.toml`` scopes ``flash-attn`` to
x86_64, so a Grace node (GB200 / GH200) has CUDA and no ``flash_attn``. Selecting
the backend on ``IS_CUDA_AVAILABLE`` therefore turns a missing wheel into a
module-level ``ModuleNotFoundError`` that takes the entire ``deepseek_v41``
package down at import -- not a degraded ViT, no model at all. The selection has
to be made on the package being importable.

Two failure modes, one test each:

* the import guard regresses to ``IS_CUDA_AVAILABLE`` and every aarch64 CUDA box
  loses the package. Invisible on x86 CI, where FA2 is always installed, so the
  first test runs a subprocess with FA2 made unimportable;
* the SDPA fallback is reachable but disagrees with FA2, which no shape or
  finiteness check would catch -- the ViT would simply encode images wrongly on
  one architecture. The second test pins it against FA2 on packed varlen input.
"""

import subprocess
import sys
import textwrap

import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE
from veomni.utils.import_utils import is_flash_attn_2_available


# Runs in a fresh interpreter: patching the flag in-process would poison
# ``sys.modules`` for every other test sharing this pytest session.
_IMPORT_WITHOUT_FA2 = textwrap.dedent(
    """
    import sys

    import veomni.utils.import_utils as import_utils

    assert "veomni.models.seed_omni.modules.deepseek_v41.vision.modeling" not in sys.modules, (
        "vision.modeling was already imported; the guard under test never ran"
    )

    import_utils._PACKAGE_FLAGS["flash_attn"] = False
    # Any surviving unconditional ``import flash_attn`` now raises instead of
    # quietly succeeding because this x86 runner happens to have the wheel.
    sys.modules["flash_attn"] = None

    from veomni.models.seed_omni.modules.deepseek_v41.vision import modeling

    assert modeling.USE_FLASH_ATTN_VARLEN is False, modeling.USE_FLASH_ATTN_VARLEN

    import veomni.models.seed_omni.modules as modules

    assert hasattr(modules, "deepseek_v41")
    print("OK")
    """
)


def test_package_imports_when_flash_attn_2_is_absent():
    """A CUDA box without FA2 -- i.e. aarch64 -- still gets the whole package."""
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_FA2],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "OK" in result.stdout


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and is_flash_attn_2_available()),
    reason="needs CUDA and FA2 to compare the fallback against the kernel it replaces",
)
@pytest.mark.parametrize("lengths", [(37, 64, 13), (1, 128), (256,)])
def test_sdpa_fallback_matches_flash_attn_varlen(lengths):
    """The fallback the aarch64 path takes must agree with FA2, not merely run."""
    from flash_attn import flash_attn_varlen_func

    from veomni.models.seed_omni.modules.deepseek_v41.vision.modeling import _sdpa_varlen_attn

    torch.manual_seed(0)
    n_heads, head_dim = 8, 64
    total = sum(lengths)
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    q, k, v = (torch.randn(total, n_heads, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    expected = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max(lengths), max(lengths), causal=False)
    actual = _sdpa_varlen_attn(q, k, v, cu_seqlens)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
