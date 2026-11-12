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

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "sync_hf_modeling.py"
_SPEC = importlib.util.spec_from_file_location("sync_hf_modeling", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
sync_hf_modeling = importlib.util.module_from_spec(_SPEC)
sys.modules["sync_hf_modeling"] = sync_hf_modeling
_SPEC.loader.exec_module(sync_hf_modeling)


def test_parse_modeling_metadata_valid_and_none(tmp_path: Path):
    non_hf = tmp_path / "modeling_wan.py"
    non_hf.write_text("# Custom diffusion modeling\nimport torch\n", encoding="utf-8")
    assert sync_hf_modeling.parse_modeling_metadata(non_hf) is None

    hf_derived = tmp_path / "modeling_qwen3.py"
    hf_derived.write_text(
        "# Upstream-Module: transformers.models.qwen3.modeling_qwen3\n"
        "# Base-Transformers-Version: 5.16.1\n\n"
        "import torch\n",
        encoding="utf-8",
    )
    meta = sync_hf_modeling.parse_modeling_metadata(hf_derived)
    assert meta is not None
    assert meta.upstream_module == "transformers.models.qwen3.modeling_qwen3"
    assert meta.base_version == "5.16.1"

    partial = tmp_path / "modeling_broken.py"
    partial.write_text("# Upstream-Module: transformers.models.qwen3.modeling_qwen3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete upstream metadata"):
        sync_hf_modeling.parse_modeling_metadata(partial)


def test_rewrite_relative_imports():
    raw = (
        "import torch\n"
        "from ...activations import ACT2FN\n"
        "from ..llama.modeling_llama import LlamaRMSNorm\n"
        "from .configuration_qwen3 import Qwen3Config\n"
        "if True:\n"
        "    from ...modeling_utils import PreTrainedModel\n"
    )
    out = sync_hf_modeling.rewrite_relative_imports(raw, "transformers.models.qwen3.modeling_qwen3")
    assert "from transformers.activations import ACT2FN" in out
    assert "from transformers.models.llama.modeling_llama import LlamaRMSNorm" in out
    assert "from transformers.models.qwen3.configuration_qwen3 import Qwen3Config" in out
    assert "    from transformers.modeling_utils import PreTrainedModel" in out


def test_repo_hf_modeling_check_strict():
    assert sync_hf_modeling.main(["check", "--strict-version"]) == 0

    bad_dir = Path(sync_hf_modeling.TRANSFORMERS_MODELS_ROOT).parent.parent
    assert bad_dir.is_dir()


def test_three_way_merge_preserves_local_opslot_and_absorbs_upstream_edit():
    base_hf = (
        "import torch\n"
        "from ...activations import ACT2FN\n\n\n"
        "class DemoMLP:\n"
        "    def __init__(self, hidden_size: int):\n"
        "        self.hidden_size = hidden_size\n\n"
        "    def forward(self, x):\n"
        "        y = x * 2\n"
        "        return y\n"
    )
    ours = (
        "# Upstream-Module: transformers.models.demo.modeling_demo\n"
        "# Base-Transformers-Version: 5.16.1\n\n"
        "import torch\n"
        "from transformers.activations import ACT2FN\n"
        "from veomni.ops.dispatch import OpSlot\n\n"
        "veomni_mlp = OpSlot('swiglu_mlp', 'standard')\n\n\n"
        "class DemoMLP:\n"
        "    def __init__(self, hidden_size: int):\n"
        "        self.hidden_size = hidden_size\n\n"
        "    def forward(self, x):\n"
        "        if veomni_mlp.use_non_eager_impl:\n"
        "            return veomni_mlp(self, x)\n"
        "        y = x * 2\n"
        "        return y\n"
    )
    # Upstream bumps init and bottom of forward without touching our OpSlot guard!
    new_hf = (
        "import torch\n"
        "from ...activations import ACT2FN\n\n\n"
        "class DemoMLP:\n"
        "    def __init__(self, hidden_size: int, bias: bool = False):\n"
        "        self.hidden_size = hidden_size\n"
        "        self.bias = bias\n\n"
        "    def forward(self, x):\n"
        "        y = x * 2\n"
        "        return y + 1\n"
    )

    merged, had_conflicts = sync_hf_modeling.three_way_merge(
        ours_text=ours,
        base_upstream_text=base_hf,
        theirs_upstream_text=new_hf,
        upstream_module="transformers.models.demo.modeling_demo",
        new_version="5.20.0",
    )
    assert not had_conflicts
    assert "# Base-Transformers-Version: 5.20.0" in merged
    assert "if veomni_mlp.use_non_eager_impl:" in merged
    assert "self.bias = bias" in merged
    assert "return y + 1" in merged

    sig_notes = sync_hf_modeling.diff_signatures(base_hf, new_hf)
    assert any("CHANGED  DemoMLP.__init__" in s for s in sig_notes)
