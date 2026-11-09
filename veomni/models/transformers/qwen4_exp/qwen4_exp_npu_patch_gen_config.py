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
"""Patch configuration for the Qwen4-Exp NPU correctness path.

Regen command:
patchgen veomni.models.transformers.qwen4_exp.qwen4_exp_npu_patch_gen_config -o veomni/models/transformers/qwen4_exp/generated --diff

Qwen4-Exp uses partial interleaved mRoPE and experimental QSA/PLE code paths,
so this initial NPU build deliberately inherits the device-agnostic GPU patch
set without adding incompatible fused rotary or attention replacements.
"""

from copy import deepcopy

from veomni.models.transformers.qwen4_exp.qwen4_exp_gpu_patch_gen_config import config as gpu_config


config = deepcopy(gpu_config)
config.target_file = "patched_modeling_qwen4_exp_npu.py"
config.description = "Qwen4-Exp NPU VLM-SFT correctness integration with PLE sharding"

for _patch in config.patches:
    if _patch.target == "Qwen4ExpTextQSAIndexer.__init__":
        _patch.name_map = {
            "self.veomni_rope = VeomniOp(\"rope\", \"partial\", config.ops_config.rotary_pos_emb_implementation)": (
                "self.veomni_rope = VeomniOp(\"rope\", \"partial\", config.ops_config.rotary_pos_emb_implementation)\n"
                "        self.veomni_qsa_indexer = (\n"
                "            VeomniOp(\"qsa_indexer\", \"standard\", \"npu_ascendc\")\n"
                "            if config.ops_config.qsa_indexer_implementation == \"npu_ascendc\"\n"
                "            else None\n"
                "        )"
            ),
        }
        break
else:
    raise RuntimeError("Expected Qwen4ExpTextQSAIndexer.__init__ in qwen4_exp GPU patch config")

# Two forward edits: invisible-block masks become -1.0 (NPU TopKV2 faults on
# -inf rows), and the eager TopK branch re-arms with the Lightning Indexer.
for _patch in config.patches:
    if _patch.target == "Qwen4ExpTextQSAIndexer.forward":
        _patch.name_map = {
            'scores = scores.masked_fill(~visible, float("-inf"))': (
                "scores = scores.masked_fill(~visible, -1.0)  # NPU TopKV2 faults on sparse -inf rows (ReLU scores >= 0)"
            ),
            "if block_starts.numel():": (
                "if self.veomni_qsa_indexer is not None and block_starts.numel():\n"
                "            selected_local = self.veomni_qsa_indexer(\n"
                "                q,\n"
                "                pooled_keys,\n"
                "                cu_seq_lens_q=cu_seq_lens_q,\n"
                "                packed_seq_lens=packed_seq_lens,\n"
                "                compress_ratio=self.compress_ratio,\n"
                "                block_topk=self.block_topk,\n"
                "            )\n"
                "        elif block_starts.numel():"
            ),
        }
        break
else:
    raise RuntimeError("Expected Qwen4ExpTextQSAIndexer.forward in qwen4_exp GPU patch config")
