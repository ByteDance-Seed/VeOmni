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

# Warn once per process when the QSA CANN indexer cannot take a layout.
config.add_post_import_block("_QSA_CANN_LAYOUT_WARNED = False")

# Bind the qsa_indexer VeomniOp next to the indexer's partial rope op.
for _patch in config.patches:
    if _patch.target == "Qwen4ExpTextQSAIndexer.__init__":
        _patch.name_map = {
            "self.veomni_rope = VeomniOp(\"rope\", \"partial\", config.ops_config.rotary_pos_emb_implementation)": (
                "self.veomni_rope = VeomniOp(\"rope\", \"partial\", config.ops_config.rotary_pos_emb_implementation)\n"
                "        self.veomni_qsa_indexer = VeomniOp(\"qsa_indexer\", \"standard\", "
                "config.ops_config.qsa_indexer_implementation)"
            ),
        }
        break
else:
    raise RuntimeError("Expected Qwen4ExpTextQSAIndexer.__init__ in qwen4_exp GPU patch config")

# Replace the GPU QSA indexer mask value in-place so NPU masks invisible block
# scores with -1.0 instead of -inf. ReLU block scores are non-negative, and
# CANN TopKV2 faults (AICore error 507015) on sparse rows dominated by -inf.
# Then dispatch the single-segment causal case (world_size == 1, no packed
# batch) to the qsa_indexer VeomniOp's "cann" row — the CANN Lightning Indexer
# from cann_ops_transformer — keeping the eager einsum+ReLU+topk loop as the
# fallback for Ulysses/CP and packed layouts. name_map entries apply in dict
# order, so the selection-loop entry below sees the -1.0 mask line above.
for _patch in config.patches:
    if _patch.target == "Qwen4ExpTextQSAIndexer.forward":
        _patch.name_map = {
            'scores = scores.masked_fill(~visible, float("-inf"))': (
                "scores = scores.masked_fill(~visible, -1.0)  # NPU TopKV2 faults on sparse -inf rows (ReLU scores >= 0)"
            ),
            "if block_starts.numel():\n"
            "            top_count = min(self.block_topk, block_starts.numel())\n"
            "            block_offsets = torch.arange(self.compress_ratio, dtype=torch.int32, "
            "device=hidden_states.device)\n"
            "            elements_per_query = max(1, batch_size * block_starts.numel() * self.index_n_heads)\n"
            "            query_chunk_size = max(1, min(local_seq_len, 16 * 1024 * 1024 // elements_per_query))\n"
            "            selected_chunks = []\n"
            "            for query_start in range(0, local_seq_len, query_chunk_size):\n"
            "                query_end = min(query_start + query_chunk_size, local_seq_len)\n"
            "                scores = torch.einsum(\n"
            '                    "blhd,nd->blnh",\n'
            "                    q[:, query_start:query_end].float(),\n"
            "                    pooled_keys[0].float(),\n"
            "                )\n"
            "                scores = torch.relu(scores).sum(dim=-1) / self.index_head_dim**0.5\n"
            "                visible = (block_segment_ids[None, None] == "
            "local_segment_ids[:, query_start:query_end, None]) & (\n"
            "                    block_starts[None, None] + self.compress_ratio - 1\n"
            "                    <= query_positions[None, query_start:query_end, None]\n"
            "                )\n"
            "                scores = scores.masked_fill(~visible, -1.0)  # NPU TopKV2 faults on sparse -inf rows "
            "(ReLU scores >= 0)\n"
            "                top_blocks = scores.topk(top_count, dim=-1).indices\n"
            "                top_valid = (\n"
            "                    visible.to(torch.int8).gather(-1, top_blocks).bool()\n"
            '                    if hidden_states.device.type == "npu"\n'
            "                    else visible.gather(-1, top_blocks)\n"
            "                )\n"
            "                selected_starts = block_starts[top_blocks].to(torch.int32)\n"
            "                selected_chunk = selected_starts[..., None] + block_offsets\n"
            "                selected_chunks.append(selected_chunk.masked_fill(~top_valid[..., None], -1).flatten(-2))\n"
            "            selected_local = torch.cat(selected_chunks, dim=1)": (
                "if block_starts.numel():\n"
                "            top_count = min(self.block_topk, block_starts.numel())\n"
                "            block_offsets = torch.arange(self.compress_ratio, dtype=torch.int32, "
                "device=hidden_states.device)\n"
                "            # The CANN Lightning Indexer covers the single-segment causal case: no\n"
                "            # sequence/CP parallelism (world_size == 1) and no packed batch. Packed\n"
                "            # layouts set cu_seq_lens_q, so its absence also rules out packing. The\n"
                "            # flat block list is batch-major with equal-length segments there, so the\n"
                "            # kernel's per-element block ids map back through batch-strided flat ids.\n"
                "            use_cann = self.veomni_qsa_indexer.impl == \"cann\" and world_size == 1 "
                "and cu_seq_lens_q is None\n"
                "            if use_cann:\n"
                "                n_blocks = local_seq_len // self.compress_ratio\n"
                "                if block_starts.numel() != batch_size * n_blocks:\n"
                "                    raise ValueError(\n"
                "                        \"Qwen4-Exp compact QSA CANN indexer needs batch-major equal-length blocks; \"\n"
                "                        f\"got {block_starts.numel()} blocks for batch_size={batch_size}, \"\n"
                "                        f\"local_seq_len={local_seq_len}, compress_ratio={self.compress_ratio}.\"\n"
                "                    )\n"
                "                block_ids = self.veomni_qsa_indexer(\n"
                "                    q,\n"
                "                    pooled_keys[0].reshape(batch_size, n_blocks, self.index_head_dim),\n"
                "                    self.compress_ratio,\n"
                "                    self.block_topk,\n"
                "                )\n"
                "                valid_ids = block_ids >= 0\n"
                "                element_offsets = torch.arange(batch_size, device=hidden_states.device) * n_blocks\n"
                "                safe_ids = (block_ids + element_offsets.view(-1, 1, 1)).clamp(min=0)\n"
                "                selected_starts = block_starts[safe_ids].to(torch.int32)\n"
                "                selected_local = (\n"
                "                    (selected_starts[..., None] + block_offsets).masked_fill(~valid_ids[..., None], -1)"
                ".flatten(-2)\n"
                "                )\n"
                "            else:\n"
                "                if self.veomni_qsa_indexer.impl == \"cann\":\n"
                "                    global _QSA_CANN_LAYOUT_WARNED\n"
                "                    if not _QSA_CANN_LAYOUT_WARNED:\n"
                "                        _QSA_CANN_LAYOUT_WARNED = True\n"
                "                        logger.warning(\n"
                "                            \"Qwen4-Exp QSA CANN indexer covers only single-segment causal selection \"\n"
                "                            \"(no sequence/CP parallelism, no packed batch); falling back to the eager loop.\"\n"
                "                        )\n"
                "                elements_per_query = max(1, batch_size * block_starts.numel() * self.index_n_heads)\n"
                "                query_chunk_size = max(1, min(local_seq_len, 16 * 1024 * 1024 // elements_per_query))\n"
                "                selected_chunks = []\n"
                "                for query_start in range(0, local_seq_len, query_chunk_size):\n"
                "                    query_end = min(query_start + query_chunk_size, local_seq_len)\n"
                "                    scores = torch.einsum(\n"
                '                        "blhd,nd->blnh",\n'
                "                        q[:, query_start:query_end].float(),\n"
                "                        pooled_keys[0].float(),\n"
                "                    )\n"
                "                    scores = torch.relu(scores).sum(dim=-1) / self.index_head_dim**0.5\n"
                "                    visible = (block_segment_ids[None, None] == "
                "local_segment_ids[:, query_start:query_end, None]) & (\n"
                "                        block_starts[None, None] + self.compress_ratio - 1\n"
                "                        <= query_positions[None, query_start:query_end, None]\n"
                "                    )\n"
                "                    scores = scores.masked_fill(~visible, -1.0)  # NPU TopKV2 faults on sparse -inf rows "
                "(ReLU scores >= 0)\n"
                "                    top_blocks = scores.topk(top_count, dim=-1).indices\n"
                "                    top_valid = (\n"
                "                        visible.to(torch.int8).gather(-1, top_blocks).bool()\n"
                '                        if hidden_states.device.type == "npu"\n'
                "                        else visible.gather(-1, top_blocks)\n"
                "                    )\n"
                "                    selected_starts = block_starts[top_blocks].to(torch.int32)\n"
                "                    selected_chunk = selected_starts[..., None] + block_offsets\n"
                "                    selected_chunks.append(selected_chunk.masked_fill(~top_valid[..., None], -1)"
                ".flatten(-2))\n"
                "                selected_local = torch.cat(selected_chunks, dim=1)"
            ),
        }
        break
else:
    raise RuntimeError("Expected Qwen4ExpTextQSAIndexer.forward in qwen4_exp GPU patch config")
