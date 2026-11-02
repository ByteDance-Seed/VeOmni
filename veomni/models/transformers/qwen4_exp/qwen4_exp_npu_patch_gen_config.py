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
                "            # The CANN Lightning Indexer's TND (packed-flat) layout carries the\n"
                "            # segment boundaries natively, so both single-segment rows and the\n"
                "            # padding-free packed training batches take the kernel: it\n"
                "            # reproduces the eager loop's per-segment floor-causal selection,\n"
                "            # and the wrapper offsets its per-segment block ids into this flat\n"
                "            # segment-major block list. Only sequence/CP parallelism\n"
                "            # (world_size > 1) and callers without host-side segment lengths\n"
                "            # (a cu_seq_lens_q tensor but no packed_seq_lens) stay on eager.\n"
                "            if cu_seq_lens_q is None:\n"
                "                host_segment_lens = [global_seq_len] * batch_size\n"
                "            elif packed_seq_lens is not None:\n"
                "                host_segment_lens = [int(value) for value in packed_seq_lens]\n"
                "            else:\n"
                "                host_segment_lens = None\n"
                "            use_cann = (\n"
                "                self.veomni_qsa_indexer.impl == \"cann\"\n"
                "                and world_size == 1\n"
                "                and host_segment_lens is not None\n"
                "                and sum(host_segment_lens) == batch_size * local_seq_len\n"
                "            )\n"
                "            if use_cann:\n"
                "                seg_block_counts = [length // self.compress_ratio for length in host_segment_lens]\n"
                "                if sum(seg_block_counts) != block_starts.numel():\n"
                "                    raise ValueError(\n"
                "                        \"Qwen4-Exp compact QSA CANN indexer needs the flat block list to hold exactly \"\n"
                "                        f\"the per-segment blocks; got {block_starts.numel()} blocks for segment lengths \"\n"
                "                        f\"{host_segment_lens} with compress_ratio={self.compress_ratio}.\"\n"
                "                    )\n"
                "                cu_q_host = [0]\n"
                "                cu_k_host = [0]\n"
                "                for length, count in zip(host_segment_lens, seg_block_counts):\n"
                "                    cu_q_host.append(cu_q_host[-1] + length)\n"
                "                    cu_k_host.append(cu_k_host[-1] + count)\n"
                "                device = hidden_states.device\n"
                "                block_ids = self.veomni_qsa_indexer(\n"
                "                    q,\n"
                "                    pooled_keys[0],\n"
                "                    self.compress_ratio,\n"
                "                    self.block_topk,\n"
                "                    cu_seq_lens_q=torch.tensor(cu_q_host, dtype=torch.int32, device=device),\n"
                "                    cu_seq_lens_k=torch.tensor(cu_k_host, dtype=torch.int32, device=device),\n"
                "                    cmp_residual_k=torch.tensor(\n"
                "                        [length % self.compress_ratio for length in host_segment_lens],\n"
                "                        dtype=torch.int32,\n"
                "                        device=device,\n"
                "                    ),\n"
                "                    batch_size=len(host_segment_lens),\n"
                "                    max_seqlen_q=max(host_segment_lens),\n"
                "                    max_seqlen_k=max(seg_block_counts),\n"
                "                )\n"
                "                valid_ids = block_ids >= 0\n"
                "                safe_ids = block_ids.clamp(min=0).to(torch.long)\n"
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
                "                            \"Qwen4-Exp QSA CANN indexer covers the flat packed layout with host-side \"\n"
                "                            \"segment lengths and no sequence/CP parallelism; falling back to the eager loop.\"\n"
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
