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

"""Cross-PR gate: cyclic packing plus the opt-in weighted Qwen loss."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_collator import DEFAULT_DATA_COLLATE_INFO


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))


def _worker(rank, rendezvous):
    from veomni.data import data_collator
    from veomni.data.cyclic_packing import CyclicPackedLoader, _StatelessMainCollator
    from veomni.models.loss_utils import cross_entropy_loss as causal
    from veomni.ops import VeomniOp
    from veomni.utils import loss_utils

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        parallel = SimpleNamespace(sp_enabled=True, sp_size=2, sp_rank=rank, sp_group=dist.group.WORLD, fsdp_size=2)
        data_collator.get_parallel_state = lambda: parallel
        loss_utils.get_parallel_state = lambda: parallel
        causal.get_parallel_state = lambda: parallel
        rows = []
        for index, tokens in enumerate(([1, 2], [3, 4, 5], [6, 7, 8, 9], [1], [2, 3])):
            ids = torch.tensor(tokens)
            row = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
            if index % 2 == 0:
                row["loss_weights"] = torch.full(ids.shape, 0.25 + 0.25 * index / 2)
            rows.append(row)

        class Dataset:
            def __len__(self):
                return len(rows)

            def __getitem__(self, index):
                return [rows[index]]

        step = SimpleNamespace(value=0)
        loader = CyclicPackedLoader(
            Dataset(),
            _StatelessMainCollator(data_collator.MainCollator()),
            max_seq_len=5,
            micro_batches_per_step=2,
            buffer_size=2,
            max_buffer_bytes=4096,
            input_contract="sha256:" + "c" * 64,
            source_fingerprint="weighted-fixture",
            deterministic=True,
            train_steps=2,
            train_epochs=1,
            sp_size=2,
            step_getter=lambda: step.value,
            group=dist.group.WORLD,
        )
        initial = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 100
        weight = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([weight], lr=0.01)
        reference_weight = torch.nn.Parameter(initial.clone())
        reference_optimizer = torch.optim.AdamW([reference_weight], lr=0.01)
        op = VeomniOp("cross_entropy_loss", "standard", "chunk_loss")
        micro_counts = []
        for _ in range(2):
            batches = next(loader)
            micro_counts.append(len(batches))
            local_mass = loss_utils.count_loss_token(batches, weighted_normalization=True)
            global_mass = loss_utils.reduce_global_loss_token(local_mass)
            actual_total = torch.tensor(0.0)
            ref_terms, ref_masses = [], []
            for batch in batches:
                ids = batch["input_ids"].flatten()
                hidden = torch.stack((ids.float(), ids.float().square() / 10, torch.ones_like(ids).float()), dim=-1)
                labels = batch["labels"].flatten()
                weights = batch.get("loss_weights")
                if weights is None:
                    weights = (labels != -100).float()
                else:
                    weights = weights.flatten()
                loss, _, _ = causal.ForCausalLMLoss(
                    hidden_states=hidden,
                    labels=labels,
                    weights=weight,
                    loss_weights=weights,
                    op=op,
                    loss_reduction_group=dist.group.WORLD,
                )
                token_counts = loss_utils.count_loss_token(batch, weighted_normalization=True)
                scaled = loss_utils.mean_global_loss(loss, token_counts, local_mass, global_mass)["foundation_loss"]
                actual_total = actual_total + scaled.detach()
                scaled.backward()

                shards = [None, None]
                dist.all_gather_object(shards, (hidden, labels, weights))
                for shard_hidden, shard_labels, shard_weights in shards:
                    terms = F.cross_entropy(
                        F.linear(shard_hidden, reference_weight), shard_labels, ignore_index=-100, reduction="none"
                    )
                    ref_terms.append((terms * shard_weights).sum())
                    ref_masses.append((shard_weights * (shard_labels != -100)).sum())
            reference = sum(ref_terms) / sum(ref_masses)
            reference.backward()
            dist.all_reduce(weight.grad)
            weight.grad /= 2
            torch.testing.assert_close(actual_total, reference.detach(), rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=1e-5, atol=1e-6)
            optimizer.step()
            reference_optimizer.step()
            torch.testing.assert_close(weight, reference_weight, rtol=1e-6, atol=1e-7)
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            step.value += 1
            loader.commit_step()
        assert micro_counts == [2, 1]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif("loss_weights" not in DEFAULT_DATA_COLLATE_INFO, reason="requires weighted-loss PR")
def test_cyclic_weighted_sp_loss_gradient_and_update(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
