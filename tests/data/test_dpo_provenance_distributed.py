# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Exercise real rank agreement before DPO reaches any model collective."""

import datetime
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.trainer.dpo_provenance import next_dpo_microbatches
from veomni.utils.dpo_provenance import (
    DPO_AGREED_KEY,
    DPO_APPROVED_KEY,
    DPO_PROOF_KEY,
    DPOInvalidProof,
    DPOPackedProof,
)


def _worker(rank, world_size, scenario, endpoint):
    dist.init_process_group(
        "gloo",
        store=endpoint.connect(),
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        alternate_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        group_id = rank // 2
        state = SimpleNamespace(sp_enabled=True, sp_size=2, sp_rank=rank % 2, sp_group=groups[group_id])
        if scenario in ("dp_only", "dp_only_count_mismatch"):
            state = SimpleNamespace(sp_enabled=False)
        state.fsdp_group = dist.group.WORLD
        reference = state
        if scenario == "reference_mismatch" and rank == 1:
            reference = SimpleNamespace(sp_enabled=False)
        if scenario == "reference_members_mismatch":
            reference = SimpleNamespace(
                sp_enabled=True, sp_size=2, sp_rank=rank % 2, sp_group=alternate_groups[rank % 2]
            )
        width = 2 if state.sp_enabled else 4
        batch = {
            "input_ids": torch.zeros((1, width), dtype=torch.long),
            "labels": torch.zeros((1, width), dtype=torch.long),
            "position_ids": torch.zeros((1, width), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "cu_seq_lens_q": torch.tensor([0, 2, 4]),
            "cu_seq_lens_k": torch.tensor([0, 2, 4]),
            "linear_attn_cu_seq_lens_q": torch.tensor([0, 2, 4]),
            "packed_sequence_slices": ((0, 2), (2, 4)),
            "max_length_q": 2,
            "max_length_k": 2,
            "attention_mask_is_all_ones": True,
        }
        if scenario == "different_global_metadata" and rank == 1:
            batch["max_length_q"] = 1
        proof = DPOPackedProof(
            (2, 2),
            (4,),
            0,
            layout_sp_size=2 if state.sp_enabled else 1,
            layout_sp_rank=rank % 2 if state.sp_enabled else 0,
            producer_contract_version=1,
        )
        if scenario == "source_range":
            index = 42 if scenario == "source_range" and rank == 1 else 0
            batch["ds_idx"] = torch.tensor([index])
            proof = replace(
                proof,
                source_indices=(index,),
            )
        batch[DPO_PROOF_KEY] = proof
        batches = [batch]
        if scenario == "missing" and rank == 1:
            batch.pop(DPO_PROOF_KEY)
        elif scenario == "mixed_dp" and group_id == 1:
            batch.pop(DPO_PROOF_KEY)
        elif scenario == "invalid" and rank == 1:
            batch[DPO_PROOF_KEY] = DPOInvalidProof("injected invalid metadata")
        elif scenario == "zero_target" and rank == 1:
            batch[DPO_PROOF_KEY] = replace(batch[DPO_PROOF_KEY], segment_lengths=(1, 3))
        elif (scenario == "microbatch_count" and group_id == 1) or (
            scenario == "dp_only_count_mismatch" and rank == 1
        ):
            batches = [batch, batch.copy()]
        elif scenario == "all_missing":
            batch.pop(DPO_PROOF_KEY)
        elif scenario == "empty_batch" and rank == 1:
            batches = []
        elif scenario == "bad_label_shape" and rank == 1:
            batch["labels"] = torch.cat((batch["labels"], batch["labels"][:, -1:]), dim=-1)
        elif scenario == "bad_attention_cu" and rank == 1:
            batch["cu_seq_lens_k"] = torch.tensor([0, 4])
        elif scenario == "invalid_local_cu" and rank == 1:
            batch["cu_seq_lens_q"] = torch.tensor([0, 2, 2, 4])
        elif scenario == "invalid_local_tail" and rank == 1:
            batch["tail_padding_length"] = torch.tensor(1)
            batch[DPO_PROOF_KEY] = replace(proof, collator_tail=1)
        if (scenario == "one_eof" and rank == 1) or scenario == "all_eof":
            iterator = iter([])
        elif scenario == "loader_error" and rank == 1:

            def failing_iterator():
                raise ValueError("injected loader failure")
                yield

            iterator = failing_iterator()
        else:
            iterator = iter([batches])
        from veomni.trainer.text_dpo_trainer import _dpo_pair_weighting

        out, loader_error = None, None
        try:
            out = next_dpo_microbatches(
                iterator,
                state,
                reference,
                global_step=1,
                epoch=1,
                source_count=(lambda: 1) if scenario == "source_range" else None,
            )
        except Exception as exc:
            loader_error = exc
        original_gather, original_object_gather = dist.all_gather, dist.all_gather_object
        if scenario == "consistent":

            def checked_gather(output, tensor, **kwargs):
                assert tensor.shape == (3,) and tensor.dtype == torch.int64
                return original_gather(output, tensor, **kwargs)

            def unexpected_object_gather(*args, **kwargs):
                raise AssertionError("normal pair-count reduction must not gather Python objects")

            dist.all_gather = checked_gather
            dist.all_gather_object = unexpected_object_gather
        try:
            _dpo_pair_weighting(out, SimpleNamespace(), state, True, loader_error=loader_error)
            result = "approved" if DPO_APPROVED_KEY in out[0] else "legacy"
        except StopIteration:
            result = "eof"
        except RuntimeError as exc:
            if scenario in ("invalid_local_cu", "invalid_local_tail"):
                assert "DPO pair-count preflight failed" in str(exc)
                assert "DPO packed sequence boundaries changed after collation" in str(exc)
            result = "error"
        finally:
            dist.all_gather, dist.all_gather_object = original_gather, original_object_gather
        results = [None] * world_size
        dist.all_gather_object(results, result)
        expected = "approved" if scenario in ("consistent", "dp_only") else "eof" if scenario == "all_eof" else "error"
        assert results == [expected] * world_size
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
@pytest.mark.parametrize(
    "scenario",
    [
        "consistent",
        "missing",
        "mixed_dp",
        "invalid",
        "zero_target",
        "microbatch_count",
        "one_eof",
        "all_eof",
        "reference_mismatch",
        "reference_members_mismatch",
        "loader_error",
        "empty_batch",
        "all_missing",
        "dp_only",
        "dp_only_count_mismatch",
        "bad_label_shape",
        "bad_attention_cu",
        "invalid_local_cu",
        "invalid_local_tail",
        "different_global_metadata",
        "source_range",
    ],
)
def test_world_agreement_and_uniform_pre_forward_rejection(scenario):
    spawn_with_rendezvous(_worker, args=(4, scenario), nprocs=4)


def _training_worker(rank, world_size, endpoint):
    import veomni.data.data_collator as data_collator
    import veomni.trainer.text_dpo_trainer as dpo
    from veomni.data.data_transform import process_dpo_example
    from veomni.data.dpo_collator import DPOProvenanceCollator

    dist.init_process_group(
        "gloo",
        store=endpoint.connect(),
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        state = SimpleNamespace(sp_enabled=True, sp_size=world_size, sp_rank=rank, sp_group=dist.group.WORLD)
        data_collator.get_parallel_state = dpo.get_parallel_state = lambda: state

        class Tokenizer:
            dpo_unpadded_output_contract = True
            dpo_unpadded_output_contract_id = "test-tokenizer-v1"

            def encode(self, text, add_special_tokens=True):
                return list(range(1, len(text) + 1))

        features = [
            process_dpo_example(example, tokenizer=Tokenizer(), record_segment_provenance=True)[0]
            for example in (
                {"chosen": "abc", "rejected": "def"},
                {"prompt": "g", "chosen": "h", "rejected": "i"},
            )
        ]
        batch = DPOProvenanceCollator(data_collator.MainCollator(pad_to_length=12))(features)
        approved = next_dpo_microbatches(iter([[batch]]), state, state, global_step=1)[0]
        state.fsdp_group = dist.group.WORLD
        dpo._dpo_pair_weighting([approved], data_collator.PostCollator(), state, True)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(-torch.arange(1, 13, dtype=torch.float32)[rank * 6 : (rank + 1) * 6])

            def forward(self, input_ids, **kwargs):
                assert all(key not in kwargs for key in (DPO_PROOF_KEY, DPO_APPROVED_KEY, DPO_AGREED_KEY))
                return SimpleNamespace(fused_linear_aux=SimpleNamespace(log_probs=self.values[None]))

        trainer = dpo.TextDPOTrainer.__new__(dpo.TextDPOTrainer)
        trainer.base = SimpleNamespace(
            args=SimpleNamespace(dpo_config=SimpleNamespace(average_log_prob=False, explicit_segment_provenance=True))
        )
        trainer.post_forward = data_collator.PostCollator()
        model = Model()
        chosen, rejected = trainer.concatenated_forward(model, approved)
        loss, _, _ = trainer.dpo_loss(chosen, rejected, torch.zeros_like(chosen), torch.zeros_like(rejected), beta=0.1)
        loss.mean().backward()

        reference = -torch.arange(1, 13, dtype=torch.float32).requires_grad_()
        ref_chosen = torch.stack((reference[0] + reference[1], reference[6]))
        ref_rejected = torch.stack((reference[3] + reference[4], reference[8]))
        ref_loss = -F.logsigmoid(0.1 * (ref_chosen - ref_rejected)).mean()
        reference_gradient = torch.autograd.grad(ref_loss, reference)[0]
        torch.testing.assert_close(chosen, ref_chosen, rtol=0, atol=0)
        torch.testing.assert_close(rejected, ref_rejected, rtol=0, atol=0)
        torch.testing.assert_close(loss.mean(), ref_loss, rtol=0, atol=0)
        # _Gather sums each rank's identical loss gradient; FSDP averaging is not part of this synthetic test.
        torch.testing.assert_close(
            model.values.grad, 2 * reference_gradient[rank * 6 : (rank + 1) * 6], rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_real_sp_gather_loss_and_gradients_with_single_token_and_tail():
    spawn_with_rendezvous(_training_worker, args=(2,), nprocs=2)
