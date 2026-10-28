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
    DPO_APPROVED_KEY,
    DPO_POSTCHECK_KEY,
    DPO_PROOF_KEY,
    DPOInvalidProof,
    DPOPackedProof,
    callback_metadata_digests,
    packed_batch_digest,
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
        group_id = rank // 2
        state = SimpleNamespace(sp_enabled=True, sp_size=2, sp_rank=rank % 2, sp_group=groups[group_id])
        if scenario in ("dp_only", "dp_only_count_mismatch"):
            state = SimpleNamespace(sp_enabled=False)
        reference = state
        if scenario == "reference_mismatch" and rank == 1:
            reference = SimpleNamespace(sp_enabled=False)
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
            ("a" * 64,),
            ("b" if group_id == 0 else "c") * 64,
            packed_batch_digest(batch),
            packed_batch_digest(batch, after_callbacks=True),
            packed_batch_digest(batch, global_only=True),
            layout_sp_size=2 if state.sp_enabled else 1,
            layout_sp_rank=rank % 2 if state.sp_enabled else 0,
            producer_contract_version=1,
        )
        if scenario in ("source_range", "catalog_mismatch"):
            index = 42 if scenario == "source_range" and rank == 1 else 0
            batch["ds_idx"] = torch.tensor([index])
            proof = replace(
                proof,
                source_indices=(index,),
                packed_digest=packed_batch_digest(batch),
                forward_digest=packed_batch_digest(batch, after_callbacks=True),
                callback_metadata_digests=callback_metadata_digests(batch, ()),
            )
        if scenario in ("dp_only", "dp_only_count_mismatch"):
            proof = replace(proof, input_digest=str(rank) * 64)
        batch[DPO_PROOF_KEY] = proof
        batches = [batch]
        if scenario == "same_shape_different_input" and rank == 1:
            batch[DPO_PROOF_KEY] = replace(proof, input_digest="d" * 64)
        elif scenario == "missing" and rank == 1:
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
        elif scenario == "callback_drop":
            batches = [batch, batch.copy()]
        elif scenario == "all_missing":
            batch.pop(DPO_PROOF_KEY)
        elif scenario == "empty_batch" and rank == 1:
            batches = []
        elif scenario == "mutated_labels" and rank == 1:
            batch["labels"] = torch.cat((batch["labels"], batch["labels"][:, -1:]), dim=-1)
        elif scenario == "mutated_attention" and rank == 1:
            batch["cu_seq_lens_k"] = torch.tensor([0, 4])
        if (scenario == "one_eof" and rank == 1) or scenario == "all_eof":
            iterator = iter([])
        elif scenario == "loader_error" and rank == 1:

            def failing_iterator():
                raise ValueError("injected loader failure")
                yield

            iterator = failing_iterator()
        else:
            iterator = iter([batches])
        try:
            step = 2 if scenario == "wrong_step" and rank == 1 else 1
            epoch = 2 if scenario == "wrong_epoch" and rank == 1 else 1
            out = next_dpo_microbatches(
                iterator,
                state,
                reference,
                global_step=step,
                epoch=epoch,
                source_count=(lambda: 1) if scenario == "source_range" else None,
                source_catalog=(lambda: ("a", "different") if rank == 1 else ("a", "b"))
                if scenario == "catalog_mismatch"
                else None,
                vocab_size=(lambda: 17 if rank == 1 else 16) if scenario == "vocab_mismatch" else None,
            )
            expected_proofs = tuple(batch[DPO_APPROVED_KEY] for batch in out)
            if scenario == "callback_drop":
                out.pop()
            if scenario == "callback_mutation" and rank == 1:
                out[0]["labels"][0, 0] += 1
            if scenario == "callback_metadata_added" and rank == 1:
                out[0]["cur_token_num"] = 12345
            next_dpo_microbatches(
                iter([out]),
                state,
                reference,
                global_step=step,
                epoch=epoch,
                after_callbacks=True,
                expected_proofs=expected_proofs,
            )
            result = "approved" if DPO_APPROVED_KEY in out[0] else "legacy"
        except StopIteration:
            result = "eof"
        except RuntimeError:
            result = "error"
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
        "same_shape_different_input",
        "missing",
        "mixed_dp",
        "invalid",
        "zero_target",
        "microbatch_count",
        "one_eof",
        "all_eof",
        "reference_mismatch",
        "loader_error",
        "wrong_step",
        "wrong_epoch",
        "empty_batch",
        "all_missing",
        "dp_only",
        "dp_only_count_mismatch",
        "mutated_labels",
        "mutated_attention",
        "different_global_metadata",
        "vocab_mismatch",
        "catalog_mismatch",
        "callback_mutation",
        "callback_metadata_added",
        "callback_drop",
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
        next_dpo_microbatches(
            iter([[approved]]),
            state,
            state,
            global_step=1,
            after_callbacks=True,
            expected_proofs=(approved[DPO_APPROVED_KEY],),
        )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(-torch.arange(1, 13, dtype=torch.float32)[rank * 6 : (rank + 1) * 6])

            def forward(self, input_ids, **kwargs):
                assert all(key not in kwargs for key in (DPO_PROOF_KEY, DPO_APPROVED_KEY, DPO_POSTCHECK_KEY))
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
