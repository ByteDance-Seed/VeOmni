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

"""CPU distributed checks for cyclic cursor consensus and training failures."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.trainer.callbacks import global_state_callback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))


class _Rows:
    def __init__(self, rank=0, changed=False):
        self.rows = []
        for index, length in enumerate((3, 2, 4, 1, 2, 3, 2)):
            value = index + 1 + int(changed and rank == 1 and index == 0)
            self.rows.append(
                {
                    "input_ids": torch.full((length,), value),
                    "labels": torch.full((length,), value),
                    "attention_mask": torch.ones(length, dtype=torch.long),
                    "sample_id": index,
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return [self.rows[index]]


def _worker(rank, rendezvous):
    from datasets import Dataset

    from veomni.arguments.arguments_types import DataloaderConfig
    from veomni.data import data_collator
    from veomni.data.cyclic_packing import CyclicPackedLoader, _StatelessMainCollator
    from veomni.data.dataset import MappingDataset
    from veomni.trainer import base as trainer_base
    from veomni.trainer.base import BaseTrainer

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        parallel = SimpleNamespace(sp_enabled=True, sp_size=2, sp_rank=rank)
        data_collator.get_parallel_state = lambda: parallel
        trainer_base.get_parallel_state = lambda: SimpleNamespace(
            dp_size=1, tp_size=1, sp_size=2, sp_group=dist.group.WORLD
        )
        if not hasattr(global_state_callback, "_saved_multi_optimizer_generations"):
            global_state_callback._saved_multi_optimizer_generations = lambda *_args: {}

        # Exercise the standard trainer builder, optimizer hook and commit
        # hook with an actual HF mapping dataset and MainCollator.
        def transform(row):
            ids = torch.tensor(row["ids"], dtype=torch.long)
            return [{"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}]

        trainer = BaseTrainer.__new__(BaseTrainer)
        config = DataloaderConfig(
            type="cyclic_packed",
            num_workers=0,
            drop_last=False,
            shuffle=False,
            pin_memory=False,
        )
        checkpoint = SimpleNamespace(save_async=False, load_path=None)
        trainer.args = SimpleNamespace(
            data=SimpleNamespace(
                dataloader=config,
                data_type="conversation",
                datasets_type="mapping",
                enable_multisource=False,
                max_seq_len=5,
                dyn_bsz_buffer_size=3,
            ),
            train=SimpleNamespace(
                checkpoint=checkpoint,
                cyclic_input_contract="sha256:" + "b" * 64,
                cyclic_input_deterministic=True,
                micro_batch_size=1,
                global_batch_size=2,
                dyn_bsz=True,
                dyn_bsz_runtime="main",
                dyn_bsz_count_mode="total",
                bsz_warmup_ratio=0,
                num_train_epochs=1,
            ),
            train_steps=4,
        )
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        optimizer._is_multi_optimizer = True
        trainer.model = SimpleNamespace(
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
            checkpoint=SimpleNamespace(_last_saved_optimizer_generation=None),
        )
        trainer.train_dataset = MappingDataset(
            Dataset.from_dict({"ids": [[1, 2], [3, 4, 5], [6, 7, 8, 9]]}), transform
        )
        trainer.collate_fn = data_collator.MainCollator()
        trainer.state = SimpleNamespace(global_step=0)
        trainer._callbacks = []
        trainer._build_sft_dataloader()
        trainer.state.global_step = 1
        batches = next(trainer.train_dataloader)
        assert batches and len(batches) == 2
        (parameter * 2).backward()
        trainer._complete_optimizer_step()
        trainer.on_step_end()
        assert trainer.train_dataloader.state_dict()["step"] == 1
        assert parameter.item() < 1.0

        def make_loader(step, changed=False):
            collator = _StatelessMainCollator(data_collator.MainCollator())
            return CyclicPackedLoader(
                _Rows(rank, changed),
                collator,
                max_seq_len=5,
                micro_batches_per_step=2,
                buffer_size=3,
                max_buffer_bytes=4096,
                input_contract="sha256:" + "a" * 64,
                source_fingerprint="distributed-fixture",
                deterministic=True,
                train_steps=6,
                train_epochs=1,
                sp_size=2,
                step_getter=lambda: step.value,
                group=dist.group.WORLD,
            )

        step = SimpleNamespace(value=0)
        loader = make_loader(step)
        saved = []
        for _ in range(2):
            batches = next(loader)
            assert batches and all(batch["labels"].ndim == 2 for batch in batches)
            step.value += 1
            loader.commit_step()
            saved.append(loader.state_dict())

        partial = make_loader(SimpleNamespace(value=0))
        for row in partial.dataset.rows:
            row["labels"].fill_(-100)
        partial.dataset.rows[0]["labels"][1] = 1
        partial_batches = next(partial)
        local_valid = any(bool((batch["labels"] != -100).any()) for batch in partial_batches)
        validity = [None, None]
        dist.all_gather_object(validity, local_valid)
        assert validity == [True, False]

        no_labels = make_loader(SimpleNamespace(value=0))
        for row in no_labels.dataset.rows:
            row["labels"].fill_(-100)
        with pytest.raises(ValueError, match="no supervised tokens"):
            next(no_labels)

        # A mixed checkpoint may be locally valid on both ranks. All ranks
        # must still reject it before one advances into a model collective.
        restored_step = SimpleNamespace(value=rank + 1)
        restored = make_loader(restored_step)
        restored.load_state_dict(saved[rank])
        with pytest.raises(RuntimeError, match="inputs differ"):
            restored.validate_restore_consensus()
        flag = torch.tensor([1])
        dist.all_reduce(flag)
        assert flag.item() == 2

        # Source content diverges on only one rank: both ranks reject the
        # batch, and the process group remains usable afterwards.
        changed = make_loader(SimpleNamespace(value=0), changed=True)
        with pytest.raises(RuntimeError, match="inputs differ"):
            next(changed)
        flag.fill_(1)
        dist.all_reduce(flag)
        assert flag.item() == 2

        class BadRows(_Rows):
            def __getitem__(self, index):
                if rank == 1:
                    raise ValueError("injected source failure")
                return super().__getitem__(index)

        failed_source = make_loader(SimpleNamespace(value=0))
        failed_source.dataset = BadRows()
        with pytest.raises((ValueError, RuntimeError), match="source failure|another rank"):
            next(failed_source)
        flag.fill_(1)
        dist.all_reduce(flag)
        assert flag.item() == 2

        class Optimizer:
            def step(self):
                if rank == 1:
                    raise ValueError("injected optimizer failure")

            def zero_grad(self):
                pass

        trainer = SimpleNamespace(
            model=SimpleNamespace(optimizer=Optimizer(), lr_scheduler=SimpleNamespace(step=lambda: None)),
            train_dataloader=SimpleNamespace(_veomni_requires_optimizer_commit=True),
        )
        with pytest.raises((ValueError, RuntimeError), match="optimizer failure|another rank"):
            BaseTrainer._complete_optimizer_step(trainer)
        flag.fill_(1)
        dist.all_reduce(flag)
        assert flag.item() == 2

        class Callback:
            def on_train_begin(self, _state):
                pass

        callback = Callback()
        cursor = SimpleNamespace(_veomni_requires_optimizer_commit=True, _restored=rank == 0)
        restoring = SimpleNamespace(
            _callbacks=[callback],
            global_state_callback=callback,
            train_dataloader=cursor,
            args=SimpleNamespace(train=SimpleNamespace(checkpoint=SimpleNamespace(load_path="checkpoint"))),
            state=SimpleNamespace(global_step=1),
        )
        with pytest.raises((ValueError, RuntimeError), match="matching data cursor|another rank"):
            BaseTrainer.on_train_begin(restoring)
        flag.fill_(1)
        dist.all_reduce(flag)
        assert flag.item() == 2

    finally:
        dist.destroy_process_group()


def test_two_rank_cursor_and_failure_consensus(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)


def _checkpoint_failure_worker(rank, rendezvous, root):
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = SimpleNamespace(
            train_dataloader=SimpleNamespace(_veomni_requires_optimizer_commit=True),
            args=SimpleNamespace(train=SimpleNamespace(global_rank=rank, checkpoint=SimpleNamespace(save_path=root))),
        )

        def checkpoint_state(_state):
            if rank == 1:
                raise ValueError("injected checkpoint state failure")
            return {"train_dataloader": {}}

        callback.state_dict = checkpoint_state
        with pytest.raises((ValueError, RuntimeError), match="checkpoint state failure|another rank"):
            callback.save_global_state(SimpleNamespace(global_step=1))
        flag = torch.tensor([1])
        dist.all_reduce(flag)
        assert flag.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not callable(getattr(global_state_callback, "_saved_multi_optimizer_generations", None)),
    reason="requires owner-aware DCP PR",
)
def test_two_rank_checkpoint_failure_consensus(tmp_path):
    mp.spawn(
        _checkpoint_failure_worker,
        args=(f"file://{tmp_path / 'checkpoint-rendezvous'}", str(tmp_path)),
        nprocs=2,
        join=True,
    )
