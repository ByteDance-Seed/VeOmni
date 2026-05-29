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

"""Dummy-data training entrypoint for Qwen3-VL-MoE under Ulysses sp>1.

VLMTrainer holds ``self.base = BaseTrainer.__new__(BaseTrainer)`` and explicitly
calls ``self.base._build_dataset()``, so a subclass method override would never
intercept dataset construction. This entrypoint instead inlines VLMTrainer.__init__
and substitutes ``build_dummy_dataset`` at the dataset stage.
"""

from veomni.arguments import parse_args
from veomni.data.dummy_dataset import build_dummy_dataset
from veomni.trainer.base import BaseTrainer
from veomni.trainer.vlm_trainer import VeOmniVLMArguments, VLMTrainer


class DummyVLMTrainer(VLMTrainer):
    def __init__(self, args: VeOmniVLMArguments):
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args
        self.base._setup()
        self._build_model()
        self._freeze_model_module()
        self._build_model_assets()
        self._build_data_transform()

        # DummyQwenVLDataset (task_type='qwen3vl') already emits the MainCollator
        # tensor-dict layout, so no MappingDataset wrapper is needed.
        self.base.train_dataset = build_dummy_dataset(
            task_type="qwen3vl",
            size=max(args.train.max_steps * args.train.global_batch_size, 1),
            max_seq_len=args.data.max_seq_len,
        )
        # Mirror BaseTrainer._build_dataset bookkeeping.
        dataset_length = len(self.base.train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.train.accelerator.dp_size
        args.compute_train_steps(dataset_length)
        self.base.train_steps = args.train_steps

        self._build_collate_fn()
        self.base._build_dataloader()
        self.base._build_parallelized_model()
        self._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()


if __name__ == "__main__":
    args = parse_args(VeOmniVLMArguments)
    trainer = DummyVLMTrainer(args)
    trainer.train()
