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

"""Train a DeepSpec speculative-decoding draft model on VeOmni.

Usage:
    bash train.sh tasks/train_deepspec_draft.py configs/deepspec/dspark_qwen3_4b.yaml

The draft model, its target-cache dataset, and the DeepSpec loss are wired in by
``veomni.trainer.deepspec.DraftModelTrainer``. See
``veomni/integrations/deepspec/`` for the bridge and
``scripts/deepspec/prepare_draft_init.py`` for building the draft init checkpoint
and config the run consumes.
"""

from veomni.arguments import VeOmniArguments, parse_args
from veomni.trainer.deepspec import DraftModelTrainer


if __name__ == "__main__":
    args = parse_args(VeOmniArguments)
    trainer = DraftModelTrainer(args)
    trainer.train()
