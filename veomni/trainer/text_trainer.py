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

from functools import partial

from ..data import (
    build_chat_template,
)
from ..data.data_transform import process_pretrain_example, process_sft_example
from ..models import build_tokenizer
from ..utils import helper
from .base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)


class TextTrainer(BaseTrainer):
    def build_model_assets(self):
        args: VeOmniArguments = self.args
        self.tokenizer = build_tokenizer(args.model.tokenizer_path)
        if args.data.data_type == "plaintext":
            return [self.tokenizer]
        else:
            self.chat_template = build_chat_template(args.data.chat_template, self.tokenizer)
            return [self.chat_template]

    def build_data_transform(self):
        args: VeOmniArguments = self.args
        if args.data.data_type == "plaintext":
            data_transform = partial(
                process_pretrain_example,
                tokenizer=self.tokenizer,
                max_seq_len=args.data.max_seq_len,
                text_keys=args.data.text_keys,
            )
        elif args.data.data_type == "conversation":
            data_transform = partial(
                process_sft_example,
                chat_template=self.chat_template,
                max_seq_len=args.data.max_seq_len,
                text_keys=args.data.text_keys,
            )
        else:
            raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")
        return data_transform
