"""
Trainer implementation based on BaseTrainer.

This module provides a concrete implementation of BaseTrainer that follows
the training logic from train_torch.py.
"""

from functools import partial

from ..data import (
    build_chat_template,
)
from ..data.data_transform import process_pretrain_example, process_sft_example
from ..models import build_tokenizer
from ..utils import helper
from .base import BaseTrainer


logger = helper.create_logger(__name__)


class TextTrainer(BaseTrainer):
    def build_model_assets(self):
        self.tokenizer = build_tokenizer(self.args.model.tokenizer_path)
        if self.args.data.data_type == "plaintext":
            self.model_assets = [self.tokenizer]
        else:
            self.chat_template = build_chat_template(self.args.data.chat_template, self.tokenizer)
            self.model_assets = [self.chat_template]

    def build_data_transform(self):
        # Build transform function
        if self.args.data.data_type == "plaintext":
            data_transform = partial(
                process_pretrain_example,
                tokenizer=self.tokenizer,
                max_seq_len=self.args.data.max_seq_len,
                text_keys=self.args.data.text_keys,
            )
        elif self.args.data.data_type == "conversation":
            data_transform = partial(
                process_sft_example,
                chat_template=self.chat_template,
                max_seq_len=self.args.data.max_seq_len,
                text_keys=self.args.data.text_keys,
            )
        else:
            raise NotImplementedError(f"Unsupported data type: {self.args.data.data_type}.")
        return data_transform
