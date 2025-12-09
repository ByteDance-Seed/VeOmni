# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import List

import torch
from PIL.Image import Image
from transformers import BatchFeature, LlamaTokenizerFast, ProcessorMixin

from .image_processing_janus import JanusImageProcessor


class JanusProcessor(ProcessorMixin):
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["image_processor", "tokenizer"]

    system_prompt = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )
    valid_kwargs = ["image_tag", "num_image_tokens", "add_special_token"]

    def __init__(
        self,
        image_processor: JanusImageProcessor,
        tokenizer: LlamaTokenizerFast,
        image_tag: str = "<image_placeholder>",
        num_image_tokens: int = 576,
        add_special_token: bool = False,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        image_id = self.tokenizer.vocab.get(image_tag)
        if image_id is None:
            special_tokens = [image_tag]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Add image tag = {image_tag} to the tokenizer")

        self.image_tag = image_tag
        self.image_start_tag = "<begin_of_image>"
        self.image_end_tag = "<end_of_image>"
        self.pad_tag = "<｜▁pad▁｜>"

        self.num_image_tokens = num_image_tokens
        self.add_special_token = add_special_token
        self.seps = ["\n\n", "<｜end▁of▁sentence｜>"]
        super().__init__(image_processor, tokenizer)

    def apply_chat_template(self, conversation, task="gen"):
        ret = ""
        if task != "gen":
            ret += self.system_prompt + self.seps[0]

        for i, message in enumerate(conversation):
            role = message["role"]
            content = message["content"]
            if content:
                ret += role + ": " + content + self.seps[i % 2]
            else:
                ret += role + ":"
        if task == "gen":
            ret += self.image_start_tag
        return ret

    @property
    def image_token(self):
        return self.image_tag

    @property
    def image_id(self):
        image_id = self.tokenizer.vocab.get(self.image_tag)
        return image_id

    @property
    def image_start_id(self):
        image_start_id = self.tokenizer.vocab.get(self.image_start_tag)
        return image_start_id

    @property
    def image_end_id(self):
        image_end_id = self.tokenizer.vocab.get(self.image_end_tag)
        return image_end_id

    @property
    def image_start_token(self):
        return self.image_start_tag

    @property
    def image_end_token(self):
        return self.image_end_tag

    @property
    def pad_id(self):
        pad_id = self.tokenizer.vocab.get(self.pad_tag)
        return pad_id

    def add_image_token(
        self,
        image_indices: List[int],
        input_ids: torch.LongTensor,
    ):
        """

        Args:
            image_indices (List[int]): [index_0, index_1, ..., index_j]
            input_ids (torch.LongTensor): [N]

        Returns:
            input_ids (torch.LongTensor): [N + image tokens]
            num_image_tokens (torch.IntTensor): [n_images]
        """

        input_slices = []

        start = 0
        for index in image_indices:
            if self.add_special_token:
                end = index + 1
            else:
                end = index

            # original text tokens
            input_slices.append(input_ids[start:end])

            # add boi, image tokens, eoi and set the mask as False
            input_slices.append(self.image_start_id * torch.ones((1), dtype=torch.long))
            input_slices.append(self.image_id * torch.ones((self.num_image_tokens,), dtype=torch.long))
            input_slices.append(self.image_end_id * torch.ones((1), dtype=torch.long))
            start = index + 1

        # the left part
        input_slices.append(input_ids[start:])

        # concat all slices
        input_ids = torch.cat(input_slices, dim=0)
        num_image_tokens = torch.IntTensor([self.num_image_tokens] * len(image_indices))

        return input_ids, num_image_tokens

    def __call__(
        self,
        prompt: str = None,
        images: List[Image] = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        # tokenize
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        # add image tokens to the input_ids
        image_token_mask: torch.BoolTensor = input_ids == self.image_id
        image_indices = image_token_mask.nonzero()
        input_ids, num_image_tokens = self.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )
        input_ids = input_ids.unsqueeze(0)
        image_mask = input_ids == self.image_id
        attention_mask = torch.ones_like(input_ids)
        # load images
        images_outputs = self.image_processor(images, return_tensors="pt")

        return BatchFeature(
            data={
                "input_ids": input_ids,
                "pixel_values": images_outputs.pixel_values,
                "image_mask": image_mask,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )
