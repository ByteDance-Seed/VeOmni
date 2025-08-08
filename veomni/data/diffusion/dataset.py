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


import os

import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from veomni.utils import logging


logger = logging.get_logger(__name__)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, datasets_repeat=1):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        logger.info_rank0(f"{len(self.path)} videos in metadata.")
        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        logger.info_rank0(f"{len(self.path)} tensors cached in metadata.")
        assert len(self.path) > 0
        self.datasets_repeat = datasets_repeat

    def __getitem__(self, index):
        data_id = (index) % len(self.path)  # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        data["latents"] = data["latents"].squeeze(0)
        return [data]

    def __len__(self):
        return len(self.path) * self.datasets_repeat


class Text2ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        metadata_path,
        height=1024,
        width=1024,
        center_crop=True,
        random_flip=False,
        datasets_repeat=1,
    ):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(dataset_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        self.height = height
        self.width = width
        self.image_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        logger.info_rank0(f"{len(self.path)} tensors cached in metadata.")
        assert len(self.path) > 0
        self.datasets_repeat = datasets_repeat

    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path)  # For fixed seed.
        text = self.text[data_id]
        image = Image.open(self.path[data_id]).convert("RGB")
        target_height, target_width = self.height, self.width
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        shape = [round(height * scale), round(width * scale)]
        image = torchvision.transforms.functional.resize(
            image, shape, interpolation=transforms.InterpolationMode.BILINEAR
        )
        image = self.image_processor(image)
        return [{"text": text, "image": image}]

    def __len__(self):
        return len(self.path) * self.datasets_repeat


def build_tensor_dataset(base_path, metadata_path, datasets_repeat=1):
    return TensorDataset(base_path, metadata_path, datasets_repeat)


def build_text_image_dataset(base_path, metadata_path, height, width, center_crop, random_flip, datasets_repeat=1):
    return Text2ImageDataset(base_path, metadata_path, height, width, center_crop, random_flip, datasets_repeat)
