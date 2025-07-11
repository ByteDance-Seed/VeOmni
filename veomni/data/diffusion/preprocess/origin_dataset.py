# Copyright 2023 Zhongjie Duan
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

import imageio
import numpy as np
import pandas as pd
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        metadata_path,
        max_num_frames=81,
        frame_interval=1,
        num_frames=81,
        height=480,
        width=832,
        is_i2v=False,
    ):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v

        self.frame_process = v2.Compose(
            [
                v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_frames_using_imageio(
        self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process
    ):
        reader = imageio.get_reader(file_path)
        if (
            reader.count_frames() < max_num_frames
            or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval
        ):
            reader.close()
            return None

        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()
        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames

    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(
            file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process
        )
        return frames

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
        else:
            print(f"Loading {path}...")
            video = self.load_video(path)
            if video is None:
                print("invalid video:", path)
                return None
        if self.is_i2v:
            video, first_frame = video
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
        else:
            data = {"text": text, "video": video, "path": path}
        return data

    def __len__(self):
        return len(self.path)


class TextImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False
    ):
        self.steps_per_epoch = steps_per_epoch
        metadata = pd.read_csv(os.path.join(dataset_path, "train/metadata.csv"))
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
        return {"text": text, "image": image}

    def __len__(self):
        return self.steps_per_epoch
