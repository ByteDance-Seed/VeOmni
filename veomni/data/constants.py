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


IGNORE_INDEX = -100

# input index
IMAGE_INPUT_INDEX = -200
VIDEO_INPUT_INDEX = -300
AUDIO_INPUT_INDEX = -400
# output index
IMAGE_OUTPUT_INDEX = -201
VIDEO_OUTPUT_INDEX = -301
AUDIO_OUTPUT_INDEX = -401


TYPE2INDEX = {
    "input": {
        "image": IMAGE_INPUT_INDEX,
        "video": VIDEO_INPUT_INDEX,
        "audio": AUDIO_INPUT_INDEX,
    },
    "output": {
        "image": IMAGE_OUTPUT_INDEX,
        "video": VIDEO_OUTPUT_INDEX,
        "audio": AUDIO_OUTPUT_INDEX,
    },
}
