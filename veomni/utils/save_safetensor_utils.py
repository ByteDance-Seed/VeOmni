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
from typing import Optional, Sequence

from veomni.checkpoint import ckpt_to_state_dict
from veomni.models import save_model_weights
from veomni.utils import helper

logger = helper.create_logger(__name__)


def save_hf_safetensor(
    save_checkpoint_path: str,
    model_assets: Optional[Sequence],
    ckpt_manager: str,
    train_architecture: Optional[str],
    output_dir: Optional[str] = None,
):
    """Save model weights in HuggingFace safetensors format.

    Args:
        save_checkpoint_path: Path to the distributed checkpoint.
        model_assets: Model assets (e.g., config, tokenizer) to save alongside weights.
        ckpt_manager: Checkpoint manager type.
        train_architecture: Training architecture type. If "lora", only LoRA weights are saved.
        output_dir: Output directory for checkpoint conversion. Required only by omnistore ckpt_manager.
    """
    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=save_checkpoint_path,
        ckpt_manager=ckpt_manager,
        output_dir=output_dir,
    )
    if train_architecture == "lora":
        model_state_dict = {k: v for k, v in model_state_dict.items() if "lora" in k}
    save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
    logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
