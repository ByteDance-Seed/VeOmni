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

from unittest.mock import patch

from transformers import AutoConfig

from veomni.lora.config import FUSED_MOE_LORA_MODULES, LORA_MODULES_BY_MODEL_TYPE, VeOmniLoraConfig
from veomni.utils.count_flops import VeomniFlopsCounter


BATCH_SEQLENS = [2048, 2048, 2048, 2048]
IMAGES_SEQLENS = [256, 256, 256, 256]
DELTA_TIME = 1.0
LORA_RANKS = [8, 64, 256, 1024, 4096]

MODELS = [
    ("Qwen2 0.5B", "Qwen/Qwen2-0.5B", "qwen2", "dense GQA"),
    ("Qwen2 72B", "Qwen/Qwen2-72B", "qwen2", "dense GQA"),
    ("Qwen3 0.6B", "Qwen/Qwen3-0.6B", "qwen3", "dense GQA + QK norm"),
    ("Qwen3 32B", "Qwen/Qwen3-32B", "qwen3", "dense GQA + QK norm"),
    ("Qwen3 30B-A3B", "Qwen/Qwen3-30B-A3B", "qwen3_moe", "MoE; 3B active"),
    ("Qwen3-Next 80B-A3B", "Qwen/Qwen3-Next-80B-A3B-Instruct", "qwen3_next", "hybrid MoE; 3B active"),
    ("Qwen2.5-VL 3B", "Qwen/Qwen2.5-VL-3B-Instruct", "qwen2_5_vl", "dense VLM"),
    ("Qwen3-VL 2B", "Qwen/Qwen3-VL-2B-Instruct", "qwen3_vl", "dense VLM"),
    ("Qwen3.5 9B", "Qwen/Qwen3.5-9B", "qwen3_5", "hybrid dense VLM"),
    ("Qwen3.5 27B", "Qwen/Qwen3.5-27B", "qwen3_5", "hybrid dense VLM"),
    ("Qwen3.5 35B-A3B", "Qwen/Qwen3.5-35B-A3B", "qwen3_5_moe", "hybrid MoE VLM; 3B active"),
]

ROUTED_MOE_TYPES = {"qwen3_moe", "qwen3_vl_moe", "qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text"}
SHARED_EXPERT_TYPES = {"qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text"}
ROUTED_EXPERT_TARGETS = ["*.mlp.experts.gate_up_proj", "*.mlp.experts.down_proj"]


def get_lora_configs(model_type: str, rank: int) -> dict[str, VeOmniLoraConfig]:
    supported_modules = LORA_MODULES_BY_MODEL_TYPE[model_type]
    mlp_modules = [module for module in supported_modules if module in FUSED_MOE_LORA_MODULES]
    attention_modules = [module for module in supported_modules if module not in FUSED_MOE_LORA_MODULES]
    target_parameters = ROUTED_EXPERT_TARGETS if model_type in ROUTED_MOE_TYPES else None
    shared_mlp_modules = mlp_modules if model_type in SHARED_EXPERT_TYPES or target_parameters is None else None

    def make_config(target_modules, routed_targets=None):
        return VeOmniLoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=target_modules,
            target_parameters=routed_targets,
            moe_mode="independent" if routed_targets else None,
        )

    return {
        "LoRA all": make_config([*attention_modules, *(shared_mlp_modules or [])], target_parameters),
        "LoRA attention": make_config(attention_modules),
        "LoRA MLP": make_config(shared_mlp_modules, target_parameters),
    }


def compare_model(model_name: str, model_id: str, model_type: str, architecture: str) -> None:
    # AutoConfig downloads only the small config metadata, never model weights.
    counter = VeomniFlopsCounter(AutoConfig.from_pretrained(model_id))
    input_kwargs = {"images_seqlens": IMAGES_SEQLENS} if getattr(counter.config, "vision_config", None) else {}
    full_flops, _ = counter.estimate_flops(BATCH_SEQLENS, DELTA_TIME, **input_kwargs)

    print(f"\n{model_name} ({model_id}; {architecture})")
    print("=" * 78)
    print(f"{'Mode':<24} {'Rank':>6} {'TFLOPs/step':>16} {'vs full':>12}")
    print("-" * 78)
    print(f"{'Full fine-tuning':<24} {'-':>6} {full_flops:>16.6f} {100:>11.2f}%")

    for rank in LORA_RANKS:
        for label, lora_config in get_lora_configs(model_type, rank).items():
            flops, _ = counter.estimate_flops(
                BATCH_SEQLENS,
                DELTA_TIME,
                lora_config=lora_config,
                **input_kwargs,
            )
            print(f"{label:<24} {rank:>6} {flops:>16.6f} {flops / full_flops * 100:>11.2f}%")


def main() -> None:
    # Only achieved FLOPs are compared, so no physical accelerator is required.
    with patch("veomni.utils.count_flops.get_device_flops", return_value=1.0):
        for model in MODELS:
            compare_model(*model)


if __name__ == "__main__":
    main()
