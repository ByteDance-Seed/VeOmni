import argparse
import json
import os
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from transformers import AutoModel, AutoProcessor

from veomni.data.multimodal.image_utils import load_image_bytes_from_path, load_image_from_bytes
from veomni.data.multimodal.video_utils import load_video_from_bytes, save_video_tensors_to_file
from veomni.models import build_foundation_model, build_processor
from veomni_patch.models.seed.seedance.condition.text import SeedanceTextEncoder, SeedanceTextEncoderProcessor
from veomni_patch.models.seed.seedance.dit import SeedanceDiT
from veomni_patch.models.seedream.data.transforms.bucket_resize import (
    BUCKET_MAPPINGS,
    BaseSRBucketResize,
)
from veomni_patch.models.seedream.dit.modules import na
from veomni_patch.models.seedream.vae.video_vae.modeling_video_vae import VideoAutoencoderKL
from veomni_patch.models.seedream.vae.video_vae.processing_video_vae import VideoVAEProcessor
from veomni.utils import helper


OmegaConf.register_new_resolver("eval", eval)

logger = helper.create_logger(__name__)


def enable_torch_deterministic_mode():
    """Configure PyTorch to ensure torch-level deterministics."""
    seed = 0
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.set_printoptions(precision=8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # torch fill deterministic doesn't support fp8 dtype. Disable it for current version (torch <= 2.4)
    # When set to True, uninitialized memory will be filled with a known value.
    # Since all the uninitialized memory will be filled with known value, this shouldn't change deterministic behavior.
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
    except AttributeError:
        # fill_uninitialized_memory is not supported in torch version < 2.4
        pass


def load_video(video_info):
    if video_info["type"] == "video_bytes":
        video_bytes = video_info["video_bytes"].encode("latin-1")
        video, video_fps, _, _ = load_video_from_bytes(video_bytes, use_audio_in_video=False)
        return video
    elif video_info["type"] == "video_url":
        raise NotImplementedError


def _move_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.to("cpu")
    elif isinstance(data, torch.nn.Module):
        return data.to("cpu")
    elif isinstance(data, dict):
        return {k: _move_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_move_to_cpu(elem) for elem in data]
    elif isinstance(data, tuple):
        return tuple(_move_to_cpu(elem) for elem in data)
    else:
        raise TypeError("Unsupported data type")


def _move_to_cuda(data):
    if isinstance(data, torch.Tensor):
        return data.to("cuda")
    elif isinstance(data, torch.nn.Module):
        return data.to("cuda")
    elif isinstance(data, dict):
        return {k: _move_to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_move_to_cuda(elem) for elem in data]
    elif isinstance(data, tuple):
        return tuple(_move_to_cuda(elem) for elem in data)
    else:
        raise TypeError("Unsupported data type")


def read_raw_data(data_path: str, negative_prompts_path: str):
    with open(negative_prompts_path, encoding="utf-8") as f:
        negative_text = f.readline().strip()
    raw_data = []

    if data_path.endswith(".jsonl"):
        with open(data_path, encoding="utf-8") as f:
            for line in f.readlines():
                data = json.loads(line)
                data["negative_prompts"] = negative_text
                raw_data.append(
                    {
                        "prompt": data["prompt"],
                        "image_bytes": data["image_bytes"], # convert to image
                        "negative_prompts": {
                            "negative_text": negative_text, 
                            "vid_negative_text": negative_text, 
                            "sr_negative_text": negative_text, 
                            "sr_vid_negative_text": negative_text,
                        }, # TODO: negative_text, vid_negative_text, sr_negative_text, sr_vid_negative_text 可不同
                    }
                )
    else:
        raise NotImplementedError(f"Not support reading data path: {data_path}")

    return raw_data

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, 
        default="/mnt/hdfs/_BYTE_DATA_SEED_/ssd_hldy/user/shizhelun/seedance/ckpt/origin/7B_0403/i2v_prompt.jsonl"
        # default="/mnt/hdfs/_BYTE_DATA_SEED_/ssd_hldy/user/shizhelun/seedance/dataset/seedance_v1_offline"
    )
    parser.add_argument(
        "--negative_prompts", type=str, 
        default="/opt/tiger/VeOmni/tasks/multimodal/seedance/configs/negative_prompt.txt"
    )
    parser.add_argument(
        "--dit_model_path", type=str,
        # default="/mnt/hdfs/_BYTE_DATA_SEED_/ssd_hldy/user/shizhelun/seedance/ckpt/veomni_hf/7B_24fps_480p_i2v_rlhf_v2_dit",
        # default="/mnt/hdfs/_BYTE_DATA_SEED_/ssd_hldy/user/shizhelun/seedance/ckpt/veomni_hf/7B_0403/i2v_dit",
        default="/mnt/hdfs/_BYTE_DATA_SEED_/ssd_hldy/user/shizhelun/seedance/ckpt/veomni_hf/7B_0403/i2v_dit_12nfe",
        
    )
    parser.add_argument("--lora_model_path", type=str, default=None)
    parser.add_argument("--use_offline_embedding", type=bool, default=False)
    parser.add_argument("--num_samples_per_prompt", type=int, default=1)
    parser.add_argument(
        "--condition_model_path", type=str,
        default="/mnt/hdfs/__MERLIN_USER_DIR__/seedance/ckpt/veomni_hf/7B_0403/condition",
    )
    args = parser.parse_args() 

    helper.enable_third_party_logging()

    enable_torch_deterministic_mode()

    raw_data_list = read_raw_data(args.data_path, args.negative_prompts) 

    dit_model_path = args.dit_model_path
    dit: SeedanceDiT = SeedanceDiT.from_pretrained(
        dit_model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    if args.lora_model_path is not None:
        from peft import PeftModel
        dit = PeftModel.from_pretrained(dit, args.lora_model_path)

    condition_model = AutoModel.from_pretrained(args.condition_model_path, torch_dtype="bfloat16") 
    condition_model.text_model.requires_grad_(False).eval().to("cuda")
    condition_model.vae_model.requires_grad_(False).eval().to("cuda")
    condition_model.vae_model.set_causal_slicing(split_size=4, memory_device="same")

    condition_processor = AutoProcessor.from_pretrained(args.condition_model_path)

    for i, raw_data in enumerate(tqdm(raw_data_list)):
        # 0. raw_data
        # 包含 prompt (image_prompt, video_promt)，image，negative_prompts（image, video）x（origin，sr）
        if "image_bytes" in raw_data:
            raw_data["image"] = load_image_from_bytes(raw_data["image_bytes"].encode("latin-1")) 
            del raw_data["image_bytes"]

        # 1. processed_data
        # 包含 clip_n_frames，neg_clip_n_frames，
        #     text_inputs（tokenize and attention mask），text_neg_inputs，text_sr_neg_inputs，
        #     image（image transform + stack），image_sr，image_index
        processed_data = condition_processor.preprocess_infer(raw_data) 

        # 2. cond_embed
        # image -> vae latent，text -> text embed
        # 包含 text_pos_embeds，text_neg_embeds，text_neg_pooled_embeds
        #     text_sr_pos_embeds，text_sr_neg_embeds，text_sr_neg_pooled_embeds，text_sr_neg_pooled_embeds
        #     image_latent，image_sr_latent，image_index，image_index
        cond_embed = condition_model.get_condition(processed_data, num_samples_per_prompt=args.num_samples_per_prompt)

        # 3. processed_cond
        # 如果是训练，就 latents 进行 sample_conditions+get_condition, 如果是 infer 就 noise get_condition
        processed_cond = condition_model.process_condition_infer(cond_embed) 

        # 4. dit generate，return unflatten latents
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            latents = dit.generate(**processed_cond, cfg_scale=3.5, ada_precompute=False)

        # 5. vae postprocess
        decoded_latents = condition_model.postprocess(latents)

        # 6. postprocess videos
        videos = condition_processor.postprocess(decoded_latents)

        os.makedirs("output", exist_ok=True)
        for j, video in enumerate(videos):
            save_video_tensors_to_file(video.cpu().numpy(), f"output/pred_{i}_{j}.mp4")


if __name__ == "__main__":
    main()
