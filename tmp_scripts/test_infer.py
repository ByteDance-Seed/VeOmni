import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "/mnt/hdfs/veomni/models/diffusers/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# lora_dir = "/opt/tiger/exp/Wan2.1-T2V-1.3B-Diffusers_lora/checkpoints/global_step_20"
# pipe.transformer.load_lora_adapter(lora_dir, prefix="base_model.model", adapter_name="wan_lora")
# pipe.set_adapters("wan_lora", adapter_weights=1.0)  # 可调强度


prompt = "Tom, the mischievous gray cat, is sprawled out on a vibrant red pillow, his body relaxed and his eyes half-closed, as if he's just woken up or is about to doze off. His white paws are stretched out in front of him, and his tail is casually draped over the edge of the pillow. The setting appears to be a cozy corner of a room, with a warm yellow wall in the background and a hint of a wooden floor. The scene captures a rare moment of tranquility for Tom, contrasting with his usual energetic and playful demeanor."

negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    prompt=prompt, negative_prompt=negative_prompt, height=480, width=832, num_frames=81, guidance_scale=5.0
).frames[0]
export_to_video(output, "origin.mp4", fps=15)
