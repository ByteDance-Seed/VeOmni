import torch
from PIL import Image

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.data.multimodal.video_utils import write_video_audio
from veomni.models.diffusers.minimax_h3.inference import MiniMaxH3Pipeline
from veomni.utils.device import get_device_type


device = get_device_type()
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    condition_model_path="pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA",
    condition_model_cfg={
        "base_model_path": "pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA",
        "use_keyframe_condition": True,
        "keyframe_indices": [0, -1],
    },
    transformer_config_path="pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA/transformer/config.json",
    transformer_weights_path="pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA/transformer",
    ops_implementation=OpsImplementationConfig(
        attn_implementation="eager",
        rotary_pos_emb_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        cross_entropy_loss_implementation="eager",
        moe_implementation="eager",
        load_balancing_loss_implementation="eager",
    ),
)

# Text -> Video + Audio
prompt = "A girl is very happy, she is speaking in english: “I enjoy working with VeOmni, it's a perfect framework.”"
video, audio = pipe(
    prompt=prompt,
    height=480,
    width=832,
    num_frames=124,
    num_inference_steps=50,
    seed=0,
)
write_video_audio(
    video=video,
    audio=audio,
    output_path="t2va.mp4",
    fps=24,
    audio_sample_rate=32000,
)

# Text + First Frame + Last Frame -> Video + Audio
# Keyframes are loaded from the local dataset copy.
first_frame = Image.open("dataset/minimax-h3-demo/minimax_h3/MiniMax-H3-FL2VA/first.png")
last_frame = Image.open("dataset/minimax-h3-demo/minimax_h3/MiniMax-H3-FL2VA/last.png")
prompt = "室内家庭争吵短剧场景，竖屏短剧质感，真实真人表演，中式家庭/小饭馆室内环境，暖色灯光，背景有红色装饰和书法字幅，浅景深，情绪强烈，剪辑节奏紧凑。表演要求：真实短剧表演风格，不要夸张舞台腔。男人的语气是愤怒、委屈、急切的反驳，他说“你到底想干什么？”；中老年女性的语气是尖锐、强势、咄咄逼人的质问，她说“你必须赔钱！”。两人之间有强烈对峙感，节奏逐步升级。画面风格：竖屏9:16，手机短剧质感，真人实拍感，浅景深，室内暖光，中近景为主，频繁正反打剪辑，背景保持生活化，不要科幻、不要古装、不要动画感。画面中不要出现任何字幕、文字、平台水印或贴片。 "
video, audio = pipe(
    prompt=prompt,
    height=832,
    width=480,
    num_frames=124,
    num_inference_steps=50,
    seed=0,
    keyframes=[first_frame, last_frame],
    keyframe_indices=[0, -1],
)
write_video_audio(
    video=video,
    audio=audio,
    output_path="fl2va.mp4",
    fps=24,
    audio_sample_rate=32000,
)
