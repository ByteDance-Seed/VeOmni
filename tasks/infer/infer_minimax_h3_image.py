import torch
from PIL import Image

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.diffusers.minimax_h3_image.inference import MiniMaxH3ImagePipeline
from veomni.utils.import_utils import is_flash_attn_2_available


OPS = OpsImplementationConfig(
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "eager",
    moe_implementation="eager",
    cross_entropy_loss_implementation="eager",
    rms_norm_implementation="eager",
    swiglu_mlp_implementation="eager",
    rotary_pos_emb_implementation="eager",
    load_balancing_loss_implementation="eager",
    rms_norm_gated_implementation="eager",
    causal_conv1d_implementation="eager",
    chunk_gated_delta_rule_implementation="eager",
)


pipe = MiniMaxH3ImagePipeline.from_pretrained(
    "pretrained_models/MiniMax-H3-Image",
    torch_dtype=torch.bfloat16,
    ops_implementation=OPS,
)

image = pipe(
    prompt="A quiet mountain lake at sunrise, photorealistic, wide angle",
    height=2048,
    width=2048,
    seed=0,
)
image.save("t2i.png")

reference = Image.open("reference.png")
image = pipe(
    prompt="<Picture 1> The same character, now standing in a snowy forest at night",
    height=2048,
    width=2048,
    references=[reference],
    seed=0,
)
image.save("edit.png")
