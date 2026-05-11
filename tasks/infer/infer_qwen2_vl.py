import json
from dataclasses import asdict, dataclass, field

import requests
from PIL import Image

from veomni.arguments import InferArguments, parse_args
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models import build_foundation_model, build_processor
from veomni.utils import helper
from veomni.utils.device import get_device_type
from veomni.utils.import_utils import is_flash_attn_2_available


logger = helper.create_logger(__name__)


@dataclass
class Arguments:
    infer: "InferArguments" = field(default_factory=InferArguments)


# Inference doesn't need fused training kernels — pin every per-op field to
# eager. ``attn_implementation`` falls back to eager on hosts without
# flash-attn so CPU / minimal-deps environments don't crash.
_INFERENCE_OPS = OpsImplementationConfig(
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


def main() -> None:
    args = parse_args(Arguments)
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    helper.set_seed(args.infer.seed)
    helper.enable_third_party_logging()
    model = (
        build_foundation_model(args.infer.model_path, args.infer.model_path, ops_implementation=_INFERENCE_OPS)
        .eval()
        .to(get_device_type())
    )
    processor = build_processor(args.infer.tokenizer_path)
    image_token_id = processor.tokenizer.encode(processor.image_token)[0]
    model.config.image_token_id = image_token_id

    processor.chat_template = (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )

    messages = [
        {"role": "user", "content": "Describe this image. <|vision_start|><|image_pad|><|vision_end|>"},
    ]
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    inputs["image_mask"] = inputs["input_ids"] == image_token_id
    inputs = inputs.to(get_device_type())
    gen_kwargs = {
        "do_sample": args.infer.do_sample,
        "temperature": args.infer.temperature,
        "top_p": args.infer.top_p,
        "max_new_tokens": args.infer.max_tokens,
    }
    generated_tokens = model.generate(**inputs, **gen_kwargs)
    response = processor.decode(generated_tokens[0, len(inputs["input_ids"][0]) :], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()
