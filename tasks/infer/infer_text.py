import json
import readline  # noqa: F401
from dataclasses import asdict, dataclass, field

import torch
from transformers import AutoTokenizer, TextStreamer

from veomni.arguments import InferArguments, parse_args
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models import build_foundation_model
from veomni.utils import helper
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
    model = build_foundation_model(
        config_path=args.infer.model_path,
        weights_path=args.infer.model_path,
        ops_implementation=_INFERENCE_OPS,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.infer.tokenizer_path, padding_side="left")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    logger.info("tips: use `clear` to remove the history, use `exit` to exit the conversation.")

    messages = []
    while True:
        query = input("\nUser: ")

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = input_ids.to(model.device)
        gen_kwargs = {
            "do_sample": args.infer.do_sample,
            "temperature": args.infer.temperature,
            "top_p": args.infer.top_p,
            "max_new_tokens": args.infer.max_tokens,
            "streamer": streamer,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
        print("Assistant: ", end="", flush=True)
        generated_tokens = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **gen_kwargs)
        response = tokenizer.decode(generated_tokens[0, len(input_ids[0]) :], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
