"""Generate text with DiffusionGemma by block-autoregressive denoising.

A thin CLI over `veomni.models.transformers.diffusion_gemma.generation`, which holds the
sampler itself. Fluent output here is strong evidence that weight loading, the shared
encoder/decoder weights, the KV-cache handoff and the attention masks are all wired
correctly -- none of which a loss value alone can demonstrate.

Usage:
    python -m scripts.generate_diffusion_gemma <checkpoint_path> --prompt "..."
    python -m scripts.generate_diffusion_gemma <checkpoint_path> --image cat.jpg
"""

import argparse

import torch

from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
from veomni.models.transformers.diffusion_gemma.generation import (
    EntropyBoundedSamplerConfig,
    block_autoregressive_generate,
)
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion


#: Gemma chat templates close the model's turn with this rather than with `eos_token`.
END_OF_TURN_TOKEN = "<end_of_turn>"


def build_model(checkpoint_path: str, device: str, load_device: str = "cpu"):
    from scripts.verify_diffusion_gemma import _move_shared_model
    from veomni.models.checkpoint.weights import init_empty_weights, load_model_weights

    config = DiffusionGemmaConfig.from_pretrained(checkpoint_path)
    config.dtype = torch.bfloat16
    config._attn_implementation = "sdpa"
    # A training checkpoint round-trips `attn_ffpa: true`, and an unhonourable request now
    # refuses to build. Sampling has no use for the fast path anyway -- a 256-long canvas is
    # under the kernel's 512-query floor -- so don't make its optional package a hard one.
    config.text_config.attn_ffpa = False
    with init_empty_weights():
        model = DiffusionGemmaForBlockDiffusion(config)
    load_model_weights(model, checkpoint_path, init_device=load_device)
    model = model.to(torch.bfloat16).eval()
    if device != load_device:
        _move_shared_model(model, device)
    return model


def resolve_end_of_turn_ids(tokenizer) -> list[int]:
    """Token ids that end the model's turn, so generation can stop at a canvas boundary."""
    candidates = {tokenizer.convert_tokens_to_ids(END_OF_TURN_TOKEN), tokenizer.eos_token_id}
    return sorted(i for i in candidates if i is not None and i != tokenizer.unk_token_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_path")
    parser.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    parser.add_argument("--image", default=None, help="optional image path for the text+image path")
    parser.add_argument("--max-blocks", type=int, default=4, help="canvases to generate before giving up on an EOT")
    parser.add_argument("--num-steps", type=int, default=48, help="denoising-step ceiling per canvas")
    parser.add_argument("--entropy-budget", type=float, default=0.1, help="nats a single step may commit")
    parser.add_argument("--stop-threshold", type=float, default=0.005, help="mean entropy that ends a canvas")
    parser.add_argument("--tau-max", type=float, default=0.8, help="temperature at the first, fully noised step")
    parser.add_argument("--tau-min", type=float, default=0.4, help="temperature at the last step")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    tokenizer = processor.tokenizer
    model = build_model(args.checkpoint_path, args.device)

    content = []
    if args.image:
        from PIL import Image

        content.append({"type": "image", "image": Image.open(args.image).convert("RGB")})
    content.append({"type": "text", "text": args.prompt})
    encoded = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_inputs = {k: v.to(args.device) for k, v in encoded.items() if k != "token_type_ids"}

    sampler = EntropyBoundedSamplerConfig(
        num_steps=args.num_steps,
        entropy_budget=args.entropy_budget,
        stop_threshold=args.stop_threshold,
        temperature_max=args.tau_max,
        temperature_min=args.tau_min,
    )
    canvas_index = -1

    def report(step: int, prediction: torch.Tensor, entropies: torch.Tensor) -> None:
        nonlocal canvas_index
        if step == 0:
            canvas_index += 1
        elif step % max(1, args.num_steps // 4):
            return
        preview = tokenizer.decode(prediction[0], skip_special_tokens=True)
        print(f"  canvas {canvas_index} step {step:>3} mean-entropy {entropies.mean():.4f} | {preview[:96]!r}")

    print(f"prompt: {args.prompt!r}" + (f" + image {args.image}" if args.image else ""))
    print(f"up to {args.max_blocks} canvases of {model.config.canvas_length}, {args.num_steps} steps each\n")
    generated = block_autoregressive_generate(
        model,
        prompt_inputs,
        eos_token_ids=resolve_end_of_turn_ids(tokenizer),
        config=sampler,
        max_blocks=args.max_blocks,
        on_step=report,
    )
    print(f"\n--- {generated.shape[-1]} generated tokens ---")
    print(tokenizer.decode(generated[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
