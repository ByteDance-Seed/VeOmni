"""Sanity-check the DiffusionGemma training objective against the real checkpoint.

A single loss number from a random sample is not interpretable, and on this model it is
actively misleading: the instruct checkpoint opens its answers with a
``<|channel>thought`` block, so an SFT target written without one is misaligned by a few
tokens and scores *worse* than the uniform baseline even though nothing is wrong.

This script separates the two questions:

1. Is the model wired up correctly? Denoise a fully random canvas in one step and read
   what it predicts. Coherent text proves weight loading, encoder/decoder weight tying,
   the KV-cache handoff and the attention masks are all correct.
2. Is the training objective correct? Score a target the model actually agrees with
   (its own greedy output) across corruption rates. The loss must sit far below the
   uniform baseline throughout, which shows the canvas, labels and loss weights line up
   position for position. The curve is often flat rather than rising with corruption:
   on a prompt the model is confident about, it reconstructs the answer from the prompt
   alone and barely reads the canvas.

Usage:
    python -m scripts.diagnose_diffusion_gemma_loss <checkpoint_path>
"""

import argparse
import math

import torch

from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion
from veomni.utils.constants import IGNORE_INDEX


def build_model(checkpoint_path: str, device: str, load_device: str = "cpu"):
    from scripts.verify_diffusion_gemma import _move_shared_model
    from veomni.models.checkpoint.weights import init_empty_weights, load_model_weights

    config = DiffusionGemmaConfig.from_pretrained(checkpoint_path)
    config.dtype = torch.bfloat16
    config._attn_implementation = "sdpa"
    # A training checkpoint round-trips `attn_ffpa: true`, and an unhonourable request now
    # refuses to build. This script only reads losses, so keep its optional package optional.
    config.text_config.attn_ffpa = False
    with init_empty_weights():
        model = DiffusionGemmaForBlockDiffusion(config)
    load_model_weights(model, checkpoint_path, init_device=load_device)
    model = model.to(torch.bfloat16).eval()
    if device != load_device:
        _move_shared_model(model, device)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_path")
    parser.add_argument("--question", default="What is the capital of France?")
    parser.add_argument("--answer-length", type=int, default=24, help="canvas positions to score")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    tokenizer = processor.tokenizer
    model = build_model(args.checkpoint_path, args.device)
    config = model.config
    vocab_size = config.text_config.vocab_size
    canvas_length = config.canvas_length
    uniform_baseline = math.log(vocab_size)
    device = args.device

    prompt_ids = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": args.question}]}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"].to(device)
    attention_mask = torch.ones_like(prompt_ids)
    decoder_attention_mask = torch.ones(1, prompt_ids.shape[1] + canvas_length, dtype=torch.bool, device=device)

    def forward(canvas, labels=None, loss_weight=None):
        with torch.no_grad():
            return model(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                decoder_input_ids=canvas,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                loss_weight=loss_weight,
                use_cache=False,
            )

    print(f"question: {args.question!r}")
    print(f"uniform baseline ln({vocab_size}) = {uniform_baseline:.3f}\n")

    print("--- 1. one-step denoise of a fully random canvas ---")
    torch.manual_seed(0)
    random_canvas = torch.randint(0, vocab_size, (1, canvas_length), device=device)
    logits = forward(random_canvas).logits
    prediction = logits[0].argmax(dim=-1)
    n = args.answer_length
    print(f"predicted: {tokenizer.decode(prediction[:n])!r}")
    print("(coherent text here means the checkpoint is loaded and wired correctly)\n")

    print("--- 2. loss vs corruption, scored against a target the model agrees with ---")
    target = prediction[:n].clone()
    labels = torch.full((1, canvas_length), IGNORE_INDEX, dtype=torch.long, device=device)
    labels[0, :n] = target
    loss_weight = torch.zeros(1, canvas_length, device=device)
    loss_weight[0, :n] = 1.0 / n

    print(f"{'corruption':>12} {'loss':>10} {'vs uniform':>12}")
    losses = []
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        torch.manual_seed(0)
        canvas = torch.full((1, canvas_length), config.text_config.pad_token_id or 0, dtype=torch.long, device=device)
        canvas[0, :n] = target
        corrupt = torch.rand(1, canvas_length, device=device) < rate
        noise = torch.randint(0, vocab_size, (1, canvas_length), device=device)
        canvas = torch.where(corrupt, noise, canvas)

        loss = forward(canvas, labels, loss_weight).loss.item()
        losses.append(loss)
        print(f"{rate:>11.0%} {loss:>10.3f} {loss - uniform_baseline:>+12.3f}")

    worst = max(losses)
    ok = worst < uniform_baseline / 2
    print(f"\nworst loss {worst:.3f} vs uniform baseline {uniform_baseline:.3f}")
    print("OBJECTIVE: OK" if ok else "OBJECTIVE: SUSPECT (target and canvas are probably misaligned)")


if __name__ == "__main__":
    main()
