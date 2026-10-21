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
"""Verify the real DiffusionGemma checkpoint against the vendored implementation.

Two stages, because the first one is cheap and catches most of what can go wrong:

1. ``--layout-only``: build the model on the meta device and diff its deduplicated
   parameter/buffer names and shapes against the checkpoint index plus the safetensors
   headers. No tensor data is read, so this runs in seconds without a GPU and works
   even with shards missing. This is what proves the three risky claims: the encoder
   contributes nothing but ``layer_scalar``, the fused expert tensors already match
   VeOmni's ``[E, 2I, H]`` / ``[E, H, I]`` layout (so no ``CheckpointTensorConverter``
   is needed), and the 25 (not 30) ``v_proj`` tensors line up with the KV-sharing
   layer types.

2. Full run (default): additionally load the weights in bf16 on one GPU through
   VeOmni's own loader and run a forward pass, checking that encoder and decoder still
   share storage afterwards and that the logits are finite.

Usage:
    python scripts/verify_diffusion_gemma.py /path/to/diffusiongemma-26B-A4B-it
    python scripts/verify_diffusion_gemma.py /path/to/ckpt --layout-only
"""

import argparse
import json
import os
import struct

import torch

from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion


def _safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(header_size))


def _build_meta_model(checkpoint_path: str, text_only: bool = False) -> DiffusionGemmaForBlockDiffusion:
    config = DiffusionGemmaConfig.from_pretrained(checkpoint_path)
    if text_only:
        config.vision_config = None  # drops the vision tower entirely
    # Inspecting the layout says nothing about attention kernels, and `attn_ffpa` arrives
    # from the checkpoint, so don't let it turn an optional package into a hard one here.
    config.text_config.attn_ffpa = False
    with torch.device("meta"):
        return DiffusionGemmaForBlockDiffusion(config)


def check_layout(checkpoint_path: str, text_only: bool = False) -> bool:
    model = _build_meta_model(checkpoint_path, text_only)
    config = model.config

    model_shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    buffer_shapes = {n: tuple(b.shape) for n, b in model.named_buffers()}
    model_shapes.update(buffer_shapes)
    # Non-persistent buffers are recomputed at init and never stored in a checkpoint.
    # VeOmni's loader carries them across `to_empty` in its `buffer_dict`.
    non_persistent = set(buffer_shapes) - set(model.state_dict())

    with open(os.path.join(checkpoint_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    checkpoint_shapes, missing_shards = {}, []
    for shard in sorted(set(weight_map.values())):
        shard_path = os.path.join(checkpoint_path, shard)
        if not os.path.exists(shard_path):
            missing_shards.append(shard)
            continue
        for name, meta in _safetensors_header(shard_path).items():
            if name != "__metadata__":
                checkpoint_shapes[name] = tuple(meta["shape"])

    print(f"model: {len(model_shapes) - len(buffer_shapes)} parameters + {len(buffer_shapes)} buffers")
    print(f"checkpoint: {len(weight_map)} tensors in {len(set(weight_map.values()))} shards")
    if missing_shards:
        print(f"WARNING: shards not on disk, their shapes are unchecked: {missing_shards}")

    # In text-only mode the vision tensors are legitimately unclaimed; otherwise nothing is.
    unexpected = sorted(k for k in set(weight_map) - set(model_shapes) if not (text_only and "vision" in k))
    # Only non-persistent buffers may be in the model but not in the checkpoint.
    uncovered = sorted(set(model_shapes) - set(weight_map) - non_persistent)
    mismatched = {
        name: (model_shapes[name], shape)
        for name, shape in checkpoint_shapes.items()
        if name in model_shapes and model_shapes[name] != shape
    }

    print(f"\nnon-vision checkpoint keys the model cannot accept: {len(unexpected)} {unexpected[:5]}")
    print(f"model tensors no checkpoint key feeds: {len(uncovered)} {uncovered[:5]}")
    print(f"shape mismatches: {len(mismatched)} {list(mismatched.items())[:3]}")
    print(f"non-persistent buffers (rebuilt at init, correctly absent): {len(non_persistent)}")

    num_layers = config.text_config.num_hidden_layers
    encoder_keys = [k for k in weight_map if k.startswith("model.encoder.") and "vision" not in k]
    encoder_is_scalar_only = all(k.endswith("layer_scalar") for k in encoder_keys)
    fused_experts = [k for k in weight_map if k.endswith("experts.gate_up_proj")]
    num_v_proj = len([k for k in weight_map if k.endswith("v_proj.weight") and "vision" not in k])
    num_v_proj_model = len([k for k in model_shapes if k.endswith("v_proj.weight")])

    print("\n--- the three claims the integration rests on ---")
    print(f"encoder stores only layer_scalar ({len(encoder_keys)} text keys): {encoder_is_scalar_only}")
    print(f"fused expert tensors present for every layer: {len(fused_experts)}/{num_layers}")
    example = "model.decoder.layers.0.experts.gate_up_proj"
    print(f"   {example}: model {model_shapes.get(example)} vs checkpoint {checkpoint_shapes.get(example)}")
    print(f"v_proj tensors, KV sharing: checkpoint {num_v_proj} == model {num_v_proj_model} (< {num_layers} layers)")

    ok = (
        not unexpected
        and not uncovered
        and not mismatched
        and encoder_is_scalar_only
        and len(fused_experts) == num_layers
        and num_v_proj == num_v_proj_model
    )
    print(f"\nLAYOUT: {'OK' if ok else 'MISMATCH'}")
    return ok


def _move_shared_model(model: torch.nn.Module, device: str) -> None:
    """Move the model without going through `Module._apply`.

    `.to()` installs a *new* `Parameter` per owning module whenever the source and
    destination are not shallow-copy compatible (CPU vs CUDA). For a model whose encoder
    and decoder share every transformer weight, that both doubles peak memory and
    silently unties them -- the same mechanism that makes `to_empty` need
    `retie_shared_weights`. Assigning `.data` in place keeps the one shared object, so
    the 49GB checkpoint stays 49GB instead of 98GB.
    """
    for param in model.parameters():  # deduplicated, so shared tensors move once
        param.data = param.data.to(device)
    for module in model.modules():
        for name, buffer in module._buffers.items():
            if buffer is not None:
                module._buffers[name] = buffer.to(device)


def check_load_and_forward(
    checkpoint_path: str, text_only: bool = False, device: str = "cuda", load_device: str = "cpu"
) -> bool:
    from veomni.models.checkpoint.weights import init_empty_weights, load_model_weights

    config = DiffusionGemmaConfig.from_pretrained(checkpoint_path)
    if text_only:
        config.vision_config = None
    config.dtype = torch.bfloat16
    config._attn_implementation = "sdpa"
    # A training checkpoint round-trips `attn_ffpa: true`, and an unhonourable request now
    # refuses to build. This check's prompts are far under the kernel's 512-query floor, so
    # requiring its out-of-band package here would buy nothing.
    config.text_config.attn_ffpa = False
    with init_empty_weights():
        model = DiffusionGemmaForBlockDiffusion(config)
    # Loading onto the CPU first: `to_empty` transiently needs ~2x the checkpoint size
    # because the shared encoder weights have not been retied yet, which does not fit on
    # one GPU. Real training does not hit this -- FSDP2 shards the parameters per rank.
    load_model_weights(model, checkpoint_path, init_device=load_device)
    model = model.to(torch.bfloat16).eval()
    if device != load_device:
        _move_shared_model(model, device)

    assert not any(p.is_meta for p in model.parameters()), "parameters left on the meta device"

    encoder_layer = model.model.encoder.language_model.layers[0]
    decoder_layer = model.model.decoder.layers[0]
    tied = (
        encoder_layer.experts.gate_up_proj is decoder_layer.experts.gate_up_proj
        and encoder_layer.self_attn.q_proj.weight is decoder_layer.self_attn.q_proj.weight
        and model.lm_head.weight is model.model.decoder.embed_tokens.weight
    )
    private = encoder_layer.layer_scalar is not decoder_layer.layer_scalar
    print(f"encoder/decoder share transformer weights after loading: {tied}")
    print(f"layer_scalar stays per-side: {private}")

    canvas_length = config.canvas_length
    torch.manual_seed(0)
    prompt = torch.randint(10, 1000, (1, 16), device=device)
    canvas = torch.randint(10, 1000, (1, canvas_length), device=device)
    with torch.no_grad():
        outputs = model(input_ids=prompt, decoder_input_ids=canvas, use_cache=False)
    logits = outputs.logits
    finite = bool(logits.isfinite().all())
    print(f"text-only logits {tuple(logits.shape)} dtype={logits.dtype} finite={finite} std={logits.float().std():.4f}")

    image_ok = True if text_only else _check_image_forward(model, checkpoint_path, device)

    ok = tied and private and finite and image_ok
    print(f"\nLOAD + FORWARD: {'OK' if ok else 'FAILED'}")
    return ok


def _check_image_forward(model, checkpoint_path: str, device: str) -> bool:
    """Run the text+image path exactly as the training data pipeline would."""
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    from veomni.data import DiffusionGemmaCollator
    from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY

    print("\n--- text+image forward ---")
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    transform = DATA_TRANSFORM_REGISTRY["diffusion_gemma_mm"]
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))

    config = model.config
    features = transform(
        {
            "messages": [
                {"from": "human", "value": "<image>What is in this image?"},
                {"from": "gpt", "value": "A block of random noise."},
            ],
            "images": [image],
        },
        processor=processor,
        canvas_length=config.canvas_length,
        vocab_size=config.text_config.vocab_size,
        max_seq_len=4096,
    )
    batch = DiffusionGemmaCollator(pad_token_id=config.text_config.pad_token_id or 0)(features)
    num_image_tokens = int((batch["input_ids"] == config.image_token_id).sum())
    print(f"prompt {tuple(batch['input_ids'].shape)} with {num_image_tokens} image tokens")
    print(f"pixel_values {tuple(batch['pixel_values'].shape)}")

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch, use_cache=False)
    finite = bool(outputs.logits.isfinite().all()) and bool(outputs.loss.isfinite())
    print(f"logits {tuple(outputs.logits.shape)} loss={outputs.loss.item():.4f} finite={finite}")

    # A wrong image pathway is easy to hide: features can be computed and then dropped.
    # Perturbing the pixels must move the loss.
    other = dict(batch)
    other["pixel_values"] = torch.randn_like(batch["pixel_values"])
    with torch.no_grad():
        other_loss = model(**other, use_cache=False).loss
    responds = not torch.allclose(outputs.loss, other_loss)
    print(f"loss responds to the image content: {responds} ({outputs.loss.item():.4f} vs {other_loss.item():.4f})")
    return finite and responds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_path")
    parser.add_argument("--layout-only", action="store_true", help="skip the bf16 load and forward pass")
    parser.add_argument("--text-only", action="store_true", help="drop the vision tower, as text-only training does")
    parser.add_argument("--device", default="cuda", help="device the forward pass runs on")
    parser.add_argument("--load-device", default="cpu", help="device weights are materialized on; see the docstring")
    args = parser.parse_args()

    ok = check_layout(args.checkpoint_path, args.text_only)
    if not args.layout_only:
        print()
        ok = check_load_and_forward(args.checkpoint_path, args.text_only, args.device, args.load_device) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
