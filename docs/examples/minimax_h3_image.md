# MiniMax-H3-Image inference

MiniMax-H3-Image is the image-generation member of the MiniMax-H3 family.
It reuses the H3 text encoder and the H3 diffusion transformer without the
audio branch, samples on a fixed six-step schedule, and supports text-to-image
and reference-image editing.

## Download

```shell
hf download MiniMaxAI/MiniMax-H3-Image --local-dir pretrained_models/MiniMax-H3-Image
```

The checkpoint directory contains:

```text
MiniMax-H3-Image/
├── transformer/                 # native MiniMax-H3 DiT layout (audio disabled)
├── vae_diffusers/               # image VAE in diffusers layout
├── text_encoder/                # Qwen3-VL, read at layer 50
├── processor/
├── tokenizer/
├── pdd_fused_heads.safetensors  # one output head per sampling step
└── pdd_config.json
```

## Text-to-image

```python
import torch
from veomni.models.diffusers.minimax_h3_image.inference import MiniMaxH3ImagePipeline

pipe = MiniMaxH3ImagePipeline.from_pretrained(
    "pretrained_models/MiniMax-H3-Image",
    torch_dtype=torch.bfloat16,
)
image = pipe(
    prompt="A quiet mountain lake at sunrise, photorealistic, wide angle",
    height=2048,
    width=2048,
    seed=0,
)
image.save("t2i.png")
```

## Image editing

References are referred to as `<Picture 1>`, `<Picture 2>`, ... in the prompt.
Each reference is encoded at its own size; only the generated image uses the
requested canvas.

```python
from PIL import Image

reference = Image.open("reference.png")
image = pipe(
    prompt="<Picture 1> The same character, now standing in a snowy forest at night",
    height=2048,
    width=2048,
    references=[reference],
    seed=0,
)
image.save("edit.png")
```

## Notes

- The released checkpoint is distilled to six steps with no classifier-free
  guidance. `num_inference_steps` is therefore not exposed.
- Both sides of the generated canvas are multiples of 32. Requests below
  `min_generation_area` (default `2048 * 2048`) are generated on a larger
  canvas and resized back.
- The text encoder and DiT are about 66 GB each. The pipeline stages the text
  encoder and VAE on demand and keeps them off the device while the DiT runs.
- Training support is not part of this integration; the released PDD heads are
  an inference-time distillation of the H3 backbone.
