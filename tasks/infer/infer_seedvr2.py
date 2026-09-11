"""Restore an image/video with the official SeedVR2-3B weights in VeOmni."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from veomni.models.diffusers.seedvr2.inference import SeedVR2Restorer, read_media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.height < 16 or args.width < 16:
        parser.error("height and width must be at least 16")
    frames, fps = read_media(args.input)
    image_input = fps is None
    restorer = SeedVR2Restorer(args.weights, device=args.device, dtype=getattr(torch, args.dtype))
    result = restorer(frames, target_area=args.height * args.width, seed=args.seed)
    result = result.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if image_input:
        Image.fromarray(result[0]).save(args.output)
    else:
        import av

        with av.open(str(args.output), mode="w") as destination:
            stream = destination.add_stream("libx264", rate=fps)
            stream.height, stream.width = result.shape[1:3]
            stream.pix_fmt = "yuv420p"
            for frame in result:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    destination.mux(packet)
            for packet in stream.encode():
                destination.mux(packet)


if __name__ == "__main__":
    main()
