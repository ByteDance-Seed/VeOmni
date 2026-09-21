import argparse
import csv
import os
import random

import pandas as pd
import torch


# --- Fixed Architectural Constants (Internal Only) ---

# channel number of video lantent
FIXED_C = 16
# sequence length of text
FIXED_L_TEXT = 512
# embedding dimension of text
FIXED_D_TEXT = 4096
# dimension of clip feature (video)
FIXED_D_CLIP = 1280
# channel number of image latent
FIXED_C_Y = 20


def generate_random_text(length=100):
    """Generates a random pseudo-natural language string."""
    words = [
        "a",
        "beautiful",
        "large",
        "ancient",
        "stone",
        "structure",
        "surrounded",
        "by",
        "lush",
        "greenery",
        "ocean",
        "cat",
        "dog",
        "running",
        "jumping",
        "flying",
        "in",
        "the",
        "sky",
        "city",
        "street",
    ]
    text = []
    for _ in range(max(1, length // 10)):
        text.append(random.choice(words))
    return " ".join(text).capitalize() + "..."


def generate_structured_i2v_data(num_files: int, T: int, H: int, W: int, output_dir: str):
    """
    Generates I2V training data.
    Structure:
    output_dir/
        train/ (contains .pth files)
        metadata.csv
    """

    # 1. Define and create sub-directory for tensors
    train_dir = os.path.join(output_dir, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        print(f"Created training folder: {train_dir}")

    metadata_rows = []

    for i in range(1, num_files + 1):
        base_name = f"video{i:03d}"
        full_file_name = f"{base_name}.tensors.pth"
        # Files are saved inside the 'train' subfolder
        file_path = os.path.join(train_dir, full_file_name)

        # --- Tensor Generation ---
        video_latents = torch.randn(FIXED_C, T, H, W, dtype=torch.bfloat16)
        clip_feature = torch.randn(1, 1, FIXED_D_CLIP, dtype=torch.bfloat16)
        y_feature = torch.randn(1, FIXED_C_Y, T, H, W, dtype=torch.bfloat16)
        context_emb = torch.randn(1, FIXED_L_TEXT, FIXED_D_TEXT, dtype=torch.bfloat16)

        sample = {
            "latents": video_latents,
            "image_emb": {"clip_feature": clip_feature, "y": y_feature},
            "prompt_emb": {
                "context": context_emb,
            },
        }

        torch.save(sample, file_path)

        # --- Metadata Recording ---
        raw_text = generate_random_text(random.randint(60, 150))
        clean_text = raw_text.replace('"', "")

        metadata_rows.append({"file_name": base_name, "text": clean_text})

        print(f"  - Generated: train/{full_file_name} [T={T}, H={H}, W={W}]")

    # 2. Save metadata.csv in the root of output_dir
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(metadata_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"\nSuccess! Data structure ready in: {output_dir}")
    print(f"Tensors: {train_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I2V Data Generator with Structured Output")

    # The 5 exposed parameters
    parser.add_argument("--num_files", type=int, default=8, help="Number of files to generate")
    parser.add_argument("--T", type=int, default=16, help="Temporal dimension (frames)")
    parser.add_argument("--H", type=int, default=32, help="Height")
    parser.add_argument("--W", type=int, default=48, help="Width")
    parser.add_argument("--output_dir", type=str, default="Wanmini480", help="Dataset output directory")

    args = parser.parse_args()

    generate_structured_i2v_data(num_files=args.num_files, T=args.T, H=args.H, W=args.W, output_dir=args.output_dir)
