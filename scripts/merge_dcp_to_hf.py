import argparse
import os

from transformers import AutoConfig, AutoProcessor

from veomni.checkpoint import dcp_to_torch_state_dict
from veomni.models import save_model_weights
from veomni.utils import helper


logger = helper.create_logger(__name__)


def merge_to_hf_pt(load_dir: str, save_path: str, model_assets_dir: str = None, model_only: bool = False):
    # save model in huggingface's format
    # By default loads all checkpoint data; use model_only=True to load only model weights
    state_dict = dcp_to_torch_state_dict(
        save_checkpoint_path=load_dir,
        model_only=model_only,
    )
    # logger.info_rank0(f"Converting state_dict: {}")
    if model_assets_dir is not None:
        config = AutoConfig.from_pretrained(model_assets_dir)
        processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=True)

        save_model_weights(save_path, state_dict, model_assets=[config, processor])
    else:
        save_model_weights(save_path, state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--model_assets_dir", type=str, default=None)
    parser.add_argument(
        "--model-only",
        action="store_true",
        default=False,
        help="Only load model weights (ignore optimizer states). By default, loads all checkpoint data.",
    )
    args = parser.parse_args()
    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "hf_ckpt") if args.save_dir is None else args.save_dir
    model_assets_dir = args.model_assets_dir
    logger.info(f"Merge Args: {args}")
    merge_to_hf_pt(load_dir, save_dir, model_assets_dir, model_only=args.model_only)
    logger.info(f"Merge to hf pt success! Save to: {save_dir}")
