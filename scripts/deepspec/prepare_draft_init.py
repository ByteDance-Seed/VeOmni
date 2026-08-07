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

"""Prepare a DeepSpec draft-model *init checkpoint* + config for VeOmni.

VeOmni builds models on the ``meta`` device and streams weights from a
HuggingFace-format checkpoint (``config.json`` + ``*.safetensors``). DeepSpec's
draft model has two frozen sub-modules copied from the target (``embed_tokens``,
``lm_head``); the rest is trained from scratch.

This script produces an init checkpoint that VeOmni can meta-load:

* ``config.json`` — the DeepSpec draft config, with ``model_type`` rewritten to
  ``"deepspec_draft"`` and ``base_model_type`` recording the target's original
  type (so ``DeepSpecDraftConfig`` can rebuild a faithful target config).
* ``model.safetensors`` — the full randomly-initialized draft state dict with
  the target's ``embed_tokens`` / ``lm_head`` weights copied in. VeOmni loads
  every param from here (missing keys would be re-initialized on meta, which we
  avoid so the run is deterministic and the frozen weights are correct).

It runs on CPU and needs the target model only here (never during training),
matching DeepSpec's "target is offline" design.

Example:
    python scripts/deepspec/prepare_draft_init.py \
        --algorithm dspark \
        --arch qwen3 \
        --target_model_name_or_path Qwen/Qwen3-4B \
        --output_dir ~/deepspec_init/dspark_qwen3_4b \
        --block_size 7 --num_draft_layers 5 \
        --target_layer_ids 1 9 17 25 33 \
        --mask_token_id 151669 --num_anchors 512 \
        --markov_rank 256 --markov_head_type vanilla \
        --confidence_head_alpha 1.0 --confidence_head_with_markov \
        --loss_decay_gamma 4.0 --ce_loss_alpha 0.1 --l1_loss_alpha 0.9
"""

import argparse
import json
import os
import sys


def _add_deepspec_to_path() -> None:
    """Make ``deepspec`` and ``veomni`` importable when run as a script."""
    # veomni bridge does the DeepSpec discovery.
    here = os.path.dirname(os.path.abspath(__file__))
    veomni_root = os.path.abspath(os.path.join(here, "..", ".."))
    if veomni_root not in sys.path:
        sys.path.insert(0, veomni_root)
    from veomni.integrations.deepspec import ensure_deepspec_importable

    ensure_deepspec_importable()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a DeepSpec draft init checkpoint for VeOmni.")
    p.add_argument("--algorithm", required=True, choices=["dspark", "dflash", "eagle3"])
    p.add_argument("--arch", required=True, choices=["qwen3", "gemma4"])
    p.add_argument("--target_model_name_or_path", required=True)
    p.add_argument("--output_dir", required=True)

    # Shared draft fields.
    p.add_argument("--target_layer_ids", type=int, nargs="+", required=True)

    # DSpark / DFlash fields.
    p.add_argument("--block_size", type=int, default=7)
    p.add_argument("--num_draft_layers", type=int, default=5)
    p.add_argument("--mask_token_id", type=int, default=None)
    p.add_argument("--num_anchors", type=int, default=512)
    p.add_argument("--markov_rank", type=int, default=0)
    p.add_argument("--markov_head_type", type=str, default="vanilla")
    p.add_argument("--confidence_head_alpha", type=float, default=0.0)
    p.add_argument("--confidence_head_with_markov", action="store_true")
    p.add_argument("--loss_decay_gamma", type=float, default=4.0)
    p.add_argument("--ce_loss_alpha", type=float, default=1.0)
    p.add_argument("--l1_loss_alpha", type=float, default=0.0)

    # Eagle3 fields.
    p.add_argument("--draft_num_hidden_layers", type=int, default=1)
    p.add_argument("--ttt_length", type=int, default=7)
    p.add_argument("--step_loss_decay", type=float, default=0.8)

    return p.parse_args()


def _build_model_args(args: argparse.Namespace):
    """Assemble the ``model_args``-style object DeepSpec's config builders read.

    DeepSpec config builders access fields via attribute *and* ``in`` (dict-style)
    membership (e.g. ``assert "target_layer_ids" in model_args``), so use a dict
    subclass that supports both.
    """
    from deepspec.utils.config import ConfigNode

    model_args = ConfigNode()
    model_args.target_model_name_or_path = args.target_model_name_or_path
    model_args.target_layer_ids = list(args.target_layer_ids)

    if args.algorithm in ("dspark", "dflash"):
        model_args.block_size = args.block_size
        model_args.num_draft_layers = args.num_draft_layers
        model_args.num_anchors = args.num_anchors
        model_args.mask_token_id = args.mask_token_id
        # DFlash == DSpark with the extra heads disabled.
        if args.algorithm == "dflash":
            model_args.markov_rank = 0
            model_args.confidence_head_alpha = 0.0
            model_args.loss_decay_gamma = args.loss_decay_gamma
            model_args.ce_loss_alpha = 1.0
            model_args.l1_loss_alpha = 0.0
        else:
            model_args.markov_rank = args.markov_rank
            model_args.markov_head_type = args.markov_head_type
            model_args.confidence_head_alpha = args.confidence_head_alpha
            model_args.confidence_head_with_markov = args.confidence_head_with_markov
            model_args.loss_decay_gamma = args.loss_decay_gamma
            model_args.ce_loss_alpha = args.ce_loss_alpha
            model_args.l1_loss_alpha = args.l1_loss_alpha
    else:  # eagle3
        model_args.draft_num_hidden_layers = args.draft_num_hidden_layers
        model_args.ttt_length = args.ttt_length
        model_args.step_loss_decay = args.step_loss_decay

    return model_args


def _build_draft_config_and_class(args, target_config, model_args):
    """Return (draft_config, draft_model_class) for the requested variant."""
    if args.algorithm in ("dspark", "dflash"):
        if args.arch == "qwen3":
            from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
            from deepspec.modeling.dspark.qwen3.config import build_draft_config

            return build_draft_config(target_config=target_config, model_args=model_args), Qwen3DSparkModel
        else:
            from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
            from deepspec.modeling.dspark.gemma4.config import build_draft_config

            return build_draft_config(target_config=target_config, model_args=model_args), Gemma4DSparkModel
    else:  # eagle3
        if args.arch == "qwen3":
            from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model
            from deepspec.modeling.eagle3.qwen3.config import build_draft_config

            return build_draft_config(target_config=target_config, model_args=model_args), Qwen3Eagle3Model
        else:
            from deepspec.modeling.eagle3.gemma4 import Gemma4Eagle3Model
            from deepspec.modeling.eagle3.gemma4.config import build_draft_config

            return build_draft_config(target_config=target_config, model_args=model_args), Gemma4Eagle3Model


def main() -> None:
    _add_deepspec_to_path()

    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[prepare_draft_init] Loading target config: {args.target_model_name_or_path}")
    target_config = AutoConfig.from_pretrained(args.target_model_name_or_path)

    model_args = _build_model_args(args)
    draft_config, draft_cls = _build_draft_config_and_class(args, target_config, model_args)

    # Base model_type for faithful config reconstruction in VeOmni. For Gemma4 the
    # DeepSpec builder derives the draft config from the *text* sub-config, so use
    # that model_type; for qwen3 the top-level type is correct.
    base_model_type = str(getattr(draft_config, "model_type", target_config.model_type))

    print(f"[prepare_draft_init] Building {draft_cls.__name__} on CPU (float32)...")
    draft_model = draft_cls(draft_config).to(device="cpu", dtype=torch.float32).eval()

    print("[prepare_draft_init] Copying frozen target embeddings + lm_head...")
    target_model = (
        AutoModelForCausalLM.from_pretrained(args.target_model_name_or_path, dtype=torch.float32)
        .to(device="cpu")
        .eval()
    )
    target_embed = target_model.get_input_embeddings()
    target_lm_head = target_model.get_output_embeddings()
    assert target_embed is not None and target_lm_head is not None
    draft_model.initialize_embeddings_and_head(
        embed_tokens=target_embed,
        lm_head=target_lm_head,
        freeze=True,
    )
    del target_model

    # Write config.json with the VeOmni-facing model_type / base_model_type.
    config_dict = draft_config.to_dict()
    config_dict["model_type"] = "deepspec_draft"
    config_dict["base_model_type"] = base_model_type
    # Ensure the concrete draft class is discoverable via architectures.
    config_dict["architectures"] = [draft_cls.__name__]
    # Persist the DSpark loss weights into the config so the trainer reads them
    # from the model (single source of truth); the YAML model_config can still
    # override them at runtime. Eagle3 reads ttt_length / step_loss_decay, which
    # the DeepSpec config builder already put on draft_config.
    if args.algorithm in ("dspark", "dflash"):
        config_dict.setdefault("loss_decay_gamma", model_args.loss_decay_gamma)
        config_dict.setdefault("ce_loss_alpha", model_args.ce_loss_alpha)
        config_dict.setdefault("l1_loss_alpha", model_args.l1_loss_alpha)
        config_dict.setdefault("confidence_head_alpha", model_args.get("confidence_head_alpha", 0.0))
    with open(os.path.join(args.output_dir, "config.json"), "w") as fh:
        json.dump(config_dict, fh, indent=2)
    print(f"[prepare_draft_init] Wrote config.json (model_type=deepspec_draft, base={base_model_type}).")

    # Save the full state dict (bf16 to match training precision) as safetensors.
    state_dict = {key: value.to(torch.bfloat16).contiguous() for key, value in draft_model.state_dict().items()}
    save_file(state_dict, os.path.join(args.output_dir, "model.safetensors"), metadata={"format": "pt"})
    n_params = sum(v.numel() for v in state_dict.values())
    print(
        f"[prepare_draft_init] Wrote model.safetensors "
        f"({len(state_dict)} tensors, {n_params/1e6:.1f}M params) to {args.output_dir}."
    )
    print("[prepare_draft_init] Done. Point the training config's model.config_path / model.model_path here.")


if __name__ == "__main__":
    main()
