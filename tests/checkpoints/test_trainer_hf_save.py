import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field

import torch
import torch.distributed as dist
import yaml
from checkpoint_verification_utils import load_hf_checkpoint, verify_hf_checkpoint
from torch.distributed.checkpoint import HuggingFaceStorageWriter

from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.checkpoint import build_checkpointer
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


# To prevent DCP from complaining "too many open files"
# see: https://github.com/pytorch/pytorch/issues/11201
torch.multiprocessing.set_sharing_strategy("file_system")

logger = helper.create_logger(__name__)


def read_output_dir_from_yaml(yaml_path: str) -> str:
    """Read output_dir from yaml config file."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config.get("train", {}).get("output_dir", None)


def read_model_path_from_yaml(yaml_path: str) -> str:
    """Read model_path from yaml config file."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config.get("model", {}).get("model_path", None)


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend())
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    helper.enable_high_precision_for_bf16()

    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    assert args.model.model_path is not None, (
        "model_path must be set to a directory containing safetensors weights for this test"
    )

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
        ep_outside=args.train.ep_outside,
    )

    logger.info_rank0("Prepare model")

    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
    )

    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    # Save via HuggingFaceStorageWriter
    hf_save_path = args.train.save_safetensor_path
    logger.info_rank0(f"Saving HF safetensors to {hf_save_path}")

    storage_writer = HuggingFaceStorageWriter(
        path=hf_save_path,
        save_distributed=True,
        fqn_to_index_mapping=args.model.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    dist.barrier()
    Checkpointer.save(
        path=hf_save_path,
        state={"model": model},
        storage_writer=storage_writer,
    )
    logger.info_rank0("HF safetensors save completed")

    # Wait for async save if applicable
    if Checkpointer.dcp_save_future is not None:
        logger.info_rank0("Waiting for async save to finish...")
        Checkpointer.dcp_save_future.result()

    dist.barrier()

    # Verify: compare saved safetensors against input safetensors from model_path (rank 0 only)
    if args.train.global_rank == 0:
        logger.info_rank0(f"Loading input safetensors from {args.model.model_path}")
        input_state_dict = load_hf_checkpoint(args.model.model_path, safe_serialization=True)
        logger.info_rank0(f"Input safetensors has {len(input_state_dict)} keys")

        logger.info_rank0("Verifying saved safetensors match input safetensors...")
        assert verify_hf_checkpoint(
            hf_checkpoint_dir=hf_save_path,
            original_state_dict=input_state_dict,
            safe_serialization=True,
        ), "HF checkpoint verification failed: saved safetensors do not match input safetensors!"
        logger.info_rank0("HF checkpoint verification passed!")

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0)


def _run_test(yaml_config_path: str):
    """Helper to run the test via torchrun subprocess."""
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--master_port=4321",
        "tests/checkpoints/test_trainer_hf_save.py",
        yaml_config_path,
    ]
    result = subprocess.run(command, check=True)
    assert result.returncode == 0

    # Verify the output directory has safetensors files
    output_dir = read_output_dir_from_yaml(yaml_config_path)
    assert output_dir is not None, f"output_dir not found in {yaml_config_path}"
    hf_ckpt_dir = os.path.join(output_dir, "hf_ckpt")
    assert os.path.exists(hf_ckpt_dir), f"HF checkpoint directory not found: {hf_ckpt_dir}"
    safetensor_files = [f for f in os.listdir(hf_ckpt_dir) if f.endswith(".safetensors")]
    assert len(safetensor_files) > 0, f"No safetensors files found in {hf_ckpt_dir}"
    logger.info(f"HF checkpoint files: {os.listdir(hf_ckpt_dir)}")


def test_hf_save_no_ep():
    _run_test("tests/checkpoints/no_ep.yaml")


def test_hf_save_ep4():
    _run_test("tests/checkpoints/ep4.yaml")


if __name__ == "__main__":
    main()
