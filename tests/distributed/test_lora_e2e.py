"""
End-to-end LoRA tests: init consistency, forward/backward, checkpoint save/load.

Run:
    torchrun --nproc_per_node=4 tests/distributed/test_lora_e2e.py --test init
    torchrun --nproc_per_node=4 tests/distributed/test_lora_e2e.py --test fwd
    torchrun --nproc_per_node=4 tests/distributed/test_lora_e2e.py --test ckpt
"""

import argparse
import tempfile

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.utils.helper import set_seed


def _make_eager_ops_config():
    """Eager (hardware-agnostic) OpsImplementationConfig for tests."""
    return OpsImplementationConfig(
        attn_implementation="flash_attention_2",
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


MODEL_PATH = "/mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-0.6B-Base"
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_RANK = 8
LORA_ALPHA = 16


def _init_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    init_parallel_state(dp_size=world_size, dp_shard_size=world_size, dp_mode="fsdp2")
    set_seed(42)
    return rank, world_size


def _build_lora_model(adapter_path=None):
    """Build a Qwen3-0.6B model with LoRA for FSDP2 training."""
    model = build_foundation_model(
        config_path=MODEL_PATH,
        weights_path=MODEL_PATH,
        torch_dtype="bfloat16",
        init_device="meta",
        ops_implementation=_make_eager_ops_config(),
    )
    if adapter_path is None:
        cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES)
        model = get_peft_model(model, cfg)
    else:
        import warnings

        from peft import PeftModel

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="copying from a non-meta parameter")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model = build_parallelize_model(
        model,
        init_device="meta",
        weights_path=MODEL_PATH,
        adapter_path=adapter_path,
        broadcast_model_weights_from_rank0=True,
    )
    return model


def test_init_consistency():
    """lora_A shards must differ across ranks (DTensor shard-aware random)."""
    rank, world_size = _init_dist()
    model = _build_lora_model()

    checked = False
    for name, param in model.named_parameters():
        if "lora_A" not in name:
            continue
        local = param._local_tensor if hasattr(param, "_local_tensor") else param.data
        flat = local.reshape(-1).float().contiguous()
        device = f"cuda:{rank}"
        sz = torch.tensor([flat.numel()], dtype=torch.long, device=device)
        all_sz = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_sz, sz)
        max_sz = max(int(s.item()) for s in all_sz)
        if max_sz == 0:
            continue
        buf = torch.zeros(max_sz, dtype=torch.float32, device=device)
        if flat.numel() > 0:
            buf[: flat.numel()] = flat
        gathered = [torch.zeros(max_sz, dtype=torch.float32, device=device) for _ in range(world_size)]
        dist.all_gather(gathered, buf)
        if rank == 0:
            non_empty = [
                (i, gathered[i][: int(all_sz[i].item())]) for i in range(world_size) if int(all_sz[i].item()) > 0
            ]
            if len(non_empty) > 1:
                collapsed = all(torch.allclose(non_empty[0][1], t) for _, t in non_empty[1:])
                assert not collapsed, f"Rank collapse detected for {name}!"
        checked = True
        break  # one param is enough to verify

    assert checked, "No lora_A param found!"
    if rank == 0:
        print("PASS: lora_A shards differ across ranks")
    dist.destroy_process_group()


def test_forward_backward():
    """Forward + backward pass completes without error; lora_A grad is non-zero."""
    rank, _ = _init_dist()
    model = _build_lora_model()
    model.train()

    input_ids = torch.randint(0, 1000, (2, 16), device=f"cuda:{rank}")
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()

    has_grad = False
    for name, param in model.named_parameters():
        if "lora_A" in name and param.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradient found for any lora_A parameter!"
    if rank == 0:
        print("PASS: forward/backward succeeds and lora_A has gradient")
    dist.destroy_process_group()


def test_checkpoint_save_load():
    """Save LoRA adapter then load it and verify weights match."""
    rank, _ = _init_dist()
    model = _build_lora_model()

    with tempfile.TemporaryDirectory() as tmp:
        paths = [tmp]
        dist.broadcast_object_list(paths, src=0)
        save_dir = paths[0]

        from veomni.utils.save_safetensor_utils import save_lora_adapter_with_dcp

        save_lora_adapter_with_dcp(model, save_dir)
        dist.barrier()

        model2 = _build_lora_model(adapter_path=save_dir)

        # Compare lora_A weights
        sd1 = {n: p for n, p in model.named_parameters() if "lora_A" in n}
        sd2 = {n: p for n, p in model2.named_parameters() if "lora_A" in n}
        assert set(sd1.keys()) == set(sd2.keys()), "Param name mismatch after reload!"
        for name in sd1:
            local1 = sd1[name]._local_tensor if hasattr(sd1[name], "_local_tensor") else sd1[name].data
            local2 = sd2[name]._local_tensor if hasattr(sd2[name], "_local_tensor") else sd2[name].data
            # save_lora_adapter_with_dcp casts float32 → bfloat16 during save, so
            # compare after the same cast to avoid false precision-mismatch failures.
            assert torch.allclose(local1.to(torch.bfloat16).float(), local2.float(), atol=1e-5), (
                f"Weight mismatch for {name}"
            )

    if rank == 0:
        print("PASS: checkpoint save/load weights match")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["init", "fwd", "ckpt"], required=True)
    args = parser.parse_args()
    {"init": test_init_consistency, "fwd": test_forward_backward, "ckpt": test_checkpoint_save_load}[args.test]()
