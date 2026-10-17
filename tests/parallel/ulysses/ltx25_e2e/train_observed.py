"""Run the production DiTTrainer with a read-only validation callback."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend

from veomni.arguments import parse_args
from veomni.models.diffusers.ltx2_3.ltx_core.model.transformer.attention import PytorchAttention
from veomni.models.diffusers.ltx2_3.ltx_transformer.modeling_ltx2_3_transformer import Attention
from veomni.trainer.callbacks.base import Callback
from veomni.trainer.dit_trainer import DiTTrainer, VeOmniDiTArguments


class Audit(Callback):
    def __init__(self, trainer):
        """Initialize rank-local audit output without changing the training state."""
        super().__init__(trainer)
        self.rank = dist.get_rank()
        self.path = Path(os.environ["LTX25_AUDIT_DIR"])
        self.path.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.previous = None

    def snapshot(self):
        """Copy local trainable parameter shards to CPU for update comparisons."""
        return {
            name: (p.to_local() if isinstance(p, DTensor) else p).detach().float().cpu().clone()
            for name, p in self.trainer.model.named_parameters()
            if p.requires_grad
        }

    def write(self, record, tensors):
        """Record finite parameter hashes and persist the rank-local audit history."""
        digest = hashlib.sha256()
        exported_digest = hashlib.sha256()
        for name, value in sorted(tensors.items()):
            assert torch.isfinite(value).all(), name
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
            exported_digest.update(name.encode())
            exported_digest.update(value.bfloat16().float().numpy().tobytes())
        record.update(
            rank=self.rank,
            parameter_sha256=digest.hexdigest(),
            exported_parameter_sha256=exported_digest.hexdigest(),
            trainable_local_numel=sum(v.numel() for v in tensors.values()),
        )
        self.records.append(record)
        (self.path / f"rank-{self.rank}.json").write_text(json.dumps(self.records, indent=2))
        print("LTX25_E2E_TRAIN " + json.dumps(record), flush=True)

    def on_train_begin(self, state, **kwargs):
        """Capture initial adapter parameters after model setup or checkpoint restoration."""
        self.previous = self.snapshot()
        assert self.previous and all("lora_" in name for name in self.previous)
        self.write(dict(event="begin", step=state.global_step), self.previous)

    def on_step_end(self, state, loss, loss_dict, grad_norm, **kwargs):
        """Require finite loss and gradients and record a nonzero local adapter update."""
        loss, grad_norm = float(loss), float(grad_norm)
        assert math.isfinite(loss) and math.isfinite(grad_norm) and grad_norm > 0
        values = self.snapshot()
        delta = sum(float((v - self.previous[n]).abs().sum()) for n, v in values.items())
        assert delta > 0, "No local LoRA update"
        self.write(
            dict(event="step", step=state.global_step, loss=loss, gradient_norm=grad_norm, update_l1=delta), values
        )
        self.previous = values


args = parse_args(VeOmniDiTArguments)
trainer = DiTTrainer(args)
# Pin the validation backend: cuDNN SDPA backward is nondeterministic on the
# tested runtime and obscures checkpoint-equivalence checks at BF16 precision.
attention_modules = 0
for root in (trainer.base.model.model, trainer.base.model.condition_model):
    for module in root.modules():
        if isinstance(module, Attention):
            module.attention_function = PytorchAttention(priority=[SDPBackend.MATH])
            module.masked_attention_function = PytorchAttention(priority=[SDPBackend.MATH])
            attention_modules += 1
assert attention_modules > 0
torch.use_deterministic_algorithms(True, warn_only=False)
print(f"LTX25_E2E_ATTENTION backend=SDPA_MATH modules={attention_modules}", flush=True)
trainer.base._callbacks.append(Audit(trainer.base))
trainer.train()
if dist.is_initialized():
    dist.barrier()
if int(os.environ["RANK"]) == 0:
    print("LTX25_E2E_TRAIN_PASS", flush=True)
if dist.is_initialized():
    dist.destroy_process_group()
