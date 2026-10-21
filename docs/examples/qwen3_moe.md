# Qwen3 MoE training guide

1. Download qwen3 moe model

```shell
python3 scripts/download_hf_model.py \
  --repo_id Qwen/Qwen3-30B-A3B \
  --local_dir .
```

2. Train directly on the downloaded checkpoint

VeOmni's runtime `CheckpointTensorConverter` folds the per-expert HF
safetensor keys (`experts.{j}.gate_proj.weight`, …) into VeOmni's fused
`gate_up_proj` / `down_proj` layout at load time. The stock HF checkpoint
can be passed straight to training — no offline merge step is required.
See `docs/transformers_v5/transformers_v5_moe_weight_loading.md` for the
full format matrix and how to convert a VeOmni-format training checkpoint
back to per-expert HF keys for inference engines.

`scripts/moe_ckpt_merge/moe_merge.py` is deprecated. It still works and
may be useful as a one-time optimization for very large checkpoints
(e.g. Qwen3-235B) where you want to amortize the per-load stacking cost
across many runs, but it is no longer a prerequisite.

Most of the MoE models in Transformers referenced the open-source implementation of Mixtral MoE. In this implementation, MoE experts are divided into multiple blocks instead of being combined into a single `nn.Parameters`. Additionally, there are cpu-block operators like `torch.where()` and for loop, which are not very friendly for integrating MoE fusion operators.

Origin [Qwen3MoeMLP](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L200C1-L213C25) code
```python
class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):

            ...

        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

            ...

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

```

- Combine the per-expert MLPs into stacked `Qwen3MoeExperts` weights and always call a local `moe_experts` `VeomniOp`. `eager` is a registered row, not a separate `ModuleList` path.

```python
from veomni.ops.config import resolve_op_impl
from veomni.ops import VeomniOp


class Qwen3MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.veomni_moe = VeomniOp("moe_experts", "standard", resolve_op_impl("moe_implementation"))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        unused = self.gate_up_proj.new_empty(0)
        return self.veomni_moe(
            hidden_states,
            top_k_weights,
            top_k_index,
            unused,
            unused,
            self.down_proj,
            self.gate_up_proj,
            num_experts=self.num_experts,
        )
```

See `veomni/models/transformers/qwen3_moe/qwen3_moe_gpu_patch_gen_config.py` for the live patch. Selection comes from `model.ops_implementation.moe_implementation`.

3. Train qwen3 moe model
```
bash train.sh tasks/train_text.py configs/text/qwen3-moe.yaml
```
