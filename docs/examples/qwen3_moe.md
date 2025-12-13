# Qwen3 MoE training guide

1. Download qwen3 moe model

```shell
python3 scripts/download_hf_model.py \
  --repo_id Qwen/Qwen3-30B-A3B \
  --local_dir .
```

2. Merge qwen3 moe model experts to support GroupGemm optimize
``` shell
python3 scripts/moe_ckpt_merge/moe_merge.py --raw_hf_path Qwen3-30B-A3B  --merge_hf_path Qwen3-30B-A3B-merge
```

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

- Combine Qwen3MoeMLP to Qwen3MoeExperts, then use fused moe operator

```python
class Qwen3MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_size),
            requires_grad=True,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, expert_idx=None, cumsum=None):
        gate_proj_out = torch.matmul(hidden_states, self.gate_proj[expert_idx].transpose(0, 1))
        up_proj_out = torch.matmul(hidden_states, self.up_proj[expert_idx].transpose(0, 1))

        out = self.act_fn(gate_proj_out) * up_proj_out
        out = torch.matmul(out, self.down_proj[expert_idx].transpose(0, 1))
        return out


class Qwen3MoeSparseFusedMoeBlock(nn.Module):
    def __init__(self, config):

            ...

      self.experts = Qwen3MoeExperts(config)

    def forward(self, hidden_states, expert_idx=None, routing_weights=None, selected_experts=None) -> torch.Tensor:

          ...

        out = fused_moe_forward(
            module=self,
            num_experts=self.num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=self.gate_proj,
            fc1_2_weight=self.up_proj,
            fc2_weight=self.down_proj,
        )
      return out

```

3. Train qwen3 moe model
```
bash train.sh tasks/train_torch.py configs/pretrain/qwen3-moe.yaml
```
