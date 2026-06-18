import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from veomni.models.loader import get_model_config


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class SFTExample:
    input_ids: torch.Tensor
    labels: torch.Tensor


class JsonlSFTDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.rows = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.rows.append(
                    SFTExample(
                        input_ids=torch.tensor(row["input_ids"], dtype=torch.long),
                        labels=torch.tensor(row["labels"], dtype=torch.long),
                    )
                )
        if not self.rows:
            raise ValueError(f"SFT dataset is empty: {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_sft(batch):
    input_ids = torch.stack([item.input_ids for item in batch])
    labels = torch.stack([item.labels for item in batch])
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class MiniMaxM3TinyDenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.dense_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.dense_intermediate_size, config.hidden_size, bias=False)
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states):
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(up * torch.sigmoid(self.swiglu_alpha * gate))


class MiniMaxM3TinySparseMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = max(1, min(config.num_experts_per_tok, self.num_experts))
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.shared_experts = MiniMaxM3TinyDenseMLP(config)

    def forward(self, hidden_states):
        routing = torch.softmax(self.gate(hidden_states), dim=-1)
        top_values, top_indices = torch.topk(routing, self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts], dim=-2)
        expert_mask = F.one_hot(top_indices, num_classes=self.num_experts).to(expert_outputs.dtype)
        expert_weights = (expert_mask * top_weights.unsqueeze(-1)).sum(dim=-2)
        return (expert_outputs * expert_weights.unsqueeze(-1)).sum(dim=-2) + self.shared_experts(hidden_states)


class MiniMaxM3TinyBlock(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads for the tiny smoke model.")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        mlp_type = config.mlp_layer_types[layer_index]
        self.mlp = MiniMaxM3TinySparseMoe(config) if mlp_type == "sparse" else MiniMaxM3TinyDenseMLP(config)

    def forward(self, hidden_states, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        batch_size, seq_len, hidden_size = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, torch.finfo(attn_scores.dtype).min)
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].to(torch.bool)
            attn_scores = attn_scores.masked_fill(~padding_mask, torch.finfo(attn_scores.dtype).min)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = (
            torch.matmul(attn_weights, value).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        )
        hidden_states = residual + self.o_proj(attn_output)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class MiniMaxM3TinyForSFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        text_config = config.text_config
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.embed_positions = nn.Embedding(text_config.max_position_embeddings, text_config.hidden_size)
        self.layers = nn.ModuleList(
            [MiniMaxM3TinyBlock(text_config, layer_index) for layer_index in range(text_config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


def load_weights_manifest(path: str):
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("actual_weights_loaded") is not False:
        raise ValueError("MiniMax M3 tiny SFT smoke expects a no-real-weights manifest.")
    return manifest


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def plot_loss(losses, output_path):
    steps = list(range(1, len(losses) + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, marker="o", linewidth=1.5, markersize=2.5, label="training loss")
    if len(losses) >= 5:
        ema = []
        value = losses[0]
        for loss in losses:
            value = 0.85 * value + 0.15 * loss
            ema.append(value)
        plt.plot(steps, ema, linewidth=2.0, label="EMA")
    plt.xlabel("SFT step")
    plt.ylabel("cross entropy loss")
    plt.title("MiniMax M3 VL tiny SFT smoke loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run MiniMax M3 VL tiny SFT smoke without real weights.")
    parser.add_argument("--config-path", default="./tests/toy_config/minimax_m3_vl_toy")
    parser.add_argument(
        "--weights-manifest", default="./tests/fixtures/minimax_m3_vl_sft/random_init_weights_manifest.json"
    )
    parser.add_argument("--dataset-path", default="./tests/fixtures/minimax_m3_vl_sft/tiny_sft.jsonl")
    parser.add_argument("--output-dir", default="/tmp/veomni_minimax_m3_vl_sft_smoke")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=20260617)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("MODELING_BACKEND", "veomni")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_model_config(args.config_path)
    weights_manifest = load_weights_manifest(args.weights_manifest)
    dataset = JsonlSFTDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sft)
    model = MiniMaxM3TinyForSFT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    losses = []
    data_iter = iter(dataloader)
    model.train()
    for _step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    curve_path = output_dir / "loss_curve.png"
    log_path = output_dir / "loss_log.json"
    plot_loss(losses, curve_path)
    summary = {
        "config_path": args.config_path,
        "dataset_path": args.dataset_path,
        "weights_manifest_path": args.weights_manifest,
        "weights_manifest": weights_manifest,
        "actual_weights_loaded": False,
        "num_examples": len(dataset),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "model_type": config.model_type,
        "text_model_type": config.text_config.model_type,
        "num_hidden_layers": config.text_config.num_hidden_layers,
        "mlp_layer_types": config.text_config.mlp_layer_types,
        "layer_types": config.text_config.layer_types,
        "losses": losses,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "loss_curve": str(curve_path),
    }
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: summary[k] for k in ("first_loss", "last_loss", "min_loss", "loss_curve")}, indent=2))
    if not losses[-1] < losses[0]:
        raise RuntimeError(f"Expected final loss to be lower than initial loss, got {losses[0]} -> {losses[-1]}")


if __name__ == "__main__":
    main()
