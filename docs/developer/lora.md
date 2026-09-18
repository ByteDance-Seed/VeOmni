# LoRA Implementation and Tests

For configuration and training commands, start with the
[LoRA user guide](../key_features/lora.md). This page covers runtime ownership,
weight loading, accounting, and regression tests for contributors.

## 2. LoRA Initialization in the Model Runtime

LoRA wrapping happens in `VeOmniModelRuntime._setup_lora()`, called from `_freeze_model_module()`.
Trainers reach this code through their model runtime; the trainer does not inherit these methods.
A single native path wraps the model with `VeOmniLoraModel`, handling dense `nn.Linear`
LoRA, MoE expert LoRA, and the two combined:

```python
# veomni/models/model_runtime.py
def _setup_lora(self):
    lora_config = self.args.lora_config
    if not bool(lora_config):
        return

    from ..lora import VeOmniLoraConfig, VeOmniLoraModel

    cfg = VeOmniLoraConfig.from_yaml(lora_config)
    lora_adapter_path = lora_config.get("lora_adapter", None)
    if lora_adapter_path is not None:
        # Resume: rebuild dense + MoE wrappers from the on-disk adapter_config.json
        # (MoE mode lives in its `veomni_lora` block). Weights load later.
        self.model = VeOmniLoraModel.from_pretrained(
            self.model,
            lora_adapter_path,
            is_trainable=lora_config.get("is_trainable", True),
        )
    else:
        self.model = VeOmniLoraModel(self.model, cfg)
```

`VeOmniLoraModel` mirrors `peft.PeftModel` structurally: `model.base_model.model` is the
original model, so every LoRA parameter FQN and every saved adapter key carries the
`base_model.model.` prefix — byte-identical to a PEFT checkpoint. Attribute access and
`forward` are forwarded to the wrapped model, so the trainer, loss computation, and
`generate` are unchanged. After wrapping, the base model is fully frozen and only the LoRA
parameters (dense `LoraLinear` and MoE-LoRA, if any) have `requires_grad=True`.

`BaseTrainer._init_callbacks()` registers one `CheckpointCallback` either way. The export format
is the model's decision, not the callback's: a model that trains only adapters exports the
adapter, so there is nothing for a LoRA-specific callback to do.

### 2.1 LoRA MFU and FLOPs accounting

`EnvironMeterCallback` passes the effective `VeOmniLoraConfig` from the wrapped model to
`VeomniFlopsCounter`, so the reported FLOPs and MFU reflect the work performed by LoRA
training instead of using the full-fine-tuning estimate. Full-fine-tuning accounting is
unchanged when no LoRA config is present.

LoRA FLOPs estimation currently supports only the Qwen-family model types registered in
`LORA_MODULES_BY_MODEL_TYPE`. An unsupported model or LoRA target emits a warning and
reports zero achieved FLOPs and zero MFU rather than interrupting training or reporting a
misleading value. `scripts/profile/compare_lora_flops.py` exercises this path against real
Hugging Face Qwen configs and compares full fine-tuning with several LoRA target sets and
ranks.

`VLMTrainer` uses the same native LoRA setup, FSDP2 loading, and adapter checkpoint path as
the text trainer. During LoRA training, the LoRA configuration determines which adapters
and optional bias parameters are trainable:

- `freeze_vit` and `freeze_audio_tower` are ignored. They control only full tuning.
- Vision and audio modules explicitly matched by `target_modules` remain trainable.
- With the default `bias: none`, untargeted towers remain frozen and do not participate
  in backward. Other bias policies explicitly opt additional bias parameters into training.
- Full merger or audio-projection weights that are normally retained by the non-LoRA freeze
  policy are not trained in LoRA mode, because those full weights are not part of the
  exported adapter.

Qwen3-VL vision adapters may target `qkv`, `proj`, `linear_fc1`, and `linear_fc2`;
language targets such as `q_proj` and `v_proj` can be used in the same config.

For Qwen VLM configurations, vision-tower accounting uses the vision-token sequence lengths
collected from the current batch and follows this logic:

1. If the batch has no vision tokens, ViT FLOPs are zero.
2. If the batch has vision tokens but no vision parameter is trainable, the ViT is treated
   as frozen and only its forward-pass FLOPs are counted.
3. If a supported ViT module has a trainable adapter, the tower uses the existing coarse
   forward/backward estimate and includes matched LoRA work.
4. If only vision biases are trainable (for example, `bias: all` with language-only adapter
   targets), base forward/input-gradient work is counted without adapter FLOPs.

The estimator intentionally uses the framework's existing 2x/4x/6x approximation rather
than reconstructing the exact autograd graph: 2x is forward-only, 4x includes activation
gradients through frozen weights, and 6x includes trainable-weight or adapter work.
During full fine-tuning, `freeze_vit: true` selects the forward-only 2x estimate for the
entire vision tower. During LoRA training the flag is ignored and adapter targets determine
the estimate. This intentionally treats Omni's trainable merger as part of the frozen tower.

The native LoRA implementation currently adapts two kinds of weights:

- `nn.Linear` modules selected through `target_modules`.
- Fused routed-MoE expert weights, represented as 3-D `nn.Parameter`s and selected through
  `target_parameters`.

The MoE experts are not `nn.Embedding` layers. The FLOPs estimator counts adapter work for
both supported categories above. Within the ViT, only matched linear layers receive LoRA
FLOPs; non-linear projections such as Qwen's `Conv3d` patch embedding are not supported LoRA
targets. Adding a convolution name to `target_modules` does not adapt it.

---

## 3. Weight Loading with LoRA

VeOmni LoRA training uses FSDP2 with `init_device: meta`. Weight loading goes through
`build_parallelize_model` and then `post_process_after_weight_loading` in
`torch_parallelize.py`. The LoRA-specific path:

1. **Base-model weights**: loaded via `rank0_load_and_broadcast_weights` or
   `load_model_weights` — the standard FSDP2 path, unchanged for LoRA.

2. **Adapter weights** (resume only): `build_parallelize_model` passes `adapter_path`
   to the FSDP2/DDP wrap, which — for a `VeOmniLoraModel` — calls the native
   `veomni.lora.weight_loading.load_lora_weights` (all-ranks read) or
   `rank0_load_and_broadcast_lora_weights` (rank-0 reads then broadcasts). Both read the
   PEFT-format adapter file natively (safetensors / torch, **no `peft` import**) and remap
   on-disk keys to model FQNs before dispatching into DTensors. EP-sharded LoRA tensors are
   sliced from the disk-side `[E, ...]` shape to the local `[E_local, ...]` via the runtime
   parallel plan, exactly as on the base-weight path.

3. **Adapter weight initialisation from scratch**: `post_process_after_weight_loading`
   calls `_init_lora_parameter` for any LoRA parameter not yet filled, invoking
   `reset_lora_parameters` (kaiming `A` / zero `B`). For MoE wrappers the decision is made
   against the missing-name set: reset only when *all* of that wrapper's LoRA tensors are
   still missing; skip when none are missing; raise on a partial load so already-loaded
   `A`/`B` tensors are never clobbered.

**Key difference from base model loading:** the on-disk adapter keys omit the adapter-name
infix (PEFT convention — e.g. `lora_A.weight`), whereas the live model stores them as
`lora_A.<adapter_name>.weight`. `veomni.lora.state_dict.insert_adapter_name` /
`strip_adapter_name` handle the translation in both directions.

---

## 7. Testing

The test suite is under `tests/lora/` and uses small MoE toy models.
Suite layout:

| Test | Coverage |
|------|----------|
| `test_veomni_lora_native.py` | CPU, single-process: `VeOmniLoraConfig` round-trips, dense injection structure / no-op-at-init, `get_lora_state_dict` PEFT key format, save→`from_pretrained`→load parity, `merge_and_unload`, rank/alpha patterns, exclude_modules, rslora scaling, and **bidirectional PEFT interop** (gated on `peft`). |
| `test_veomni_lora_moe_native.py` | CPU, single-process: native MoE injection (independent/shared), MoE metadata embedded in `adapter_config.json` (no sidecar), state-dict key format/rank, save→reload param round-trip, and MoE-mode inference when the `veomni_lora` block is absent. |
| `test_moe_lora_eager.py` | Wrapper layout + autograd parity for both Mode 1 (independent) and Mode 2 (shared); covers all v5 MoE configs (qwen3_moe / qwen3_5_moe / qwen3_vl_moe / qwen3_omni_moe / deepseek_v3). |
| `test_moe_lora_fused.py` | Triton fused MoE-LoRA kernel parity vs eager (forward + backward) and EP autograd-class parity vs non-EP under controlled inputs. |
| `test_moe_lora_trainer.py` | Production save/load/resume round-trip: writer (DCP shard + HF adapter) → DCP-resume subprocess → adapter-resume subprocess; bit-exact LoRA reload assertion. The yaml enables both `lora_modules` (linear LoRA on `q_proj`/`v_proj`) and `target_parameters` (MoE-LoRA wrappers), so both LoRA flavors round-trip end-to-end. |
| `test_moe_lora_ep2.py` | Trainer-driven EP=2 coverage: (a) integration assertions that the EP plumbing engages (plan-bridge fires, EP slicing happens at the right ratio, DCP consolidates EP shards before HF save); (b) `test_moe_lora_ep_save_load_parallel_align` -- one EP=2 seeder writes both an HF adapter (full `[E, r, H]`) and DCP shards, then three resumer subprocesses validate cross-EP adapter parity (EP=1 vs EP=2 adapter-load trajectories match) and EP=2 DCP round-trip parity (DCP-resumer trajectory matches the seeder's tail). |

Run the suite on an environment with the GPUs required by the selected tests:

```shell
pytest -s tests/lora/
```

Run just the production-shaped save/load/resume round-trip:

```shell
pytest -s -v tests/lora/test_moe_lora_trainer.py::test_save_load_resume_round_trip
```

Or as a manual torchrun against either `independent` or `shared`:

```shell
torchrun --nproc_per_node=4 tests/lora/test_moe_lora_trainer.py \
    tests/lora/qwen3_moe_toy_lora_independent.yaml \
    --train.checkpoint.output_dir /tmp/test_moe_lora_run
```

**What the round-trip test verifies:**
1. Writer trains `max_steps=4`, writes DCP shards at step 2 / step 4 and the HF LoRA adapter (`adapter_config.json` with the `veomni_lora` block + `adapter_model.safetensors`, no sidecar) at the same cadence; snapshots the full LoRA tensors at `on_train_begin` and `on_train_end`.
2. DCP-resume subprocess loads the step-2 DCP shard, continues to step 4, and the resulting LoRA tensors must be **bit-exact** vs the writer's `on_train_end` snapshot (DCP ships model + optimizer + RNG + dataloader state).
3. LoRA-adapter-resume subprocess loads the step-4 HF adapter via `model.lora_config.lora_adapter`, and the resulting `on_train_begin` snapshot (= post-load, pre-train) must be bit-exact vs the writer's `on_train_end` snapshot in bf16 (the on-disk adapter dtype).
