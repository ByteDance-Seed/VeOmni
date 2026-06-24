# BAGEL-7B-MoT (SeedOmni V2)

End-to-end recipe for training and inferring **BAGEL-7B-MoT** as a SeedOmni V2
graph model. The upstream BAGEL checkpoint is split into five OmniModules: text
embedding / LM head, SigLIP-NaViT understanding tower, VAE codec, flow connector,
and Qwen2-MoT backbone.

All paths below assume the upstream BAGEL checkpoint lives at
`/mnt/hdfs/user_dir/veomni_omni/models/transformers/BAGEL-7B-MoT`. Adjust to your
own storage.

Config dir: `configs/seed_omni/Bagel/bagel_7b_mot/`.

| Module | Holds | Role |
|--------|-------|------|
| `bagel_text_encoder` | tokenizer, token embedding, LM head | text/template markers, text logits, image-marker embeddings |
| `bagel_siglip_navit` | SigLIP-NaViT tower + connector | user image context for understanding and edit prompts |
| `bagel_vae` | VAE encoder/decoder | assistant image training latents, edit context latents, generated latent decode |
| `bagel_flow_connector` | VAE↔LLM projections, timestep embedding | latent patch embedding, velocity prediction, denoise state |
| `bagel_qwen2_mot` | Qwen2-MoT decoder backbone | text AR and flow-denoise hidden states |

The omni config layout uses the same `base.yaml` for training and inference. The
training block references `modules_train.yaml` and `graph_train.yaml`; the
inference block maps each scenario to a separate generation graph.

| File | Role |
|------|------|
| `base.yaml` | Top-level launcher: model paths, accelerator, data, train, and `infer` block. |
| `modules_train.yaml` | Per-module training paths. |
| `modules_infer_eager.yaml` | Single-process inference: every module loads eager. |
| `modules_infer_fsdp.yaml` | Distributed inference: every module uses FSDP2. |
| `graph_train.yaml` | Training DAG. |
| `graph_infer_und.yaml` | Image/text understanding to text. |
| `graph_infer_gen.yaml` | Text to image generation. |
| `graph_infer_edit.yaml` | Text+image to image edit. |
| `data.yaml` | Weighted multisource data list. |

The V2 Bagel wiring currently exposes understanding, generation, and edit.

---

## 1. Convert the checkpoint

The converter reads upstream `llm_config.json`, `vit_config.json`,
`ema.safetensors`, and `ae.safetensors`, then writes one sub-checkpoint per
module:

```bash
python scripts/convert_model.py \
  --model_type bagel \
  --model_path /mnt/hdfs/user_dir/veomni_omni/models/transformers/BAGEL-7B-MoT \
  --output_dir /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/BAGEL-7B-MoT
```

The output root becomes `model.model_path` and `infer.model_path` in
`base.yaml`. It must contain:

```text
BAGEL-7B-MoT/
├── bagel_text_encoder/
├── bagel_siglip_navit/
├── bagel_vae/
├── bagel_flow_connector/
└── bagel_qwen2_mot/
```

`bagel_text_encoder` also stores tokenizer assets copied from the upstream
checkpoint; SigLIP-NaViT and VAE save their processors next to their weights.

---

## 2. Prepare data

`data.yaml` lists a weighted multisource mixture:

```yaml
sources:
  - /mnt/hdfs/user_dir/dataset/imagenet1k_train
  - /mnt/hdfs/veomni/datasets/tulu-3-sft-mixture/data
  - /mnt/hdfs/veomni/datasets/sharegpt4v_cap_100k
names:
  - imagenet1k
  - tulu-3-sft-mixture
  - sharegpt4v_cap_100k
schedule:
  - { schedule_type: const, weights: [0.5, 0.2, 0.3] }
```

The Bagel CPU preprocessor routes images by role:

- user images become `BAGEL_SIGLIP_CONTEXT` for SigLIP-NaViT.
- assistant images become `BAGEL_VAE_CONTEXT` for VAE latent targets.

`ConversationItem.source` is the real producer/consumer branch identity.
`meta["source"]` is only an alignment hint for dummy/worker paths and should not
drive real routing.

---

## 3. Train

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/base.yaml
```

The training DAG is:

```text
bagel_text_encoder.encode          -> bagel_qwen2_mot
bagel_siglip_navit                 -> bagel_qwen2_mot
bagel_vae.encode                   -> bagel_flow_connector.embed_latent
bagel_flow_connector.embed_latent  -> bagel_qwen2_mot
bagel_qwen2_mot                    -> bagel_text_encoder.decode
bagel_qwen2_mot                    -> bagel_flow_connector.decode_velocity
```

`bagel_flow_connector.embed_latent` patchifies VAE latents, samples the training
noise/timestep, writes the velocity target, and projects noised latent patches
into LLM space. `bagel_flow_connector.decode_velocity` consumes the MoT hidden
states for those latent query positions and computes the flow velocity loss.

Quick smoke run:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/base.yaml \
  --model.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/BAGEL-7B-MoT \
  --train.max_steps 10 \
  --train.global_batch_size 8 \
  --train.micro_batch_size 1 \
  --train.wandb.enable false
```

---

## 4. Inference

`tasks/omni/infer_omni.py` selects a generation graph with `--infer.infer_type`.
Use `modules_infer_eager.yaml` for a single-process run or
`modules_infer_fsdp.yaml` with `bash train.sh` for a torchrun/FSDP2 run.

### 4.1 Understanding

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/base.yaml \
  --infer.infer_type infer_und \
  --infer.modules configs/seed_omni/Bagel/bagel_7b_mot/modules_infer_eager.yaml \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/BAGEL-7B-MoT \
  --infer.image /path/to/image.jpg \
  --infer.prompt "Describe this image." \
  --infer.output_dir bagel_out
```

`infer_und` runs SigLIP-NaViT for the prompt image, inserts image marker
embeddings via the text encoder, and then uses Qwen2-MoT + text encoder AR
decode until `text_done`.

### 4.2 Text-to-image generation

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/base.yaml \
  --infer.infer_type infer_gen \
  --infer.modules configs/seed_omni/Bagel/bagel_7b_mot/modules_infer_eager.yaml \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/BAGEL-7B-MoT \
  --infer.prompt "A watercolor painting of a small cabin beside a lake." \
  --infer.output_dir bagel_out \
  --infer.generation_kwargs.num_timesteps 50 \
  --infer.generation_kwargs.timestep_shift 3.0
```

`infer_gen` runs prompt prefill, then loops:

```text
flow_connector.prepare_denoise_query
  -> text_encoder.encode_image_markers
  -> qwen2_mot.denoise_branch
  -> flow_connector.decode_velocity_from_hidden
  -> qwen2_mot.collect_velocity
  -> flow_connector.advance_denoise
```

When the denoise state emits `image_complete`, `bagel_vae.decode_generated`
decodes the final `BAGEL_GENERATED_LATENT`.

### 4.3 Image edit

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Bagel/bagel_7b_mot/base.yaml \
  --infer.infer_type infer_edit \
  --infer.modules configs/seed_omni/Bagel/bagel_7b_mot/modules_infer_eager.yaml \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/BAGEL-7B-MoT \
  --infer.image /path/to/source.jpg \
  --infer.prompt "Make it look like a snowy evening." \
  --infer.output_dir bagel_out
```

Edit first builds context from both image branches: VAE encodes the edit image
as `BAGEL_VAE_CONTEXT`, SigLIP-NaViT keeps the raw prompt image as visual
context, and `flow_connector.embed_context_latents` projects the VAE context into
the denoise prompt. The downstream denoise loop is shared with `infer_gen`.

---

## 5. Visualize the graphs

```bash
python scripts/visualize_omni_graph.py \
  configs/seed_omni/Bagel/bagel_7b_mot/base.yaml
# -> graphs/bagel_7b_mot_base/{training,infer_edit,infer_gen,infer_und}.mmd
```

The visualized graphs are generated from the same config loader path as training
and inference, so regenerate them after changing module entrypoints or graph
YAML.

---

## 6. Contract checks

The Bagel module and graph contracts cover carrier/source routing, generation
state transitions, prompt-parallel behavior, and graph config structure:

```bash
PYTHONPATH=/app/projects/bagel-migration/Open-VeOmni-bagel-v2 \
  ../Open-VeOmni/.venv/bin/python -m pytest -q \
  tests/seed_omni/bagel/contracts/test_processing_contracts.py \
  tests/seed_omni/bagel/contracts/test_generation_contracts.py \
  tests/seed_omni/bagel/contracts/test_graph_config_contracts.py \
  tests/seed_omni/bagel/contracts/test_prompt_parallel_contract.py \
  tests/seed_omni/bagel/contracts/test_source_routing_contracts.py
```
