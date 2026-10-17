# Janus Example Map

Use current Janus files instead of synthetic module examples.

## Modules

| Need | File or folder |
|---|---|
| Minimal stand-in module A (accelerated hooks) | `veomni/models/seed_omni/modules/fake_model/fake_module_a/accelerated/accelerated.py` |
| Minimal stand-in module A (native modeling) | `veomni/models/seed_omni/modules/fake_model/fake_module_a/modeling.py` |
| Minimal stand-in module B (loss sink) | `veomni/models/seed_omni/modules/fake_model/fake_module_b/accelerated/accelerated.py` |
| Vision encoder CPU preprocessing and dummy images | `veomni/models/seed_omni/modules/<family>/siglip/accelerated/accelerated.py` |
| Vision encoder modeling + `generate()` (`InferenceMixin`) | `veomni/models/seed_omni/modules/<family>/siglip/modeling.py` |
| Vision processor asset | `veomni/models/seed_omni/modules/<family>/siglip/processing.py` |
| Text encoder CPU preprocessing and FSM signals | `veomni/models/seed_omni/modules/<family>/text_encoder/accelerated/accelerated.py` |
| Text encoder chat template | `veomni/models/seed_omni/modules/<family>/text_encoder/chat_template.py` |
| Base text encoder shared behavior | `veomni/models/seed_omni/modules/<base>/text_encoder/` |
| AR backbone | `veomni/models/seed_omni/modules/<family>/llama/` |
| VQ codec encode/decode | `veomni/models/seed_omni/modules/<family>/vqvae/` |
| Packed training (CPU pack + mask scatter) | `veomni/models/seed_omni/modules/<family>/packing.py` + per-module `accelerated/packed.py` |
| Split checkpoint conversion | `veomni/models/seed_omni/modules/fake_model/convert_model.py` (`veomni/models/seed_omni/modules/<family>/convert_model.py`) |

## Configs

| Need | File |
|---|---|
| Minimal stand-in launcher | `configs/seed_omni/fake_model/train/base.yaml` |
| Minimal stand-in training module overrides | `configs/seed_omni/fake_model/train/modules_train.yaml` |
| Minimal stand-in eager inference overrides | `configs/seed_omni/fake_model/infer/modules_infer_eager.yaml` |
| Minimal stand-in distributed inference overrides | `configs/seed_omni/fake_model/infer/modules_infer_fsdp.yaml` |
| Minimal stand-in training graph | `configs/seed_omni/fake_model/train/graph_train.yaml` |
| Minimal stand-in generation FSMs | `configs/seed_omni/fake_model/infer/graph_infer_gen.yaml`, `configs/seed_omni/fake_model/infer/graph_infer_und.yaml` |
| Family launcher shape | `configs/seed_omni/<Family>/<variant>/train/base.yaml` |
| Family training module overrides | `configs/seed_omni/<Family>/<variant>/train/modules_train.yaml` |
| Family eager inference module overrides | `configs/seed_omni/<Family>/<variant>/infer/modules_infer_eager.yaml` |
| Family distributed inference module overrides | `configs/seed_omni/<Family>/<variant>/infer/modules_infer_fsdp.yaml` |
| Family training graph | `configs/seed_omni/<Family>/<variant>/train/graph_train.yaml` |
| Family packed training graph / launcher | `configs/seed_omni/<Family>/<variant>/packed/graph_train.yaml`, `configs/seed_omni/<Family>/<variant>/packed/base.yaml` |
| Family model-level FSDP launcher | `configs/seed_omni/<Family>/<variant>/train/base_model_fsdp.yaml` |
| Family generation FSMs | `configs/seed_omni/<Family>/<variant>/infer/graph_infer_<scenario>.yaml` |

## How To Use

1. Pick the row that matches the task shape.
2. Read the live file.
3. Copy conventions, not names.
4. Verify against source if docs disagree.
