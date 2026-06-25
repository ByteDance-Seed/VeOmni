# Janus Example Map

Use current Janus files instead of synthetic module examples.

## Modules

| Need | File or folder |
|---|---|
| Vision encoder CPU preprocessing and dummy images | `veomni/models/seed_omni/modules/janus/siglip/modulemixin.py` |
| Vision encoder modeling | `veomni/models/seed_omni/modules/janus/siglip/modeling.py` |
| Vision processor asset | `veomni/models/seed_omni/modules/janus/siglip/processing.py` |
| Text encoder CPU preprocessing and FSM signals | `veomni/models/seed_omni/modules/janus/text_encoder/modulemixin.py` |
| Text encoder chat template | `veomni/models/seed_omni/modules/janus/text_encoder/chat_template.py` |
| Base text encoder shared behavior | `veomni/models/seed_omni/modules/base/text_encoder/` |
| AR backbone | `veomni/models/seed_omni/modules/janus/llama/` |
| VQ codec encode/decode | `veomni/models/seed_omni/modules/janus/vqvae/` |
| Split checkpoint conversion | `veomni/models/seed_omni/modules/janus/convert_model.py` |

## Configs

| Need | File |
|---|---|
| Launcher shape | `configs/seed_omni/Janus/janus_1.3b/base.yaml` |
| Training module overrides | `configs/seed_omni/Janus/janus_1.3b/modules_train.yaml` |
| Eager inference module overrides | `configs/seed_omni/Janus/janus_1.3b/modules_infer_eager.yaml` |
| Distributed inference module overrides | `configs/seed_omni/Janus/janus_1.3b/modules_infer_fsdp.yaml` |
| Training graph | `configs/seed_omni/Janus/janus_1.3b/graph_train.yaml` |
| Text-to-image FSM | `configs/seed_omni/Janus/janus_1.3b/graph_infer_gen.yaml` |
| Image understanding FSM | `configs/seed_omni/Janus/janus_1.3b/graph_infer_und.yaml` |
| Interleave FSM | `configs/seed_omni/Janus/janus_1.3b/graph_infer_interleave.yaml` |

## How To Use

1. Pick the row that matches the task shape.
2. Read the live file.
3. Copy conventions, not names.
4. Verify against source if docs disagree.
