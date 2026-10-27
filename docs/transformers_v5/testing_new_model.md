# Testing a New Model

When adding a model under `veomni/models/transformers/<model>/`, keep
the model-specific coverage beside the maintained implementation. The old
shared model-patch suite has been retired; new families should not add tests
under a separate legacy model tree.

## 1. Registry and model parity

Add one canonical tiny-config factory to `tests/models/tiny_configs.py`.
Reuse that factory from both places below instead of copying the config:

- `tests/models/base/test_auto_registry.py` lists the registered model
  type, supported architectures, aliases, and build prerequisites.
- The corresponding test under `tests/models/transformers/` covers the
  model's eager forward/backward parity and any family-specific behavior.

For a Qwen family, place the test under
`tests/models/transformers/qwen/`; otherwise use a family file or
subdirectory under `tests/models/transformers/`.

Keep implementation assertions capability-based. A test may prove that eager
and an optimized implementation can be selected, but it should not require a
particular YAML file to choose one fixed implementation.

If an on-disk HuggingFace checkpoint needs key or tensor-layout conversion,
implement the converter in `veomni/models/` and cover it in
`tests/models/base/test_checkpoint_tensor_converter.py`.

Run the focused model tests first:

```bash
pytest tests/models/base/test_auto_registry.py -k <model>
pytest tests/models/transformers -k <model>
```

## 2. Optimized operator coverage

Model tests should verify binding and model-level parity. Put numerical,
backward, availability, and hardware-guard tests for each optimized operator
under `tests/ops/`. Hardware-specific tests should skip with an explicit
prerequisite when the required accelerator or package is unavailable.

## 3. End-to-end parallel coverage

Add the model to `tests/e2e/test_e2e_parallel.py`. Text models use
`text_test_cases`; VLM and omni models use the matching family list and data
fixture. Set `max_sp_size=1` only when sequence parallelism is unsupported.
MoE cases should exercise the supported expert-parallel configurations.

The suite launches short FSDP2 training runs and compares loss and gradient
norm across SP/EP configurations:

```bash
pytest tests/e2e/test_e2e_parallel.py -k <model>
```

## 4. VLM and omni contracts

VLM models also need the trainer freeze test in
`tests/trainer/test_vlm_trainer.py`. Add the canonical tiny config to the
freeze-ViT case list and verify both `freeze_vit=False` and
`freeze_vit=True`.

Any model whose generated modeling overrides `dummy_forward` must add a case
to `tests/distributed/test_dummy_forward.py`. This 2-GPU test sends a
multimodal batch to one rank and text-only batches to the others, checking that
all ranks still participate in FSDP collectives.

```bash
pytest tests/trainer/test_vlm_trainer.py -k <model>
pytest tests/distributed/test_dummy_forward.py -k <model>
```

## Checklist

- [ ] One canonical tiny config added to `tests/models/tiny_configs.py`
- [ ] Registry case added to `tests/models/base/test_auto_registry.py`
- [ ] Family test covers eager forward/backward parity and model-specific contracts
- [ ] Optimized operators have focused coverage under `tests/ops/`
- [ ] E2E SP/EP case added to `tests/e2e/test_e2e_parallel.py`
- [ ] VLM: freeze-ViT case added to `tests/trainer/test_vlm_trainer.py`
- [ ] VLM/omni with `dummy_forward`: asymmetric 2-GPU case added
- [ ] `pytest --collect-only -k <model>` collects the expected tests
- [ ] Patch generation is current: `make check-patchgen`
- [ ] Focused tests and `make quality` pass
