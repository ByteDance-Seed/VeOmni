# Testing a New Model

This is the coverage policy for adding a model under
`veomni/models/transformers/<model>/`. Keep family-specific tests beside the
maintained implementation and reuse the shared model and E2E suites. Select
coverage by model structure, operator implementation, and supported parallel
modes; adding a model does not require repeating every backend combination.

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

## 2. Accelerator model parity and optimized operators

For each supported accelerator, cover one supported training configuration
against an independent reference, including loss and backward checks. Reuse
`tests/models/test_models_patch.py` for its supported families, or extend
`tests/models/base/test_low_precision_oracle.py` or a family test when the model
needs a different reference or attention path. The shared model-patch suite
still runs; eager family parity and low-precision training parity have
different purposes.

Choose the configuration that represents the model's supported training path
on that device. FA2 plus the device's fused MoE implementation is a common
choice, but models with DSA, GatedDeltaNet, or GPT-OSS-specific attention and
experts need their supported implementations. Do not force an unsupported
backend to make the matrix look uniform.

Additional attention versions, eager fallback, and alternate fused backends
use representative models chosen by the implementation they exercise. Cover
a new layout, activation, bias, or backward implementation explicitly. For
example, `moe_experts/standard/fused_quack` and
`moe_experts/gpt_oss/fused_quack` share Quack GEMM but have different expert
math and backward implementations; one does not qualify the other.

Put operator-level numerical, backward, availability, and hardware-guard tests
under `tests/ops/`. Cover each distinct EP/non-EP implementation and relevant
split/merged layouts there. An EP-local GEMM test does not exercise token
all-to-all or the complete distributed training path. Model tests own operator
binding and model-level parity; E2E tests own the integrated training path.

## 3. End-to-end parallel coverage

Add the model to `tests/e2e/test_e2e_parallel.py`. Text models use
`text_test_cases`; VLM and omni models use the matching family list and data
fixture. Reuse an existing test function and its launch-plan provider.

### Default matrix

Each pair below is `(SP, EP)`. The baseline has both SP and EP disabled;
FSDP2 remains enabled.

| Model | Modes | Processes per run | Assertion |
|---|---|---|---|
| Dense, with SP support | `(1,1)`, `(2,1)` | 4 | Loss and grad norm align to `(1,1)` |
| MoE, with SP and EP support | `(1,1)`, `(2,2)` | 4 | Loss and grad norm align to `(1,1)` |
| MoE, without SP support | `(1,1)`, `(1,2)` | 4 | EP alignment; document the SP restriction |
| MoE, without EP support | `(1,1)`, `(2,1)` | 4 | SP alignment; document the EP restriction |

The two-mode MoE default checks the combined SP+EP configuration. It does not
isolate SP and EP independently. Shared parallel primitives and the
representatives below provide that additional coverage. A model with a
distinct implementation may need extra modes, with a stated reason.

### Representatives and capability exceptions

| Case | Modes | Processes per run | Reason |
|---|---|---|---|
| `qwen3_moe` | `(1,1)`, `(1,2)`, `(2,1)`, `(2,2)` | 4 at SP1, 8 at SP2 | Shared full-matrix representative; preserves the larger topology |
| `gpt_oss` | `(1,1)`, `(2,1)`, `(2,2)` | 4 | Dedicated GPT-OSS Quack variant; baseline plus SP and EP configurations |

GPT-OSS compares all three runs against the `(1,1)` baseline. Both EP and
non-EP experts use Quack; they have separate custom backward implementations.
That split also exists in standard Quack and does not require every MoE model
to run three modes. Additional model/backend representatives should be added
only when existing coverage misses a distinct implementation or topology.

Capability exceptions are separate from these representatives. For example,
GLM-MoE-DSA sets `max_sp_size=1` because its attention/indexer path does not
support SP. DeepSeek-V4 has a dedicated SP1 EP-alignment test plus SP2 smoke
tests for other paths. Record the supported configuration, the missing
capability or assertion, and any complementary test; do not describe a smoke
test as numerical alignment.

### Numerical and resource contract

Alignment runs use the same initial weights, data, global batch size, and
optimizer settings, with two optimizer steps. At fixed four-process world
size, changing SP changes the DP size; the training callbacks reduce losses
with `dp_mean` before comparison. Compare loss and grad norm with an explicit,
justified tolerance. These aggregate metrics do not prove per-parameter
gradient/update equivalence or convergence.

Keep `compare_alignment=True` for alignment cases. A successful process exit
and nonempty logs only establish smoke coverage. Use smoke only when it is
the intended contract, explain why, and identify any remaining alignment gap.

The mode list in `tests/e2e/utils.py::parallel_launches` drives both commands
and resource plans. Modes execute sequentially within a case. Ordinary cases
reserve four devices; adding a model must not introduce an eight-device
reservation unless the tested topology needs it. A new launcher must declare
the same process count in its `pytest.mark.launch_plan` and worker startup.

## 4. Hardware eligibility and CI collection

Declare support and prerequisites independently for GPU and NPU. Check
hardware eligibility before importing an optional accelerator kernel, and
skip with a concrete reason when its device or package is unavailable.
GPT-OSS FA4/Quack E2E requires SM90+ CUDA, FA4, and Quack; it skips on SM89
GPU runners and NPUs. A green job that skips a case is not execution evidence
for that backend. Record hardware, executed modes, and skips in validation.

Check the selections and filters in
`scripts/ci/accelerator/suites/gpu.json` and `npu.json`. A case added under an
already-selected file/directory normally needs no manifest edit, but a new
file outside those selections does. See
[Registering Tests in CI](../testing.md#registering-tests-in-ci) and
[launch-plan authoring](../design/ci_case_scheduling.md).

Collect the exact family/test IDs before running. E2E model names can differ
from registry names, especially for VLMs:

```bash
pytest tests/models --collect-only -q -k <model>
pytest tests/e2e/test_e2e_parallel.py --collect-only -q -k <model>
pytest tests/e2e/test_e2e_parallel.py -k <model>
```

## 5. VLM and omni contracts

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
- [ ] One supported accelerator training configuration has loss/backward parity coverage
- [ ] Distinct optimized variants/backends have focused forward/backward coverage under `tests/ops/`
- [ ] E2E modes follow the default matrix or document a representative/capability exception
- [ ] Alignment compares loss and grad norm; any smoke-only gap is explicit
- [ ] Launch plan matches worker process counts and sequential execution
- [ ] GPU/NPU selections, prerequisites, executed cases, and skips checked
- [ ] VLM: freeze-ViT case added to `tests/trainer/test_vlm_trainer.py`
- [ ] VLM/omni with `dummy_forward`: asymmetric 2-GPU case added
- [ ] `pytest --collect-only -k <model>` collects the expected tests
- [ ] Patch generation is current: `make check-patchgen`
- [ ] Focused tests and `make quality` pass
