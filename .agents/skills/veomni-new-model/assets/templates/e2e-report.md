# {{MODEL_NAME}} VeOmni integration report

Status: TBD (PASS, FAIL, or BLOCKED)
Scope: TBD (requested inference/training, components, backend/topology)

Use only sections applicable to the task. Mark omitted checks N/A with a reason;
mark missing prerequisites for required checks BLOCKED. A partial check does not
establish full-model parity, training resume, or hardware support.

## Source and target

| Item | Value |
|---|---|
| Source repository/revision and local modifications | TBD |
| Components and implementation frameworks | TBD |
| Source and weight licenses | TBD |
| Weights/revision or checksum, when used | TBD |
| VeOmni revision | TBD |
| Python/framework/runtime and relevant dependencies | TBD |
| Claimed backend/topology | TBD |

## Integration decisions

- Source-to-VeOmni interface mapping and changed files: TBD
- Direct loading or necessary conversion, with rationale: TBD
- Data/condition/model boundaries: TBD
- Deliberate differences from source behavior: TBD

## Reproduction

- Documented input/fixture and provenance: TBD
- Model/data/configuration inputs: TBD
- Environment setup and exact run commands: TBD
- Logs or small result artifacts: TBD

## Applicable checks

| Check | Applies/reason | Command/artifact | Criterion | Observation | Status |
|---|---|---|---|---|---|
| Import/config/forward | Always | TBD | TBD | TBD | TBD |
| Patchgen | Generated modeling changed | TBD | TBD | TBD | TBD |
| Weight loading | Weights used | TBD | TBD | TBD | TBD |
| Conversion | Conversion needed | TBD | TBD | TBD | TBD |
| Source parity | Implementation/layout/operations adapted | TBD | TBD | TBD | TBD |
| Real data path and training | Training in scope | TBD | TBD | TBD | TBD |
| Convergence | Loss evidence requested | TBD | TBD | TBD | TBD |
| Fresh-process resume | Resume requested or state handling changed | TBD | TBD | TBD | TBD |
| Export | Export in scope | TBD | TBD | TBD | TBD |
| Backend/distributed run | Support claimed | TBD | TBD | TBD | TBD |
| Performance | Requested or affected | TBD | TBD | TBD | TBD |
| Quality/focused verification | Code changed | TBD | TBD | TBD | TBD |

## Detailed evidence, when applicable

- Weight mapping: source/target counts, drops/initialization, fingerprints, strict load.
- Parity: matched weights/inputs/mode, tensor names/shapes/dtypes, observed errors/tolerances.
- Training: steps, loss/gradient series, optimizer update; acceptance rule if convergence claimed.
- Resume: restored state, next-batch identity, control/resumed loss and next update.
- Performance: workload, hardware/runtime, warmup/synchronization, time, throughput, memory.

Replace these prompts with evidence for applicable checks, or remove them.

## Supported behavior and remaining limitations

TBD: state only claims supported by the observations above, with explicit blocked checks.
