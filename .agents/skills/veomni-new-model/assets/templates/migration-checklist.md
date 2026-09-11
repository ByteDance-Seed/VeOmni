# {{MODEL_NAME}} integration checklist

Source revision: {{UPSTREAM_REVISION}}
Requested behavior and backend/topology: TBD

Use this checklist for a substantial integration. Set each item to PASS, FAIL,
BLOCKED, or N/A with evidence or a reason. Remove inapplicable detail; do not
turn the template into additional requirements.

| Check | Applies when | Status and evidence/reason |
|---|---|---|
| Source revision, license, dependencies | Every integration | TBD |
| Actual model/component routes and interface mapping | Every integration | TBD |
| Forward contract and nearest VeOmni implementation | Every integration | TBD |
| Config, registry, and clean-process forward | Every integration | TBD |
| Patchgen generation and applicable patch tests | Transformers generated modeling changes | TBD |
| Weight source, format, required key/shape coverage | Pretrained weights are used | TBD |
| Conversion accounting and strict target load | Conversion is necessary | TBD |
| Fixed-input source comparison with declared tolerances | Model semantics/layouts/operations are ported or changed | TBD |
| Real-format preprocessing and collation | Data integration is in scope | TBD |
| Cache provenance and compatibility | Cached conditions are introduced | TBD |
| Finite loss/gradients, optimizer update, short trainer run | Training is in scope | TBD |
| Controlled loss experiment | Convergence evidence is requested | TBD |
| Fresh-process resume compared with uninterrupted next step | Resume is requested or checkpoint/state handling changes | TBD |
| Exported weights load and run in intended consumer | Export is in scope | TBD |
| Actual backend/topology execution | Backend/distributed support is claimed | TBD |
| Comparable performance measurement | Performance is requested or affected | TBD |
| Quality checks and focused verification | Every code change | TBD |
| Reproduction commands and supported-model documentation | Applicable supported behavior | TBD |
| E2E or Ascend report | Requested or useful for substantial end-to-end/backend work | TBD |

## Decisions and limitations

- Chosen per-component route and source evidence: TBD
- Changed files and their responsibilities: TBD
- Intentional semantic differences: TBD
- Required checks blocked by unavailable inputs/hardware: TBD

Replace placeholders in applicable sections before delivery. N/A is not a
substitute for BLOCKED when a required check cannot run.
