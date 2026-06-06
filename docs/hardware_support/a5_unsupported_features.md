# A5 (Ascend 950) Features Pending Validation

This document lists features and scenarios on Ascend 950 (A5) products that have **not yet been fully validated** in VeOmni.

> For the full hardware support matrix, see [Get Started with Ascend NPU](get_started_npu.md).

## Validation Status

All existing unit tests (UT) and system tests (ST) in VeOmni have been verified to pass on A5 hardware. The features listed below are **theoretically supported** but have not yet undergone the same level of validation as on A2 (910B) products.

## Scenarios Pending Further Validation on A5

| Feature / Scenario | Current Status | Notes |
|---|---|---|
| Multi-node Expert Parallelism (EP) for MoE models > 8B | Pending validation | A5 inter-chip bandwidth differs from A2 (910B); multi-node EP for large MoE models requires additional verification |
| Wan2.1 (1.3B) training | Pending validation | Wan2.1 on NPU uses the FSDP1 backend; validation on A5 is pending |
| Training models > 8B parameters | Pending validation | A5 HBM capacity differs from A2; large model training requires additional verification |
| Sequence length > 32K for models ≥ 30B | Pending validation | A5 memory bandwidth and capacity for very long sequences require additional verification |
| `flash_attention_2` Triton backend | Pending validation | The Triton-based flash attention backend availability on A5 is pending verification |
| `liger_kernel` MoE backend | Pending validation | Liger-Kernel support on A5 hardware is pending verification |
| Multi-node training with > 8 nodes | Pending validation | A5 cluster configurations beyond 8 nodes require additional verification |

## NPU Operator Implementation Recommendations for A5

When running on A5, the following operator configurations are recommended until further validation is complete:

```yaml
model:
  ops_implementation:
    attn_implementation: "sdpa"
    moe_implementation: "fused_npu"
    cross_entropy_loss_implementation: "npu"
    rms_norm_implementation: "npu"
    rotary_pos_emb_implementation: "npu"
    swiglu_mlp_implementation: "eager"
    load_balancing_loss_implementation: "eager"
    rms_norm_gated_implementation: "eager"
```

> **Note**: The above configuration is recommended for A5 based on current validation status. As validation progresses, additional operator backends may become available on A5.

## Verification Status

| Category | A5 Status |
|---|---|
| Unit Tests (UT) | ✅ All passed |
| System Tests (ST) | ✅ All passed |
| CI Runner | A5 self-hosted runner (in preparation) |

For questions about A5 support, please open an [issue](https://github.com/ByteDance-Seed/VeOmni/issues) with the `ascend-npu` label.
