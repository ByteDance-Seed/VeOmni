# Ascend Docker Supported Tags

A full list of tags published at
[quay.io/ascend/veomni](https://quay.io/repository/ascend/veomni?tab=tags).

## Tag Naming Schemes

Two tag schemes coexist on quay.io:

- **Product-based** (current): `<veomni_version>-cann<CANN>-torch_npu<torch_npu>-<chip_series>-<os>-py<python>-veomni`
- **Legacy**: `veomni-<cann>-<chip_series>-<os>-py<python>[-torch<torch>][-<suffix>]`

> Product-based tags use dedicated CANN / torch-npu Dockerfiles that include the
> triton-ascend and fla_npu GDN stack. General-purpose Dockerfiles remain
> available for the legacy tags.

## CANN 9.2.1

### Product-based tags

| Tag | Dockerfile | Content |
|---|---|---|
| `v0.1.12-cann9.2.1-torch_npu2.10.0.post7-910b-ubuntu22.04-py3.12-veomni` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_910b.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_910b.arm) | veomni / torch-npu / triton-ascend / fla_npu |
| `v0.1.12-cann9.2.1-torch_npu2.10.0.post7-a3-ubuntu22.04-py3.12-veomni` | [Dockerfile] [Atlas A3 training products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a3) | veomni / torch-npu / triton-ascend / fla_npu |
| `v0.1.12-cann9.2.1-torch_npu2.10.0.post7-a5-ubuntu22.04-py3.12-veomni` | [Dockerfile] [Ascend 950PR&950DT products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5) | veomni / torch-npu / triton-ascend / fla_npu |

## CANN 9.1.0

### Product-based tags

| Tag | Dockerfile | Content |
|---|---|---|
| `v0.1.11-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12-veomni` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.arm) | veomni / torch-npu / triton-ascend / fla_npu |
| `v0.1.11-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12-veomni` | [Dockerfile] [Atlas A3 training products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_a3) | veomni / torch-npu / triton-ascend / fla_npu |

## CANN 9.0.0

### Product-based tags

| Tag | Dockerfile | Content |
|---|---|---|
| `v0.1.11-cann9.0.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.11-veomni` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_910b.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_910b.arm) | veomni / torch-npu / triton-ascend / fla_npu |
| `v0.1.11-cann9.0.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.11-veomni` | [Dockerfile] [Atlas A3 training products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_a3) | veomni / torch-npu / triton-ascend / fla_npu |
| `v0.1.11-cann9.0.0-torch_npu2.10.0.post2-A2-ubuntu22.04-py3.11-veomni-latest` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_910b.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_910b.arm) | veomni / torch-npu / triton-ascend / fla_npu |
| `v0.1.11-cann9.0.0-torch_npu2.10.0.post2-A3-ubuntu22.04-py3.11-veomni-latest` | [Dockerfile] [Atlas A3 training products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_torch_npu2.10.0.post2_a3) | veomni / torch-npu / triton-ascend / fla_npu |

### Legacy tags

| Tag | Dockerfile | Content |
|---|---|---|
| `veomni-9.0.0-910b-ubuntu22.04-py3.11-latest` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.arm) | veomni / torch-npu |
| `veomni-9.0.0-a3-ubuntu22.04-py3.11-latest` | [Dockerfile] [Atlas A3 training products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3) | veomni / torch-npu |
| `veomni-9.0.0-910b-ubuntu22.04-py3.11-torch2.10.0-latest` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.arm) | veomni / torch-npu |
| `veomni-9.0.0-a3-ubuntu22.04-py3.11-torch2.10.0-latest` | [Dockerfile] [Atlas A3 training products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3) | veomni / torch-npu |
| `veomni-9.0.0-910b-ubuntu22.04-py3.11-torch2.9.0-latest` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2.arm) | veomni / torch-npu |
| `veomni-9.0.0-a3-ubuntu22.04-py3.11-torch2.9.0-latest` | [Dockerfile] [Atlas A3 training products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3) | veomni / torch-npu |
| `veomni-9.0.0-a3-ubuntu22.04-py3.11-torch2.7.1-latest` | [Dockerfile] [Atlas A3 training products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3) | veomni / torch-npu |

## CANN 8.3.rc2

### Legacy tags

| Tag | Dockerfile | Content |
|---|---|---|
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-latest` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
| `veomni-8.3.rc2-a3-ubuntu22.04-py3.11-latest` | [Dockerfile] [Atlas A3 training products](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a3) | veomni / torch-npu |
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-torch2.7.1-latest` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-v0.1.9a5` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-v0.1.9a4` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-v0.1.9a3` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
| `veomni-8.3.rc2-910b-ubuntu22.04-py3.11-v0.1.9a2` | [Dockerfile] [Atlas A2 training products (x86)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.x86) / [Atlas A2 training products (ARM)](https://github.com/ByteDance-Seed/VeOmni/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc2_a2.arm) | veomni / torch-npu |
