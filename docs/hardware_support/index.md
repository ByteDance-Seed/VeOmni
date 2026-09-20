# Hardware Support

Select the environment for your accelerator. Platform guides describe their own tested revisions and limitations; they do not imply that every model and kernel combination is validated on current main.

## Environment and validation scope

| Platform | Installation / environment | Evidence and limits |
| --- | --- | --- |
| NVIDIA | [GPU installation](../get_started/installation/install.md), `gpu` extra | Locked CUDA environment; model/kernel tests have hardware-specific skips. A passing toy test is not a full-size training certification. |
| Ascend x86 / ARM | [x86](../get_started/installation/install_ascend_x86.md), [ARM](../get_started/installation/install_ascend_arm.md) | Separate `npu` / `npu_aarch64` extras. See recipe-specific NPU kernel settings and [container tags](AscendDockerUsage/supported_tags.md). |
| AMD ROCm | [ROCm guide](rocm/README.md) | Guide reports validation on 8×MI308X at a named revision. Its container uses a different torch/Transformers stack from current main; review the stated checkpoint/kernel limits before upgrading. |
| Cambricon MLU | [MLU guide](mlu/README.md) | Vendor-provided environment and MLU-specific MoE backends; the guide does not establish a model-by-model current-main validation matrix. |

Follow each guide's dependency instructions. Do not apply the NVIDIA uv extra
to a vendor-provided ROCm or MLU environment. For model configurations, use the
[recipe catalog](../models/index.md).

```{toctree}
:maxdepth: 1
:caption: AMD and Cambricon

rocm/README
mlu/README
```

```{toctree}
:maxdepth: 1
:caption: Ascend operations

get_started_npu
typical_usage
npu_variables
precision_analysis
profiling_analysis
FAQ
```

```{toctree}
:maxdepth: 1
:caption: Ascend containers

AscendDockerUsage/overview
AscendDockerUsage/supported_tags
AscendDockerUsage/build_a2_docker
AscendDockerUsage/build_a3_docker
```
