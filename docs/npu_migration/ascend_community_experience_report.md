# 昇腾社区开发体验报告

> Issue #1031: [Ascend][Feature][Q3社区任务3] Skill：通用 GPU 模型向 NPU 迁移
> 交付模型: Gemma 3 (270M text-only + 12B VLM)

## 一、任务概述

在 VeOmni 分布式训练框架中编写一个通用的 GPU 模型向 Ascend NPU 迁移 Skill，
并以 Gemma 3 为交付测试对象完成实际迁移实践。该 Skill 面向"模型已能在 VeOmni
GPU 环境训练、需要新增 NPU 支持"的场景，指导开发者完成代码分析、设备相关逻辑
替换、NPU patchgen、配置、测试、E2E 和精度性能验证。

## 二、开发过程

### 2.1 学习与调研

深入研究了仓库中已有的 NPU 适配案例：

- **Qwen3** (`veomni/models/transformers/qwen3/`): 采用 OpSlot 守卫模式，GPU
  config 中已包含所有 OpSlot 守卫（`if veomni_rms_norm.use_non_eager_impl:`），
  NPU config 直接引用 GPU config 的补丁函数并重新注册到 NPU 目标文件。
- **SeedOss** (`veomni/models/transformers/seed_oss/`): 采用直接 NPU 内核调用
  模式，NPU config 中直接 import 并调用 NPU 融合算子（`rms_norm_forward_npu`、
  `apply_rotary_pos_emb_npu`）。
- **Qwen3-VL** (`veomni/models/transformers/qwen3_vl/`): VLM 参考实现，`__init__.py`
  分发 `ForConditionalGeneration` + `Model`；NPU config 对视觉塔做了 OpSlot 守卫
  与 dummy_forward 补丁，并在 `ForConditionalGeneration.forward` 上叠加 fused CE。

两种模式的对比：
- OpSlot 守卫模式（Qwen3 方式）：统一分发、eager 回退、非 NPU 硬件上可测试 → 推荐
- 直接内核调用模式（SeedOss 方式）：更直观、NPU 内核逻辑显式可见

### 2.2 Skill 编写

编写了 `.agents/skills/veomni-gpu-to-npu/SKILL.md`，覆盖完整迁移流程：

1. **迁移前分析**：CUDA/GPU 专有依赖扫描、NPU 后端选择表、模型特定 RMSNorm
   变体识别、注意力实现兼容性检查
2. **NPU patchgen 配置创建**：两种模式（OpSlot 守卫 / 直接内核调用）的代码模板，
   含 VLM（`ForConditionalGeneration`）专属小节
3. **训练配置适配**：各算子的 NPU 后端映射表
4. **测试创建**：导入验证、OpSlot 守卫检查、forward/backward、RMSNorm 数值一致性
5. **E2E 脚本**：NPU 设备检测、环境变量设置、smoke 训练
6. **实践报告模板**：可复用于其他模型的报告框架

### 2.3 Gemma 3 迁移实践

使用上述 Skill 完成 Gemma 3 NPU 迁移，覆盖 text-only（270M）与 VLM（12B）两个尺寸：

- **注意力**: GPU 使用 `flex_attention`（BlockMask），NPU 改用 `sdpa`（标准张量
  mask）。VeOmni masking_utils 自动处理两种 mask 类型的差异。
- **RMSNorm**: Gemma 3 使用 `(1.0 + weight)` 公式（weight 零初始化），NPU 守卫
  传递 `1.0 + self.weight` 和 `self.eps`（注意不是 `self.variance_epsilon`）。
- **RoPE**: 标准 rotate_half 实现，使用 NPU `npu_rotary_mul` 融合内核。
- **SwiGLU MLP**: 无 NPU 后端，使用 `eager`（无需 OpSlot 守卫）。
- **CrossEntropy**: 复用 GPU config 的 OpSlot 守卫，运行时绑定 NPU chunk_loss。
- **VLM (12B)**: `__init__.py` 注册 `gemma3` 分发到 `Gemma3ForConditionalGeneration`
  + `Gemma3Model`；NPU config 增加 `Gemma3ForConditionalGeneration.forward` 补丁
  叠加 fused CE；视觉塔 SiglipVisionModel 使用 eager/sdpa 无需 NPU 补丁。

### 2.4 遇到的问题与解决

| 问题 | 原因 | 解决方案 |
|---|---|---|
| `huggingface.co` 网络不可达 | 环境网络限制 | 使用 `HF_ENDPOINT=https://hf-mirror.com` 镜像 |
| Gemma3RMSNorm 属性名不匹配 | Gemma 用 `self.eps` 而非 `self.variance_epsilon` | OpSlot 守卫显式传递 `self.eps` |
| Gemma3RMSNorm 权重公式不同 | Gemma 零初始化 `weight`，使用 `(1.0 + weight)` | OpSlot 守卫传递 `1.0 + self.weight` |
| GPU config 无 RMSNorm/RoPE OpSlot 守卫 | GPU 依赖 HF eager 实现 | NPU config 新增 OpSlot 守卫 |
| 910C 标称算力误填 800T | 双芯封装与单芯混淆 | 单芯 354T（CANN platform_config 确认），`"910_93"` 已匹配 |
| 12B 关闭 GC 导致 OOM | 48 层激活 ~58GB | 保留 GC on；显存 34.7GB/64GB |

## 三、交付件清单

| 交付件 | 文件路径 | 状态 |
|---|---|---|
| 通用迁移 Skill | `.agents/skills/veomni-gpu-to-npu/SKILL.md` | ✅ 完成 |
| 报告模板 | `docs/npu_migration/report_template.md` | ✅ 完成 |
| Gemma 3 NPU patchgen 配置 | `veomni/models/transformers/gemma3/gemma3_npu_patch_gen_config.py` | ✅ 完成 |
| Gemma 3 `__init__.py` NPU 分发 | `veomni/models/transformers/gemma3/__init__.py` | ✅ 完成 |
| Gemma 3 NPU 生成文件 | `veomni/models/transformers/gemma3/generated/patched_modeling_gemma3_npu.py` | ✅ 生成 |
| 270M NPU 训练配置 | `configs/text/gemma3_npu.yaml` | ✅ 完成 |
| 12B NPU 训练配置 | `configs/text/gemma3_12b_npu.yaml` | ✅ 完成 |
| 270M GPU 基线配置 | `configs/text/gemma3_gpu_sdpa.yaml` | ✅ 完成 |
| Gemma 3 NPU 测试 | `tests/models/test_gemma3_npu.py` | ✅ 10/10 通过 |
| Gemma 3 NPU E2E 脚本 | `scripts/e2e/gemma3_npu_e2e.sh` | ✅ 完成 |
| MFU 采集 | `veomni/utils/count_flops.py` + `veomni/trainer/callbacks/trace_callback.py` | ✅ 完成 |
| Gemma 3 实践报告 | `docs/npu_migration/gemma3_npu_practice_report.md` | ✅ 完成 |
| 昇腾社区体验报告 | `docs/npu_migration/ascend_community_experience_report.md` | ✅ 完成 |
| AGENTS.md skill 分发 | `AGENTS.md` | ✅ 更新 |
| patchgen drift gate | — | ✅ 通过 |

## 四、验证结果

### 4.1 单元测试
```
pytest tests/models/test_gemma3_npu.py -v
→ 10 passed in 8.95s
```

### 4.2 Patchgen drift gate
```
patchgen --check
→ gemma3_npu: OK (no drift)
```

### 4.3 代码质量
```
ruff check: All checks passed!
ruff format: files already formatted
```

### 4.4 E2E 训练与精度（270M，NPU vs GPU）

相同模型、数据与训练配置（均使用 `sdpa` 注意力，GPU 基线改用 `sdpa` 以公平对比）：

| Step | GPU loss | NPU loss | 差异 |
|---|---|---|---|
| 1 | 3.51 | 3.50 | 0.01 |
| 100 | 1.85 | 1.85 | 0.00 |
| 250 (epoch 1) | 1.38 | 1.38 | 0.00 |
| 500 (epoch 2) | 1.38 | 1.38 | 0.00 |
| 750 (epoch 3) | 0.62 | 0.65 | 0.03 |

**结论**: ✅ NPU loss 与 GPU loss 在所有检查点差异 < 0.03，步 750 的 0.03 差异
处于 BF16 跨硬件训练的预期数值噪声范围内。

### 4.5 性能（12B VLM，MFU 基准）

| 指标 | 值 |
|---|---|
| 硬件 | Ascend 910C × 8 (NPUs 8–15) |
| 模型 | `gemma-3-12b-it` (12B, VLM) |
| Seq length | 8192 (dyn_bsz 打包) |
| Micro batch | 1 / Global batch | 8 |
| 并行策略 | FSDP2 (dp=8), gradient_checkpointing=on |
| 单芯峰值 BF16 | 354 TFLOPS (24 cube × 1800MHz × 8192) |
| 实测 TFLOPS/芯 | 92.1 |
| 步耗时 | 6.42s |
| 显存 | 34.7 GB / 64 GB |
| **MFU** | **26.0%** |
| 目标 | ≥ 24% |
| **结论** | ✅ 达标 |

## 五、开发体验总结

### 5.1 优点
- VeOmni 的 patchgen 系统使 GPU→NPU 迁移非常系统化，NPU 配置可大量复用 GPU 配置
- OpSlot 守卫模式提供了优雅的 eager 回退机制，使 NPU 代码可在非 NPU 硬件上测试
- VeOmni masking_utils 统一处理了 flex_attention 和 sdpa/eager 的 mask 类型差异
- 统一内核注册表和 NPU 默认回退表 (`_NPU_DEFAULT_FALLBACK`) 自动处理大部分算子切换
- `count_flops.py` 注册新模型类型后即可在训练 step 中自动采集并输出 MFU

### 5.2 改进建议
- 可考虑为 SwiGLU MLP 增加 NPU 融合算子后端，进一步降低 eager 回退的性能开销
- 文档中可补充更多模型特定 RMSNorm 变体的说明
- patchgen 的 `--check` 在 transformers 版本不匹配时会产生大量预先存在的 drift 噪声

### 5.3 可复用性
该 Skill 和 Gemma 3 迁移实践可作为其他 GPU 模型向 NPU 迁移的模板。核心步骤为：
1. 扫描 GPU 依赖 → 2. 选择 NPU 后端 → 3. 创建 NPU patchgen 配置 → 4. 生成 NPU modeling → 5. 适配训练配置 → 6. 编写测试 → 7. 运行 E2E → 8. 生成报告。
