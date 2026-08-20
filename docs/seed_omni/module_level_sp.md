# Module-Level Sequence Parallel (SeedOmni V2)

> **架构决策（2026-07-20，已实现）：从 Arch A（per-module *looped* SP）迁移到 Arch B（uniform outer SP + classic single-pass Ulysses）。**
>
> **现状**：代码即 Arch B —— 全局统一 outer SP，dataloader 按 SP 组**复制**数据，每个开 SP 的模块把复制样本**切 1/sp**、跑**一次**前向、再 **all-gather** 回全序列。Arch A 的 looped/offload/ckpt/gather-to-owner/`fsdp2_ac_patch` 全部删除。
>
> **尚未实现（未来工作，见 §8）**：per-module data-balance（scatter item / gather embed）、dataloader 级 compute 均衡 packing、音视频 halo 处理。当前所有模块一律 uniform SP-slice。
>
> 实验数字（Arch A 时代，作为放弃 Arch A 的证据）：[sp_loop_memory_experiments.md](./sp_loop_memory_experiments.md)。配置约定：`.agents/skills/seedomni-v2/references/per-module-parallel.md`。

---

## 1. 为什么放弃 Arch A（looped per-module SP）

**Arch A**：outer SP 强制 =1，每卡持有一条 distinct sample；开 SP 的模块 loop `sp` 次（broadcast owner → slice → forward → gather-to-owner）。因为**整图只有一次 `loss.backward()`**，若保留 `sp` 份 sharded 激活，bwd 峰值会回到 ≈ 一条满样本；于是必须靠 **per-sample checkpoint（recompute 税）** 或 **同步 CPU offload（H2D/D2H 税）** 把峰值压回 ~1 条样本。

问题是**这个税压在了最贵、最慢的 LLM 上**：

- **实测**：Janus llamasp4 `offload_sync` 的同步拷贝 ≈ **49–60% 墙钟**（实验文档 §3.3），带宽受 per-tensor 延迟 + pageable 限制。
- **大模型上只增不减、且躲不掉**：激活字节随规模线性涨；MoE 激活/计算比高 → 相对占比也可能升；而大模型恰恰是**非开 SP 不可（OOM）**的场景 —— 税无法回避。
- **基线错觉**：Arch A 里 LLM 的 looped SP，真正的对照**不是「免费的 classic SP」**，而是「把不可避免的 SP 开销压在哪个模块」。压在 LLM 是压错了地方。

---

## 2. Arch A vs Arch B

| | **Arch A（已删）** | **Arch B（现，uniform outer SP）** |
|--|--|--|
| SP mesh | 每模块不同 SP size，outer=1 | **全局统一 outer SP，所有模块一致** |
| 数据 | 每卡 distinct sample，模块内 gather 组内 distinct 序列 | **dataloader 按 SP 组复制**（`dp=world/sp`），模块内切 1/sp |
| LLM | looped：`sp` 次前向 + gather + **offload/ckpt 税** | **classic Ulysses：每卡常驻 1/sp，一次前向一次反向，无 loop / 无 offload / 无 ckpt** |
| 输出 | gather-to-owner（每卡拿回自己那条） | **all-gather 回全序列**（每卡相同），`post_forward` SP-agnostic |
| 税落在 | **最贵的 LLM** | **便宜的 encoder（1/sp 切分 + 一次 all-gather）** |
| 复杂度 | loop + broadcast + gather-to-owner + zero-link + fsdp2_ac_patch + offload/ckpt | 标准 Ulysses（slice + all-gather）+ uniform SP 校验 |

**一句话：Arch B 把不可避免的 SP 开销从最贵的 LLM，挪到便宜的 encoder，并删掉一大票 Arch A 补丁。**

---

## 3. Arch B 设计（已实现部分）

### 3.1 Uniform outer SP + 复制数据

- 最外侧 `accelerator.ulysses_size` 承载**统一** SP size；各模块经 `build_module_runtime_args` 的 accelerator deep-merge **继承**该值。框架**不**校验一致性——模块 YAML 里写 per-module `ulysses_size` 覆盖会静默产生非 uniform SP，不要这么用。
- dataloader 用 `BaseTrainer` 的标准 build-time sharded loader：给出 `dp_size = world / sp` 条 **distinct** shard，并把每条 shard **复制**到其 SP 组的所有 rank（collator 不做按模态切分）。因此一个 SP 组内每卡持有**相同**样本。
- 每个开 SP 的模块在 `pre_forward` 内以 `if get_parallel_state().sp_size > 1:` 分支把复制样本**切 1/sp**（`sp_pad` + `slice_input_tensor` / `sp_pad_and_slice`），跑**一次**前向（attention 组内 all-to-all），再在 `post_forward` 内的同名分支 **all-gather** 回全序列（`gather_outputs`）。SP 逻辑完全收在模块自己的 `pre_forward` / `post_forward` 里（与 veomni v1 单模型 SP 一致，无独立 sp hook）。forward 与 backward 峰值均 ≈ `1/sp`。

### 3.2 梯度正确性

- `gather_outputs` 用 autograd-aware `_Gather`：backward 对 SP 组做 all-reduce(SUM) 再取本 rank 分片，引入的 `×sp` 被 FSDP2 的 `÷|dp_shard_sp|` 梯度平均抵消（参数在含 sp 的 mesh 上分片/规约），最终梯度与非 SP 基线一致 —— 与 VeOmni 单模型 Ulysses 同一不变式。
- decode/loss 侧：`reduce_sequence_parallel_loss(..., group=ps.fsdp_group)` 在 `dp_sp` 上 token 加权规约。复制数据下 SP 组各 rank 的 `(ce_sum, n_valid)` 相同，`dp_sp` 规约让每个 distinct DP shard 计一次、sp 副本抵消，结果与非 SP 基线一致（constraints 7b）。
- 计量：`metric_meter_set_seqlens` 在 `pre_forward`（切分前）记 FULL 样本长度；`OmniEnvironMeter` 在 `dp_group`（不含 sp）上规约，复制的 sp rank 不重复计（constraints 7c）。

### 3.3 encoder→LLM 边界

encoder 与 LLM 的输出都 all-gather 回**全序列**（每卡相同），embed 装配进 LLM 序列在全序列上进行，LLM 再自行按 token 切 1/sp。当前实现下 encoder / LLM 各自独立 slice + all-gather，边界处是普通的全序列装配，无额外 reshard 机制。

---

## 4. 关键洞察：SP-slice 与 data-balance 的关系

- **块对角模块**（每张图/每段独立，无 cross-item attention）+ 切分对齐 item 边界 → **SP-slice ≡ data-balance**：每个 item 只算一次、不触发 all-to-all。SigLIP / VQVAE 走的正是 batch 维 SP-slice（复制 batch → 切 1/sp 张图 → all-gather），等价于「每卡分一份图」的 data-balance。
- **负载均衡上 SP-slice 严格更优**：等 token 切 → 永远均衡，单图也四等分**无 bubble**，只需在总长补 ≤ `sp` 个 token 的 tiny dummy（一次，不是每图）。反而是 **data-balance（整 item 分配）才会 bubble**（单图 / item 数 < sp / item 不等大）。
- 因此：**若 encoder 是 SP-aware（全局注意力 + 无 halo，如 ViT），全局 SP-slice 一把梭即可，data-balance 冗余。** 这正是当前实现的选择：所有模块 uniform SP-slice。

---

## 5. 多模态（image / audio / video）：data-balance 的未来必要性

上一节的「data-balance 冗余」只在 image-only + SP-aware ViT 的友好场景成立。**含音视频时，data-balance 会从「可选回退」升回「一等机制」**（当前**未实现**，属未来工作）：

1. **局部/卷积/窗口结构破坏 SP-slice**：Whisper 类 audio 的 conv1d 下采样、video 的 3D conv patchify / 时序窗口，在 **all-to-all 之前**就作用在已分片序列上，shard 边界需 **halo 交换**，Ulysses all-to-all 给不了 → naive SP-slice 直接算错。**data-balance（整 item 留一个 rank、用未改动 encoder）绕开 halo。** ViT（非重叠 patchify + 全局注意力）才 SP-friendly。
2. **天然 item 并行**：视频/音频逐 clip / 逐帧独立，自然并行维度就是 item；item 多 → 可整除性好、bubble 小。
3. **全局 compute 均衡**（dataloader 级）：纯文本 vs 60s 视频 compute 差数量级 → uniform outer SP 下各 dp 组啃一条样本 → **DP straggler**。需按长度/compute 分桶 packing —— 另一种、且始终需要的 data-balance，与「组内分 item」正交。

---

## 6. 对「module-level SP 必要性」的收敛结论

- **「per-module 不同 SP size + outer=1 + loop」这个具体机制：已放弃并删除。**
- **「module-level 意识」仍必要**，但正确形态是：**每模块的并行策略（SP-slice / data-balance / replicate，由计算结构推出），在一个 uniform outer SP mesh 之下**。
- 当前实现只做了 **SP-slice**（所有模块一致）。真正需要 per-module 决策的，几乎只剩一个二元判断：**这个模块的 attention 能否/是否 SP-aware（全局+无 halo）** —— 能则 SP-slice，不能则 data-balance（未来）。

---

## 7. 权衡与决定性变量

- **dp_size 缩小**：uniform outer sp=S → `dp=world/S`。小卡数下 dp 可能 =1（无数据并行），需靠更大 global batch / grad-accum 补；大卡数（如 32 卡 sp=4→dp=8）健康。这是经典 Ulysses 常态。
- **单图 / 少 item 边角料**：SP-slice → tiny dummy（总长补齐）；data-balance → bubble。两者都**落在便宜的 encoder 阶段**，影响有界（LLM 永远干净）。
- **决定性变量**（Arch B 相对 Arch A 的收益，需真实数字确认）：
  1. 目标数据的 **items（图/clip）per sample 分布** —— 多媒体多 → Arch B 大胜；少 → 仍胜（税在便宜模块）。
  2. **encoder / LLM 计算占比** —— 只有 encoder 极重且普遍单 item 时，Arch A 才可能扳回。

---

## 8. 已删除的 Arch A 复杂度 & 未来工作

**已删除（Arch B 不再需要）**：

- `run_sp_looped_endpoint`（loop + `sp_broadcast_from_rank` + `sp_gather_to_owner` + zero-link fold）及 `dispatch.py` 相关机制。
- `fsdp_config.sp_activation_offload` 整套压 bwd 机制（`ckpt` / `offload_sync` + `_SyncCpuOffload` + profiler）与 `sp_keep_params_unsharded`。
- `veomni/distributed/fsdp2_ac_patch.py`（multi-forward AC recompute unshard，pytorch#171779 backport）。
- `sequence_parallel/data.py` 的 `sp_gather_seqs` / `sp_take_own_seq` / `sp_broadcast_from_rank` / `sp_gather_to_owner` / `_GatherConcatSP` / `_sp_unify_dtype` 等 Arch A 重分发原语。
- `OmniTrainer` 的 outer-SP=1 **硬禁**，改为**驱动** uniform outer SP（由各模块继承 outer `accelerator.ulysses_size` 达成）。

**未来工作（尚未实现）**：

- dataloader 级 **compute 均衡 packing**（跨样本 compute 方差）。
- encoder 的 **data-balance**（scatter item / gather embed）路径，用于 conv/局部/SP-unaware（音视频）模块。
- 音视频 encoder 的 halo 处理 / SP-unaware 判定与自动策略选择。

---

## 9. 现状数据流（Arch B）

```text
# 外层：dataloader 给 dp=world/sp 条 distinct shard，每条复制到 SP 组
for each node:
    kw  = module.pre_forward(**data)      # 若 sp_size>1：pad + 切 1/sp（Ulysses / batch 维）
    out = module(**kw)                    # 一次前向，attention 组内 all-to-all
    out = module.post_forward(**out)      # 若 sp_size>1：gather_outputs all-gather 回全序列 + 去 pad；再 SP-agnostic 全序列上跑
# 全图一次 loss.backward()；FSDP2 在含 sp 的 mesh 上规约梯度
```

- 入口：`veomni/models/seed_omni/accelerator/executor.py::execute_train_node`（`pre_forward → endpoint → post_forward`；`TrainingGraph` 只负责选节点。SP 收在模块 `pre_forward` / `post_forward` 的 `if sp_size>1` 分支里）。
- 原语：`slice_input_tensor` / `sp_pad` / `sp_pad_and_slice` / `gather_outputs`（`sequence_parallel/data.py`）。
- 细节见 `.agents/knowledge/constraints.md` §7-outer / §7a / §7b / §7c / §7d / §7e。
