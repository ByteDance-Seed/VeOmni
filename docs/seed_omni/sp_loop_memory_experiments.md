# Looped per-module SP：显存 / 激活实验记录

> **⚠️ 历史文档（2026-07-20 已废弃该实现）：本文记录的 looped per-module SP 是 Arch A，代码已删除并迁移到 Arch B（uniform outer SP + classic single-pass Ulysses，见 [module_level_sp.md](./module_level_sp.md)）。** 下面的显存/时间数字**仍作为「促成放弃 Arch A 的动因证据」保留**：它们量化了把 SP 税压在最贵 LLM 上的代价（尤其 §3.3 的同步拷贝 ≈49–60% 墙钟）。文中提到的 `run_sp_looped_endpoint` / `sp_activation_offload` / `fsdp2_ac_patch` 等均已不存在。
>
> 日期：2026-07-19 · 机型：8×A100-80GB · 框架：VeOmni SeedOmni V2 · 对照模型：Janus-1.3B  
> 目标（Arch A 语境）：outer SP=1、module `ulysses_size=4` 时，**forward 与 backward 峰值显存都降到约 1/4**（相对「每卡只跑自己那条满长 sample」）。  
> 设计与图例：[module_level_sp.md](./module_level_sp.md)。

---

## 1. V1 vs V2 语义对比（先对齐理解）

| | **V1（经典 Ulysses / 整段切分）** | **V2（looped per-module SP，当前）** |
|--|--|--|
| 数据布局 | 组内样本先拼/对齐后，**每卡永久只持有序列的 1/sp** | outer SP=1：每卡一条**完整** distinct sample；loop 时广播 → 切成 L/sp 共 forward |
| Forward 激活 | 天然 ≈ **1/sp**，**不依赖** activation-checkpoint / offload | 每轮 fwd 工作集 ≈ 1/sp；但 loop `sp` 次后，若图不释放，存留 ≈ **sp × (1/sp) = 一整条** |
| Backward | 一次 bwd，激活仍 ≈ 1/sp | 只有**一次** `loss.backward()`；若 4 份 boundary/activation 仍在 GPU → bwd 峰值 ≈ 满样本 |
| 与 GC 的关系 | SP 能跑时通常**不必**开 layer GC | 压 bwd 有两条已验证路径：per-sample ckpt（recompute）或 **同步 blanket CPU offload（无 recompute）**；两者都**不依赖** HF layer GC，layer GC 可关 |
| Offload | 一般不需要靠 offload 维持 1/sp | loop 内 `saved_tensors_hooks`→CPU：**同步版梯度正确且峰值更低**；**异步 prefetch + 跨 stream 缓冲版会静默坏梯度**（caching-allocator 跨流复用踩内存，非 hook 机制本身）见 §3 |

一句话：

- V1：切分后「人手里永远只有 1/sp」，**不开 activation offload / GC 也能维持约 1/4 显存**。
- V2：loop 四次共算四条样本；**若不做 per-sample 释放（ckpt / 真·分次 bwd / 可靠 offload），bwd 显存回不到 1/4**。  
  这和「每次只取 1/sp 数据算一遍」在**单次迭代工作集**上类似，但在**一次 backward 前 GPU 上留着几份**这一点上不如 V1。

---

## 2. 公平对照定义（显存 bench）

脚本：`scripts/seed_omni/_bench_sp_bwd_mem.py`  
代理激活：`FatAct`，体积 ∝ `seq_len`（`L=32768, H=1024, blowup=24, sp=4`）。

| 模式 | 含义 |
|--|--|
| **nosp** | 每卡只对自己那条长度 L 的 sample 做 fwd+bwd（基线满样本） |
| **SP no ckpt** | 4×（全员共 fwd 一条 @ L/4 + gather-to-owner）→ 一次 bwd |
| **SP + per-sample ckpt** | 同上，但每个 sample 的 module forward 包在 `torch.utils.checkpoint` 里（**当前生产路径**） |

理论 FatAct：`full_sample ≈ 1.50 GiB`，`shard ≈ 0.375 GiB`。

---

## 3. 实验结果一览

### 3.1 合成激活（FatAct）显存

| 配置 | fwd 峰值 | 相对 nosp fwd | after_fwd / after_loop | bwd 峰值 | 相对 nosp bwd | 结论 |
|--|--|--|--|--|--|--|
| nosp | 1.891 GiB | 1.00× | 1.766 | 2.266 GiB | 1.00× | 基线 |
| SP no ckpt | iter0 0.719 / δ 0.578 | **0.38× / 0.31×** | 1.750 | 2.438 GiB | **1.08×** | fwd 已降；**bwd 几乎不降**（4 份 shard act ≈ 1 条满样本） |
| SP + per-sample ckpt（现实现） | iter0 0.609 | **0.32×** | 0.500 | 1.188 GiB | **0.52×** | fwd≈1/4 量级；bwd 明显下降（合成场景因 recompute+残留 out，难到严格 0.25×） |

### 3.2 Janus e2e（4 GPU，`max_steps=2`，GC off，`micro_batch_size=4`）

| 配置 | modules | loss | grad_norm | e2e 峰值显存 | recompute | 稳定性 | 备注 |
|--|--|--|--|--|--|--|--|
| SP + per-sample ckpt（现生产路径） | `modules_train_sp.yaml`（llamasp4） | 9.38 → 8.62 | **43.9 → 50.75** | **21.40 GB** | 有 | OK | 与改前数值对齐，梯度健康 |
| **SP + 同步 blanket CPU offload** | llamasp4 | 9.38 → 8.62 | **43.93 → 50.73** | **16.87 GB** | 无 | OK | 配置项 `fsdp_config.sp_activation_offload: offload_sync`（env `VEOMNI_SP_OFFLOAD_MODE` 可临时覆盖）；grad 与 ckpt 逐位对齐，峰值更低、无 recompute |
| SP + per-sample ckpt | `modules_train_allsp4.yaml` | ~7.56 → ~7.59（或同类） | 正常量级 | — | 有 | EXIT 0 | 需 FSDP2 AC recompute unshard patch（`fsdp2_ac_patch.py`） |
| SP + loop 内 CPU offload，**异步 prefetch + 跨 stream 缓冲** | llamasp4 / allsp4 | loss 仍像 9.38→8.62 | **2e3 → 1e5~1e6** | 无 | 曾 step2 hang / 梯度静默坏 | **已回滚**；坏在异步/跨流，非 blanket hook 本身 |
| gather-to-owner only（无 per-sample 释放） | — | — | — | — | — | — | 仅缩 out，**bwd 峰值不降**（见 3.1 no-ckpt） |
| all-gather outs（历史） | — | — | — | — | — | — | 每 rank 留 sp 份满 out → bwd OOM 更差 |

> 关键更正：**blanket `saved_tensors_hooks`→CPU 本身是对的**（`veomni/distributed/offloading.py::custom_save_on_cpu` 就是可用的 blanket 例子）。之前 grad 爆炸源于我那版**异步 prefetch 侧流 + 复用 caching-allocator 缓冲**的跨流数据竞争；改成纯同步 `.cpu()` / `.to(device)` 后，梯度完全正确，且峰值比 per-sample ckpt 更低（省掉了 recompute 的瞬时 forward 尖峰）。

### 3.3 时间 & offload 拷贝 profiler（Janus 2-step）

粗测墙钟：

| 配置 | 约 wall（2 steps） | 相对 |
|--|--|--|
| llamasp4 + per-sample ckpt | ~30–33 s | 基线（含 recompute） |
| llamasp4 + `offload_sync` | ~49.6 s（24.8 s/it） | 慢，几乎一半墙钟卡在同步拷贝上（见下） |

`offload_sync` 拷贝探针（`VEOMNI_SP_OFFLOAD_PROFILE=1`，per rank，整段 2-step 累计）。两个 scale 点（Janus，mbs=4 vs mbs=16）：

| scale | 方向 | 总耗时 | 数据量 | 张量数 | 单张量 | 有效带宽 |
|--|--|--|--|--|--|--|
| mbs=4 | D2H（fwd pack `.cpu()`） | 13.96 s | 17.7 GB | 4824 | 3.67 MB | **1.24 GB/s** |
| mbs=4 | H2D（bwd unpack `.to(cuda)`） | 10.29 s | 17.7 GB | 4824 | 3.67 MB | **1.68 GB/s** |
| mbs=16 | D2H | 34.2 s | 68.3 GB | 5608 | 12.2 MB | **1.95 GB/s** |
| mbs=16 | H2D | 15.0 s | 68.3 GB | 5608 | 12.2 MB | **4.44 GB/s** |

- **延迟受限已证实**：张量数几乎不变（4824→5608），单张量 3.67→12.2 MB，带宽随之升（D2H ↑1.6×、H2D ↑2.6×）。小拷贝被单次 `.cpu()`/`.to()` 的固定开销 + pageable 内存吃掉。
- **但相对占比在 Janus 上仍高**（mbs=4 ≈49%、mbs=16 ≈60%+ 墙钟）——**Janus-1.3B 是 copy fraction 的最坏情形**：copy_time ≈ 激活字节/带宽，compute_time ≈ FLOPs ≈ 参数×token；小模型 compute/byte 低 → copy 占比高。**大模型（如 30B）compute/byte 高 → copy 相对占比大幅下降，且 async overlap 能藏进更多 compute**。换言之：**copy 痛点最重的恰是「本来就装得下、不需要 offload」的小模型；真正需要 offload 的大模型痛点反而轻。**
- **D2H/H2D 不对称**：pack `.cpu()` 即使张量变大仍只 ~2 GB/s（pageable + `.cpu()` 强制把 forward 计算流 drain，串行化）；unpack `.to(cuda)` 能到 4.44 GB/s。→ **pinned host + 合并小张量**尤其能救 D2H，且跨 scale、零正确性风险。
- Qwen3-VL-2B 更大模型 smoke **被无关的 checkpoint `tie input/output embeddings`（`KeyError: 'weight'`）问题挡住**，未取得大模型数据点。

> caveat：均为 smoke run（`max_steps=2`、Janus 小序列）。探针入口：`dispatch.py` `_OFFLOAD_PROFILE` / `_prof_dump`。

### 3.4 激活占比 / 显存占比（概念）

在 **layer AC 已开、无 per-sample ckpt** 时（仅讨论存留，不含 params）：

| 项 | nosp（长 L） | V2 SP=4 loop、无释放 | V2 SP=4 + per-sample ckpt |
|--|--|--|--|
| 单次 fwd 工作集 | ~L | ~L/4 | ~L/4 |
| fwd 结束后 GPU 上 layer-boundary / saved | ~N_layers·L | ~4·N_layers·(L/4) = **N_layers·L**（同阶满样本） | ~4× module 输入（量级小）或 recompute 时一次一份 |
| bwd 峰值（激活主导时） | ~满样本 | ~满样本 | ~**一份 L/4**（+少量 fold/out） |

Params / FSDP：

- 默认 `reshard_after_forward=True`：每次 sample fwd 都 shard↔unshard（SP/OOM 常态）。
- `sp_keep_params_unsharded=True`：省 allgather，但大模型通常 **unshard 常驻会 OOM**，默认 False，冷门。

---

## 4. 方案取舍（对应你的理解）

| 诉求 | V1 | V2 现状 |
|--|--|--|
| fwd 峰值 ~1/4 | ✅ 天然 | ✅ 每轮 loop 工作集已是 1/sp |
| bwd 峰值 ~1/4，且**不开** GC/offload | ✅ | ❌ 做不到：一次 `loss.backward()` 前会攒满约「一条」的存留 |
| bwd 峰值 ~1/4，允许 ckpt/recompute | （通常不需要） | ✅ per-sample `checkpoint`（现默认路径，21.40 GB） |
| bwd 峰值 ~1/4，**不开 GC/recompute**，靠 CPU offload | — | ✅ **同步 blanket offload 实测可用**（16.87 GB，grad 正确）；异步 prefetch 版才会坏 |
| SP 优先于 GC（能 SP 就尽量不开 GC 保速度） | ✅ | ✅ 同步 offload 路径**不 recompute**，更贴近「有 SP 就不开 GC」的产品直觉（代价换成同步 D2H/H2D 在关键路径上） |

「不开 GC 也 1/4」现在有了可用手段：

1. **同步 blanket CPU offload**（本次验证，推荐方向）：loop 内 `saved_tensors_hooks` 同步 `.cpu()`，bwd 时 autograd 按子图顺序同步取回 → GPU 上一次≈一条样本的激活，无 recompute；或  
2. 回到 V1 式「常驻只持有 1/sp」的切分；或  
3. 在能拿到 loss 的层级做 **真·4 次 fwd+bwd 梯度累计**（抬循环，改 trainer/graph——此前明确不做）。

---

## 5. 相关代码 / 配置

| 路径 | 作用 |
|--|--|
| `veomni/models/seed_omni/graphs/dispatch.py` | `run_sp_looped_endpoint` + per-sample `checkpoint`（`ckpt`）/ 同步 offload（`offload_sync`，`_SyncCpuOffload`） |
| `train.accelerator.fsdp_config.sp_activation_offload` | 选 `ckpt`（默认）或 `offload_sync`；可整体或 per-module 设置（env `VEOMNI_SP_OFFLOAD_MODE` 临时覆盖） |
| `veomni/distributed/fsdp2_ac_patch.py` | FSDP2 multi-fwd + AC recompute unshard（pytorch#171779  backport；仅 `ckpt` 路径需要） |
| `veomni/distributed/sequence_parallel/data.py` | `sp_broadcast_from_rank` / `sp_gather_to_owner` |
| `scripts/seed_omni/_bench_sp_bwd_mem.py` | FatAct 公平显存 bench |
| `configs/.../modules_train_sp.yaml` / `modules_train_allsp4.yaml` | Janus SP 布局 |
| `train.accelerator.fsdp_config.sp_keep_params_unsharded` | 可选 keep-unsharded（默认 False） |
| `train.accelerator.offload_config.enable_activation` | 训练器级 activation→CPU（勿与已回滚的 loop 内 hooks 混淆） |

---

## 6. 结论（给后续决策）

1. **Gather-to-owner 必要但不充分**：只解决 out 放大，不解决 bwd 激活存留。  
2. **两条已验证、梯度正确、能明显压 bwd 的 V2 手段**：  
   - per-sample activation checkpoint（现默认，e2e 21.40 GB，付 recompute 税）；  
   - **同步 blanket CPU offload（e2e 16.87 GB，无 recompute，grad 与 ckpt 逐位对齐）**——峰值更低。  
3. **更正**：Loop 内 CPU offload 方向本身可行；之前判「不可用/坏梯度」的是**异步 prefetch + 跨 stream 缓冲**那版，坏在跨流数据竞争，不是 blanket hook 机制。  
4. **相对 V1**：同步 offload 已能做到「不开 GC 也压 bwd」，代价从 recompute 换成关键路径上的同步 D2H/H2D 拷贝（未做严格 step-time profiler；2-step 墙钟 offload≈96s vs ckpt≈85s，含启动噪声）。
