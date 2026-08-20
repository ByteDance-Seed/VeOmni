# SeedOmni V2 架构设计

> **2026-08 更新**：本文档是历史设计记录，其中 `module.py` / `ModuleMixin` / `modulemixin.py`
> 的写法已重构为「`modeling.py`（HF-native，权重 + `forward`，外加模块自带的
> `InferenceMixin`——类比 HF `GenerationMixin`，持有 `generate()` 及其 FSM 状态/采样
> helper，`class X(InferenceMixin, OmniPreTrainedModel)`，`InferenceMixin` 必须排在
> `OmniPreTrainedModel` 之前以免被其 no-op 默认实现遮蔽）+ `accelerated.py`（VeOmni
> 训练图钩子，仅 `TrainingMixin` / `VeOmniMixin`，不再需要 `InferenceMixin`）」两层——
> 见 [`seed_omni_v2.md` §2.1](seed_omni_v2.md) 获取当前权威说明；本文里凡提到
> `modulemixin.py` 或 `XxxModuleMixin` 的地方，一律理解为 `accelerated.py` 里的
> `VeOmniMixin`，且 `generate` 已经不在其列——它现在是 native `modeling.py` 里
> `InferenceMixin` 的方法。以下正文保留原文，只作历史参考。

> SeedOmni V2 (`veomni/models/seed_omni/`) 重写——把固定的 `Encoder → Foundation → Decoder` 三元结构换成**显式图声明**的模块化系统。`ModuleMixin` 是共享 mixin 基类；每个子模型再写 `XxxModuleMixin(ModuleMixin)`（`modulemixin.py`）并与 HuggingFace `PreTrainedModel` 多继承（`modeling.py`）。`training_graph` 是一条条 edge（`{from, to}`，端点为 `module[.method]`），node 由 endpoints 自动并出；同一 module 可挂多个 method。每个 node 必有出边——指向另一个 node 或保留关键字 `end`（虚拟终点），保证图无孤岛、无环。训练执行序由 topo sort 推导（可视化时画出 forward queue + `data` 伪节点）；推理由 FSM 驱动，每个 state 的 `body` 也是一条条内联 edge，可无限循环（text→image→text→image→...）。**数据完全 model-agnostic**：raw_batch 起点只有 `conversation_list`（list of dict，含 type / value / role / loss_mask），chat template / tokenize / image processor / boundary marker 注入全部由对应 module 在 forward 阶段自管——同一份数据可同时喂给任意 ug 模型；每个 module 的 `forward(**kwargs) -> Dict` 返回 dict 被框架写回共享 `raw_batch`（data 100% 走 raw_batch、module 之间不互相返回值）；collator helper / SP slice 由各 module 自己在 pre_forward 中按需调用（ViT 切 image batch、text encoder 切 sequence，各管各的）。loss 按 `_loss` 后缀隐式收集——每个 module 一次 forward 内部把所有 micro-batch 跑完，`post_forward` 自己做 token-level mean，OmniModel 顶层只把各 module 的标量 `_loss` 加起来。并行采用全局单一 `ParallelState`，OmniModel 顶层单次 `build_parallelize_model` 包装，`ParallelPlan` 由子模块递归聚合。生命周期上 weights 走 `build_foundation_model` + `build_parallelize_model`（多模块 path dict）、save 由各 module-trainer 的 `OmniModuleHfCallback` / `OmniModuleLoraCallback` 写到各 subfolder（config + 可选 processor/tokenizer 资产）。**配置拆分**：`base.yaml`（`model.model.model_path` + `model.model.model_config.modules` + `model.model.model_config.train_graph` + `model.accelerator` + `infer` 块）→ `OmniArguments.resolve_model()`（`arguments/omni_arguments_types.py`）合并 train/infer module 覆盖并解析相对 `model.model.model_path`，产出 runtime config（`OmniModelRuntimeArguments`）；`.to_hf_config()` 才投影成 HF `OmniConfig`。**FSM 转移**：只有 `module_signal` 与 `default` 两种 condition；text 侧由 `JanusTextEncoder` 通过 `module._tokenizer` 解析后发出 `start_image_gen` / `text_done` 等信号。**不保留 V1 兼容**。

## 总纲（不变量）

1. **`module` ≠ `node` ≠ `edge`**：实例 / 调用 / 数据流，三层各司其职。
2. **一个 module instance 可挂任意多个 node**；同 method 也可承担多个角色（按 kwargs 自分派）。
3. **训练 = DAG（一次拓扑遍历），推理 = FSM（含环、按状态转移循环）**。
4. **永远不自动推导"图结构本身"**：edge 列表必须 config 显式给出。但**执行顺序可由 topo sort 从 edges 推导**——可视化训练图时画出 forward queue；FSM 因含环不可推导执行序，只可视化状态转移图。

## 背景与问题

当前 [`modeling_seed_omni.py`](veomni/models/seed_omni/modeling_seed_omni.py) 采用固定的三元结构 `Encoder → Foundation → Decoder`，存在以下根本性局限：

- **结构写死**：`encoder`、`foundation`、`decoder` 是硬编码字段，无法表达 Qwen-Omni 的 thinker+talker（两个 LLM 串联）、BAGEL 的 AR+DiT 联合等架构
- **同模态只能有一个 encoder**：`self.image_encoder` 是单一字段，无法让理解图走 ViT、生成图走 VAE
- **SP 在外层**：`gather_seq_scatter_heads` 写在 `SeedOmniEncoderModel.forward()` 里，不随模块封装
- **ParallelPlan 不可组合**：`get_parallel_plan()` 只委托 foundation，encoder/decoder 即使本身是 MoE/带 embed 并行也无法把 plan 透出来 → 多模态 MoE（例如 ar_llm 是 MoE + vision_vae 也想加自己的 EP plan）只能改顶层模型逻辑

---

## 设计目标

| 目标 | 说明 |
|------|------|
| 完全模块化 | 所有组件以 `*ModuleMixin` + HF 模型多继承形态平等存在；只改 YAML（path / nodes / edges）即可替换任意模块 |
| 支持 AR + DiT | 同一训练框架内同时支持自回归和扩散两种生成范式 |
| 并行可组合 | 每个子模块可在完整 world 上跑**自己的**并行拓扑（异构 FSDP2 / FSDP2+ExtraParallel(`emb`/`ep`) / DDP）：拓扑与全局一致则复用全局 `ParallelState`，不同则自建独立 mesh；ParallelPlan 由各子模块 `get_parallel_plan()` 贡献 ExtraParallel 切分 |
| 训推一致（RL） | training `forward()` 和 inference `generate_step()` 共用同一底层实现 |
| 多模态对话驱动 | 同模态数据根据 conversation role 路由到不同模块（understanding vs. generation） |
| 推理循环生成 | 推理时可以反复循环（text→image→text→image），不是 DAG |
| 拆模型 / 多 path 加载 | 拆模型脚本输出 family 子模型目录，trainer 多 path 加载，per-module callback 各自存 subfolder |

---

Related Models
Lance https://arxiv.org/pdf/2605.18678
Cola-dlm https://hongcanguo.github.io/Cola-DLM/
Interaction Models https://thinkingmachines.ai/blog/interaction-models/
Cheers https://github.com/AI9Stars/Cheers
SenseNova-U1 https://github.com/OpenSenseNova/SenseNova-U1
Tuna-2 https://github.com/facebookresearch/tuna-2

---

## 核心设计

### 为什么训练是 DAG、推理不是

**训练**（teacher forcing）：AR LLM 一次 forward 处理完整序列，所有图像 output 位置已知且固定，其他模块在固定位置提取 hidden states 计算 loss。整个计算图**一次拓扑遍历**即可完成，是 DAG。

**推理**：token-by-token 驱动，生成一段文字后触发图像模块，图像模块完成后控制权归还文字模块，可无限循环（`text → image → text → image → ...`）。这**不是 DAG，是有限状态机（FSM）**。

两套执行语义分开实现：`OmniModel.forward()` 跑 DAG 遍历，`OmniModel.generate()` 跑状态机。

### 核心思路：扁平 edge 列表 + `end`（虚拟终点）

去掉 encoder / foundation / decoder 的固定角色，用**扁平 edge 列表**描述整张图——每条 edge 只有两个端点 `{from, to}`，端点是 `module[.method]` 字符串：

- **裸 module 名**（如 `janus_siglip`）→ 训练默认 `.forward`，推理默认 `.generate`
- **带点 method**（如 `janus_vqvae.encode`、`janus_text_encoder.emit_image_start`）→ 训推都调该 method
- **`end`** — 保留关键字，所有 sink 必须有一条 `to: end` 的边。**任何 node 至少有一条出边**，无孤岛；自环 / 任何环严格禁止（自环 = for-loop，应在模块内部实现）

**node 由 edge endpoints 自动并出**：node 的身份是其规范化的 `"<module>.<method>"` 字符串，无需独立的 `nodes:` 池。同一 module 可以挂多个 method 端点（如 `janus_vqvae.encode` 与 `janus_vqvae.decode`），共享一份参数但是图上两个独立节点。

edge **只声明拓扑顺序**，不携带 `output:` / `as:` 路由字段——数据通过共享 `conversation_list` / `raw_batch` / `ctx` 流动，各 module 从 carrier 按自己的 input keys 取。

`graph_train.yaml` **就是** training DAG 的 edge 列表（顶层无 wrapper key）；`graph_infer_*.yaml` **就是** 一张 FSM（顶层 `initial:` + `states:`），每个 state 的 `body` 也是内联 edge 列表。

```
modules pool                    graph_train.yaml（扁平 edge 列表）
─────────────────────           ────────────────────────────────────────────────
janus_siglip      ──→            - {from: janus_siglip,              to: janus_llama}
janus_vqvae       ──→            - {from: janus_vqvae.encode,        to: janus_llama}
janus_text_encoder──→            - {from: janus_text_encoder.encode, to: janus_llama}
janus_llama       ──→            - {from: janus_llama,               to: janus_text_encoder.decode}
                                 - {from: janus_llama,               to: janus_vqvae.decode}
                                 - {from: janus_text_encoder.decode, to: end}
                                 - {from: janus_vqvae.decode,        to: end}
                                 ↑ to: end 是 sink 锚（拓扑标记）；loss 仍按 _loss 后缀收集
```

---

## 核心抽象

### 1. Native / accelerated 两层：`modeling.py`（HF-native）+ `accelerated.py`（VeOmni 训练钩子）

> **2026-08 更新**：原先 `module.py` / `ModuleMixin` + `modulemixin.py` 的写法已重构为
> native/accelerated 两层，见下。历史决策记录（"D2.x" 等）中提到 `modulemixin.py` 的地方
> 保留不改，只反映当时的状态；本节描述**当前**的实现约定。

每个子模型现在有**两个类**：`modeling.py` 里的纯 HuggingFace-native 类
（权重 + `forward`；可以脱离 VeOmni 用普通 `from_pretrained` 加载运行）和
`accelerated.py` 里的 VeOmni 训练包装类（组合训练图 mixin）。类比 HF 自身
`forward` / `GenerationMixin.generate` 的分工：`modeling.py` 里除了模型类本身，
还定义一个同文件的 `InferenceMixin`（omni 版 `GenerationMixin`），持有
`generate()` 及其 FSM 状态 / 采样 helper：

```python
# modeling.py —— 纯 HF-native。InferenceMixin 持有 generate() + FSM 状态；
# 模型类持有权重 + forward，同时继承二者。
class InferenceMixin:
    """FSM generate() —— 类比 HF 的 GenerationMixin。"""

    def reset_local_inference_state(self) -> None: ...
    def reset_global_inference_state(self) -> None: ...

    def generate(self, conversation_list=None, generation_kwargs=None, **kwargs):
        ...  # FSM 单步推理（CFG cache 等 Janus 特有状态也在这里）

class JanusLlama(InferenceMixin, OmniPreTrainedModel):
    def forward(self, ...): ...

# accelerated.py —— 仅 VeOmni 训练图钩子，不再需要 InferenceMixin。
class TrainingMixin(TrainingModuleMixin): ...
class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin): ...
class JanusLlamaAccelerated(VeOmniMixin, JanusLlama): ...
```

判断标准：**脱离 VeOmni、单纯拿这个 checkpoint 做 HF 推理（chat / `generate` /
`AutoModel.from_pretrained`）时用户会期望能跑通的东西，都放 `modeling.py`**；
只有离开 VeOmni 图 runtime 就没意义的东西（FSDP dummy 输入、SP slice、
per-module 计量、训练 pre/post 钩子）才放 `accelerated.py`。两个类分别注册到
`OMNI_MODEL_REGISTRY`（native，供 `OmniModel.from_pretrained` / eager 推理）和
`OMNI_ACCELERATED_MODEL_REGISTRY`（accelerated，供 `ModuleRuntime` 训练 / 分布式推理）。

**`InferenceMixin`（`modeling.py`）为什么要排在 `OmniPreTrainedModel` 之前**：
`OmniPreTrainedModel` 自带 no-op 的 `reset_local_inference_state` /
`reset_global_inference_state` / `finalize`（作为安全网，供不需要真实推理状态的
模块兜底——例如 FSM runtime 会无条件调用 `module.finalize(ctx=...)`）。Python
MRO 从左到右解析，所以模型类必须写成
`class JanusLlama(InferenceMixin, OmniPreTrainedModel)`——若顺序反过来，
`OmniPreTrainedModel` 的 no-op 会**先于** `InferenceMixin` 里真正的实现被解析到，
静默 shadow 掉它们。`accelerated.py` 不再需要自己的 `InferenceMixin`：
`JanusLlamaAccelerated` 继承 `JanusLlama`，`generate()` / `reset_*` / `finalize`
通过对 native 类的常规继承、不经 shadow 直达 accelerated 包装类。只有当某个模块
确实需要"纯 accelerated、没有 native 对应实现"的推理行为时，才直接在
`Accelerated` 类上 override 对应方法——不要在 `accelerated.py` 里重新引入空的
`InferenceMixin` 标记类。少数骨干（`qwen3/llm`、`qwen3_moe/llm`）复用跨家族的
`SimpleArGenerationMixin`（`modules/base/llm_packing.py`）代替各自的
`InferenceMixin`，同样的 MRO 规则——排在 `OmniPreTrainedModel` 之前。

**初始化链**（不要在 mixin 里 override `post_init`）：

```python
# modeling.py —— native 类的 __init__ 只知道 PreTrainedModel
# （InferenceMixin 不定义 __init__，不影响这条链）。
class JanusSiglip(InferenceMixin, OmniPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)      # → PreTrainedModel
        ... 构建子模块 ...
        self.post_init()              # HF 权重初始化 / tied keys / parallel plan

# accelerated.py —— JanusSiglipAccelerated(VeOmniMixin, JanusSiglip) 时，
# cooperative __init__ 先过 VeOmniMixin 的 mixin 链，再落到上面这个 __init__。
```

**`accelerated.py` 训练图钩子**（除训练图节点外均可选；`generate` 类钩子已不在此列，见下）：

| 钩子 | 用途 |
|------|------|
| `pre_forward(method, **kwargs)` | 从 `conversation_list` 抽输入；FSDP dummy；SP slice |
| `forward(**kwargs)` | 训练计算；可返回标量 `_loss` |
| `post_forward(method, **outputs)` | 写回 `conversation_list`；`loss` → `_loss` |
| `dummy_inputs` | 缺模态时的零张量（训练 FSDP 对齐） |
| `get_parallel_plan` / `get_assets` | 并行与 checkpoint 资产 |

**`modeling.py` native 推理钩子**（推理期专用，不依赖 VeOmni 图 runtime）：

| 钩子 | 用途 |
|------|------|
| `generate(conversation_list, generation_kwargs, **kwargs)` | FSM 单步推理；每个模块各自实现 |
| `reset_local_inference_state()` / `reset_global_inference_state()` | 清理单次请求 / 单次生成的 FSM 状态 |
| `finalize(*, ctx)` | `max_new_tokens` 达到时 flush 缓冲输出 |

模块**实际继承形态**：

```python
# modeling.py
class InferenceMixin:
    """generate() + FSM 状态；见本文件（下同）。"""

class JanusLlama(InferenceMixin, OmniPreTrainedModel):
    """纯 backbone；wte/lm_head 在 janus_text_encoder。"""

class JanusVqvae(InferenceMixin, OmniPreTrainedModel):
    """VQ codec：encode / decode 两个 graph method；forward 在 modeling.py。"""

class JanusTextEncoder(TextEncoder):
    """继承 base TextEncoder（已含 InferenceMixin）；Janus chat template +
    FSM module_signal + 覆写 generate()。"""

# accelerated.py
class JanusLlamaAccelerated(VeOmniMixin, JanusLlama): ...
class JanusVqvaeAccelerated(VeOmniMixin, JanusVqvae): ...
class JanusTextEncoderAccelerated(VeOmniMixin, JanusTextEncoder): ...
```

- `model_type` 写在 `configuration.py`（HF `PretrainedConfig.model_type`），**不写在 YAML**——train YAML 的 `modules.*.model.model_path` 指向子目录，`read_model_type` 读 `config.json` 后在 `OMNI_MODEL_REGISTRY` 解析类。
- Tokenizer / processor 是**模块私有资产**（如 `janus_text_encoder/tokenizer/`），由 `OmniModuleTrainer` 在 build 时挂到 `module._tokenizer` / `module._processor`。

### 2. `OmniConfig`：modules + nodes + edges + 训练子集 + 推理状态机

**拆分 YAML 分工**（omni 配置；canonical 示例：`configs/seed_omni/Janus/janus_1.3b/`）：

| 文件 | 职责 |
|------|------|
| **Launcher**（`base.yaml`） | VeOmni 训练 / 推理共用入口：`model.model_path`（split checkpoint 根）、`model.modules` / `model.train_graph`、顶层 `accelerator`（v2 把 accelerator 从 `train` 提到与 `model`/`data`/`train` 平级）、`train.*` / `data.*`，以及 `infer` 块（`infer.modules` / `infer.infer_graph`（scenario → infer YAML）/ `infer.infer_type` / `infer.generation_kwargs` / 可选 `infer.model_path`） |
| **Train modules**（`modules_train.yaml`） | 每个 module 的训练覆盖（`model` / `train` / `accelerator`）。``modules.*.model.model_path`` 写**相对**子目录名（如 `janus_siglip`）或绝对路径 |
| **Train graph**（`graph_train.yaml`） | 文件本身**就是**扁平 edge 列表（顶层无 wrapper key），端点为 `module[.method]` |
| **Infer modules**（`modules_infer.yaml`，可选） | 每个 module 的推理覆盖，与 train modules **按模块名 deep-merge**；默认每个 module 走 eager 加载 |
| **Infer graph**（`graph_infer_*.yaml`） | 一个文件一个场景，文件本身**就是**那张 FSM（顶层 `initial:` / `states:`，无 wrapper key），由 `infer.infer_graph` 映射 |

运行时加载（训练 / 推理均通过 `veomni.arguments.omni_arguments_types`，由 `OmniArguments.resolve_model()` 调用；推理时传 `for_inference=True`）：

```python
from veomni.arguments import OmniArguments

args = OmniArguments(...)  # 或 parse_omni_args(...)
runtime_cfg = args.resolve_model(for_inference=True)

cfg = runtime_cfg.to_hf_config()     # 需要 HF checkpoint 形态时才转
```

### runtime config vs `OmniConfig`

**`resolve_omni_model()` 返回的是 runtime config（`OmniModelRuntimeArguments`），不是 `OmniConfig`。**
保留 launcher 的全部信息：拆分 checkpoint 根目录 `model_path`、deep-merge 后的每模块块
（含绝对路径与 `accelerator` / `train` 设置）、以及全部图。**这里不丢任何东西。**

投影到 checkpoint 形态的 `OmniConfig`（只留 `subfolder` + 可选 `model.model_config`）是一步
**显式**转换 `.to_hf_config()`，只在真正需要 HF 产物的地方调用：

- `OmniModelRuntime.from_model_runtime()` / `OmniInferencer` —— `OmniModel` 是 `PreTrainedModel`，
  需要它来 save assets / `save_pretrained`；
- `scripts/seed_omni/export_omni_checkpoint.py` —— 要写出 HF checkpoint。

只读图的消费者（如 `scripts/visualize_omni_graph.py`）直接用 runtime config，不需要转换。

`OmniConfig` 本身（`configuration_omni.py`）是纯 HF `PretrainedConfig`，
不 import `veomni.arguments`。

**所有场景都会被载入** `cfg.generation_graphs`（`{infer_type: FSM}`），`cfg.infer_type` 只决定哪个生效，
`cfg.generation_graph` 属性返回激活的那张图（runtime config 与 `OmniConfig` 都有这组属性）。因此一份导出的
checkpoint 能同时服务 gen / und / edit——换场景只需改 `config.infer_type` 再重建模型（`OmniModel` 在
`__init__` 绑定单张 FSM，不支持已建模型上热切换）。

图可视化（一条命令出四张图）：

```bash
python scripts/visualize_omni_graph.py configs/seed_omni/Janus/janus_1.3b/base.yaml
# → graphs/janus_1.3b_base/{training,infer_gen,infer_und,infer_interleave}.mmd
```

顶层 section 各司其职：

| Section | 职责 |
|---|---|
| `modules` | 模型实例池：name → launcher args 深合并后的 per-module 配置块（含 `model.model_path`）。**不写 model_type** |
| `graph_train.yaml` | 文件本身就是扁平 edge 列表（`{from, to}`，端点为 `module[.method]` 字符串）；`TrainingGraph` 据 endpoints 自动并出 nodes、按 topo 排序（DAG 视图） |
| `graph_infer_*.yaml` | 文件本身就是推理 FSM（顶层 `initial:` + `states:`）；每个 state.body 是内联 `{from, to}` edge（裸 module 默认 `.generate`） |

同一个 module 可以挂多个 method 端点，但**模型实例不拆分**——`janus_vqvae.encode` 和 `janus_vqvae.decode` 是图上两个独立 node，共享一个 `JanusVQDecoder` 实例；同一个 method 也可以承担训练 + 推理两条 input pathway，靠 kwargs 自分派（`janus_vqvae.decode` 是这种统一 head 的典型例子）。

> **`text_encoder`：model-specific 的 chat-template + wte + lm_head 模块。** 这一层是 V2 数据流的核心枢纽：
> - **Tokenizer 资产** 住在 ``modules/<family>/text_encoder/tokenizer/``；build 时挂到 ``module._tokenizer``，special-token id **不落 config**；
> - 在 `forward.encode` 中把 raw `conversation_list` 拼接成 `input_ids` / `inputs_embeds` / `labels` / `attention_mask`（含 chat template / system prompt / EOS / boi-eoi marker token）；
> - 在 `forward.decode` 中把 hidden_states 投影回 vocab（lm_head；tied weights 时 encode/decode 共享同一份矩阵）；推理时采样后写 **FSM ``module_signal``**（如 Janus 的 `start_image_gen` / `text_done`），YAML 不硬编码 token id。
>
> Janus 子类额外提供 ``emit_image_start`` / ``emit_image_end`` 两个 call-site（推理 bridge state 用），边界 token id 由 ``module._tokenizer`` 解析。
>
> 跟 V1 的"通用 wte + lm_head"不同，V2 的 `text_encoder` 是 **model-specific**——每个 family 一份 `modules/<family>/text_encoder/`。``scripts/convert_model.py``（family 实现见 ``modules/janus/convert_model.py``）把 ``embed_tokens`` + ``lm_head`` 拆到 ``text_encoder/`` 子目录，**全局 tokenizer 写到 output 根**。

```yaml
# ── 模块注册表（不写 model_type，HF AutoConfig 自动读）──────────────
# model.model_path 相对 launcher 的 model.model_path
modules:
  janus_siglip:       {model: {model_path: janus_siglip}}
  janus_vqvae:        {model: {model_path: janus_vqvae}}
  janus_llama:        {model: {model_path: janus_llama}}
  janus_text_encoder: {model: {model_path: janus_text_encoder}}

# ── graph_train.yaml：文件本身就是 training DAG 的 edge 列表 ────────
# 每个端点是 module[.method] 字符串；裸 module → .forward
- { from: janus_siglip,              to: janus_llama }
- { from: janus_vqvae.encode,        to: janus_llama }
- { from: janus_text_encoder.encode, to: janus_llama }
- { from: janus_llama,               to: janus_text_encoder.decode }
- { from: janus_llama,               to: janus_vqvae.decode }
- { from: janus_text_encoder.decode, to: end }
- { from: janus_vqvae.decode,        to: end }

# ── graph_infer_gen.yaml：文件本身就是一张 FSM ────────────────────
# 顶层 initial + states；每个 state.body 是内联 edge 列表
initial: prompt_encode

states:
  prompt_encode:
    body:
      - { from: janus_siglip,       to: janus_llama }
      - { from: janus_text_encoder, to: janus_llama }
      - { from: janus_llama,        to: end }
    transitions:
      - { condition: { type: default }, next_state: image_vq_start }

  image_vq_start:
    body:
      - { from: janus_text_encoder.emit_image_start, to: end }
    transitions:
      - { condition: { type: default }, next_state: image_vq }

  image_vq:
    body:
      - { from: janus_llama, to: janus_vqvae }
      - { from: janus_vqvae, to: janus_llama }
    transitions:
      - { condition: { type: module_signal, key: image_complete }, next_state: image_vq_end }

  image_vq_end:
    body:
      - { from: janus_text_encoder.emit_image_end, to: end }
    transitions:
      - { condition: { type: default }, next_state: done }
```

**只改 config 即可完成模块替换**：
- 把 `janus_llama` 的 `model.model_path` 指向其他 backbone 子目录 → 换了 backbone（`model_type` 自动从新 path 的 config.json 读）
- 把 `janus_siglip` 改成另一份 vision encoder ckpt → 换了 vision encoder
- 新增 `talker` 模块 + 对应 edge 端点 → 支持 Qwen-Omni 风格的双 LLM

### 3. `OmniModel`：两套执行语义

```python
class OmniModel(PreTrainedModel, GenerationMixin):
    modules_dict: nn.ModuleDict          # 模块实例（一份 module 一个 key）
    graph:        TrainingGraph          # 训练 DAG（端点 = module[.method]）
    fsm:          GenerationGraph        # 推理 FSM（state.body = 内联 edge 列表）

    # ── 训练路径：node DAG 一次遍历 ──────────────────────────────────────
    def forward(self, **batch) -> OmniOutput:
        node_outputs = {}                  # 索引 = node 名
        losses = {}
        for endpoint in self.graph.execution_order:   # 由 edge topo 推出的 node 序
            module_name = self.graph.module_of(endpoint)
            method      = self.graph.method_of(endpoint)  # 默认 forward
            module      = self.modules_dict[module_name]
            inputs      = batch   # edge 仅声明拓扑顺序；共享 conversation_list 载体
            # 一次调用内部把本 step 的所有 micro-batch 跑完：模块 forward
            # 自己迭代 micro-batches → 累加 token-sum loss / 累加 token_count →
            # post_forward 里做一次 token-level mean，吐出标量 `*_loss`
            if method == "forward":
                outputs = module(**inputs)              # 走 FSDP 包装层
            else:
                outputs = getattr(_unwrap(module), method)(**inputs)  # 直调 raw module
            node_outputs[endpoint] = outputs
            # _loss 后缀隐式收集；此时每个 _loss 已经是 mean 后的标量
            losses |= {f"{endpoint}/{k}": v for k, v in outputs.items() if k.endswith("_loss")}
        # 顶层只把各 module 已 mean 的标量 loss 求和（无须再加权）
        total_loss = sum(losses.values()) if losses else None
        return OmniOutput(
            losses=losses,
            total_loss=total_loss,
            **{f"{ep}_out": o for ep, o in node_outputs.items()},
        )

    # ── 推理路径：状态机分发 ────────────────────────────────────────────
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self._fsm.step(input_ids, self.modules_dict, **kwargs)

    # ── ParallelPlan 递归聚合（供顶层 build_parallelize_model 使用）─────
    # 注意：sub-modules 直接挂为 OmniModel 顶层 attribute（不通过 ModuleDict
    # 中介，D2.2 已落地），所以 self.named_parameters() 看到的 fqn 是
    # <name>.<rest>，无中间 prefix。`modules_dict` 是 property dict view，
    # 用于向后兼容老 callsite。
    def get_parallel_plan(self) -> ParallelPlan | None:
        merged: dict[str, dict[str, Shard]] = {}
        for name in self._module_names:
            mod = getattr(self, name)
            plan = mod.get_parallel_plan() if hasattr(mod, "get_parallel_plan") else None
            if plan is None:
                continue
            plan.update_prefix(name)                      # 加 <name>. 前缀
            for para, sub_plan in plan.extra_parallel_plan.items():
                merged.setdefault(para, {}).update(sub_plan)
        return ParallelPlan(merged) if merged else None
```

**Loss 协议（单键 `_loss`）**：每个 module 一次 `forward` 内部**自己遍历所有 micro-batch**——所有 micro-batch 跑完 → 在 `post_forward` 内部按 token-sum / token-count 做 mean → 吐出标量 `<name>_loss`（已经是正确的 token-level mean）。OmniModel 顶层只是把各 module 的标量 loss 加起来，不需要 token count 元数据。

为什么在 module 内部 loop micro-batch 而不是外层：
- **正确性**：不同 micro-batch 的 image token 数不同时，必须先 sum loss / sum tokens 再 mean——这是 token-level mean。如果外层每个 micro-batch 调一次 module、各自吐 mean，再外层做 batch-mean，会得到 **batch-weighted** 而非 **token-weighted** 的错误结果。
- **简洁性**：单键 `_loss` 协议足够；无需 `_loss_sum + _loss_token_count` 双键；OmniModel 不感知 token 数。
- **执行**：依赖每个 module 自己实现 `pre_forward` / `forward` 中的 micro-batch 循环（即"一个 module 一次性跑完所有 micro-batch"），相当于把 trainer 现有 `mean_global_loss`（参考 [`base.py:530-532`](veomni/trainer/base.py)）的语义内化到模块。

---

## 训练数据流

训练时 teacher forcing，AR LLM 一次 forward 处理完整序列，整体是个 node DAG。每个 node 跑一遍 `forward`：返回 dict 被框架写回共享 raw_batch；下游 module 从同一 raw_batch 按自己的 input keys 取。同一 module 可以挂多个 method 端点（如 `janus_vqvae.encode` / `janus_vqvae.decode`），共享一份参数。

```mermaid
flowchart TD
    data["data[(data)]"]

    subgraph exec ["OmniModel.forward — training_graph 拓扑一次遍历"]
        siglip["janus_siglip<br/><i>.forward</i><br/>← conversation_list<br/>→ {image_embeds, conversation_list (+ boi/eoi items)}"]
        vae_enc["janus_vqvae.encode<br/><i>.encode</i><br/>← conversation_list<br/>→ {gen_embeds, vq_token_ids, conversation_list (+ boi/eoi items)}"]
        tok_enc["janus_text_encoder.encode<br/><i>.encode</i><br/>← conversation_list (含 boundary markers)<br/>→ {conversation_list (split, value=inputs_embeds), input_ids, labels, attention_mask}"]
        ar["janus_llama<br/><i>.forward</i><br/>← conversation_list (split)<br/>← image embeds (siglip)<br/>← gen embeds (vqvae.encode)<br/>← labels / attention_mask (text_encoder.encode)<br/>splice: 按 segment 顺序替换 placeholder → N patch tokens<br/>→ {hidden_states}"]
        tok_dec["janus_text_encoder.decode<br/><i>.decode</i><br/>← hidden_states<br/>← labels (raw_batch)<br/>→ {_loss}  scalar, post_forward 内 token-mean"]
        vae_dec["janus_vqvae.decode<br/><i>.decode</i><br/>← hidden_states<br/>← gt_token_ids (vqvae.encode)<br/>→ {_loss}  scalar, post_forward 内 token-mean"]
        endN((end))
    end

    data -.-> siglip & vae_enc & tok_enc
    siglip -.modify.-> data
    vae_enc -.modify.-> data
    tok_enc -.modify.-> data
    siglip --> ar
    vae_enc --> ar
    tok_enc --> ar
    ar --> tok_dec
    ar --> vae_dec
    tok_dec -.->|"to: end (_loss)"| endN
    vae_dec -.->|"to: end (_loss)"| endN
```

**Forward queue 由 topo sort 自动推导**——`scripts/visualize_omni_graph.py`（传入 launcher YAML）会基于 `training_graph` edge 列表跑 Kahn topo sort，并画 **`data[(data)]` 伪节点**指向所有 source node（表示 kwargs 来自共享 batch dict）。注意 **`janus_text_encoder.encode` 必须等 `janus_siglip` / `janus_vqvae.encode` 完成**——它们对 `conversation_list` 的修改（插入 boi/eoi marker item）是 `janus_text_encoder.encode` 拼接 chat template 时的必要前置：

```
forward queue:
  1. janus_siglip              (no deps; reads conversation_list, mutates it with boi/eoi)
  2. janus_vqvae.encode        (no deps; reads conversation_list, mutates it with boi/eoi)
  3. janus_text_encoder.encode (waits: siglip + vqvae.encode; reads mutated conversation_list)
  4. janus_llama               (waits: siglip + vqvae.encode + text_encoder.encode)
  5. janus_text_encoder.decode (waits: janus_llama)
  6. janus_vqvae.decode        (waits: janus_llama + vqvae.encode)
  → end            (sink)
```

> 注：`janus_siglip` / `janus_vqvae.encode` 之间互不依赖（都只读 conversation_list 的不同 item type），它们对 conversation_list 的修改在不同位置（`image` item 处 vs `vq_image` item 处），插入操作满足交换律。框架不强求两者的相对顺序，但相对 `janus_text_encoder.encode` 必须都在它之前。

无环要求保证 topo sort 可解；任何环（含自环）会在 `TrainingGraph` 构造时直接报错。

**关于 janus_vqvae 的双角色——靠两个 method 端点表达**：

- `janus_vqvae.encode`：从 `conversation_list` 读 vq_image item，产出 gen embeds 喂给 `janus_llama`，并把 ground-truth token ids 写回 carrier 供 decode 用。
- `janus_vqvae.decode`：**统一 VQ head**——同一个 node 同时承担训练 loss 和推理反馈：
  - 训练：吃 `janus_llama.hidden_states` 和 encode 阶段写回的 ground-truth token ids → 吐标量 `gen_loss`（走 `generation_head` + CE，`post_forward` 内已做 token-level mean）
  - 推理：吃 `janus_llama` 采样的 `token_id` → 吐 `embed`（走 `generation_embeddings` + `aligner`）
  - 两条路径互不干扰，按 kwargs 分派——HF 风格的 "input present → run, absent → skip / dummy"

两个 method 端点共享同一个 `JanusVQDecoder` 实例（同一份参数），但**图论上是两个独立 node**（`janus_vqvae.encode` / `janus_vqvae.decode`），分别在 `janus_llama` 之前和之后执行——没有环、没有"同模块跑两次"的特殊处理，就是标准 DAG。

**端点边 (`to: end`) 与 loss 收集**：

- `to: end` 边是**拓扑标记**：保证图无孤岛、可视化时所有 sink 都汇入 end 节点，**不携带数据语义**。
- loss 仍由 `*_loss` **后缀**隐式收集：OmniModel 扫描每个 module 的输出 dict，把 `_loss` 后缀键（已经是 module 内部 token-level mean 后的标量）收齐求和。
- 因此即便某个 sink 边漏写了，只要模块输出有 `_loss` 后缀键就还会被收集——但**强烈建议每个 sink 都补一条 `to: end` 边**，保证拓扑完整，避免可视化丢节点。

**Dummy forward**：node 一旦出现在 `training_graph` edge 列表里，**必跑一遍 forward**——data 全 0 / dummy 也必须走完整张图，避免 FSDP backward hang。模块自己在 `pre_forward` / `forward` 里写 dummy 路径（输入为 None / 全 0 时构造形状一致的 dummy tensor、loss 标量为 0），保证计算图静态一致。

**训推一致性**：训练用 teacher forcing（ground truth VQ embeds 直接送入 `janus_llama`），推理用 `image_vq` body loop（`janus_llama` 采样 vq_token_id → 同一个 `janus_vqvae` module 走 generate 路径产 embed → 下一步 input）。训练和推理共用同一份参数、同一个 module，仅 method / kwargs 不同。

### Q：同一个模块在数据流上出现两次怎么办？

典型场景：一个统一的 `image_codec`，输入图像过它得到 embeds 喂给 LLM，LLM 输出再过它得到生成图像。直觉上 `image_codec` 是一个节点被调用两次，但这会让图带环。

**做法：声明两个 node，共享一个 module 实例。**

```yaml
# graph_train.yaml — 文件本身就是 edge 列表
- { from: image_codec.encode, to: ar_llm }
- { from: ar_llm,              to: image_codec.decode }
- { from: image_codec.decode, to: end }
```

`OmniModel.modules_dict["image_codec"]` 只 init 一次、参数只一份；`janus_vqvae.encode` 和 `janus_vqvae.decode` 是图上两个独立 node（规范 endpoint `"image_codec.encode"` / `"image_codec.decode"`），但 `module_of()` 解析到的是**同一个 Python 对象**。反向传播时两次调用的梯度自动累加到同一份参数上，就是普通的 weight sharing，没有任何 magic。

为什么不允许"一个 node 跑多次"：那会让图带环，endpoint 失去唯一性，loss key 失去唯一性，拓扑排序退化成"输入到齐就跑"的数据流调度，且 torch.compile / FSDP2 都假设 sub-module 调用顺序在一次 forward 中静态可枚举。把它写成两个 method 端点，等价于**显式静态展开**那次循环——表达力一样，YAML 多几行，换来全程纯 DAG。

至于"自回归推理时 image_codec 在每个 token step 都被调用"这种**时序上的重复**——交给 FSM：训练图保持静态 DAG，FSM 在每个 step 内执行一段 body edge 序列，整段 body 由外层步数循环驱动，不会污染 DAG 的"每个 node 跑一次"语义。

---

## 推理：生成图（FSM 视图）

### 核心统一抽象

推理和训练的本质差异在于：训练时 edge 列表做**一次拓扑遍历**，推理时 state.body 做**N 步循环**。两者都是**扁平 edge 列表**——端点是 `module[.method]` 字符串，激活的 nodes 由 endpoints 自动并出。

**FSM 一步执行规则**：

- 按 `state.body` 列出的 edge 顺序遍历；遇到 `from` endpoint 首次时**执行该 node**（裸 module → `.generate`；显式 method → 直调），module 返回 dict 被框架**写回 ctx**（推理时 `ctx == raw_batch`）。
- 同 step 内同一 endpoint 不重复执行。
- **module 之间不直接传值**：下游 module 从共享 `ctx` / `conversation_list` 按自己的 input keys 取，跟训练时一样。edge 只声明拓扑顺序，不携带 `output:` / `as:` 路由字段。

典型形态：

- **单节点循环**：文本 AR（`janus_text_encoder → janus_llama`，variable 步）；DiT（`bagel_dit` 循环 forward，1 步）
- **多节点串接 + 反馈循环**：VQ 图像生成（`janus_llama → janus_vqvae → 反馈回 janus_llama`，循环 576 步）

```
训练时：graph_train.yaml（edge 列表） → 拓扑排序 → 一次 forward 遍历
推理时：state.body（edge 列表）       → 按序执行 (endpoint 首次激活) → 循环 N 步
```

### 状态机定义

```yaml
# graph_infer_interleave.yaml — 文件本身就是 FSM（text → image → text 循环）
initial: prompt_encode

states:
  prompt_encode:
    body:
      - { from: janus_siglip,       to: janus_llama }
      - { from: janus_text_encoder, to: janus_llama }
      - { from: janus_llama,        to: end }
    transitions:
      - { condition: { type: module_signal, key: start_image_gen }, next_state: image_vq_start }
      - { condition: { type: module_signal, key: text_done },        next_state: done }

  image_vq_start:
    body:
      - { from: janus_text_encoder.emit_image_start, to: end }
    transitions:
      - { condition: { type: default }, next_state: image_vq }

  image_vq:
    body:
      - { from: janus_llama, to: janus_vqvae }
      - { from: janus_vqvae, to: janus_llama }
    transitions:
      - { condition: { type: module_signal, key: image_complete }, next_state: image_vq_end }

  image_vq_end:
    body:
      - { from: janus_text_encoder.emit_image_end, to: end }
    transitions:
      - { condition: { type: default }, next_state: text_ar }

  text_ar:
    body:
      - { from: janus_text_encoder, to: janus_llama }
      - { from: janus_llama,        to: end }
    transitions:
      - { condition: { type: module_signal, key: start_image_gen }, next_state: image_vq_start }
      - { condition: { type: module_signal, key: text_done },        next_state: done }
```

### FSM 转移条件

| 类型 | 谁决定 | YAML | 典型场景 |
|------|--------|------|----------|
| `module_signal` | 模块在 return dict 写一次性 flag | `{type: module_signal, key: K}` | VQ 末 patch → `image_complete`；text 采样 boi/eos → `start_image_gen` / `text_done` |
| `default` | catch-all 兜底——无条件匹配，**必须放在最后** | `{type: default}` | 单趟 bridge / leaf state（prompt encode、emit `<boi>` / `<eoi>`），或 `module_signal` 之后的 else 分支 |

> 框架只有这两种 condition；不再支持 YAML 硬编码 vocab id 的 `token_match`——special-token 语义全部留在 module 内部，由 module emit `module_signal`。

> **state 没有步数预算**：state body 跑一次后持续循环，直到某条转移触发。"跑多少步、何时结束 state" 完全由模块决定——AR 循环靠 `module_signal`（模块在 return dict 写 ``ctx["module_signal"] = "<K>"``，YAML 的 ``module_signal.key: K`` 做字符串相等匹配，转移后框架 pop），单趟 bridge state 靠 `default`。转移按顺序求值、首个匹配生效，所以 `default` 无条件匹配 ⇒ 它是最低优先级的兜底分支，必须排在最后（否则框架在构图时报错，因为其后的转移是死代码）。框架不再有 `token_length` 这个概念。

(如果要解析得到生成图像大小，这个东西可能做成一个 node，输出的 size 信息直接交给 image decoder，由该模块自行决定循环步数。)

### KV cache 由模块自管

KV 状态完全 module-specific：
- **Janus 风格**（每个 token 都过 `janus_llama` 生成 → 文本/图像/文本切换时 KV 可复用）→ `janus_llama.generate_step` 内部维护 KV，状态切换时不清。
- **DiT 后回到 LLM**（DiT 不消耗 LLM 的 KV，DiT 后切回文本要重新过 prompt）→ `dit.generate_step` 完成后，下次 `janus_llama.generate_step` 检测到上下文变化、清空 KV 重算。

何时复用、何时清空、是否保存 conversation history——都由各模块自己实现，OmniModel 不感知。

### 状态机实现

```python
class GenerationGraph:
    """
    每次推理 step：
      1. 按 state.body 顺序遍历 edges：edge.from endpoint 首次命中时调 module.method
         （裸 module → .generate），写 outputs 到 ctx
      2. 检查所有转移条件（first-match）
      3. 若触发转移，更新 _current_state
    （无步数预算——是否结束 state 完全取决于模块 raise 的 module_signal 或 default）
    """
    _current_state: str
```

### 状态机示意

```mermaid
stateDiagram-v2
    [*] --> text_ar : 开始推理

    text_ar : text_ar\nbody: janus_text_encoder→janus_llama
    text_ar --> text_ar : 普通文本 token
    text_ar --> image_vq_start : module_signal start_image_gen
    text_ar --> video_dit : module_signal start_video
    text_ar --> [*] : module_signal text_done

    image_vq_start : image_vq_start\nemit_image_start + 1 AR step
    image_vq_start --> image_vq : always

    image_vq : image_vq\nbody: janus_llama→janus_vqvae (反馈循环)
    image_vq --> image_vq_end : module_signal image_complete

    image_vq_end : image_vq_end\nemit_image_end + 1 AR step
    image_vq_end --> text_ar : always

    video_dit : video_dit\nbody: dit
    video_dit --> text_ar : module_signal video_complete
```

---

## 配置示例：不同模型架构

### Seed-Omni（AR + VQ 图像生成）

两个 module 各出现两次（通过 method 端点），共享一份参数：

* `janus_vqvae.encode` / `janus_vqvae.decode`（**统一 VQ head**——训练算 `gen_loss`、推理 hidden→sample→embed）。
* `janus_text_encoder.encode` / `janus_text_encoder.decode`。推理-only method `emit_image_start` / `emit_image_end` 由 `JanusTextEncoder` 提供。

`janus_llama` 自身不再持有 `wte` / `lm_head`——就是个纯 backbone（`inputs_embeds → hidden_states`）。

`scripts/convert_model.py` 把原始 Janus checkpoint 拆成 4 份 module 子目录：`janus_siglip/`、`janus_vqvae/`、`janus_text_encoder/`（含 tokenizer 资产）、`janus_llama/`。YAML 里 ``model.model_path`` 写相对名（如 `janus_siglip`），launcher 的 ``model.model_path`` 指向 split 根。

```yaml
# graph_train.yaml — 文件本身就是 edge 列表
- { from: janus_siglip,              to: janus_llama }
- { from: janus_vqvae.encode,        to: janus_llama }
- { from: janus_text_encoder.encode, to: janus_llama }
- { from: janus_llama,               to: janus_text_encoder.decode }
- { from: janus_llama,               to: janus_vqvae.decode }
- { from: janus_text_encoder.decode, to: end }
- { from: janus_vqvae.decode,        to: end }

# graph_infer_interleave.yaml — 文件本身就是 FSM
initial: prompt_encode
states:
  prompt_encode:
    body:
      - { from: janus_siglip,       to: janus_llama }
      - { from: janus_text_encoder, to: janus_llama }
      - { from: janus_llama,        to: end }
    transitions:
      - { condition: { type: module_signal, key: start_image_gen }, next_state: image_vq_start }
      - { condition: { type: module_signal, key: text_done },        next_state: done }
  image_vq_start:
    body:
      - { from: janus_text_encoder.emit_image_start, to: end }
    transitions:
      - { condition: { type: default }, next_state: image_vq }
  image_vq:
    body:
      - { from: janus_llama, to: janus_vqvae }
      - { from: janus_vqvae, to: janus_llama }
    transitions:
      - { condition: { type: module_signal, key: image_complete }, next_state: image_vq_end }
  image_vq_end:
    body:
      - { from: janus_text_encoder.emit_image_end, to: end }
    transitions:
      - { condition: { type: default }, next_state: text_ar }
  text_ar:
    body:
      - { from: janus_text_encoder, to: janus_llama }
      - { from: janus_llama,        to: end }
    transitions:
      - { condition: { type: module_signal, key: start_image_gen }, next_state: image_vq_start }
      - { condition: { type: module_signal, key: text_done },        next_state: done }
```

### Qwen-Omni（thinker + talker 双 LLM + 音频）

两个 LLM 各配一份 `text_encoder`（`tie_word_embeddings=true` 时 encode/decode 共用一矩阵）。

```yaml
# graph_train.yaml — 文件本身就是 edge 列表（thinker + talker 双 LLM 骨架）
- { from: qwen_vision,                 to: thinker_llm }
- { from: qwen_audio,                  to: thinker_llm }
- { from: thinker_text_encoder.encode, to: thinker_llm }
- { from: thinker_llm,                 to: thinker_text_encoder.decode }
- { from: thinker_llm,                 to: talker_llm }
- { from: talker_text_encoder.encode,  to: talker_llm }
- { from: talker_llm,                  to: talker_text_encoder.decode }
- { from: thinker_text_encoder.decode, to: end }
- { from: talker_text_encoder.decode,  to: end }

# graph_infer.yaml — 文件本身就是 FSM
initial: thinking
states:
  thinking:
    body:
      - { from: thinker_text_encoder, to: thinker_llm }
      - { from: thinker_llm,          to: end }
    transitions:
      - { condition: { type: module_signal, key: start_speaking }, next_state: speaking }
  speaking:
    body:
      - { from: thinker_llm,         to: talker_llm }
      - { from: talker_text_encoder, to: talker_llm }
      - { from: talker_llm,          to: end }
    transitions:
      - { condition: { type: module_signal, key: resume_thinking }, next_state: thinking }
      - { condition: { type: module_signal, key: text_done },       next_state: done }
```

`thinker_llm` 内部决定如何将 vision/audio embeds merge 进 embedding；`talker_llm` 内部决定如何用 thinker hidden states 作为 cross-attention key。**与 vllm-omni 中 thinker2talker `custom_process_input_func` 对应，但移入模块内部。**

### BAGEL（AR + DiT 图像生成）

```yaml
# graph_train.yaml — 文件本身就是 edge 列表（来自 configs/seed_omni/Bagel/bagel_7b_mot/）
- { from: bagel_text_encoder.encode,            to: bagel_qwen2_mot }
- { from: bagel_siglip_navit,                   to: bagel_qwen2_mot }
- { from: bagel_vae.encode,                     to: bagel_flow_connector.embed_latent }
- { from: bagel_flow_connector.embed_latent,    to: bagel_qwen2_mot }
- { from: bagel_qwen2_mot,                      to: bagel_text_encoder.decode }
- { from: bagel_qwen2_mot,                      to: bagel_flow_connector.decode_velocity }
- { from: bagel_text_encoder.decode,            to: end }
- { from: bagel_flow_connector.decode_velocity, to: end }

# graph_infer_gen.yaml — 文件本身就是 FSM
initial: prompt_encode
states:
  prompt_encode:
    body:
      - { from: bagel_text_encoder, to: bagel_qwen2_mot }
      - { from: bagel_siglip_navit,  to: bagel_qwen2_mot }
      - { from: bagel_qwen2_mot,    to: end }
    transitions:
      - { condition: { type: default }, next_state: query_denoise }
  query_denoise:
    body:
      - { from: bagel_flow_connector.prepare_denoise_query, to: bagel_text_encoder.encode_image_markers }
      - { from: bagel_text_encoder.encode_image_markers, to: bagel_qwen2_mot.denoise_branch }
      - { from: bagel_qwen2_mot.denoise_branch,      to: end }
    transitions:
      - { condition: { type: default }, next_state: velocity_collect }
  velocity_collect:
    body:
      - { from: bagel_flow_connector.decode_velocity_from_hidden, to: bagel_qwen2_mot.collect_velocity }
      - { from: bagel_qwen2_mot.collect_velocity,                 to: bagel_flow_connector.advance_denoise }
      - { from: bagel_flow_connector.advance_denoise,             to: end }
    transitions:
      - { condition: { type: module_signal, key: image_complete }, next_state: image_decode }
      - { condition: { type: default }, next_state: query_denoise }
  image_decode:
    body:
      - { from: bagel_vae.decode_generated, to: end }
    transitions:
      - { condition: { type: default }, next_state: done }
```

---

## 离线 Embedding：不是特殊模式，就是不同的 training_graph + 不同的 dataset

V2 框架中不存在 `offline_embedding` / `offline_training` / `online_training` 三种特殊模式的概念。它们只是**三份不同的 `training_graph` 配置**加上**不同的数据集格式**：

| 场景 | graph_train.yaml（edge 列表） | 数据集格式 | raw_batch 起点 | 产出 |
|------|---------------------------|--------|--------|------|
| **A. 生成 embedding** | 激活前置 module 到断点 module 的 edge 列表 + sink edge 到 end | 原始 jsonl + 多模态文件 | `{conversation_list}` | trainer 收集断点 module 输出，dump 成 pickle dataset |
| **B. 离线训练 DiT** | 只列 dit 子图的 edge 列表 + sink | 上一步 dump 出来的 **pickle dataset**（已含 pre-computed tensors） | `{condition: <Tensor>, dit_target: <Tensor>, ...}`（直接含张量字段，**无 conversation_list**） | dit_loss |
| **C. 在线全图训练** | 全部 edge 列表 | 原始 jsonl + 多模态文件 | `{conversation_list}` | 各 module 的 _loss 求和 |

```yaml
# ── 场景 A：生成 condition embedding（只跑到 bagel_qwen2_mot，保存 hidden_states）
# graph_train_offline_cache.yaml — 文件本身就是 edge 列表
- { from: bagel_text_encoder.encode, to: bagel_qwen2_mot }
- { from: bagel_siglip_navit,        to: bagel_qwen2_mot }
- { from: bagel_qwen2_mot,           to: end }
# 数据集：原始 jsonl + 多模态文件 → conversation_list
# trainer 从 OmniModel.forward() 输出里读 raw_batch['hidden_states'] 并 pickle 到磁盘

# ── 场景 B：用预存 pickle 训练 flow connector
# graph_train_with_cache.yaml — 只列 flow connector 相关 edge
- { from: bagel_vae.online_process,             to: bagel_flow_connector.embed_latent }
- { from: bagel_flow_connector.embed_latent,    to: bagel_qwen2_mot }
- { from: bagel_qwen2_mot,                      to: bagel_flow_connector.decode_velocity }
- { from: bagel_flow_connector.decode_velocity, to: end }
# 数据集：场景 A dump 出来的 pickle（已含 pre-computed condition 字段）
# raw_batch 起点不再是 conversation_list——离线场景的特殊性

# ── 场景 C：在线全图训练
# graph_train.yaml — 完整 edge 列表（见 BAGEL 示例）
```

**关键不变量**：
- raw_batch 是 mutable dict，**起点 schema 由 dataset 决定**——在线场景下 dataset 输出 conversation_list；离线场景下 dataset 输出已 pickle 好的张量字典。两种 schema 都通过相同的 OmniModel.forward 入口，只是 `graph_train.yaml` edge 列表决定走哪些 module。
- **跳过前置 module 的方法 = 不在 graph_train.yaml edge 列表里列它**——`TrainingGraph` 据 endpoints 自动并出 active node 集合，没列的 module 不会被实例化、也不会跑 forward；离线场景下连 vit/text_encoder 实例都不会被构造（节省显存）。
- **OfflineEmbeddingSaver 是 trainer 层的工具，模型不感知**——`OmniModel` 本身没有任何 mode 切换，所有 mode 差异都封在"哪份 training_graph 配置 + 哪份 dataset"。
- **Pickle 格式由 saver / loader 协议决定**：scripts/`save_offline_embeddings.py`（场景 A 之后）输出 pickle；scripts 里的 OfflineDataset 类负责加载 pickle 并按 batch 喂出。两边的 schema 约定（字段名 / 张量 dtype）跟训练 yaml 里 dit module 的 input keys 对齐。

---

## 并行配置（按模块 ParallelState + 递归 ParallelPlan）

每个子模块可在**完整 world** 上跑自己的并行拓扑——同一个 OmniModel 内可同时存在异构的
**FSDP2 / FSDP2 + ExtraParallel（`emb`/`ep`）/ DDP**。机制如下：

- trainer 先建**一份全局 `ParallelState`**（`OmniTrainer.base._setup()`，来自顶层 `accelerator`）。
- 每个子模块比较自己的 accelerator 拓扑与全局拓扑（`OmniTrainer._build_model` 用
  `_accelerator_topology(module_acc) != _accelerator_topology(global_acc)`）：**一致则复用**全局
  `ParallelState`（不重复建进程组）；**不同则**由 `OmniModuleTrainer._setup` 调
  `init_parallel_state(...)` **自建独立 mesh**。
- 每个子模块**各自 wrap**（FSDP2 原地分片 / DDP 包装），权重也按各自路径加载：每个 `ModuleRuntime`
  单独调一次 `build_parallelize_model(weights_path=args.model_path)`。
- ExtraParallel（`emb`/`ep`）切分仍由各子模块 `get_parallel_plan()` 贡献，但应用在该模块自己的 mesh 上。

per-module 拓扑通过 `modules_{train,infer}.yaml` 里每个模块的 `accelerator:` 块声明。

| 层级 | 职责 |
|------|------|
| 全局 `ParallelState` | trainer 层一次 `init_parallel_state(...)`（顶层 `accelerator`）；拓扑相同的子模块复用它 |
| 每个 `OmniModuleTrainer._setup` | 拓扑不同的子模块自建独立 `ParallelState`（先 `_dedup_extra_parallel` 折叠重复的 `ep`），并以**模块名**（`self.module_name`，同时也是 `<module>/` ckpt 子目录名）注册进全局 registry（`init_parallel_state(name=self.module_name)`）；module-trainer 不保留 `parallel_state` 句柄，读取方按名 `get_parallel_state_by_name(module_name)` |
| `ModuleMixin.forward()` | 内部自管 SP `gather/scatter`；运行时被包在 `use_parallel_state(模块名)` 中（registry 按名解析）|
| `OmniModel` | `set_module_parallel_state_names(names)` 记录已注册模块名 + `module_context(name)` → `use_parallel_state(name)`：每个 node 的 forward/generate 在该模块 state 下执行，使 `get_parallel_state()` 解析到该模块 mesh（registry 是 state 对象的唯一来源；eager 推理模块不注册、不 scope）|
| `ModuleMixin.get_parallel_plan()` | 返回**模块本地** fqn 的 `ParallelPlan`（如 `embed_tokens.weight` / `layers.*.mlp.experts.gate_up_proj`）|
| 梯度裁剪 | `_omni_clip_grad_norm`：按各模块拓扑分别 reduce pᵗʰ-power → 合成全局范数 → 共享系数裁剪（见下）|

### 拓扑判定与 `ep` 去重

`AcceleratorConfig.__post_init__` 总会追加一个 `ep` 维；而 `build_module_runtime_args` 会
`_instantiate_recursive` 重新实例化 accelerator，于是 per-module accelerator 末尾会出现**两个 `ep`**。
建 mesh / 比拓扑前必须 `_dedup_extra_parallel(acc)` 折叠这个重复（真实的 `emb`+`ep` 布局会保留）：

| 模块 override | 重新实例化后 names | dedup 后 | vs 全局 |
|---------------|--------------------|----------|---------|
| 无 | `["ep","ep"]` | `["ep"]` | 相同 → 复用 |
| `emb` size 4 | `["emb","ep"]` | `["emb","ep"]` | 不同 → 自建 |
| `fsdp_mode: ddp` | `["ep","ep"]` | `["ep"]`（mode 不同）| 不同 → 自建 |

### 异构梯度裁剪

对 `OmniModel.parameters()` 直接 `clip_grad_norm_` 会失败（不能混 DTensor 与普通 Tensor）。
`_omni_clip_grad_norm` 对每个模块在其拓扑对应进程组上算 world-complete 的 pᵗʰ-power 和
（FSDP2 → `fsdp_group`；FSDP2+ExtraParallel → `{p}_fsdp` 再 `{p}`；DDP → 不再 reduce，backward 已 all-reduce），
合成一个全局范数后用共享系数裁剪所有模块。

### DDP 细节

- **构建**：`parallelize_model_ddp`（`torch_parallelize.py`）在 meta-init 下先 materialize + 加载全量权重，
  **再** `DDP(...)`（DDP 只复制 + 注册梯度同步 hook，不会 materialize meta 参数、不加载权重）。
- **分发**：`DistributedDataParallel` 不代理属性访问，所以 `TrainingGraph.step`（训练单节点驱动）/
  `OmniModuleTrainer.on_step_end` 要 `_unwrap_module(...)` 取 `pre_forward`/`post_forward`/`metric_meter_collect`，
  但实际前向仍调 **wrapper** 以触发 hook（FSDP2 原地 `raw is wrapper`；DDP 包装 `raw = wrapper.module`）。
  composed 后的 `OmniModel` 把每个 module 的 wrapped model 作为子模块持有，`TrainingGraph.step` 通过 `OmniModel`
  传入的 wrapped modules 字典取用（与 `GenerationGraph.step` 完全对称）。

### FQN 视角对齐（重要细节）

`OmniModel` 把每个 sub-module 直接 attach 为顶层 attribute（**不**通过 `nn.ModuleDict` 中介），所以
`model.named_parameters()` 看到的 fqn 形如 `<module_name>.<rest>`；`model.named_children()` 直接枚举
`[(<module_name>, sub_module), ...]`，与 checkpoint / parallel plan 的 key 对齐。`modules_dict` 是
property dict view，**不**是 `nn.ModuleDict`。

### 举例

- `janus_text_encoder`：`accelerator` 声明 `emb` ExtraParallel（`extra_parallel_names: ["emb"]`），
  `get_parallel_plan()` 返回 `{"emb": {"embed_tokens.weight": Shard(0)}}`，自建 FSDP2+emb 的独立 state；
  查表用 `AllToAllEmbedding`（`veomni/ops/kernels/embed/`）。
- `janus_llama`：`accelerator.fsdp_config.fsdp_mode: ddp` → 自建 ddp `ParallelState`，复制式 backbone。
- `janus_siglip` / `janus_vqvae`：无 `accelerator` override → 复用全局 FSDP2 mesh；SP 在自己 `forward()` 里处理。

### micro_batch_size 仍全局；dp / fsdp / ExtraParallel 可 per-module

`micro_batch_size` 与数据管线是全局共享的（数据集 / collator / dataloader 只有一份），**不**接受 per-module
`micro_batch_size`。但 **dp / fsdp_mode / ExtraParallel 现在可以 per-module**——通过模块的 `accelerator:` 块声明，
拓扑不同即自建独立 mesh。`OmniConfig.modules.<name>.accelerator.*` 即承载这些覆盖。

`init_device` / `broadcast_model_weights_from_rank0` / `ep_sharded_stream_load` / `gradient_checkpointing` /
`torch_compile` / `chunk_mbs_config` 同样挂在 `accelerator.*` 上（而不是 `train.*`），因此也是 **per-module**：
不同模块可以各自声明 `accelerator.init_device` / `accelerator.gradient_checkpointing.enable` 等。这些字段的
交叉校验（`init_device` 与 `fsdp_mode`/`ep_size` 的关系、`chunk_mbs_config` 与 `pad_to_length`/
`gradient_checkpointing.enable_reentrant` 的关系、`torch_compile.enable` 的整体禁用）由
`veomni.arguments.omni_arguments_types._validate_omni_accelerator` 负责，分别对顶层默认 `model.accelerator`
（`OmniArguments.__post_init__` 时）和每个模块解析后的 `accelerator`（`resolve_omni_model` 里，模块合并之后）各
校验一次——只放在 Omni 侧，不写进 V1 也会用到的共享 `AcceleratorConfig.__post_init__`（V1 的 `torch_compile` 是
支持的，且校验逻辑不同）。

`init_device: meta` 只有 `fsdp_mode: fsdp2` 才是必须的（`torch_parallelize.py` 里 `parallelize_model_fsdp2`
的硬断言）；`ddp` 模块不需要 meta-init 的 materialize+broadcast 流程——`parallelize_model_ddp` 对非 meta
的 `init_device` 直接跳过该分支，`build_foundation_model` 已经把权重加载到位，DDP 构造时再广播一次即可,故
`ddp` 模块可以直接声明 `accelerator.init_device: cuda`（见 `janus_siglip` / `janus_vqvae` 的 `modules_train.yaml`
/ `modules_infer_fsdp.yaml`）。这也规避了 `build_parallelize_model` 里 `not parallel_state.fsdp_enabled`
（即 `world_size // (pp_size * tp_size) == 1`，例如单卡场景）时对 `init_device` 必须是 `cuda`/`npu` 的硬校验——
继承全局 `meta` 默认值的 `ddp` 模块在这种拓扑下会直接报错。

### 推理侧

推理同样支持 per-module 拓扑：`OmniInferencer` 用 `_module_needs_distributed`（fsdp_mode 非 `eager` 即为分布式，
含 `fsdp2` / `ddp`）判断是否需要 torchrun + 进程组，并为每个非 eager 模块建独立 `ParallelState`；eager 模块走
`from_pretrained(device_map=...)` 单卡副本。两套 `modules_infer_*.yaml`（分布式 / 全 eager）+ 对应启动脚本。
- ❌ per-module micro_batch_size / dp_size / sp_size / tp_size / cp_size（OmniTrainer 整体可工作后再支持）

### 与现有基础设施

`torch_parallelize.py` 的 `build_parallelize_model` **不需要**改：`OmniTrainer._build_model` 走 `build_omni_model()`，由每个 `ModuleRuntime` 各自调一次,传自己那一个 `args.model_path`（见下一节）。早期设计里"顶层一次 wrap 整个 OmniModel + 多 weights_path"的两步流程没有采用。

---

## 生命周期

### Build & 权重加载

trainer 现有的两个组件函数**原样复用**，只是每个 module 各调一次:

| 组件 | 用法 |
|------|------|
| [`build_foundation_model`](veomni/models/auto.py) | `ModuleRuntime._build_module_model()` 传该模块的 `args.model_path` 同时作为 `config_path` 与 `weights_path`，meta-init 出一个子模型 |
| [`build_parallelize_model`](veomni/distributed/torch_parallelize.py) | `ModuleRuntime._parallelize_module_model()` 传同一个 `args.model_path`，wrap + 加载该模块自己的 snapshot |

**Meta device + per-module 加载流程**（`ModuleRuntime.__init__`，整段在 `use_parallel_state(module_name)` 作用域内）：

1. 该 module 在 meta device 上按 HF AutoConfig + AutoModel 构造（自动从 `<model_path>/config.json` 读 `model_type`）。
2. 用 `OMNI_MODEL_REGISTRY[model_type]` 解析预定义的 `XxxModuleMixin + PreTrainedModel` 合体类。
3. freeze / LoRA 之后，ParallelPlan 应用 + `fully_shard()`（或 DDP）wrap **该模块**。
4. 从 `<model_path>` 加载该模块自己的 weights；子模块之间互不感知。装配成 `OmniModel` 后，参数 fqn 自然带上 `<module_name>.` 前缀。

**Key convert**：`scripts/convert_model/split_<family>.py` 拆分时只关心 family 内子模型，不知道用户在 YAML 里给这个子模型起什么 node 名。所以约定：
- 拆模型脚本输出固定的子目录命名（如 `janus_siglip/`、`janus_vqvae/`），子目录里 weights 用模块**本地** fqn 命名。
- 加载时按 YAML `modules.<name>.model.model_path` 读取，state_dict 套上 `<name>.` 前缀放到 `omni_model.<name>` 子树。
- 用户在 YAML 里改 module 的 key（如把 `janus_llama` 改成 `my_backbone`），不影响加载——前缀由 YAML key 决定。

**实施进度**：早期设计的终态是"扩展 `build_foundation_model` 接 `dict[str, str]`、`build_parallelize_model` 多 path meta-init、顶层一次 wrap"，按 PR 拆分推进时改成了上面的 per-module 方案。各阶段落点：

| 阶段 | 状态 | 说明 |
|------|------|------|
| stale cleanup | ✅ 已完成 | 删 `OmniTrainer` 里 stale 的 `OmniBuildArgs` / `OmniModel.build_from_args` 引用；`_build_model` / `_build_model_assets` 留 `NotImplementedError` stub。文件可 import，D1 collator 路径 (`_build_collate_fn`) 正常单测。 |
| `build_parallelize_model` 多 path 扩展 | 🟡 部分保留 | **保留**：`OmniModel` sub-modules 提升为顶层 attribute（取代 `nn.ModuleDict` 中介、`modules_dict` 改 property dict view），`model.named_children()` 直接枚举 `[(<name>, sub_module), ...]`；`parallelize_model_fsdp2` 抽出的 `_materialize_and_load_weights` helper（后来被 DDP 路径复用）。**已删除**：它的 `Mapping[str, str]` 分支（按 named_children 分子树加载）——D2.3 最终改成每个 `ModuleRuntime` 各自调一次 `build_parallelize_model(weights_path=args.model_path)`，顶层一次性 wrap 整个 OmniModel 的方案没有落地，该分支始终没有生产调用方。若将来恢复顶层 wrap，可从 commit `5d0abc453` 取回（含 strict bijection 与 PEFT 拒绝的完整单测）。 |
| MODELING_REGISTRY 注册 + OmniTrainer 重写 | ✅ 已完成 | V2 子模块经 `OMNI_MODEL_REGISTRY` 解析；`ModuleRuntime` 按模块 `args.model_path` 构建并 wrap，`build_omni_model()` 把它们装配成 `OmniModel`。 |

> 旧版本（被回退）尝试过用 single-path `build_foundation_model` 直接加载到 cpu/cuda、再让 parallelize 阶段 `weights_path=None` 跳过 weight load。这条路有三个 runtime 阻断点：(1) V2 子模块未注册到 `MODELING_REGISTRY`，`build_foundation_model` 第一次 call 就抛 `Unknown Modeling name: janus_siglip`；(2) `parallelize_model_fsdp2` 在 `weights_path=None` 时会跑 `model.to_empty + init_weights()` 重置权重；(3) `init_device='cpu'` 下 `auto.py:242` 让 rank>0 拿空权重又没后续 broadcast，多 rank 静默发散。所以直接做终态比绕道更稳。

### Save：每个 module 自己的 callback

每个 module-trainer 在初始化时挂一个自己的 `OmniModuleHfCallback` / `OmniModuleLoraCallback` 实例（定义于 [`veomni/trainer/omni/omni_module_trainer.py`](veomni/trainer/omni/omni_module_trainer.py)，分别继承 `HuggingfaceCkptCallback` / `HFLoraCkptCallback`）。orchestrator 的 `on_*` cascade 触发 save 时，各 module-trainer 各自写自己的 subfolder：

```
output_ckpt_dir/
├── tokenizer/                          # global; written by OmniTrainer top-level callback
│   ├── tokenizer.json
│   ├── special_tokens_map.json
│   └── tokenizer_config.json
├── janus_siglip/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── janus_vqvae/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── janus_llama/
│   ├── config.json
│   └── model.safetensors
├── janus_text_encoder/                   # wte + lm_head（无 per-module tokenizer 副本）
│   ├── config.json
│   └── model.safetensors
└── omni_config.yaml                    # OmniConfig snapshot（modules / nodes / edges / graphs）
                                        # tokenizer_path 指向 output_ckpt_dir 根
```

- "整体打包存"由顶层 callback 触发（不重复写每个 module）。
- config.json 由各 callback 顺带保存（HF 风格 `save_pretrained`）；`model_type` 字段自动随 config 落盘——下次加载直接 from_pretrained。
- 训练继续时 weights_path 直接指向各 subdir，无需再过拆模型脚本。

### Assets

| 类别 | 是否全局 | 存放位置 |
|------|---------|---------|
| **tokenizer** | per-module（text_encoder 持有） | 住在 ``janus_text_encoder/tokenizer/`` 等模块子目录；build 时挂 ``module._tokenizer`` |
| vision processor / image processor | per-module | 跟随该 module subdir（如 `janus_siglip/preprocessor_config.json`） |
| audio feature extractor | per-module | 跟随该 module subdir |
| chat template 逻辑 | per-module（代码） | 住在 ``text_encoder/modeling.py``，不是独立 asset 文件 |

- **每个 vision/audio module 0 或 1 个 processor asset**——不重复。
- **special-token id 不进 config.json / YAML**：boi / eoi / eos 等由 ``module._tokenizer`` 在运行时解析；FSM 只听 ``module_signal``。
- **pure DiT 不需要 tokenizer**：纯 DiT 配置里完全不写 text_encoder module。
- **vocab-bound backbone**（``janus_llama``）本身没有 tokenizer asset——读 ``inputs_embeds`` / ``hidden_states``。

### freeze、gradient_checkpointing 等模块特化字段

写在 `modules.<name>.<field>`，由各模块自己读取并应用。当前版本：
- `freeze: true` → 模块构造完后冻结所有参数（不参与训练）。
- `gradient_checkpointing: true` → 模块 init 后调 `gradient_checkpointing_enable()`。

并行配置（`micro_batch_size` / `dp_size` / `sp_size` / `tp_size` / `cp_size`）**不在** `modules.<name>` 下接受——参见 § "micro_batch_size / DP / SP 一致（暂时全局对齐）" 一节。

---

## 数据路由：raw_batch = `conversation_list` + module-driven processing

> **状态**：本节描述 V2 的**目标契约**。当前 `veomni/data/multimodal/multimodal_chat_template.py` 沿用 V1 的"chat-template 工具层 + N 倍预展开 + backbone scatter"形态。本节描述的"raw_batch 单字段 + module 全责处理"是后续按 feature 迁移的目标形态，按 feature 一项一项实施。

### 设计原则

V2 框架的数据流由两条核心契约定义：

1. **数据完全 model-agnostic**：raw_batch 里只有 `conversation_list` 这**唯一字段**——每条 sample 是一个 `list[dict]`，每个 item 仅含通用字段 `type` / `value` / `role` / `loss_mask`。**没有 input_ids、没有 pixel_values、没有 image_pos**。同一份 SFT 数据集可同时喂给 Janus / Qwen-Omni / Bagel 等任意 ug 模型，每个模型自己解析、自己 tokenize、自己处理 image —— **数据集和模型解耦**。

2. **module 通过 forward `return dict` 修改 raw_batch**（不是直接 mutate）：每个 module 的 forward 仍是 `forward(**kwargs) -> Dict[str, Any]` 风格（HF 兼容、单测纯函数）；OmniModel 框架收到返回 dict 后**立即按 edge.as 写回 raw_batch**（不通过 edge 通道传递给下游）。下游 module 从同一 raw_batch 按自己声明的 input keys 取。这等价于"data 100% 走 raw_batch、module 之间不互相返回值"，但保留了 kwargs 风格 API 和 edge 显式契约。

### Raw conversation item schema

```python
{
    "type":      "text" | "image" | "video" | "audio" | "vq_image"
                 | "boi" | "eoi"      # ← module forward 阶段插入的边界 marker
                 | "audio_bos" | "audio_eos" | ...,
    "value":     <str | torch.Tensor>,  # text: string；
                                        # image/video: torch.Tensor (C, H, W) uint8 已 resize、未 normalize；
                                        # audio: torch.Tensor (T,) float32 已 resample；
                                        # boundary marker (boi/eoi/...): None 或省略
    "role":      "system" | "user" | "assistant",   # 角色标签
    "loss_mask": 0 | 1,                  # 是否参与 loss（默认 int(role=="assistant")，dataset 可覆盖）
}
```

> **schema 说明**：
> - `role` 是**唯一的"谁说的"语义字段**——`role == "assistant"` 是 backbone splice / labels 计算时识别 supervised 段的依据；`role == "system"` 让 text_encoder 拼 chat template 时识别 system prompt 前缀；`role == "user"` 走普通 user turn。框架不再持有冗余的布尔字段（如 from_assistant）；模块内部如需该语义直接写 `item["role"] == "assistant"`。
> - `loss_mask` 是**显式 per-item override**——常见场景下 dataset 不写它，框架按 `int(role == "assistant")` 默认；多轮对话里某些 assistant turn 不算 loss（e.g. revision step）时 dataset 显式置 0。
> - `value` 类型按 `type` 决定：image/video 是 `(C, H, W) uint8` tensor（IPC 友好、保留所有原始信息；下游 ViT/VAE 自己 normalize + patchify）；audio 是 1D float32 waveform。

例如"理解一张图 + 生成一段文 + 生成一张图"对话进入 raw_batch 时的形态：

```python
raw_batch["conversation_list"][0] = [   # 第 0 个 sample
    {"type": "text",     "value": "You are a helpful assistant.",      "role": "system",    "loss_mask": 0},
    {"type": "text",     "value": "Describe this and draw similar:",  "role": "user",      "loss_mask": 0},
    {"type": "image",    "value": <Tensor (C, H, W) uint8>,            "role": "user",      "loss_mask": 0},
    {"type": "text",     "value": "A cat on a sofa.",                   "role": "assistant", "loss_mask": 1},
    {"type": "vq_image", "value": <Tensor (C, H, W) uint8>,            "role": "assistant", "loss_mask": 1},
]
```

注意 `image` / `vq_image` item 的 `value` 已经是 **resized uint8 tensor**（不是 path、不是 PIL，也未做 channel-mean normalize）—— resize 由 `multimodal_transform.py` 的减重版工具层在数据加载阶段完成（见下）。下游 ViT/VAE 在 forward 阶段自己跑 normalize + patchify。

### 数据流分层（六层串行）

```
┌─ Layer 1: jsonl on disk ────────────────────────────────────────────────┐
│  每行 = 一条 sample = list[dict]，item.value 是 path / string             │
└───────────────────────────────┬────────────────────────────────────────┘
                                ▼
┌─ Layer 2: multimodal_transform.py（减重版工具层）─────────────────────────┐
│  对 conversation_list 中的每个 item 按 type 做基础 IO + resize：           │
│    type=image  : item["value"] = read_image(path) → resize → Tensor(C,H,W)│
│    type=video  : item["value"] = read_frames(path) → resize → Tensor(...)│
│    type=audio  : item["value"] = load_audio(path) → Tensor(...)           │
│    type=text   : item["value"] 保持 string                                │
│  ❌ 不做 chat template；❌ 不做 tokenize；❌ 不做 image processor          │
│  （后两者下放到对应 module 在 forward 阶段做）                            │
│  输出仍是 conversation_list（schema 不变，只是 value 升级为 tensor）       │
└───────────────────────────────┬────────────────────────────────────────┘
                                ▼
┌─ Layer 3: dataloader / collator（基础版）────────────────────────────────┐
│  仅把 N 条 sample 包成 batch:                                            │
│    raw_batch = {"conversation_list": [conv_0, conv_1, ..., conv_{N-1}]}  │
│  ❌ 不做任何 sequence-domain padding（input_ids 还不存在）                 │
│  ❌ 不做 SP slice（留给 module 自己在 pre_forward 调）                    │
└───────────────────────────────┬────────────────────────────────────────┘
                                ▼
[OmniModel.forward / generate 入口；raw_batch 起点只有 conversation_list]
                                │
                                ▼
┌─ Layer 4: vision / audio encoder modules（forward 阶段）─────────────────┐
│  ViT / VAE / audio_encoder 各自：                                         │
│    1. pre_forward: 按需调本模块的 collator helper 把 batch 内对应 type    │
│       的 item.value 抽出来 + stack 成 (B*N, C, H, W) tensor，再做该字段   │
│       的 SP slice（注意：切的是 image batch 维 / patch 维，不切 sequence）│
│    2. forward: 跑本模块的 image processor（patch / normalize）→ encoder   │
│       → 产出 image_embeds / vq_token_ids / audio_embeds                  │
│    3. 同时修改 conversation_list: 在每个 image item 前后插 boi/eoi item   │
│       (audio 模块插 audio_bos/audio_eos，video 插 video_bos/video_eos);   │
│       新插入的 marker item 继承原 item 的 role/loss_mask                  │
│    4. return dict 含 conversation_list (modified) + image_embeds + ...    │
│       框架按 edge.as 立即写回 raw_batch                                   │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ raw_batch 现含: conversation_list (含所有
                                │   boundary markers), und_image_embeds,
                                │   gen_image_embeds, audio_embeds, ...
                                ▼
┌─ Layer 5: text_encoder module（base 提供默认实现，family-specific 覆写 chat-template）──┐
│  base 模块：modules/base/text_encoder/                                                  │
│    通用方法：拼接 conversation_list 中的文本/marker、tokenize、产 input_ids/labels、   │
│              wte lookup、按模态 split 输出新的 conversation_list（value=inputs_embeds）│
│  family 模块：modules/<family>/text_encoder/  继承 base/text_encoder/TextEncoder       │
│    自带 tokenizer asset；只 override chat-template 拼接细节（system prompt 前缀格式、  │
│    role marker、boundary token id 表）                                                  │
│  pre_forward: 调 collator helper 抽 batch                                              │
│  forward.encode (一气呵成):                                                            │
│    1. 对每个 sample 的 conversation_list（已被 ViT/VAE 在 Layer 4 加入 boi/eoi marker  │
│       item）按本 family 的 chat template 规则拼接：                                    │
│       - system prompt 前缀 / user / assistant 角色 token                               │
│       - 每个 item 翻译为 token id 序列：                                                │
│           type=text:        tokenizer.encode(item["value"])                            │
│           type=boi/eoi/...: tokenizer.convert_tokens_to_ids("<boi>")                   │
│           type=image/video/audio/vq_image: **1 个** placeholder token id               │
│             （供 backbone splice 时识别 → 扩展成 N patch tokens）                      │
│       - 末尾加 EOS                                                                     │
│    2. 算 labels（image/audio 段对应位置填 -100；text 段按 role + loss_mask）           │
│    3. 算 attention_mask                                                                │
│    4. 过 wte → inputs_embeds（每个 sample 一个 (L, D) 张量；含 text + 1 placeholder    │
│       per modality + boundary token，**还没**展开到 N patch tokens）                   │
│    5. **按模态 split 出新的 conversation_list**——按原 conversation_list 的 item       │
│       边界把 inputs_embeds 切片，每个 item 的 value 替换成该段的 embedding tensor：    │
│       [{type:"text",  value:<Tensor(L_text1, D)>, role:"system",  loss_mask:0},        │
│        {type:"image", value:<Tensor(1, D)>,        role:"user",    loss_mask:0},       │
│        {type:"boi",   value:<Tensor(1, D)>,        role:"user",    loss_mask:0},       │
│        ...]                                                                            │
│       text segment 的 value 是该段所有 token 的 wte embedding（多 token）；image /     │
│       video / audio segment 的 value 是 1 个 placeholder 的 wte embedding（单 token）；│
│       boundary marker segment 的 value 是 1 个 marker token 的 wte embedding。         │
│    6. SP slice（input_ids / inputs_embeds / labels / attention_mask）                  │
│  return: {                                                                             │
│    input_ids, inputs_embeds (flat), labels, attention_mask,                            │
│    conversation_list,    # ← 覆盖 Layer 4 的版本：现在是按模态 split 的，value=embeds  │
│  }                                                                                     │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                │ raw_batch 现含:
                                │   conversation_list (按模态 split, value=inputs_embeds segment),
                                │   und_image_embeds (来自 Layer 4 ViT，每张图 N patch tokens),
                                │   gen_image_embeds (来自 Layer 4 VAE),
                                │   input_ids, inputs_embeds (flat), labels, attention_mask
                                ▼
┌─ Layer 6: backbone（JanusLlama / QwenOmniThinker / ...）──────────────────────────────┐
│  pre_forward:                                                                          │
│    1. 多模态 splice：遍历 split 后的 conversation_list，                               │
│       - text/boundary segment：直接拿 segment.value（已经是 wte embedding）            │
│       - image segment：把 segment.value（1 placeholder embedding）替换成               │
│         und_image_embeds[i] / gen_image_embeds[i]（N patch tokens）                    │
│       - audio segment：同上替换成 audio_embeds[i]                                      │
│       按 segment 顺序 concat 得到完整 inputs_embeds（长度从 L_text+1·M 变成 L_text+ΣN）│
│    2. 同步 splice labels（image 段 -100）/ attention_mask（1）                         │
│    3. compute_position_ids 从 splice 后的最终长度算 position_ids                       │
│    4. SP pad_and_slice                                                                 │
│  forward → hidden_states                                                               │
│  post_forward → SP gather                                                              │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

> **为什么 text_encoder 输出按模态 split 的 conversation_list 而不是 flat tensor**：
> - backbone splice 不需要再扫 input_ids 找 placeholder token id 的位置——直接按 segment 顺序处理；
> - 多个 image / audio / video / vq_image segment 跟 ViT/VAE 输出的 embedding list 一一对应（按 segment 在 conversation_list 中出现的顺序匹配），不需要额外的 image_pos 索引字段；
> - segment 的 role / loss_mask 字段保留——backbone splice 同步 labels 时直接读 segment 元信息；
> - mental model 跟 user 看到的对话结构一致：每个 segment 仍然对应一段语义，只是 value 从原始 string/path 升级到了 embedding。

### 关键不变量

- **数据 100% model-agnostic**：raw_batch 起点只有 `conversation_list`，schema 通用。同一份数据可同时喂给任意 ug 模型。
- **chat-template / tokenize / image processor / audio feature extractor 全部下放给模型**：multimodal_transform.py 工具层只保留基础 IO + resize；不存在框架层级的 chat-template helper、不存在框架层级的 image_pattern 注册表。
- **每个 module 自管自己的 token 拼接**：text encoder（text_encoder）拼 system prompt + 文本 item + 加 eos；ViT 在 conversation_list 中给 image item 加 boi/eoi；audio encoder 给 audio item 加 audio_bos/audio_eos；framework 不感知。
- **collator 在 module pre_forward 中按需调用，不再是 dataloader final-step**：每个 module 自己知道关心哪些字段、怎么 batch、SP 怎么切（ViT 切 image batch 维；text encoder 切 sequence 维）。
- **可选：module 自管的 CPU 预处理可下放到 DataLoader worker（不破坏上面的契约）**：module 的 chat-template+tokenize、image normalize 等**纯 CPU、无权重、无梯度**的 input-prep，默认在 `pre_forward`（主进程）跑，会阻塞 GPU 且无法 prefetch。有此类预处理的子模块在自己的 `processing.py` 下定义 `XxxPreprocessor(Preprocessor)`（对应 HF `AutoProcessor` 哲学：可直接 `XxxPreprocessor.from_pretrained(module_path)` 构建，tokenizer / chat_template / image_processor 都在这个文件内自行 load，不依赖任何 model 实例），并通过 `ModuleMixin.preprocessor_class` 类属性注册；`OmniProcessor`（`processing.py` 顶层）按 checkpoint 目录逐模块调用 `preprocessor_class.from_pretrained`（不经过 `modeling` 的 `build_processor`）收集成 `dict[module_name, Preprocessor]`。dummy 输入不再在构建期由 model 现算现塞：`Preprocessor.bind_dummy_inputs(config, dtype)` 是纯 `(config, dtype)` 的可选后置步骤，preprocessor 构建阶段完全不依赖 model 实例。`OmniTrainer._build_train_dataloader` 在 `_build_model` 之后运行，因此统一调用 `OmniProcessor.bind_dummy_inputs(module_configs, dtype=...)`，直接把每个已建好模块的 `ModuleRuntime.model_config`（内存中已解析好、含 overrides 的配置）传给它——不再重新读盘、也不用再套一遍 config overrides。于是这段逻辑在 **worker** 内执行、借 prefetch 与 GPU 计算重叠，`pre_forward` 退化为「检测 sentinel → 已就绪则只 `.to(device)`，否则回退完整路径」。处理逻辑仍归 module 所有、数据集仍 model-agnostic；只有执行位置从主进程移到 worker。需要权重/梯度的真正 encode 计算（ViT/VAE/wte、VQ 量化）必须留在 forward，不可进 worker。已接入：Janus（text_encoder / siglip / vqvae）、Bagel（text_encoder / siglip_navit / vae）、Qwen3 text_encoder（同时覆盖 qwen3_moe，二者共用该 text_encoder）、Qwen3-VL（text_encoder + vision）。各 backbone（`janus_llama` / `qwen3_*_llm`）只读 embeds/hidden_states、无 CPU 预处理，不声明 `preprocessor_class`（`from_pretrained` 返回 `None`、零开销）。
  - 配套：`naflatten`/`unflatten`（`veomni/utils/tensor_utils.py`）的形状元数据保持在 **CPU**，避免 `unflatten().tolist()` 在 backbone / text_encoder 的 post_forward 触发逐段 device→host 同步（这些同步会阻塞在尚未跑完的 forward kernel 上）。
  - **推理复用同一套 preprocessor（训练/推理对齐）**：`OmniInferencer._preprocess_request` 在进入 FSM 之前，对 request 的 `conversation_list`（单样本，包成 batch-of-1）按 graph 顺序跑一遍各模块的 `Preprocessor.__call__`，与训练时 `OmniProcessor` 收集的这套 preprocessor 完全同一份实例、同一套逻辑。`Preprocessor.__call__(conversation_list, *, inference=False)` 用 `inference` flag 区分训练/推理专属行为：image/codec 模块**跳过 dummy 注入**（推理无 FSDP anchor 需求，且 `bind_dummy_inputs` 在推理路径下本就不会被调用），text encoder **追加 generation prompt**（`tokenize_conversation(..., add_generation_prompt=True)`）。于是 module 的 `generate` 不再现场 process，只负责 pack → encode → scatter，和训练 forward 同构：vision 直接读回 patches 跑 ViT；text encoder 首个 FSM step 由 `TextEncoderModuleMixin._encode_prompt` 直接 embed 已 tokenize 的 prompt（`_prompt_encoded` flag 区分首步 vs 后续 AR）。唯一例外是推理 FSM **中途**生成的 item（如 janus siglip 对生成图的 `image_output` 回编码），preprocessor 见不到，仍在 `generate` 内现场处理。
  - **dummy 输入也由 worker 构造**：vision/codec 模块（siglip / vqvae / qwen3vl_vision）在某 micro-batch 没有真实 image/video 时，预处理器按模块几何 append 一个 `role="dummy"` 占位 item（value=零 pixels，`item.source` 标识模块，qwen3vl 另带 grid）。真实 item 也在 CPU 预处理时打上 `item.source`，于是**真实数据与 dummy 同构**：`forward_pre` 用单个 `iter_desired_items(sources=[_SOURCE])` 一把取出（一个 batch 要么全真要么全 dummy），stack→喂 forward + `is_dummy` 标记；`forward_post` 同样按 source 取回、逐 item 回写 embed，**不判断 None / dummy / role**。
  - **dummy 仅在 training + FSDP 下需要**（「零输入 GPU forward 跑过可训练权重拿梯度」的 reduce_scatter anchor），所有 fsdp/training 判断统一收敛到 `modeling`，且**永远产出真实形状的零张量而非 `None`**，让 pre/post 保持 branch-free：
    - `modeling.forward/encode`：`if is_dummy and not (self.training and fsdp_enabled): 造零张量 else: 真跑 encode`。即只有「训练 + 开 FSDP」才真跑 codec/ViT 当 anchor；推理（eval）或不开 FSDP 时跳过 forward，用 `_dummy_*` helper 造一份与真实 encode 同 batch/同 token 数的零张量（vqvae: `image_embeds`+`vq_token_ids`；siglip: `image_embeds`；qwen3vl: `image_embeds`+每层 deepstack feature）。
    - vqvae 还有 encode→decode 两段依赖（decode 需要 encode 产出的 label），因 dummy 现在恒有真实形状的 label/embed，`decode_pre` 拼 dummy span 与真实 gen image 完全同构；`modeling.decode` 同样按 `self.training and fsdp_enabled` 决定 dummy 走 `generation_head` anchor 还是直接返回 `0.0` loss。

    CPU 预处理一定会跑（worker 或主进程 collator），故 `pre_forward` 无需 eager 回退。
  - **「worker 已 normalize」无需独立哨兵**：siglip / vqvae 真实 item 经 role 即可区分（`forward_pre` 直接 stack 已是 model-dtype 的 `value`）；qwen3vl_vision 用 per-item `_OMNI_GRID`（grid 元数据）的存在与否区分「训练已处理 patches」与「eager 推理 raw 图」，复用同一个 grid key，不再额外维护 `*_pixels` 哨兵。
- **forward 阶段两次"形态变换"**：(1) Layer 4 ViT/VAE 在 conversation_list 中**插入新 item**（boi/eoi marker）、原 image item 的 value 不变（仍是 resized tensor）；(2) Layer 5 text_encoder 把 conversation_list **按模态 split** 输出新 list，每个 item 的 value 升级为 inputs_embeds（text segment 多 token；image/audio/marker segment 单 token placeholder）；(3) Layer 6 backbone splice 把每个 modality segment 的 1-token placeholder embedding 替换成 N 个 patch tokens，concat 输出 flat inputs_embeds。labels / attention_mask / position_ids 在 Layer 5 和 Layer 6 各重新计算一次。
- **module forward = kwargs + Dict 返回**（W2 风格）：API 不变，但语义改成"返回 dict 立刻被框架按 edge.as 写回 raw_batch"，data 不通过 edge 通道传递。下游 module 从同一 raw_batch 按 input keys 取。
- **graph topology 自动从 edge dependency 推**：因为 ViT/VAE 修改 conversation_list、text_encoder 读 conversation_list，topo 序自动 ViT/VAE → text_encoder → backbone，**不需要显式顺序约束 edge**。

### 与 V1 主线的迁移路径（每条 feature 独立 PR）

1. **Feature D1**（基础）：multimodal_transform.py 减重 + list-only collator——`process_seedomni_example` 移除 chat_template + tokenize + image_processor 调用，只保留 IO + resize；输出 `[{"conversation_list": [...]}]`。`SeedOmniCollator` 不做任何 batching/SP/padding，只把每个 sample 的 `conversation_list` 收集成 `list[list[dict]]`。OmniTrainer 在 `data_type='seedomni'` 时改走该 collator。
2. **Feature D2**（基础）：OmniTrainer build flow 重写。拆三段独立 PR 推进——
   1. ✅ **D2.1 — stale cleanup**：删 `OmniTrainer` 里失效的 `OmniBuildArgs` / `OmniModel.build_from_args` 引用；`_build_model` / `_build_model_assets` 留 `NotImplementedError` stub。让文件可 import，启用 D1 wiring tests；`OmniTrainer.__init__` 仍是软失败状态（在 `_build_model` raise）。
   2. ✅ **D2.2 — extend `build_parallelize_model`**：把 `OmniModel` sub-modules 提为顶层 attribute（取代 `nn.ModuleDict` 中介，`modules_dict` 改 property dict view），让 `model.named_children()` 直接出 `[(<name>, sub_module), ...]`；`parallelize_model_fsdp2` 抽出 `_materialize_and_load_weights` helper（`None` 随机 init / `str` 单 snapshot），后来 DDP 路径也复用它。当时一并加的 `weights_path: Mapping[str, str]` 分支是为下面的 D2.3 顶层 wrap 铺路，D2.3 换方案后已删除（见 § "Build & 权重加载"）。
   3. ✅ **D2.3 — registry + build flow**：V2 子模块经 `OMNI_MODEL_REGISTRY` 解析；`_build_model` 走 `build_omni_model()` → 每个 `ModuleRuntime` 在自己的 `use_parallel_state(module_name)` 作用域里 build + wrap + 按 `args.model_path` 加载。顶层一次性 wrap 整个 OmniModel（以及 D2.2 的 dict 分支）**未采用**。

   **注意**：即使 D2.3 全部完成，trainer 仍**无法**端到端 train —— module forward 的输入契约仍是 V1 风格 flat tensor batch，`conversation_list` 喂不进去；这要等 D3-D5 把 chat template / image processor / splice 全部迁移到 module forward 后才能跑通。详细 build flow 设计见 § "Build & 权重加载"。
3. **Feature D3**（vision）：把 image processor + boundary marker 注入逻辑搬进 ViT/VAE 的 forward。
4. **Feature D4**（text）：把 chat template + tokenize 搬进 text_encoder 的 forward；text_encoder 升级为 model-specific（modules/<family>/text_encoder/）。
5. **Feature D5**（backbone）：splice + compute_position_ids 在 backbone pre_forward 中接管最终长度对齐（这条之前讨论过）。

D1-D2 是数据层 / 训练入口减重；D3-D5 是模型层接管。每步都向后兼容（中间状态**可 import / 可单元测**，但 D2 之前的 trainer.train() 不会跑通——这是已知的过渡期，由 D3+ 收尾）；最终目标是上述六层架构。

### Backbone `pre_forward` 完成多模态 splice + 长度对齐（target contract）

> **状态**：本节同样描述目标契约。当前 `JanusLlama.pre_forward` 走的是 V1 兼容的 scatter 路径（input_ids 里已经有 N 个 image_pad placeholder，`masked_scatter` 替换 placeholder embedding 而不改长度，配合 `+ x.sum() * 0.0` 锚点保证 FSDP grad sync）。迁移到本节描述的 splice 形态是 Feature D5——前置依赖 Feature D4（text_encoder 在 forward 中产出"每张 image 仅 1 个 placeholder"的 input_ids）。

text_encoder（Layer 5）已经把 conversation_list 拼接 + tokenize + wte 得到一份**按模态 split** 的新 conversation_list，其中：
- text / boundary segment 的 value 是该段所有 token 的 wte embedding；
- image / audio / video / vq_image segment 的 value 是 1 个 placeholder token 的 wte embedding（`(1, D)` 张量）。

vision / audio 等 encoder 各自吐出 embedding list（Layer 4 已写到 raw_batch，每张图 N_i 个 patch tokens）。**真正的拼接（splice）在 backbone（如 `janus_llama`）的 `pre_forward` 里完成**——遍历 split 后的 conversation_list，**按 segment 顺序**把每个 image / audio / video 的 1-token placeholder embedding 替换成 N 个 patch token embedding，最后 concat 出完整 inputs_embeds。inputs_embeds 长度从 `L_text + 1·num_modal` 变成 `L_text + sum(N_i)`：

```python
class JanusLlama(JanusLlamaModuleMixin, PreTrainedModel):
    def pre_forward(
        self,
        conversation_list,                   # 按模态 split 的 list of dict，每个 item 含 type/value/role/loss_mask
                                             # value 已经是 wte embedding 张量
        und_image_embeds=None,               # 来自 vit_encode（list[Tensor]，每张图 N_i 个 patch token）
        gen_image_embeds=None,               # 来自 vae_encode
        attention_mask=None, labels=None, position_ids=None,
        **_,
    ):
        # 按 segment 顺序遍历，替换 image / audio segment 的 placeholder embedding，
        # 然后 concat 出完整 inputs_embeds。
        # labels / attention_mask 同步用 segment 元信息（type / role / loss_mask）
        # 重新生成（image 段 labels=-100、attention_mask=1）。
        # position_ids 在 splice 后由 backbone 自己的 compute_position_ids 重新算
        # （M-RoPE 类模型需要 image grid 才能算位置；1D RoPE 直接 arange 新长度）。
        inputs_embeds, attention_mask, labels = splice_by_segments(
            conversation_list,
            und_image_embeds_iter=iter(und_image_embeds or []),
            gen_image_embeds_iter=iter(gen_image_embeds or []),
        )
        position_ids = self.compute_position_ids(
            attention_mask=attention_mask, image_grid_thw=...
        )["position_ids"]
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
        }
```

注意：
- backbone **不需要扫 input_ids 找 placeholder 位置**——按 segment 顺序就能定位（image segment N 个 patch tokens 替换 1 个 placeholder embedding 即可）；
- backbone **不需要 image_pos / und_image_pos / gen_image_pos 等索引字段**——这些字段在 V1 的"N 倍预展开 + masked_scatter"里有用，新设计里多余；
- ViT/VAE 输出的 image_embeds list 顺序跟 conversation_list 中的 image / vq_image segment 顺序一一对应（按出现顺序匹配），不需要额外的对齐字段。

这样：
- **chat template 不在 HF tokenizer 内部**——是 `text_encoder` module 的 forward 实现细节；HF tokenizer（住在 `text_encoder` 内部）只懂 string → token id。
- **多模态拼接没有全局路由表**——每个 backbone 自己决定如何 splice（不同 backbone 可能 cross-attn 而非 splice，比如 DiT 风格用 cross-attn 消费 text_encoders，根本不做 splice）。
- **HF tokenizer / text_encoder 都不感知 image patch 数**——text_encoder 在 input_ids 里每张 image 只放 1 个 placeholder token（不需要懂 N）；展开 N 倍由 backbone 在 splice 阶段完成，依赖 image_embeds 的实际形状（来自 image processor，是 vision encoder 模块的私有 asset）。
- **labels / attention_mask / position_ids 同步在 splice 阶段对齐**——image 段 labels 填 -100，attention_mask 填 1，position_ids 由 backbone 重新算（参见"Position IDs"一节）。
- **模态新增 = 加一个 module + 一条 edge**——`pre_forward` 在 `embeds_per_modality` 里多接一种模态、多写一段 splice 即可。

### Position IDs：backbone 私有 schema，splice 后由 backbone 重算

> **状态**：本节描述目标契约。当前 `JanusLlama.pre_forward` 接受外部传入的 `position_ids`（沿用 V1 主线 `multimodal_transform.py` 在数据预处理阶段调 `position_id_func` 算好的形态）；本节描述 V2 把这一步迁移到 backbone 内部 `compute_position_ids` 钩子。

position_ids 计算的输入依赖 image / video 的 patch grid（M-RoPE 类模型给 image 内部分配 2D `(h_idx, w_idx)`，给 video 分配 3D `(t_idx, h_idx, w_idx)`），所以**必然**要在 image_embeds 已经 splice 进 inputs_embeds 之后才能算 —— 因为 splice 本身就是 image 占多少 token、occupy 哪些位置的最终 source of truth。

backbone 可在 `modulemixin.py` 按需 override：

```python
class JanusLlamaModuleMixin(ModuleMixin):
    def compute_position_ids(
        self,
        *,
        input_ids: torch.Tensor,        # splice 后的最终 token 序列（不是 placeholder 序列）
        attention_mask: torch.Tensor,   # splice 后的 mask
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        audio_lengths: torch.Tensor = None,
        **_,
    ) -> Dict[str, torch.Tensor]:
        """从 splice 后的 input_ids 算出 backbone 期望形状的 position_ids。

        默认 1D arange（适用于 LLaMA / 普通 Transformer）；M-RoPE 类
        backbone（Qwen-VL / Qwen-Omni）override 这个方法返回 (3, L) 的
        多维 position_ids。
        """
        L = input_ids.shape[-1]
        return {"position_ids": torch.arange(L, device=input_ids.device).unsqueeze(0)}
```

调用时机有两条路：

| 场景 | 调用位置 | 备注 |
|---|---|---|
| **训练 / 推理 prefill** | backbone 的 `pre_forward`，在 splice 完成之后 | 一次性算整段 prompt 的 position_ids；这条路是默认 |
| **推理增量 decode**（FSM 每个 step） | backbone 的 `generate_step` 内部，按 `rope_deltas` 增量算 | 不再调 `compute_position_ids`；新 token 的 position 由 backbone 自己根据 prev + rope_deltas 算 |

不变量：
- **数据预处理阶段不再算 position_ids**——这条信息流之前在 V1 `multimodal_transform.py` 里走 `position_id_func` 的形态，迁移到 V2 后由 backbone 自己拥有，因为 splice 在 backbone 内部，splice 之前算的 position_ids 没有意义。
- **`compute_position_ids` 是 backbone 的私有 schema**——其他 module（vision encoder / VQ codec / text embed）不需要这个钩子；图层 / 数据 / collator 都不感知 position_ids 的形状（1D vs 3D vs 含 audio time）。
- **SP slice 跟 input_ids 同步**——splice 后再算 position_ids，再 SP `pad_and_slice`，这部分跟现有 `JanusLlama.pre_forward` 的 SP 处理顺序一致（只是顺序变成 splice → compute_position_ids → SP slice）。

### Per-module 数据处理责任清单（target contract）

每个 module 都通过 `pre_forward` / `forward` / `post_forward` 中的某些步骤参与下面的责任分布。**collator helper / SP slice 由各 module 在自己的 `pre_forward` 内按需调用**——没有全局 collator、没有全局 SP slice 节点。

| 模块 | 主要职责 |
|------|----------------------|
| `vit_encode` / `vae_encode` 等 **vision encoder** | (1) 从 conversation_list 抽 image / vq_image item.value（已是 resized uint8 tensor）→ stack 成 patch batch tensor + normalize；(2) 用本模块自带的 image processor 跑 patch / normalize；(3) encoder forward → image_embeds；(4) **修改 conversation_list**：在每个 image / vq_image item 前后插 `{type: "boi"}` / `{type: "eoi"}` item（继承原 role / loss_mask）；(5) 按需 SP slice 自己的字段（image batch / patch 维），不动 sequence 维 |
| `audio_encode` | 同上但模态是 audio：抽 audio item.value → feature extractor → encoder → audio_embeds；在 conversation_list 中给 audio item 加 `audio_bos` / `audio_eos` marker |
| **text_encoder**（base 在 `modules/base/text_encoder/`，family 在 `modules/<family>/text_encoder/` 继承之）| (1) 自带 tokenizer asset；(2) 接受已经被 vision/audio module 修改过的 conversation_list；(3) 按本 family 的 chat template 规则拼接 token_id 序列（按 role 写 system/user/assistant prefix、含 EOS / boi-eoi 等 marker token；image / audio / video / vq_image item 用 1 个 placeholder token id 占位）；(4) 算 flat labels（image/audio 段填 -100，text 段按 role + loss_mask）；(5) 算 flat attention_mask；(6) 过 wte → inputs_embeds（flat）；(7) **按模态 split** 输出新 conversation_list（item.value=该段 inputs_embeds segment）；(8) SP slice sequence-domain tensors |
| `<backbone>`（`janus_llama` / `qwen_omni_thinker` / ...）| **按 segment 顺序遍历 split 后的 conversation_list 做 splice**：image/audio segment 的 1-token placeholder embedding → N 个 patch token embedding（来自 ViT/VAE）；text/boundary segment 的 value 直接保留；concat 出最终 inputs_embeds。同步用 segment 元信息重算 labels / attention_mask；调 `compute_position_ids` 算 position_ids；最终 SP pad_and_slice |
| `tok_decode` / `vae_decode` | 直接读上游 hidden_states，跑 head + sample / 算 loss；SP-agnostic（backbone post_forward 已 gather） |

注意几点：
- **没有 chat_template 这个独立 module**：chat template 拼接逻辑住在 text_encoder（每个 family 一份）；boundary marker 注入由对应模态的 encoder 负责。
- **同一 module 不同 method 用不同 node 标识**：text_encoder 上典型两个 node —— `tok_encode`（method=encode：chat-template + tokenize + wte → split conversation_list）和 `tok_decode`（method=decode：hidden_states → logits → lm_loss / sample）。tied weights 时两个 node 共享同一份 `embed_tokens.weight` 矩阵。如果某些场景需要把 chat-template / tokenize / wte 三步进一步拆成多个 node，也是同样的模式（在 text_encoder 上声明多个 method，对应多个 node 共享同一实例）。
- **graph topology 顺序**：因为 ViT/VAE 输出 `conversation_list`、text_encoder 输入 `conversation_list`，edge dependency 自动让 ViT/VAE 排在 text_encoder 之前；text_encoder 输出 `inputs_embeds`，backbone 输入 `inputs_embeds`，自动让 text_encoder 在 backbone 之前。**不需要显式顺序约束 edge**。

### 采样策略与 CFG（per-request runtime state）

> **状态**：本节描述目标契约。当前 V2 SeedOmni 代码尚未实现推理 CFG（V1 主线只有训练侧的 `cfg_ratio` 随机 condition drop，发生在 `MultimodalChatTemplate` 工具层；新设计里训练 CFG 改在 `text_encoder.forward` 内部对随机选中的 sample 把 condition 段替换成 pad token；推理 CFG 是从零设计的 V2 feature）。

`temperature` / `top_p` / `repetition_penalty` / `cfg_weight` 这一类 **per-request runtime sampling state**，跟 KV cache 同质，**不进入 graph / YAML 抽象**。它们的存在不影响 FSM 结构、不增加 node 数、不改变 edge schema —— 只通过 `OmniModel.generate()` 的 `sampling: dict` 参数传入，写入 ctx，由 backbone module 自己消费。

```python
ctx = model.generate(
    request=...,
    sampling={
        "temperature": 1.0,
        "top_p": 1.0,
        "cfg_weight": 5.0,                 # 1.0 = 不启用 CFG，零开销
        # parallel_size 不在 sampling dict——它是 module config 字段，见下文
    },
)
```

#### CFG 是 backbone 私有的 batch-axis 机制

CFG 的 cond / uncond 双路 forward 通过 **batch 维 2x 平铺** 实现（不是两次串行 forward call、不是 graph 上的 cond/uncond 分叉），这跟 Janus 官方 T2I 推理一致。具体由 backbone module 自己处理：

1. **prefill 第一步**（backbone 的 `pre_forward`）：检测 `ctx["sampling"].get("cfg_weight", 1.0) != 1.0`。
2. 若启用，调用 `build_cfg_uncond_inputs` 构造 uncond 分支（pad token 从 `module._tokenizer` 读取）。
3. multimodal splice 之后，把 `inputs_embeds` / `attention_mask` / `position_ids` 在 batch 维复制成 2x（偶数行 cond，奇数行 uncond），送进 backbone forward。
4. **每个 image_vq generate_step**：backbone forward 得到 (2N, V) logits，自己拆 `cond = logits[0::2]`、`uncond = logits[1::2]`，按 `cfg_weight` merge，sample 出 next_token，再在 batch 维 2x 复制喂下一步。**FSM / graph / 上层 caller 看到的 batch 始终是 1x**（即 `parallel_size`，见下）。
5. **退出 image_vq state 时**（FSM transition 触发的 `module_signal(image_complete)`）：backbone 在 hook 中把 2x batch shape 的 KV cache **丢弃**（不能复用给后续 text state，因为 batch shape 不兼容）。这条跟 `#13 KV cache 由模块自管` 一致。

#### `parallel_size`：backbone 推理时 config（不是 sampling 参数）

Janus 风格的 T2I 推理一次生成 `parallel_size` 张图（共享 prompt，独立 sampling）—— 这是 **module 自己的推理优化**，跟模型实现耦合（`JanusLlama` 的 KV cache 布局、`JanusVQVAE` 的 batch decode 都依赖这个值），所以放进 **module 的 PretrainedConfig**，不放进 sampling dict：

```python
# JanusLlamaConfig / JanusVQVAEConfig
class JanusLlamaConfig(PretrainedConfig):
    parallel_size: int = 1   # T2I 推理时一次生成多少张图；interleave / understanding 必须是 1

class JanusVQVAEConfig(PretrainedConfig):
    parallel_size: int = 1   # 必须与 JanusLlama 的 parallel_size 一致
```

约束：
- **进入 image_vq state 时**，backbone 的 hook 一次性把 KV cache batch 维扩展成 `parallel_size`（cond 路径 N 张图，N=parallel_size）。如果同时启用 CFG，再 2x 扩展到 `2 * parallel_size`。
- **`parallel_size > 1` 仅 T2I 模式支持**，不支持 interleave。原因：interleave 模式下 image_vq state 之后还要切回 text state，而 `parallel_size > 1` 把 batch 维彻底改写（每个 prompt 实例膨胀成 N 张独立图），切回 text 时无法干净地降回 batch=1。`graph_infer_gen.yaml`（T2I）是唯一允许 `parallel_size > 1` 的入口；`graph_infer_interleave.yaml` / `graph_infer_und.yaml` 必须 `parallel_size = 1`。
- **同一对 backbone + VQ codec 必须配同一个 `parallel_size`**（否则 KV cache batch 跟 VQ decode batch 错位）。OmniModel 在 build 时校验：`JanusLlama.config.parallel_size == JanusVQVAE.config.parallel_size`。
- 用户在 `model.generate()` 调用时仍可通过 sampling 字段 override `parallel_size`，OmniModel 在 generate 入口把值写回相关 module 的 config 副本（一次 generate 一个值），允许同一 weights 多种 parallel_size 推理。

#### `build_cfg_uncond_inputs` 钩子（backbone 可选）

```python
class JanusLlamaModuleMixin(ModuleMixin):
    def build_cfg_uncond_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **mm_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """构造 CFG uncond 分支的输入。

        默认 raise NotImplementedError——backbone 不实现就不允许 cfg_weight != 1.0
        （generate() 入口校验时直接 ValueError，避免 silent garbage）。子类按
        自己的 condition-drop 方式 override；pad token id 由 `self.tokenizer.pad_token_id`
        自取（``module._tokenizer``）。

        返回 dict 至少含 `input_ids`（uncond 版）；其他字段未 override 时
        fallback 到 cond 输入。
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support classifier-free guidance "
            "(cfg_weight != 1.0). Implement build_cfg_uncond_inputs to enable it."
        )
```

JanusLlama 的实现（伪码）：

```python
class JanusLlama(JanusLlamaModuleMixin, PreTrainedModel):
    def init_omni_state(self):
        super().init_omni_state()
        # _boi_token_id / _pad_id 等在 build 后从 module._tokenizer 解析

    def build_cfg_uncond_inputs(self, *, input_ids, attention_mask, **_):
        uncond_ids = input_ids.clone()
        # Janus 约定：保留 BOS 和最后的 image_start，其余替成 pad
        uncond_ids[..., 1:-1] = self._pad_id
        return {"input_ids": uncond_ids}
```

#### Sampling / CFG / parallel_size 不变量小结

- **sampling 状态完全不进 graph 抽象**：YAML 没有 `cfg_*` 字段，`generation_graph` 没有 cfg-aware state，`edges` 没有 cond / uncond 分叉。
- **`cfg_weight=1.0` 与 `parallel_size=1` 是零成本默认**：backbone 检测后跳过 batch 维平铺，性能跟无 CFG / 单图完全一致。
- **`parallel_size > 1` 仅 T2I 模式支持**：interleave / understanding 推理强制 `parallel_size=1`，OmniModel build 时校验 + generate 入口再 assert 一次。
- **2x batch shape KV cache 在退出 image_vq state 时由 backbone 丢弃**：跟 KV cache 由模块自管一致，不引入新生命周期概念。
- **`build_cfg_uncond_inputs` 默认 NotImplementedError**：未实现的 backbone 不允许 `cfg_weight != 1.0`，generate 入口校验时抛 ValueError。
- **pad token id / image_start id 由 `module._tokenizer` 自取**，不写入 sampling dict。

---

## 文件结构

模块按 **model family** 组织：每个 family 的子模型放到 `modules/<family>/`，跨 family 复用的轻量模块放到 `modules/base/`。每个子模型一组 (`configuration_xxx.py`, `modeling_xxx.py`, `processing_xxx.py`) 三件套，`model_type` 写在 `configuration_xxx.py`（参考 [`veomni/models/diffusers/wan_t2v/wan_condition/configuration_wan_condition.py`](veomni/models/diffusers/wan_t2v/wan_condition/configuration_wan_condition.py) 第 7 行 `model_type = "WanTransformer3DConditionModel"`）。

```
veomni/models/seed_omni/                    # 整个目录完全重写，不保留 V1
├── module.py                               # ModuleMixin 基类（共享 hook 默认 + init_omni_state）
├── graph.py                                # NodeDef / EdgeDef：节点 / 边的共享数据类型 + end 关键字
├── training_graph.py                       # TrainingGraph：DAG 视图，按 edges topo 推执行序
├── generation_graph.py                     # GenerationGraph：FSM 视图，按 state.body (edges) 分发
├── configuration_omni.py                   # OmniConfig：纯 HF PretrainedConfig（checkpoint 读写 + graph sidecar）
├── arguments/omni_arguments_types.py       # OmniArguments launcher schema + resolve_omni_model / build_omni_model_runtime：launcher YAML -> OmniModelRuntimeArguments；.to_hf_config() 投影成 HF OmniConfig
├── modeling_omni.py                        # OmniModel：DAG forward + FSM generate + parallel plan 聚合 + 多模块 build/load/save
└── modules/                                # 每个子模块：configuration + modeling（native）+ accelerated（训练钩子）[+ processing]
    ├── base/                                # 跨 family 复用的轻量模块
    │   ├── text_encoder/
    │   │   ├── configuration.py
    │   │   ├── modeling.py                  # TextEncoder(InferenceMixin, OmniPreTrainedModel) —— InferenceMixin 含 generate()
    │   │   └── accelerated.py               # TextEncoderAccelerated(VeOmniMixin, TextEncoder) —— 训练钩子
    │   └── mlp_adapter/                     # 计划中：1024→2048 等通用投影
    │       ├── configuration.py
    │       └── modeling.py
    ├── janus/                               # janus 全家桶
    │   ├── llama/        {configuration, modeling, accelerated}.py
    │   ├── siglip/       {configuration, modeling, accelerated, processing}.py
    │   ├── vqvae/        {configuration, modeling, accelerated, processing}.py
    │   └── text_encoder/ {configuration, modeling, accelerated, processing}.py
    ├── qwen_omni/                           # qwen-omni 全家桶（thinker + talker + ...）
    │   ├── thinker/
    │   │   ├── configuration.py
    │   │   └── modeling.py
    │   └── ...
    ├── qwen3/                                # qwen3-moe/text 全家桶
    │   └── text_encoder/ {configuration, modeling, accelerated, processing}.py
    ├── qwen3vl/                              # qwen3-vl 全家桶
    │   ├── text_encoder/ {configuration, modeling, accelerated, processing}.py
    │   └── vision/        {configuration, modeling, accelerated, processing}.py
    ├── bagel/
    │   ├── llama/
    │   │   ├── configuration.py
    │   │   └── modeling.py
    │   ├── siglip_navit/ {configuration, modeling, accelerated, processing}.py
    │   ├── vae/           {configuration, modeling, accelerated, processing}.py
    │   ├── text_encoder/  {configuration, modeling, accelerated, processing}.py
    │   └── ...
    └── ...
```

每个含 `processing.py` 的子模块都在其中定义 `XxxPreprocessor(Preprocessor)`（`__call__` + `from_pretrained` + 可选 `bind_dummy_inputs`），通过 `preprocessor_class` 类属性挂到对应 `modulemixin.py` 的 `XxxModuleMixin` 上；没有 CPU 预处理需求的子模块（如各 backbone）不设该属性。

文件夹名（如 `janus/siglip/`）已经给出了 `<family>_<sub_module>` 的命名空间，所以子模块内部的文件就用裸 `configuration.py` / `modeling.py` / `processing.py`，不再重复写 `configuration_janus_siglip.py`。每个子模块文件夹有自己的 `__init__.py`，把公开符号 re-export 给上一层（`from .siglip import JanusSiglip, JanusSiglipConfig, JanusSiglipProcessor`）。

`modules/__init__.py` 的 `OMNI_MODEL_REGISTRY` / `OMNI_CONFIG_REGISTRY` 把 HF `model_type` 映射到 `modeling.py` 里的合体类。

---

## 命名规范

| 对象 | 规则 | 例子 |
|------|------|------|
| **module name**（YAML modules 池 key） | 具体模型简名（不带前缀） | `janus_llama`, `janus_siglip`, `janus_vqvae`, `janus_text_encoder`；通用模块用单名（`siglip`、`vqvae`） |
| **edge 端点**（`from` / `to`） | `module[.method]` 字符串；裸 module 训练默认 `.forward`、推理默认 `.generate`，带点原样使用 | `janus_siglip`, `janus_vqvae.encode`, `janus_vqvae.decode`, `janus_text_encoder.encode`, `janus_text_encoder.decode`, `janus_llama` |
| **node 身份**（自动派生） | 规范化 `"<module>.<method>"`（不手写，由端点并出） | `janus_vqvae.encode`, `janus_llama.forward`, `janus_llama.generate` |
| **model_type**（HF config） | 由模型 config 决定，写在 `configuration_xxx.py` | `model_type = "janus_llama"` 等 |
| **拆模型脚本** | 每子模型独立文件夹 + 三件套（短文件名） | `janus/llama/{configuration.py, modeling.py}` + `janus/siglip/{configuration.py, modeling.py, processing.py}` |

**拆模型脚本怎么定子模型 `model_type`**：
- 从某个 family 拆出新子模型时（如 Janus 拆出 `janus_llama` / `janus_siglip` / `janus_vqvae` / `janus_text_encoder`），每个子模型在 `<family>/<sub>/configuration.py` 里写明自己的 `model_type` 字符串。
- 拆模型脚本（`scripts/convert_model/split_<family>.py`）按 sub-config 分别生成 `<output_dir>/<sub_name>/config.json`，`model_type` 字段会随 `save_pretrained` 自动落盘。
- YAML 里只填相对 `model.model_path` 即可，HF AutoConfig 从子目录 `config.json` 读出 `model_type`，再到 `OMNI_MODEL_REGISTRY` 解析类。

---

## 关键设计决策

1. **模块按 model family 组织**：`modules/<family>/` 下放该 family 拆出的子模型；`modules/base/` 放跨 family 复用的小模块。每个子模块三件套：`modulemixin.py`（图钩子）+ `modeling.py`（HF 权重与 `forward`）+ `configuration.py`。

2. **`ModuleMixin` 是 mixin，不是基类**：`XxxModuleMixin(ModuleMixin)` + `PreTrainedModel` 多继承；`init_omni_state` 在 `super().__init__` 后自动调用，`post_init` 留在 `modeling.py`。除训练图 `forward` 外钩子均可选。

3. **raw_batch 全局透明**：raw_batch 是整个 OmniModel forward / generate 共享的 mutable dict，每个 node 默认拿到完整 raw_batch（按自己声明的 input keys 取）。中间输出（hidden states / embeds 等）也写回同一 raw_batch（详见 #15/#16）——edges 是数据依赖契约和拓扑标记，**不是数据通道**。

4. **loss 收集按 `_loss` 后缀**（隐式）+ `to: end` sink 边（拓扑显式）：模块输出的 `*_loss` 键（已 mean 的标量）由 `OmniModel.forward()` 自动收集求和；`to: end` 是拓扑标记，保证图无孤岛，不携带数据语义。

5. **Loss mean 在 module 内部完成**：每个 module 一次 `forward` 把所有 micro-batch 跑完，`post_forward` 内部按 token-sum / token-count 做 token-level mean，吐出标量 `*_loss`。**外层只求和**——这样既保证 token-level 加权正确性（不同 micro-batch 的 token 数不同时不会退化为 batch-weighted），又让 OmniModel 协议简单（单键 `_loss`，无需 `*_loss_token_count`）。

6. **endpoint 即 node**：edge 端点是 `module[.method]` 字符串，node 身份是其规范化的 `"<module>.<method>"`。无独立 `nodes:` / `edges:` 池，无 `output:` / `as:` 路由字段。

7. **无孤岛、无环**：每个 node 至少一条出边（指向另一 node 或 `end`）；任何环（含自环）严格禁止——自环=for-loop，应在模块内部实现。

8. **node 与 module 解耦**：图 node 是 **endpoint**（`module[.method]` 字符串），不是 module 实例。同一 module 可挂多个 method 端点（`janus_vqvae.encode` / `janus_vqvae.decode`），module 实例只有一份，参数共享。**同一个 method 也可承担多重角色**——VQ head 的 `decode` 训练出 loss、推理采样产 embed，按 kwargs 自分派。

9. **method 默认值**：裸 module 端点时，**训练默认 `.forward`、推理默认 `.generate`**。训练时 `forward` 走 FSDP 包装层，其他 method 直调 raw module（FSDP2 透明）。

10. **配置层不写 `model_type`**：YAML modules 池只写相对 `model.model_path`，`model_type` 由 HF AutoConfig 从子目录 `config.json` 读出。

11. **training / generation graph 同构**：`graph_train.yaml` 文件本身就是 edge 列表（一次 DAG 遍历）；`graph_infer_*.yaml` 文件本身就是 FSM，每个 state.body 也是内联 edge 列表。**激活 nodes 由 endpoints 自动并出，执行序由 topo sort 推导**——这是框架唯一的"自动"，结构本身仍要显式给出。

12. **state 步数完全由模块控制**：AR / VQ / DiT 的循环步数不在 YAML 表达——模块的推理方法（`generate_step` 或显式 method）内部实现无论是 next-token 采样还是完整去噪循环，对状态机均透明；何时结束一个 state 由模块 raise 的 `module_signal`（AR/反馈循环）或 `always`（单趟 bridge）决定。框架不持有任何步数预算。

13. **KV cache 由模块自管**：何时复用、何时清空，是 model-specific——Janus 风格（每 token 都过同一 LLM）可复用；DiT 后回到 LLM 必须重算。OmniModel 不感知。

14. **生命周期分层**：weights 加载走 `build_foundation_model` + `build_parallelize_model`；保存由各 module-trainer 的 `OmniModuleHfCallback` / `OmniModuleLoraCallback` 写到 `<ckpt>/<module_name>/`（config + 可选 processor/tokenizer 资产）。Special-token id 不进 ``config.json``——build 时挂 `module._tokenizer` 后运行时解析。

15. **数据流单一抽象 raw_batch；起点 conversation_list**：raw_batch 是 mutable dict，初始只含一个 key `conversation_list`（`list[list[dict]]`，每个 item dict 含 `type` / `value` / `role` / `loss_mask`）。其他所有衍生字段（input_ids、image_embeds、attention_mask、labels、position_ids、hidden_states、...）由各 module 在 forward 阶段产出并通过返回 dict 写回 raw_batch。multimodal_transform.py 工具层只做基础 IO + resize（path → tensor 填回 item.value），不做 chat template 拼接、不做 tokenize、不做 image processor。同一份数据可同时喂给任意 ug 模型——chat template / tokenize / image processor / boundary marker 注入全部由对应 module 自管。

16. **module forward = kwargs + Dict 返回；data 100% 走 raw_batch**：每个 module 的 `forward` 仍是 `forward(**kwargs) -> Dict[str, Any]` 风格（HF 兼容、单测纯函数）；OmniModel 收到返回 dict 后**写回 raw_batch**，**不通过 edge 通道传给下游 module**。下游从同一 raw_batch 按自己声明的 input keys 取。edge 只声明拓扑顺序，不携带 `output:` / `as:` 路由字段。collator helper / SP slice 由各 module 在自己 `pre_forward` 中按需调用。

17. **Sampling state 是 per-request runtime ctx，不进 graph**：`temperature` / `top_p` / `cfg_weight` 等通过 `generation_kwargs` 写入 ctx，由 backbone / text_encoder 消费。CFG batch 2x 平铺、`build_cfg_uncond_inputs` 等是 backbone `modulemixin` 可选能力。`parallel_size` 是 module config 字段（非 sampling 参数）。

18. **token 拼接 / boundary marker / chat template 全部下放给对应 module**：text encoder（text_encoder）拼接 system prompt + 文本 item + EOS + role marker，自带 tokenizer 自带 chat template 实现；ViT/VAE 在 forward 阶段往 `conversation_list` 中给 image / vq_image item 加 `boi` / `eoi` marker；audio encoder 给 audio item 加 `audio_bos` / `audio_eos` marker；video 同理。**没有 chat_template 这个独立 module、没有顶层 chat template 工具层、没有顶层 image_pattern 注册表**。每个 family 的 chat template 写在自己的 `modules/<family>/text_encoder/modeling.py` 里，互不干扰。两次 input_ids 长度变化（text_encoder 输出"每张 image 1 个 placeholder"序列 → backbone splice 扩展成 N patch tokens）的同步 labels / mask / position_ids 对齐由 backbone 在 splice 时一次性处理（参见"Backbone pre_forward 完成多模态 splice + 长度对齐"和"Position IDs"两节）。

19. **RL 一致性**：训练 node 的 `forward()` 和推理 node 的 `generate_step()` 共用同一底层模型实现，log-prob 直接从 logits 提取，无两套实现分叉。一个 module type 一个 instance；RL 场景的 reference model 和 actor model 是两个独立 instance（model_type 可以相同）。

20. **FSM 转移完全由模块驱动，state 无步数预算**：state body 跑一次后持续循环，直到某条转移触发——"跑多少步、何时结束 state" 由模块决定，不由 YAML 的步数预算控制（框架已无 `token_length` 概念）。AR 循环靠 `module_signal`：模块在 return dict 写语义化一次性 flag（`image_complete`、`start_image_gen`、`text_done` …），YAML 用 `{type: module_signal, key: K}`，框架 pop key 防 stale；单趟 bridge / leaf state（prompt encode、emit `<boi>`/`<eoi>`）靠 `{type: default}`（catch-all 无条件匹配，body 跑一次即转移；与 `module_signal` 并列时必须排在最后做 fallback，否则框架报错）。框架只有 `module_signal` 与 `default` 两种 condition，不在 YAML 硬编码 vocab id。

21. **配置拆分：base launcher + 拆分的 module / graph 文件**：`base.yaml` 管 `model.model.model_path` / `model.model.model_config.modules` / `model.model.model_config.train_graph` / `model.accelerator` / 训练超参 / `infer` 块；`modules_train.yaml` 管每模块训练覆盖，`graph_train.yaml` 本身就是 training DAG 的 edge 列表；`modules_infer.yaml`（可选）覆盖推理模块（按模块名 deep-merge，默认 eager），`graph_infer_*.yaml` 本身就是一张 generation FSM。graph 文件顶层不再包 `training_graph:` / `generation_graph:` —— 文件即 graph（checkpoint sidecar 例外，它要用 key 承载多场景 map）。加载走 `OmniArguments.resolve_model()`（底层 `resolve_omni_model`，见 `arguments/omni_arguments_types.py`），返回 **runtime config**（`OmniModelRuntimeArguments`）；需要 HF checkpoint 形态时再显式 `.to_hf_config()`。`infer_graph` 传整张场景 map，全部载入 `generation_graphs`，`infer_type` 选激活项。`visualize_omni_graph.py` 与 trainer 共用这条路径（且只需 build 一次 config 就能画出全部场景的 FSM；可视化只读图，直接用 runtime config 不做转换）。

---

## 设计笔记：含音频的视频（Qwen-Omni 风格 av-video）

> **状态：未实现（design-only）**。当前 SeedOmni V2 仅支持纯图像理解（见
> `example_models/qwen3vl.md`）。本节记录"含音频视频"的目标设计，供后续实现参考。

### Qwen-Omni 参考实现（transformers `qwen2_5_omni` / `qwen3_omni_moe`）

- **message 层**：用户只写一个 `video` turn，不单独写 audio item；音频由
  `process_mm_info(..., use_audio_in_video=True)` 从 mp4 抽出，单独塞进
  `audio` 列表。chat template 只渲染 `<|vision_bos|><|VIDEO|><|vision_eos|>`，
  `use_audio_in_video` 是 **processor 标志**而非模板 token。
- **token 层**：`processor.replace_multimodal_special_tokens` 把这一个 `<|VIDEO|>`
  块**按时间轴交错**展开为
  `<|vision_bos|><|audio_bos|> …(交错的 VIDEO/AUDIO 占位)… <|audio_eos|><|vision_eos|>`。
  - Qwen2.5-Omni：按固定时间块（`seconds_per_chunk`，默认 2s）交错。
  - Qwen3-Omni：按 per-token 时间戳归并排序（谁的下一个 token 时间早就先放谁）。
  - 时间索引由 `video_second_per_grid` / `position_id_per_seconds` 算出，video/audio
    落在同一时间轴上——这也是 **TMRoPE**（时间对齐 multimodal RoPE）的依据。
- **拆分到两个 encoder**：拆分发生在 processor，从同一个 mp4 产出两路张量——
  video（`pixel_values_videos` + `video_grid_thw`）与 audio（`input_features` +
  `feature_attention_mask`）。模型里两个 encoder 各自独立编码，交错的占位 token 由
  `masked_scatter` 分别回填（VIDEO 槽 ← video embeds，AUDIO 槽 ← audio embeds）；
  因占位已按时间排好，scatter 后即恢复时间对齐。

### SeedOmni V2 目标设计（决定方案）

- **载体**：一个 `conversation_list` item，`type="video"`，`value = video_inputs`，
  `meta["audio_stream"]` 可选（存在 ⇒ 该片段带声音）；编码器还需把时间轴元数据
  （video 的 `second_per_grid`/fps、audio 帧率）写进 `meta`，否则 backbone 无法
  算交错顺序与 TMRoPE。
- **text_encoder（只管布局）**：根据 `meta["audio_stream"]` 是否存在决定外包裹——
  纯视频 `<|vision_bos|> … <|vision_eos|>`；带音频
  `<|vision_bos|><|audio_bos|> … <|audio_eos|><|vision_eos|>`（内层 audio、外层
  vision）。**不做交错**。沿用 janus 式压缩 loss（av item 在 decode 只贡献 1 行
  `-100`），从而 text_encoder 无需预展开精确的 video/audio 占位数。
- **video module**：`item.value`（帧）→ video embeds。
- **audio module**（全新模态，Whisper 式 mel 特征 + audio encoder）：
  `item.meta["audio_stream"]` → audio embeds。
- **llm backbone**：拥有时间交错——把该 item 的 video embeds + audio embeds 按时间戳
  归并织入扁平 `inputs_embeds`，并在同一趟里构建对齐的 **TMRoPE** position ids；
  av span 的总长度与 labels 也由 backbone 负责。

设计要点：数据保持 model-agnostic（一个 media item），两个 encoder 都是纯 embed
provider，唯二耦合的两件事（交错顺序 + 时间对齐 position）集中在已经负责 splice +
position 的 backbone 一处。
