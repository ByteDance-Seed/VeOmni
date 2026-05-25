# SeedOmni V2 架构设计

> SeedOmni V2 (`veomni/models/seed_omni/`) 重写——把固定的 `Encoder → Foundation → Decoder` 三元结构换成**显式图（nodes + edges）声明**的模块化系统。`OmniModule` 是 mixin，模型本体来自 transformers / diffusers，通过多继承获得框架钩子。两个平级的池子描述全图：`nodes:` 池每个 entry 是一次 `module.method` 调用（不指定 method 时训练用 `forward`、推理用 `generate_step`），`edges:` 池每条边路由 `from → to` 的输出。同一 module 可挂多个 node。每个 node 必有出边——指向另一个 node 或保留关键字 `end`（虚拟终点），保证图无孤岛、无环。训练子集只列 `edges`（nodes 由 endpoints 自动并出，执行序由 topo sort 推导，可视化时画出 forward queue）；推理由 FSM 驱动，每个 state 的 `body` 也只列 edges，可无限循环（text→image→text→image→...）。**数据完全 model-agnostic**：raw_batch 起点只有 `conversation_list`（list of dict，含 type / value / loss_mask / from_assistant），chat template / tokenize / image processor / boundary marker 注入全部由对应 module 在 forward 阶段自管——同一份数据可同时喂给任意 ug 模型；每个 module 的 `forward(**kwargs) -> Dict` 返回 dict 被框架按 edge.output 立刻写回 raw_batch（data 100% 走 raw_batch、module 之间不互相返回值）；collator helper / SP slice 由各 module 自己在 pre_forward 中按需调用（ViT 切 image batch、text encoder 切 sequence，各管各的）。loss 按 `_loss` 后缀隐式收集——每个 module 一次 forward 内部把所有 micro-batch 跑完，`post_forward` 自己做 token-level mean，OmniModel 顶层只把各 module 的标量 `_loss` 加起来。并行采用全局单一 `ParallelState`，OmniModel 顶层单次 `build_parallelize_model` 包装，`ParallelPlan` 由子模块递归聚合。生命周期上 weights 走 `build_foundation_model` + `build_parallelize_model`（多模块 weights_path 是 dict）、save 由每个 module 自己的 `CheckpointCallback` 写到自己的 subfolder（自带 config + 可选 asset，**所有 processor 和 tokenizer 都跟随 module subfolder**——OmniConfig 顶层不再有全局 `tokenizer_path` 字段）。多模态拼接（text embedding + image feature 等）由 AR backbone 在 `pre_forward` 内部完成。**不保留 V1 兼容**。

## 总纲（不变量）

1. **`module` ≠ `node` ≠ `edge`**：实例 / 调用 / 数据流，三层各司其职。
2. **一个 module instance 可挂任意多个 node**；同 method 也可承担多个角色（按 kwargs 自分派）。
3. **训练 = DAG（一次拓扑遍历），推理 = FSM（含环、按状态转移循环）**。
4. **永远不自动推导"图结构本身"**：`edges` 必须 config 显式给出。但**执行顺序可由 topo sort 从 edges 推导**——可视化训练图时画出 forward queue；FSM 因含环不可推导执行序，只可视化状态转移图。

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
| 完全模块化 | 所有组件以 OmniModule mixin 形态平等存在；只改 YAML（path / nodes / edges）即可替换任意模块 |
| 支持 AR + DiT | 同一训练框架内同时支持自回归和扩散两种生成范式 |
| 并行可组合 | 全局单一 `ParallelState`；OmniModel 顶层一次 `build_parallelize_model`；ParallelPlan 由各子模块的 `get_parallel_plan()` 递归聚合（**不**做异构 mesh / per-module FSDP wrap） |
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

### 核心思路：nodes（call-site）+ edges（数据依赖）+ end（虚拟终点）

去掉 encoder / foundation / decoder 的固定角色，用两个平级的池子 + 一个保留关键字描述整张图：

- **`nodes:`** 图节点池——每个 entry 是一个 call-site，对应一次 `module.method` 调用。同一 module 可以挂多个 node（如 VAE 的 `encode` 与 `decode`，共享一份参数但是图上两个独立节点）。不指定 `method` 时**训练默认 `forward`、推理默认 `generate_step`**。
- **`edges:`** 图边池——每条边把上游 node 输出 dict 里的某个 key 路由到下游 node 的某个 kwarg：`{from: A, output: k, to: B, as: m}`。
- **`end`：保留关键字**——所有 sink（如 `*_loss` 产出位）必须有一条 `to: end` 的边。**任何 node 至少有一条出边**，无孤岛；自环 / 任何环严格禁止（自环= for-loop，应在模块内部实现）。

**nodes 与 edges 是独立命名空间**：FSM body 只查 edges 池、edges 的 `from`/`to` 只查 nodes 池（外加 `end` 关键字），名字可以重名互不冲突。

激活子集 `training_graph` 只列 `edges`，nodes 由 edge endpoints 自动并出；执行顺序由 edges 拓扑序自动推导（**这是唯一的"自动"**，结构本身仍要显式给出）。`generation_graph.states.<name>.body` 同样只列 edges。

```
modules pool             nodes pool                              edges pool
─────────────────────    ───────────────────────────────         ────────────────────────────────────────────────
janus_siglip      ──→    vit_encode   → siglip.forward           vit_to_llama:        vit_encode → janus_llama
janus_vqvae       ──→    vae_encode   → vqvae.encode             vae_enc_to_llama:    vae_encode → janus_llama
janus_wte_lm_head ──→    vae_decode   → vqvae.decode             tok_enc_to_llama:    tok_encode → janus_llama
janus_llama       ──→    tok_encode   → wte_lm.encode            llama_to_tok_decode: janus_llama → tok_decode
                         tok_decode   → wte_lm.decode            llama_to_vae_decode: janus_llama → vae_decode
                         janus_llama  → ar_llm.forward           tok_decode_to_end:   tok_decode → end  (lm_loss)
                                                                 vae_decode_to_end:   vae_decode → end  (gen_loss)
                                                                 ↑ to: end 是 sink 锚（拓扑标记）；loss 仍按 _loss 后缀收集
```

---

## 核心抽象

### 1. `OmniModule`：mixin 形式的钩子集

`OmniModule` 不是独立的基类，而是**多继承 mixin**——模型本体来自 transformers / diffusers，再多继承 `OmniModule` 拿到框架钩子。所有钩子都是**可选**的，按模块自身需要决定实现哪些。

```python
class OmniModule:
    """Mixin for HF / diffusers models. All hooks optional."""

    # ── 训练入口（通常继承自 HF 基类）─────────────────────────────
    def forward(self, **kwargs) -> Dict[str, Any]:
        """
        训练 node 默认入口。kwargs = 当前 raw_batch 中本 module 声明
        要消费的 key 集合（由出入边的 ``as`` 字段决定具体名字）。返回
        命名 dict——OmniModel 框架**立刻把返回 dict 按 edge.output 写
        回 raw_batch**（不通过 edge 通道传给下游 module）。下游从同一
        raw_batch 按自己的 input keys 取。``*_loss`` 后缀的标量被
        OmniModel.forward 自动收集求和，不写回 raw_batch。

        如果 module 想**修改** raw_batch 的某个已有字段（比如 ViT 想改
        ``conversation_list`` 加 boundary marker、text_embed 想从
        conversation_list 产出 ``input_ids``），就在返回 dict 里写同名
        key，框架会覆写 raw_batch[<key>]。同一 sample 内多个 module
        写同一 key 的语义由 graph 顺序决定（后写覆盖先写）。

        多 method 模块通常让 forward = 主 method（如 encode）。
        """

    # ── 任意命名 method（YAML `method:` 引用）─────────────────────
    # 同 method 可承担多角色，靠 kwargs 自分派：
    # - VQ head decode：吃 (hidden+gt) → gen_loss；吃 hidden → 采 vq_token_id + embed；吃 token_id → codebook lookup
    # - text head decode：吃 (hidden+labels) → lm_loss；吃 hidden → 采 next token
    # tied weights 时 encode/decode 共享 `embed_tokens.weight`。

    # ── FSM 推理入口（可选）────────────────────────────────────────
    def generate_step(self, **kwargs) -> Dict[str, Any]:
        """
        FSM 推理时 method 为默认（`forward`）的 node 自动改走 generate_step。
        AR backbone 在此做一次 next-token 采样（KV cache 由模块自管）；
        DiT 类模块可在此跑完整去噪循环；其他显式 method（如 `decode`）
        由 FSM 按名直调，训推共用同一实现。
        """

    # ── 数据钩子（可选）────────────────────────────────────────────
    def pre_forward(self, **kwargs) -> Dict[str, Any]:
        """缺输入时构造 dummy（保持计算图一致，避免 FSDP bwd hang）；SP slice 等。"""

    def post_forward(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """SP gather；token-level loss scale（按 sum + token_count 形式吐出）。"""

    # ── tokenizer 共享（可选；少数场景才用）────────────────────────
    def set_tokenizer(self, tokenizer: "PreTrainedTokenizer") -> None:
        """
        Optional. tokenizer 在 V2 是 per-module asset（住在 text_embed module
        的 subdir）；大多数情况下每个 module 自带 / 不需要 tokenizer。这个钩子
        仅用于**少数需要共享 tokenizer 引用**的场景：
          - 一个 ViT 模块需要懂 boi/eoi token 字符串，但不想自己再带一份
            tokenizer，可在 OmniModel.__init__ 阶段通过本钩子拿到 text_embed
            的 tokenizer 引用，仅用于 ``convert_tokens_to_ids`` 之类的查询；
          - CFG ``build_cfg_uncond_inputs`` 取 ``pad_token_id`` 时，backbone
            可通过本钩子借用 text_embed 的 tokenizer。
        OmniModel 在 init 后由 text_embed 模块（如果存在）发起一次广播：把
        text_embed.tokenizer 注入给所有实现了 ``set_tokenizer`` 的 sibling
        module。默认 no-op。
        """

    # ── 并行（可选）────────────────────────────────────────────────
    def get_parallel_plan(self) -> Optional[ParallelPlan]:
        """
        本模块的 ParallelPlan，fqn 用**模块本地命名**（不带任何前缀）。
        OmniModel 在聚合时调 ``ParallelPlan.update_prefix(name)`` 加上
        ``<name>.`` 前缀，得到整模 plan 后传给顶层一次性的
        ``build_parallelize_model``。本方法只负责声明 EP / embed 等
        ExtraParallel 切分；SP gather/scatter 走 ``pre_forward``/``post_forward``，
        FSDP 由顶层单次 wrap 统一应用。
        """
```

模块**实际继承形态**：

```python
class JanusLlama(LlamaModel, OmniModule):
    """Janus 的 LLM backbone：LlamaModel + OmniModule 合体。
    forward 沿用 LlamaModel.forward；generate_step / post_forward 自定义。"""

class JanusVQDecoder(PreTrainedModel, OmniModule):
    """VQ codec：自定义 encode / decode 两个 method；forward = encode。"""

class TextEmbed(nn.Module, OmniModule):
    """generic wte + lm_head：encode / decode 对称；forward = encode。"""
```

- `model_type` 字段写在该模型的 `configuration_xxx.py`（HF 风格 `PretrainedConfig.model_type`），**不写在 YAML**——YAML modules 池只给 `weights_path` / `config_path`，`model_type` 由 HF AutoConfig 自动从 `<path>/config.json` 读出，再到模块注册表里找对应合体类。
- 没有 `_no_split_modules`、`config_class` 这类强约束——按 HF 原生写就行，FSDP 必要的 `_no_split_modules` 用 HF 基类自带的或在合体类里覆盖。

### 2. `OmniConfig`：modules + nodes + edges + 训练子集 + 推理状态机

顶层 5 个 section 各司其职（V2 不再有顶层 `tokenizer_path`）：

| Section | 职责 |
|---|---|
| `modules` | 模型实例池：name → `{weights_path / config_path, ...特化字段}`。**不写 model_type**（自动从 HF config.json 读）。每个 module 自带需要的 asset（tokenizer / image processor / feature extractor）跟随其 weights subdir |
| `nodes` | 图节点池：name → `{module: <name>(.<method>)?}`，声明一次 `module.method` 调用 |
| `edges` | 图边池：name → `{from, output, to, as}`，声明一条数据依赖（`to: end` 表示 sink） |
| `training_graph` | 只列 `edges:` 子集；`TrainingGraph` 据 endpoints 自动并出 nodes、按 topo 排序（DAG 视图） |
| `generation_graph` | 推理 FSM；`states.<name>.body` 也只列 edges 子集（FSM 视图） |

同一个 module 可以挂多个 node，每次以不同 method 被调用，但**模型实例不拆分**——`janus_vqvae.encode` 和 `janus_vqvae.decode` 是图上两个独立节点，共享一个 `JanusVQDecoder` 实例；同一个 method 也可以承担训练 + 推理两条 input pathway，靠 kwargs 自分派（`vae_decode` 是这种统一 head 的典型例子）。

> **`text_embed`：model-specific 的 chat-template + tokenizer + wte + lm_head 模块。** 这一层是 V2 数据流的核心枢纽：
> - **自带 tokenizer**（住在 `modules/<family>/text_embed/tokenizer/` 子目录）；
> - 在 `forward.encode` 中把 raw `conversation_list` 拼接成 `input_ids` / `inputs_embeds` / `labels` / `attention_mask`（含 chat template / system prompt / EOS / boi-eoi marker token）；
> - 在 `forward.decode` 中把 hidden_states 投影回 vocab（lm_head；tied weights 时 encode/decode 共享同一份矩阵）。
>
> 跟 V1 的"通用 wte + lm_head"不同，V2 的 `text_embed` 是 **model-specific**——每个 family 一份 `modules/<family>/text_embed/`，因为 chat template 拼接逻辑、特殊 token 集合、is system prompt 处理都是 family 相关的。`scripts/split_<family>.py` 会自动把 HF model 的 `embed_tokens` + `lm_head` + `tokenizer` 一并拆到 `text_embed/` 子目录。

```yaml
# ── 模块注册表（不写 model_type，HF AutoConfig 自动读）──────────────
# tokenizer 不在顶层，跟随 text_embed module 的 weights subdir
modules:
  janus_siglip:      {weights_path: /path/to/janus_siglip}
  janus_vqvae:       {weights_path: /path/to/janus_vqvae, freeze: true}
  janus_llama:       {weights_path: /path/to/janus_llama}        # 纯 backbone，无 vocab 层
  janus_wte_lm_head: {weights_path: /path/to/janus_wte_lm_head}

# ── 图节点池（每个 entry 是一次 module.method 调用）─────────────────
#
# {module: X}              → 训练用 X.forward / 推理用 X.generate_step
# {module: X.method}       → 训推都用 X.method（dotted 简写）
# {module: X, method: m}   → 同上（等价展开）
nodes:
  vit_encode:  {module: janus_siglip}                      # 默认：训练 forward / 推理 generate_step
  vae_encode:  {module: janus_vqvae,       method: encode} # pixels → gen_embeds + vq_token_ids
  vae_decode:  {module: janus_vqvae,       method: decode} # 统一 VQ head（见下方说明）
  tok_encode:  {module: janus_wte_lm_head, method: encode} # input_ids → inputs_embeds
  tok_decode:  {module: janus_wte_lm_head, method: decode} # 统一 text head（见下方说明）
  janus_llama: {module: janus_llama}                       # 纯 backbone

# ── 图边池（数据依赖；to: end 表示 sink）────────────────────────────
#
# {from: A, output: k, to: B, as: m}
#   把 node A 输出 dict 里的 k 路由成 node B 的 kwarg m
# {from: A, output: k, to: end}
#   sink 边——保证图无孤岛；loss 数值仍由 *_loss 后缀隐式收集
#   （edge 到 end 是拓扑标记，不携带数据）
edges:
  # ── 训练数据边
  vit_to_llama:       {from: vit_encode,  output: image_embeds,  to: janus_llama, as: und_image_embeds}
  vae_enc_to_llama:   {from: vae_encode,  output: gen_embeds,    to: janus_llama, as: gen_image_embeds}
  tok_enc_to_llama:   {from: tok_encode,  output: inputs_embeds, to: janus_llama, as: inputs_embeds}
  llama_to_tok_decode:{from: janus_llama, output: hidden_states, to: tok_decode,  as: hidden_states}
  llama_to_vae_decode:{from: janus_llama, output: hidden_states, to: vae_decode,  as: hidden_states}
  vae_tok_to_decode:  {from: vae_encode,  output: vq_token_ids,  to: vae_decode,  as: gt_token_ids}

  # ── 训练 sink 边（to: end，保证无孤岛；loss 仍按 _loss 后缀收集）
  tok_decode_to_end:  {from: tok_decode,  output: lm_loss,       to: end}
  vae_decode_to_end:  {from: vae_decode,  output: gen_loss,      to: end}

  # ── 推理反馈边
  vae_decode_to_llama:{from: vae_decode,  output: embed,         to: janus_llama, as: inputs_embeds}
  tok_decode_to_input:{from: tok_decode,  output: input_ids,     to: tok_encode,  as: input_ids}

# ── 训练 DAG（只列 edges；nodes 由 endpoints 自动并出，topo 推执行序）
training_graph:
  edges:
    - vit_to_llama
    - vae_enc_to_llama
    - tok_enc_to_llama
    - llama_to_tok_decode
    - llama_to_vae_decode
    - vae_tok_to_decode
    - tok_decode_to_end
    - vae_decode_to_end

# ── 推理图（FSM）；state.body 也只列 edges
generation_graph:
  initial: text_ar
  states: { ... }                 # 见 "推理：生成图" 节
```

**只改 config 即可完成模块替换**：
- 把 `janus_llama.weights_path` 指向其他 backbone 的拆分目录 → 换了 backbone（`model_type` 自动从新 path 的 config.json 读）
- 把 `janus_siglip` 改成另一份 vision encoder ckpt → 换了 vision encoder
- 新增 `talker` 模块 + 对应 node/edges → 支持 Qwen-Omni 风格的双 LLM

### 3. `OmniModel`：两套执行语义

```python
class OmniModel(PreTrainedModel, GenerationMixin):
    modules_dict: nn.ModuleDict          # 模块实例（一份 module 一个 key）
    graph:        TrainingGraph          # 训练 DAG（节点 = call-site，边 = 数据依赖）
    fsm:          GenerationGraph        # 推理 FSM（基于同一对 nodes/edges 池）

    # ── 训练路径：node DAG 一次遍历 ──────────────────────────────────────
    def forward(self, **batch) -> OmniOutput:
        node_outputs = {}                  # 索引 = node 名
        losses = {}
        for n in self.graph.execution_order:            # 由 edges topo 推出的 node 序
            module_name = self.graph.module_of(n)
            method      = self.graph.method_of(n)       # 默认 forward
            module      = self.modules_dict[module_name]
            inputs      = self.graph.collect_inputs(n, node_outputs, batch)
            # 一次调用内部把本 step 的所有 micro-batch 跑完：模块 forward
            # 自己迭代 micro-batches → 累加 token-sum loss / 累加 token_count →
            # post_forward 里做一次 token-level mean，吐出标量 `*_loss`
            if method == "forward":
                outputs = module(**inputs)              # 走 FSDP 包装层
            else:
                outputs = getattr(_unwrap(module), method)(**inputs)  # 直调 raw module
            node_outputs[n] = outputs
            # _loss 后缀隐式收集；此时每个 _loss 已经是 mean 后的标量
            losses |= {f"{n}/{k}": v for k, v in outputs.items() if k.endswith("_loss")}
        # 顶层只把各 module 已 mean 的标量 loss 求和（无须再加权）
        total_loss = sum(losses.values()) if losses else None
        return OmniOutput(
            losses=losses,
            total_loss=total_loss,
            **{f"{n}_out": o for n, o in node_outputs.items()},
        )

    # ── 推理路径：状态机分发 ────────────────────────────────────────────
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self._fsm.step(input_ids, self.modules_dict, **kwargs)

    # ── ParallelPlan 递归聚合（供顶层 build_parallelize_model 使用）─────
    # 注意：本类把 parameters() / named_parameters() delegate 到
    # self.omni_modules，bypass 了 "omni_modules." 这一段，所以
    # self.named_parameters() 看到的 fqn 是 <name>.<rest>，前缀只加 <name>.
    def get_parallel_plan(self) -> ParallelPlan | None:
        merged: dict[str, dict[str, Shard]] = {}
        for name, mod in self.omni_modules.items():
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

训练时 teacher forcing，AR LLM 一次 forward 处理完整序列，整体是个 node DAG。每个 node 跑一遍：fan-in 边把多个上游输出合并进它的 kwargs，fan-out 让多个下游共享同一份输出。同一 module 可以挂多个 node，分别调不同 method。

```mermaid
flowchart TD
    subgraph raw_batch [raw batch — 所有 node 全局透明可见]
        R1[input_ids / labels]
        R2[und_image_patches]
        R3[gen_image_patches / gen_image_mask]
    end

    subgraph exec ["OmniModel.forward — training_graph.edges 拓扑一次遍历"]
        vit["vit_encode<br/><i>janus_siglip.forward</i><br/>← und_image_patches<br/>→ {image_embeds}"]
        vae_e["vae_encode<br/><i>janus_vqvae.encode</i><br/>← gen_image_patches<br/>→ {gen_embeds, vq_token_ids}"]
        tok_e["tok_encode<br/><i>janus_wte_lm_head.encode</i><br/>← input_ids<br/>→ {inputs_embeds}"]
        ar["janus_llama<br/><i>janus_llama.forward</i><br/>← inputs_embeds (tok_encode)<br/>← und_image_embeds (vit)<br/>← gen_image_embeds (vae_encode)<br/>→ {hidden_states}"]
        tok_d["tok_decode<br/><i>janus_wte_lm_head.decode</i><br/>← hidden_states (janus_llama)<br/>← labels (raw)<br/>→ {lm_loss}  scalar, post_forward 内 token-mean"]
        vae_d["vae_decode<br/><i>janus_vqvae.decode</i><br/>← hidden_states (janus_llama)<br/>← gt_token_ids (vae_encode)<br/>→ {gen_loss}  scalar, post_forward 内 token-mean"]
        endN((end))
    end

    raw_batch --> vit & vae_e & tok_e
    vit -->|"as: und_image_embeds"| ar
    vae_e -->|"as: gen_image_embeds"| ar
    tok_e -->|"as: inputs_embeds"| ar
    ar -->|"as: hidden_states"| tok_d
    ar -->|"as: hidden_states"| vae_d
    vae_e -->|"as: gt_token_ids"| vae_d
    tok_d -.->|"to: end (lm_loss)"| endN
    vae_d -.->|"to: end (gen_loss)"| endN
```

**Forward queue 由 topo sort 自动推导**——`scripts/visualize_omni_graph.py` 会基于 `training_graph.edges` 跑 Kahn topo sort，输出严格的执行序列：

```
forward queue:
  1. vit_encode    (no deps)
  2. vae_encode    (no deps)
  3. tok_encode    (no deps)
  4. janus_llama   (waits: vit_encode, vae_encode, tok_encode)
  5. tok_decode    (waits: janus_llama)
  6. vae_decode    (waits: janus_llama, vae_encode)
  → end            (sink)
```

无环要求保证 topo sort 可解；任何环（含自环）会在 `TrainingGraph` 构造时直接报错。

**关于 janus_vqvae 的双角色——靠两个 node 表达**：

- `vae_encode` (`janus_vqvae.encode`)：吃 `gen_image_patches`，吐 `gen_embeds` 喂给 `janus_llama`。同时把 `vq_token_ids` 通过 `vae_tok_to_decode` 边送给下游做 ground truth。
- `vae_decode` (`janus_vqvae.decode`)：**统一 VQ head**——同一个 node 同时承担训练 loss 和推理反馈：
  - 训练：吃 `janus_llama.hidden_states` 和 `vae_encode.vq_token_ids` → 吐标量 `gen_loss`（走 `generation_head` + CE，`post_forward` 内已做 token-level mean）
  - 推理：吃 `janus_llama` 采样的 `token_id` → 吐 `embed`（走 `generation_embeddings` + `aligner`）
  - 两条路径互不干扰，按 kwargs 分派——HF 风格的 "input present → run, absent → skip / dummy"

两个 node 共享同一个 `JanusVQDecoder` 实例（同一份参数），但**图论上是两个独立节点**，分别在 `janus_llama` 之前和之后执行——没有环、没有"同模块跑两次"的特殊处理，就是标准 DAG。

**端点边 (`to: end`) 与 loss 收集**：

- `to: end` 边是**拓扑标记**：保证图无孤岛、可视化时所有 sink 都汇入 end 节点，**不携带数据语义**。
- loss 仍由 `*_loss` **后缀**隐式收集：OmniModel 扫描每个 module 的输出 dict，把 `_loss` 后缀键（已经是 module 内部 token-level mean 后的标量）收齐求和。
- 因此即便某个 sink 边漏写了，只要模块输出有 `_loss` 后缀键就还会被收集——但**强烈建议每个 sink 都补一条 `to: end` 边**，保证拓扑完整，避免可视化丢节点。

**Dummy forward**：node 一旦进了 `training_graph.edges`，**必跑一遍 forward**——data 全 0 / dummy 也必须走完整张图，避免 FSDP backward hang。模块自己在 `pre_forward` / `forward` 里写 dummy 路径（输入为 None / 全 0 时构造形状一致的 dummy tensor、loss 标量为 0），保证计算图静态一致。

**训推一致性**：训练用 teacher forcing（ground truth VQ embeds 直接送入 `janus_llama`），推理用 `image_vq` body loop（`janus_llama` 采样 vq_token_id → 同一个 `vae_decode` node 走推理路径产 embed → 下一步 input）。训练和推理共用同一份参数、同一个 node、同一个 `decode` 方法，仅 kwargs 不同。

### Q：同一个模块在数据流上出现两次怎么办？

典型场景：一个统一的 `image_codec`，输入图像过它得到 embeds 喂给 LLM，LLM 输出再过它得到生成图像。直觉上 `image_codec` 是一个节点被调用两次，但这会让图带环。

**做法：声明两个 node，共享一个 module 实例。**

```yaml
modules:
  image_codec: {weights_path: /path/to/image_codec}
  ar_llm:      {weights_path: /path/to/ar_llm}

nodes:
  img_encode: {module: image_codec, method: encode}     # 第一次调用：raw image → embeds
  ar_llm:     {module: ar_llm}
  img_decode: {module: image_codec, method: decode}     # 第二次调用：hidden_states → image

edges:
  img_to_ar:    {from: img_encode, output: embeds, to: ar_llm, as: image_embeds}
  ar_to_img:    {from: ar_llm, output: hidden_states, to: img_decode, as: hidden_states}
  img_dec_end:  {from: img_decode, output: gen_loss, to: end}

training_graph:
  edges: [img_to_ar, ar_to_img, img_dec_end]
```

`OmniModel.modules_dict["image_codec"]` 只 init 一次、参数只一份；`module_of("img_encode") == module_of("img_decode") == "image_codec"`，两个 node 通过 module 名拿到的是**同一个 Python 对象**。反向传播时两次调用的梯度自动累加到同一份参数上，就是普通的 weight sharing，没有任何 magic。

为什么不允许"一个 node 跑多次"：那会让图带环，`from`/`to` 失去唯一性，loss key（`{node}/{loss_name}`）失去唯一性，拓扑排序退化成"输入到齐就跑"的数据流调度，且 torch.compile / FSDP2 都假设 sub-module 调用顺序在一次 forward 中静态可枚举。把它写成两个 node，等价于**显式静态展开**那次循环——表达力一样，YAML 多几行，换来全程纯 DAG。

至于"自回归推理时 image_codec 在每个 token step 都被调用"这种**时序上的重复**——交给 FSM：训练图保持静态 DAG，FSM 在每个 step 内执行一段 body 序列（也只是 edges），整段 body 由外层步数循环驱动，不会污染 DAG 的"每个 node 跑一次"语义。

---

## 推理：生成图（FSM 视图）

### 核心统一抽象

推理和训练的本质差异在于：训练时 edges 做**一次拓扑遍历**，推理时 state.body 做**N 步循环**。两者都**只列 edges**——edge 自带 from/to 节点信息，激活的 nodes 由 endpoints 自动并出。

**FSM 一步执行规则**：

- 按 `state.body` 列出的 edges 顺序遍历；遇到 `from` 节点首次时**执行该节点**（method 为默认时 → 调 `generate_step`；显式 method → 直调），把输出 dict 写入 ctx。
- 该 edge 把 `ctx[output]` 复制到 `ctx[as]` 作为下游节点的 kwarg。
- 同 step 内同一节点不重复执行——后续命中的 edge 只做路由。

典型形态：

- **单节点循环**：文本 AR（`tok_encode → janus_llama → tok_decode`，variable 步）；DiT（`dit` 循环 forward，1 步）
- **多节点串接 + 反馈循环**：VQ 图像生成（`janus_llama → vae_decode → 反馈回 janus_llama`，循环 576 步）

```
训练时：training_graph.edges → 拓扑排序 → 一次 forward 遍历
推理时：state.body (edges)   → 按序执行 (node 首次激活 / edge 路由) → 循环 N 步
```

### 状态机定义

```yaml
generation_graph:
  initial: text_ar

  states:

    # ── 文本生成：每步 tok_encode → janus_llama → tok_decode ────
    text_ar:
      body:
        - tok_enc_to_llama       # tok_encode 执行 → inputs_embeds 路由到 janus_llama
        - llama_to_tok_decode    # janus_llama 执行（generate_step）→ hidden_states 路由到 tok_decode
        - tok_decode_to_input    # tok_decode 执行 → next input_ids 写回，下一步从 raw_batch 续上
      token_length: {type: variable}
      transitions:
        - {condition: {type: token_match, token_id: 151859}, next_state: image_vq}
        - {condition: {type: token_match, token_id: 151866}, next_state: video_dit}

    # ── VQ 图像生成：每步 janus_llama → vae_decode → 反馈 ───────
    # janus_llama generate_step 输出 hidden_states 给 vae_decode；
    # vae_decode 在 generation_head 上采 vq_token_id 并查 codebook 出 embed；
    # 路由回 janus_llama 作为下一步 inputs_embeds
    image_vq:
      body:
        - llama_to_vae_decode    # janus_llama 执行 → hidden_states
        - vae_decode_to_llama    # vae_decode 执行 → embed 路由回 janus_llama
      token_length:
        type: fixed
        value: 256
        # type: from_request            # 从用户请求参数读取
        # field: image_token_count
        # type: from_generated_text     # 从已生成文本中解析
        # extractor: parse_image_tokens
      transitions:
        - {condition: {type: steps_complete}, next_state: text_ar}
        - {condition: {type: token_match, token_id: 151860}, next_state: text_ar}

    # ── DiT 图像/视频生成：每步只执行 dit ─────────────────────────
    # dit.generate_step() 内部跑完整去噪循环，一次调用即产出完整结果
    # （所以 token_length=1）
    video_dit:
      body:
        - dit_to_end             # dit 执行 → loss/产物到 end（推理时 to: end 是终结标记）
      token_length: {type: fixed, value: 1}
      transitions:
        - {condition: {type: steps_complete}, next_state: text_ar}
```

### Token Length 策略

| 策略 | 说明 | 典型场景 |
|------|------|----------|
| `variable` | 无上限，持续循环直到转移条件或 EOS | 文本 AR、audio stream |
| `fixed` | 静态固定步数 | VQ 图像 token（256/1024 等）；DiT 单次调用（value=1） |
| `from_request` | 从用户的推理请求参数读取 | 用户指定分辨率 → 对应 token 数 |
| `from_generated_text` | 调用模块内定义的解析函数，从已生成内容提取 | AR 先生成 `<image w=1024 h=1024>`，再据此决定 VQ token 数 |

(如果要解析得到生成图像大小，这个东西可能做成一个node，输出的size信息直接交给image decoder)

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
      1. 按 state.body 顺序遍历 edges：edge.from 首次命中时调 module.method
         （forward → generate_step），写 outputs 到 ctx；edge 把 ctx[output] → ctx[as]
      2. 检查所有转移条件（first-match）
      3. 若触发转移，更新 _current_state 并重置 _step_count
    """
    _current_state: str
    _step_count: int
    _token_budget: int | None      # fixed / from_request / from_text 解析结果缓存
```

### 状态机示意

```mermaid
stateDiagram-v2
    [*] --> text_ar : 开始推理

    text_ar : text_ar\nbody: tok_encode→janus_llama→tok_decode\ntoken_length: variable
    text_ar --> text_ar : 普通文本 token
    text_ar --> image_vq : token_match 151859
    text_ar --> video_dit : token_match 151866
    text_ar --> [*] : EOS

    image_vq : image_vq\nbody: janus_llama→vae_decode (反馈循环)\ntoken_length: fixed/from_request/from_text
    image_vq --> text_ar : steps_complete
    image_vq --> text_ar : token_match 151860

    video_dit : video_dit\nbody: dit\ntoken_length: fixed=1
    video_dit --> text_ar : steps_complete
```

---

## 配置示例：不同模型架构

### Seed-Omni（AR + VQ 图像生成）

两个模块各出现两次，共享一份参数：

* `janus_vqvae` 挂 `vae_encode`（teacher-forcing embeds + ground-truth tokens）和 `vae_decode`（**统一 VQ head**——训练算 `gen_loss`、推理 hidden→sample→embed）。
* `janus_wte_lm_head` 挂 `tok_encode`（input_ids → inputs_embeds，wte lookup）和 `tok_decode`（**统一 text head**——训练算 `lm_loss`、推理 hidden→sample→next token）。`tie_word_embeddings=true` 时 encode/decode 共用 `embed_tokens.weight`，否则各持一份矩阵。

`janus_llama` 自身不再持有 `wte` / `lm_head`——就是个纯 backbone（`inputs_embeds → hidden_states`）。

`scripts/split_janus.py` 把原始 Janus checkpoint 拆成 4 份：`janus_siglip/`、`janus_vqvae/`、`janus_wte_lm_head/`、`janus_llama/`，每份带 config.json（`model_type` 已写入）。YAML 里只填 `weights_path` 即可，model_type 自动从 config 读。

```yaml
# tokenizer 跟随 janus_text_embed module，不在顶层

modules:
  janus_siglip:      {weights_path: /path/to/janus_siglip}
  janus_vqvae:       {weights_path: /path/to/janus_vqvae, freeze: true}
  janus_wte_lm_head: {weights_path: /path/to/janus_wte_lm_head}
  janus_llama:       {weights_path: /path/to/janus_llama}              # 纯 backbone，无 vocab 层

nodes:
  vit_encode:  {module: janus_siglip}                            # 默认 forward / generate_step
  vae_encode:  {module: janus_vqvae,       method: encode}       # pixels → gen_embeds + vq_token_ids
  tok_encode:  {module: janus_wte_lm_head, method: encode}       # input_ids → inputs_embeds
  janus_llama: {module: janus_llama}                             # inputs_embeds → hidden_states
  tok_decode:  {module: janus_wte_lm_head, method: decode}       # 训练: +labels → lm_loss
                                                                 # 推理: hidden → sample → next id
  vae_decode:  {module: janus_vqvae,       method: decode}       # 训练: +gt → gen_loss
                                                                 # 推理: hidden → sample → vq_id + embed

edges:
  # ── 训练数据边
  vit_to_llama:        {from: vit_encode,  output: image_embeds,  to: janus_llama, as: und_image_embeds}
  vae_enc_to_llama:    {from: vae_encode,  output: gen_embeds,    to: janus_llama, as: gen_image_embeds}
  tok_enc_to_llama:    {from: tok_encode,  output: inputs_embeds, to: janus_llama, as: inputs_embeds}
  llama_to_tok_decode: {from: janus_llama, output: hidden_states, to: tok_decode,  as: hidden_states}
  llama_to_vae_decode: {from: janus_llama, output: hidden_states, to: vae_decode,  as: hidden_states}
  vae_tok_to_decode:   {from: vae_encode,  output: vq_token_ids,  to: vae_decode,  as: gt_token_ids}
  # ── 训练 sink 边（保证图无孤岛；loss 仍按 _loss 后缀收集）
  tok_decode_to_end:   {from: tok_decode,  output: lm_loss,       to: end}
  vae_decode_to_end:   {from: vae_decode,  output: gen_loss,      to: end}
  # ── 推理反馈边
  vae_decode_to_llama: {from: vae_decode,  output: embed,         to: janus_llama, as: inputs_embeds}
  tok_decode_to_input: {from: tok_decode,  output: input_ids,     to: tok_encode,  as: input_ids}

training_graph:
  edges:
    - vit_to_llama
    - vae_enc_to_llama
    - tok_enc_to_llama
    - llama_to_tok_decode
    - llama_to_vae_decode
    - vae_tok_to_decode
    - tok_decode_to_end
    - vae_decode_to_end

generation_graph:
  initial: text_ar
  states:
    text_ar:
      # 每步：tok_encode → janus_llama → tok_decode；
      # tok_decode 把 next id 写回 ctx，下一步从 raw_batch 续上。
      body: [tok_enc_to_llama, llama_to_tok_decode, tok_decode_to_input]
      token_length: {type: variable}
      transitions:
        - {condition: {type: token_match, token_id: 151859}, next_state: image_vq}
    image_vq:
      # vae_decode 在 generation_head 上采 vq_id 并查 codebook 出 embed；
      # janus_llama 失去 lm_head 后 VQ 采样回到正确的 VQ vocab（不再借 text vocab）。
      body: [llama_to_vae_decode, vae_decode_to_llama]
      token_length: {type: fixed, value: 256}
      transitions:
        - {condition: {type: steps_complete}, next_state: text_ar}
```

### Qwen-Omni（thinker + talker 双 LLM + 音频）

两个 LLM 各配一份 `wte_lm_head`（`tie_word_embeddings=true` 时 encode/decode 共用一矩阵；下面省略 thinker / talker 各自的 `tok_encode` / `tok_decode` 两节点和对应 edges，结构同 Seed-Omni）。

```yaml
# tokenizer 跟随 thinker_wte_lm 和 talker_wte_lm（如果两个 text encoder 用不同 vocab，各自带一份）

modules:
  qwen_vision:        {weights_path: /path/to/qwen_vision}
  qwen_audio:         {weights_path: /path/to/qwen_audio}
  thinker_wte_lm:     {weights_path: /path/to/thinker_wte_lm}            # thinker 的 wte + lm_head
  thinker_llm:        {weights_path: /path/to/thinker_llm}               # 纯 backbone
  talker_wte_lm:      {weights_path: /path/to/talker_wte_lm}             # talker 的 wte + lm_head
  talker_llm:         {weights_path: /path/to/talker_llm}                # 纯 backbone
  codec2wav:          {weights_path: /path/to/codec_decoder}

nodes:
  vision_encode: {module: qwen_vision}
  audio_encode:  {module: qwen_audio}
  thinker_llm:   {module: thinker_llm}
  talker_llm:    {module: talker_llm}
  # 每个 LLM 自己的 tok_encode / tok_decode 略（结构同 Seed-Omni）

edges:
  vision_to_thinker: {from: vision_encode, output: image_embeds, to: thinker_llm, as: vision_embeds}
  audio_to_thinker:  {from: audio_encode,  output: audio_embeds, to: thinker_llm, as: audio_embeds}
  thinker_to_talker: {from: thinker_llm,   output: hidden_states, to: talker_llm,  as: thinker_hidden_states}
  # tok_*/sink 略

training_graph:
  edges: [vision_to_thinker, audio_to_thinker, thinker_to_talker, ...]

generation_graph:
  initial: thinking
  states:
    thinking:
      body: [tok_enc_to_thinker, thinker_to_tok_dec, thinker_tok_dec_to_input]
      token_length: {type: variable}
      transitions:
        - {condition: {type: token_match, token_id: 151860}, next_state: speaking}
    speaking:
      body: [thinker_to_talker, talker_to_tok_dec, talker_tok_dec_to_input]
      token_length: {type: variable}
      transitions:
        - {condition: {type: token_match, token_id: 151861}, next_state: thinking}
        - {condition: {type: eos}, next_state: done}
```

`thinker_llm` 内部决定如何将 `vision_embeds`、`audio_embeds` merge 进 embedding；`talker_llm` 内部决定如何用 `thinker_hidden_states` 作为 cross-attention key。**与 vllm-omni 中 thinker2talker `custom_process_input_func` 对应，但移入模块内部。**

### BAGEL（AR + DiT 图像生成）

```yaml
# tokenizer 跟随 bagel_text_embed module

modules:
  bagel_siglip:    {weights_path: /path/to/bagel_siglip}
  bagel_wte_lm:    {weights_path: /path/to/bagel_wte_lm}
  bagel_llama:     {weights_path: /path/to/bagel_llama}        # 纯 backbone
  bagel_dit:       {weights_path: /path/to/bagel_dit}

nodes:
  vision_encode: {module: bagel_siglip}
  tok_encode:    {module: bagel_wte_lm, method: encode}
  bagel_llama:   {module: bagel_llama}
  tok_decode:    {module: bagel_wte_lm, method: decode}
  bagel_dit:     {module: bagel_dit}

edges:
  vit_to_llama:    {from: vision_encode, output: image_embeds,  to: bagel_llama, as: vision_embeds}
  tok_enc_to_llama:{from: tok_encode,    output: inputs_embeds, to: bagel_llama, as: inputs_embeds}
  llama_to_tok_d:  {from: bagel_llama,   output: hidden_states, to: tok_decode,  as: hidden_states}
  llama_to_dit:    {from: bagel_llama,   output: hidden_states, to: bagel_dit,   as: condition}
  tok_dec_to_end:  {from: tok_decode,    output: lm_loss,       to: end}
  dit_to_end:      {from: bagel_dit,     output: dit_loss,      to: end}

training_graph:
  edges: [vit_to_llama, tok_enc_to_llama, llama_to_tok_d, llama_to_dit, tok_dec_to_end, dit_to_end]

generation_graph:
  initial: text_ar
  states:
    text_ar:
      body: [tok_enc_to_llama, llama_to_tok_d, tok_dec_to_input]
      token_length: {type: variable}
      transitions:
        - {condition: {type: token_match, token_id: 151870}, next_state: image_dit}
    image_dit:
      body: [llama_to_dit, dit_to_end]
      token_length:
        type: from_generated_text      # AR 先生成 "<image w=1024 h=768>"，DiT 解析出尺寸
        extractor: parse_image_config
      transitions:
        - {condition: {type: steps_complete}, next_state: text_ar}
```

---

## 离线 Embedding：不是特殊模式，就是不同的 training_graph

V2 框架中不存在 `offline_embedding` / `offline_training` / `online_training` 三种特殊模式的概念。它们只是**三份不同的 `training_graph` 配置**加上**不同的数据集**：

| 场景 | training_graph.edges | 数据集 | 产出 |
|------|---------------------------|--------|------|
| 生成 embedding | 激活 condition 模块到 backbone 的 edges + sink edge 到 end | 原始数据 | trainer 收集模块输出，存盘 |
| 离线训练 DiT | 只列 `dit` 的 sink 边到 end | 预存的 embedding 数据 | loss |
| 在线训练（全图） | 全部 edges | 原始数据 | loss |

```yaml
# ── 场景 A：生成 condition embedding（只跑到 bagel_llama，保存 hidden_states）
training_graph:
  edges:
    - vit_to_llama        # 端点并出 {vision_encode, bagel_llama}
    - tok_enc_to_llama    # 并出 {tok_encode, bagel_llama}
    - llama_to_end        # sink：保证 bagel_llama 不是孤岛
# 数据集：原始图文数据
# trainer 逻辑：从 OmniModel.forward() 输出里读 bagel_llama_out['hidden_states']，写到 parquet

# ── 场景 B：用预存 embedding 训练 DiT
training_graph:
  edges: [dit_to_end]    # 端点并出 {bagel_dit}
# 数据集：上一步保存的 parquet（batch 里已带 condition 字段）
# bagel_dit.forward() 直接从 **kwargs 里取 raw_batch['condition']

# ── 场景 C：在线全图训练
training_graph:
  edges: [vit_to_llama, tok_enc_to_llama, llama_to_tok_d, llama_to_dit, tok_dec_to_end, dit_to_end]
# 数据集：原始图文数据
```

**关键**：raw batch 对所有 node 全局透明。在场景 B 中，预存的 `condition` 字段已经在 batch 里，`bagel_dit.forward()` 直接从 `**kwargs` 里取，不需要任何特殊注入逻辑。OmniModel 本身没有任何 mode 切换，模型行为完全由 `training_graph.edges` 决定。`OfflineEmbeddingSaver` 是 trainer 层的工具，模型不感知。

---

## 并行配置（单一 ParallelState + 递归 ParallelPlan）

本版本采用与 `BaseTrainer` 完全一致的并行入口：**全局一份 `ParallelState`**（FSDP / SP / EP / CP / TP 维度共享），**OmniModel 顶层单次** `build_parallelize_model` 包装。子模块**不**单独 wrap FSDP、**不**单独定义 mesh / dp_mode；它们只通过 `get_parallel_plan()` 贡献自己那部分的 ExtraParallel（EP / embed 等）切分声明，由 OmniModel 收集后递归合并成整模 plan。

| 层级 | 职责 |
|------|------|
| 全局 `ParallelState` | trainer 层一次 `init_parallel_state(...)`；所有子模块共享 fsdp_mesh / sp_group / ep_group |
| 每个 `OmniModule.forward()` | 内部自管 SP `gather/scatter`（数据进出 SP 区域），对外完全透明 |
| `OmniModel.forward()` | 不含任何 SP / FSDP / EP 操作，仅做图遍历 + loss 聚合 |
| `OmniModule.get_parallel_plan()` | 返回**模块本地** fqn 的 `ParallelPlan`（如 `layers.*.mlp.experts.gate_up_proj`），不带任何前缀 |
| `OmniModel.get_parallel_plan()` | 调 `plan.update_prefix(name)` 加 `<name>.` 前缀，把所有子模块 plan 合并成一份整模 plan |
| trainer | 一次 `build_parallelize_model(omni_model, ...)`：内部读 `omni_model.get_parallel_plan()` 应用 EP，再 `fully_shard()` 顶层 |

```python
class OmniModel(nn.Module):
    def get_parallel_plan(self) -> ParallelPlan | None:
        merged: dict[str, dict[str, Shard]] = {}
        for name, mod in self.omni_modules.items():
            plan = mod.get_parallel_plan() if hasattr(mod, "get_parallel_plan") else None
            if plan is None:
                continue
            plan.update_prefix(name)                     # 加 <name>. 前缀
            for para, sub_plan in plan.extra_parallel_plan.items():
                merged.setdefault(para, {}).update(sub_plan)
        return ParallelPlan(merged) if merged else None
```

### FQN 视角对齐（重要细节）

`OmniModel` 把 `parameters()` / `named_parameters()` delegate 到 `self.omni_modules`（bypass 了 `omni_modules.` 这一段），所以 `model.named_parameters()` 看到的 fqn 形如 `<module_name>.<rest>`。`ParallelPlan.apply` 按这一视角做 fqn 匹配，与 `update_prefix(name)` 加的 `<name>.` 前缀**对齐一致**。FSDP2 仍按真实 attribute 路径（`omni_modules.<name>.<...>`）找 child module，两套视角各走各的，互不干扰。

### 举例

- `janus_llama`（MoE）：模块自身 `get_parallel_plan()` 返回 `{"ep": {"layers.*.mlp.experts.gate_up_proj": Shard(0), ...}}`；OmniModel 加前缀后变成 `{"ep": {"janus_llama.layers.*.mlp.experts.gate_up_proj": Shard(0), ...}}`
- `janus_siglip`（VLM ViT）：`get_parallel_plan()` 返回 `None`（无 ExtraParallel）；SP 在自己的 `forward()` 里通过 `gather_seq_scatter_heads` 处理
- `bagel_dit`：当前版本不做 per-module 不同 mesh / TP；后续如有需要再扩展

### micro_batch_size 一致

跨 node 的 `micro_batch_size` **强制全局一致**——暂不考虑不同 node 用不同 micro batch 数的场景（实现复杂度高、收益有限）。模块特化字段 `micro_batch_size` 仍保留在 `modules:` 池里以备未来扩展，但当前版本由顶层统一传入。

### 本版本明确**不做**的

- ❌ 子模块各自 wrap FSDP（异构 dp_mode）
- ❌ 子模块持有自己的 `ParallelState` / mesh
- ❌ 子模块声明独立 SP / EP group（SP / EP group 全局唯一）
- ❌ DDP 路径（暂只支持 FSDP2）
- ❌ 跨 node 不同 micro_batch_size

### 与现有基础设施

`torch_parallelize.py` 的 `build_parallelize_model` 需扩展支持**多 weights_path**（见下一节）；`BaseTrainer._build_parallelized_model` 复用——`OmniTrainer` 走 `_build_model`（构造 raw OmniModel）→ `_build_parallelized_model`（顶层一次 wrap）的两步流程，与 `VLMTrainer` / `TextTrainer` 对齐。

---

## 生命周期

### Build & 权重加载

复用 trainer 现有两个组件函数（**需要扩展支持多模块**）：

| 组件 | 现状 | OmniModel 改动 |
|------|------|----------------|
| [`build_foundation_model`](veomni/trainer/base.py) (255-262) | 单一 `weights_path` → 单一 `nn.Module` | 接 `dict[str, str]`：`{module_name: path, ...}`，按 modules 池 init `ModuleDict` |
| [`build_parallelize_model`](veomni/trainer/base.py) (387-404) | 单一 `weights_path` 加载到 single model | 接多 path，按 module subdir 分别从 meta device 加载 |

**Meta device + 多 path 加载流程**：

1. 各 module 在 meta device 上按 HF AutoConfig + AutoModel 构造（自动从 `<weights_path>/config.json` 读 `model_type`）。
2. 用 module mixin 注册表 `MODULE_MIXIN_REGISTRY[hf_model_type]` 找到合体类（`type(name, (HFClass, mixin), {})` 或预先定义的合体类），把 mixin 钩子挂上。
3. ParallelPlan 应用、`fully_shard()` 顶层 wrap。
4. 按 module subdir 分别 `_load_state_dict_from_safetensors(<weights_path>/model.safetensors)` 到对应子树（FQN 前缀 `<module_name>.`）。

**Key convert**：`scripts/split_<family>.py` 拆分时只关心 family 内子模型，不知道用户在 YAML 里给这个子模型起什么 node 名。所以约定：
- 拆模型脚本输出固定的子目录命名（如 `janus_siglip/`、`janus_vqvae/`），子目录里 weights 用模块**本地** fqn 命名。
- 加载时按 YAML `modules.<name>.weights_path` 读取，state_dict 套上 `<name>.` 前缀就能放到 `omni_model.modules_dict.<name>` 子树。
- 用户在 YAML 里改 module 的 key（如把 `janus_llama` 改成 `my_backbone`），不影响加载——前缀由 YAML key 决定。

### Save：每个 module 自己的 callback

每个 module 在初始化时挂一个自己的 [`CheckpointCallback`](veomni/trainer/callbacks/checkpoint_callback.py) 实例。trainer 触发 save 时遍历所有 callback，各自写自己的 subfolder：

```
output_ckpt_dir/
├── janus_siglip/
│   ├── config.json                   # 含 model_type，HF save_pretrained 写
│   └── model.safetensors
├── janus_vqvae/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json      # 此模块的 asset（vision processor）
├── janus_llama/
│   ├── config.json                   # 无 asset（tokenizer 全局）
│   └── model.safetensors
├── janus_wte_lm_head/
│   ├── config.json                   # 无 asset（tokenizer 全局）
│   └── model.safetensors
├── tokenizer/                         # 全局，OmniModel 顶层 callback
│   ├── tokenizer.json
│   └── special_tokens_map.json
└── omni_config.yaml                   # 顶层 OmniConfig（modules / nodes / edges / graphs）
```

- "整体打包存"由顶层 callback 触发（不重复写每个 module）。
- config.json 由各 callback 顺带保存（HF 风格 `save_pretrained`）；`model_type` 字段自动随 config 落盘——下次加载直接 from_pretrained。
- 训练继续时 weights_path 直接指向各 subdir，无需再过拆模型脚本。

### Assets

| 类别 | 是否全局 | 存放位置 |
|------|---------|---------|
| **tokenizer** | **per-module**（住在 text_embed module 的 subdir） | 由 `text_embed` module 自带（如 `modules/janus/text_embed/tokenizer/`）。OmniConfig **顶层不再有** `tokenizer_path`。如果一个模型有多个 text encoder 或场景需要多 tokenizer，每个 text-side module 各自带一份。纯 DiT 模型（无 text encoder）配置里完全不出现 tokenizer，框架不强求。 |
| vision processor / image processor | per-module | 跟随该 module subdir（如 `janus_siglip/preprocessor_config.json`、`janus_vqvae/preprocessor_config.json`） |
| audio feature extractor | per-module | 跟随该 module subdir |
| 任何其他 processor（video frame extractor / chat template 拼接 helper / ...） | per-module | 跟随该 module subdir。chat template 不是独立 asset，它是 text_embed 模块的 forward 实现细节（用 Python 代码或 jinja 模板都可以）。 |

- **每个 module 0 或 1 个 asset，不会重复**——一个 module 只对应一种 processor / tokenizer。
- **tokenizer 不再是顶层全局**：raw data 进入 OmniModel 时只是 conversation_list（list[dict]），还没 tokenize；tokenize 发生在 text_embed.forward 内部（数据流 Layer 5）。这条契约让：
  - **emit 边界 token / chat template 拼接 / labels 计算 / 构造 CFG uncond 分支** 全部封进 text_embed 或对应模态的 encoder—— OmniModel 框架不感知任何 model-specific 的 token id；
  - **pure DiT 不需要 tokenizer**：纯 DiT 配置里完全不写 text_embed module，raw_batch 起点是空 conversation_list 或者纯 image conversation；
  - **多 text encoder** 场景（比如某个模型同时用 LLaMA tokenizer 和 T5 tokenizer 给两个 text 分支）：每个 text encoder 各自带自己的 tokenizer，互不干扰。
- **vocab-bound 模块（如 `janus_llama` backbone 等）本身没有 asset**——它们读 inputs_embeds / hidden_states，不直接读原始字节或 token id。
- **vision encoder 类模块（如 `janus_siglip`、`janus_vqvae`）的 image processor 跟随该模块**——避免顶层维护一个 processor 注册表。模块的 `forward` 直接从 conversation_list 取已 resize 的 image tensor 并调本模块自己的 processor 完成 patch / normalize。

### micro_batch_size、freeze、gradient_checkpointing 等模块特化字段

写在 `modules.<name>.<field>`，由各模块自己读取并应用。当前版本：
- `freeze: true` → 模块构造完后冻结所有参数（不参与训练）。
- `gradient_checkpointing: true` → 模块 init 后调 `gradient_checkpointing_enable()`。
- `micro_batch_size` → 字段保留但当前必须全局一致。

---

## 数据路由：raw_batch = `conversation_list` + module-driven processing

> **状态**：本节描述 V2 的**目标契约**。当前 `veomni/data/multimodal/multimodal_chat_template.py` 沿用 V1 的"chat-template 工具层 + N 倍预展开 + backbone scatter"形态。本节描述的"raw_batch 单字段 + module 全责处理"是后续按 feature 迁移的目标形态，按 feature 一项一项实施。

### 设计原则

V2 框架的数据流由两条核心契约定义：

1. **数据完全 model-agnostic**：raw_batch 里只有 `conversation_list` 这**唯一字段**——每条 sample 是一个 `list[dict]`，每个 item 仅含通用字段 `type` / `value` / `loss_mask` / `from_assistant`。**没有 input_ids、没有 pixel_values、没有 image_pos**。同一份 SFT 数据集可同时喂给 Janus / Qwen-Omni / Bagel 等任意 ug 模型，每个模型自己解析、自己 tokenize、自己处理 image —— **数据集和模型解耦**。

2. **module 通过 forward `return dict` 修改 raw_batch**（不是直接 mutate）：每个 module 的 forward 仍是 `forward(**kwargs) -> Dict[str, Any]` 风格（HF 兼容、单测纯函数）；OmniModel 框架收到返回 dict 后**立即按 edge.as 写回 raw_batch**（不通过 edge 通道传递给下游）。下游 module 从同一 raw_batch 按自己声明的 input keys 取。这等价于"data 100% 走 raw_batch、module 之间不互相返回值"，但保留了 kwargs 风格 API 和 edge 显式契约。

### Raw conversation item schema

```python
{
    "type":           "text" | "image" | "video" | "audio" | "vq_image"
                      | "boi" | "eoi"      # ← module forward 阶段插入的边界 marker
                      | "audio_bos" | "audio_eos" | ...,
    "value":          <str | tensor>,       # text: string；image/video/audio: 已 resize 的 tensor；
                                            # boundary marker (boi/eoi/...): None 或省略
    "loss_mask":      0 | 1,                # 是否参与 loss（assistant 段=1，其他段=0）
    "from_assistant": bool,                 # 由 assistant 产出（推理 ctx 也用这个字段）
}
```

`role: "user" | "assistant" | "system"` 字段被 `from_assistant: bool` 替代——后者在 forward 中更直接（labels 计算时只关心是否参与 loss，role 字符串语义由各 module 的 chat-template 拼接逻辑解读）。system prompt 通过约定（比如约定第一个 `from_assistant=False, loss_mask=0` 的 text item 就是 system prompt，或者由 dataset 配置注入）由 text encoder 自己识别。

例如"理解一张图 + 生成一段文 + 生成一张图"对话进入 raw_batch 时的形态：

```python
raw_batch["conversation_list"][0] = [   # 第 0 个 sample
    {"type": "text",     "value": "You are a helpful assistant.", "loss_mask": 0, "from_assistant": False},
    {"type": "text",     "value": "Describe this and draw similar:", "loss_mask": 0, "from_assistant": False},
    {"type": "image",    "value": <Tensor C×H×W>,                  "loss_mask": 0, "from_assistant": False},
    {"type": "text",     "value": "A cat on a sofa.",              "loss_mask": 1, "from_assistant": True},
    {"type": "vq_image", "value": <Tensor C×H×W>,                  "loss_mask": 1, "from_assistant": True},
]
```

注意 `image` / `vq_image` item 的 `value` 已经是 **resized tensor**（不是 path）—— resize 由 `multimodal_transform.py` 的减重版工具层在数据加载阶段完成（见下）。

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
│       新插入的 marker item 继承原 item 的 from_assistant/loss_mask        │
│    4. return dict 含 conversation_list (modified) + image_embeds + ...    │
│       框架按 edge.as 立即写回 raw_batch                                   │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ raw_batch 现含: conversation_list (含所有
                                │   boundary markers), und_image_embeds,
                                │   gen_image_embeds, audio_embeds, ...
                                ▼
┌─ Layer 5: text_embed module（model-specific，包揽 chat-template + wte）──┐
│  modules/<family>/text_embed/                                             │
│  Asset: tokenizer/（per-module，本模块自带；OmniConfig 顶层不再有         │
│         全局 tokenizer_path）                                             │
│  pre_forward: 调 collator helper 抽 batch                                 │
│  forward (一气呵成):                                                      │
│    1. 对每个 sample 的 conversation_list 按本 family 的 chat template     │
│       规则拼接：                                                          │
│       - system prompt 前缀 / user / assistant 角色 token                  │
│       - 每个 item 翻译为 token 序列：                                     │
│           type=text:        tokenizer.encode(item["value"])               │
│           type=boi/eoi/...: tokenizer.convert_tokens_to_ids("<boi>")      │
│           type=image/video/audio/vq_image: 1 个 placeholder token id      │
│             （供 backbone splice 时识别）                                 │
│       - 末尾加 EOS                                                        │
│    2. 算 labels（image/audio 段填 -100；text 段按 from_assistant 决定     │
│       loss_mask=1 时复用 input_id，否则填 -100）                          │
│    3. 算 attention_mask                                                   │
│    4. 过 wte → inputs_embeds                                              │
│    5. SP slice（input_ids / inputs_embeds / labels / attention_mask）     │
│  return: {input_ids, inputs_embeds, labels, attention_mask}               │
│                                                                            │
│  各 family 的 text_embed 实现差异完全包在这一层：                         │
│    Janus 的 text_embed 写 Janus 的 chat template 拼接逻辑；               │
│    Qwen-Omni 的写自己的；Bagel 的写自己的。框架不感知。                   │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ raw_batch 现含: conversation_list, image
                                │   embeds, vq embeds, input_ids,
                                │   inputs_embeds, labels, attention_mask
                                ▼
┌─ Layer 6: backbone（JanusLlama / QwenOmniThinker / ...）─────────────────┐
│  pre_forward:                                                             │
│    1. 多模态 splice（单 placeholder → N patch tokens；image_embeds 来自   │
│       Layer 4 的 ViT/VAE encoder，placeholder 来自 Layer 5 的 text_embed）│
│    2. 同步 splice labels (image 段 -100) / attention_mask (1) /           │
│       input_ids 概念                                                      │
│    3. compute_position_ids 从 splice 后的最终长度算 position_ids          │
│    4. SP pad_and_slice                                                    │
│  forward → hidden_states                                                  │
│  post_forward → SP gather                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 关键不变量

- **数据 100% model-agnostic**：raw_batch 起点只有 `conversation_list`，schema 通用。同一份数据可同时喂给任意 ug 模型。
- **chat-template / tokenize / image processor / audio feature extractor 全部下放给模型**：multimodal_transform.py 工具层只保留基础 IO + resize；不存在框架层级的 chat-template helper、不存在框架层级的 image_pattern 注册表。
- **每个 module 自管自己的 token 拼接**：text encoder（text_embed）拼 system prompt + 文本 item + 加 eos；ViT 在 conversation_list 中给 image item 加 boi/eoi；audio encoder 给 audio item 加 audio_bos/audio_eos；framework 不感知。
- **collator 在 module pre_forward 中按需调用，不再是 dataloader final-step**：每个 module 自己知道关心哪些字段、怎么 batch、SP 怎么切（ViT 切 image batch 维；text encoder 切 sequence 维）。
- **forward 阶段允许修改 input_ids 长度**：ViT 在 conversation_list 加 boundary marker → text_embed 在 input_ids 里产出对应 token id（每个 marker 1 个 token）→ backbone splice 把 image placeholder 扩展成 N patch tokens。两次长度变化，labels / attention_mask / position_ids 在每一步都同步对齐。
- **module forward = kwargs + Dict 返回**（W2 风格）：API 不变，但语义改成"返回 dict 立刻被框架按 edge.as 写回 raw_batch"，data 不通过 edge 通道传递。下游 module 从同一 raw_batch 按 input keys 取。
- **graph topology 自动从 edge dependency 推**：因为 ViT/VAE 修改 conversation_list、text_embed 读 conversation_list，topo 序自动 ViT/VAE → text_embed → backbone，**不需要显式顺序约束 edge**。

### 与 V1 主线的迁移路径（每条 feature 独立 PR）

1. **Feature D1**（基础）：multimodal_transform.py 减重——移除 chat_template + tokenize + image_processor 调用，只保留 IO + resize；输出 conversation_list 而非张量 batch。
2. **Feature D2**（基础）：dataloader / collator 减重——只 batch list；不做任何 sequence padding。
3. **Feature D3**（vision）：把 image processor + boundary marker 注入逻辑搬进 ViT/VAE 的 forward。
4. **Feature D4**（text）：把 chat template + tokenize 搬进 text_embed 的 forward；text_embed 升级为 model-specific（modules/<family>/text_embed/）。
5. **Feature D5**（backbone）：splice + compute_position_ids 在 backbone pre_forward 中接管最终长度对齐（这条之前讨论过）。

D1-D2 是数据层减重；D3-D5 是模型层接管。每步都向后兼容（中间状态可跑），但最终目标是上述六层架构。

### Backbone `pre_forward` 完成多模态 splice + 长度对齐（target contract）

> **状态**：本节同样描述目标契约。当前 `JanusLlama.pre_forward` 走的是 V1 兼容的 scatter 路径（input_ids 里已经有 N 个 image_pad placeholder，`masked_scatter` 替换 placeholder embedding 而不改长度，配合 `+ x.sum() * 0.0` 锚点保证 FSDP grad sync）。迁移到本节描述的 splice 形态是 Feature D5——前置依赖 Feature D4（text_embed 在 forward 中产出"每张 image 仅 1 个 placeholder"的 input_ids）。

text_embed（Layer 5）已经把 `input_ids` 全量过 wte 得到一份**每张 image 占 1 个 placeholder 槽位**的 text embedding 序列；vision / audio 等 encoder 各自吐出 embedding list（Layer 4 已写到 raw_batch）。**真正的拼接（splice）在 backbone（如 `janus_llama`）的 `pre_forward` 里完成**——这一步把每张 image / audio / video 的 1 个 placeholder 槽位**膨胀**成 N 个 patch token 槽位，相应地 inputs_embeds 长度从 `L_text + 1·num_images` 变成 `L_text + sum(N_i)`：

```python
class JanusLlama(LlamaModel, OmniModule):
    def pre_forward(
        self,
        input_ids,                           # placeholder 序列（每张图占 1 个 image_pad token）
        inputs_embeds,                       # 来自 tok_encode（每张图占 1 个槽位）
        und_image_embeds=None,               # 来自 vit_encode（list[Tensor]，每张图 N_i 个 patch token）
        gen_image_embeds=None,               # 来自 vae_encode
        und_image_pos=None, gen_image_pos=None,    # 每张图 placeholder 在 input_ids 里的 token_idx
        attention_mask=None, labels=None, position_ids=None,
        **_,
    ):
        # splice：把每个 single placeholder 槽位替换成 N_i 个 patch token embedding
        #
        # 此处 inputs_embeds / input_ids / attention_mask / labels 必须**同步**做 splice
        # 扩展（image 段 labels=-100、attention_mask=1），否则 sequence-domain 对齐会
        # 错位。position_ids 在 splice 后由 backbone 自己的 compute_position_ids 重新
        # 算（M-RoPE 类模型需要 image grid 才能算位置；1D RoPE 直接 arange 新长度）。
        inputs_embeds, attention_mask, labels = splice_multimodal(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            embeds_per_modality={
                "und_image": (und_image_embeds, und_image_pos),
                "gen_image": (gen_image_embeds, gen_image_pos),
            },
        )
        position_ids = self.compute_position_ids(
            input_ids=input_ids, attention_mask=attention_mask, image_grid_thw=...
        )["position_ids"]
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
        }
```

这样：
- **chat template 不在 HF tokenizer 内部**——是 `text_embed` module 的 forward 实现细节；HF tokenizer（住在 `text_embed` 内部）只懂 string → token id。
- **多模态拼接没有全局路由表**——每个 backbone 自己决定如何 splice（不同 backbone 可能 cross-attn 而非 splice，比如 DiT 风格用 cross-attn 消费 text_embeds，根本不做 splice）。
- **HF tokenizer / text_embed 都不感知 image patch 数**——text_embed 在 input_ids 里每张 image 只放 1 个 placeholder token（不需要懂 N）；展开 N 倍由 backbone 在 splice 阶段完成，依赖 image_embeds 的实际形状（来自 image processor，是 vision encoder 模块的私有 asset）。
- **labels / attention_mask / position_ids 同步在 splice 阶段对齐**——image 段 labels 填 -100，attention_mask 填 1，position_ids 由 backbone 重新算（参见"Position IDs"一节）。
- **模态新增 = 加一个 module + 一条 edge**——`pre_forward` 在 `embeds_per_modality` 里多接一种模态、多写一段 splice 即可。

### Position IDs：backbone 私有 schema，splice 后由 backbone 重算

> **状态**：本节描述目标契约。当前 `JanusLlama.pre_forward` 接受外部传入的 `position_ids`（沿用 V1 主线 `multimodal_transform.py` 在数据预处理阶段调 `position_id_func` 算好的形态）；本节描述 V2 把这一步迁移到 backbone 内部 `compute_position_ids` 钩子。

position_ids 计算的输入依赖 image / video 的 patch grid（M-RoPE 类模型给 image 内部分配 2D `(h_idx, w_idx)`，给 video 分配 3D `(t_idx, h_idx, w_idx)`），所以**必然**要在 image_embeds 已经 splice 进 inputs_embeds 之后才能算 —— 因为 splice 本身就是 image 占多少 token、occupy 哪些位置的最终 source of truth。

`OmniModule` 提供一个可选钩子，每个 backbone module 按需 override：

```python
class OmniModule:
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
| `vit_encode` / `vae_encode` 等 **vision encoder** | (1) 从 conversation_list 抽 image / vq_image item.value（已是 resized tensor）→ stack 成 patch batch tensor；(2) 用本模块自带的 image processor 跑 patch / normalize；(3) encoder forward → image_embeds；(4) **修改 conversation_list**：在每个 image / vq_image item 前后插 `{type: "boi"}` / `{type: "eoi"}` item（继承原 from_assistant / loss_mask）；(5) 按需 SP slice 自己的字段（image batch / patch 维），不动 sequence 维 |
| `audio_encode` | 同上但模态是 audio：抽 audio item.value → feature extractor → encoder → audio_embeds；在 conversation_list 中给 audio item 加 `audio_bos` / `audio_eos` marker |
| **text_embed**（model-specific，住 `modules/<family>/text_embed/`）| (1) 自带 tokenizer asset；(2) 接受已经被 vision/audio module 修改过的 conversation_list；(3) 按本 family 的 chat template 规则拼接 token_id 序列（含 system prompt / EOS / boi-eoi 等 marker token）；(4) 算 labels（image/audio 段填 -100，text 段按 from_assistant + loss_mask）；(5) 算 attention_mask；(6) 过 wte → inputs_embeds；(7) SP slice sequence-domain tensors |
| `<backbone>`（`janus_llama` / `qwen_omni_thinker` / ...）| **多模态 splice**（单 placeholder → N patch tokens）+ labels / mask / position_ids 同步对齐 + `compute_position_ids` 重算 + dummy fill + 最终一次 SP pad_and_slice |
| `tok_decode` / `vae_decode` | 直接读上游 hidden_states，跑 head + sample / 算 loss；SP-agnostic（backbone post_forward 已 gather） |

注意几点：
- **没有 chat_template 这个独立 module**：chat template 拼接逻辑住在 text_embed（每个 family 一份）；boundary marker 注入由对应模态的 encoder 负责。
- **没有 tok_encode 这个独立 node**：text_embed.forward 一气呵成 chat-template + tokenize + wte，输出 `inputs_embeds`。如果某些场景需要把"text_embed.encode = chat-template + tokenize"和"text_embed.wte = embed lookup"拆成两个 node 调用（比如先看到 input_ids 再决定下游路径），可在 text_embed 上声明两个 method（`encode` / `embed`），通过两个 node 挂同一 module 实现。
- **graph topology 顺序**：因为 ViT/VAE 输出 `conversation_list`、text_embed 输入 `conversation_list`，edge dependency 自动让 ViT/VAE 排在 text_embed 之前；text_embed 输出 `inputs_embeds`，backbone 输入 `inputs_embeds`，自动让 text_embed 在 backbone 之前。**不需要显式顺序约束 edge**。

### 采样策略与 CFG（per-request runtime state）

> **状态**：本节描述目标契约。当前 V2 SeedOmni 代码尚未实现推理 CFG（V1 主线只有训练侧的 `cfg_ratio` 随机 condition drop，发生在 `MultimodalChatTemplate` 工具层；新设计里训练 CFG 改在 `text_embed.forward` 内部对随机选中的 sample 把 condition 段替换成 pad token；推理 CFG 是从零设计的 V2 feature）。

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
2. 若启用，调用本 module 的 `build_cfg_uncond_inputs(input_ids, attention_mask, **mm_kwargs)` 钩子构造 uncond 分支的 token 输入（pad token 由 module 通过 `set_tokenizer` 注入到的 tokenizer 自取，**不**作为 sampling 参数传入）。
3. multimodal splice 之后，把 `inputs_embeds` / `attention_mask` / `position_ids` 在 batch 维复制成 2x（偶数行 cond，奇数行 uncond），送进 backbone forward。
4. **每个 image_vq generate_step**：backbone forward 得到 (2N, V) logits，自己拆 `cond = logits[0::2]`、`uncond = logits[1::2]`，按 `cfg_weight` merge，sample 出 next_token，再在 batch 维 2x 复制喂下一步。**FSM / graph / 上层 caller 看到的 batch 始终是 1x**（即 `parallel_size`，见下）。
5. **退出 image_vq state 时**（FSM transition 触发的 `ctx_flag(image_complete)`）：backbone 在 hook 中把 2x batch shape 的 KV cache **丢弃**（不能复用给后续 text state，因为 batch shape 不兼容）。这条跟 `#13 KV cache 由模块自管` 一致。

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
- **`parallel_size > 1` 仅 T2I 模式支持**，不支持 interleave。原因：interleave 模式下 image_vq state 之后还要切回 text state，而 `parallel_size > 1` 把 batch 维彻底改写（每个 prompt 实例膨胀成 N 张独立图），切回 text 时无法干净地降回 batch=1。`infer_t2i.yaml` 是唯一允许 `parallel_size > 1` 的入口；`infer_interleave.yaml` / `infer_understanding.yaml` 必须 `parallel_size = 1`。
- **同一对 backbone + VQ codec 必须配同一个 `parallel_size`**（否则 KV cache batch 跟 VQ decode batch 错位）。OmniModel 在 build 时校验：`JanusLlama.config.parallel_size == JanusVQVAE.config.parallel_size`。
- 用户在 `model.generate()` 调用时仍可通过 sampling 字段 override `parallel_size`，OmniModel 在 generate 入口把值写回相关 module 的 config 副本（一次 generate 一个值），允许同一 weights 多种 parallel_size 推理。

#### `build_cfg_uncond_inputs` 钩子（OmniModule 可选钩子）

```python
class OmniModule:
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
        自取（tokenizer 通过 ``set_tokenizer`` 注入）。

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
class JanusLlama(LlamaModel, OmniModule):
    def set_tokenizer(self, tokenizer):
        self._pad_id = tokenizer.pad_token_id
        self._image_start_id = tokenizer.convert_tokens_to_ids("<begin_of_image>")

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
- **pad token id / image_start id 由 module 通过 `set_tokenizer` 自取**，不污染 sampling dict 也不污染 generate API。

---

## 文件结构

模块按 **model family** 组织：每个 family 的子模型放到 `modules/<family>/`，跨 family 复用的轻量模块放到 `modules/base/`。每个子模型一组 (`configuration_xxx.py`, `modeling_xxx.py`, `processing_xxx.py`) 三件套，`model_type` 写在 `configuration_xxx.py`（参考 [`veomni/models/diffusers/wan_t2v/wan_condition/configuration_wan_condition.py`](veomni/models/diffusers/wan_t2v/wan_condition/configuration_wan_condition.py) 第 7 行 `model_type = "WanTransformer3DConditionModel"`）。

```
veomni/models/seed_omni/                    # 整个目录完全重写，不保留 V1
├── module.py                               # OmniModule mixin（所有 hook 可选）
├── graph.py                                # NodeDef / EdgeDef：节点 / 边的共享数据类型 + end 关键字
├── training_graph.py                       # TrainingGraph：DAG 视图，按 edges topo 推执行序
├── generation_graph.py                     # GenerationGraph：FSM 视图，按 state.body (edges) 分发
├── configuration_seed_omni.py              # OmniConfig：modules + nodes + edges + training_graph + generation_graph（无顶层 tokenizer_path）
├── modeling_omni.py                        # OmniModel：DAG forward + FSM generate + parallel plan 聚合 + 多模块 build/load/save
└── modules/                                # 每个子模块一个文件夹；文件名固定 configuration / modeling / processing
    ├── base/                                # 跨 family 复用的轻量模块
    │   ├── text_embed/
    │   │   ├── configuration.py             # TextEmbedConfig
    │   │   └── modeling.py                  # TextEmbed —— 通用 wte + lm_head（tied/untied 自适应）
    │   └── mlp_adapter/                     # 计划中：1024→2048 等通用投影
    │       ├── configuration.py
    │       └── modeling.py
    ├── janus/                               # janus 全家桶
    │   ├── llama/                           # AR backbone（无 vocab 层）
    │   │   ├── configuration.py
    │   │   └── modeling.py
    │   ├── siglip/                          # 视觉 understanding 编码
    │   │   ├── configuration.py
    │   │   ├── modeling.py
    │   │   └── processing.py
    │   ├── vqvae/                           # 视觉 generation codec
    │   │   ├── configuration.py
    │   │   ├── modeling.py
    │   │   └── processing.py
    │   └── text_embed/                      # JanusTextEmbed（继承 TextEmbed + 边界 token emit）
    │       ├── configuration.py
    │       └── modeling.py
    ├── qwen_omni/                           # qwen-omni 全家桶（thinker + talker + ...）
    │   ├── thinker/
    │   │   ├── configuration.py
    │   │   └── modeling.py
    │   └── ...
    ├── bagel/
    │   ├── llama/
    │   │   ├── configuration.py
    │   │   └── modeling.py
    │   └── ...
    └── ...
```

文件夹名（如 `janus/siglip/`）已经给出了 `<family>_<sub_module>` 的命名空间，所以子模块内部的文件就用裸 `configuration.py` / `modeling.py` / `processing.py`，不再重复写 `configuration_janus_siglip.py`。每个子模块文件夹有自己的 `__init__.py`，把公开符号 re-export 给上一层（`from .siglip import JanusSiglip, JanusSiglipConfig, JanusSiglipProcessor`）。

`modules/__init__.py` 的 `MODULE_MIXIN_REGISTRY: dict[str, type]` 把 HF `model_type` 字符串映射到对应的合体类（或 mixin 类，由 `type()` 动态合成）。

---

## 命名规范

| 对象 | 规则 | 例子 |
|------|------|------|
| **module name**（YAML modules 池 key） | 具体模型简名（不带前缀） | `janus_llama`, `janus_siglip`, `janus_vqvae`, `janus_wte_lm_head`；通用模块用单名（`siglip`、`vqvae`） |
| **node name**（YAML nodes 池 key） | `<模型简名>_<功能>` 或裸 `<模型简名>`（backbone 类） | `siglip_encode`, `vae_encode`, `vae_decode`, `tok_encode`, `tok_decode`, `janus_llama`（不要写 `run_ar`） |
| **edge name**（YAML edges 池 key） | `<from>_to_<to>` 或语义化名字 | `vit_to_llama`, `llama_to_tok_decode`, `tok_decode_to_end` |
| **model_type**（HF config） | 由模型 config 决定，写在 `configuration_xxx.py` | `model_type = "janus_llama"` 等 |
| **拆模型脚本** | 每子模型独立文件夹 + 三件套（短文件名） | `janus/llama/{configuration.py, modeling.py}` + `janus/siglip/{configuration.py, modeling.py, processing.py}` |

**拆模型脚本怎么定子模型 `model_type`**：
- 从某个 family 拆出新子模型时（如 Janus 拆出 `janus_llama` / `janus_siglip` / `janus_vqvae` / `janus_text_embed`），每个子模型在 `<family>/<sub>/configuration.py` 里写明自己的 `model_type` 字符串。
- 拆模型脚本（`scripts/split_<family>.py`）按 sub-config 分别生成 `<output_dir>/<sub_name>/config.json`，`model_type` 字段会随 `save_pretrained` 自动落盘。
- YAML 里只填 `weights_path` 即可，HF AutoConfig 会从 `<weights_path>/config.json` 读出 `model_type`，再到 `MODULE_MIXIN_REGISTRY` 找合体类。

---

## 关键设计决策

1. **模块按 model family 组织**：`modules/<family>/` 下放该 family 拆出的子模型；`modules/base/` 放跨 family 复用的小模块（通用 text_embed、MLP adapter 等）。代码层面所有模块都是 OmniModule mixin 形态——HF / diffusers 模型 + OmniModule 多继承。

2. **OmniModule 是 mixin，不是基类**：所有钩子（`forward` / `generate_step` / `pre_forward` / `post_forward` / `get_parallel_plan`）都可选，模块按需实现。这避免了为继承模型而 ad-hoc 改 transformers 的 inheritance hierarchy。

3. **raw_batch 全局透明**：raw_batch 是整个 OmniModel forward / generate 共享的 mutable dict，每个 node 默认拿到完整 raw_batch（按自己声明的 input keys 取）。中间输出（hidden states / embeds 等）也写回同一 raw_batch（详见 #15/#16）——edges 是数据依赖契约和拓扑标记，**不是数据通道**。

4. **loss 收集按 `_loss` 后缀**（隐式）+ `to: end` sink 边（拓扑显式）：模块输出的 `*_loss` 键（已 mean 的标量）由 `OmniModel.forward()` 自动收集求和；`to: end` 是拓扑标记，保证图无孤岛，不携带数据语义。

5. **Loss mean 在 module 内部完成**：每个 module 一次 `forward` 把所有 micro-batch 跑完，`post_forward` 内部按 token-sum / token-count 做 token-level mean，吐出标量 `*_loss`。**外层只求和**——这样既保证 token-level 加权正确性（不同 micro-batch 的 token 数不同时不会退化为 batch-weighted），又让 OmniModel 协议简单（单键 `_loss`，无需 `*_loss_token_count`）。

6. **nodes / edges 平级、独立命名空间**：FSM body 只查 edges 池、edges.from/to 只查 nodes 池（外加 `end` 关键字），名字可以重名。两个平级池让结构与图论一致，错配字段在解析阶段就报错。

7. **无孤岛、无环**：每个 node 至少一条出边（指向另一 node 或 `end`）；任何环（含自环）严格禁止——自环=for-loop，应在模块内部实现。

8. **node 与 module 解耦**：图节点是 **node**（YAML 中 `nodes:` 池 key），不是 module。同一 module 可挂多个 node（`vae_encode` / `vae_decode`），module 实例只有一份，参数共享。**同一个 method 也可承担多重角色**——VQ head 的 `decode` 训练吃 `hidden+gt` 出 loss、推理吃 `hidden` 采样、吃 `token_id` lookup，按 kwargs 自分派。

9. **method 默认值**：node 不指定 `method:` 时，**训练默认 `forward`、推理默认 `generate_step`**。训练时 `forward` 走 FSDP 包装层，其他 method 直调 raw module（FSDP2 透明）；推理时 `generate_step` 是 FSM 的 next-token 采样入口。

10. **配置层不写 `model_type`**：YAML modules 池只写 `weights_path` / `config_path`，`model_type` 由 HF AutoConfig 自动从 `<path>/config.json` 读出。这让"换 backbone"等价于"换 path"，无需改 YAML 其他字段。

11. **training_graph / generation_graph 同构**：两者都基于同一组 `nodes:` / `edges:` 池，`training_graph` 只列 edges 子集（一次 DAG 遍历），`generation_graph.states.<name>.body` 也只列 edges 子集（FSM 一步循环执行）。**激活 nodes 由 edges endpoints 自动并出，执行序由 topo sort 推导**——这是框架唯一的"自动"，结构本身仍要显式给出。

12. **token_length 可插拔**：AR / VQ / DiT 都通过 `token_length` 策略统一表达，模块的推理方法（`generate_step` 或显式 method）内部实现无论是 next-token 采样还是完整去噪循环，对状态机均透明。

13. **KV cache 由模块自管**：何时复用、何时清空，是 model-specific——Janus 风格（每 token 都过同一 LLM）可复用；DiT 后回到 LLM 必须重算。OmniModel 不感知。

14. **生命周期分层**：weights 加载走 `build_foundation_model` + `build_parallelize_model`（多 path 扩展）；保存由每个 module 自己的 `CheckpointCallback` 写到 subfolder（自带 config + 可选 asset，**含 tokenizer**）；OmniConfig 顶层不再有全局 `tokenizer_path` 字段——tokenizer 是 text_embed module 的私有 asset。

15. **数据流单一抽象 raw_batch；起点 conversation_list**：raw_batch 是 mutable dict，初始只含一个 key `conversation_list`（`list[list[dict]]`，每个 item dict 含 `type` / `value` / `loss_mask` / `from_assistant`）。其他所有衍生字段（input_ids、image_embeds、attention_mask、labels、position_ids、hidden_states、...）由各 module 在 forward 阶段产出并通过返回 dict 写回 raw_batch。multimodal_transform.py 工具层只做基础 IO + resize（path → tensor 填回 item.value），不做 chat template 拼接、不做 tokenize、不做 image processor。同一份数据可同时喂给任意 ug 模型——chat template / tokenize / image processor / boundary marker 注入全部由对应 module 自管。

16. **module forward = kwargs + Dict 返回；data 100% 走 raw_batch**：每个 module 的 `forward` 仍是 `forward(**kwargs) -> Dict[str, Any]` 风格（HF 兼容、单测纯函数）；OmniModel 收到返回 dict 后**立刻按 edge.output 写回 raw_batch**，**不通过 edge 通道传给下游 module**。下游从同一 raw_batch 按自己声明的 input keys 取。这等价于"data 完全走 raw_batch、module 之间不互相返回"，但保留了 kwargs API 和 edge 显式契约。collator helper / SP slice 由各 module 在自己 `pre_forward` 中按需调用——没有全局 collator final-step、没有全局 SP slice 节点；ViT 切 image batch 维、text encoder 切 sequence 维，各管各的。

17. **Sampling state 是 per-request runtime ctx，不进 graph**：`temperature` / `top_p` / `cfg_weight` 等 sampling 超参与 KV cache 同级，通过 `OmniModel.generate(request, *, sampling=...)` 写入 ctx（推理时 ctx == 持续修改的 raw_batch），由 backbone 自己消费。CFG 的 batch-axis 2x 平铺、cond/uncond logits merge、退出 image_vq state 时 KV cache 丢弃，全部在 backbone 内部完成，graph / YAML 一行不动。`build_cfg_uncond_inputs` 是 OmniModule 可选钩子——未实现该钩子的 backbone 不允许 `cfg_weight != 1.0`，generate 入口直接 ValueError。`parallel_size` 走另一条路：是 backbone module 的 PretrainedConfig 字段（不是 sampling 参数），仅 T2I 模式支持 `parallel_size > 1`，interleave / understanding 强制 `parallel_size = 1`。同一对 backbone + VQ codec 必须配同一个 `parallel_size`，OmniModel build 时校验。

18. **token 拼接 / boundary marker / chat template 全部下放给对应 module**：text encoder（text_embed）拼接 system prompt + 文本 item + EOS + role marker，自带 tokenizer 自带 chat template 实现；ViT/VAE 在 forward 阶段往 `conversation_list` 中给 image / vq_image item 加 `boi` / `eoi` marker；audio encoder 给 audio item 加 `audio_bos` / `audio_eos` marker；video 同理。**没有 chat_template 这个独立 module、没有顶层 chat template 工具层、没有顶层 image_pattern 注册表**。每个 family 的 chat template 写在自己的 `modules/<family>/text_embed/modeling.py` 里，互不干扰。两次 input_ids 长度变化（text_embed 输出"每张 image 1 个 placeholder"序列 → backbone splice 扩展成 N patch tokens）的同步 labels / mask / position_ids 对齐由 backbone 在 splice 时一次性处理（参见"Backbone pre_forward 完成多模态 splice + 长度对齐"和"Position IDs"两节）。

19. **RL 一致性**：训练 node 的 `forward()` 和推理 node 的 `generate_step()` 共用同一底层模型实现，log-prob 直接从 logits 提取，无两套实现分叉。一个 module type 一个 instance；RL 场景的 reference model 和 actor model 是两个独立 instance（model_type 可以相同）。
