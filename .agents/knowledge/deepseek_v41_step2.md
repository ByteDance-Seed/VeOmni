# DeepSeek-V4.1-Flash — Step 2 落地记录

> 本文是飞书工作文档 `Gaf4djLBFoveF0xlIQVcJwydnWe` 的增量草稿。本机到
> `open.feishu.cn` 的网络不通（IPv4 立即失败、IPv6 超时），无法用 `lark-cli`
> 直接写入，故先落在仓库内。同步后可删除本文件。

## 模块切分与规模

四个模块在真实 `config.json` 下建出的参数量：

| 模块 | 参数量 | 并行计划 |
|---|---|---|
| `deepseek_v41_vision` | 0.49 B | 无（FSDP 足够） |
| `deepseek_v41_text_encoder` | 1.32 B | `emb` |
| `deepseek_v41_ngram` | 196.61 B | `emb` |
| `deepseek_v41_llm` | 550.56 B | `ep` |
| **合计** | **748.98 B** | |

其中 llm 的 543.4 B 是路由专家。DSpark 的三个 draft 阶段未实现，其 2401 个张量
被转换器跳过并计数。

## backbone：三处顺序

layer loop 里有三处顺序照抄参考实现，写反了都不会报错，只会安静地给出错误结果：

1. **Engram 写在其所属层之前**。参考实现在 loop body 顶部把查表结果加到流上，
   因此该层的 attention 是第一个看到它的。写在层之后会把 n-gram 交给下一层，
   而让本层读到一个训练权重从未产生过的流。
2. **DSpark hook 读目标层的输入而非输出**。参考实现里
   `main_hiddens.append(h.mean(dim=2))` 位于 `layer(...)` 之上。
3. **最终 collapse 用最后一层 FFN 的 `pre`**，而不是从最终流重新算一个。后者正是
   `DeepseekV4HyperHead` 的做法，也正是它不能被复用的原因。

样本边界只从 `position_ids` 推导（`sequence_slices_from_positions`），不再另设一个
可能与之矛盾的入参。

## accelerated 层

### 拒绝 Ulysses SP

压缩分组、候选块、压缩历史可见性都是在任何 kernel 之前用 Python 按整条序列构建的。
序列被切分的 rank 会为一条不存在的序列建计划、并跨分片边界池化。要支持必须同时切分
计划本身，那是另一个改动；静默接受则是错误结果。

### 稀疏 FLOPs 口径

V4.1 每个 query 只读 `sliding_window + index_topk` 个条目，与序列长度无关，因此**不能**
继承稠密 backbone 的 `12 * S^2` 项。只有 eager indexer 随平方增长，且仅在
`index_source_layer_ids` 上，故单列。

校验：把估算换算成等效激活参数量，2k / 8k / 32k token 分别为 26.49 / 26.59 / 26.99 G
——基本恒定，说明计量被 per-token 项主导。

### image mask 与 packing 同步构建

不做第二遍扫描。mask 与 packing 不一致会路由错 token 而不改变任何形状。

## packing 不变量（GPU bf16 实测）

- **单层、索引未平移的样本**：逐比特相同（0 / 65536 个元素不同）。
- **单层、索引平移的样本**：0.86 ulp，74 / 49152 个元素移动。不对称有因——packed 里
  sample 0 仍从索引 0 开始，sample 1 整体平移，kernel 分块边界随之改变，累加顺序不同。
- **6 层累积**：2.45 / 1.72 ulp。中位元素在 6 层时仍逐比特相等。
- **隔离性**：改写 sample 0 对 sample 1 的输出影响为**精确的 0**。

结论：packed 与单独跑的差异是 bf16 舍入放大，不是信息泄漏。容差应按 ulp 而非绝对值定。

## ngram 节点接图

共享的 `pack_input_ids` 只收 text 项（文本编码器只需要这些），**不能**用于 ngram：
ngram 发出的行被 backbone 按位置索引，必须覆盖含图像跨段的完整打包序列。用它会短掉
所有图像的长度，并把 n-gram 直接哈希穿过本该被图像打断的跨段。

`pack_ngram_inputs` 覆盖全长，图像跨段填 `pad_token_id` + `False` mask。用 pad 而非
图像占位 token，是因为被阻断的回看槽本来就压缩成 pad——两种「不可哈希」的成因由此一致，
而不是各自碰巧都能工作。

图像跨段的长度是 ViT 为它产出的 token 数，所以 `vision -> ngram` 是真实依赖边，已写入
`configs/seed_omni/DeepSeek/deepseek_v41_flash/train/graph_train.yaml`。

文本 ids 改从 `meta['input_ids']` 读：`encode` 会把文本项的 `value` 覆写成 embedding，
而 ngram 与 `text_encoder.encode` 都只依赖 vision，DAG 不规定谁先跑。模板在 `tokenize`
存副本并在 `merge_text_embeds` 保持同步（BAGEL 模板已有同样做法）。该改动经验证是纯增量的：
`value` / `labels` / `attention_mask` 与基类逐比特相同。

## checkpoint 转换器

### 命名方向与原计划相反的一点

发布的 checkpoint **已经是参考推理栈的命名**（`layers.N.attn.wq_a`、`ffn.gate.bias`、
`hc_attn_fn`），不是 HF 命名——`inference/convert.py` 的重命名已经施加其上。因此转换器是
把它们**走回去**：`attn_norm`/`ffn_norm` → HF 拼写，`ffn.gate.bias` →
`e_score_correction_bias`，`hc_attn_*` 按 site 拆分，`w1/w2/w3` → gate/down/up。

vision 是例外：其参数名与 `DeepseekV41Vision` 完全一致，纯直通。

### 不建模型

发布件 510 GB / 48 分片 / 96085 个张量，反量化到 bf16 约三倍。`from_pretrained` 后
`save_pretrained` 需要整个模型常驻两份。改为通过 checkpoint 自带的 index 随机读取、
变换、边走边刷出分片，峰值内存是**一个 MoE 层的专家**而非一个模型。

### 量化

| 部位 | 发布格式 | 输出 | 体积 |
|---|---|---|---|
| 稠密投影 | fp8 + 32×32 块 scale | bf16 | — |
| 路由专家 | fp4（两值一字节）+ 块 scale | bf16 | 272 GB → 1087 GB |
| Engram 表 | fp8 + 块 scale | bf16 | 196.6 GB → 393 GB |

专家必须出 bf16：`DeepseekV4Experts` 的 FP4 QAT 是**从 bf16 做 fake-quant**，留在 fp4
存储等于对已量化张量再量化一次。

**Engram 表出 bf16 是临时选择。** 表本身预期冻结，fp8 存储 + gather 时反量化可以省一半，
但 `DeepseekV41Ngram` 用 `nn.Embedding` 持有，没有地方放 per-row scale。这是一个已知的
实现与文档不一致点（docstring 声称「stay frozen and in FP8，8 rank 各 ~23 GiB」）。

### token_map

从 tokenizer 烘焙，并校验其压缩词表大小等于 config 的 99092。所有哈希乘子都由该大小导出，
tokenizer 版本对不上会让两张 384M 行的表被重新寻址——每次查表都返回错误的行，而下游不会
有任何失败。

### 验证

- 22 个测试跑在合成的「发布布局」checkpoint 上（fp8 稠密带块 scale、fp4 打包专家、
  跨分片 index），终点是对**真实模块定义**做 `load_state_dict`，无 unexpected、无 missing。
  仅自洽的重命名通不过这一关。
- 对真实 checkpoint：从 safetensors header 读形状，与真实 config 在 meta device 上建出的
  模块比对，**915 个张量、0 个错配**。
- 反量化数学单独对官方 `convert.py` 验数（20 个测试），并在真实张量上确认：fp8 稠密层
  std 0.0239、fp4 专家 std 0.0236——两条独立量化路径解出同一分布。

## 待办

- DSpark（三个 draft 阶段 + Markov head + confidence head），2401 个张量。
- Engram 表的 fp8 存储路径（省 196 GB）。
- `base.yaml` / `modules_train.yaml`（目前只有 `graph_train.yaml`）。
