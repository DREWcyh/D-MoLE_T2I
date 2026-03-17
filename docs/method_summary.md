# D-MoLE_T2I 方法总结

基于当前代码实现生成，面向“项目方法说明、实验记录、后续改法对照”用途。

- 生成日期：2026-03-16
- 代码基线：`9dfa0182797f402322fd21ba3f0aab4744467dac`
- 代码根目录：`/home/chenyiha24/D-MoLE_T2I`

## 1. 项目目标

本项目实现的是一个面向文本到图像持续学习的 D-MoLE 风格方法。核心目标是在 PixArt-XL-2 的基础上，按任务顺序逐个学习多个 item domain，而不是为每个新任务全量微调整个 Diffusion Transformer。

当前实现的主线思想是：

1. 基于 PixArt-XL-2 的 `Transformer2DModel` 作为 backbone。
2. 对每个新任务只新增一组 LoRA expert，而不是改动全部参数。
3. 新 expert 的挂载位置不是固定的，而是通过 ZCP 打分动态决定。
4. 每个任务训练结束后，用一个轻量自编码器 router 学习该任务的特征分布。
5. 新任务到来时，router 先判断它更像哪个历史任务；如果相似，就把对应旧 expert 的权重拷贝给新 expert 作为初始化。
6. 推理时再由 router 按 prompt 自动路由到最合适的 expert。

从“系统组成”上看，本项目是：

\[
\text{PixArt Backbone} + \text{Dynamic LoRA Allocation} + \text{Task Router} + \text{Sequential Continual Training}
\]

## 2. 代码模块总览

| 文件 | 作用 | 关键接口 |
| --- | --- | --- |
| `main.py` | 训练主入口，串起持续学习流程 | `main()`、`train_on_dataset()` |
| `dataset.py` | DreamBooth 数据集、prompt 编码、batch 拼接 | `DreamBoothDataset`、`collate_fn` |
| `feature_extractor.py` | 文本/图像特征抽取与融合 | `extract_and_fuse_features()` |
| `router.py` | 基于 AE 的任务路由器 | `DMoLE_Router` |
| `zcp_allocator.py` | ZCP 打分与动态 LoRA expert 挂载 | `compute_zcp_scores()`、`add_dmole_lora_adapter()` |
| `inference_dmole.py` | 推理入口，负责加载 experts 与 router 并生成图片 | `main()` |
| `scripts/train_dmole.sh` | 训练脚本，配置默认数据、模型和超参数 | shell launcher |
| `scripts/infer_dmole.sh` | 推理脚本，配置 prompt 文件和输出目录 | shell launcher |
| `ds_config/item.json` | DeepSpeed 配置 | ZeRO Stage 1、fp16、micro-batch |

## 3. 记号与变量

为了便于把代码和公式对齐，下面统一使用这组记号：

- \(x\)：输入图像
- \(p\)：输入文本 prompt
- \(E_{\text{text}}(p)\)：T5 文本编码器输出
- \(E_{\text{vae}}(x)\)：VAE 编码器输出的 latent
- \(z_0\)：无噪 latent
- \(z_t\)：加噪后的 latent
- \(\epsilon\)：高斯噪声
- \(t\)：扩散时间步
- \(f_t\)：文本特征
- \(f_v\)：图像特征
- \(g_b\)：第 \(b\) 个 transformer block 的 saliency 分数
- \(\mathcal{B}_t\)：任务 \(t\) 选中的 expert 挂载 block 集合
- \(A_k(\cdot)\)：第 \(k\) 个任务的 autoencoder router
- \(\theta_k\)：第 \(k\) 个 expert 的 LoRA 参数

## 4. Backbone 与基础训练对象

### 4.1 基础模型

当前训练和推理主干都来自 PixArt-XL-2：

- tokenizer：`T5Tokenizer`
- text encoder：`T5EncoderModel`
- latent encoder：`AutoencoderKL`
- denoiser backbone：`Transformer2DModel`

训练时实际更新的不是 backbone 本身，而是 PEFT/LoRA 附加参数。

### 4.2 LoRA 参数化

项目采用标准 LoRA 思路，把线性层的权重更新写成低秩形式：

\[
W' = W + \Delta W,\qquad \Delta W = BA
\]

其中：

- \(W\) 是冻结的原始权重
- \(A \in \mathbb{R}^{r \times d_{\text{in}}}\)
- \(B \in \mathbb{R}^{d_{\text{out}} \times r}\)
- \(r\) 是 LoRA rank

当前代码里：

- rank 来自 `--lora_rank`
- `lora_alpha = 2r`
- 初始化为 `gaussian`
- 只训练当前 expert 对应的 LoRA 参数，其余参数全部冻结

对应代码见：

- `main.py` 中的 `add_new_lora_adapter()`
- `zcp_allocator.py` 中的 `add_dmole_lora_adapter()`

## 5. 数据与输入表示

### 5.1 DreamBoothDataset

`dataset.py` 里的 `DreamBoothDataset` 负责读取：

- instance images
- instance prompt
- 可选的 class images
- 可选的 class prompt

图像变换为：

\[
x' = \text{Normalize}(\text{Crop}(\text{Resize}(x)))
\]

当前具体实现：

1. Resize 到 `size`
2. `CenterCrop(size)` 或 `RandomCrop(size)`
3. `ToTensor()`
4. `Normalize([0.5], [0.5])`

### 5.2 文本编码

prompt 先被 tokenizer 编码成：

\[
\text{input\_ids},\ \text{attention\_mask}
\]

再送入 T5 文本编码器：

\[
H = E_{\text{text}}(p) \in \mathbb{R}^{B \times L \times 4096}
\]

如果开启 `--pre_compute_text_embeddings`，则会预先计算并缓存每个任务对应的文本 embedding。

### 5.3 Prior Preservation

如果启用 `--with_prior_preservation`，batch 会被拼接成 instance + class 两部分，并在 loss 中加入 prior 项：

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{instance}} + \lambda_{\text{prior}} \mathcal{L}_{\text{prior}}
\]

其中：

\[
\mathcal{L}_{\text{prior}} = \text{MSE}(\hat{y}_{\text{prior}}, y_{\text{prior}})
\]

但是需要注意：当前默认训练脚本 `scripts/train_dmole.sh` 传入了 `CLASS_DIRS` 和 `CLASS_PROMPTS`，却没有传 `--with_prior_preservation`，因此默认配置下 prior preservation 实际并没有被启用。

## 6. 扩散训练目标

### 6.1 Latent 构造

图像先经过 VAE 编码得到 latent：

\[
z_0 = E_{\text{vae}}(x) \cdot s
\]

其中 \(s\) 是 `vae.config.scaling_factor`。

### 6.2 前向加噪

代码通过 `noise_scheduler.add_noise()` 完成 DDPM 风格加噪，本质上对应：

\[
z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon,\qquad \epsilon \sim \mathcal{N}(0, I)
\]

### 6.3 去噪预测

denoiser 的输入是：

- noisy latent \(z_t\)
- 文本条件 \(H\)
- 时间步 \(t\)
- 分辨率条件 `resolution`
- 宽高比条件 `aspect_ratio`

网络输出：

\[
\hat{\epsilon}_\theta(z_t, H, t)
\]

或在 `v_prediction` 模式下输出速度项。

### 6.4 训练损失

若 `prediction_type == epsilon`：

\[
\mathcal{L}_{\text{diff}} = \|\hat{\epsilon}_\theta - \epsilon\|_2^2
\]

若 `prediction_type == v_prediction`：

\[
\mathcal{L}_{\text{diff}} = \|\hat{v}_\theta - v(z_0,\epsilon,t)\|_2^2
\]

若启用了 `snr_gamma`，则使用 SNR reweighting：

\[
w_t = \frac{\min(\mathrm{SNR}(t), \gamma)}{\mathrm{SNR}(t)}
\]

\[
\mathcal{L}_{\text{diff}} = w_t \cdot \text{MSE}
\]

实际代码中如果是 `v_prediction`，会先执行：

\[
\mathrm{SNR}(t) \leftarrow \mathrm{SNR}(t) + 1
\]

## 7. 特征提取模块

文件：`feature_extractor.py`

### 7.1 文本特征

给定 T5 输出：

\[
H \in \mathbb{R}^{B \times L \times 4096}
\]

先对序列维做 max pooling：

\[
t_{\text{raw}} = \max_{i=1,\dots,L} H_i \in \mathbb{R}^{B \times 4096}
\]

然后用固定随机正交投影降维到 512：

\[
R \in \mathbb{R}^{4096 \times 512},\qquad R^\top R = I
\]

\[
t = t_{\text{raw}} R
\]

其中 \(R\) 的构造方式是：

1. 固定随机种子 `42`
2. 采样高斯矩阵
3. 用 QR 分解取正交基

最后归一化：

\[
f_t = \frac{t}{\|t\|_2}
\]

### 7.2 图像特征

若提供 `latents`，则对 latent 做 `AdaptiveMaxPool2d` 到 \(11 \times 11\)：

\[
v_{\text{pool}} = \text{AdaptiveMaxPool2d}(z_0, 11, 11)
\]

展平后得到：

\[
v \in \mathbb{R}^{B \times 484}
\]

再归一化：

\[
f_v = \frac{v}{\|v\|_2}
\]

### 7.3 注释设计 vs 当前实际实现

代码注释写的是“最终输出文本与图像拼接后的 996 维特征”：

\[
z = [f_t ; f_v] \in \mathbb{R}^{996}
\]

但当前文件最后实际返回的是：

\[
z = f_t \in \mathbb{R}^{512}
\]

即：

- 图像分支被计算了
- 拼接张量 `z_feat_batch` 也被构造了
- 但最终 `return t_feat_norm`

因此当前项目的 router 与任务特征实际上是“文本单模态 512 维特征”，而不是代码注释中描述的“文本+图像融合 996 维特征”。

这也是当前实现里最重要的一个方法细节。

## 8. Router 模块

文件：`router.py`

### 8.1 总体思路

router 采用“每个任务一个自编码器”的方式建模任务分布。给定一个输入特征 \(z\)，如果某个任务的 AE 能较好重构它，说明这个输入更接近该任务。

### 8.2 基础特征剥离

router 维护一个 base feature：

\[
b \in \mathbb{R}^{1 \times d}
\]

它由固定 prompt `"A photo of a item"` 生成。

预处理函数为：

\[
\tilde{z} = z - b
\]

\[
\bar{z} = 10 \cdot \frac{\tilde{z}}{\|\tilde{z}\|_2}
\]

代码里就是：

\[
\texttt{\_process\_features}(z)=10\cdot \text{normalize}(z-b)
\]

### 8.3 AE 结构

每个任务的 autoencoder 结构为：

\[
512 \rightarrow 256 \rightarrow 64 \rightarrow 256 \rightarrow 512
\]

中间使用：

- `LayerNorm`
- `GELU`

更具体地说：

\[
h_1 = \text{GELU}(\text{LN}(W_1 z + b_1))
\]

\[
h_2 = \text{GELU}(W_2 h_1 + b_2)
\]

\[
h_3 = \text{GELU}(\text{LN}(W_3 h_2 + b_3))
\]

\[
\hat{z} = W_4 h_3 + b_4
\]

### 8.4 Router 训练目标

对任务 \(k\) 的 feature matrix \(Z_k\)，训练对应 AE：

\[
\mathcal{L}_{\text{AE}}^{(k)} = \frac{1}{N}\sum_{i=1}^{N}\|A_k(z_i) - z_i\|_2^2
\]

优化器是：

- `Adam`
- 学习率 `1e-3`
- epoch 数固定 `50`

### 8.5 路由判定

对输入 \(z\)，分别计算所有任务 AE 的重构误差：

\[
e_k(z) = \|A_k(z) - z\|_2^2
\]

选择误差最小的任务：

\[
k^* = \arg\min_k e_k(z)
\]

若最小误差仍高于阈值 \(\tau\)，则判为 OOD：

\[
e_{k^*}(z) > \tau \Rightarrow \text{fallback}
\]

否则：

\[
\text{route}(z)=k^*
\]

## 9. ZCP 打分模块

文件：`zcp_allocator.py`

### 9.1 目标

ZCP 的作用是估计“当前任务最值得插 expert 的 transformer block”，从而避免把 LoRA 挂满所有层。

### 9.2 当前实际打分方式

代码对以下参数临时开放梯度：

- `to_q`
- `to_k`
- `to_v`
- `to_out.0`

然后从 dataloader 中只取一个 batch，执行一次前向和反向，得到噪声预测损失：

\[
\mathcal{L}_{\text{zcp}} = \text{MSE}(\hat{\epsilon}_\theta, \epsilon)
\]

之后按 block 聚合梯度范数：

\[
g_b = \sum_{p \in \Theta_b} \left\|\frac{\partial \mathcal{L}_{\text{zcp}}}{\partial p}\right\|_2
\]

这里：

- \(\Theta_b\) 是第 \(b\) 个 `transformer_block` 中本次有梯度的参数集合
- 由于仅对 `to_q/to_k/to_v/to_out.0` 打开梯度，实际 saliency 主要来自这些 attention 相关参数

### 9.3 重要实现说明

虽然项目参数里定义了 `--zcp_sample_ratio`，但当前 `compute_zcp_scores()` 并没有使用这个参数，而是直接：

1. `next(iter(dataloader))`
2. 只用第一个 batch 打分

因此当前实现的 ZCP 是“单 batch 近似打分”。

## 10. 动态 LoRA expert 分配

文件：`zcp_allocator.py`

### 10.1 Block 选择策略

给定所有 block 的 saliency 分数 \(\{g_b\}\)，先降序排序，然后选择最小的 block 集合 \(\mathcal{B}_t\)，使其累计显著性达到阈值 \(\rho\)：

\[
\frac{\sum_{b \in \mathcal{B}_t} g_b}{\sum_j g_j} \ge \rho
\]

当前约束还有两条：

- 最多选 `param_budget` 个 block
- 最少选 2 个 block

所以最终规则是：

1. 按 \(g_b\) 从大到小遍历
2. 如果累计比例达到 `zcp_rho` 则停止
3. 如果达到 `param_budget` 则停止
4. 如果最后不足 2 个 block，则强制取前 2 个

### 10.2 实际挂载的层

在被选中的 block 内，LoRA 只挂到以下模块：

- `to_k`
- `to_q`
- `to_v`
- `to_out.0`
- `ff.net.0.proj`
- `ff.net.2`

所以当前 expert 不是“整层 LoRA”，而是“被选 block 内特定 attention/FFN 子模块的 LoRA”。

### 10.3 当前任务 expert

第 \(t\) 个任务的新 expert 名称为：

\[
\text{adapter\_name} = \text{stage}(t+1)
\]

也就是说：

- `task_0` 对应 `stage1`
- `task_1` 对应 `stage2`
- `task_2` 对应 `stage3`
- 以此类推

## 11. 持续学习流程

文件：`main.py`

### 11.1 总流程

对任务序列 \(\{(D_t, p_t)\}_{t=1}^{T}\)，当前训练流程可以概括为：

1. 加载 PixArt backbone、T5、VAE、scheduler
2. 若需要，预先计算全部任务 prompt embedding
3. 对每个任务 \(t\)：
4. 构造当前任务的临时 dataloader
5. 用 ZCP 给各 block 打分
6. 为该任务新增一个 LoRA expert
7. 从当前任务数据抽取 feature matrix
8. 用 router 判断最相似历史任务并可选地进行 mentor 初始化
9. 训练当前 expert
10. 保存 `task_t/transformer` 和 `router.bin`

### 11.2 任务特征矩阵

对当前任务的数据，项目会遍历一个临时 dataloader，提取每个 batch 的特征：

\[
Z_t = \{z_i\}_{i=1}^{N_t}
\]

并计算任务中心：

\[
c_t = \frac{1}{N_t}\sum_{i=1}^{N_t} z_i
\]

这里：

- `task_feature_matrix = torch.cat(all_z_feats, dim=0)`
- `task_centroid = task_feature_matrix.mean(dim=0, keepdim=True)`

### 11.3 Mentor 初始化

对第 \(t>1\) 个任务，若 router 认为它与历史 expert \(k\) 最相近，则当前代码会把旧 expert 参数直接复制给新 expert：

\[
\theta_t \leftarrow \theta_k
\]

更准确地说，是对当前 expert 名称匹配到的参数逐一做字符串替换并复制值。

这一步不是蒸馏，也不是加正则，而是“显式权重继承”。

### 11.4 训练阶段

每个任务只训练当前 expert，对 backbone 和历史 expert 保持冻结。

优化器：

- `AdamW`
- 学习率来自 `--learning_rate`

调度器：

- `get_scheduler()`
- 默认脚本使用 `constant`

混合精度：

- `fp16` 或 `bf16`
- 默认脚本使用 `fp16`

DeepSpeed：

- 使用 `ds.initialize()`
- 默认配置为 ZeRO Stage 1

## 12. 推理流程

文件：`inference_dmole.py`

### 12.1 推理阶段加载内容

推理时会加载：

1. 基础 PixArt 模型
2. `adapter_dir` 下所有 `task_*` expert
3. 最后一个任务目录下的 `router.bin`

其中 `find_adapter_config()` 允许兼容三种目录结构：

1. `task_N/transformer/stageN/`
2. `task_N/transformer/`
3. `task_N/`

### 12.2 推理特征与路由

推理时只根据文本 prompt 计算路由特征：

\[
z_{\text{infer}} = \text{extract\_and\_fuse\_features}(H,\ \text{latents=None})
\]

然后用 router 选择 expert：

\[
k^* = \arg\min_k e_k(z_{\text{infer}})
\]

若最小误差超过阈值，则 fallback 到 `stage1`。

需要特别注意：

- 推理里的路由阈值不是命令行参数，而是硬编码的 `0.2`
- 训练脚本里的 `router_threshold=0.000001` 不会自动传到推理阶段

### 12.3 图像生成

选定 expert 后：

1. `pipeline.transformer.set_adapter(best_adapter)`
2. 调用 `PixArtAlphaPipeline`
3. 生成图像并保存

当前默认生成参数：

\[
\text{height}=512,\quad \text{width}=512,\quad \text{steps}=30,\quad \text{CFG}=4.5
\]

## 13. 训练与推理脚本默认配置

### 13.1 `scripts/train_dmole.sh`

默认训练 4 个任务：

1. `dog`
2. `dog3`
3. `cat2`
4. `shiny_sneaker`

默认超参数：

- GPU：`0,1`
- batch size：`4`
- max train steps：`500`
- learning rate：`2e-5`
- LoRA rank：`16`
- `use_dmo_le`
- `param_budget=28`
- `router_threshold=1e-6`
- `zcp_sample_ratio=0.01`

### 13.2 `scripts/infer_dmole.sh`

默认读取 4 个 prompt 文件，并把输出保存到 `outputs/inference/...` 下。

## 14. DeepSpeed 配置

文件：`ds_config/item.json`

当前配置摘要：

- `train_batch_size = 4`
- `train_micro_batch_size_per_gpu = 2`
- `gradient_accumulation_steps = 1`
- `gradient_clipping = 1.0`
- `zero_optimization.stage = 1`
- `fp16.enabled = auto`

即默认是一个轻量的 ZeRO Stage 1 + fp16 训练配置。

## 15. 当前实现中的重要方法注记

这部分不是“理论应该怎样”，而是“当前代码实际怎样”。

### 15.1 特征提取当前是文本单模态

虽然 `feature_extractor.py` 里构造了图像分支，并且注释声称会输出 996 维文本图像融合特征，但当前实际返回的是 512 维文本特征：

\[
z = f_t
\]

因此 router 当前学习到的是“文本域差异”，不是“文本+图像联合域差异”。

### 15.2 `zcp_sample_ratio` 当前未生效

参数存在，但 `compute_zcp_scores()` 没有用它控制样本数；当前只用 dataloader 的第一个 batch 做 ZCP 估计。

### 15.3 `use_inter_modal_curriculum` 当前未生效

训练脚本传了 `--use_inter_modal_curriculum`，但在 `main.py` 中没有看到对应的训练逻辑。

### 15.4 默认训练脚本并未真正启用 prior preservation

原因是没有传 `--with_prior_preservation`。

### 15.5 训练/推理默认输出目录有一处不一致

当前默认脚本中：

- 训练输出默认到 `outputs/train/dmole_without_prior/items_sequential`
- 推理脚本默认从 `outputs/train/dmole_without_prior_3/items_sequential/run` 读取 adapter

如果直接用默认脚本训练后立刻推理，需要手动把 `ADAPTER_DIR` 调整到实际训练输出目录。

### 15.6 推理脚本中的 `num_validation_images` 参数当前未真正生效

虽然命令行里有 `--num_validation_images`，但推理实现里每个 prompt 实际只生成并保存一张图：

\[
\text{one prompt} \rightarrow \text{one image}
\]

## 16. 一句话概括当前项目方法

当前项目实现的是一个基于 PixArt-XL-2 的持续学习文本到图像方法：它使用单任务单 expert 的 LoRA 增量学习框架，用单 batch ZCP 估计当前任务最重要的 transformer block，在这些 block 上动态挂载 LoRA expert，再用基于文本特征的自编码器 router 学习任务分布并完成训练期 mentor 初始化与推理期 expert 路由。

## 17. 后续文档维护建议

如果后续继续改方法，建议每次更新这个文件时都补三项：

1. 改动前后的公式差异
2. 改动影响到的模块文件
3. 实验脚本和结果目录

最实用的写法可以是：

- `docs/method_summary.md`：记录“当前主实现”
- `docs/method_history/xxxx-xx-xx-xxx.md`：记录“某次改法”

这样你后面做 ablation、写论文方法部分、回溯旧版本都会轻松很多。
