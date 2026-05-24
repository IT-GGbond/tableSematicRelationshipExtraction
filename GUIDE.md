# 表格语义关系提取：从基础运行到上分优化指南

这份文档旨在指导您跑通本次“表格语义关系提取”赛题的全流程，并针对赛题少样本（Few-Shot）结合特定公式的核心评分机制，提供后续提升成绩（Score）的明确方向。

---

## 1. 基础运行指南

### 1。1 环境与代码准备

无论您是在本地还是在 **百度飞桨 (AI Studio)** 上运行，请严格按照以下步骤准备：

1. **拉取项目代码**
   在终端中使用 git 进行克隆：

   ```bash
   git clone https://github.com/IT-GGbond/tableSematicRelationshipExtraction.git
   cd tableSematicRelationshipExtraction
   ```

2. **安装依赖**
   由于我们已排除了 `aistudio-sdk` 导致的错误，这里建议使用稳定的依赖列表。
   ```bash
   pip install -r environment/requirements.txt
   ```
   _(注：在飞桨中，建议先执行 `pip uninstall -y paddlenlp aistudio-sdk` 删掉旧版，再进行安装。)_

### 1。2 模型训练 (Train)

配置好环境之后，直接执行 `baseline/train.py` 开始核心的模型微调过程。

```bash
python baseline/train.py --epoch 20 --batch_size 32 --train_dir ./dataset/Train_Set
```

_(注意：`train_dir`需要指向您解压好后的训练集csv文件夹。如果您还没准备好数据，请将数据集下载后放到 `dataset/Train_Set` 里面)_

**训练流程说明：**

- 日志中会输出每个 Epoch 的 Loss 和在验证集上的准确率 (Val Acc)。
- 代码自带 **Early Stopping (早停机制)**，默认当验证集指标连续 3 轮没有破新高时，将自动停止训练以防止过拟合。
- 训练完成后，在 `./cpa_output/` 会生成一个时间戳命名的模型存放目录，里面会有类似 `best_model.pdparams`，这就是训练出来最好的权重文件。

### 1。3 加速训练 (针对 V100 16GB 等高级显卡的推荐配置)

如果您觉得默认训练时间过长，可以利用以下参数大幅缩短训练时间，且基本不影响甚至有时会提高训练效率：

```bash
python baseline/train.py \
    --epoch 10 \
    --batch_size 64 \
    --use_amp \
    --shortcut_name distilbert-base-uncased \
    --train_dir ./dataset/Train_Set
```

**参数加速原理说明：**
- `--epoch 10`：强制降低最大训练轮数。配合早停机制（默认3轮无提升即停），通常跑不满10轮就会输出最佳模型。
- `--batch_size 64`：V100 16GB 的显存足够支撑到 batch size 64（在 max_length=128 的情况下），单轮速度直接翻倍。
- `--use_amp`：开启自动混合精度计算（AMP），利用 V100 的 Tensor Core 提速并降低显存占用。
- `--shortcut_name distilbert-base-uncased`：使用参数量更小的轻量化预训练模型，计算速度成倍加快。

### 1。4 运行推理与生成提交文件 (Inference)

模型训练结束且得到 `best_model.pdparams` 后，通过 `baseline/infer.py` 对测试集进行预测：

```bash
python baseline/infer.py \
    --input_csv ./dataset/test.csv \
    --labels_path ./dataset/labels.txt \
    --model_path ./cpa_output/cpa_20260521_150928/best_model.pdparams \
    --output_file ./submission.csv
```

**产出物**：
执行完成后，项目根目录会生成一个 **`submission.csv`**，这就是您在比赛官方页面可直接提交的文件。

---

## 2. 优化与提分指南 (How to improve score)

官方基线（Baseline Score）为 **0.66135**。观察本赛题的**特殊评估方式**：
**分数 = (少样本权重 × 准确率) 的加权求和**。
越是罕见、冷门的长尾数据关系（数量最少），它的权重占比极大。因此，单纯追求整体准確率 (Accuracy) 用处受限，**您必须重点打击那些数据量极少的罕见关系。**

以下提供几个逐步深入的提分(上分)方法：

### 方法 1：解决样本极度不均衡（针对计分公式的核心策略）

赛题中 `counts_min` 是决定权重的关键。对于那些在 `train_set` 中出现次数小于 10 甚至只有 1~2 次的关系，目前基础交叉熵函数（CrossEntropyLoss）通常会让模型趋向于把它们预测为常见类别（俗称“被淹没”了）。

- **修改 Loss 损失函数：Focal Loss 或 Class-Balanced Loss**
  在 `train.py` 里，原来的损失函数如果是 `nn.CrossEntropyLoss()`，您可以替换为加权交叉熵或 `FocalLoss`。您可以自己编写一个逻辑，求出每个 Label 在本地训练集出现的次数，给出倒数或者指数衰减权重传入 Loss，**狠狠惩罚预测错罕见类别的行为**。
- **小样本数增强 (Oversampling / Data Augmentation)**
  写个脚本盘点一下训练数据，对于出现次数很少的类别行，通过简单的交换位置（如果在关系上对称）、替换实体等数据增强方式，强制塞回训练集里。

### 方法 2：更换更强大的前沿预训练模型 (最快速见效)

Baseline 采用的是比较旧且基础的 `bert-base-uncased`。

- 您在 `train.py` 和 `infer.py` 运行时，可以更改参数 `--shortcut_name`。
- **推荐进阶模型**：`roberta-base`，或者更大型的 `roberta-large`。如果是垂类英文数据集，考虑飞桨模型库里面的 `ernie-3.0-base-zh` 或者支持英中混合的语义模型，它们通常能带来 2~5% 的立竿见影的硬提分。

### 方法 3：引入提示学习 (Prompt Engineering)

在基线代码中，我们通常只是将 `Subjetct` 和 `Object` 作为输入拼接进去。
但是在**少样本 (Few-Shot)** 场景，激发模型本身自带的知识储备极其关键。

- **做法：** 在将数据输入 Tokenizer 前，把两列拼成自然的句式，例如：
  _原来：_ `Jobcenter [SEP] Enzo Cormann`
  _改进后：_ `[Subject] Jobcenter and [Object] Enzo Cormann have a relation of [MASK].`
  然后利用模型去恢复这个 Mask，这样往往比粗暴的拼接特征进行分类提取有明显得多的理解能力提升。

### 方法 4：模型融合 (Ensemble)

打比赛后期必备。

- 用不同的随机种子 (Random Seed) 训练 3-5 遍模型。
- 用不同架构的模型（比如一个跑 RoBERTa，一个跑 BERT）分别产出对 `test.csv` 每个类别的预测概率得分。
- 将这几个预测的概率(Logits) 加在一起，选取分数最大的那一个作为这一行的猜测关系（软投票 Soft Voting）。这种做法通常非常稳定，甚至可以再往上拔高 1~2%。

---

## 3. 已落地的代码改进 (Improved Baseline v3)

### 3.0 踩坑记录：为什么 roberta + FocalLoss + 自然语言 prompt 反而降分了？

v2 版本尝试了 roberta-base + Focal Loss + `"Subject: ..." / "Object: ..."` 自然语言前缀，实测 submission 得分从 0.67 **降到 0.66**。分析 726 条预测差异后发现根本原因：

**赛题评分公式决定了"死记硬背 > 举一反三"。**

```
m_weights = (counts_max - counts_m + counts_min × 0.1) / (counts_max + counts_min × 0.1)
```

样本越少的关系权重越大（趋近 1.0），样本越多的关系权重越小（趋近 0.0）。这意味着：

| 策略 | 常见关系 (低权重) | 罕见关系 (高权重) | 最终得分 |
|------|-------------------|-------------------|----------|
| roberta + 自然语言提示 | 常识推理更准 ✓ | 语义先验"覆盖"了少样本训练信号 ✗ | **变差** |
| bert + 简单拼接 | 保守预测 | 老老实实记住训练样本 ✓ | **更好** |

具体来说三个改动各自的问题：

- **自然语言前缀 (`Subject:` / `Object:`)**：这些词对所有样本都一样，不提供判别信息，反而消耗 token 预算。对于罕见关系（如 `parity quantum number`、`tussenvoegsel`），模型的强语义先验会被激活并"纠正"训练信号，导致预测成语义相近但错误的类别。
- **Focal Loss (gamma=2.0)**：罕见关系样本在几轮后被模型记住时，`pt → 1`，Focal Loss 将其梯度压到几乎为 0，模型停止从这些关键样本学习。而普通加权 CE 会持续给罕见样本高梯度，确保彻底记牢。
- **roberta-base**：预训练语义知识更丰富 → 对少样本的"先验偏见"更强 → 不如 bert-base-uncased 愿意老老实实拟合少量样本。

**核心教训：这个赛题要的不是语义理解，而是对罕见类的精确记忆。越"聪明"的模型反而越难教。**

---

### 3.1 v3 改进方案：简单即可靠

当前代码（v3）回到 bert-base-uncased，保留经过验证有效的改进，去掉适得其反的"优化"。

#### 改进 1：Train/Infer 统一使用 Text-Pair 编码（无前缀）

**原来 (v1 baseline)：** 训练用 `"{Subject} [SEP] {Object}"` 单文本，推理用 text-pair `(Subject, Object)`。两者的 segment embedding 不同——训练学到的特征和推理时输入不匹配。

**v2 (失败)：** text-pair + `"Subject: ..."` / `"Object: ..."` 前缀。前缀增加了无用 token，干扰了罕见类的记忆。

**v3 (当前)：** 统一使用 text-pair 编码，传原始 Subject 和 Object，不加任何前缀：

```python
# train.py & infer.py — 完全一致
input_ids, mask = encode_text_pair(tokenizer, subject, object, max_length)
```

→ `[CLS] Jobcenter [SEP] Enzo Cormann [SEP]`

- segment embedding 明确区分 Subject (seg=0) 和 Object (seg=1)
- 不加任何多余 token，让模型直接关注实体对本身
- 训练和推理严格一致，消除分布偏移

#### 改进 2：保留加权 CrossEntropyLoss（不用 Focal Loss）

```python
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
```

- `class_weights_tensor` 使用赛题公式计算，罕见类权重高
- 不用 Focal Loss——罕见样本被记住后仍需高梯度持续强化，不能在 `pt` 变高后降低学习强度

#### 改进 3：少样本过采样

```python
raw_train_df = oversample_rare_classes(raw_train_df, min_count=10, max_multiplier=10)
```

- 样本数 < 10 的类别按比例重复（1条→10倍，2条→5倍，3条→3倍）
- 让罕见类在每轮训练中被多次看到，加速记忆

#### 改进 4：保持 bert-base-uncased

- 不做模型升级。bert-base-uncased 语义先验适中，愿意拟合少样本数据
- 如需更大的模型，用 `--shortcut_name` 手动指定即可

#### 改进 5：默认参数保持与原版一致

v2 改了大量默认参数（lr、dropout、weight_decay、patience 等）。v3 全部恢复原版默认值，仅在用户显式传参时生效：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `shortcut_name` | bert-base-uncased | 不前升级模型 |
| `lr` | 5e-5 | 原版学习率 |
| `max_length` | 128 | 原版长度 |
| `dropout` | 0.1 | 原版 dropout |
| `weight_decay` | 0.0 | 原版无衰减 |
| `patience` | 3 | 原版早停 |

---

## 4. 运行命令（v3）

训练命令与 baseline 完全兼容，没有任何必填的新参数：

```bash
# 标准训练
python baseline/train.py --epoch 20 --batch_size 32 --train_dir ./dataset/Train_Set

# 加速训练（推荐）
python baseline/train.py --epoch 20 --batch_size 64 --train_dir ./dataset/Train_Set --use_amp

# 推理（注意 labels_path 指向训练生成的 label_classes.txt）
python baseline/infer.py \
    --input_csv ./dataset/test.csv \
    --labels_path ./cpa_output/cpa_20260523_191304/label_classes.txt \
    --model_path ./cpa_output/cpa_20260523_191304/best_model.pdparams \
    --output_file ./submission.csv
```

---

## 5. 继续上分的后续方向

### 5.1 模型融合（最稳）

用不同 seed 训练 3 个模型，推理时对 logits 取平均再 argmax：

```bash
python baseline/train.py --random_seed 42 --train_dir ./dataset/Train_Set
python baseline/train.py --random_seed 123 --train_dir ./dataset/Train_Set
python baseline/train.py --random_seed 456 --train_dir ./dataset/Train_Set
```

修改 infer.py 加载多个模型做软投票，通常稳定提升 1~3%。

### 5.2 更激进的少样本增强

当前过采样是简单重复。可以尝试：
- **实体替换**：对罕见关系，从其他关系的 Subject/Object 池中随机替换一侧实体，生成"伪样本"。
- **回译增强**：用翻译模型将 Subject 和 Object 分别翻译到其他语言再译回，保持关系不变的同时增加表达多样性。

### 5.3 两阶段训练

1. 第一阶段：在全量数据上正常训练，让模型学会通用特征
2. 第二阶段：冻结 encoder，仅用罕见关系数据微调 classifier（或降低 encoder 学习率 100 倍）

这样可以避免罕见类的梯度被常见类"淹没"。

### 5.4 对抗训练 (FGM)

在训练循环中加入 FGM 对抗扰动，增强模型对输入微小变化的鲁棒性。实现简单，对少样本泛化有帮助。

### 5.5 更大的 bert 变体（谨慎尝试）

- `bert-large-uncased`：参数量更大，但需要更多显存，且更强的先验可能再次伤害少样本性能
- 建议先跑通融合方案，确认有效后再尝试大模型
