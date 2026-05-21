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
