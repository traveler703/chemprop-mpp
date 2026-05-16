# 下一步计划：面向 Research Report 的项目增强

## 目标

老师要求最终提交 **research report 或 PPT 二选一**。既然我们组不做 PPT，就应该把项目改造成一份 **10 页以内的研究报告**。

当前项目已经有：

- 基于 Chemprop MPNN 的 ESOL 水溶解度预测
- 数据规模实验：20%、50%、80%、100%
- 较好的结果：RMSE 约 0.59，R2 约 0.92
- 图表、分子案例和实验结果文件

接下来最值得做的增强有三个：

1. 加一个更大的数据集：Lipophilicity
2. 加一个传统机器学习 baseline：RDKit 描述符 + Random Forest
3. 加一个更专业的泛化实验：random split vs scaffold split

这三个增强能让报告从“项目复现记录”变成“有研究问题、有实验对比、有科学分析的小型 AI for Science 研究”。

---

## 报告核心问题

建议把报告的核心研究问题写成：

> 分子图神经网络在不同科学数据条件下，能否稳定预测分子性质？数据规模、分子性质类型、模型范式和数据划分方式如何影响预测性能？

英文版可写：

> How does molecular graph learning perform under different scientific data conditions, including data scale, molecular property type, model paradigm, and chemical generalization split?

最终报告不要只写：

> 我们训练了 Chemprop 来预测 ESOL 溶解度。

而要写成：

> 本研究以分子性质预测为 AI for Science 场景，比较 Chemprop MPNN 与传统机器学习 baseline 在不同数据规模、不同分子性质任务和不同划分方式下的表现。

---

## 1. 加更大的数据集：Lipophilicity

### 为什么值得做

ESOL 很适合入门，但它只有约 1,128 个分子，作为唯一数据集略显单薄。

Lipophilicity 更适合作为扩展数据集，因为：

- 它也是 MoleculeNet 里的经典分子性质预测数据集。
- 它约有 4,200 个分子，比 ESOL 大，但仍然能在个人电脑上训练。
- 预测目标是脂溶性/logD，和药物发现密切相关。
- 它和水溶解度天然形成对照：一个偏“水溶性”，一个偏“脂溶性”，都属于药物 ADMET 性质相关任务。

### 报告里怎么写

可以把它写成：

> To evaluate whether the model is limited to a single small dataset, we further include the MoleculeNet Lipophilicity dataset, which contains a larger number of compounds and measures a different but related ADMET property.

中文版：

> 为了避免实验只停留在单一小数据集上，本文进一步引入 MoleculeNet Lipophilicity 数据集。该数据集规模更大，预测目标为脂溶性/logD，能够检验模型在不同分子性质任务上的适用性。

### 实现任务

- 扩展 `scripts/prepare_data.py`，支持准备两种数据集：
  - `esol`
  - `lipophilicity`
- 保存 Lipophilicity 数据文件：
  - `data/lipo.csv`
  - `data/lipo_train.csv`
  - `data/lipo_test.csv`
  - `data/lipo_train_20.csv`
  - `data/lipo_train_50.csv`
  - `data/lipo_train_80.csv`
  - `data/lipo_train_100.csv`
- 统一数据格式，每个数据集都保留两列：
  - `smiles`
  - `target`
- 修改训练脚本，不要再把目标列硬编码成 `logS`。

### 推荐数据来源

使用 MoleculeNet/DeepChem 公开的 Lipophilicity CSV：

```text
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv
```

通常有用的列：

- `smiles`
- `exp` 或类似的实验脂溶性目标列

### 预期结果表

报告中放一个数据集说明表：

| 数据集 | 任务 | 分子数量 | 预测目标 | 指标 |
|---|---|---:|---|---|
| ESOL | 水溶解度预测 | 约 1,128 | LogS | RMSE, MAE, R2 |
| Lipophilicity | 脂溶性预测 | 约 4,200 | logD | RMSE, MAE, R2 |

---

## 2. 加 Random Forest Baseline

### 为什么值得做

现在 Chemprop 的结果不错，但没有 baseline。报告里如果只写一个模型，容易显得像“只跑通了一个框架”。

Random Forest baseline 可以回答这个问题：

> Chemprop MPNN 相比传统机器学习模型到底有什么优势？

### Baseline 设计

使用：

```text
RDKit 分子描述符 + Random Forest Regressor
```

每个分子的流程：

- 用 RDKit 解析 SMILES
- 计算分子描述符
- 训练 `sklearn.ensemble.RandomForestRegressor`
- 在和 Chemprop 相同的训练/测试划分上评估

### 推荐描述符

先用简单、可解释的描述符即可：

- 分子量
- MolLogP
- TPSA
- 氢键供体数量
- 氢键受体数量
- 可旋转键数量
- 环数量
- Fraction Csp3

这些描述符的好处是报告里容易解释，不会只剩一串黑箱指标。

### 实现任务

新建脚本：

```text
scripts/baseline_rf.py
```

脚本支持：

```bash
python scripts/baseline_rf.py --dataset esol
python scripts/baseline_rf.py --dataset lipo
```

输出：

```text
results/rf_esol_results.json
results/rf_lipo_results.json
results/model_comparison.png
```

每个结果 JSON 里应包含：

- 数据集名称
- 训练集/测试集样本数
- RMSE
- MAE
- R2
- feature importance

### 报告里怎么写

可以这样写方法对比：

> The Random Forest baseline uses hand-crafted molecular descriptors computed by RDKit, while Chemprop directly learns molecular representations from graph structures. This comparison allows us to examine whether learned graph representations provide advantages over traditional descriptor-based machine learning.

中文版：

> Random Forest baseline 基于 RDKit 人工分子描述符，而 Chemprop MPNN 直接从分子图结构中学习表示。两者对比可以检验自动学习的图表示是否优于传统手工特征方法。

### 预期结果表

报告中放一个模型对比表：

| 数据集 | 模型 | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| ESOL | Random Forest | 待实验 | 待实验 | 待实验 |
| ESOL | Chemprop MPNN | 0.59 | 0.41 | 0.92 |
| Lipophilicity | Random Forest | 待实验 | 待实验 | 待实验 |
| Lipophilicity | Chemprop MPNN | 待实验 | 待实验 | 待实验 |

### 报告分析重点

无论 Chemprop 是否明显超过 Random Forest，都可以分析：

- Chemprop 的优势：自动从分子图结构中学习表示。
- Random Forest 的优势：训练快、可解释性强、对小数据集友好。
- 如果 Chemprop 提升有限，可以解释为：ESOL 数据集较小，传统描述符已经包含很多与溶解度相关的信息。

---

## 3. 加 Random Split vs Scaffold Split

### 为什么值得做

这是最像研究实验的增强点。

Random split 可能比较容易，因为相似分子可能同时出现在训练集和测试集中。Scaffold split 更难，因为它按照化学骨架划分分子，更接近真实药物发现中遇到新结构的情况。

这个实验能让报告从“模型性能展示”升级到“模型泛化能力分析”。

### 科学表述

报告里可以直接写：

> 随机划分更像在相似分子之间插值，骨架划分更接近真实药物发现中面对新化学结构的泛化问题。

英文版：

> Random split tests interpolation among similar molecules, while scaffold split tests generalization to new chemical structures.

### 实现任务

给 `prepare_data.py` 增加 scaffold split。

支持两种 split：

```bash
python scripts/prepare_data.py --dataset esol --split random
python scripts/prepare_data.py --dataset esol --split scaffold
python scripts/prepare_data.py --dataset lipo --split random
python scripts/prepare_data.py --dataset lipo --split scaffold
```

输出文件名包含 split 类型：

```text
data/esol_random_train.csv
data/esol_random_test.csv
data/esol_scaffold_train.csv
data/esol_scaffold_test.csv
data/lipo_random_train.csv
data/lipo_random_test.csv
data/lipo_scaffold_train.csv
data/lipo_scaffold_test.csv
```

使用 RDKit 的 Bemis-Murcko scaffold：

```python
from rdkit.Chem.Scaffolds import MurckoScaffold
```

### 预期结果

Scaffold split 的指标大概率会比 random split 差。这不是坏事。

这反而是一个很好的讨论点：

> 模型在结构相似的分子上表现很好，但面对新化学骨架时性能下降。这说明科学机器学习模型的可靠性受到数据覆盖范围和化学多样性的限制。

### 报告图表

报告中可以放一张柱状图：

```text
Random Split vs Scaffold Split 下的 RMSE 对比
```

图里至少包含：

- ESOL + Chemprop
- ESOL + Random Forest
- 如果时间允许，再加 Lipophilicity

---

## 推荐实现顺序

### Step 1：统一数据格式

先把脚本改成统一使用：

```text
smiles,target
```

而不是：

```text
smiles,logS
```

需要改的文件：

- `scripts/prepare_data.py`
- `scripts/train.py`
- `scripts/data_scale_experiment.py`
- `scripts/visualize.py`
- `scripts/molecule_cases.py`

如果可以，保留对旧 `logS` 列的兼容，但后续新数据集统一用 `target`。

### Step 2：加入 Lipophilicity 数据集

数据格式统一后，再加入：

```bash
python scripts/prepare_data.py --dataset lipo
python scripts/train.py --dataset lipo
```

需要验证：

- 数据能正确下载和解析
- Chemprop 能训练
- 能生成结果 JSON
- 能生成对应图表

### Step 3：加入 Random Forest baseline

实现：

```text
scripts/baseline_rf.py
```

运行：

```bash
python scripts/baseline_rf.py --dataset esol
python scripts/baseline_rf.py --dataset lipo
```

然后生成一张模型对比图：

```text
Chemprop MPNN vs Random Forest
```

### Step 4：加入 Scaffold Split

最后再加 scaffold split。

先跑四个关键实验：

```bash
python scripts/train.py --dataset esol --split random
python scripts/train.py --dataset esol --split scaffold
python scripts/train.py --dataset lipo --split random
python scripts/train.py --dataset lipo --split scaffold
```

如果时间足够，再给 Random Forest 也跑 split 对比：

```bash
python scripts/baseline_rf.py --dataset esol --split random
python scripts/baseline_rf.py --dataset esol --split scaffold
python scripts/baseline_rf.py --dataset lipo --split random
python scripts/baseline_rf.py --dataset lipo --split scaffold
```

---

## Research Report 建议结构

报告控制在 10 页以内，建议结构如下：

### 1. Introduction

内容：

- AI for Science 背景
- 分子性质预测在药物发现中的意义
- 本文研究问题

建议长度：0.8-1 页。

### 2. Dataset

内容：

- ESOL 数据集说明
- Lipophilicity 数据集说明
- SMILES 和目标值介绍
- random split 与 scaffold split 说明

建议长度：1-1.5 页。

### 3. Methods

内容：

- Chemprop MPNN
- RDKit descriptors + Random Forest baseline
- 评价指标：RMSE、MAE、R2
- 实验设置：数据规模实验、模型对比、划分方式对比

建议长度：1.5-2 页。

### 4. Experiments and Results

内容：

- ESOL Chemprop 主结果
- 数据规模实验
- Chemprop vs Random Forest
- ESOL vs Lipophilicity
- Random split vs Scaffold split

建议长度：2.5-3 页。

### 5. Discussion

内容：

- 为什么数据规模增加会提升性能
- 为什么 80% 到 100% 可能出现平台期
- Chemprop 和 Random Forest 的优劣
- Scaffold split 为什么更难
- 失败案例和数据分布限制

建议长度：1.5-2 页。

### 6. Conclusion

内容：

- 总结主要发现
- 说明 AI for Science 中数据质量、结构和物理/化学先验的重要性
- 未来工作：更多 ADMET 数据集、更复杂模型、更严格外部测试集

建议长度：0.5 页。

### References

内容：

- Chemprop 论文
- MoleculeNet 论文
- ESOL / Delaney 论文
- Lipophilicity 数据集来源

建议长度：0.5-1 页。

---

## 报告图表清单

建议最终报告放 5-6 张图/表，不要堆太多。

必放：

1. 数据集对比表：ESOL vs Lipophilicity
2. Chemprop 主结果图：Predicted vs Measured
3. 数据规模实验图：RMSE vs training data size
4. 模型对比表：Chemprop vs Random Forest
5. 划分方式对比图：Random split vs Scaffold split

可选：

6. 分子案例表：Aspirin、Caffeine、Glucose 等
7. Random Forest feature importance 图
8. 误差分布图

如果页数紧张，优先保留：

```text
数据规模图 + 模型对比表 + split 对比图
```

这三类最能体现“数据挖掘实验”的价值。

