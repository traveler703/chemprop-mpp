# Chemprop Workshop: Molecular Property Prediction for Drug Discovery

基于 Chemprop MPNN 的分子水溶解度预测 — AI for Science Workshop

## 核心问题

给定一个分子的 SMILES 结构，能否用图神经网络（MPNN）预测它的水溶解度（LogS）？

## 项目结构

```
workshop/
├── scripts/
│   ├── setup_env.bat / .sh       # Conda 环境安装脚本
│   ├── prepare_data.py           # 下载并准备 ESOL 数据集
│   ├── train.py                  # 训练 Chemprop MPNN 模型
│   ├── data_scale_experiment.py  # 数据规模实验 (20/50/80/100%)
│   ├── visualize.py              # 生成实验图表
│   ├── molecule_cases.py         # 分子案例预测
│   ├── create_ppt.py             # 生成 Workshop PPT
│   └── run_all.py                # 一键运行完整流程
├── notebooks/
│   └── demo.ipynb                # 交互式 Demo（输入 SMILES → 预测溶解度）
├── data/                         # ESOL 数据集（运行 prepare_data.py 后生成）
├── models/                       # 训练好的模型（运行后生成）
├── results/                      # 图表和实验结果（运行后生成）
└── ppt/                          # Workshop PPT（运行后生成）
```

## 快速开始

### 1. 环境配置（一次性）

打开 Anaconda Prompt，进入项目目录：

```bash
cd E:\College\3down\dataAM\workshop

# Windows
scripts\setup_env.bat

# 或 Linux/Mac
bash scripts/setup_env.sh
```

然后激活环境：

```bash
conda activate chemprop-workshop
```

验证安装：

```bash
python -c "import chemprop; import torch; import rdkit; print('OK')"
```

### 2. 一键运行完整流程

```bash
python scripts/run_all.py
```

这会依次执行：数据准备 → 训练 ESOL 和 Lipophilicity baseline → 双数据集数据规模实验 → 生成双数据集图表 → 创建 PPT。

如果想加速（跳过数据规模实验，省约 20-40 分钟）：

```bash
python scripts/run_all.py --skip-scales
```

### 3. 分步运行

```bash
# Step 1: 准备数据（ESOL + Lipophilicity）
python scripts/prepare_data.py

# Step 2: 训练 baseline 模型（ESOL + Lipophilicity）
python scripts/train.py

# Step 3: 数据规模实验（ESOL + Lipophilicity，可选，耗时较长）
python scripts/data_scale_experiment.py

# Step 4: 分子案例分析（ESOL + Lipophilicity）
python scripts/molecule_cases.py

# Step 5: 生成所有图表（ESOL + Lipophilicity）
python scripts/visualize.py

# Step 6: 生成 PPT
python scripts/create_ppt.py
```

### 4. 运行 Demo Notebook

```bash
cd notebooks
jupyter notebook demo.ipynb
```

在 Notebook 中可以：
- 输入任意 SMILES，选择 ESOL 或 Lipophilicity 模型进行预测
- 查看分子结构图
- 查看已知药物分子的 LogS / logD 预测案例
- 查看 ESOL 和 Lipophilicity 的实验图表

## 实验设计

### Baseline
- 模型: Chemprop MPNN (hidden=300, depth=5, dropout=0.1)
- 数据: ESOL 水溶解度数据集 (1,128 molecules) 与 MoleculeNet Lipophilicity 数据集 (4,200 molecules)
- 指标: RMSE, MAE, R²
- 当前 baseline 结果:
  - ESOL: RMSE=0.584, MAE=0.427, R²=0.928
  - Lipophilicity: RMSE=0.542, MAE=0.399, R²=0.801

### 数据规模实验
- 分别在 ESOL 和 Lipophilicity 上使用 20%, 50%, 80%, 100% 训练数据
- 固定测试集
- 每个规模 3 次重复 → 误差棒
- 展示 RMSE 和 R² 随训练数据量增加的变化

## 数据集

ESOL (Delaney) 水溶解度数据集：
- 1,128 个有机小分子
- SMILES + 实测 LogS (log₁₀ mol/L)
- 来源: Delaney, J.S. *J. Chem. Inf. Comput. Sci.* 2004, 44, 1000-1005

Lipophilicity 脂溶性数据集：
- 4,200 个药物样小分子
- SMILES + 实测 logD
- 来源: MoleculeNet / DeepChem Lipophilicity 公开数据集
- 作用: 与 ESOL 形成 ADMET 性质对照，一个关注水溶性，一个关注脂溶性

## 原始项目与参考文献

本 workshop 项目基于 Chemprop 官方开源框架和公开 ESOL 数据集实现，核心思想是使用 message passing neural network (MPNN) 从分子图结构中学习分子性质。

### 原始 GitHub / 官方资源

- Chemprop 官方 GitHub: https://github.com/chemprop/chemprop
- Chemprop 官方组织主页: https://github.com/chemprop
- Chemprop 文档: https://chemprop.readthedocs.io/

### 相关论文

- Yang, K.; Swanson, K.; Jin, W.; et al. **Analyzing Learned Molecular Representations for Property Prediction.** *Journal of Chemical Information and Modeling*, 2019, 59(8), 3370-3388. DOI: https://doi.org/10.1021/acs.jcim.9b00237
- Heid, E.; Greenman, K. P.; Chung, Y.; et al. **Chemprop: A Machine Learning Package for Chemical Property Prediction.** *Journal of Chemical Information and Modeling*, 2024, 64(1), 9-17. DOI: https://doi.org/10.1021/acs.jcim.3c01250
- **Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical Property Prediction.** *Journal of Chemical Information and Modeling*, 2026. DOI: https://doi.org/10.1021/acs.jcim.5c02332
- Delaney, J. S. **ESOL: Estimating Aqueous Solubility Directly from Molecular Structure.** *Journal of Chemical Information and Computer Sciences*, 2004, 44(3), 1000-1005. DOI: https://doi.org/10.1021/ci034243x

## 依赖

- Python 3.10+
- PyTorch >= 2.0 (CUDA 可选但推荐)
- Chemprop v2
- RDKit
- pandas, numpy, matplotlib, seaborn, scikit-learn
- jupyter
- python-pptx (用于生成 PPT)

## PPT 结构（7 页）

1. 标题与研究问题
2. AI for Science 背景：为什么分子性质预测重要
3. 数据与方法：ESOL 数据集 + Chemprop MPNN
4. 实验设计：训练流程 + 数据规模实验
5. 结果：预测 vs 真实散点图 + 误差分布
6. 数据规模分析：RMSE vs 训练数据量
7. 结论：科学数据 + 分子结构 + GNN 如何加速药物发现
