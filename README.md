# 期货保证金调整与价格波动动态关系研究

## 项目概述
本项目旨在研究期货保证金调整对市场波动性的影响，采用差分法(DID)和局部投影(LP-IRF)等方法分析保证金变动前后的期货价格行为特征。

## 目录结构
- `data/`: 存放原始数据和处理后的数据
  - `raw/`: 原始数据文件
  - `processed/`: 处理后的数据文件
- `notebooks/`: Jupyter笔记本，用于数据探索和原型设计
- `src/`: 源代码目录
  - `data_processing/`: 数据处理相关代码
  - `analysis/`: 分析模型相关代码
  - `robustness/`: 稳健性检验相关代码
  - `visualization/`: 数据可视化和表格生成代码
  - `utils/`: 通用工具函数
- `output/`: 输出结果目录
  - `figures/`: 图形输出
  - `tables/`: 表格输出
  - `logs/`: 日志文件

## 环境设置

### Python环境配置

本项目使用Python 3.9版本，选择该版本是因为其稳定性和与关键包的良好兼容性。我们已经创建并配置了一个名为`futures_analysis`的conda环境，其中包含了所有必要的依赖包。

#### 方法1：使用已创建的Conda环境（推荐）

```bash
# 激活已创建的环境
conda activate futures_analysis

# 验证环境
python -c "import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; import scipy as sp; import statsmodels.api as sm; import linearmodels as lm; import arch; print('环境设置成功！')"
```

#### 方法2：在新机器上重新创建环境

```bash
# 使用environment.yml创建新环境
conda env create -f environment.yml
conda activate futures_analysis
```

#### 方法3：使用pip安装依赖（不推荐）

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### R环境配置

本项目的某些分析也可以使用R完成，特别是对于面板数据差分法分析和局部投影方法。

```bash
# 安装R包
Rscript R_requirements.R
```

### 已安装的核心依赖包

我们的`futures_analysis`环境包含以下主要组件，所有版本均经过精心选择以确保兼容性和稳定性：

- **Python版本**: 3.9.18
- **数据处理**: 
  - numpy 1.24.3
  - pandas 2.0.3 
  - scipy 1.11.4
  - pyarrow 20.0.0
  
- **统计与计量经济学**: 
  - statsmodels 0.14.4
  - linearmodels 6.1
  - arch 7.2.0 (GARCH模型)
  - scikit-learn 1.3.0
  
- **数据获取与处理**:
  - pandas-datareader 0.10.0
  - openpyxl 3.1.5
  - xlrd 2.0.1
  
- **可视化**:
  - matplotlib 3.7.2
  - seaborn 0.12.2
  - plotly 6.0.1
  
- **并行计算与效率**:
  - dask 2024.8.0
  - joblib 1.4.2
  - tqdm 4.67.1
  
- **交互式开发**:
  - jupyter 1.0.0
  - jupyterlab 4.3.4
  - ipython 8.15.0

## 使用指南

### 数据准备

1. 将原始期货数据文件放入 `data/raw/` 目录
2. 将节假日和保证金调整公告日期数据放入 `data/raw/` 目录

### 运行分析

#### 完整分析流程

```bash
# 激活环境
conda activate futures_analysis

# 运行完整分析
./main.sh
```

#### 分步骤运行

1. **数据探索与处理**:
   ```bash
   # 启动JupyterLab进行交互式探索
   jupyter lab notebooks/01_data_exploration.ipynb
   
   # 或启动Jupyter Notebook
   jupyter notebook notebooks/01_data_exploration.ipynb
   
   # 处理数据
   python -m src.data_processing.build_features
   ```

2. **差分法(DID)分析**:
   ```bash
   python -m src.analysis.did_cs_analysis
   ```

3. **局部投影(LP-IRF)分析**:
   ```bash
   python -m src.analysis.lp_irf_analysis
   ```

4. **稳健性检验**:
   ```bash
   python -m src.robustness.run_robustness
   ```

5. **生成结果图表**:
   ```bash
   python -m src.visualization.plot_did_results
   python -m src.visualization.plot_lp_irf_results
   python -m src.visualization.generate_tables
   ```

### 环境维护

- **添加新包**:
   ```bash
   conda activate futures_analysis
   conda install 包名
   # 或
   pip install 包名
   ```

- **环境导出**（便于在其他机器上复制）:
   ```bash
   conda env export -n futures_analysis > environment_full.yml
   ```

### 可配置参数

主要配置参数位于 `src/config.py`，可以调整以下设置：

- 差分法(DID)分析的窗口期和控制组选择方法
- 局部投影(LP-IRF)分析的预测期数和自回归滞后阶数
- 稳健性检验的自助法迭代次数
- 输出图表的格式和分辨率

## 分析流程

1. 数据探索与清洗
   - 处理缺失值、异常值
   - 初步的描述性统计

2. 特征工程与变量构建
   - 构建波动率指标
   - 识别保证金调整事件窗口

3. 差分法(DID)与横截面分析
   - 定义处理组和控制组
   - 估计处理效应

4. 局部投影脉冲响应函数(LP-IRF)分析
   - 估计保证金调整的动态效应
   - 分析效应持续时间

5. 稳健性检验
   - 安慰剂测试
   - 不同窗口期的敏感性分析

6. 可视化与结果呈现
   - 生成图表
   - 编写报告

## 贡献者
[您的姓名]
