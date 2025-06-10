# 期货保证金波动动态研究

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Academic-orange.svg)](https://github.com)

本项目是一个关于期货保证金调整对市场波动率动态影响的学术研究项目。采用局部投影脉冲响应函数（LP-IRF）和事件研究方法，分析中国期货市场中保证金政策变化的市场效应。

## 项目概述

### 研究目的
本研究旨在分析期货保证金率调整对市场波动率的动态影响，探索：
- 保证金调整的短期和中期市场效应
- 不同市场状态下政策效果的异质性
- 保证金增加与减少的非对称影响
- 不同期货品种的差异化响应

### 研究方法
1. **局部投影脉冲响应函数（LP-IRF）**：分析保证金冲击的动态传导机制
2. **事件研究方法**：测量保证金调整事件的短期异常收益率
3. **状态依赖分析**：考虑市场状态和波动率状态的调节效应
4. **稳健性检验**：多种替代指标和方法验证结果可靠性

### 主要发现
- 保证金调整对波动率具有显著的动态影响
- 保证金增加和减少呈现非对称效应
- 市场状态显著影响政策传导效果
- 不同期货品种存在异质性响应

## 快速开始

### 环境要求
- Python 3.8+
- 内存：建议8GB以上
- 存储：至少2GB可用空间

### 安装依赖
```bash
# 克隆项目
git clone <repository-url>
cd futures_margin_volatility_dynamics

# 安装Python依赖
pip install -r requirements.txt

# 或使用conda环境
conda env create -f environment.yml
conda activate futures_margin_analysis
```

### 运行分析
```bash
# 完整分析流程（推荐）
./main.sh

# 快速模式（跳过传统稳健性检验）
./main.sh --quick

# 查看帮助信息
./main.sh --help
```

### 单步运行
```bash
# 1. 数据处理
python src/data_processing/build_features.py

# 2. LP-IRF分析
python src/analysis/lp_irf_analysis.py

# 3. 事件研究分析
python src/analysis/run_event_study.py

# 4. 生成可视化图表
python src/visualization/plot_lp_irf_results.py
```

## 项目结构

```
futures_margin_volatility_dynamics/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── environment.yml             # Conda环境配置
├── main.sh                     # 主执行脚本
├── QUICK_START_GUIDE.md        # 快速开始指南
│
├── src/                        # 源代码目录
│   ├── __init__.py
│   ├── config.py               # 全局配置参数
│   ├── analysis/               # 分析模块
│   │   ├── __init__.py
│   │   ├── lp_irf_analysis.py          # LP-IRF分析
│   │   ├── event_study_analysis.py     # 事件研究分析
│   │   ├── run_event_study.py          # 事件研究执行脚本
│   │   └── comparative_analysis.py     # 比较分析
│   ├── data_processing/        # 数据处理模块
│   │   ├── __init__.py
│   │   └── build_features.py           # 特征构建
│   ├── visualization/          # 可视化模块
│   │   ├── __init__.py
│   │   ├── plot_lp_irf_results.py      # LP-IRF结果可视化
│   │   ├── plot_event_study_results.py # 事件研究结果可视化
│   │   └── generate_tables.py          # 描述性统计表格
│   ├── robustness/            # 稳健性检验模块
│   │   ├── __init__.py
│   │   └── run_robustness.py           # 稳健性检验
│   └── utils/                 # 工具函数
│       ├── __init__.py
│       └── helpers.py                  # 辅助函数
│
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   │   └── futures_margin_data.csv     # 期货保证金数据
│   └── processed/             # 处理后数据
│       ├── panel_data_features.parquet         # 面板数据特征
│       └── panel_data_features_cleaned.parquet # 清洗后数据
│
├── output/                    # 输出结果目录
│   ├── tables/               # 结果表格
│   │   ├── descriptive_statistics.csv         # 描述性统计
│   │   ├── lp_irf_results_baseline.csv        # LP-IRF基准结果
│   │   ├── event_study_aar_increase.csv       # 事件研究AAR结果（增加）
│   │   ├── event_study_aar_decrease.csv       # 事件研究AAR结果（减少）
│   │   └── ...                                # 其他结果表格
│   ├── figures/              # 结果图表
│   │   ├── lp_irf_baseline_increase.png       # LP-IRF基准图（增加）
│   │   ├── lp_irf_baseline_decrease.png       # LP-IRF基准图（减少）
│   │   ├── event_study_aar_increase.png       # 事件研究AAR图（增加）
│   │   ├── event_study_aar_decrease.png       # 事件研究AAR图（减少）
│   │   └── ...                                # 其他图表文件
│   ├── logs/                 # 日志文件
│   └── event_study_analysis_report.md         # 分析报告
│
└── docs/                     # 文档目录
    └── event_study_methodology.md             # 事件研究方法文档
```

## 数据描述

### 数据来源
本研究使用的数据包含中国期货市场的历史交易数据和保证金调整记录，时间跨度为2010年1月至2024年12月。

### 主要变量
| 变量名 | 描述 | 数据类型 |
|--------|------|----------|
| `date` | 交易日期 | datetime |
| `variety` | 期货品种代码 | string |
| `margin_rate` | 保证金率 | float |
| `close_price` | 收盘价 | float |
| `trading_volume` | 成交量 | float |
| `open_interest` | 持仓量 | float |
| `high_price` | 最高价 | float |
| `low_price` | 最低价 | float |

### 数据样本（前5行）
```csv
date,variety,margin_rate,close_price,trading_volume,open_interest,high_price,low_price
2010-01-04,C.DCE,5.0,4057.0,176352.0,284296.0,1900.0,1875.0
2010-01-04,C.DCE,5.0,4126.0,176352.0,92.0,1900.0,1875.0
2010-01-04,C.DCE,5.0,3009.0,176352.0,237164.0,1900.0,1875.0
2010-01-04,C.DCE,5.0,1892.0,176352.0,217210.0,1900.0,1875.0
2010-01-04,C.DCE,5.0,3085.7,176352.0,237164.0,1900.0,1875.0
```

### 数据统计特征
- **总观测数**：148,895个合约-日观测值
- **期货品种**：涵盖农产品、金属、能源等主要品种
- **保证金调整事件**：保证金增加事件122个，减少事件117个
- **数据完整性**：经过严格的数据清洗和质量控制

## 使用指南

### 配置参数
主要参数在 `src/config.py` 中配置：

```python
# 分析时间范围
ANALYSIS_START_DATE = "2010-01-01"
ANALYSIS_END_DATE = "2024-12-31"

# LP-IRF参数
LP_HORIZON = 10                    # 脉冲响应期数
LP_CONTROL_LAGS = 1               # 控制变量滞后阶数

# 事件研究参数
EVENT_STUDY_PRE_WINDOW = 10       # 事件前窗口期
EVENT_STUDY_POST_WINDOW = 10      # 事件后窗口期
EVENT_STUDY_ESTIMATION_WINDOW = 120  # 估计窗口期长度
```

### 自定义分析
```python
# 导入核心分析模块
from src.analysis.lp_irf_analysis import run_lp_irf_analysis
from src.analysis.event_study_analysis import run_event_study_analysis_core

# 运行LP-IRF分析
lp_results = run_lp_irf_analysis(
    data=your_data,
    outcome_var='log_gk_volatility',
    shock_vars=['margin_increase_shock', 'margin_decrease_shock']
)

# 运行事件研究分析
event_results = run_event_study_analysis_core(
    data=your_data,
    outcome_var='log_gk_volatility',
    output_table_dir='output/tables'
)
```

### 结果解读
1. **LP-IRF结果**：查看 `output/tables/lp_irf_results_baseline.csv`
   - 系数估计值表示保证金冲击对波动率的影响大小
   - t统计量和p值用于判断统计显著性
   - 置信区间提供不确定性度量

2. **事件研究结果**：查看 `output/tables/event_study_aar_*.csv`
   - AAR（平均异常收益率）衡量事件的平均影响
   - CAAR（累积平均异常收益率）衡量累积影响
   - t统计量检验影响的统计显著性

## 主要结果摘要

### LP-IRF分析发现
1. **保证金增加效应**：
   - 对波动率产生显著正向影响，峰值出现在第2-3期
   - 影响持续约6-8期后逐渐消散
   - 在高波动率状态下效应更为显著

2. **保证金减少效应**：
   - 对波动率产生负向影响，但幅度小于增加效应
   - 影响相对持久，持续约8-10期
   - 在牛市状态下效应更为明显

### 事件研究发现
1. **短期市场反应**：
   - 保证金调整在事件日当天即产生显著影响
   - 保证金增加导致异常波动率上升约2-3%
   - 保证金减少的影响相对温和

2. **累积效应**：
   - 事件后10天内累积效应达到峰值
   - 不同期货品种存在显著差异
   - 市场状态显著影响事件效应大小

### 稳健性检验结果
- 使用Parkinson波动率的结果与基准结果高度一致
- 不同事件识别阈值下结论保持稳定
- 排除极端事件后主要发现依然成立
- 分品种分析证实了结果的普遍性

完整依赖列表请参见 `requirements.txt` 文件。

## 问题报告
如发现bug或有功能建议，请通过GitHub Issues提交：
- 详细描述问题或建议
- 提供复现步骤（如适用）
- 包含相关的错误信息和日志

## 许可信息

本项目采用MIT许可证，适合学术研究和教育用途。

```
MIT License

Copyright (c) 2024 期货保证金波动动态研究项目

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 学术引用
如果您在学术研究中使用了本项目，请考虑引用：

```bibtex
@software{futures_margin_volatility_dynamics,
  title={中国期货保证金波动动态研究},
  author={[XXX]},
  year={2024},
  url={https://github.com/willetjiang/futures_margin_volatility_dynamics},
  note={中国期货保证金调整对市场波动率动态影响的实证研究}
}
```
