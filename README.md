# 期货保证金波动率动态研究项目

本项目研究期货市场保证金调整对市场波动率和价格动态的影响，采用局部投影脉冲响应函数(LP-IRF)和事件研究方法进行分析。

## 🆕 新增功能：事件研究方法

本项目现已实现完整的事件研究方法框架，作为LP-IRF分析的重要补充：

### 主要特性
- **完整的事件研究流程**：事件识别、正常收益率估计、异常收益率计算、统计检验
- **多种正常收益率模型**：市场模型、均值调整模型、市场调整模型
- **稳健性检验**：替代波动率指标、不同阈值、排除极端事件、品种分组分析
- **比较分析**：LP-IRF与事件研究结果的系统性比较
- **丰富的可视化**：AAR/CAAR时间序列图、事件分布图、比较分析图

### 核心优势
- **互补验证**：两种方法的一致性结果增强研究结论可信度
- **时间维度**：事件研究关注短期影响(±10天)，LP-IRF关注中期影响(0-10期)
- **方法论多样性**：不同的识别策略和统计框架提供更全面的分析视角

## 项目结构

```
futures_margin_volatility_dynamics/
├── data/
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后的数据
├── src/
│   ├── analysis/               # 分析模块
│   │   ├── lp_irf_analysis.py         # LP-IRF分析
│   │   ├── event_study_analysis.py    # 事件研究分析 🆕
│   │   ├── comparative_analysis.py    # 比较分析 🆕
│   │   └── run_event_study.py         # 事件研究主执行脚本 🆕
│   ├── data_processing/        # 数据处理
│   ├── robustness/            # 稳健性检验
│   ├── utils/                 # 工具函数
│   └── visualization/         # 可视化
│       ├── plot_lp_irf_results.py     # LP-IRF结果可视化
│       └── plot_event_study_results.py # 事件研究结果可视化 🆕
├── output/
│   ├── tables/                # 结果表格
│   ├── figures/               # 图表
│   └── logs/                  # 日志文件
├── docs/
│   └── event_study_methodology.md     # 事件研究方法文档 🆕
└── test_event_study.py        # 事件研究测试脚本 🆕
```

## 快速开始

### 1. 环境配置

确保已安装所需依赖：
```bash
pip install -r requirements.txt
```

### 2. 数据准备

将原始数据文件放置在 `data/raw/` 目录下，然后运行数据处理：
```bash
python src/data_processing/build_features.py
```

### 3. 运行分析

#### 选项A：运行完整的事件研究分析 🆕
```bash
python src/analysis/run_event_study.py
```

这将执行：
- 主要事件研究分析
- 稳健性检验（替代指标、不同阈值、排除极端事件、品种分组）
- 与LP-IRF结果的比较分析
- 自动生成所有图表和报告

#### 选项B：运行LP-IRF分析
```bash
python src/analysis/lp_irf_analysis.py
```

#### 选项C：测试事件研究功能 🆕
```bash
python test_event_study.py
```

### 4. 查看结果

分析完成后，结果将保存在：
- **表格数据**：`output/tables/`
- **图表**：`output/figures/`
- **分析报告**：`output/event_study_analysis_report.md` 🆕

## 事件研究方法详解

### 方法论框架

1. **事件识别**
   - 保证金增加事件：`dlog_margin_rate > threshold`
   - 保证金减少事件：`dlog_margin_rate < -threshold`
   - 自动过滤重叠事件（最小间隔30天）

2. **事件窗口设定**
   - 事件窗口：[-10, +10] 天
   - 估计窗口：120天（与事件窗口间隔5天）
   - 最小估计观测：60个

3. **正常收益率模型**
   - **市场模型**（默认）：`E(R_it) = α_i + β_i * R_mt`
   - **均值调整模型**：`E(R_it) = μ_i`
   - **市场调整模型**：`E(R_it) = R_mt`

4. **异常收益率计算**
   - 异常收益率：`AR_it = R_it - E(R_it)`
   - 累积异常收益率：`CAR_i(t1,t2) = Σ AR_it`
   - 平均异常收益率：`AAR_t = (1/N) * Σ AR_it`

5. **统计检验**
   - t统计量检验
   - 95%置信区间
   - 显著性标记

### 主要输出文件

#### 数据表格 (`output/tables/`)
- `event_study_aar_increase.csv` - 保证金增加事件AAR结果
- `event_study_aar_decrease.csv` - 保证金减少事件AAR结果
- `event_study_events_*.csv` - 识别的事件列表
- `event_study_car_*.csv` - 详细CAR数据
- `lp_irf_vs_event_study_correlation_*.csv` - 相关性分析结果

#### 图表文件 (`output/figures/`)
- `event_study_aar_*.png` - AAR时间序列图
- `event_study_distribution_*.png` - 事件分布图
- `event_study_summary.png` - 结果汇总图
- `lp_irf_vs_event_study_*.png` - 方法比较图

### 稳健性检验

项目实现了多种稳健性检验：

1. **替代波动率指标**：使用Parkinson波动率
2. **不同事件阈值**：更严格的识别标准
3. **排除极端事件**：移除最极端1%的调整
4. **品种分组分析**：按期货品种分别分析

每种稳健性检验都会生成相应的结果文件（带特定后缀）。

## 比较分析

### LP-IRF vs 事件研究

| 维度 | LP-IRF | 事件研究 |
|------|--------|----------|
| 时间范围 | 中期 (0-10期) | 短期 (±10天) |
| 基准设定 | 控制变量 | 历史正常收益率 |
| 因果识别 | 强 | 中等 |
| 计算复杂度 | 高 | 中等 |
| 解释直观性 | 中等 | 高 |

### 互补性分析

1. **验证性**：两种方法结果一致性增强结论可信度
2. **时间维度**：短期和中期影响的完整图景
3. **稳健性**：不同方法论框架下的一致发现

## 配置参数

主要参数在 `src/config.py` 中配置：

```python
# 事件研究参数
EVENT_STUDY_PRE_WINDOW = 10          # 事件前窗口
EVENT_STUDY_POST_WINDOW = 10         # 事件后窗口
EVENT_STUDY_ESTIMATION_WINDOW = 120  # 估计窗口长度
EVENT_STUDY_GAP_DAYS = 5             # 窗口间隔
EVENT_STUDY_MIN_ESTIMATION_OBS = 60  # 最小估计观测数
EVENT_STUDY_MIN_EVENT_GAP = 30       # 事件最小间隔
EVENT_STUDY_NORMAL_RETURN_MODEL = "market_model"  # 正常收益率模型
EVENT_STUDY_SIGNIFICANCE_LEVEL = 0.05  # 显著性水平
```

## 技术特点

### 数据处理
- 自动处理缺失值和异常值
- 支持不平衡面板数据
- 灵活的事件筛选机制

### 统计方法
- 多种正常收益率模型
- 标准化异常收益率
- 稳健的统计检验

### 可视化
- 中文字体支持
- 专业的学术图表风格
- 自动化批量生成

### 扩展性
- 模块化设计
- 支持不同结果变量
- 灵活的参数配置

## 使用示例

### 基本使用

```python
from src.analysis.event_study_analysis import run_event_study_analysis_core
import pandas as pd

# 加载数据
df = pd.read_parquet('data/processed/panel_data_features.parquet')

# 运行事件研究
results = run_event_study_analysis_core(
    data=df,
    outcome_var='log_gk_volatility',
    output_table_dir='output/tables',
    threshold=1e-6
)
```

### 自定义分析

```python
from src.analysis.event_study_analysis import EventStudyAnalyzer

# 创建分析器
analyzer = EventStudyAnalyzer(
    data=df,
    outcome_var='log_gk_volatility',
    event_var='dlog_margin_rate'
)

# 执行分析
results = analyzer.run_event_study(threshold=1e-6)
```

### 可视化生成

```python
from src.visualization.plot_event_study_results import generate_event_study_plots

generate_event_study_plots(
    results_dict=results,
    outcome_var='log_gk_volatility',
    output_dir='output/figures'
)
```

## 注意事项

1. **数据质量**：确保日期列为datetime格式，保证金数据有足够变异
2. **参数选择**：根据市场特征调整事件窗口和估计窗口长度
3. **结果解释**：事件研究为描述性分析，需结合LP-IRF进行因果推断
4. **计算资源**：大数据集可能需要较长计算时间

## 故障排除

### 常见问题

1. **无事件识别**：检查阈值设置和数据质量
2. **估计窗口不足**：调整`EVENT_STUDY_MIN_ESTIMATION_OBS`参数
3. **内存不足**：使用数据子集或增加系统内存
4. **图表显示问题**：确保中文字体正确安装

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。
