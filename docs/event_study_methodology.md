# 事件研究方法实现文档

## 概述

本文档详细说明了为期货保证金波动率动态研究项目实现的事件研究方法。事件研究方法作为局部投影脉冲响应函数(LP-IRF)分析的补充，提供了另一个角度来理解保证金调整对期货市场动态的影响。

## 方法论框架

### 1. 事件研究基本原理

事件研究方法通过比较事件发生前后的实际收益率与正常收益率，来衡量特定事件对资产价格的影响。在本研究中：

- **事件**: 期货合约保证金率的调整
- **结果变量**: 期货合约的波动率（主要使用Garman-Klass波动率）
- **事件窗口**: 事件发生前后的时间区间 [-10, +10] 天
- **估计窗口**: 用于估计正常收益率的时间区间（事件前120天，与事件窗口间隔5天）

### 2. 事件识别

#### 2.1 事件定义
- **保证金增加事件**: `dlog_margin_rate > threshold` (默认 1e-6)
- **保证金减少事件**: `dlog_margin_rate < -threshold`

#### 2.2 事件筛选规则
- 同一合约内事件间隔至少30天，避免重叠效应
- 如果事件重叠，选择影响幅度更大的事件
- 估计窗口至少需要60个有效观测值

### 3. 正常收益率模型

实现了三种正常收益率模型：

#### 3.1 市场模型 (Market Model) - 默认
```
E(R_it) = α_i + β_i * R_mt + ε_it
```
其中：
- `R_it`: 合约i在时间t的收益率
- `R_mt`: 市场收益率（所有合约的平均收益率）
- `α_i, β_i`: 通过估计窗口的OLS回归估计

#### 3.2 均值调整模型 (Mean-Adjusted Model)
```
E(R_it) = μ_i
```
其中 `μ_i` 是合约i在估计窗口的平均收益率

#### 3.3 市场调整模型 (Market-Adjusted Model)
```
E(R_it) = R_mt
```
直接使用市场收益率作为预期收益率

### 4. 异常收益率计算

#### 4.1 异常收益率 (Abnormal Returns, AR)
```
AR_it = R_it - E(R_it)
```

#### 4.2 标准化异常收益率 (Standardized AR)
```
SAR_it = AR_it / σ_i
```
其中 `σ_i` 是估计窗口的残差标准差

#### 4.3 累积异常收益率 (Cumulative AR, CAR)
```
CAR_i(t1,t2) = Σ(t=t1 to t2) AR_it
```

### 5. 统计检验

#### 5.1 平均异常收益率 (Average AR, AAR)
```
AAR_t = (1/N) * Σ(i=1 to N) AR_it
```

#### 5.2 t统计量
```
t_AAR = AAR_t / SE(AAR_t)
```
其中 `SE(AAR_t) = σ_AAR / √N`

#### 5.3 累积平均异常收益率 (CAAR)
```
CAAR(t1,t2) = Σ(t=t1 to t2) AAR_t
```

## 实现架构

### 核心类和函数

#### 1. `EventStudyAnalyzer` 类
- **位置**: `src/analysis/event_study_analysis.py`
- **功能**: 实现完整的事件研究分析流程
- **主要方法**:
  - `identify_events()`: 事件识别和筛选
  - `calculate_normal_returns()`: 正常收益率模型估计
  - `calculate_abnormal_returns()`: 异常收益率计算
  - `calculate_cumulative_abnormal_returns()`: CAR计算
  - `calculate_average_abnormal_returns()`: AAR和CAAR计算

#### 2. 可视化模块
- **位置**: `src/visualization/plot_event_study_results.py`
- **功能**: 生成事件研究结果图表
- **主要图表**:
  - AAR时间序列图
  - CAAR时间序列图
  - 事件分布图
  - 个别事件CAR图

#### 3. 比较分析模块
- **位置**: `src/analysis/comparative_analysis.py`
- **功能**: LP-IRF与事件研究结果比较
- **主要分析**:
  - 相关性分析
  - 系数比较图
  - 统计显著性比较

### 配置参数

在 `src/config.py` 中新增的事件研究参数：

```python
# 事件窗口定义
EVENT_STUDY_PRE_WINDOW = 10    # 事件前窗口期
EVENT_STUDY_POST_WINDOW = 10   # 事件后窗口期
EVENT_STUDY_ESTIMATION_WINDOW = 120  # 估计窗口期长度
EVENT_STUDY_GAP_DAYS = 5       # 估计窗口与事件窗口间隔

# 数据质量要求
EVENT_STUDY_MIN_ESTIMATION_OBS = 60  # 估计窗口最少观测数
EVENT_STUDY_MIN_EVENT_GAP = 30       # 事件间最小间隔

# 模型和检验参数
EVENT_STUDY_NORMAL_RETURN_MODEL = "market_model"  # 正常收益率模型
EVENT_STUDY_SIGNIFICANCE_LEVEL = 0.05  # 显著性水平
EVENT_STUDY_BOOTSTRAP_ITERATIONS = 1000  # Bootstrap迭代次数
```

## 使用方法

### 1. 运行完整分析
```bash
python src/analysis/run_event_study.py
```

### 2. 单独运行事件研究
```python
from src.analysis.event_study_analysis import run_event_study_analysis_core

results = run_event_study_analysis_core(
    data=your_data,
    outcome_var='log_gk_volatility',
    output_table_dir='output/tables',
    output_suffix=""
)
```

### 3. 生成可视化图表
```python
from src.visualization.plot_event_study_results import generate_event_study_plots

generate_event_study_plots(
    results_dict=results,
    outcome_var='log_gk_volatility',
    output_dir='output/figures'
)
```

### 4. 比较分析
```python
from src.analysis.comparative_analysis import run_comparative_analysis_core

comparison_results = run_comparative_analysis_core(
    lp_irf_results_dir='output/tables',
    event_study_results_dir='output/tables',
    output_dir='output/figures'
)
```

## 输出文件

### 1. 数据表格 (保存在 `output/tables/`)
- `event_study_aar_increase.csv`: 保证金增加事件的AAR结果
- `event_study_aar_decrease.csv`: 保证金减少事件的AAR结果
- `event_study_events_increase.csv`: 保证金增加事件列表
- `event_study_events_decrease.csv`: 保证金减少事件列表
- `event_study_car_increase.csv`: 保证金增加事件的详细CAR数据
- `event_study_car_decrease.csv`: 保证金减少事件的详细CAR数据

### 2. 图表文件 (保存在 `output/figures/`)
- `event_study_aar_increase.png`: 保证金增加事件AAR时间序列图
- `event_study_aar_decrease.png`: 保证金减少事件AAR时间序列图
- `event_study_distribution_increase.png`: 保证金增加事件分布图
- `event_study_distribution_decrease.png`: 保证金减少事件分布图
- `event_study_summary.png`: 事件研究结果汇总图
- `lp_irf_vs_event_study_increase.png`: LP-IRF与事件研究比较图（增加）
- `lp_irf_vs_event_study_decrease.png`: LP-IRF与事件研究比较图（减少）

### 3. 比较分析表格
- `lp_irf_vs_event_study_correlation_increase.csv`: 相关性分析结果（增加）
- `lp_irf_vs_event_study_correlation_decrease.csv`: 相关性分析结果（减少）

## 稳健性检验

实现了多种稳健性检验：

### 1. 替代波动率指标
- 使用Parkinson波动率替代Garman-Klass波动率
- 后缀: `_parkinson_vol`

### 2. 不同事件识别阈值
- 使用更严格的阈值 (1e-4) 识别事件
- 后缀: `_strict_threshold`

### 3. 排除极端事件
- 排除最极端的1%保证金调整事件
- 后缀: `_filtered_extreme`

### 4. 按品种分组分析
- 对主要期货品种分别进行分析
- 后缀: `_variety_{品种名}`

## 与LP-IRF方法的比较

### 相似点
1. 都研究保证金调整对波动率的影响
2. 都区分保证金增加和减少事件
3. 都提供统计显著性检验

### 差异点
1. **时间范围**: 事件研究关注短期影响（±10天），LP-IRF关注中期影响（0-10期）
2. **基准设定**: 事件研究使用历史数据估计正常收益率，LP-IRF使用控制变量
3. **因果识别**: LP-IRF更适合因果推断，事件研究更适合描述性分析
4. **数据要求**: 事件研究需要足够的估计窗口，LP-IRF对数据结构要求更灵活

### 互补性
1. **验证性**: 两种方法结果的一致性增强研究结论的可信度
2. **时间维度**: 结合短期和中期影响的完整图景
3. **稳健性**: 不同方法论框架下的一致发现增强结果稳健性

## 技术特点

### 1. 数据处理
- 自动处理缺失值和异常值
- 支持不平衡面板数据
- 灵活的事件筛选机制

### 2. 统计方法
- 多种正常收益率模型选择
- 标准化异常收益率计算
- Bootstrap方法支持（预留接口）

### 3. 可视化
- 中文字体支持
- 交互式图表设计
- 自动化批量生成

### 4. 扩展性
- 模块化设计便于扩展
- 支持不同结果变量
- 灵活的参数配置

## 注意事项

### 1. 数据质量要求
- 确保日期列为datetime格式
- 保证金率变化数据需要足够的变异
- 估计窗口需要足够的观测值

### 2. 参数选择
- 事件窗口长度应根据市场特征调整
- 估计窗口长度影响正常收益率模型稳定性
- 事件识别阈值影响事件数量和质量

### 3. 解释注意
- 事件研究结果为描述性，不能直接推断因果关系
- 需要结合LP-IRF结果进行综合分析
- 注意市场环境变化对结果的影响

## 未来扩展方向

1. **多元事件研究**: 同时考虑多个相关事件
2. **非参数方法**: 实现基于排序的非参数检验
3. **动态事件窗口**: 根据事件特征调整窗口长度
4. **机器学习方法**: 使用ML方法改进正常收益率预测
5. **高频数据**: 扩展到日内高频数据分析
