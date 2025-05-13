# 保证金调整对期货市场波动率影响的双重差分分析

本项目使用Callaway & Sant'Anna (2021)方法，分析所有期间保证金调整对期货市场波动率的影响。

## 主要文件说明

1. **数据处理脚本**:
   - `did_analysis_improved_part1.R`: 数据加载和基本处理
   - `did_analysis_improved_part2.R`: 保证金率缺失值处理和保证金调整事件识别
   - `did_analysis_improved_part3.R`: 波动率计算和其他分析变量创建
   - `did_analysis_improved_part4_fixed.R`: DID分析变量创建和人工对照组创建
   - `did_analysis_improved_part5_with_plots.R`: Callaway & Sant'Anna DID分析和结果可视化

2. **运行脚本**:
   - `run_analysis.R`: 一键运行完整分析流程

3. **输出目录**:
   - `output/did_analysis_improved/`: 所有分析结果
   - `output/did_analysis_improved/tables/`: 表格结果
   - `output/did_analysis_improved/plots/`: 图表结果
     - `dynamic_effects.png`: 动态处理效应图
     - `volatility_distribution.png`: 波动率分布图

## 主要发现

分析发现保证金调整对期货市场波动率有轻微的正向影响，即保证金调整后波动率略有增加。处理前的平均对数波动率为-4.60，处理后上升到-4.48，增加了0.12。

动态处理效应分析显示，保证金调整后的不同时期可能存在不同的影响模式，部分期间显示显著的效应。

## 使用说明

1. 确保已安装必要的R包：dplyr, tidyr, ggplot2, data.table, lubridate, did, zoo等。
2. 运行`./run_analysis.R`执行完整分析。
3. 结果将保存在`output/did_analysis_improved/`目录下，包括表格和图表。

## 技术细节

- **方法**: Callaway & Sant'Anna (2021) 的双重差分方法
- **控制组策略**: notyettreated（由于所有合约最终都被处理，创建了人工对照组）
- **控制变量**: 滞后交易量、滞后持仓量、滞后收益率
- **图表**: 动态处理效应图和波动率分布图

## 附加说明

本分析重点关注所有期间的保证金调整，而不限于2014年春节期间。分析使用了全量数据，包括在2010年至2024年间的所有保证金调整事件。

由于数据结构特性（所有合约最终都被处理），分析创建了人工对照组以实现Callaway & Sant'Anna方法的应用。

## 参考文献

Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
