# R依赖包安装脚本

# 基础分析包
install.packages(c(
  "tidyverse",  # 数据处理与可视化
  "lubridate",  # 日期时间处理
  "zoo",        # 时间序列处理
  "xts",        # 扩展时间序列
  "tseries",    # 时间序列分析
  "forecast",   # 预测方法
  "urca",       # 单位根和协整分析
  "vars"        # 向量自回归模型
))

# 计量经济学包
install.packages(c(
  "lmtest",     # 线性模型测试
  "car",        # 回归分析附属函数
  "sandwich",   # 稳健标准误
  "plm",        # 面板数据模型
  "nlme",       # 非线性混合效应模型
  "lme4",       # 线性混合效应模型
  "multiwayvcov", # 多维度聚类标准误
  "fixest",     # 高效固定效应估计
  "did",        # 差分法包
  "lpirfs",     # 局部投影脉冲响应函数
  "marginaleffects", # 边际效应计算
  "estimatr",   # 稳健标准误估计
  "rdrobust",   # 回归断点设计
  "hdm",        # 高维计量经济学方法
  "panelvar"    # 面板向量自回归
))

# 可视化包
install.packages(c(
  "ggplot2",    # 图形语法
  "ggthemes",   # ggplot2主题
  "scales",     # 尺度与坐标变换
  "gridExtra",  # 网格图形安排
  "patchwork",  # 组合ggplot图形
  "corrplot",   # 相关图
  "viridis",    # 色盲友好调色板
  "RColorBrewer", # 颜色渐变方案
  "ggridges",   # 岭线图
  "ggrepel",    # 避免文本重叠
  "gganimate"   # 图形动画
))

# 表格输出包
install.packages(c(
  "stargazer",  # 回归表格输出
  "xtable",     # LaTeX表格
  "knitr",      # 报告生成
  "kableExtra", # 表格美化
  "gt",         # 美观表格
  "flextable",  # 灵活表格
  "modelsummary", # 模型汇总
  "gtsummary"   # 描述性统计表格
))

# 数据输入输出
install.packages(c(
  "readxl",     # 读取Excel
  "writexl",    # 写入Excel
  "haven",      # 读取SAS/STATA/SPSS
  "openxlsx",   # 读写Excel
  "data.table", # 高效数据处理
  "arrow",      # Apache Arrow集成
  "jsonlite",   # JSON处理
  "readr"       # 高效读取文本文件
))

# 并行计算
install.packages(c(
  "parallel",   # 并行计算
  "doParallel", # 并行后端
  "foreach",    # 并行迭代
  "future",     # 统一的并行框架
  "furrr"       # 结合purrr和future的并行处理
))

# 保证金调整研究专用包
install.packages(c(
  "moments",    # 高阶矩计算
  "rugarch",    # GARCH模型
  "rmgarch",    # 多元GARCH模型
  "cvar",       # 条件风险价值
  "PerformanceAnalytics", # 金融绩效分析
  "quantmod",   # 量化金融建模
  "roll",       # 滚动统计量
  "riskmetrics", # 风险度量
  "changepoint" # 变点检测
))

print("所有R包安装完成！")
