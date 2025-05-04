#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目配置文件
包含所有项目所需的全局配置参数
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 输出目录
OUTPUT_DIR = ROOT_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
LOGS_DIR = OUTPUT_DIR / "logs"

# 确保目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 数据文件路径
DATA_PATH = RAW_DATA_DIR / "futures_margin_data.csv"

# 处理后的数据文件路径
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_futures_data.csv"
FEATURES_DATA_PATH = PROCESSED_DATA_DIR / "features_data.csv"
EVENT_WINDOWS_PATH = PROCESSED_DATA_DIR / "event_windows.csv"

# 分析参数
# 差分法(DID)分析参数
DID_WINDOW_PRE = 20  # 事件前窗口天数
DID_WINDOW_POST = 20  # 事件后窗口天数
DID_CONTROL_METHOD = "matching"  # 控制组选择方法: "matching" 或 "synthetic"

# 局部投影(LP-IRF)分析参数
LP_HORIZON = 30  # 局部投影的预测期数
LP_LAGS = 5  # 自回归滞后阶数

# 稳健性检验参数
BOOTSTRAP_ITERATIONS = 1000  # 自助法迭代次数
PLACEBO_TESTS = True  # 是否进行安慰剂测试

# 可视化参数
PLOT_DPI = 300  # 图表DPI
FIGURE_FORMAT = "png"  # 图表保存格式: "png", "pdf", "svg"等
PLOT_STYLE = "seaborn-whitegrid"  # 图表样式

# 随机种子，确保结果可重复
RANDOM_SEED = 42
