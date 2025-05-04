#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具函数模块
提供在整个项目中可重用的辅助功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path

from ..config import LOGS_DIR, RANDOM_SEED

# 设置随机种子，确保结果可重复
np.random.seed(RANDOM_SEED)

# 配置日志
def setup_logger(name, log_file=None, level=logging.INFO):
    """设置项目日志记录器"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"{name}_{timestamp}.log"
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 日期处理函数
def is_trading_day(date, holidays_df):
    """检查日期是否为交易日"""
    # 如果是周末，不是交易日
    if date.weekday() >= 5:
        return False
    
    # 如果是假期，不是交易日
    if date in holidays_df['holiday_date'].values:
        return False
        
    return True

def get_trading_days(start_date, end_date, holidays_df):
    """获取两个日期之间的所有交易日"""
    trading_days = []
    current_date = start_date
    
    while current_date <= end_date:
        if is_trading_day(current_date, holidays_df):
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    return pd.DataFrame({'trading_date': trading_days})

def get_next_trading_day(date, holidays_df, n=1):
    """获取给定日期后的第n个交易日"""
    current_date = date + timedelta(days=1)
    count = 0
    
    while count < n:
        if is_trading_day(current_date, holidays_df):
            count += 1
            if count == n:
                return current_date
        current_date += timedelta(days=1)
    
    return current_date

# 数据处理函数
def calculate_returns(prices, method='log'):
    """计算收益率"""
    if method == 'log':
        return np.log(prices / prices.shift(1))
    else:  # simple returns
        return prices / prices.shift(1) - 1

def calculate_volatility(returns, window=20):
    """计算波动率"""
    return returns.rolling(window=window).std() * np.sqrt(252)  # 年化

def calculate_realized_volatility(returns, window=20):
    """计算已实现波动率"""
    return np.sqrt((returns ** 2).rolling(window=window).sum() * (252 / window))

def winsorize_series(series, limits=(0.01, 0.01)):
    """Winsorize系列数据，处理极端值"""
    lower_limit = series.quantile(limits[0])
    upper_limit = series.quantile(1 - limits[1])
    
    return pd.Series([
        lower_limit if x < lower_limit else (upper_limit if x > upper_limit else x)
        for x in series
    ], index=series.index)

# 图表函数
def save_figure(fig, filename, dpi=300, format='png'):
    """保存图表"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 保存图表
    fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
    plt.close(fig)

# 统计函数
def compute_t_stat(series):
    """计算t统计量"""
    return series.mean() / (series.std() / np.sqrt(len(series)))

def newey_west_se(series, lags=5):
    """使用Newey-West方法计算标准误"""
    from statsmodels.stats.sandwich_covariance import cov_hac
    
    # 将系列数据转换为数组形式
    X = np.ones((len(series), 1))
    y = np.array(series)
    
    # 估计参数和残差
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X.dot(beta)
    
    # 计算Newey-West协方差矩阵
    cov = cov_hac(X, resid, nlags=lags)
    
    # 返回标准误
    return np.sqrt(np.diag(cov))
