# -*- coding: utf-8 -*-
"""
生成描述性统计表格

加载处理后的数据，计算关键变量的描述性统计量，并保存为表格。
"""

import pandas as pd
import numpy as np
import os
import logging

# --- 项目配置导入 ---
try:
    from src import config
except ImportError:
    import sys
    PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PACKAGE_DIR)
    from src import config

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 主函数 ---

def generate_descriptive_stats_table():
    """生成并保存描述性统计表格"""
    logging.info("开始生成描述性统计表格...")

    # --- 1. 加载处理好的数据 ---
    try:
        df = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        logging.info(f"已加载处理好的面板数据: {df.shape}")
    except FileNotFoundError:
        logging.error(f"处理后的面板数据文件未找到: {config.PANEL_DATA_FILEPATH}")
        logging.error("请先运行 src/data_processing/build_features.py")
        return
    except Exception as e:
        logging.error(f"加载处理后数据时出错: {e}")
        return

    # --- 2. 选择用于描述性统计的变量 ---
    # 包含因变量、关键自变量、状态变量、控制变量等
    # 注意：只选择数值型变量进行 describe()
    desc_vars = [
        # 因变量
        'log_gk_volatility',
        # 关键自变量 (冲击)
        'dlog_margin_rate',
        'margin_increase_shock',
        'margin_decrease_shock',
        # 状态变量 (哑变量，计算均值即为比例)
        # 查找所有 market_regime_* 和 volatility_regime_* 列
    ] + [col for col in df.columns if col.startswith('market_regime_')] \
      + [col for col in df.columns if col.startswith('volatility_regime_')] \
      + [
        'State_HolidayAdjust', 'State_NonHolidayAdjust', # t 时刻
        # 控制变量 (t-1)
    ] + config.CONTROL_VARIABLES + [
        # 其他相关变量
        'turnover_rate', # t 时刻流动性代理
        'limit_hit_dummy', # t 时刻涨跌停状态
        'return' # t 时刻收益率
    ]


    # 确保选择的变量存在于 DataFrame 中，并去重
    desc_vars_exist = sorted(list(set(v for v in desc_vars if v in df.columns)))

    missing_vars = set(desc_vars) - set(desc_vars_exist)
    if missing_vars:
        # 过滤掉 config.CONTROL_VARIABLES 中可能不存在的变量的警告
        config_missing = missing_vars.intersection(set(config.CONTROL_VARIABLES))
        other_missing = missing_vars - config_missing
        if other_missing:
             logging.warning(f"以下变量未在数据中找到，将从描述性统计中排除: {sorted(list(other_missing))}")
        if config_missing:
             logging.debug(f"配置文件中指定的控制变量未找到: {sorted(list(config_missing))}")


    if not desc_vars_exist:
        logging.error("没有可用于描述性统计的变量。")
        return

    df_desc = df[desc_vars_exist]

    # --- 3. 计算描述性统计量 ---
    logging.info(f"为以下变量计算描述性统计量: {desc_vars_exist}")
    try:
        # 使用 describe() 计算主要统计量
        stats_table = df_desc.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).transpose() # 添加更多分位数
        # describe() 结果包含 count, mean, std, min, 25%, 50%, 75%, max

        # 格式化输出 (可选，例如保留特定小数位数)
        stats_table = stats_table.round(4) # 保留4位小数

        logging.info("描述性统计计算完成。")
        # print("\n描述性统计结果预览:") # 在实际运行时取消注释
        # print(stats_table)

    except Exception as e:
        logging.error(f"计算描述性统计时出错: {e}")
        return

    # --- 4. 保存表格 ---
    output_filename = "descriptive_statistics.csv"
    output_path = os.path.join(config.PATH_OUTPUT_TABLES, output_filename)
    try:
        stats_table.to_csv(output_path, index=True) # index=True 保留变量名作为索引列
        logging.info(f"描述性统计表格已保存到: {output_path}")
    except Exception as e:
        logging.error(f"保存描述性统计表格时出错: {e}")

# --- 主执行块 ---
if __name__ == "__main__":
    generate_descriptive_stats_table()