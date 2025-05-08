# -*- coding: utf-8 -*-
"""
可视化 DID 分析结果

加载 Callaway & Sant'Anna DID 分析的聚合结果，并生成事件研究图。
包含可被其他脚本调用的核心绘图函数 plot_did_core。
"""

import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- 重构后的核心绘图函数 ---

def plot_did_core(results_filepath, output_filepath, title_suffix=""):
    """
    生成并保存 DID 事件研究图的核心逻辑。

    Args:
        results_filepath (str): 包含聚合 DID 结果的 CSV 文件路径。
                                (预期列: event_time, estimate, conf.low, conf.high 或其变体)
        output_filepath (str): 保存图表的文件路径 (如 .png)。
        title_suffix (str, optional): 添加到图表标题末尾的后缀。Defaults to "".
    """
    logging.info(f"开始生成 DID 事件研究图 (Suffix: '{title_suffix}'), 数据来源: {results_filepath}")

    # --- 1. 加载结果数据 ---
    try:
        results_df = pd.read_csv(results_filepath)
        if results_df.empty or results_df.columns.size == 0:
            logging.warning(f"DID 结果文件 {results_filepath} 为空或无表头，跳过绘图。")
            return True
        logging.info(f"已加载 DID 结果数据: {results_df.shape}")
        logging.info(f"结果文件列名: {results_df.columns.tolist()}")
    except FileNotFoundError:
        logging.error(f"DID 结果文件未找到: {results_filepath}")
        return False # 返回失败状态
    except Exception as e:
        logging.error(f"加载 DID 结果 {results_filepath} 时出错: {e}")
        return False

    # --- 2. 检查必需的列 ---
    required_cols = {'event_time': ['event_time', 'time', 't', 'relative_time', 'term'],
                     'estimate': ['estimate', 'att', 'coef', 'point.estimate'],
                     'conf.low': ['conf.low', 'ci.low', 'lower'],
                     'conf.high': ['conf.high', 'ci.high', 'upper']}

    col_mapping = {}
    missing_found = False
    for standard_name, possible_names in required_cols.items():
        found = False
        for name in possible_names:
            if name in results_df.columns:
                col_mapping[standard_name] = name
                found = True
                logging.info(f"找到列 '{name}' 对应标准名 '{standard_name}'")
                break
        if not found:
            logging.error(f"结果文件 {results_filepath} 中缺少必需的列: {standard_name} (尝试过: {possible_names})")
            missing_found = True

    if missing_found:
        return False

    # 重命名为标准列名
    results_df_renamed = results_df.rename(columns={v: k for k, v in col_mapping.items()})

    # 确保 event_time 是数值类型
    try:
        results_df_renamed['event_time'] = pd.to_numeric(results_df_renamed['event_time'])
    except ValueError as e:
        logging.error(f"无法将事件时间列 '{col_mapping['event_time']}' 转换为数值: {e}")
        return False

    # 按事件时间排序
    results_df_renamed = results_df_renamed.sort_values(by='event_time')

    # --- 3. 创建图表 ---
    logging.info("创建事件研究图...")
    try:
        plt.style.use(config.PLOT_STYLE)
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制效应估计值
        ax.plot(results_df_renamed['event_time'], results_df_renamed['estimate'], marker='o', linestyle='-', label='ATT Estimate')

        # 绘制置信区间阴影
        ax.fill_between(results_df_renamed['event_time'], results_df_renamed['conf.low'], results_df_renamed['conf.high'],
                        alpha=0.2, label='95% Confidence Interval')

        # 添加参考线
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(-1, color='red', linestyle=':', linewidth=0.8, label='Pre-Treatment Period End')

        # 设置坐标轴标签和标题
        ax.set_xlabel("Event Time Relative to Holiday Adjustment")
        # 尝试从文件名或参数推断 outcome var? 暂时硬编码
        ax.set_ylabel("Estimated Average Treatment Effect (ATT)") # 更通用
        title = f"Dynamic Effect of Holiday Margin Adjustments (C&S DID){title_suffix}"
        ax.set_title(title)

        # 设置 x 轴范围
        min_time = results_df_renamed['event_time'].min()
        max_time = results_df_renamed['event_time'].max()
        plot_min = min(min_time, -config.DID_EVENT_WINDOW_PRE) if pd.notna(min_time) else -config.DID_EVENT_WINDOW_PRE
        plot_max = max(max_time, config.DID_EVENT_WINDOW_POST) if pd.notna(max_time) else config.DID_EVENT_WINDOW_POST
        ax.set_xlim(plot_min -1 , plot_max + 1)

        ax.legend()
        ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')

        # --- 4. 保存图表 ---
        plt.tight_layout()
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        fig.savefig(output_filepath, dpi=300)
        logging.info(f"事件研究图已保存到: {output_filepath}")
        plt.close(fig)
        return True # 返回成功状态

    except Exception as e:
        logging.error(f"创建或保存 DID 图表 {output_filepath} 时出错: {e}")
        return False


# --- 主执行块 (用于直接运行此脚本) ---
if __name__ == "__main__":
    logging.info("直接运行 plot_did_results.py 脚本...")

    # 查找默认的聚合结果文件
    did_agg_results_filename = "did_aggregate_results.csv"
    did_agg_results_path = os.path.join(config.PATH_OUTPUT_TABLES, did_agg_results_filename)

    output_figure_filename = "did_event_study_plot.png"
    output_figure_path = os.path.join(config.PATH_OUTPUT_FIGURES, output_figure_filename)

    if os.path.exists(did_agg_results_path):
        logging.info(f"找到默认聚合结果文件: {did_agg_results_path}")
        success = plot_did_core(results_filepath=did_agg_results_path,
                                output_filepath=output_figure_path,
                                title_suffix="") # 主分析无后缀
        if success:
            logging.info("默认 DID 事件研究图生成成功。")
        else:
            logging.error("默认 DID 事件研究图生成失败。")
    else:
        logging.error(f"未找到默认 DID 聚合结果文件: {did_agg_results_path}")
        logging.error("请先运行 DID 分析脚本 (did_cs_analysis.py) 并确保其生成了聚合结果。")
