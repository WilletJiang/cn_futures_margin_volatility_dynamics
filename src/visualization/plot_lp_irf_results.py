# -*- coding: utf-8 -*-
"""
可视化局部投影 (LP-IRF) 分析结果

加载基准和状态依赖的 IRF 结果，并生成相应的图表。
包含可被其他脚本调用的核心绘图函数 plot_lp_irf_core。
"""

import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import itertools # 用于生成颜色和线型

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

# --- 辅助绘图函数 (保持不变) ---

def plot_single_irf(data, title, output_filepath):
    """绘制单个脉冲响应函数图 (带置信区间)"""
    if data.empty:
        logging.warning(f"没有数据用于绘制图表: {title}")
        return

    plt.style.use(config.PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data['horizon'], data['coeff'], marker='o', linestyle='-', label='IRF Estimate')
    ax.fill_between(data['horizon'], data['conf_low'], data['conf_high'],
                    alpha=0.2, label='95% Confidence Interval')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

    ax.set_xlabel("Horizon (Trading Days)")
    # 尝试从标题或其他地方获取 outcome var name? 暂时硬编码
    ax.set_ylabel("Impulse Response") # 更通用
    ax.set_title(title)
    ax.set_xlim(-1, config.LP_HORIZON + 1) # 从 horizon 0 开始
    ax.legend()
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')

    plt.tight_layout()
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        fig.savefig(output_filepath, dpi=300)
        logging.info(f"图表已保存到: {output_filepath}")
    except Exception as e:
        logging.error(f"保存图表 {output_filepath} 时出错: {e}")
    plt.close(fig)


def plot_comparative_irf(data, state_var_prefix, title, output_filepath, label_map=None):
    """绘制状态依赖的脉冲响应函数对比图"""
    if data.empty:
        logging.warning(f"没有数据用于绘制对比图表: {title}")
        return

    plt.style.use(config.PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))

    state_vars = sorted(data['state_variable'].unique())

    colors = sns.color_palette("colorblind", len(state_vars))
    linestyles = ['-', '--', ':', '-.']
    style_cycler = itertools.cycle(zip(colors, linestyles))

    for i, state in enumerate(state_vars):
        state_data = data[data['state_variable'] == state].sort_values('horizon')
        if state_data.empty:
            continue

        color, linestyle = next(style_cycler)
        label = label_map.get(state, state) if label_map else state

        ax.plot(state_data['horizon'], state_data['coeff'], marker='o', markersize=4, linestyle=linestyle, color=color, label=f'{label}')
        ax.fill_between(state_data['horizon'], state_data['conf_low'], state_data['conf_high'],
                        alpha=0.15, color=color)

    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Horizon (Trading Days)")
    ax.set_ylabel("Impulse Response") # 更通用
    ax.set_title(title)
    ax.set_xlim(-1, config.LP_HORIZON + 1)
    ax.legend(title="Conditioning State", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        fig.savefig(output_filepath, dpi=300, bbox_inches='tight')
        logging.info(f"对比图表已保存到: {output_filepath}")
    except Exception as e:
        logging.error(f"保存对比图表 {output_filepath} 时出错: {e}")
    plt.close(fig)


# --- 重构后的核心绘图函数 ---

def plot_lp_irf_core(table_dir, figure_dir, suffix=""):
    """
    加载指定目录和后缀的 LP-IRF 结果并生成所有图表。

    Args:
        table_dir (str): 包含 LP 结果 CSV 文件的目录。
        figure_dir (str): 保存输出图表的目录。
        suffix (str, optional): 要加载和保存的文件名后缀。Defaults to "".
    """
    logging.info(f"--- 开始核心 LP-IRF 可视化 (Suffix: '{suffix}') ---")

    # 确保输出目录存在
    os.makedirs(figure_dir, exist_ok=True)

    # --- 1. 加载基准 IRF 结果 ---
    baseline_results_filename = f"lp_irf_results_baseline{suffix}.csv"
    baseline_results_path = os.path.join(table_dir, baseline_results_filename)
    try:
        baseline_df = pd.read_csv(baseline_results_path)
        logging.info(f"已加载基准 IRF 结果: {baseline_results_path} ({baseline_df.shape})")
    except FileNotFoundError:
        logging.error(f"基准 IRF 结果文件未找到: {baseline_results_path}")
        baseline_df = pd.DataFrame()
    except Exception as e:
        logging.error(f"加载基准 IRF 结果时出错: {e}")
        baseline_df = pd.DataFrame()

    # --- 2. 绘制基准 IRF 图 ---
    if not baseline_df.empty:
        inc_data = baseline_df[baseline_df['shock_type'] == 'increase'].sort_values('horizon')
        plot_single_irf(inc_data,
                        f"Baseline IRF: Response to Margin Increase Shock{suffix.replace('_', ' ').title()}",
                        os.path.join(figure_dir, f"lp_irf_baseline_increase{suffix}.png"))

        dec_data = baseline_df[baseline_df['shock_type'] == 'decrease'].sort_values('horizon')
        plot_single_irf(dec_data,
                        f"Baseline IRF: Response to Margin Decrease Shock{suffix.replace('_', ' ').title()}",
                        os.path.join(figure_dir, f"lp_irf_baseline_decrease{suffix}.png"))
    else:
        logging.warning("跳过绘制基准 IRF 图，因为未加载到数据。")


    # --- 3. 加载状态依赖 IRF 结果 ---
    statedep_results_filename = f"lp_irf_results_statedep{suffix}.csv"
    statedep_results_path = os.path.join(table_dir, statedep_results_filename)
    try:
        statedep_df = pd.read_csv(statedep_results_path)
        logging.info(f"已加载状态依赖 IRF 结果: {statedep_results_path} ({statedep_df.shape})")
    except FileNotFoundError:
        logging.error(f"状态依赖 IRF 结果文件未找到: {statedep_results_path}")
        statedep_df = pd.DataFrame()
    except Exception as e:
        logging.error(f"加载状态依赖 IRF 结果时出错: {e}")
        statedep_df = pd.DataFrame()

    # --- 4. 绘制状态依赖 IRF 对比图 ---
    if not statedep_df.empty:
        state_label_map = {
            'market_regime_Bull': 'Bull Market',
            'market_regime_Bear': 'Bear Market',
            'market_regime_Neutral': 'Neutral Market',
            'volatility_regime_High': 'High Volatility',
            'volatility_regime_Low': 'Low Volatility',
            'State_HolidayAdjust_lag1': 'Holiday Adjustment',
            'State_NonHolidayAdjust_lag1': 'Non-Holiday Adjustment'
        }

        title_suffix_pretty = suffix.replace('_', ' ').title()

        market_regime_data = statedep_df[statedep_df['state_variable'].str.startswith('market_regime_')]
        plot_comparative_irf(market_regime_data, 'market_regime_',
                             f"State-Dependent IRF: By Market Regime (t-1){title_suffix_pretty}",
                             os.path.join(figure_dir, f"lp_irf_statedep_market_regime{suffix}.png"),
                             label_map=state_label_map)

        vol_regime_data = statedep_df[statedep_df['state_variable'].str.startswith('volatility_regime_')]
        plot_comparative_irf(vol_regime_data, 'volatility_regime_',
                             f"State-Dependent IRF: By Volatility Regime (t-1){title_suffix_pretty}",
                             os.path.join(figure_dir, f"lp_irf_statedep_volatility_regime{suffix}.png"),
                             label_map=state_label_map)

        adj_type_data = statedep_df[statedep_df['state_variable'].str.contains('Adjust_lag1')]
        plot_comparative_irf(adj_type_data, 'State_',
                             f"State-Dependent IRF: By Adjustment Type (t-1){title_suffix_pretty}",
                             os.path.join(figure_dir, f"lp_irf_statedep_adjustment_type{suffix}.png"),
                             label_map=state_label_map)
    else:
         logging.warning("跳过绘制状态依赖 IRF 图，因为未加载到数据。")

    logging.info(f"--- 完成核心 LP-IRF 可视化 (Suffix: '{suffix}') ---")


# --- 主执行块 (用于直接运行此脚本) ---
if __name__ == "__main__":
    logging.info("直接运行 plot_lp_irf_results.py 脚本...")
    # 执行默认绘图 (主分析结果)
    plot_lp_irf_core(table_dir=config.PATH_OUTPUT_TABLES,
                     figure_dir=config.PATH_OUTPUT_FIGURES,
                     suffix="")

    # --- 为流动性机制检验结果绘图 ---
    logging.info("为流动性机制检验 (换手率) 的 LP-IRF 结果生成图表...")
    plot_lp_irf_core(table_dir=config.PATH_OUTPUT_TABLES,
                     figure_dir=config.PATH_OUTPUT_FIGURES,
                     suffix="_liquidity_turnover")

    logging.info("为流动性机制检验 (对数成交量) 的 LP-IRF 结果生成图表...")
    plot_lp_irf_core(table_dir=config.PATH_OUTPUT_TABLES,
                     figure_dir=config.PATH_OUTPUT_FIGURES,
                     suffix="_liquidity_logvolume")

    # --- 为排除极端冲击的敏感性分析结果绘图 ---
    logging.info("为排除极端冲击的敏感性分析的 LP-IRF 结果生成图表...")
    plot_lp_irf_core(table_dir=config.PATH_OUTPUT_TABLES,
                     figure_dir=config.PATH_OUTPUT_FIGURES,
                     suffix="_trimmed_shocks")

    # --- 为按品种类型分解的敏感性分析结果绘图 ---
    logging.info("为按品种类型分解的敏感性分析的 LP-IRF 结果生成图表...")
    try:
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
            logging.warning(f"主面板数据文件未找到: {config.PANEL_DATA_FILEPATH}，无法确定品种类型以绘图。")
        else:
            main_df_plot_variety = pd.read_parquet(config.PANEL_DATA_FILEPATH)
            variety_type_col = 'variety' # 更新为 'variety' 以匹配 lp_irf_analysis.py
            if variety_type_col not in main_df_plot_variety.columns:
                logging.warning(f"品种列 '{variety_type_col}' 不在数据中，无法为按品种分解的结果绘图。")
            else:
                unique_varieties_plot = main_df_plot_variety[variety_type_col].unique()
                for variety_plot in unique_varieties_plot:
                    if pd.isna(variety_plot):
                        continue # 跳过缺失
                    
                    df_subset_plot_check = main_df_plot_variety[main_df_plot_variety[variety_type_col] == variety_plot]
                    if df_subset_plot_check.empty or len(df_subset_plot_check) < 100: # 与 lp_irf_analysis.py 中条件一致
                        logging.info(f"品种类型 '{variety_plot}' 数据子集过小，可能未生成分析结果，跳过绘图。")
                        continue

                    safe_variety_suffix_plot = "".join(filter(str.isalnum, str(variety_plot)))
                    if not safe_variety_suffix_plot:
                         safe_variety_suffix_plot = f"unknownvariety_{hash(variety_plot) % 1000}"
                    
                    plot_suffix_variety = f"_vt_{safe_variety_suffix_plot}"
                    
                    # 检查对应的结果文件是否存在，避免对未成功运行的分析尝试绘图
                    baseline_results_filename = f"lp_irf_results_baseline{plot_suffix_variety}.csv"
                    baseline_results_path = os.path.join(config.PATH_OUTPUT_TABLES, baseline_results_filename)
                    statedep_results_filename = f"lp_irf_results_statedep{plot_suffix_variety}.csv"
                    statedep_results_path = os.path.join(config.PATH_OUTPUT_TABLES, statedep_results_filename)

                    if not os.path.exists(baseline_results_path) and not os.path.exists(statedep_results_path):
                        logging.info(f"未找到品种 '{variety_plot}' (后缀 {plot_suffix_variety}) 的LP-IRF结果文件，跳过绘图。")
                        continue

                    logging.info(f"为品种 '{variety_plot}' (后缀 {plot_suffix_variety}) 的 LP-IRF 结果生成图表...")
                    plot_lp_irf_core(table_dir=config.PATH_OUTPUT_TABLES,
                                     figure_dir=config.PATH_OUTPUT_FIGURES,
                                     suffix=plot_suffix_variety)
    except Exception as e:
        logging.error(f"为按品种类型分解的结果绘图时发生错误: {e}")