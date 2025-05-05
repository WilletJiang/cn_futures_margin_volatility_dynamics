# -*- coding: utf-8 -*-
"""
执行稳健性检验

协调运行研究设计中定义的各种稳健性检验。
"""

import pandas as pd
import numpy as np
import os
import logging
import argparse # 用于命令行参数

# --- 项目配置和分析脚本导入 ---
try:
    from src import config
    # 导入重构后的核心分析和绘图函数
    from src.analysis.did_cs_analysis import run_did_analysis_core
    from src.analysis.lp_irf_analysis import run_lp_analysis_core
    from src.visualization.plot_did_results import plot_did_core as plot_did_results_core # 避免命名冲突
    from src.visualization.plot_lp_irf_results import plot_lp_irf_core as plot_lp_irf_results_core # 避免命名冲突
except ImportError as e:
    import sys
    logging.error(f"导入模块时出错: {e}")
    PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PACKAGE_DIR)
    try:
        from src import config
        from src.analysis.did_cs_analysis import run_did_analysis_core
        from src.analysis.lp_irf_analysis import run_lp_analysis_core
        from src.visualization.plot_did_results import plot_did_core as plot_did_results_core
        from src.visualization.plot_lp_irf_results import plot_lp_irf_core as plot_lp_irf_results_core
    except ImportError as e_inner:
         logging.error(f"尝试从 sys.path 添加后再次导入失败: {e_inner}")
         exit()


# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 确保稳健性输出目录存在 ---
ROBUSTNESS_TABLE_DIR = os.path.join(config.PATH_OUTPUT_TABLES, 'robustness')
ROBUSTNESS_FIGURE_DIR = os.path.join(config.PATH_OUTPUT_FIGURES, 'robustness')
os.makedirs(ROBUSTNESS_TABLE_DIR, exist_ok=True)
os.makedirs(ROBUSTNESS_FIGURE_DIR, exist_ok=True)


# --- 稳健性检验函数 ---

def run_alt_vol_measure(base_data):
    """
    4.1 使用替代波动率度量 (Parkinson) 运行分析
    """
    logging.info("--- 开始稳健性检验: 替代波动率度量 (Parkinson) ---")
    # 从 config 获取替代波动率变量名 (去除 _lag1 后缀得到当期变量)
    alt_outcome_var = config.ALT_VOLATILITY_VAR.replace('_lag1', '')

    if alt_outcome_var not in base_data.columns:
        logging.error(f"替代波动率变量 '{alt_outcome_var}' 不在数据中，无法执行此检验。")
        return

    output_suffix = "_alt_vol" # 文件名后缀
    # 定义此检验特定的输出子目录
    alt_table_dir = os.path.join(ROBUSTNESS_TABLE_DIR, 'alt_volatility')
    alt_figure_dir = os.path.join(ROBUSTNESS_FIGURE_DIR, 'alt_volatility')
    os.makedirs(alt_table_dir, exist_ok=True)
    os.makedirs(alt_figure_dir, exist_ok=True)

    # --- 运行 DID 分析 (使用替代波动率) ---
    logging.info(f"运行 DID 分析 (结果变量: {alt_outcome_var})...")
    did_success = False
    try:
        did_success = run_did_analysis_core(data=base_data.copy(), # 使用副本
                                            outcome_var=alt_outcome_var,
                                            output_table_dir=alt_table_dir,
                                            output_suffix=output_suffix)
        if did_success:
             logging.info("替代波动率 DID 分析核心逻辑执行成功。")
             # 尝试绘图
             did_results_file = os.path.join(alt_table_dir, f"did_aggregate_results{output_suffix}.csv")
             did_plot_file = os.path.join(alt_figure_dir, f"did_event_study_plot{output_suffix}.png")
             if os.path.exists(did_results_file):
                  plot_did_results_core(results_filepath=did_results_file,
                                        output_filepath=did_plot_file,
                                        title_suffix=output_suffix.replace('_', ' ').title())
             else:
                  logging.warning(f"未找到预期的 DID 结果文件 {did_results_file}，无法绘图。")
        else:
             logging.error("替代波动率 DID 分析核心逻辑执行失败。")

    except Exception as e:
        logging.error(f"运行替代波动率 DID 分析或绘图时出错: {e}")


    # --- 运行 LP-IRF 分析 (使用替代波动率) ---
    logging.info(f"运行 LP-IRF 分析 (结果变量: {alt_outcome_var})...")
    lp_success = False
    try:
        lp_success = run_lp_analysis_core(data=base_data.copy(), # 使用副本
                                          outcome_var=alt_outcome_var,
                                          output_table_dir=alt_table_dir,
                                          output_suffix=output_suffix)
        if lp_success:
             logging.info("替代波动率 LP-IRF 分析核心逻辑执行成功。")
             # 尝试绘图
             plot_lp_irf_results_core(table_dir=alt_table_dir,
                                      figure_dir=alt_figure_dir,
                                      suffix=output_suffix)
        else:
             logging.error("替代波动率 LP-IRF 分析核心逻辑执行失败。")

    except Exception as e:
        logging.error(f"运行替代波动率 LP-IRF 分析或绘图时出错: {e}")

    logging.info("--- 完成稳健性检验: 替代波动率度量 ---")


def run_alt_model_specs(base_data):
    """
    4.2 运行替代模型设定检验
    """
    logging.info("--- 开始稳健性检验: 替代模型设定 ---")
    # TODO: 实现 LP 不同滞后阶数、LP-DID 等
    # 例如: 修改 LP_CONTROL_LAGS 或实现 LP-DID 逻辑
    logging.warning("替代模型设定检验尚未完全实现。")
    # ... 实现逻辑 ...
    logging.info("--- 完成稳健性检验: 替代模型设定 ---")


def run_subsample_analysis(base_data):
    """
    4.3 运行子样本分析
    """
    logging.info("--- 开始稳健性检验: 子样本分析 ---")
    # TODO: 实现按时间、合约类型、流动性、交易所过滤数据并重新运行分析
    # 例如:
    # df_filtered = base_data[base_data['date'] < '2020-01-01']
    # run_did_analysis_core(df_filtered, ...)
    # run_lp_analysis_core(df_filtered, ...)
    logging.warning("子样本分析检验尚未完全实现。")
    # ... 实现逻辑 ...
    logging.info("--- 完成稳健性检验: 子样本分析 ---")


def run_placebo_tests(base_data):
    """
    4.4 运行安慰剂检验
    """
    logging.info("--- 开始稳健性检验: 安慰剂检验 ---")
    # TODO: 实现 DID 和 LP 的安慰剂检验逻辑
    # DID: 需要修改 treat_date_g 为随机日期
    # LP: 需要生成随机的 dlog_margin_rate
    logging.warning("安慰剂检验尚未完全实现。需要实现随机化逻辑。")
    # ... 实现逻辑 ...
    logging.info("--- 完成稳健性检验: 安慰剂检验 ---")


def run_anticipation_effects(base_data):
    """
    4.5 分析预期效应
    """
    logging.info("--- 开始稳健性检验: 预期效应分析 ---")
    # TODO: 实现预期效应的分析逻辑
    # 可能需要利用 announcement_date 列，调整事件窗口或模型
    logging.warning("预期效应分析尚未完全实现。可能需要更复杂的数据处理和模型设定。")
    # ... 实现逻辑 ...
    logging.info("--- 完成稳健性检验: 预期效应分析 ---")


# --- 主执行块 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行期货保证金调整研究的稳健性检验")
    parser.add_argument('--test', type=str, nargs='+', default=['all'],
                        choices=['all', 'alt_vol', 'alt_spec', 'subsample', 'placebo', 'anticipation'],
                        help='指定要运行的稳健性检验类型 (默认: all)')

    args = parser.parse_args()
    tests_to_run = args.test

    logging.info("加载基础面板数据用于稳健性检验...")
    try:
        # 确保加载最新的数据
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
             logging.error(f"无法加载基础面板数据: {config.PANEL_DATA_FILEPATH}")
             logging.error("请先运行 src/data_processing/build_features.py")
             exit()
        df_main = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        logging.info(f"基础数据加载成功: {df_main.shape}")
    except Exception as e:
        logging.error(f"加载基础数据时出错: {e}")
        exit()

    run_all = 'all' in tests_to_run

    if run_all or 'alt_vol' in tests_to_run:
        run_alt_vol_measure(df_main.copy()) # 传递数据的副本

    if run_all or 'alt_spec' in tests_to_run:
        run_alt_model_specs(df_main.copy())

    if run_all or 'subsample' in tests_to_run:
        run_subsample_analysis(df_main.copy())

    if run_all or 'placebo' in tests_to_run:
        run_placebo_tests(df_main.copy())

    if run_all or 'anticipation' in tests_to_run:
        run_anticipation_effects(df_main.copy())

    logging.info("所有指定的稳健性检验运行完毕。")
    logging.info("注意：部分检验 (alt_spec, subsample, placebo, anticipation) 尚未完全实现，需要进一步开发。")