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
import sys

# 确保能够找到项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # 上两级目录
sys.path.insert(0, PROJECT_ROOT)  # 将项目根目录添加到Python路径

# --- 项目配置和分析脚本导入 ---
try:
    from src import config
    # 导入重构后的核心分析和绘图函数
    from src.analysis.did_cs_analysis import run_did_analysis_core
    from src.analysis.lp_irf_analysis import run_lp_analysis_core
    from src.visualization.plot_did_results import plot_did_core as plot_did_results_core  # 避免命名冲突
    from src.visualization.plot_lp_irf_results import plot_lp_irf_core as plot_lp_irf_results_core  # 避免命名冲突
except ImportError as e:
    logging.error(f"导入模块时出错: {e}")
    logging.error(f"当前Python路径: {sys.path}")
    sys.exit(1)


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
    main_outcome_var = config.OUTCOME_VAR.replace('_lag1', '') # 假设主结果变量也需要移除_lag1

    # 定义不同的模型设定
    specs_to_run = []

    # --- DID 的替代设定 ---
    # 设定 1: DID 无控制变量
    specs_to_run.append({
        "name": "did_no_controls",
        "type": "did",
        "params": {"control_vars": []},
        "output_suffix": "_did_no_ctrl"
    })
    # 设定 2: DID 使用 IPW 估计方法 (如果 DR 是默认)
    if "dr" == config.DID_EST_METHOD: # 假设 config 中有 DID_EST_METHOD
        specs_to_run.append({
            "name": "did_ipw_method",
            "type": "did",
            "params": {"est_method": "ipw"},
            "output_suffix": "_did_ipw"
        })
    # 设定 3: DID 使用 never treated 控制组 (如果 notyettreated 是默认)
    # 需要确认数据中是否有真正的 never treated 组，并且 config.DID_CONTROL_GROUP_TYPE 存在
    if hasattr(config, 'DID_CONTROL_GROUP_TYPE') and config.DID_CONTROL_GROUP_TYPE == "notyettreated":
         specs_to_run.append({
             "name": "did_nevertreated_group",
             "type": "did",
             "params": {"control_group_type": "nevertreated"},
             "output_suffix": "_did_never_treated"
         })


    # --- LP-IRF 的替代设定 ---
    # 设定 4: LP-IRF 使用较短的 horizon
    short_horizon = max(1, config.LP_HORIZON // 2) # 例如，主 horizon 的一半，至少为1
    if short_horizon < config.LP_HORIZON:
        specs_to_run.append({
            "name": f"lp_short_horizon_{short_horizon}",
            "type": "lp",
            "params": {"lp_horizon": short_horizon},
            "output_suffix": f"_lp_short_h{short_horizon}"
        })
    # 设定 5: LP-IRF 使用较长的 horizon
    long_horizon = config.LP_HORIZON + 5 # 例如，主 horizon + 5
    specs_to_run.append({
        "name": f"lp_long_horizon_{long_horizon}",
        "type": "lp",
        "params": {"lp_horizon": long_horizon},
        "output_suffix": f"_lp_long_h{long_horizon}"
    })
    # 设定 6: LP-IRF 无控制变量
    specs_to_run.append({
        "name": "lp_no_controls",
        "type": "lp",
        "params": {"controls": []},
        "output_suffix": "_lp_no_ctrl"
    })

    if not specs_to_run:
        logging.warning("没有定义可运行的替代模型设定。")
        logging.info("--- 完成稳健性检验: 替代模型设定 (无操作) ---")
        return

    logging.info(f"将运行以下替代模型设定: {[s['name'] for s in specs_to_run]}")

    for spec in specs_to_run:
        logging.info(f"--- 开始运行替代设定: {spec['name']} ---")
        spec_data = base_data.copy() # 每个设定使用独立的数据副本
        spec_table_dir = os.path.join(ROBUSTNESS_TABLE_DIR, f"alt_spec_{spec['name']}")
        spec_figure_dir = os.path.join(ROBUSTNESS_FIGURE_DIR, f"alt_spec_{spec['name']}")
        os.makedirs(spec_table_dir, exist_ok=True)
        os.makedirs(spec_figure_dir, exist_ok=True)

        output_suffix = spec['output_suffix']

        try:
            if spec['type'] == 'did':
                logging.info(f"运行 DID 分析 (设定: {spec['name']}, 结果变量: {main_outcome_var})...")
                # 获取当前 spec 的参数，如果参数未在 spec 中定义，则使用 None (让 core 函数使用其默认值或 config 值)
                did_params = {
                    "outcome_var": main_outcome_var,
                    "output_table_dir": spec_table_dir,
                    "output_suffix": output_suffix,
                    "control_vars": spec['params'].get('control_vars'), # None if not set
                    "control_group_type": spec['params'].get('control_group_type'), # None if not set
                    "est_method": spec['params'].get('est_method') # None if not set
                }
                # 移除值为 None 的参数，以便 core 函数使用其内部默认值
                did_params_cleaned = {k: v for k, v in did_params.items() if v is not None}

                did_success = run_did_analysis_core(data=spec_data, **did_params_cleaned)

                if did_success:
                    logging.info(f"DID 分析 ({spec['name']}) 核心逻辑执行成功。")
                    did_results_file = os.path.join(spec_table_dir, f"did_aggregate_results{output_suffix}.csv")
                    did_plot_file = os.path.join(spec_figure_dir, f"did_event_study_plot{output_suffix}.png")
                    if os.path.exists(did_results_file):
                        plot_did_results_core(results_filepath=did_results_file,
                                              output_filepath=did_plot_file,
                                              title_suffix=f" ({spec['name']})")
                    else:
                        logging.warning(f"未找到 DID 结果文件 {did_results_file} ({spec['name']})，无法绘图。")
                else:
                    logging.error(f"DID 分析 ({spec['name']}) 核心逻辑执行失败。")

            elif spec['type'] == 'lp':
                logging.info(f"运行 LP-IRF 分析 (设定: {spec['name']}, 结果变量: {main_outcome_var})...")
                # 获取当前 spec 的参数
                lp_params = {
                    "outcome_var": main_outcome_var,
                    "output_table_dir": spec_table_dir,
                    "output_suffix": output_suffix,
                    "controls": spec['params'].get('controls'), # None if not set
                    "lp_horizon": spec['params'].get('lp_horizon') # None if not set
                }
                # 移除值为 None 的参数
                lp_params_cleaned = {k: v for k, v in lp_params.items() if v is not None}

                lp_success = run_lp_analysis_core(data=spec_data, **lp_params_cleaned)

                if lp_success:
                    logging.info(f"LP-IRF 分析 ({spec['name']}) 核心逻辑执行成功。")
                    plot_lp_irf_results_core(table_dir=spec_table_dir,
                                             figure_dir=spec_figure_dir,
                                             suffix=output_suffix)
                else:
                    logging.error(f"LP-IRF 分析 ({spec['name']}) 核心逻辑执行失败。")

        except Exception as e:
            logging.error(f"运行替代模型设定 {spec['name']} 时出错: {e}", exc_info=True)

        logging.info(f"--- 完成运行替代设定: {spec['name']} ---")

    logging.info("--- 完成稳健性检验: 替代模型设定 ---")


def run_subsample_analysis(base_data):
    """
    4.3 运行子样本分析
    """
    logging.info("--- 开始稳健性检验: 子样本分析 ---")
    main_outcome_var = config.OUTCOME_VAR.replace('_lag1', '') # 主结果变量

    # --- 定义子样本 ---
    subsample_definitions = [
        # --- 1. 按时间分割 ---
        {
            "name": "time_pre_2020", # 2020年1月1日之前
            "filter_lambda": lambda df: df[df['date'] < pd.Timestamp('2020-01-01')],
            "output_suffix": "_sub_pre2020"
        },
        {
            "name": "time_post_2019", # 2020年1月1日及以后
            "filter_lambda": lambda df: df[df['date'] >= pd.Timestamp('2020-01-01')],
            "output_suffix": "_sub_post2019"
        },

        # --- 2. 按市场状态 (Market Regime) ---
        {
            "name": "market_bull",
            "filter_lambda": lambda df: df[df['market_regime_Bull'] == 1],
            "output_suffix": "_sub_bull"
        },
        {
            "name": "market_bear",
            "filter_lambda": lambda df: df[df['market_regime_Bear'] == 1],
            "output_suffix": "_sub_bear"
        },
        {
            "name": "market_neutral",
            "filter_lambda": lambda df: df[df['market_regime_Neutral'] == 1],
            "output_suffix": "_sub_neutral"
        },

        # --- 3. 按波动率状态 (Volatility Regime) ---
        {
            "name": "vol_high",
            "filter_lambda": lambda df: df[df['volatility_regime_High'] == 1],
            "output_suffix": "_sub_volhigh"
        },
        {
            "name": "vol_low",
            "filter_lambda": lambda df: df[df['volatility_regime_Low'] == 1],
            "output_suffix": "_sub_vollow"
        },

        # --- 4. 按调整类型 ---
        {
            "name": "adj_holiday",
            "filter_lambda": lambda df: df[df['State_HolidayAdjust'] == 1],
            "output_suffix": "_sub_adjholiday"
        },
        {
            "name": "adj_nonholiday",
            "filter_lambda": lambda df: df[df['State_NonHolidayAdjust'] == 1],
            "output_suffix": "_sub_adjnonholiday"
        },

        # --- 5. 按合约类别 (示例 - 请根据您的数据调整或取消注释) ---
        # {
        #     "name": "contract_indexfutures",
        #     "filter_lambda": lambda df: df['contract_id'].str.startswith('IF', na=False),
        #     "output_suffix": "_sub_idxft"
        # },
        # {
        #     "name": "contract_copper",
        #     "filter_lambda": lambda df: df['contract_id'].str.startswith('CU', na=False),
        #     "output_suffix": "_sub_copper"
        # },

        # --- 6. 按处理组标签 (示例 - 请根据您的数据调整或取消注释) ---
        # {
        #     "name": "group_springfestival",
        #     "filter_lambda": lambda df: df['treat_group_g_label'] == '春节',
        #     "output_suffix": "_sub_springfest"
        # },
        # {
        #     "name": "group_nationalday",
        #     "filter_lambda": lambda df: df['treat_group_g_label'] == '国庆节',
        #     "output_suffix": "_sub_nationalday"
        # },
    ]

    # 过滤掉那些依赖于 base_data 中不存在的列的子样本定义 (更安全的做法)
    active_subsample_definitions = []
    for sub_def in subsample_definitions:
        try:
            # 尝试对一小部分数据应用 lambda 以检查列是否存在
            _ = sub_def["filter_lambda"](base_data.head(2).copy()) # 使用 .copy() 避免 SettingWithCopyWarning
            active_subsample_definitions.append(sub_def)
        except KeyError as e:
            logging.warning(f"子样本定义 '{sub_def['name']}' 因缺少列 {e} 而被跳过。请检查您的数据或子样本定义。")
        except Exception as e:
            logging.warning(f"子样本定义 '{sub_def['name']}' 在测试时出错 ({type(e).__name__}: {e})，将被跳过。")

    if not active_subsample_definitions:
        logging.warning("没有可运行的子样本分析定义 (可能因缺少列或配置错误)。")
        logging.info("--- 完成稳健性检验: 子样本分析 (无操作) ---")
        return

    logging.info(f"将运行以下子样本分析: {[s['name'] for s in active_subsample_definitions]}")

    for sub_def in active_subsample_definitions:
        logging.info(f"--- 开始运行子样本分析: {sub_def['name']} ---")

        try:
            # 对完整数据的副本应用过滤器
            sub_data = sub_def["filter_lambda"](base_data.copy())
        except KeyError as e: # 再次捕获以防万一 (例如，如果 head(1) 测试未完全暴露问题)
            logging.error(f"为子样本 '{sub_def['name']}' 创建数据时列缺失: {e}。跳过此子样本。")
            continue
        except Exception as e:
            logging.error(f"为子样本 '{sub_def['name']}' 应用过滤器时出错: {e}。跳过此子样本。", exc_info=True)
            continue

        if sub_data.empty:
            logging.warning(f"子样本 '{sub_def['name']}' 数据为空 (过滤后)，跳过分析。")
            continue

        logging.info(f"子样本 '{sub_def['name']}' 数据行数: {sub_data.shape[0]}, 列数: {sub_data.shape[1]}")

        sub_table_dir = os.path.join(ROBUSTNESS_TABLE_DIR, f"subsample_{sub_def['name']}")
        sub_figure_dir = os.path.join(ROBUSTNESS_FIGURE_DIR, f"subsample_{sub_def['name']}")
        os.makedirs(sub_table_dir, exist_ok=True)
        os.makedirs(sub_figure_dir, exist_ok=True)

        output_suffix = sub_def['output_suffix']

        # --- 运行 DID 分析 (子样本) ---
        try:
            logging.info(f"运行 DID 分析 (子样本: {sub_def['name']}, 结果变量: {main_outcome_var})...")
            did_success = run_did_analysis_core(data=sub_data.copy(), # 传递子数据的副本
                                                outcome_var=main_outcome_var,
                                                output_table_dir=sub_table_dir,
                                                output_suffix=output_suffix)
            if did_success:
                logging.info(f"子样本 DID 分析 ({sub_def['name']}) 核心逻辑执行成功。")
                did_results_file = os.path.join(sub_table_dir, f"did_aggregate_results{output_suffix}.csv")
                did_plot_file = os.path.join(sub_figure_dir, f"did_event_study_plot{output_suffix}.png")
                if os.path.exists(did_results_file):
                    plot_did_results_core(results_filepath=did_results_file,
                                          output_filepath=did_plot_file,
                                          title_suffix=f" ({sub_def['name']})")
                else:
                    logging.warning(f"未找到预期的 DID 结果文件 {did_results_file} ({sub_def['name']})，无法绘图。")
            else:
                logging.error(f"子样本 DID 分析 ({sub_def['name']}) 核心逻辑执行失败。")
        except Exception as e:
            logging.error(f"运行子样本 DID 分析 ({sub_def['name']}) 或绘图时出错: {e}", exc_info=True)

        # --- 运行 LP-IRF 分析 (子样本) ---
        try:
            logging.info(f"运行 LP-IRF 分析 (子样本: {sub_def['name']}, 结果变量: {main_outcome_var})...")
            lp_success = run_lp_analysis_core(data=sub_data.copy(), # 传递子数据的副本
                                              outcome_var=main_outcome_var,
                                              output_table_dir=sub_table_dir,
                                              output_suffix=output_suffix)
            if lp_success:
                logging.info(f"子样本 LP-IRF 分析 ({sub_def['name']}) 核心逻辑执行成功。")
                plot_lp_irf_results_core(table_dir=sub_table_dir,
                                         figure_dir=sub_figure_dir,
                                         suffix=output_suffix)
            else:
                logging.error(f"子样本 LP-IRF 分析 ({sub_def['name']}) 核心逻辑执行失败。")
        except Exception as e:
            logging.error(f"运行子样本 LP-IRF 分析 ({sub_def['name']}) 或绘图时出错: {e}", exc_info=True)

        logging.info(f"--- 完成运行子样本分析: {sub_def['name']} ---")

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