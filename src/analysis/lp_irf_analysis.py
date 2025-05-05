# -*- coding: utf-8 -*-
"""
局部投影脉冲响应函数 (LP-IRF, Jordà, 2005) 分析

估计波动率对保证金变化冲击的平均和状态依赖的动态响应。
包含可被其他脚本调用的核心分析函数 run_lp_analysis_core。
"""

import pandas as pd
import numpy as np
import os
import logging
from linearmodels.panel import PanelOLS
import statsmodels.api as sm # 用于获取置信区间常数

# --- 项目配置导入 ---
try:
    from src import config
except ImportError:
    import sys
    PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PACKAGE_DIR)
    from src import config

# --- 日志配置 ---
# 注意：如果作为模块导入，日志配置可能需要在调用脚本中完成
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 核心 LP 估计函数 ---

def estimate_lp_horizon(data, horizon, outcome_var, shock_var, controls,
                        entity_col='contract_id', time_col='date',
                        cluster_entity=True, cluster_time=True,
                        interaction_state_var=None):
    """
    估计单个预测期 (horizon) 的局部投影回归。

    Args:
        data (pd.DataFrame): 输入面板数据。
        horizon (int): 预测期数 h。
        outcome_var (str): 结果变量名称 (如 'log_gk_volatility')。
        shock_var (str): 冲击变量名称 (如 'margin_increase_shock')。
        controls (list): 控制变量名称列表 (t-1 时刻)。
        entity_col (str): 实体标识列名。
        time_col (str): 时间标识列名。
        cluster_entity (bool): 是否按实体聚类标准误。
        cluster_time (bool): 是否按时间聚类标准误。
        interaction_state_var (str, optional): 用于与冲击变量交互的状态变量名称 (t-1 时刻)。

    Returns:
        tuple: (系数, 标准误, p值, 置信区间下限, 置信区间上限) 或 None (如果估计失败)。
    """
    logging.debug(f"开始估计 LP horizon={horizon}, shock={shock_var}, interaction={interaction_state_var}")

    df_h = data.copy()

    # 1. 创建因变量 Y_{t+h} - Y_{t-1}
    # Y_{t+h}
    df_h[f'{outcome_var}_h'] = df_h.groupby(entity_col)[outcome_var].shift(-horizon)
    # Y_{t-1}
    # 检查 outcome_var 是否已存在 lag1，如果 build_features 已创建则直接使用
    outcome_lag1_col = f'{outcome_var}_lag1'
    if outcome_lag1_col not in df_h.columns:
        df_h[outcome_lag1_col] = df_h.groupby(entity_col)[outcome_var].shift(1)

    # Dependent variable
    dep_var = f'{outcome_var}_diff_h{horizon}'
    df_h[dep_var] = df_h[f'{outcome_var}_h'] - df_h[outcome_lag1_col]

    # 2. 定义自变量
    exog_vars = []
    target_var = shock_var # 默认目标是冲击变量本身

    if interaction_state_var:
        # 如果有交互项，目标是交互项的系数
        interaction_term_name = f'{shock_var}_x_{interaction_state_var}'
        # 确保交互的状态变量存在
        if interaction_state_var not in df_h.columns:
             logging.error(f"交互状态变量 '{interaction_state_var}' 不在数据中。")
             return None
        # 确保冲击变量存在
        if shock_var not in df_h.columns:
             logging.error(f"冲击变量 '{shock_var}' 不在数据中。")
             return None

        df_h[interaction_term_name] = df_h[shock_var] * df_h[interaction_state_var]
        exog_vars.append(interaction_term_name)
        target_var = interaction_term_name
        # 根据模型设定，可能还需要包含冲击变量本身和状态变量本身 (如果固定效应不吸收)
        # exog_vars.append(shock_var)
        # exog_vars.append(interaction_state_var)
    else:
        # 否则，目标是冲击变量的系数
        if shock_var not in df_h.columns:
             logging.error(f"冲击变量 '{shock_var}' 不在数据中。")
             return None
        exog_vars.append(shock_var)

    # 添加控制变量
    valid_controls = [c for c in controls if c in df_h.columns]
    if len(valid_controls) != len(controls):
        missing_ctrl = set(controls) - set(valid_controls)
        logging.warning(f"Horizon {horizon}: 缺少部分控制变量: {missing_ctrl}")
    exog_vars.extend(valid_controls)

    # 3. 准备 PanelOLS 数据
    # 确保所有需要的列都存在
    required_cols_for_reg = [entity_col, time_col, dep_var] + exog_vars
    missing_in_df = [col for col in required_cols_for_reg if col not in df_h.columns]
    if missing_in_df:
        logging.error(f"Horizon {horizon}: DataFrame 缺少回归所需的列: {missing_in_df}")
        return None

    df_reg = df_h[required_cols_for_reg].dropna()
    if df_reg.empty:
        logging.warning(f"Horizon {horizon}: 删除缺失值后数据为空。")
        return None

    # 检查目标变量是否有变化
    if target_var not in df_reg.columns:
         logging.error(f"Horizon {horizon}: 目标变量 '{target_var}' 在删除 NaN 后丢失。")
         return None
    if df_reg[target_var].nunique() < 2:
         logging.warning(f"Horizon {horizon}: 目标变量 '{target_var}' 没有足够的变化 (nunique={df_reg[target_var].nunique()})，无法估计。")
         return None


    try:
        # 设置面板索引
        df_reg = df_reg.set_index([entity_col, time_col])
        exog = df_reg[exog_vars]

        # 4. 执行 PanelOLS 回归
        mod = PanelOLS(df_reg[dep_var], exog, entity_effects=True, time_effects=True, check_rank=False) # check_rank=False 可能需要处理共线性

        # 配置聚类标准误
        cluster_config = {}
        cluster_vars = []
        if cluster_entity: cluster_vars.append(entity_col)
        if cluster_time: cluster_vars.append(time_col)

        # linearmodels 的聚类需要原始数据框和索引名
        if cluster_entity and cluster_time:
            cluster_config['cluster_entity'] = True
            cluster_config['cluster_time'] = True
            cov_type_str = 'clustered'
        elif cluster_entity:
            cluster_config['cluster_entity'] = True
            cov_type_str = 'clustered'
        elif cluster_time:
             cluster_config['cluster_time'] = True
             cov_type_str = 'clustered'
        else:
            cov_type_str = 'robust' # 默认异方差稳健

        results = mod.fit(cov_type=cov_type_str, **cluster_config)

        # 5. 提取结果
        coeff = results.params[target_var]
        stderr = results.std_errors[target_var]
        pval = results.pvalues[target_var]
        conf_int = results.conf_int().loc[target_var]
        conf_low = conf_int['lower']
        conf_high = conf_int['upper']

        logging.debug(f"Horizon {horizon}, Target {target_var}: Coeff={coeff:.4f}, SE={stderr:.4f}, Pval={pval:.3f}")
        return coeff, stderr, pval, conf_low, conf_high

    except KeyError as e:
         # 检查是否是目标变量在 params/std_errors 中不存在 (可能因共线性被移除)
         if target_var in str(e):
             logging.warning(f"Horizon {horizon}: 目标变量 '{target_var}' 可能因共线性等问题从模型中移除。无法获取结果。")
         else:
             logging.error(f"Horizon {horizon}: 回归时发生 KeyError: {e}. 可能缺少变量。")
         return None
    except np.linalg.LinAlgError as e:
        logging.error(f"Horizon {horizon}: 发生线性代数错误 (可能存在完全共线性): {e}")
        # 可以尝试打印 exog.corr() 来检查
        # print(exog.corr())
        return None
    except Exception as e:
        logging.error(f"Horizon {horizon}: 估计时发生未知错误: {e}")
        return None


# --- 重构后的核心分析函数 ---

def run_lp_analysis_core(data, outcome_var, output_table_dir, output_suffix="",
                         shock_var_inc='margin_increase_shock',
                         shock_var_dec='margin_decrease_shock',
                         shock_var_total='dlog_margin_rate',
                         controls=None,
                         lp_horizon=None):
    """
    执行局部投影分析的核心逻辑。

    Args:
        data (pd.DataFrame): 输入面板数据。
        outcome_var (str): 结果变量名称。
        output_table_dir (str): 保存结果表格的目录。
        output_suffix (str, optional): 添加到输出文件名末尾的后缀。Defaults to "".
        shock_var_inc (str, optional): 保证金增加冲击变量名。Defaults to 'margin_increase_shock'.
        shock_var_dec (str, optional): 保证金减少冲击变量名。Defaults to 'margin_decrease_shock'.
        shock_var_total (str, optional): 总保证金冲击变量名 (用于交互)。Defaults to 'dlog_margin_rate'.
        controls (list, optional): 控制变量列表。如果为 None，则从 config 加载。Defaults to None.
        lp_horizon (int, optional): LP 预测期数。如果为 None，则从 config 加载。Defaults to None.

    Returns:
        bool: 如果分析成功完成则返回 True，否则返回 False。
    """
    logging.info(f"--- 开始核心 LP 分析 (Outcome: {outcome_var}, Suffix: '{output_suffix}') ---")

    # 使用 config 中的默认值（如果未提供）
    if controls is None:
        controls = config.CONTROL_VARIABLES
    if lp_horizon is None:
        lp_horizon = config.LP_HORIZON

    # 确保输出目录存在
    os.makedirs(output_table_dir, exist_ok=True)

    df = data.copy() # 使用数据的副本

    # --- 准备状态变量的滞后项 (用于交互) ---
    adj_type_states = ['State_HolidayAdjust', 'State_NonHolidayAdjust']
    state_vars_to_lag = []
    for state in adj_type_states:
        if state in df.columns:
            lagged_name = f'{state}_lag1'
            df[lagged_name] = df.groupby('contract_id')[state].shift(1)
            state_vars_to_lag.append(lagged_name)
        else:
            logging.warning(f"原始调整类型状态变量 '{state}' 不存在，无法创建滞后项。")

    # 删除因滞后产生的 NaN (仅基于成功创建的滞后变量)
    if state_vars_to_lag:
        rows_before = df.shape[0]
        df.dropna(subset=state_vars_to_lag, inplace=True)
        logging.info(f"因状态变量滞后删除 {rows_before - df.shape[0]} 行")
        if df.empty:
            logging.error("因状态变量滞后删除 NaN 后数据为空。")
            return False


    # --- 估计基准 IRFs ---
    logging.info("估计基准 IRFs...")
    baseline_results = []
    analysis_successful = True # 跟踪是否有错误发生

    for h in range(lp_horizon + 1):
        # 上升冲击
        if shock_var_inc in df.columns:
            res_inc = estimate_lp_horizon(df, h, outcome_var, shock_var_inc, controls)
            if res_inc:
                baseline_results.append({'horizon': h, 'shock_type': 'increase',
                                         'coeff': res_inc[0], 'stderr': res_inc[1], 'pval': res_inc[2],
                                         'conf_low': res_inc[3], 'conf_high': res_inc[4]})
            else:
                 logging.warning(f"基准 IRF (上升) 估计失败，Horizon={h}")
                 analysis_successful = False # 标记估计中出现问题
        else:
             logging.warning(f"冲击变量 '{shock_var_inc}' 不存在，跳过上升冲击估计。")

        # 下降冲击
        if shock_var_dec in df.columns:
            res_dec = estimate_lp_horizon(df, h, outcome_var, shock_var_dec, controls)
            if res_dec:
                baseline_results.append({'horizon': h, 'shock_type': 'decrease',
                                         'coeff': res_dec[0], 'stderr': res_dec[1], 'pval': res_dec[2],
                                         'conf_low': res_dec[3], 'conf_high': res_dec[4]})
            else:
                 logging.warning(f"基准 IRF (下降) 估计失败，Horizon={h}")
                 analysis_successful = False
        else:
             logging.warning(f"冲击变量 '{shock_var_dec}' 不存在，跳过下降冲击估计。")

    baseline_df = pd.DataFrame(baseline_results)
    logging.info("基准 IRF 估计完成。")
    # 保存基准结果
    output_baseline_filename = f"lp_irf_results_baseline{output_suffix}.csv"
    output_baseline_path = os.path.join(output_table_dir, output_baseline_filename)
    try:
        baseline_df.to_csv(output_baseline_path, index=False)
        logging.info(f"基准 IRF 结果已保存到: {output_baseline_path}")
    except Exception as e:
        logging.error(f"保存基准 IRF 结果时出错: {e}")
        analysis_successful = False


    # --- 估计状态依赖 IRFs ---
    logging.info("估计状态依赖 IRFs...")
    state_dependent_results = []

    # 定义状态变量 (t-1 时刻)
    state_variables = []
    market_regime_dummies = sorted([col for col in df.columns if col.startswith('market_regime_')])
    if market_regime_dummies: state_variables.extend(market_regime_dummies)
    vol_regime_dummies = sorted([col for col in df.columns if col.startswith('volatility_regime_')])
    if vol_regime_dummies: state_variables.extend(vol_regime_dummies)
    if state_vars_to_lag: state_variables.extend(sorted(state_vars_to_lag)) # 使用成功创建的滞后变量

    if not state_variables:
         logging.warning("未找到可用的状态变量进行交互分析。")
    elif shock_var_total not in df.columns:
        logging.error(f"基础冲击变量 '{shock_var_total}' 不存在，无法进行状态依赖分析。")
        analysis_successful = False
    else:
        for state_var in state_variables:
            logging.info(f"--- 估计状态依赖 IRF: 交互状态 = {state_var} ---")
            for h in range(lp_horizon + 1):
                # 确保状态变量在当前数据子集 df 中存在
                if state_var not in df.columns:
                    logging.warning(f"状态变量 {state_var} 在数据中缺失，跳过交互估计 Horizon={h}")
                    continue

                res_state = estimate_lp_horizon(df, h, outcome_var, shock_var_total, controls,
                                                interaction_state_var=state_var)
                if res_state:
                    state_dependent_results.append({
                        'horizon': h, 'state_variable': state_var, 'shock_variable': shock_var_total,
                        'coeff': res_state[0], 'stderr': res_state[1], 'pval': res_state[2],
                        'conf_low': res_state[3], 'conf_high': res_state[4]
                    })
                else:
                    logging.warning(f"状态依赖 IRF ({state_var}) 估计失败，Horizon={h}")
                    analysis_successful = False # 标记估计中出现问题

        state_dependent_df = pd.DataFrame(state_dependent_results)
        logging.info("状态依赖 IRF 估计完成。")

        # 保存状态依赖结果
        output_statedep_filename = f"lp_irf_results_statedep{output_suffix}.csv"
        output_statedep_path = os.path.join(output_table_dir, output_statedep_filename)
        try:
            state_dependent_df.to_csv(output_statedep_path, index=False)
            logging.info(f"状态依赖 IRF 结果已保存到: {output_statedep_path}")
        except Exception as e:
            logging.error(f"保存状态依赖 IRF 结果时出错: {e}")
            analysis_successful = False

    logging.info(f"--- 完成核心 LP 分析 (Outcome: {outcome_var}, Suffix: '{output_suffix}') ---")
    return analysis_successful


# --- 主执行块 (用于直接运行此脚本) ---
if __name__ == "__main__":
    logging.info("直接运行 lp_irf_analysis.py 脚本...")

    # 加载主数据
    try:
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
             logging.error(f"主面板数据文件未找到: {config.PANEL_DATA_FILEPATH}")
             logging.error("请先运行 src/data_processing/build_features.py")
             exit()
        main_df = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        main_df['date'] = pd.to_datetime(main_df['date']) # 确保日期类型正确
        logging.info(f"主数据加载成功: {main_df.shape}")

        # 执行默认分析
        success = run_lp_analysis_core(data=main_df,
                             outcome_var='log_gk_volatility',
                             output_table_dir=config.PATH_OUTPUT_TABLES,
                             output_suffix="") # 无后缀表示主分析
        if success:
            logging.info("默认 LP 分析成功完成。")
        else:
            logging.error("默认 LP 分析执行过程中出现错误。")

    except Exception as e:
        logging.error(f"直接运行 LP 分析时出错: {e}")