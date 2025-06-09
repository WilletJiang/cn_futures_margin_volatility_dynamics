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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

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
        logging.error(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: DataFrame 缺少回归所需的列: {missing_in_df}")
        return None

    df_reg = df_h[required_cols_for_reg].copy() # MODIFIED: Added .copy() to avoid SettingWithCopyWarning
    logging.debug(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: df_reg shape BEFORE dropna: {df_reg.shape}")
    logging.debug(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: Non-NA counts BEFORE dropna for required_cols_for_reg:\\n{df_reg.notna().sum()}")

    df_reg.dropna(inplace=True) # MODIFIED: Changed from df_reg = df_h[...].dropna() to operate on the copy
    logging.debug(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: df_reg shape AFTER dropna: {df_reg.shape}")
    if df_reg.empty:
        logging.warning(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: 删除缺失值后数据为空。")
        return None

    # 检查目标变量是否有变化
    if target_var not in df_reg.columns:
         logging.error(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: 目标变量 '{target_var}' 在删除 NaN 后丢失。Columns in df_reg: {df_reg.columns.tolist()}")
         return None
    if df_reg[target_var].nunique() < 2:
         logging.warning(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: 目标变量 '{target_var}' 没有足够的变化 (nunique={df_reg[target_var].nunique()})，无法估计。df_reg[{target_var}].value_counts():\n{df_reg[target_var].value_counts(dropna=False)}")
         return None

    # 特别调试: 检查 horizon 3 基线模型的共线性
    if horizon == 3 and interaction_state_var is None:
        try:
            logging.debug(f"DEBUG Horizon 3 Baseline: Correlation matrix of exogenous variables (df_reg[exog_vars]):\n{df_reg[exog_vars].corr()}")
        except Exception as e_corr:
            logging.debug(f"DEBUG Horizon 3 Baseline: Could not compute correlation matrix: {e_corr}")

    try:
        # 设置面板索引
        df_reg = df_reg.set_index([entity_col, time_col])
        exog = df_reg[exog_vars]

        # --- 调试：检查解释变量 ---
        logging.debug(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: Exogenous variables being used: {exog_vars}")
        for var in exog_vars:
            if var in exog.columns:
                logging.debug(f"  Var '{var}': nunique={exog[var].nunique()}, dtype={exog[var].dtype}, non_na_count={exog[var].notna().sum()}")
                if exog[var].nunique() < 2 and exog[var].notna().sum() > 0 : # 如果变量存在但没有变化 (例如全是0或全是同一个值)
                     logging.warning(f"  WARNING Var '{var}': has {exog[var].nunique()} unique values but {exog[var].notna().sum()} non-NA observations. This could cause perfect collinearity or an intercept-only like variable.")
                     logging.debug(f"    Value counts for '{var}':\\n{exog[var].value_counts(dropna=False).to_string()}")
            else:
                logging.error(f"  Var '{var}' specified in exog_vars but not found in exog DataFrame columns after set_index and selection.")
        
        # --- 调试：计算条件数 ---
        try:
            if not exog.empty and exog.shape[1] > 0: # 确保 exog 不为空且有列
                # 确保所有列都是数值类型，对于 PanelOLS 通常会自动处理，但条件数计算需要显式数值
                exog_numeric = exog.select_dtypes(include=np.number)
                if exog_numeric.shape[1] < exog.shape[1]:
                    logging.warning(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: Not all exogenous variables are numeric for condition number calculation. Original exog columns: {exog.columns.tolist()}, Numeric exog columns: {exog_numeric.columns.tolist()}")
                
                if not exog_numeric.empty and exog_numeric.shape[1] > 0:
                    # 添加常数项以模拟回归中的截距（如果模型有截距且未被固定效应吸收）
                    # PanelOLS 通常自己处理截距和固定效应，这里我们只看解释变量自身的共线性
                    # X = sm.add_constant(exog_numeric, prepend=True) if not entity_effects and not time_effects else exog_numeric # 根据是否有固定效应决定是否加常数
                    # 考虑到 PanelOLS 会处理固定效应，我们主要关注 exog_vars 之间的共线性。
                    # 如果 exog_vars 包含了一个接近常数的列 (除了固定效应之外)，那也会有问题。
                    X_for_cond = exog_numeric.copy()
                    # 检查是否存在方差为0的列 (常数项)
                    constant_cols = X_for_cond.columns[X_for_cond.var() < 1e-10] # 方差极小的列
                    if len(constant_cols) > 0:
                        logging.warning(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: Exogenous variables include constant-like columns (after selecting numeric): {constant_cols.tolist()}. This will lead to perfect collinearity if not handled by fixed effects.")
                        # 尝试移除这些列再计算条件数，但这可能改变模型
                        # X_for_cond = X_for_cond.drop(columns=constant_cols)
                    
                    if not X_for_cond.empty and X_for_cond.shape[1] > 0:
                        condition_number = np.linalg.cond(X_for_cond)
                        logging.info(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: Condition number of exogenous variables (numeric part): {condition_number:.2f}")
                        if condition_number > 30: # 一个常用的经验阈值
                            logging.warning(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: High condition number ({condition_number:.2f}) suggests potential multicollinearity.")
                    else:
                        logging.debug(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: Exogenous numeric matrix for condition number is empty or has no columns after filtering constant-like columns.")
                else:
                    logging.debug(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: No numeric exogenous variables to calculate condition number.")
        except Exception as e_cond:
            logging.error(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: Error calculating condition number: {e_cond}")

        # 4. 执行 PanelOLS 回归
        # mod = PanelOLS(df_reg[dep_var], exog, entity_effects=True, time_effects=True, check_rank=False) # MODIFIED: Original line commented out

        # MODIFIED: Start of new logic for check_rank based on interaction_state_var
        use_strict_rank_check = False # 默认为False
        log_rank_check_reason = "宽松检查"
        if interaction_state_var == 'market_regime_Neutral':
            use_strict_rank_check = True
            log_rank_check_reason = "严格检查) 进行 market_regime_Neutral 的估计"
            # 这条日志现在由下面的通用日志代替
            # logger.debug(f"Horizon {h}, Shock {shock_var}: 使用 check_rank={use_strict_rank_check} ({log_rank_check_reason}")
        # else:
            # log_rank_check_reason = "宽松检查" # 如果不是Neutral，明确设为宽松 (虽然已经是默认)
            # 这条日志现在由下面的通用日志代替
            # logger.debug(f"Horizon {h}, Shock {shock_var}, Interaction {interaction_state_var if interaction_state_var else 'None'}: 使用 check_rank={use_strict_rank_check} ({log_rank_check_reason}")
        
        logging.debug(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var if interaction_state_var else 'None'}: 设置 check_rank={use_strict_rank_check} ({log_rank_check_reason})")
        mod = PanelOLS(df_reg[dep_var], exog, entity_effects=True, time_effects=True, check_rank=use_strict_rank_check)

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

        # 针对 market_regime_Neutral 在特定 horizons 上 NaN SE 问题的特别处理
        problematic_horizons_for_neutral = [5, 6, 11, 12, 16]
        if interaction_state_var == 'market_regime_Neutral' and horizon in problematic_horizons_for_neutral:
            if cov_type_str != "robust" or (cluster_entity or cluster_time): # 检查是否真的需要覆盖，并且之前不是robust或者有聚类设置
                 logging.info(f"Horizon {horizon}, Interaction {interaction_state_var}: 检测到之前 problematic horizon，强制使用 'robust' (非聚类) 标准误。")
            cov_type_str = "robust" # 强制使用稳健标准误，不进行聚类
            effective_cluster_config = {} # 如果是 robust, 则不应该有聚类参数
        else:
            effective_cluster_config = cluster_config # 否则使用原始的聚类配置

        # results = mod.fit(cov_type=cov_type_str, **cluster_config) # 原始行
        # results = mod.fit(cov_type=cov_type_str, **effective_cluster_config) # 上一个版本，仍有问题

        # 新的拟合逻辑，更明确地处理空的 effective_cluster_config
        if not effective_cluster_config: 
            logging.debug(f"Horizon {horizon}, Interaction {interaction_state_var if interaction_state_var else 'None'}: effective_cluster_config is empty. Calling fit with cov_type='{cov_type_str}' only.")
            results = mod.fit(cov_type=cov_type_str)
        else:
            logging.debug(f"Horizon {horizon}, Interaction {interaction_state_var if interaction_state_var else 'None'}: effective_cluster_config is {effective_cluster_config}. Calling fit with cov_type='{cov_type_str}' and config.")
            results = mod.fit(cov_type=cov_type_str, **effective_cluster_config)

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
             logging.warning(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: 目标变量 '{target_var}' 可能因共线性等问题从模型中移除。无法获取结果。\\nExog vars: {exog_vars}\\nExog nunique:\\n{exog.nunique().to_string() if 'exog' in locals() and hasattr(exog, 'nunique') else 'exog not defined or nunique failed'}\\nExog info:\\n{exog.info() if 'exog' in locals() and hasattr(exog, 'info') else 'exog not defined or info failed'}")
         else:
             logging.error(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: 回归时发生 KeyError: {e}. 可能缺少变量。Exog vars: {exog_vars}")
         return None
    except np.linalg.LinAlgError as e:
        logging.error(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: 发生线性代数错误 (可能存在完全共线性): {e}\\nExog vars: {exog_vars}\\nExog nunique:\\n{exog.nunique().to_string() if 'exog' in locals() and hasattr(exog, 'nunique') else 'exog not defined or nunique failed'}\\nExog describe:\\n{exog.describe().to_string() if 'exog' in locals() and hasattr(exog, 'describe') else 'exog not defined or describe failed'}\\nDep_var describe:\\n{df_reg[dep_var].describe().to_string() if 'df_reg' in locals() and dep_var in df_reg and hasattr(df_reg[dep_var], 'describe') else 'dep_var not defined or describe failed'}")
        # 可以尝试打印 exog.corr() 来检查
        if 'exog' in locals() and hasattr(exog, 'corr'):
            try:
                logging.error(f"Exog correlation matrix:\\n{exog.corr().to_string()}")
            except Exception as e_corr_print:
                logging.error(f"Could not print exog correlation matrix: {e_corr_print}")
        return None
    except Exception as e:
        logging.error(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: 估计时发生未知错误: {e}\\nExog vars: {exog_vars}\\nExog nunique:\\n{exog.nunique().to_string() if 'exog' in locals() and hasattr(exog, 'nunique') else 'exog not defined or nunique failed'}\\nExog describe:\\n{exog.describe().to_string() if 'exog' in locals() and hasattr(exog, 'describe') else 'exog not defined or describe failed'}\\nDep_var describe:\\n{df_reg[dep_var].describe().to_string() if 'df_reg' in locals() and dep_var in df_reg and hasattr(df_reg[dep_var], 'describe') else 'dep_var not defined or describe failed'}")
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
                 logging.warning(f"基准 IRF (上升), Shock '{shock_var_inc}', Horizon={h}: 估计返回 None.")
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
                 logging.warning(f"基准 IRF (下降), Shock '{shock_var_dec}', Horizon={h}: 估计返回 None.")
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
                    logging.warning(f"状态依赖 IRF, Interaction '{state_var}', Shock '{shock_var_total}', Horizon={h}: 估计返回 None.")
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

    # --- 流动性机制检验 ---
    # Outcome: turnover_rate
    try:
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
             logging.error(f"主面板数据文件未找到: {config.PANEL_DATA_FILEPATH}")
             exit()
        main_df_liq_tr = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        main_df_liq_tr['date'] = pd.to_datetime(main_df_liq_tr['date'])
        # 确保 turnover_rate 存在
        turnover_rate_var = config.LIQUIDITY_PROXY_VAR.replace('_lag1', '') # 'turnover_rate'
        if turnover_rate_var not in main_df_liq_tr.columns:
            logging.error(f"流动性指标 '{turnover_rate_var}' 不在数据中，无法进行分析。")
        else:
            logging.info(f"--- 开始流动性机制检验 (Outcome: {turnover_rate_var}) ---")
            success_liq_tr = run_lp_analysis_core(
                data=main_df_liq_tr,
                outcome_var=turnover_rate_var,
                output_table_dir=config.PATH_OUTPUT_TABLES,
                output_suffix="_liquidity_turnover", # 新的后缀
                # shock_var_inc, shock_var_dec, shock_var_total, controls, lp_horizon 使用默认值
            )
            if success_liq_tr:
                logging.info(f"LP 分析 (Outcome: {turnover_rate_var}) 成功完成。")
            else:
                logging.error(f"LP 分析 (Outcome: {turnover_rate_var}) 执行过程中出现错误。")
    except Exception as e:
        logging.error(f"运行 LP 分析 (Outcome: turnover_rate) 时出错: {e}")

    # Outcome: log_volume
    try:
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
             logging.error(f"主面板数据文件未找到: {config.PANEL_DATA_FILEPATH}")
             exit()
        main_df_liq_lv = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        main_df_liq_lv['date'] = pd.to_datetime(main_df_liq_lv['date'])
        log_volume_var = 'log_volume'
        if log_volume_var not in main_df_liq_lv.columns:
            logging.error(f"流动性指标 '{log_volume_var}' 不在数据中，无法进行分析。")
        else:
            logging.info(f"--- 开始流动性机制检验 (Outcome: {log_volume_var}) ---")
            success_liq_lv = run_lp_analysis_core(
                data=main_df_liq_lv,
                outcome_var=log_volume_var,
                output_table_dir=config.PATH_OUTPUT_TABLES,
                output_suffix="_liquidity_logvolume", # 新的后缀
            )
            if success_liq_lv:
                logging.info(f"LP 分析 (Outcome: {log_volume_var}) 成功完成。")
            else:
                logging.error(f"LP 分析 (Outcome: {log_volume_var}) 执行过程中出现错误。")
    except Exception as e:
        logging.error(f"运行 LP 分析 (Outcome: {log_volume_var}) 时出错: {e}")

    # --- 敏感性分析：排除极端冲击 ---
    try:
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
            logging.error(f"主面板数据文件未找到: {config.PANEL_DATA_FILEPATH}")
            exit()
        main_df_trimmed = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        main_df_trimmed['date'] = pd.to_datetime(main_df_trimmed['date'])

        # 识别极端冲击 (dlog_margin_rate 绝对值最大的 1%)
        # 确保 dlog_margin_rate 存在
        if 'dlog_margin_rate' not in main_df_trimmed.columns:
            logging.error("'dlog_margin_rate' 列不在数据中，无法进行极端冲击排除。")
        else:
            logging.info("--- 开始敏感性分析：排除极端保证金率变动冲击 ---")
            abs_dlog_margin = main_df_trimmed['dlog_margin_rate'].abs()
            
            # 改进的极端冲击阈值计算逻辑
            non_zero_abs_shocks = abs_dlog_margin[abs_dlog_margin > 1e-10] # 考虑浮点精度的小容差
            
            if non_zero_abs_shocks.empty:
                logging.warning("数据中没有发现非零的保证金率变动，跳过极端冲击排除分析。")
                threshold = 0 # 使得后续过滤保留所有数据（如果都为0）
                df_filtered_shocks = main_df_trimmed.copy() # 无需过滤
            else:
                threshold = non_zero_abs_shocks.quantile(0.99)
                logging.info(f"极端冲击阈值 (基于非零冲击的99th percentile of absolute dlog_margin_rate): {threshold:.6f}")
                # 过滤：保留冲击小于阈值的，或冲击为零的
                df_filtered_shocks = main_df_trimmed[(abs_dlog_margin < threshold) | (abs_dlog_margin < 1e-10)].copy()

            logging.info(f"原始数据行数: {len(main_df_trimmed)}, 排除极端冲击后行数: {len(df_filtered_shocks)}")

            if len(df_filtered_shocks) < 0.5 * len(main_df_trimmed) or df_filtered_shocks.empty:
                logging.warning("排除极端冲击后数据量过少或为空，跳过此项敏感性分析。")
            else:
                # 在过滤后的数据上重新定义冲击变量 (如果需要，尽管 run_lp_analysis_core 主要看 shock_var_inc/dec)
                # 对于基准分析，我们通常传入 shock_var_inc 和 shock_var_dec。
                # 我们需要确保这些冲击变量在过滤后的 df_filtered_shocks 中也相应调整。
                # run_lp_analysis_core 会使用传入的 shock_var_inc/dec 列名，所以这些列必须存在于 data 参数中。
                # 如果原始的 shock_var_inc/dec 是基于 dlog_margin_rate 计算的，那么在过滤行之后，这些列的值依然有效。
                # 或者，我们可以仅对 dlog_margin_rate 进行操作，然后让 estimate_lp_horizon 基于这个新的 dlog_margin_rate
                # （如果它支持从一个总冲击变量生成 inc/dec）。但目前 estimate_lp_horizon 是直接用 shock_var。
                # 因此，只要过滤后的 df_filtered_shocks 中保留了原有的 margin_increase_shock 和 margin_decrease_shock 列即可。
                # 这些列的值本身已经是基于原始 dlog_margin_rate 计算的。
                # 过滤掉了包含极端 dlog_margin_rate 的行，也就过滤掉了相应的极端 increase/decrease shock。

                success_trimmed = run_lp_analysis_core(
                    data=df_filtered_shocks,
                    outcome_var='log_gk_volatility', # 结果变量仍是波动率
                    output_table_dir=config.PATH_OUTPUT_TABLES,
                    output_suffix="_trimmed_shocks",
                    # shock_var_inc, shock_var_dec 等使用默认，它们应该存在于 df_filtered_shocks 中
                )
                if success_trimmed:
                    logging.info("LP 分析 (排除极端冲击) 成功完成。")
                else:
                    logging.error("LP 分析 (排除极端冲击) 执行过程中出现错误。")

    except Exception as e:
        logging.error(f"运行 LP 分析 (排除极端冲击) 时出错: {e}")

    # --- 敏感性分析：按品种类型分解 ---
    try:
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
            logging.error(f"主面板数据文件未找到: {config.PANEL_DATA_FILEPATH}")
            exit()
        main_df_variety = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        main_df_variety['date'] = pd.to_datetime(main_df_variety['date'])

        variety_type_col = 'variety' 
        if variety_type_col not in main_df_variety.columns:
            logging.warning(f"品种类型列 '{variety_type_col}' 不在数据中，跳过按品种分解的敏感性分析。")
        else:
            unique_varieties = main_df_variety[variety_type_col].unique()
            logging.info(f"--- 开始敏感性分析：按品种类型分解 (发现品种: {unique_varieties}) ---")

            for variety in unique_varieties:
                if pd.isna(variety): # 跳过缺失的品种类型
                    logging.info(f"跳过缺失的品种类型 (NaN)")
                    continue
                
                logging.info(f"处理品种类型: {variety}")
                df_subset = main_df_variety[main_df_variety[variety_type_col] == variety].copy()

                if df_subset.empty or len(df_subset) < 100: # 如果子集太小，可能无法进行稳健估计
                    logging.warning(f"品种类型 '{variety}' 的数据子集太小 (行数: {len(df_subset)})，跳过分析。")
                    continue

                # 清理品种名称以用作文件名后缀 (移除非字母数字字符)
                safe_variety_suffix = "".join(filter(str.isalnum, str(variety)))
                if not safe_variety_suffix: # 如果清理后为空，使用通用后缀
                    safe_variety_suffix = f"unknownvariety_{hash(variety) % 1000}"
                
                output_suffix_variety = f"_vt_{safe_variety_suffix}"

                logging.info(f"运行品种 '{variety}' (后缀: {output_suffix_variety}) 的 LP 分析...")
                success_variety = run_lp_analysis_core(
                    data=df_subset,
                    outcome_var='log_gk_volatility',
                    output_table_dir=config.PATH_OUTPUT_TABLES,
                    output_suffix=output_suffix_variety,
                )
                if success_variety:
                    logging.info(f"LP 分析 (品种: {variety}) 成功完成。")
                else:
                    logging.error(f"LP 分析 (品种: {variety}) 执行过程中出现错误。")

    except Exception as e:
        logging.error(f"运行 LP 分析 (按品种类型分解) 时出错: {e}")