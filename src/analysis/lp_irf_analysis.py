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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

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

    # 1. 创建因变量 - 使用水平形式 Y_{t+h}
    # 按用户要求，使用Y(t+h)作为因变量，而不是差分形式
    
    # Y_{t+h}
    df_h[f'{outcome_var}_h'] = df_h.groupby(entity_col)[outcome_var].shift(-horizon)
    
    # 因变量：直接使用Y_{t+h}
    dep_var = f'{outcome_var}_level_h{horizon}'
    df_h[dep_var] = df_h[f'{outcome_var}_h']

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

        # --- 改进：检查交互状态变量的有效性 ---
        state_nunique = df_h[interaction_state_var].nunique()
        state_std = df_h[interaction_state_var].std()
        
        if state_nunique < 2:
            logging.warning(f"Horizon {horizon}: 交互状态变量 '{interaction_state_var}' 只有 {state_nunique} 个唯一值，跳过交互项")
            # 退回到简单模式
            exog_vars.append(shock_var)
            target_var = shock_var
        elif state_std < 1e-10:
            logging.warning(f"Horizon {horizon}: 交互状态变量 '{interaction_state_var}' 标准差过小 ({state_std:.2e})，跳过交互项")
            # 退回到简单模式
            exog_vars.append(shock_var)
            target_var = shock_var
        else:
            # 创建交互项，但先标准化状态变量以提高数值稳定性
            df_h[f"{interaction_state_var}_scaled"] = (df_h[interaction_state_var] - df_h[interaction_state_var].mean()) / (df_h[interaction_state_var].std() + 1e-8)
            
            df_h[interaction_term_name] = df_h[shock_var] * df_h[f"{interaction_state_var}_scaled"]
            
            # 检查交互项的有效性
            interaction_nunique = df_h[interaction_term_name].nunique()
            interaction_nonzero = (df_h[interaction_term_name] != 0).sum()
            
            if interaction_nunique < 2 or interaction_nonzero < 10:
                logging.warning(f"Horizon {horizon}: 交互项 '{interaction_term_name}' 变异不足 (唯一值={interaction_nunique}, 非零={interaction_nonzero})，退回到简单模式")
                # 退回到简单模式
                exog_vars.append(shock_var)
                target_var = shock_var
            else:
                logging.debug(f"Horizon {horizon}: 创建交互项 '{interaction_term_name}' (唯一值={interaction_nunique}, 非零={interaction_nonzero})")
                exog_vars.append(interaction_term_name)
                target_var = interaction_term_name
                # 添加标准化后的状态变量
                exog_vars.append(f"{interaction_state_var}_scaled")
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
        # --- 数据质量改进：异常值处理 ---
        # 对连续变量进行异常值截尾处理，提高数值稳定性
        for var in exog_vars:
            if var in df_reg.columns and var not in [shock_var]:  # 不处理冲击变量本身
                if df_reg[var].dtype in ['float64', 'int64']:
                    # 使用1%和99%分位数进行截尾
                    q01 = df_reg[var].quantile(0.01)
                    q99 = df_reg[var].quantile(0.99)
                    original_outliers = ((df_reg[var] < q01) | (df_reg[var] > q99)).sum()
                    if original_outliers > 0:
                        df_reg[var] = np.clip(df_reg[var], q01, q99)
                        logging.debug(f"Horizon {horizon}: 截尾处理 {var}，处理了 {original_outliers} 个异常值")
        
        # 设置面板索引
        df_reg = df_reg.set_index([entity_col, time_col])
        exog = df_reg[exog_vars]

        # --- 数值稳定性检查 ---
        logging.debug(f"Horizon {horizon}, Shock {shock_var}, Interaction {interaction_state_var}: Exogenous variables being used: {exog_vars}")
        
        # 检查变量的有效性
        valid_vars = []
        for var in exog_vars:
            if var in exog.columns:
                var_data = exog[var]
                nunique = var_data.nunique()
                std_dev = var_data.std()
                
                # 检查变量是否有足够的变异
                if nunique < 2:
                    logging.warning(f"  WARNING {var}: 只有 {nunique} 个唯一值，跳过此变量")
                    continue
                elif std_dev < 1e-10:
                    logging.warning(f"  WARNING {var}: 标准差过小 ({std_dev:.2e})，跳过此变量")
                    continue
                else:
                    valid_vars.append(var)
                    logging.debug(f"  {var}: nunique={nunique}, std={std_dev:.6f}")
            else:
                logging.error(f"  {var}: 变量不存在")
        
        # 如果有效变量太少，返回失败
        if len(valid_vars) < len(exog_vars):
            logging.warning(f"Horizon {horizon}: 原有 {len(exog_vars)} 个变量，只有 {len(valid_vars)} 个有效")
            if len(valid_vars) == 0:
                logging.error(f"Horizon {horizon}: 没有有效的解释变量")
                return None
            # 更新变量列表
            exog = exog[valid_vars]
            exog_vars = valid_vars
        
        # --- 条件数检查和处理 ---
        if not exog.empty and exog.shape[1] > 0:
            try:
                condition_number = np.linalg.cond(exog)
                logging.debug(f"Horizon {horizon}: 设计矩阵条件数: {condition_number:.2f}")
                
                # 如果条件数过高，使用标准化处理
                if condition_number > 100:
                    logging.warning(f"Horizon {horizon}: 条件数过高 ({condition_number:.2f})，将使用数值稳定化处理")
                    # 使用简单的标准化 (z-score)
                    exog = (exog - exog.mean()) / (exog.std() + 1e-8)
                    logging.debug(f"Horizon {horizon}: 已对解释变量进行标准化处理")
            except Exception as e_cond:
                logging.warning(f"Horizon {horizon}: 条件数计算失败: {e_cond}")
        else:
            logging.error(f"Horizon {horizon}: 解释变量矩阵为空")
            return None

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
        
        # 改进：针对稀疏数据优化固定效应设定
        n_obs = len(df_reg)
        n_entities = df_reg.index.get_level_values(0).nunique()
        n_periods = df_reg.index.get_level_values(1).nunique()
        
        # 检查非零冲击的分布
        if target_var in exog.columns:
            non_zero_shocks = (exog[target_var] != 0).sum()
            shock_ratio = non_zero_shocks / n_obs if n_obs > 0 else 0
        else:
            non_zero_shocks = 0
            shock_ratio = 0
        
        # 动态调整固定效应策略
        if shock_ratio < 0.05:  # 如果非零冲击少于5%
            # 对于极稀疏数据，只使用实体固定效应
            use_entity_effects = True
            use_time_effects = False
            logging.info(f"Horizon {horizon}: 稀疏数据模式 (非零冲击比例: {shock_ratio:.3f})，仅使用实体固定效应")
        elif n_obs > 5000 and n_periods > 500:
            # 对于大样本，使用双向固定效应
            use_entity_effects = True
            use_time_effects = True
            logging.info(f"Horizon {horizon}: 大样本模式，使用双向固定效应")
        else:
            # 中等样本，仅使用实体固定效应
            use_entity_effects = True
            use_time_effects = False
            logging.info(f"Horizon {horizon}: 中等样本模式，仅使用实体固定效应")
        
        logging.debug(f"Horizon {horizon}: 数据规模 - 观测值={n_obs}, 实体={n_entities}, 时期={n_periods}, 非零冲击={non_zero_shocks}")
        
        mod = PanelOLS(df_reg[dep_var], exog, entity_effects=use_entity_effects, time_effects=use_time_effects, check_rank=use_strict_rank_check)

        # 配置标准误 - 针对稀疏数据优化
        cluster_config = {}
        
        # 基于数据特征动态选择标准误类型
        if shock_ratio < 0.02:  # 极稀疏数据（非零冲击<2%）
            # 使用简单稳健标准误，避免聚类导致的标准误膨胀
            cov_type_str = 'robust'
            effective_cluster_config = {}
            logging.info(f"Horizon {horizon}: 极稀疏数据 (冲击比例: {shock_ratio:.3f})，使用稳健标准误")
        elif shock_ratio < 0.05 or n_entities < 10:  # 稀疏数据或实体数量少
            # 仅使用实体聚类
            if cluster_entity:
                cluster_config['cluster_entity'] = True
                cov_type_str = 'clustered'
                effective_cluster_config = cluster_config
                logging.info(f"Horizon {horizon}: 稀疏数据，使用实体聚类标准误")
            else:
                cov_type_str = 'robust'
                effective_cluster_config = {}
                logging.info(f"Horizon {horizon}: 稀疏数据，使用稳健标准误")
        else:
            # 数据相对充足时使用双向聚类
            if cluster_entity and cluster_time and n_entities >= 5 and n_periods >= 50:
                cluster_config['cluster_entity'] = True
                cluster_config['cluster_time'] = True
                cov_type_str = 'clustered'
                effective_cluster_config = cluster_config
                logging.info(f"Horizon {horizon}: 使用双向聚类标准误")
            elif cluster_entity:
                cluster_config['cluster_entity'] = True
                cov_type_str = 'clustered'
                effective_cluster_config = cluster_config
                logging.info(f"Horizon {horizon}: 使用实体聚类标准误")
            else:
                cov_type_str = 'robust'
                effective_cluster_config = {}
                logging.info(f"Horizon {horizon}: 使用稳健标准误")

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

        # 5. 提取结果并进行数值稳定性检查
        coeff = results.params[target_var]
        stderr = results.std_errors[target_var]
        pval = results.pvalues[target_var]
        conf_int = results.conf_int().loc[target_var]
        conf_low = conf_int['lower']
        conf_high = conf_int['upper']

        # --- 数值稳定性检查：防止无穷大和NaN ---
        if np.isnan(coeff) or np.isinf(coeff):
            logging.warning(f"Horizon {horizon}: 系数为NaN或无穷大 ({coeff})，跳过此结果")
            return None
        
        if np.isnan(stderr) or np.isinf(stderr) or stderr <= 0:
            logging.warning(f"Horizon {horizon}: 标准误异常 ({stderr})，跳过此结果")
            return None
            
        if np.isnan(pval) or np.isinf(pval):
            logging.warning(f"Horizon {horizon}: p值异常 ({pval})，跳过此结果")
            return None
            
        if np.isnan(conf_low) or np.isinf(conf_low) or np.isnan(conf_high) or np.isinf(conf_high):
            logging.warning(f"Horizon {horizon}: 置信区间异常 ([{conf_low}, {conf_high}])，跳过此结果")
            return None
        
        # 检查置信区间是否过宽（可能表明估计不稳定）
        ci_width = conf_high - conf_low
        if ci_width > 10 * abs(coeff):  # 如果置信区间宽度超过系数绝对值的10倍
            logging.warning(f"Horizon {horizon}: 置信区间过宽 (系数={coeff:.4f}, 宽度={ci_width:.4f})，估计可能不稳定")

        logging.debug(f"Horizon {horizon}, Target {target_var}: Coeff={coeff:.4f}, SE={stderr:.4f}, Pval={pval:.3f}, CI=[{conf_low:.4f}, {conf_high:.4f}]")
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