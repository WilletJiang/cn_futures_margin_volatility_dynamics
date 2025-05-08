# -*- coding: utf-8 -*-
"""
多事件双重差分法 (Callaway & Sant'Anna, 2021) 分析

使用 rpy2 调用 R 的 did 包，估计节假日保证金调整对波动率的动态处理效应。
与原始 did_cs_analysis.py 不同，本脚本将每个节假日调整日视为独立的处理事件，
而不仅仅使用每个合约的首次保证金调整。

主要修改：
1. 创建合约-节假日事件标识符，将每个节假日调整日视为独立处理
2. 为每个事件设置独立的处理时间
3. 调整 DID 分析以处理增加的处理组数量
"""

import pandas as pd
import numpy as np
import os
import logging
import warnings

# --- rpy2 设置 ---
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib.embedded import RRuntimeError
    from rpy2.rinterface import RRuntimeWarning # 用于捕获 R 警告
    # 忽略 rpy2 产生的特定警告 (例如关于 R 版本)
    warnings.filterwarnings("ignore", category=RRuntimeWarning)
    # 激活 pandas 到 R data.frame 的自动转换 (可以在函数内部或全局激活)
    pandas2ri.activate()
except ImportError:
    logging.error("rpy2 未安装或配置不正确。请确保已安装 rpy2 并且 R 环境可访问。")
    exit()
except RRuntimeError as e:
     logging.error(f"无法初始化 R 环境: {e}. 请检查 R 安装和 R_HOME 环境变量。")
     exit()


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

# --- R 包导入与检查 ---
try:
    R = ro.r
    base = importr('base')
    stats = importr('stats')
    did = importr('did')
    # dplyr = importr('dplyr') # 可能需要
    logging.info("R 环境和 did 包加载成功。")
except RRuntimeError as e:
    logging.error(f"无法加载 R 包 'did': {e}")
    logging.error("请确保已在 R 环境中安装 'did' 包 (install.packages('did'))。")
    exit()
except Exception as e:
     logging.error(f"导入 R 包时发生未知错误: {e}")
     exit()


# --- 辅助函数：保存结果 ---
def save_results_df(df, filename, directory):
    """安全地保存 DataFrame 到 CSV"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    try:
        df.to_csv(filepath, index=False)
        logging.info(f"结果已保存到: {filepath}")
        return True
    except Exception as e:
        logging.error(f"保存结果到 {filepath} 时出错: {e}")
        return False

# --- 多事件 DID 核心分析函数 ---

def run_multi_event_did_analysis_core(data, outcome_var, output_table_dir, output_suffix="_multi_event",
                                      control_vars=None,
                                      control_group_type="notyettreated", # "notyettreated" or "nevertreated"
                                      est_method="dr"):
    """
    执行多事件 Callaway & Sant'Anna DID 分析的核心逻辑。
    
    与原始 DID 分析不同，此函数将每个节假日保证金调整日期视为独立的处理事件，
    而不仅使用每个合约的首次调整日期。这允许分析师研究不同节假日调整的异质性效应。

    Args:
        data (pd.DataFrame): 输入面板数据。
        outcome_var (str): 结果变量名称。
        output_table_dir (str): 保存结果表格的目录。
        output_suffix (str, optional): 添加到输出文件名末尾的后缀。Defaults to "_multi_event".
        control_vars (list, optional): 控制变量列表。如果为 None，则从 config 加载。Defaults to None.
        control_group_type (str, optional): 控制组类型 ('notyettreated' 或 'nevertreated')。 Defaults to "notyettreated".
        est_method (str, optional): 估计方法 ('dr' 或 'ipw')。 Defaults to "dr".

    Returns:
        bool: 如果分析成功完成则返回 True，否则返回 False。
    """
    logging.info(f"--- 开始多事件 DID 分析 (Outcome: {outcome_var}, Suffix: '{output_suffix}') ---")

    if control_vars is None:
        control_vars = config.CONTROL_VARIABLES

    df = data.copy() # 使用副本

    # --- 1. 数据准备 ---
    logging.info("准备数据以适配多事件 DID 分析...")
    if outcome_var not in df.columns:
        logging.error(f"结果变量 '{outcome_var}' 不在数据中。")
        return False
        
    # 检查必要的列
    required_cols = ['contract_id', 'date', 'is_holiday_adjustment_day', 'treat_group_g_label', outcome_var]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"数据中缺少必要的列: {missing_cols}")
        return False
        
    # --- 1.1 创建多事件处理标识 ---
    logging.info("创建多事件处理标识...")
    
    # 找出所有是节假日调整的行 (is_holiday_adjustment_day == 1)
    holiday_adj_rows = df[df['is_holiday_adjustment_day'] == 1].copy()
    
    if holiday_adj_rows.empty:
        logging.error("数据中没有找到节假日调整事件 (is_holiday_adjustment_day == 1)，无法进行多事件分析。")
        return False
        
    # 为每个合约的每个节假日调整创建唯一事件ID
    # 格式: contract_id + "_" + holiday_window_label + "_" + date
    holiday_adj_rows['event_id'] = (
        holiday_adj_rows['contract_id'].astype(str) + "_" + 
        holiday_adj_rows['treat_group_g_label'].fillna('未知') + "_" + 
        holiday_adj_rows['date'].dt.strftime('%Y%m%d')
    )
    
    # 为每个事件ID分配处理时间
    event_treatment_dates = holiday_adj_rows[['event_id', 'date']].drop_duplicates()
    event_treatment_dates = event_treatment_dates.rename(columns={'date': 'treatment_date'})
    
    # 创建事件ID到处理日期的映射
    event_id_to_date = dict(zip(event_treatment_dates['event_id'], event_treatment_dates['treatment_date']))
    
    # 为每一行创建潜在的事件ID
    df['temp_event_id'] = (
        df['contract_id'].astype(str) + "_" + 
        df['treat_group_g_label'].fillna('未知') + "_" + 
        df['date'].dt.strftime('%Y%m%d')
    )
    
    # 创建一个包含所有事件ID的列表
    all_event_ids = list(event_id_to_date.keys())
    
    # 创建一个新的DataFrame来存储扩展后的面板数据
    expanded_rows = []
    
    # 统计信息
    total_events = len(all_event_ids)
    logging.info(f"找到 {total_events} 个独立的节假日调整事件。")
    
    # 对每个事件分别为所有观测值创建处理时间
    for event_id in all_event_ids:
        # 提取事件的组成部分
        parts = event_id.split("_")
        contract_id = parts[0]  # 假设合约ID不包含下划线
        holiday_label = "_".join(parts[1:-1])  # 处理组标签可能包含下划线（如"春节_2010"）
        
        # 只选择相应合约和处理组标签的行
        contract_rows = df[(df['contract_id'] == contract_id) & (df['treat_group_g_label'] == holiday_label)].copy()
        
        if contract_rows.empty:
            continue
            
        # 设置此事件的处理日期
        treatment_date = event_id_to_date[event_id]
        
        # 为此事件创建事件特定的列
        contract_rows['event_id'] = event_id
        contract_rows['event_treatment_date'] = treatment_date
        
        # 计算该事件的时间周期（年月格式）
        contract_rows['event_treatment_period'] = (
            treatment_date.year * 100 + treatment_date.month
        )
        
        # 添加到扩展数据中
        expanded_rows.append(contract_rows)
    
    # 合并所有扩展的行
    if not expanded_rows:
        logging.error("处理后的扩展数据为空，无法继续分析。")
        return False
        
    expanded_df = pd.concat(expanded_rows, ignore_index=True)
    
    # 创建时间周期列（年月格式）
    expanded_df['time_period'] = expanded_df['date'].dt.year * 100 + expanded_df['date'].dt.month
    
    # 使用事件特定的处理时间作为first_treat_period
    expanded_df['first_treat_period'] = expanded_df['event_treatment_period']
    
    # 检查数据
    logging.info(f"多事件扩展后的数据形状: {expanded_df.shape}")
    logging.info(f"时间周期范围: {expanded_df['time_period'].min()} 到 {expanded_df['time_period'].max()}")
    logging.info(f"处理时间范围: {expanded_df['first_treat_period'].min()} 到 {expanded_df['first_treat_period'].max()}")
    
    # 使用扩展的数据框继续分析
    df = expanded_df

    # 处理从未处理的组
    if control_group_type == "nevertreated":
        df['first_treat_period'].fillna(np.inf, inplace=True)
        logging.info("将从未处理组的 first_treat_period 设置为 Inf (配合 control_group='nevertreated')")
    else: # 默认为 notyettreated
        df['first_treat_period'].fillna(0, inplace=True)
        logging.info("将从未处理组的 first_treat_period 设置为 0 (配合 control_group='notyettreated')")

    # 检查控制变量
    valid_controls = [c for c in control_vars if c in df.columns]
    if len(valid_controls) != len(control_vars):
        missing_ctrl = set(control_vars) - set(valid_controls)
        logging.warning(f"缺少部分控制变量: {missing_ctrl}，将仅使用存在的控制变量。")
        control_vars = valid_controls # 更新为实际存在的控制变量

    # 选择列并处理缺失值
    # 为R分析创建唯一标识符
    df['numeric_id'] = pd.factorize(df['event_id'])[0] + 1
    cols_to_r = ['numeric_id', 'time_period', 'first_treat_period', outcome_var] + control_vars
    
    # 检查所有需要的列是否存在
    missing_cols_final = [c for c in cols_to_r if c not in df.columns]
    if missing_cols_final:
        logging.error(f"数据准备阶段缺少必需列: {missing_cols_final}")
        return False
        
    df_r = df[cols_to_r].copy()
    rows_before = df_r.shape[0]
    # 诊断：打印各列缺失值数量
    na_counts = df_r.isna().sum()
    logging.info("DID 数据缺失统计:\n" + na_counts.to_string())
    # 仅删除关键列缺失的行，避免删除所有观测
    df_r.dropna(subset=['numeric_id', 'time_period', 'first_treat_period', outcome_var], inplace=True)
    logging.info(f"因缺失值删除 {rows_before - df_r.shape[0]} 行，剩余 {df_r.shape[0]} 行。")
    # 删除控制变量的缺失观测，以免 att_gt 预处理阶段剔除全部数据
    rows_before_ctrl = df_r.shape[0]
    if control_vars:
        df_r.dropna(subset=control_vars, inplace=True)
        logging.info(f"因控制变量缺失删除 {rows_before_ctrl - df_r.shape[0]} 行，剩余 {df_r.shape[0]} 行。")

    # --- 2. 转换数据到 R 并执行 DID 估计 ---
    logging.info("将数据转换到 R 并执行多事件 DID 分析...")
    
    # 检查处理后的数据是否为空
    if df_r.empty:
        logging.error("处理后的数据为空，无法进行 DID 分析。")
        return False
        
    # 检查处理组的数量
    num_treat_groups = df_r['first_treat_period'].nunique()
    logging.info(f"处理后的数据包含 {num_treat_groups} 个不同的处理组。")
    
    try:
        # 转换 pandas DataFrame 到 R DataFrame 并将其放入R全局环境
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_r)
            # 显式地将r_df放入R全局环境
            ro.globalenv['r_df'] = r_df
        
        # R 代码：设置 att_gt DID 分析
        logging.info(f"正在执行 did::att_gt 函数，控制组类型: {control_group_type}，估计方法: {est_method}...")
        
        # 设置 R 代码
        r_code = f"""
        suppressWarnings(library(did))
        
        # 执行 att_gt
        att_gt_results <- suppressWarnings(
            att_gt(
                yname = "{outcome_var}",
                gname = "first_treat_period",
                idname = "numeric_id",
                tname = "time_period",
                xformla = {' ~ ' + ' + '.join(control_vars) if control_vars else 'NULL'},
                data = r_df,
                panel = FALSE,
                control_group = "{control_group_type}",
                bstrap = TRUE,
                cband = TRUE,
                est_method = "{est_method}",
                print_details = FALSE,
                clustervars = "numeric_id"
            )
        )
        
        # 聚合结果 - 汇总平均处理效应
        agg_results <- suppressWarnings(
            aggte(att_gt_results, 
                  type = "dynamic", 
                  min_e = -12,  # 关注处理前12个月到处理后12个月
                  max_e = 12)
        )
        
        # 将 att_gt 的原始结果转换为数据框
        att_gt_df <- as.data.frame(att_gt_results$att)
        att_gt_df$se <- att_gt_results$se
        att_gt_df$ci_lower <- att_gt_results$att - 1.96 * att_gt_results$se
        att_gt_df$ci_upper <- att_gt_results$att + 1.96 * att_gt_results$se
        att_gt_df$p_value <- 2 * pnorm(-abs(att_gt_results$att / att_gt_results$se))
        
        # 汇总聚合结果
        agg_df <- data.frame(
            event_time = agg_results$egt,
            att = agg_results$att.egt,
            se = agg_results$se.egt,
            ci_lower = agg_results$att.egt - 1.96 * agg_results$se.egt,
            ci_upper = agg_results$att.egt + 1.96 * agg_results$se.egt,
            p_value = 2 * pnorm(-abs(agg_results$att.egt / agg_results$se.egt))
        )
        
        # 统计信息
        stats_df <- data.frame(
            metric = c(
                "Total Observations",
                "Treated Units",
                "Control Units",
                "Treatment Groups",
                "First Treatment Period",
                "Last Treatment Period"
            ),
            value = c(
                nrow(r_df),
                length(unique(r_df$numeric_id[r_df$first_treat_period > 0])),
                length(unique(r_df$numeric_id[r_df$first_treat_period == 0 | is.infinite(r_df$first_treat_period)])),
                length(unique(r_df$first_treat_period[!is.na(r_df$first_treat_period) & r_df$first_treat_period > 0])),
                min(r_df$first_treat_period[r_df$first_treat_period > 0], na.rm=TRUE),
                max(r_df$first_treat_period[r_df$first_treat_period > 0], na.rm=TRUE)
            )
        )
        """
        
        # 运行 R 代码
        R(r_code)
        
        # 从 R 工作空间获取结果
        with localconverter(ro.default_converter + pandas2ri.converter):
            att_gt_results = ro.conversion.rpy2py(R("att_gt_df"))
            agg_results = ro.conversion.rpy2py(R("agg_df"))
            stats_df = ro.conversion.rpy2py(R("stats_df"))
            
        # 记录统计信息
        logging.info("DID 分析统计信息：")
        for _, row in stats_df.iterrows():
            logging.info(f"- {row['metric']}: {row['value']}")
            
        # --- 3. 保存结果 ---
        # 设置文件名
        stats_file = f"did_stats{output_suffix}.csv"
        att_gt_file = f"did_att_gt_results{output_suffix}.csv"
        agg_file = f"did_aggregate_results{output_suffix}.csv"
        
        # 保存统计信息
        save_results_df(stats_df, stats_file, output_table_dir)

        # 仅当 att_gt_results 有效时才保存
        if att_gt_results is not None and hasattr(att_gt_results, 'empty') and not att_gt_results.empty:
            save_results_df(att_gt_results, att_gt_file, output_table_dir)
        else:
            logging.warning(f"att_gt_results 为空或无效，跳过保存 {att_gt_file}")

        # 仅当 agg_results 有效时才保存
        if agg_results is not None and hasattr(agg_results, 'empty') and not agg_results.empty:
            save_results_df(agg_results, agg_file, output_table_dir)
        else:
            logging.warning(f"agg_results 为空或无效，跳过保存 {agg_file}")
        
        # --- 4. 结束分析 ---
        logging.info(f"多事件 DID 分析完成，结果已保存到 {output_table_dir}")
        return True
        
    except RRuntimeError as e:
        logging.error(f"R 运行时错误: {e}")
        # 尝试提取 R 警告信息
        try:
            warnings_list = R("warnings()")
            logging.error(f"R 警告: {warnings_list}")
        except:
            pass
        return False
    except Exception as e:
        logging.error(f"执行 DID 分析时发生未知错误: {e}")
        return False


def run_did_analysis_multi_event(data, outcome_var, output_table_dir, output_suffix="_multi_event",
                               control_vars=None,
                               control_group_type="notyettreated", 
                               est_method="dr"):
    """
    多事件 DID 分析的外部入口函数。
    
    此函数是对核心分析功能的简单包装，提供标准化的接口和额外的错误处理。
    
    Args:
        data (pd.DataFrame): 输入面板数据。
        outcome_var (str): 结果变量名称。
        output_table_dir (str): 保存结果表格的目录。
        output_suffix (str, optional): 添加到输出文件名末尾的后缀。默认为 "_multi_event"。
        control_vars (list, optional): 控制变量列表。如果为 None，则从 config 加载。默认为 None。
        control_group_type (str, optional): 控制组类型 ('notyettreated' 或 'nevertreated')。默认为 "notyettreated"。
        est_method (str, optional): 估计方法 ('dr' 或 'ipw')。默认为 "dr"。
        
    Returns:
        bool: 如果分析成功完成则返回 True，否则返回 False。
    """
    try:
        logging.info(f"开始多事件 DID 分析，结果变量: {outcome_var}")
        return run_multi_event_did_analysis_core(
            data=data,
            outcome_var=outcome_var,
            output_table_dir=output_table_dir,
            output_suffix=output_suffix,
            control_vars=control_vars,
            control_group_type=control_group_type,
            est_method=est_method
        )
    except Exception as e:
        logging.error(f"多事件 DID 分析失败: {e}")
        return False


if __name__ == "__main__":
    import pandas as pd
    import logging
    from src import config

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("直接运行 did_cs_analysis_multi_event.py 脚本...")

    try:
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
            logging.error(f"主面板数据文件未找到: {config.PANEL_DATA_FILEPATH}")
            logging.error("请先运行 src/data_processing/build_features.py 生成数据。")
            exit(1)
        
        main_df = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        main_df['date'] = pd.to_datetime(main_df['date'])  # 确保时间格式正确
        logging.info(f"主数据加载成功，数据条数: {main_df.shape[0]}")

        # 调用多事件DID分析核心函数
        success = run_did_analysis_multi_event(
            data=main_df,
            outcome_var='log_gk_volatility',
            output_table_dir=config.PATH_OUTPUT_TABLES,
            control_vars=None,
            control_group_type='notyettreated',
            est_method='dr'
        )
        if success:
            logging.info("多事件DID分析成功完成。")
        else:
            logging.error("多事件DID分析执行失败。")
    
    except Exception as e:
        logging.error(f"运行多事件DID分析时出现异常: {e}")

