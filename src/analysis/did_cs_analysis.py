# -*- coding: utf-8 -*-
"""
双重差分法 (Callaway & Sant'Anna, 2021) 分析

使用 rpy2 调用 R 的 did 包，估计节假日保证金调整对波动率的动态处理效应。
包含可被其他脚本调用的核心分析函数 run_did_analysis_core。
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

# --- 重构后的核心分析函数 ---

def run_did_analysis_core(data, outcome_var, output_table_dir, output_suffix="",
                          control_vars=None,
                          control_group_type="notyettreated", # "notyettreated" or "nevertreated"
                          est_method="dr"):
    """
    执行 Callaway & Sant'Anna DID 分析的核心逻辑。

    Args:
        data (pd.DataFrame): 输入面板数据。
        outcome_var (str): 结果变量名称。
        output_table_dir (str): 保存结果表格的目录。
        output_suffix (str, optional): 添加到输出文件名末尾的后缀。Defaults to "".
        control_vars (list, optional): 控制变量列表。如果为 None，则从 config 加载。Defaults to None.
        control_group_type (str, optional): 控制组类型 ('notyettreated' 或 'nevertreated')。 Defaults to "notyettreated".
        est_method (str, optional): 估计方法 ('dr' 或 'ipw')。 Defaults to "dr".

    Returns:
        bool: 如果分析成功完成则返回 True，否则返回 False。
    """
    logging.info(f"--- 开始核心 DID 分析 (Outcome: {outcome_var}, Suffix: '{output_suffix}') ---")
    analysis_successful = True # 初始化整体分析成功状态

    if control_vars is None:
        control_vars = config.CONTROL_VARIABLES

    df_did = data.copy() # 使用副本

    # --- 1. 数据准备 ---
    logging.info("准备数据以适配 R did 包...")
    if outcome_var not in df_did.columns:
        logging.error(f"结果变量 '{outcome_var}' 不在数据中。")
        return False

    df_did['time_period'] = df_did['date'].dt.year * 100 + df_did['date'].dt.month
    # 确保 treat_date_g 是 datetime 类型，以便 .dt 访问器工作
    # 如果它已经是 datetime64[ns] 类型则此行无害
    df_did['treat_date_g'] = pd.to_datetime(df_did['treat_date_g'], errors='coerce') 
    df_did['first_treat_period'] = df_did['treat_date_g'].dt.year * 100 + df_did['treat_date_g'].dt.month

    # 处理从未处理的组 (NaT 在 first_treat_period 中会是 NaN)
    if control_group_type == "nevertreated":
        df_did['first_treat_period'].fillna(np.inf, inplace=True)
        logging.info("将从未处理组的 first_treat_period 设置为 Inf (配合 control_group='nevertreated')")
    else: # 默认为 notyettreated
        df_did['first_treat_period'].fillna(0, inplace=True)
        logging.info("将从未处理组的 first_treat_period 设置为 0 (配合 control_group='notyettreated')")

    # 检查实际存在的控制变量
    actual_control_vars = [c for c in control_vars if c in df_did.columns]
    if len(actual_control_vars) != len(control_vars):
        missing_ctrl = set(control_vars) - set(actual_control_vars)
        logging.warning(f"DID分析中缺少部分原始控制变量: {missing_ctrl}。将仅使用实际存在的控制变量: {actual_control_vars}")
    # 如果 control_vars 为空或 None，actual_control_vars 也将为空列表，后续逻辑正确

    if 'contract_id' not in df_did.columns:
         logging.error("数据中缺少 'contract_id' 列。")
         return False
    df_did['numeric_id'] = pd.factorize(df_did['contract_id'])[0] + 1
    
    # --- 修改点：针对性 dropna ---
    # 列出 DID 模型必需的核心列
    core_did_cols = ['numeric_id', 'time_period', 'first_treat_period', outcome_var]
    # 所有 DID 分析实际需要的列 (actual_control_vars 可能为空列表)
    cols_for_did_model = core_did_cols + actual_control_vars 
    
    # 检查这些列是否存在于 df_did (在进行选择之前)
    missing_cols_in_df_did = [c for c in cols_for_did_model if c not in df_did.columns]
    if missing_cols_in_df_did:
        # 这通常不应该发生，因为 actual_control_vars 已经是 df_did 中存在的
        logging.error(f"逻辑错误：以下目标列在 df_did 中缺失: {missing_cols_in_df_did}")
        return False
        
    # 从 df_did 中只选取模型实际需要的列到 df_r
    df_r = df_did[cols_for_did_model].copy()
    
    rows_before_na_did = df_r.shape[0]
    # 对这些选定的列执行 dropna
    # 注意：如果 actual_control_vars 为空，dropna 仍会正确处理核心列
    df_r.dropna(inplace=True)
    logging.info(f"针对DID模型所需列 ({cols_for_did_model}) 进行dropna后，删除 {rows_before_na_did - df_r.shape[0]} 行，剩余 {df_r.shape[0]} 行。")

    if df_r.empty:
        logging.error("针对DID模型所需列进行dropna后数据为空，无法进行 DID 分析。")
        return False

    # --- 诊断：检查处理前时期 ---
    logging.info("--- 诊断：检查处理前时期 ---")
    # 获取所有唯一的处理组开始时间 (假设0或inf是特殊值，不代表实际处理时间)
    unique_g_periods = sorted([g for g in df_r['first_treat_period'].unique() if g > 0 and g != np.inf])

    all_groups_have_pre_treatment = True
    for g_val in unique_g_periods:
        group_data = df_r[df_r['first_treat_period'] == g_val]
        if not group_data.empty:
            min_time_for_group = group_data['time_period'].min()
            logging.info(f"处理组 G={g_val}: 最早的观测时间 T_min={min_time_for_group}")
            pre_treatment_periods_count = group_data[group_data['time_period'] < g_val]['time_period'].nunique()
            logging.info(f"处理组 G={g_val}: 拥有的严格处理前时期数量: {pre_treatment_periods_count}")
            if pre_treatment_periods_count < 2:
                logging.warning(f"警告：处理组 G={g_val} 严格处理前时期数量为 {pre_treatment_periods_count}，少于2个。这可能导致 att_gt 估计失败或结果不可靠。")
                all_groups_have_pre_treatment = False
            # 你可以根据 did 包的要求增加更严格的检查，比如 pre_treatment_periods_count < 2
        else:
            # 这通常不应该发生，因为 unique_g_periods 是从 df_r 本身提取的
            logging.warning(f"诊断信息：处理组 G={g_val} 在 df_r 中没有数据，这很奇怪。")
    if not all_groups_have_pre_treatment:
        logging.error("一个或多个处理组缺少严格的处理前时期。这很可能是导致 R did::att_gt 无法计算或返回空结果的原因。")
    logging.info("--- 完成处理前时期诊断 ---")

    # --- 2. 转换为 R Data Frame ---
    logging.info("将数据转换为 R data.frame...")
    try:
        # 在函数内部使用 localconverter 确保转换作用域
        # 将 first_treat_period 和 time_period 列转换为数值型 (df_r 中的这些列已经是数值型或可以安全转换)
        df_r['first_treat_period'] = pd.to_numeric(df_r['first_treat_period'], errors='coerce').astype(int)
        df_r['time_period'] = pd.to_numeric(df_r['time_period'], errors='coerce').astype(int)
        # outcome_var 和 control_vars 中的变量应在 dropna 后是有效的数值类型
        # numeric_id 已经是 int
        logging.info("已将 df_r 中的 first_treat_period 和 time_period 转换为数值型 (为R准备)。")

        # !!! 根据最新诊断，G=201002 和 G=202004 处理前时期不足 !!!
        problematic_group_g_values = [201002, 202004] 
        rows_before_problem_removal = df_r.shape[0]
        df_r = df_r[~df_r['first_treat_period'].isin(problematic_group_g_values)]
        rows_after_problem_removal = df_r.shape[0]
        if rows_before_problem_removal > rows_after_problem_removal:
            logging.info(f"根据诊断移除了处理组 G 值在 {problematic_group_g_values} 中的 {rows_before_problem_removal - rows_after_problem_removal} 行数据。")
        
        if df_r.empty:
            logging.error(f"移除问题组 {problematic_group_g_values} 后数据为空，无法继续分析。")
            return False
        
        # 打印 r_df 的详细信息以供诊断
        logging.info(f"传递给 R 的 DataFrame (r_df) 的形状: {df_r.shape}")
        logging.info(f"r_df 列的数据类型:\n{df_r.dtypes.to_string()}")
        logging.info(f"r_df 的描述性统计:\n{df_r.describe(include='all').to_string()}")
        for col in ['numeric_id', 'time_period', 'first_treat_period', outcome_var]:
            if col in df_r.columns:
                logging.info(f"列 '{col}' 的唯一值数量: {df_r[col].nunique()}")
                # logging.info(f"列 '{col}' 的唯一值 (前10个): {df_r[col].unique()[:10]}") # 如果需要可以取消注释
            else:
                logging.warning(f"诊断信息：列 '{col}' 不在 r_df 中。")

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_r)
    except Exception as e:
        logging.error(f"Pandas DataFrame 转换为 R data.frame 时出错: {e}")
        return False

    # --- 3. 执行 R did::att_gt ---
    logging.info("调用 R did::att_gt 函数...")
    att_gt_results = None
    try:
        yname = outcome_var
        tname = "time_period"
        idname = "numeric_id"
        gname = "first_treat_period"
        data_r = r_df
        control_group_r = ro.StrVector([control_group_type])
        est_method_r = est_method
        panel_r = True
        allow_unbalanced_r = True

        # 控制变量 formula
        if control_vars:
            controls_formula_str = " ~ " + " + ".join(control_vars)
            xformla_r = ro.Formula(controls_formula_str)
        else:
            xformla_r = ro.NULL

        # 忽略 R 警告执行
        with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=RRuntimeWarning)
             att_gt_results = did.att_gt(
                 yname=yname, tname=tname, idname=idname, gname=gname,
                 data=data_r, xformla=xformla_r, control_group=control_group_r,
                 est_method=est_method, panel=panel_r,
                 allow_unbalanced_panel=allow_unbalanced_r,
                 weightsname=ro.NULL,
                 bstrap=True, cband=True
             )
        logging.info("did::att_gt 函数执行成功。")

    except RRuntimeError as e:
        logging.warning(f"ATT(g,t) 分析失败。R 返回错误: {e}")
        # 尝试获取更详细的 R 调用栈或错误信息，但这依赖于 rpy2 的具体行为和 R 错误的类型
        try:
            # r_traceback = R('traceback()') # R traceback() 可能只在交互式会话中有用
            # logging.warning(f"R traceback (可能不完整或不可用):\n{str(r_traceback)}")
            r_warnings = R('warnings()')
            if r_warnings != ro.NULL:
                 logging.warning(f"R warnings() 输出:\n{str(r_warnings)}")
        except Exception as e_diag:
            logging.warning(f"尝试获取 R 诊断信息时出错: {e_diag}")

        save_results_df(pd.DataFrame(), f"did_att_gt_results{output_suffix}.csv", output_table_dir)
        analysis_successful = False # 更新分析状态
        att_gt_results = None
    except Exception as e:
        logging.warning(f"未知错误执行 ATT(g,t): {e}，保存空文件继续。")
        save_results_df(pd.DataFrame(), f"did_att_gt_results{output_suffix}.csv", output_table_dir)
        analysis_successful = False # 更新分析状态
        att_gt_results = None

    # --- 4. 保存 att(g,t) 结果 ---
    if att_gt_results is not None: # 仅当 did.att_gt 没有在 Python 层面立即失败时
        att_gt_summary_df = pd.DataFrame()
        try:
            summary_att_gt_r_obj = R.summary(att_gt_results)

            if summary_att_gt_r_obj == ro.NULL:
                logging.warning("R.summary(att_gt_results) 返回 NULL。这表明 att_gt 未能生成有效结果或结果为空。ATT(g,t) 摘要将记录为空，并且 AGGTE 分析将被跳过。")
                try:
                    # 尝试获取 att_gt_results 的 R 类名
                    r_class = R['class'](att_gt_results)
                    logging.info(f"att_gt_results 对象的 R 类名: {str(list(r_class))}")
                except Exception as e_class:
                    logging.warning(f"获取 att_gt_results 的 R 类名时出错: {e_class}")

                # att_gt_summary_df 保持为空 DataFrame
                analysis_successful = False # att_gt 结果无效，标记分析失败
                att_gt_results = None     # 设置为 None 以跳过 aggte
            else:
                with localconverter(ro.default_converter + pandas2ri.converter):
                    converted_summary = ro.conversion.rpy2py(summary_att_gt_r_obj)

                if isinstance(converted_summary, pd.DataFrame):
                    att_gt_summary_df = converted_summary
                    if not att_gt_summary_df.empty:
                        logging.info("ATT(g,t) 摘要结果成功提取并转换为 DataFrame。")
                    else:
                        logging.info("ATT(g,t) 摘要结果提取为空的 DataFrame。")
                else:
                    logging.warning(f"R summary of att_gt_results 转换为 Python 对象类型 {type(converted_summary)}，而非预期的 DataFrame。ATT(g,t) 摘要将记录为空。")
                    # att_gt_summary_df 保持为空 DataFrame
                    # 即使转换类型不对，也尝试保存空表，但不认为是"成功"的摘要提取
                    analysis_successful = False # 摘要提取未得到预期 DataFrame

            if not save_results_df(att_gt_summary_df, f"did_att_gt_results{output_suffix}.csv", output_table_dir):
                analysis_successful = False # 如果保存 ATT(g,t) 摘要失败

        except Exception as e:
            logging.error(f"提取或保存 ATT(g,t) 摘要结果时出错: {e}。将尝试保存空的 ATT(g,t) 摘要文件。")
            save_results_df(pd.DataFrame(), f"did_att_gt_results{output_suffix}.csv", output_table_dir)
            analysis_successful = False # 标记分析失败
            att_gt_results = None     # 设置为 None 以确保跳过 aggte
    else:
        # 此分支在 att_gt_results 为 None 时执行 (即 did.att_gt 早期失败，analysis_successful 已为 False)
        logging.info("att_gt_results 为 None (did.att_gt 调用失败或结果无效)，跳过 ATT(g,t) 摘要的提取和保存。")


    # 如果 att_gt_results 为空 (因上述任一原因变为 None)，直接保存空的聚合文件并跳过 aggte
    if att_gt_results is None:
        # 生成带有标准列的空聚合结果，供绘图脚本读取
        empty_agg_df = pd.DataFrame(columns=['event_time', 'estimate', 'conf.low', 'conf.high'])
        save_results_df(empty_agg_df, f"did_aggregate_results{output_suffix}.csv", output_table_dir) # 保存这个空文件不应改变 analysis_successful 状态
        logging.warning("由于 ATT(g,t) 阶段问题或结果无效，跳过 AGGTE 分析。已生成带有表头的空 did_aggregate_results 文件。")
        logging.info(f"--- 完成核心 DID 分析 (Outcome: {outcome_var}, Suffix: '{output_suffix}') ---")
        return analysis_successful # 返回由之前步骤决定的 analysis_successful 状态

    # --- 5. 执行 R did::aggte
    # 如果代码执行到这里，意味着 att_gt_results 不是 None 且 analysis_successful 仍然是 True
    logging.info("调用 R did::aggte 函数计算聚合效应...")
    agg_results_df = pd.DataFrame() # 初始化为空
    try:
        # type="dynamic" 获取事件研究估计值
        # min_e, max_e 定义事件研究窗口期 (相对于处理)
        # 使用 config 中的窗口期
        min_e = -config.DID_EVENT_WINDOW_PRE
        max_e = config.DID_EVENT_WINDOW_POST

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RRuntimeWarning)
            agg_results = did.aggte(
                att_gt_results, # 使用 att_gt 的结果作为输入
                type = "dynamic",
                min_e = min_e,
                max_e = max_e,
            )
        logging.info("did::aggte 函数执行成功。")

        # 提取聚合结果 (aggte 的 summary 通常更适合绘图)
        summary_agg = R.summary(agg_results)
        with localconverter(ro.default_converter + pandas2ri.converter):
            agg_results_df = ro.conversion.rpy2py(summary_agg)

        # 重命名列以便绘图脚本使用 (aggte summary 可能有 'term', 'estimate', 'conf.low', 'conf.high')
        agg_results_df.rename(columns={'term': 'event_time', 'point.estimate': 'estimate'}, inplace=True, errors='ignore')

        logging.info("聚合效应结果提取成功。")
        if not save_results_df(agg_results_df, f"did_aggregate_results{output_suffix}.csv", output_table_dir):
             analysis_successful = False # 保存失败则标记

    except RRuntimeError as e:
        logging.error(f"执行 R did::aggte 时出错: {e}")
        analysis_successful = False
    except Exception as e:
        logging.error(f"提取或保存聚合效应结果时出错: {e}")
        analysis_successful = False


    logging.info(f"--- 完成核心 DID 分析 (Outcome: {outcome_var}, Suffix: '{output_suffix}') ---")
    return analysis_successful


# --- 主执行块 (用于直接运行此脚本) ---
if __name__ == "__main__":
    logging.info("直接运行 did_cs_analysis.py 脚本...")

    # 加载主数据
    try:
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
             logging.error(f"主面板数据文件未找到: {config.PANEL_DATA_FILEPATH}")
             logging.error("请先运行 src/data_processing/build_features.py")
             exit()
        main_df = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        main_df['date'] = pd.to_datetime(main_df['date']) # 确保日期类型
        logging.info(f"主数据加载成功: {main_df.shape}")

        # 执行默认分析
        success = run_did_analysis_core(data=main_df,
                                        outcome_var='log_gk_volatility',
                                        output_table_dir=config.PATH_OUTPUT_TABLES,
                                        output_suffix="") # 无后缀表示主分析
        if success:
            logging.info("默认 DID 分析成功完成。")
        else:
            logging.error("默认 DID 分析执行过程中出现错误。")

    except Exception as e:
        logging.error(f"直接运行 DID 分析时出错: {e}")

    # 停用转换器 (如果全局激活)
    # pandas2ri.deactivate()