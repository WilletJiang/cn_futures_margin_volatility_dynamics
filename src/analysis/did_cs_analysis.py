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

    if control_vars is None:
        control_vars = config.CONTROL_VARIABLES

    df = data.copy() # 使用副本

    # --- 1. 数据准备 ---
    logging.info("准备数据以适配 R did 包...")
    if outcome_var not in df.columns:
        logging.error(f"结果变量 '{outcome_var}' 不在数据中。")
        return False

    # 时间和分组变量
    df['time_period'] = df['date'].dt.year
    df['first_treat_period'] = df['treat_date_g'].dt.year

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
    # 确保 contract_id 存在
    if 'contract_id' not in df.columns:
         logging.error("数据中缺少 'contract_id' 列。")
         return False
    # 新增数值型ID列，供 R did 包使用
    df['numeric_id'] = pd.factorize(df['contract_id'])[0] + 1
    cols_to_r = ['numeric_id', 'time_period', 'first_treat_period', outcome_var] + control_vars
    
    # 检查所有需要的列是否存在
    missing_cols_final = [c for c in cols_to_r if c not in df.columns]
    if missing_cols_final:
        logging.error(f"数据准备阶段缺少必需列: {missing_cols_final}")
        return False
        
    df_r = df[cols_to_r].copy()
    rows_before = df_r.shape[0]
    # 仅删除关键列缺失的行，避免删除所有观测
    df_r.dropna(subset=['numeric_id', 'time_period', 'first_treat_period', outcome_var], inplace=True)
    logging.info(f"因缺失值删除 {rows_before - df_r.shape[0]} 行，剩余 {df_r.shape[0]} 行。")

    if df_r.empty:
        logging.error("数据准备后为空，无法进行 DID 分析。")
        return False

    # --- 2. 转换为 R Data Frame ---
    logging.info("将数据转换为 R data.frame...")
    try:
        # 在函数内部使用 localconverter 确保转换作用域
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
                 bstrap=True, cband=True,
                 na_rm=True
             )
        logging.info("did::att_gt 函数执行成功。")

    except RRuntimeError as e:
        logging.error(f"执行 R did::att_gt 时出错: {e}")
        return False
    except Exception as e:
        logging.error(f"调用 did::att_gt 时发生未知错误: {e}")
        return False

    # --- 4. 保存 att(g,t) 结果 ---
    att_gt_summary_df = pd.DataFrame() # 初始化为空
    try:
        summary_att_gt = R.summary(att_gt_results)
        with localconverter(ro.default_converter + pandas2ri.converter):
            att_gt_summary_df = ro.conversion.rpy2py(summary_att_gt)
        logging.info("ATT(g,t) 结果提取成功。")
        save_results_df(att_gt_summary_df, f"did_att_gt_results{output_suffix}.csv", output_table_dir)
    except Exception as e:
        logging.error(f"提取或保存 ATT(g,t) 结果时出错: {e}")
        # 不在此处返回 False，继续尝试 aggte

    # --- 5. 执行 R did::aggte (计算聚合效应) ---
    logging.info("调用 R did::aggte 函数计算聚合效应...")
    agg_results_df = pd.DataFrame() # 初始化为空
    analysis_successful = True # 默认成功，除非后续步骤失败
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
                na_rm = True # 移除无法计算的时期
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