import pandas as pd
import numpy as np
import os
from datetime import timedelta

# --- 从 config.py 导入配置 ---
try:
    import config 
except ImportError:
    print("错误：无法导入 config.py。请确保它与此脚本在同一目录，或者在PYTHONPATH中。")
    exit()

# --- 日志基础配置 (有助于跟踪) ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 辅助函数：从 build_features.py 中借鉴和简化 ---

def get_holiday_windows_for_check(holiday_spec, pre_window_days, post_window_days, analysis_start, analysis_end):
    """
    (简化版) 根据 config 中的节假日定义，生成包含前后窗口期的日期区间列表。
    返回: list of tuples [(window_start, window_end, holiday_name_year), ...]
    """
    holiday_windows = []
    # 近似交易日，对于窗口边界的精确性要求不高，主要是覆盖
    # 注意：build_features.py 中使用的是 'B' (Business day frequency)
    # 为了独立性，这里简化处理，但应意识到与原脚本的细微差别
    # 更准确的做法是传入或构建一个交易日历
    trading_days_approx = pd.date_range(start=analysis_start, end=analysis_end, freq='D') 

    for holiday_name, years_spec in holiday_spec.items():
        for year, (start_str, end_str) in years_spec.items():
            holiday_start = pd.to_datetime(start_str)
            holiday_end = pd.to_datetime(end_str)

            # 简化窗口计算：直接在日期上加减天数
            # 这与 build_features.py 中基于交易日的计算方式不同，但用于检查目的应该足够
            window_start_dt = holiday_start - pd.Timedelta(days=pre_window_days)
            window_end_dt = holiday_end + pd.Timedelta(days=post_window_days)
            
            # 确保窗口在分析范围内
            window_start_dt = max(window_start_dt, pd.to_datetime(analysis_start))
            window_end_dt = min(window_end_dt, pd.to_datetime(analysis_end))

            if window_start_dt <= window_end_dt:
                 holiday_windows.append((window_start_dt, window_end_dt, f"{holiday_name}_{year}"))
    holiday_windows.sort()
    return holiday_windows

def main():
    logging.info("开始检查潜在的控制组合约...")

    # --- 1. 加载原始数据 ---
    raw_data_filename = "futures_margin_data.csv" # 与 build_features.py 一致
    raw_data_path = os.path.join(config.PATH_RAW_DATA, raw_data_filename)
    
    logging.info(f"从 {raw_data_path} 加载原始数据...")
    try:
        # 尝试明确指定与 DtypeWarning 相关的列的类型，或使用 low_memory=False
        # 从你的日志看，第3列 (index) 是 yield_rate
        # 我们先用 low_memory=False 避免警告，但这不是最佳实践
        df_raw = pd.read_csv(raw_data_path, low_memory=False)
        logging.info(f"原始数据加载成功: {df_raw.shape}")
    except FileNotFoundError:
        logging.error(f"原始数据文件未找到: {raw_data_path}")
        return
    except Exception as e:
        logging.error(f"加载原始数据时出错: {e}")
        return

    # --- 2. 基本预处理 (与 build_features.py 对齐关键步骤) ---
    df = df_raw.copy()
    df.columns = df.columns.str.lower() # 列名小写

    # 确保关键列存在 (合约ID, 日期, 保证金率)
    # !!! 根据你的原始数据调整 'variety_code_column' !!!
    variety_code_column = 'variety_code' # 假设原始数据中合约品种的列名是 variety_code
    date_column = 'date'
    margin_rate_orig_col = config.MARGIN_RATE_COLUMN # 从 config 获取保证金率列名

    required_cols_check = [variety_code_column, date_column, margin_rate_orig_col]
    missing_cols = [col for col in required_cols_check if col not in df.columns]
    if missing_cols:
        logging.error(f"原始数据缺少检查所需的列: {missing_cols} (variety_code列请根据实际情况修改)")
        return

    df[date_column] = pd.to_datetime(df[date_column])
    
    # 过滤分析时间范围
    df = df[(df[date_column] >= pd.to_datetime(config.ANALYSIS_START_DATE)) & \
            (df[date_column] <= pd.to_datetime(config.ANALYSIS_END_DATE))].copy()
    
    if df.empty:
        logging.info("在指定分析时间范围内没有数据。")
        return
        
    # 处理保证金率 (与 build_features.py 中类似，确保是数值)
    if df[margin_rate_orig_col].dtype == 'object':
        try:
            df[margin_rate_orig_col] = df[margin_rate_orig_col].astype(str).str.replace('%', '', regex=False).astype(float) / 100.0
        except ValueError:
            logging.error(f"无法将保证金率列 '{margin_rate_orig_col}' 转换为数值。请检查数据格式。")
            return
    elif not pd.api.types.is_numeric_dtype(df[margin_rate_orig_col]):
         logging.error(f"保证金率列 '{margin_rate_orig_col}' 不是数值类型。")
         return
    df.sort_values(by=[variety_code_column, date_column], inplace=True)

    # 计算 dlog_margin_rate (对数保证金率差分)
    df['log_margin_rate'] = np.log(df[margin_rate_orig_col].replace(0, 1e-10)) # 避免 log(0)
    df['log_margin_rate_lag1'] = df.groupby(variety_code_column)['log_margin_rate'].shift(1)
    df['dlog_margin_rate'] = df['log_margin_rate'] - df['log_margin_rate_lag1']
    df['dlog_margin_rate'].fillna(0, inplace=True) # 第一期差分为0

    # --- 3. 识别节假日调整窗口 (与 build_features.py 对齐) ---
    logging.info("识别节假日窗口...")
    holiday_windows = get_holiday_windows_for_check(
        config.HOLIDAY_SPECIFIC_DATES, 
        config.DID_HOLIDAY_WINDOW_PRE, 
        config.DID_HOLIDAY_WINDOW_POST,
        config.ANALYSIS_START_DATE,
        config.ANALYSIS_END_DATE
    )
    df['holiday_window_label'] = ''
    for start, end, label in holiday_windows:
        df.loc[(df[date_column] >= start) & (df[date_column] <= end), 'holiday_window_label'] = label
    
    # --- 4. 识别节假日调整日 (与 build_features.py 对齐) ---
    df['is_holiday_adjustment_day'] = (df['holiday_window_label'] != '') & (np.abs(df['dlog_margin_rate']) > 1e-8)

    # --- 5. 检查每个 variety_code ---
    logging.info("检查各合约品种的节假日调整情况...")
    all_variety_codes = df[variety_code_column].unique()
    control_candidate_codes = []

    for code in all_variety_codes:
        contract_data = df[df[variety_code_column] == code]
        if not contract_data['is_holiday_adjustment_day'].any():
            control_candidate_codes.append(code)
            
    if control_candidate_codes:
        logging.info(f"找到 {len(control_candidate_codes)} 个潜在的控制组合约品种 (在分析期间内从未发生节假日保证金调整):")
        for code in control_candidate_codes:
            logging.info(f"  - {code}")
    else:
        logging.info("在分析期间内，所有合约品种都至少发生过一次节假日期间的保证金调整。没有找到潜在的控制组。")

    logging.info("检查完成。")

if __name__ == "__main__":
    main()