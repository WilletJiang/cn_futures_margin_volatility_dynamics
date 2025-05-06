# -*- coding: utf-8 -*-
"""
数据处理与特征构建

读取原始数据，进行清洗、计算所有需要的变量（特征），
保存最终用于分析的面板数据集。
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import timedelta

# 从 src 包导入配置 (假设项目结构允许)
# 如果直接运行脚本，可能需要调整路径或使用相对导入
try:
    from src import config
except ImportError:
    import sys
    # 将项目根目录添加到 sys.path
    PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PACKAGE_DIR)
    from src import config

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 辅助计算函数 ---

def calculate_gk_volatility(high, low, close, open_):
    """计算 Garman-Klass (GK) 波动率的平方项 (未开根号和取对数)"""
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_sq = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)
    # 处理潜在的负值 (理论上不应出现，但实证中可能因数据质量产生)
    gk_sq = np.maximum(gk_sq, 1e-10) # 替换为极小的正数
    return gk_sq

def calculate_parkinson_volatility(high, low):
    """计算 Parkinson 波动率的平方项 (未开根号和取对数)"""
    log_hl = np.log(high / low)
    pk_sq = (1 / (4 * np.log(2))) * (log_hl**2)
    pk_sq = np.maximum(pk_sq, 1e-10)
    return pk_sq

def calculate_turnover_rate(volume, open_interest):
    """计算换手率"""
    # 避免除以零
    turnover = np.where(open_interest > 0, volume / open_interest, 0)
    return turnover

def identify_limit_hit(close, high, low, upper_limit, lower_limit):
    """
    识别是否触及涨跌停板
    定义：收盘价等于涨停价 或 收盘价等于跌停价
    注意：需要数据中包含 upper_limit 和 lower_limit 列
    """
    hit_upper = (close >= upper_limit) & (high >= upper_limit) # 收盘价和最高价都触及涨停
    hit_lower = (close <= lower_limit) & (low <= lower_limit) # 收盘价和最低价都触及跌停
    return np.where(hit_upper | hit_lower, 1, 0)

def get_holiday_windows(holiday_spec, pre_window_days, post_window_days):
    """
    根据 config 中的节假日定义，生成包含前后窗口期的日期区间列表。
    返回: list of tuples [(window_start, window_end, holiday_name, year), ...]
    """
    holiday_windows = []
    trading_days = pd.date_range(start=config.ANALYSIS_START_DATE, end=config.ANALYSIS_END_DATE, freq='B') # 近似交易日

    for holiday_name, years_spec in holiday_spec.items():
        for year, (start_str, end_str) in years_spec.items():
            holiday_start = pd.to_datetime(start_str)
            holiday_end = pd.to_datetime(end_str)

            # 找到假日前 pre_window_days 个交易日
            pre_days_mask = (trading_days < holiday_start)
            window_start_dt = trading_days[pre_days_mask][-pre_window_days:].min() if pre_days_mask.any() else holiday_start

            # 找到假日后 post_window_days 个交易日 (包含假日本身覆盖的交易日)
            # 注意：这里简化处理，直接在假日结束日期上加 post_window_days 个 *日历日*，
            # 更精确的方法需要结合实际交易日历。
            # 或者，更稳妥的方式是找到假日结束后的 post_window_days 个 *交易日*
            post_days_mask = (trading_days > holiday_end)
            window_end_dt = trading_days[post_days_mask][:post_window_days].max() if post_days_mask.any() else holiday_end
            # 修正：窗口结束日期应至少包含假日结束日期
            window_end_dt = max(window_end_dt, holiday_end)


            # 确保窗口在分析范围内
            window_start_dt = max(window_start_dt, pd.to_datetime(config.ANALYSIS_START_DATE))
            window_end_dt = min(window_end_dt, pd.to_datetime(config.ANALYSIS_END_DATE))

            if window_start_dt <= window_end_dt:
                 holiday_windows.append((window_start_dt, window_end_dt, f"{holiday_name}_{year}")) # 使用唯一标识符

    # 按窗口开始日期排序
    holiday_windows.sort()
    return holiday_windows


# --- 主处理函数 ---

def build_features():
    """执行数据加载、清洗、特征计算和保存"""
    logging.info("开始构建特征...")

    # --- 1. 加载数据 ---
    # !!! 需要用户确认文件名和列名 !!!
    logging.info("加载原始数据...")
    try:
        # 使用用户提供的文件名加载单个文件
        raw_data_filename = "futures_margin_data.csv" #! 确认文件扩展名是否为 .csv
        raw_data_path = os.path.join(config.PATH_RAW_DATA, raw_data_filename)
        
        df = pd.read_csv(raw_data_path)
        logging.info(f"原始数据 '{raw_data_filename}' 加载成功: {df.shape}")
        logging.info(f"原始数据列名: {df.columns.tolist()}")

    except FileNotFoundError:
        logging.error(f"原始数据文件未找到: {raw_data_path}. 请检查文件名和路径。")
        return
    except Exception as e:
        logging.error(f"加载原始数据时出错: {e}")
        return

    # --- 2. 基础清洗和预处理 ---
    logging.info("数据清洗和预处理...")

    # 统一列名为小写
    df.columns = df.columns.str.lower()

    # 定义列映射 (从原始列名到脚本使用的标准列名) - 根据用户反馈调整
    # 用户反馈列名: date, variety, yield_rate, long_margin, short_margin, volume_change,
    # price_limit, high_price, low_price, trading_volume, margin_rate, market_state,
    # max_yield, exchange, variety_code, variety_type, close_price, open_price,
    # open_interest, announcement_date
    column_mapping = {
        'date': 'date',
        'variety_code': 'contract_id', # 使用 variety_code 作为合约标识
        'open_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'close_price': 'close',
        'trading_volume': 'volume',
        'open_interest': 'open_interest',
        'margin_rate': config.MARGIN_RATE_COLUMN, # 使用 config 中定义的保证金列名
        'announce_date': 'announcement_date', # 保留公告日期
        # 保留其他可能需要的原始列
        'variety': 'variety',
        'exchange': 'exchange',
        'variety_type': 'variety_type',
        'long_margin': 'long_margin', # 保留多头保证金
        'short_margin': 'short_margin', # 保留空头保证金
        'price_limit': 'price_limit', # 保留涨跌停信息列
        'yield_rate': 'yield_rate',
        'volume_change': 'volume_change',
        'market_state': 'original_market_state', # 保留原始市场状态，避免与计算的冲突
        'max_yield': 'max_yield'
    }
    
    # 检查所需列是否存在
    required_original_cols = list(column_mapping.keys())
    missing_cols = [col for col in required_original_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"原始数据缺少以下必需列: {missing_cols}")
        return
        
    # 选择并重命名列
    df = df[required_original_cols].rename(columns=column_mapping)

    # 转换日期列
    try:
        df['date'] = pd.to_datetime(df['date'])
        # 公告日期可能为空或格式不同，尝试转换，错误则设为 NaT
        if 'announcement_date' in df.columns:
             df['announcement_date'] = pd.to_datetime(df['announcement_date'], errors='coerce')
    except Exception as e:
        logging.error(f"转换日期列时出错: {e}")
        return

    # 合并数据的逻辑不再需要，因为只加载了一个文件
    # logging.info("数据合并步骤已跳过，因为只加载了一个文件。")

    # 过滤时间范围
    df = df[(df['date'] >= config.ANALYSIS_START_DATE) & (df['date'] <= config.ANALYSIS_END_DATE)].copy()
    df.sort_values(by=['contract_id', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 处理价格和成交量中的零或负值 (重要!)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = df[col].replace(0, np.nan) # 假设 0 是无效价格
        df[col] = df[col].apply(lambda x: x if x > 0 else np.nan)
    # 确保 high >= low, high >= open, high >= close, low <= open, low <= close
    df['high'] = df[['high', 'low', 'open', 'close']].max(axis=1)
    df['low'] = df[['high', 'low', 'open', 'close']].min(axis=1)
    # 如果 high == low，波动率计算会出问题，检查这种情况
    if (df['high'] == df['low']).any():
         logging.warning("数据中存在最高价等于最低价的情况，可能影响波动率计算。")
         # 可以考虑加一个极小值，或者在计算波动率时处理
         df.loc[df['high'] == df['low'], 'high'] = df.loc[df['high'] == df['low'], 'high'] * (1 + 1e-6)


    # 处理缺失值 (简单向前填充，可根据需要改进策略)
    # 对于价格，用前一天的收盘价填充开盘价？或者组内向前填充
    # 对于成交量/持仓量，填充 0 或向前填充？
    # 保证金率缺失如何处理？向前填充？
    fill_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest', config.MARGIN_RATE_COLUMN]
    # 检查涨跌停列是否存在
    if 'upper_limit' in df.columns and 'lower_limit' in df.columns:
        fill_cols.extend(['upper_limit', 'lower_limit'])
    else:
        logging.warning("缺少涨跌停价格列 ('upper_limit', 'lower_limit')，无法计算涨跌停状态。")


    df[fill_cols] = df.groupby('contract_id')[fill_cols].ffill()
    # 填充后仍有缺失，可能需要删除该合约的早期数据
    df.dropna(subset=fill_cols, inplace=True) # 删除仍有关键信息缺失的行

    logging.info(f"数据清洗后形状: {df.shape}")
    if df.empty:
        logging.error("数据清洗后为空，请检查原始数据质量和处理步骤。")
        return

    # --- 3. 计算核心变量 ---
    logging.info("计算核心变量...")

    # 3.1 波动率
    df['gk_vol_sq'] = calculate_gk_volatility(df['high'], df['low'], df['close'], df['open'])
    df['log_gk_volatility'] = np.log(np.sqrt(df['gk_vol_sq']))

    # 添加日志量变量计算 (volume和open_interest)
    df['log_volume'] = np.log(df['volume'].replace(0, 1e-10))  # 避免log(0)
    df['log_open_interest'] = np.log(df['open_interest'].replace(0, 1e-10))  # 避免log(0)

    # 替代波动率 (用于稳健性)
    df['parkinson_vol_sq'] = calculate_parkinson_volatility(df['high'], df['low'])
    df['log_parkinson_volatility'] = np.log(np.sqrt(df['parkinson_vol_sq']))

    # 3.2 保证金冲击
    # 确保保证金率是数值类型，并处理可能的百分比表示
    if df[config.MARGIN_RATE_COLUMN].dtype == 'object':
        try:
            # 尝试去除 '%' 并转换为浮点数
            df[config.MARGIN_RATE_COLUMN] = df[config.MARGIN_RATE_COLUMN].astype(str).str.replace('%', '', regex=False).astype(float) / 100.0
        except ValueError:
            logging.error(f"无法将保证金率列 '{config.MARGIN_RATE_COLUMN}' 转换为数值。请检查数据格式。")
            # 可以尝试更复杂的清洗，或在此处停止
            return
    elif not pd.api.types.is_numeric_dtype(df[config.MARGIN_RATE_COLUMN]):
         logging.error(f"保证金率列 '{config.MARGIN_RATE_COLUMN}' 不是数值类型。")
         return

    # 计算对数保证金率的滞后值
    df['log_margin_rate'] = np.log(df[config.MARGIN_RATE_COLUMN].replace(0, 1e-10)) # 避免 log(0)
    df['log_margin_rate_lag1'] = df.groupby('contract_id')['log_margin_rate'].shift(1)
    # 计算对数差分
    df['dlog_margin_rate'] = df['log_margin_rate'] - df['log_margin_rate_lag1']
    # 处理每个合约的第一期观测 (差分为 NaN)
    df['dlog_margin_rate'].fillna(0, inplace=True) # 假设第一期无冲击

    # 区分保证金上升/下降冲击 (可选，LP 分析中可以直接用 dlog_margin_rate)
    df['margin_increase_shock'] = np.where(df['dlog_margin_rate'] > 1e-8, df['dlog_margin_rate'], 0) # 加小容差
    df['margin_decrease_shock'] = np.where(df['dlog_margin_rate'] < -1e-8, df['dlog_margin_rate'], 0)

    # 3.3 流动性代理
    df['turnover_rate'] = calculate_turnover_rate(df['volume'], df['open_interest'])

    # 3.4 涨跌停状态
    if 'upper_limit' in df.columns and 'lower_limit' in df.columns:
        df['limit_hit_dummy'] = identify_limit_hit(df['close'], df['high'], df['low'], df['upper_limit'], df['lower_limit'])
    else:
        df['limit_hit_dummy'] = 0 # 如果没有涨跌停数据，设为0

    # 3.5 日收益率
    df['close_lag1'] = df.groupby('contract_id')['close'].shift(1)
    df['return'] = np.log(df['close'] / df['close_lag1'])
    df['return'].fillna(0, inplace=True) # 第一期收益率为 0

    # --- 4. 计算状态变量 (使用 t-1 信息) ---
    logging.info("计算状态变量...")

    # 4.1 市场状态 (基于过去 N 天移动平均收益率)
    df['avg_return_lookback'] = df.groupby('contract_id')['return'].transform(
        lambda x: x.rolling(window=config.MARKET_REGIME_LOOKBACK, min_periods=int(config.MARKET_REGIME_LOOKBACK*0.8)).mean().shift(1) # 使用 t-1 信息
    )
    df['market_regime'] = 'Neutral' # 默认为震荡市
    df.loc[df['avg_return_lookback'] > config.MARKET_REGIME_UPPER_THRESHOLD, 'market_regime'] = 'Bull'
    df.loc[df['avg_return_lookback'] < config.MARKET_REGIME_LOWER_THRESHOLD, 'market_regime'] = 'Bear'
    # 转换为分类哑变量
    df = pd.get_dummies(df, columns=['market_regime'], prefix='market_regime', dummy_na=False)
    # 将 True/False 转为 1/0
    for col in df.columns:
        if col.startswith('market_regime_'):
            df[col] = df[col].astype(int)


    # 4.2 波动率状态 (基于过去 N 天滚动中位数波动率)
    df['median_log_gk_vol_lookback'] = df.groupby('contract_id')['log_gk_volatility'].transform(
        lambda x: x.rolling(window=config.VOLATILITY_REGIME_LOOKBACK, min_periods=int(config.VOLATILITY_REGIME_LOOKBACK*0.8)).median().shift(1) # 使用 t-1 信息
    )
    df['volatility_regime'] = 'Low' # 默认为低波
    df.loc[df['log_gk_volatility'].shift(1) > df['median_log_gk_vol_lookback'], 'volatility_regime'] = 'High'
     # 转换为分类哑变量
    df = pd.get_dummies(df, columns=['volatility_regime'], prefix='volatility_regime', dummy_na=False)
    for col in df.columns:
        if col.startswith('volatility_regime_'):
            df[col] = df[col].astype(int)

    # --- 5. 创建 DID 相关变量 ---
    logging.info("创建 DID 相关变量...")

    # 5.1 识别节假日调整窗口
    holiday_windows = get_holiday_windows(config.HOLIDAY_SPECIFIC_DATES, config.DID_HOLIDAY_WINDOW_PRE, config.DID_HOLIDAY_WINDOW_POST)
    df['holiday_window_label'] = '' # 标记所属的节假日窗口名称

    # 标记每个日期是否在某个节假日窗口内，以及是哪个窗口
    # 注意：这里假设窗口不重叠，如果重叠需要更复杂的逻辑
    for start, end, label in holiday_windows:
        df.loc[(df['date'] >= start) & (df['date'] <= end), 'holiday_window_label'] = label

    # 5.2 识别节假日期间的保证金调整事件 (以实施日为准)
    # 调整定义：在 holiday_window_label 非空的日期，发生了保证金变动 (dlog_margin_rate != 0)
    df['is_holiday_adjustment_day'] = (df['holiday_window_label'] != '') & (np.abs(df['dlog_margin_rate']) > 1e-8)

    # 5.3 确定每个合约首次被处理的组 (g) 和处理时间 (首次调整日)
    # g 定义为首次发生节假日调整的 holiday_window_label
    df['treat_date_g'] = df[df['is_holiday_adjustment_day']].groupby('contract_id')['date'].transform('min')
    df['treat_group_g_label'] = df.loc[df['date'] == df['treat_date_g'], 'holiday_window_label']
    df['treat_group_g_label'] = df.groupby('contract_id')['treat_group_g_label'].transform('first') # 广播到该合约所有日期
    df['treat_date_g'] = df.groupby('contract_id')['treat_date_g'].transform('first') # 广播

    # 创建 C&S DID 需要的变量:
    # - first_treatment_period: 合约首次接受处理的“时期”（这里用 treat_date_g）
    # - time_period: 当前观测的时期（这里用 date）
    # - outcome: 结果变量 (log_gk_volatility)
    # - group: 分组变量 (treat_group_g_label) - C&S 的 att_gt 会自动处理

    # 5.4 创建相对时间变量 (t) (可选，主要用于可视化或传统 DID)
    # t = 0 是处理日当天
    df['relative_time_t'] = (df['date'] - df['treat_date_g']).dt.days
    # 对于从未被处理的合约，relative_time_t 为 NaT/NaN

    # 5.5 创建 HolidayAdjust 哑变量 (用于 LP 状态依赖)
    # =1 如果当前日期 t 发生了保证金调整，并且该日期处于某个节假日窗口内
    df['State_HolidayAdjust'] = df['is_holiday_adjustment_day'].astype(int)
    df['State_NonHolidayAdjust'] = ((df['holiday_window_label'] == '') & (np.abs(df['dlog_margin_rate']) > 1e-8)).astype(int)


    # --- 6. 创建滞后控制变量 ---
    logging.info("创建滞后控制变量...")
    lag_vars = config.CONTROL_VARIABLES + [config.LIQUIDITY_PROXY_VAR, config.ALT_VOLATILITY_VAR]
    # 去除可能重复的 _lag1 后缀，并确保基础变量存在
    base_vars_needed = set(v.replace('_lag1', '') for v in lag_vars)

    missing_base_vars = base_vars_needed - set(df.columns)
    if missing_base_vars:
        logging.warning(f"缺少以下基础变量，无法创建其滞后项: {missing_base_vars}")
        # 从 lag_vars 中移除无法创建的滞后项
        lag_vars = [v for v in lag_vars if v.replace('_lag1', '') not in missing_base_vars]

    for var in lag_vars:
        base_var = var.replace('_lag1', '')
        if base_var in df.columns:
            df[var] = df.groupby('contract_id')[base_var].shift(1)
        else:
             logging.warning(f"基础变量 {base_var} 不存在于 DataFrame 中，无法创建滞后项 {var}")


    # --- 7. 清理和保存 ---
    logging.info("最后清理和保存数据...")

    # 选择最终需要的列 (根据后续分析调整)
    # DID 需要: contract_id, date, log_gk_volatility, treat_date_g (或等效的首次处理期标识), 控制变量
    # LP 需要: contract_id, date, log_gk_volatility (或其他结果变量), dlog_margin_rate, 状态变量 (t-1), 控制变量 (t-1)
    final_columns = [
        'contract_id', 'date',
        'log_gk_volatility', # 主要结果变量
        'dlog_margin_rate', 'margin_increase_shock', 'margin_decrease_shock', # 冲击变量
        # DID 相关
        'treat_date_g', 'treat_group_g_label', 'relative_time_t', 'is_holiday_adjustment_day',
        # 状态变量 (t-1 时刻) - 注意 get_dummies 后缀
        'market_regime_Bull', 'market_regime_Bear', 'market_regime_Neutral', # 示例，具体看 get_dummies 结果
        'volatility_regime_High', 'volatility_regime_Low', # 示例
        'State_HolidayAdjust', 'State_NonHolidayAdjust', # 调整类型状态
        # 控制变量 (t-1 时刻)
    ] + config.CONTROL_VARIABLES + [
        # 其他可能需要的变量
        'log_volume', 'log_open_interest', 'return', 'limit_hit_dummy', # 当期值
        config.LIQUIDITY_PROXY_VAR.replace('_lag1',''), # 当期流动性
        config.ALT_VOLATILITY_VAR.replace('_lag1',''), # 当期替代波动率
        config.LIQUIDITY_PROXY_VAR, # 滞后流动性
        config.ALT_VOLATILITY_VAR, # 滞后替代波动率
    ]
    # 确保 market_regime 和 volatility_regime 的哑变量列名正确加入
    final_columns.extend([col for col in df.columns if col.startswith('market_regime_') and col not in final_columns])
    final_columns.extend([col for col in df.columns if col.startswith('volatility_regime_') and col not in final_columns])

    # 去重并保留存在的列
    final_columns = sorted(list(set(col for col in final_columns if col in df.columns)))


    df_final = df[final_columns].copy()

    # 处理计算滞后项和滚动窗口导致的初始 NaN
    # 计算每个合约需要多少非缺失观测才能进行所有计算
    # 例如，最长的回看期 + 滞后期
    max_lookback = max(config.MARKET_REGIME_LOOKBACK, config.VOLATILITY_REGIME_LOOKBACK)
    min_obs_required = max_lookback + config.LP_CONTROL_LAGS # 加1是因为shift(1)

    # 删除每个组开头不满足最小观测期的行
    df_final = df_final.groupby('contract_id').filter(lambda x: len(x) >= min_obs_required)
    # 或者，更精确地删除开头因计算产生的 NaN 行
    
    # 准备实际存在的控制变量列表
    existing_control_vars = [var for var in config.CONTROL_VARIABLES if var in df_final.columns]
    existing_regime_vars = [col for col in df_final.columns if col.startswith('market_regime_') or col.startswith('volatility_regime_')]
    existing_extra_vars = [var for var in [config.LIQUIDITY_PROXY_VAR, config.ALT_VOLATILITY_VAR] if var in df_final.columns]

    # 使用实际存在的变量进行dropna
    if existing_control_vars or existing_regime_vars or existing_extra_vars:
        df_final.dropna(subset=existing_control_vars + existing_regime_vars + existing_extra_vars, inplace=True)


    logging.info(f"最终面板数据形状: {df_final.shape}")
    if df_final.empty:
        logging.error("最终面板数据为空，请检查数据处理流程和原始数据。")
        return

    # 保存到 Parquet 文件
    output_path = config.PANEL_DATA_FILEPATH
    try:
        df_final.to_parquet(output_path, index=False)
        logging.info(f"特征构建完成，最终面板数据已保存到: {output_path}")
    except Exception as e:
        logging.error(f"保存处理后数据时出错: {e}")

# --- 主执行块 ---
if __name__ == "__main__":
    # 可以在这里添加命令行参数处理，例如指定配置文件路径
    build_features()