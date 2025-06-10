# -*- coding: utf-8 -*-
"""
事件研究方法 (Event Study Methodology)

实现事件研究框架来补充LP-IRF分析，研究保证金调整事件对期货市场波动率和价格动态的影响。
包含事件识别、正常收益率估计、异常收益率计算和统计检验。
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 项目配置导入
try:
    from src import config
except ImportError:
    import sys
    PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PACKAGE_DIR)
    from src import config

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

class EventStudyAnalyzer:
    """
    事件研究分析器
    
    实现完整的事件研究方法，包括：
    1. 事件识别和筛选
    2. 估计窗口和事件窗口定义
    3. 正常收益率模型估计
    4. 异常收益率计算
    5. 累积异常收益率(CAR)计算
    6. 统计显著性检验
    """
    
    def __init__(self, data, outcome_var='log_gk_volatility', 
                 event_var='dlog_margin_rate',
                 entity_col='contract_id', time_col='date'):
        """
        初始化事件研究分析器
        
        Args:
            data (pd.DataFrame): 面板数据
            outcome_var (str): 结果变量名称 (如波动率)
            event_var (str): 事件变量名称 (如保证金率变化)
            entity_col (str): 实体标识列名
            time_col (str): 时间列名
        """
        self.data = data.copy()
        self.outcome_var = outcome_var
        self.event_var = event_var
        self.entity_col = entity_col
        self.time_col = time_col
        
        # 确保日期列为datetime类型
        self.data[self.time_col] = pd.to_datetime(self.data[self.time_col])
        
        # 事件研究参数
        self.pre_window = config.EVENT_STUDY_PRE_WINDOW
        self.post_window = config.EVENT_STUDY_POST_WINDOW
        self.estimation_window = config.EVENT_STUDY_ESTIMATION_WINDOW
        self.gap_days = config.EVENT_STUDY_GAP_DAYS
        self.min_estimation_obs = config.EVENT_STUDY_MIN_ESTIMATION_OBS
        self.min_event_gap = config.EVENT_STUDY_MIN_EVENT_GAP
        
        logging.info(f"事件研究分析器初始化完成 - 数据形状: {self.data.shape}")
        logging.info(f"事件窗口: [{-self.pre_window}, {self.post_window}], 估计窗口: {self.estimation_window}天")
    
    def identify_events(self, threshold=1e-6, event_type='all'):
        """
        识别保证金调整事件
        
        Args:
            threshold (float): 事件识别阈值 (绝对值)
            event_type (str): 事件类型 ('increase', 'decrease', 'all')
        
        Returns:
            pd.DataFrame: 包含事件信息的DataFrame
        """
        logging.info(f"开始识别事件 - 阈值: {threshold}, 类型: {event_type}")
        
        # 识别事件
        if event_type == 'increase':
            event_mask = self.data[self.event_var] > threshold
        elif event_type == 'decrease':
            event_mask = self.data[self.event_var] < -threshold
        else:  # 'all'
            event_mask = np.abs(self.data[self.event_var]) > threshold
        
        events = self.data[event_mask].copy()
        
        if events.empty:
            logging.warning("未发现任何事件")
            return pd.DataFrame()
        
        # 按实体和时间排序
        events = events.sort_values([self.entity_col, self.time_col])
        
        # 过滤重叠事件 (同一实体内事件间隔小于最小间隔的)
        filtered_events = []
        
        for entity in events[self.entity_col].unique():
            entity_events = events[events[self.entity_col] == entity].copy()
            
            if len(entity_events) == 1:
                filtered_events.append(entity_events)
                continue
            
            # 过滤重叠事件
            selected_events = [entity_events.iloc[0]]  # 保留第一个事件
            
            for i in range(1, len(entity_events)):
                current_event = entity_events.iloc[i]
                last_selected = selected_events[-1]
                
                days_diff = (current_event[self.time_col] - last_selected[self.time_col]).days
                
                if days_diff >= self.min_event_gap:
                    selected_events.append(current_event)
                else:
                    # 如果事件重叠，选择影响更大的事件
                    if abs(current_event[self.event_var]) > abs(last_selected[self.event_var]):
                        selected_events[-1] = current_event
            
            if selected_events:
                filtered_events.append(pd.DataFrame(selected_events))
        
        if filtered_events:
            final_events = pd.concat(filtered_events, ignore_index=True)
        else:
            final_events = pd.DataFrame()
        
        logging.info(f"事件识别完成 - 原始事件: {len(events)}, 过滤后事件: {len(final_events)}")
        
        return final_events
    
    def calculate_normal_returns(self, events_df):
        """
        计算正常收益率 (基于估计窗口)
        
        Args:
            events_df (pd.DataFrame): 事件数据
        
        Returns:
            dict: 每个事件的正常收益率模型参数
        """
        logging.info("开始计算正常收益率模型")
        
        normal_return_models = {}
        
        for _, event in events_df.iterrows():
            entity = event[self.entity_col]
            event_date = event[self.time_col]
            
            # 定义估计窗口 (事件日前gap_days到gap_days+estimation_window天)
            estimation_end = event_date - timedelta(days=self.gap_days)
            estimation_start = estimation_end - timedelta(days=self.estimation_window)
            
            # 获取估计窗口数据
            entity_data = self.data[self.data[self.entity_col] == entity].copy()
            estimation_data = entity_data[
                (entity_data[self.time_col] >= estimation_start) & 
                (entity_data[self.time_col] <= estimation_end)
            ].copy()
            
            if len(estimation_data) < self.min_estimation_obs:
                logging.warning(f"实体 {entity} 事件 {event_date} 估计窗口观测不足: {len(estimation_data)}")
                continue
            
            # 计算正常收益率模型
            if config.EVENT_STUDY_NORMAL_RETURN_MODEL == "mean_adjusted":
                # 均值调整模型
                normal_return = estimation_data[self.outcome_var].mean()
                model_params = {'type': 'mean_adjusted', 'mean': normal_return}
                
            elif config.EVENT_STUDY_NORMAL_RETURN_MODEL == "market_adjusted":
                # 市场调整模型 (这里使用所有合约的平均作为市场)
                market_returns = self.data.groupby(self.time_col)[self.outcome_var].mean()
                estimation_market = market_returns[
                    (market_returns.index >= estimation_start) & 
                    (market_returns.index <= estimation_end)
                ]
                
                if len(estimation_market) < self.min_estimation_obs:
                    continue
                
                model_params = {'type': 'market_adjusted', 'market_data': estimation_market}
                
            else:  # market_model (默认)
                # 市场模型 (CAPM)
                market_returns = self.data.groupby(self.time_col)[self.outcome_var].mean()
                estimation_market = market_returns[
                    (market_returns.index >= estimation_start) & 
                    (market_returns.index <= estimation_end)
                ]
                
                if len(estimation_market) < self.min_estimation_obs:
                    continue
                
                # 对齐数据
                aligned_data = pd.merge(
                    estimation_data[[self.time_col, self.outcome_var]],
                    estimation_market.reset_index().rename(columns={self.time_col: self.time_col, self.outcome_var: 'market_return'}),
                    on=self.time_col,
                    how='inner'
                )
                
                if len(aligned_data) < self.min_estimation_obs:
                    continue
                
                # 估计市场模型: R_i = alpha + beta * R_m + epsilon
                # 使用简单的最小二乘法实现
                X = aligned_data['market_return'].values
                y = aligned_data[self.outcome_var].values

                # 计算回归系数: beta = Cov(X,Y) / Var(X), alpha = mean(Y) - beta * mean(X)
                beta = np.cov(X, y)[0, 1] / np.var(X, ddof=1)
                alpha = np.mean(y) - beta * np.mean(X)

                # 计算残差标准差
                y_pred = alpha + beta * X
                residuals = y - y_pred
                residual_std = np.std(residuals, ddof=2)
                
                model_params = {
                    'type': 'market_model',
                    'alpha': alpha,
                    'beta': beta,
                    'residual_std': residual_std,
                    'market_data': estimation_market
                }
            
            normal_return_models[f"{entity}_{event_date.strftime('%Y%m%d')}"] = model_params
        
        logging.info(f"正常收益率模型计算完成 - 成功建模事件数: {len(normal_return_models)}")

        return normal_return_models

    def calculate_abnormal_returns(self, events_df, normal_return_models):
        """
        计算异常收益率 (Abnormal Returns, AR)

        Args:
            events_df (pd.DataFrame): 事件数据
            normal_return_models (dict): 正常收益率模型参数

        Returns:
            pd.DataFrame: 包含异常收益率的数据
        """
        logging.info("开始计算异常收益率")

        abnormal_returns_list = []

        for _, event in events_df.iterrows():
            entity = event[self.entity_col]
            event_date = event[self.time_col]
            event_key = f"{entity}_{event_date.strftime('%Y%m%d')}"

            if event_key not in normal_return_models:
                logging.warning(f"事件 {event_key} 缺少正常收益率模型，跳过")
                continue

            model_params = normal_return_models[event_key]

            # 定义事件窗口
            window_start = event_date - timedelta(days=self.pre_window)
            window_end = event_date + timedelta(days=self.post_window)

            # 获取事件窗口数据
            entity_data = self.data[self.data[self.entity_col] == entity].copy()
            window_data = entity_data[
                (entity_data[self.time_col] >= window_start) &
                (entity_data[self.time_col] <= window_end)
            ].copy()

            if window_data.empty:
                logging.warning(f"事件 {event_key} 事件窗口无数据，跳过")
                continue

            # 计算相对时间 (相对于事件日)
            window_data['relative_time'] = (window_data[self.time_col] - event_date).dt.days

            # 计算预期正常收益率
            if model_params['type'] == 'mean_adjusted':
                window_data['expected_return'] = model_params['mean']

            elif model_params['type'] == 'market_adjusted':
                # 获取对应日期的市场收益率
                market_data = model_params['market_data']
                window_data['expected_return'] = window_data[self.time_col].map(
                    lambda x: market_data.get(x, np.nan)
                )

            else:  # market_model
                # 获取对应日期的市场收益率
                market_data = self.data.groupby(self.time_col)[self.outcome_var].mean()
                window_data['market_return'] = window_data[self.time_col].map(market_data)

                # 计算预期收益率: E(R_i) = alpha + beta * R_m
                window_data['expected_return'] = (
                    model_params['alpha'] +
                    model_params['beta'] * window_data['market_return']
                )

            # 计算异常收益率: AR = 实际收益率 - 预期收益率
            window_data['abnormal_return'] = (
                window_data[self.outcome_var] - window_data['expected_return']
            )

            # 添加事件信息
            window_data['event_id'] = event_key
            window_data['event_date'] = event_date
            window_data['event_magnitude'] = event[self.event_var]
            window_data['event_type'] = 'increase' if event[self.event_var] > 0 else 'decrease'

            # 添加标准化异常收益率 (如果有残差标准差)
            if model_params['type'] == 'market_model' and 'residual_std' in model_params:
                window_data['standardized_abnormal_return'] = (
                    window_data['abnormal_return'] / model_params['residual_std']
                )

            abnormal_returns_list.append(window_data)

        if abnormal_returns_list:
            abnormal_returns_df = pd.concat(abnormal_returns_list, ignore_index=True)
        else:
            abnormal_returns_df = pd.DataFrame()

        logging.info(f"异常收益率计算完成 - 事件数: {len(abnormal_returns_list)}")

        return abnormal_returns_df

    def calculate_cumulative_abnormal_returns(self, abnormal_returns_df):
        """
        计算累积异常收益率 (Cumulative Abnormal Returns, CAR)

        Args:
            abnormal_returns_df (pd.DataFrame): 异常收益率数据

        Returns:
            pd.DataFrame: 包含CAR的数据
        """
        logging.info("开始计算累积异常收益率")

        if abnormal_returns_df.empty:
            return pd.DataFrame()

        car_results = []

        for event_id in abnormal_returns_df['event_id'].unique():
            event_data = abnormal_returns_df[
                abnormal_returns_df['event_id'] == event_id
            ].copy().sort_values('relative_time')

            # 计算累积异常收益率
            event_data['cumulative_abnormal_return'] = event_data['abnormal_return'].cumsum()

            # 计算标准化累积异常收益率 (如果有标准化异常收益率)
            if 'standardized_abnormal_return' in event_data.columns:
                event_data['cumulative_standardized_abnormal_return'] = (
                    event_data['standardized_abnormal_return'].cumsum()
                )

            car_results.append(event_data)

        if car_results:
            car_df = pd.concat(car_results, ignore_index=True)
        else:
            car_df = pd.DataFrame()

        logging.info(f"累积异常收益率计算完成")

        return car_df

    def calculate_average_abnormal_returns(self, car_df):
        """
        计算平均异常收益率 (Average Abnormal Returns, AAR) 和
        累积平均异常收益率 (Cumulative Average Abnormal Returns, CAAR)

        Args:
            car_df (pd.DataFrame): 包含CAR的数据

        Returns:
            pd.DataFrame: AAR和CAAR结果
        """
        logging.info("开始计算平均异常收益率和累积平均异常收益率")

        if car_df.empty:
            return pd.DataFrame()

        # 按相对时间计算平均异常收益率
        aar_results = car_df.groupby('relative_time').agg({
            'abnormal_return': ['mean', 'std', 'count'],
            'cumulative_abnormal_return': 'mean'
        }).round(6)

        # 展平列名
        aar_results.columns = ['_'.join(col).strip() for col in aar_results.columns.values]
        aar_results = aar_results.reset_index()

        # 重命名列
        aar_results.rename(columns={
            'abnormal_return_mean': 'AAR',
            'abnormal_return_std': 'AAR_std',
            'abnormal_return_count': 'N_events',
            'cumulative_abnormal_return_mean': 'CAAR'
        }, inplace=True)

        # 计算标准误
        aar_results['AAR_se'] = aar_results['AAR_std'] / np.sqrt(aar_results['N_events'])

        # 计算t统计量和p值
        aar_results['t_stat'] = aar_results['AAR'] / aar_results['AAR_se']
        aar_results['p_value'] = 2 * (1 - stats.t.cdf(np.abs(aar_results['t_stat']),
                                                       aar_results['N_events'] - 1))

        # 计算置信区间
        alpha = config.EVENT_STUDY_SIGNIFICANCE_LEVEL
        t_critical = stats.t.ppf(1 - alpha/2, aar_results['N_events'] - 1)
        aar_results['CI_lower'] = aar_results['AAR'] - t_critical * aar_results['AAR_se']
        aar_results['CI_upper'] = aar_results['AAR'] + t_critical * aar_results['AAR_se']

        # 标记显著性
        aar_results['significant'] = aar_results['p_value'] < alpha

        logging.info(f"平均异常收益率计算完成 - 时间点数: {len(aar_results)}")

        return aar_results

    def run_event_study(self, threshold=1e-6, event_types=['increase', 'decrease']):
        """
        执行完整的事件研究分析

        Args:
            threshold (float): 事件识别阈值
            event_types (list): 要分析的事件类型列表

        Returns:
            dict: 包含所有分析结果的字典
        """
        logging.info("开始执行事件研究分析")

        results = {}

        for event_type in event_types:
            logging.info(f"--- 分析事件类型: {event_type} ---")

            # 1. 识别事件
            events_df = self.identify_events(threshold=threshold, event_type=event_type)

            if events_df.empty:
                logging.warning(f"事件类型 {event_type} 未发现事件，跳过")
                continue

            # 2. 计算正常收益率模型
            normal_return_models = self.calculate_normal_returns(events_df)

            if not normal_return_models:
                logging.warning(f"事件类型 {event_type} 无法建立正常收益率模型，跳过")
                continue

            # 3. 计算异常收益率
            abnormal_returns_df = self.calculate_abnormal_returns(events_df, normal_return_models)

            if abnormal_returns_df.empty:
                logging.warning(f"事件类型 {event_type} 无法计算异常收益率，跳过")
                continue

            # 4. 计算累积异常收益率
            car_df = self.calculate_cumulative_abnormal_returns(abnormal_returns_df)

            # 5. 计算平均异常收益率
            aar_df = self.calculate_average_abnormal_returns(car_df)

            # 保存结果
            results[event_type] = {
                'events': events_df,
                'normal_return_models': normal_return_models,
                'abnormal_returns': abnormal_returns_df,
                'car_data': car_df,
                'aar_results': aar_df
            }

            logging.info(f"事件类型 {event_type} 分析完成 - 事件数: {len(events_df)}")

        logging.info("事件研究分析完成")

        return results


def run_event_study_analysis_core(data, outcome_var='log_gk_volatility',
                                  output_table_dir=None, output_suffix="",
                                  event_var='dlog_margin_rate',
                                  threshold=1e-6):
    """
    执行事件研究分析的核心函数 (可被其他脚本调用)

    Args:
        data (pd.DataFrame): 输入面板数据
        outcome_var (str): 结果变量名称
        output_table_dir (str): 保存结果表格的目录
        output_suffix (str): 添加到输出文件名末尾的后缀
        event_var (str): 事件变量名称
        threshold (float): 事件识别阈值

    Returns:
        dict: 分析结果字典
    """
    logging.info(f"--- 开始核心事件研究分析 (Outcome: {outcome_var}, Suffix: '{output_suffix}') ---")

    # 使用默认输出目录
    if output_table_dir is None:
        output_table_dir = config.PATH_OUTPUT_TABLES

    # 确保输出目录存在
    os.makedirs(output_table_dir, exist_ok=True)

    # 创建事件研究分析器
    analyzer = EventStudyAnalyzer(
        data=data,
        outcome_var=outcome_var,
        event_var=event_var
    )

    # 执行分析
    results = analyzer.run_event_study(threshold=threshold)

    # 保存结果
    for event_type, event_results in results.items():
        # 保存AAR结果
        if 'aar_results' in event_results and not event_results['aar_results'].empty:
            aar_filename = f"event_study_aar_{event_type}{output_suffix}.csv"
            aar_path = os.path.join(output_table_dir, aar_filename)
            event_results['aar_results'].to_csv(aar_path, index=False)
            logging.info(f"AAR结果已保存: {aar_path}")

        # 保存事件列表
        if 'events' in event_results and not event_results['events'].empty:
            events_filename = f"event_study_events_{event_type}{output_suffix}.csv"
            events_path = os.path.join(output_table_dir, events_filename)
            event_results['events'].to_csv(events_path, index=False)
            logging.info(f"事件列表已保存: {events_path}")

        # 保存详细的异常收益率数据
        if 'car_data' in event_results and not event_results['car_data'].empty:
            car_filename = f"event_study_car_{event_type}{output_suffix}.csv"
            car_path = os.path.join(output_table_dir, car_filename)
            event_results['car_data'].to_csv(car_path, index=False)
            logging.info(f"CAR数据已保存: {car_path}")

    logging.info(f"--- 完成核心事件研究分析 (Outcome: {outcome_var}, Suffix: '{output_suffix}') ---")

    return results


# 主执行块 (用于直接运行此脚本)
if __name__ == "__main__":
    logging.info("直接运行 event_study_analysis.py 脚本...")

    # 加载主数据
    try:
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
            logging.error(f"主面板数据文件未找到: {config.PANEL_DATA_FILEPATH}")
            logging.error("请先运行 src/data_processing/build_features.py")
            exit()

        main_df = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        main_df['date'] = pd.to_datetime(main_df['date'])
        logging.info(f"主数据加载成功: {main_df.shape}")

        # 执行默认分析
        results = run_event_study_analysis_core(
            data=main_df,
            outcome_var='log_gk_volatility',
            output_table_dir=config.PATH_OUTPUT_TABLES,
            output_suffix=""  # 无后缀表示主分析
        )

        if results:
            logging.info("默认事件研究分析成功完成。")
        else:
            logging.error("默认事件研究分析执行过程中出现错误。")

    except Exception as e:
        logging.error(f"直接运行事件研究分析时出错: {e}")
        import traceback
        traceback.print_exc()
