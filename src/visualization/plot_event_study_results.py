# -*- coding: utf-8 -*-
"""
事件研究结果可视化

生成事件研究分析的图表，包括：
1. 平均异常收益率(AAR)时间序列图
2. 累积平均异常收益率(CAAR)图
3. 事件分布图
4. 与LP-IRF结果的比较图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime

# 项目配置导入
try:
    from src import config
except ImportError:
    import sys
    PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PACKAGE_DIR)
    from src import config

# 设置英文字体和绘图风格
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

def plot_aar_time_series(aar_results, event_type, outcome_var, output_dir, suffix=""):
    """
    绘制平均异常收益率(AAR)时间序列图
    
    Args:
        aar_results (pd.DataFrame): AAR分析结果
        event_type (str): 事件类型
        outcome_var (str): 结果变量名称
        output_dir (str): 输出目录
        suffix (str): 文件名后缀
    """
    if aar_results.empty:
        logging.warning(f"AAR结果为空，跳过绘图: {event_type}")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上图：AAR时间序列
    x = aar_results['relative_time']
    y = aar_results['AAR']
    ci_lower = aar_results['CI_lower']
    ci_upper = aar_results['CI_upper']
    
    # 绘制AAR线
    ax1.plot(x, y, 'b-', linewidth=2, label='Average Abnormal Returns (AAR)')

    # 绘制置信区间
    ax1.fill_between(x, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% Confidence Interval')

    # 标记显著点
    significant_points = aar_results[aar_results['significant']]
    if not significant_points.empty:
        ax1.scatter(significant_points['relative_time'], significant_points['AAR'],
                   color='red', s=50, zorder=5, label='Significant Points (p<0.05)')

    # 添加零线和事件日线
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Event Day')

    ax1.set_xlabel('Relative Time (Days)')
    ax1.set_ylabel('Average Abnormal Returns')
    ax1.set_title(f'Average Abnormal Returns Time Series - {event_type.upper()} Events\n({outcome_var})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：CAAR时间序列
    if 'CAAR' in aar_results.columns:
        ax2.plot(x, aar_results['CAAR'], 'g-', linewidth=2, label='Cumulative Average Abnormal Returns (CAAR)')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Event Day')

        ax2.set_xlabel('Relative Time (Days)')
        ax2.set_ylabel('Cumulative Average Abnormal Returns')
        ax2.set_title(f'Cumulative Average Abnormal Returns Time Series - {event_type.upper()} Events')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"event_study_aar_{event_type}{suffix}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"AAR时间序列图已保存: {filepath}")

def plot_event_distribution(events_df, event_type, output_dir, suffix=""):
    """
    绘制事件分布图
    
    Args:
        events_df (pd.DataFrame): 事件数据
        event_type (str): 事件类型
        output_dir (str): 输出目录
        suffix (str): 文件名后缀
    """
    if events_df.empty:
        logging.warning(f"事件数据为空，跳过绘图: {event_type}")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 事件时间分布
    events_df['year'] = pd.to_datetime(events_df['date']).dt.year
    year_counts = events_df['year'].value_counts().sort_index()

    ax1.bar(year_counts.index, year_counts.values, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Events')
    ax1.set_title(f'{event_type.upper()} Events Distribution by Year')
    ax1.grid(True, alpha=0.3)

    # 2. 事件幅度分布
    event_var = 'dlog_margin_rate' if 'dlog_margin_rate' in events_df.columns else 'event_magnitude'
    if event_var in events_df.columns:
        ax2.hist(events_df[event_var], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Margin Rate Change Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{event_type.upper()} Events Magnitude Distribution')
        ax2.grid(True, alpha=0.3)

    # 3. 按合约类型分布 (如果有)
    if 'variety' in events_df.columns:
        variety_counts = events_df['variety'].value_counts().head(10)
        ax3.barh(range(len(variety_counts)), variety_counts.values, alpha=0.7, color='orange')
        ax3.set_yticks(range(len(variety_counts)))
        ax3.set_yticklabels(variety_counts.index)
        ax3.set_xlabel('Number of Events')
        ax3.set_title(f'{event_type.upper()} Events Distribution by Variety (Top 10)')
        ax3.grid(True, alpha=0.3)

    # 4. 月度分布
    events_df['month'] = pd.to_datetime(events_df['date']).dt.month
    month_counts = events_df['month'].value_counts().sort_index()

    ax4.bar(month_counts.index, month_counts.values, alpha=0.7, color='pink')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Number of Events')
    ax4.set_title(f'{event_type.upper()} Events Distribution by Month')
    ax4.set_xticks(range(1, 13))
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"event_study_distribution_{event_type}{suffix}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"事件分布图已保存: {filepath}")

def plot_car_individual_events(car_data, event_type, output_dir, suffix="", max_events=10):
    """
    绘制个别事件的累积异常收益率图
    
    Args:
        car_data (pd.DataFrame): CAR数据
        event_type (str): 事件类型
        output_dir (str): 输出目录
        suffix (str): 文件名后缀
        max_events (int): 最多显示的事件数量
    """
    if car_data.empty:
        logging.warning(f"CAR数据为空，跳过绘图: {event_type}")
        return
    
    # 选择前几个事件进行展示
    unique_events = car_data['event_id'].unique()[:max_events]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))
    
    for i, event_id in enumerate(unique_events):
        event_data = car_data[car_data['event_id'] == event_id].sort_values('relative_time')
        
        ax.plot(event_data['relative_time'], event_data['cumulative_abnormal_return'], 
               color=colors[i], alpha=0.7, linewidth=1.5, label=f'事件 {i+1}')
    
    # 添加零线和事件日线
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='事件日')
    
    ax.set_xlabel('相对时间 (天)')
    ax.set_ylabel('累积异常收益率')
    ax.set_title(f'个别事件累积异常收益率 - {event_type.upper()}事件 (前{len(unique_events)}个)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"event_study_individual_car_{event_type}{suffix}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"个别事件CAR图已保存: {filepath}")

def plot_event_study_summary(results_dict, outcome_var, output_dir, suffix=""):
    """
    绘制事件研究结果汇总图
    
    Args:
        results_dict (dict): 事件研究结果字典
        outcome_var (str): 结果变量名称
        output_dir (str): 输出目录
        suffix (str): 文件名后缀
    """
    if not results_dict:
        logging.warning("事件研究结果为空，跳过汇总绘图")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    
    for event_type, results in results_dict.items():
        if 'aar_results' not in results or results['aar_results'].empty:
            continue
        
        aar_data = results['aar_results']
        
        # AAR图
        if plot_idx < 4:
            ax = axes[plot_idx]
            x = aar_data['relative_time']
            y = aar_data['AAR']
            
            ax.plot(x, y, linewidth=2, label=f'{event_type.upper()} AAR')
            ax.fill_between(x, aar_data['CI_lower'], aar_data['CI_upper'], alpha=0.3)
            
            # 标记显著点
            significant_points = aar_data[aar_data['significant']]
            if not significant_points.empty:
                ax.scatter(significant_points['relative_time'], significant_points['AAR'], 
                          color='red', s=30, zorder=5)
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('相对时间 (天)')
            ax.set_ylabel('平均异常收益率')
            ax.set_title(f'{event_type.upper()}事件 AAR')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plot_idx += 1
        
        # CAAR图
        if plot_idx < 4 and 'CAAR' in aar_data.columns:
            ax = axes[plot_idx]
            ax.plot(x, aar_data['CAAR'], linewidth=2, label=f'{event_type.upper()} CAAR', color='green')
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('相对时间 (天)')
            ax.set_ylabel('累积平均异常收益率')
            ax.set_title(f'{event_type.upper()}事件 CAAR')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plot_idx += 1
    
    # 隐藏未使用的子图
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.suptitle(f'事件研究结果汇总 - {outcome_var}', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # 保存图片
    filename = f"event_study_summary{suffix}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"事件研究汇总图已保存: {filepath}")

def generate_event_study_plots(results_dict, outcome_var='log_gk_volatility', 
                              output_dir=None, suffix=""):
    """
    生成所有事件研究相关图表
    
    Args:
        results_dict (dict): 事件研究结果字典
        outcome_var (str): 结果变量名称
        output_dir (str): 输出目录
        suffix (str): 文件名后缀
    """
    if output_dir is None:
        output_dir = config.PATH_OUTPUT_FIGURES
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"开始生成事件研究图表 - 输出目录: {output_dir}")
    
    # 为每种事件类型生成详细图表
    for event_type, results in results_dict.items():
        if 'aar_results' in results and not results['aar_results'].empty:
            plot_aar_time_series(results['aar_results'], event_type, outcome_var, output_dir, suffix)
        
        if 'events' in results and not results['events'].empty:
            plot_event_distribution(results['events'], event_type, output_dir, suffix)
        
        if 'car_data' in results and not results['car_data'].empty:
            plot_car_individual_events(results['car_data'], event_type, output_dir, suffix)
    
    # 生成汇总图
    plot_event_study_summary(results_dict, outcome_var, output_dir, suffix)
    
    logging.info("事件研究图表生成完成")


if __name__ == "__main__":
    logging.info("事件研究可视化模块测试")
    # 这里可以添加测试代码
