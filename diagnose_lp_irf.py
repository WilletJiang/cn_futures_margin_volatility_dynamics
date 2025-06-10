#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断LP-IRF置信区间过宽的问题
"""

import pandas as pd
import numpy as np
import sys

sys.path.append('.')
from src import config

def analyze_data_characteristics():
    """分析数据特征"""
    print("=== 数据特征分析 ===")
    
    # 加载主数据
    main_df = pd.read_parquet(config.PANEL_DATA_FILEPATH)
    main_df['date'] = pd.to_datetime(main_df['date'])
    
    print(f"数据形状: {main_df.shape}")
    print(f"时间范围: {main_df['date'].min()} 到 {main_df['date'].max()}")
    print(f"合约数量: {main_df['contract_id'].nunique()}")
    print(f"观测天数: {main_df['date'].nunique()}")
    
    # 分析面板平衡性
    contract_obs = main_df.groupby('contract_id').size()
    print(f"\n面板平衡性:")
    print(f"平均每合约观测数: {contract_obs.mean():.1f}")
    print(f"最少观测数: {contract_obs.min()}")
    print(f"最多观测数: {contract_obs.max()}")
    print(f"观测数标准差: {contract_obs.std():.1f}")
    
    return main_df

def analyze_margin_events(df):
    """分析保证金调整事件"""
    print("\n=== 保证金调整事件分析 ===")
    
    # 基本统计
    margin_changes = df['dlog_margin_rate'] != 0
    print(f"保证金调整事件数量: {margin_changes.sum()}")
    print(f"总观测数: {len(df)}")
    print(f"事件频率: {margin_changes.mean():.4f} ({margin_changes.mean()*100:.2f}%)")
    
    # 事件幅度分析
    margin_events = df[margin_changes]['dlog_margin_rate']
    if len(margin_events) > 0:
        print(f"\n保证金调整幅度统计:")
        print(f"平均幅度: {margin_events.mean():.6f}")
        print(f"标准差: {margin_events.std():.6f}")
        print(f"最大幅度: {margin_events.max():.6f}")
        print(f"最小幅度: {margin_events.min():.6f}")
        print(f"中位数: {margin_events.median():.6f}")
        
        # 分析增加和减少事件
        increase_events = margin_events[margin_events > 0]
        decrease_events = margin_events[margin_events < 0]
        print(f"\n增加事件: {len(increase_events)} 个")
        print(f"减少事件: {len(decrease_events)} 个")
        
        if len(increase_events) > 0:
            print(f"增加事件平均幅度: {increase_events.mean():.6f}")
        if len(decrease_events) > 0:
            print(f"减少事件平均幅度: {decrease_events.mean():.6f}")
    
    return margin_events

def analyze_outcome_variable(df):
    """分析结果变量"""
    print("\n=== 结果变量分析 ===")
    
    outcome_var = 'log_gk_volatility'
    if outcome_var in df.columns:
        y = df[outcome_var].dropna()
        print(f"结果变量: {outcome_var}")
        print(f"有效观测数: {len(y)}")
        print(f"缺失值数量: {df[outcome_var].isna().sum()}")
        print(f"平均值: {y.mean():.6f}")
        print(f"标准差: {y.std():.6f}")
        print(f"最小值: {y.min():.6f}")
        print(f"最大值: {y.max():.6f}")
        
        # 分析变异性
        print(f"变异系数: {y.std()/abs(y.mean()):.4f}")
        
        # 分析时间序列特征
        df_sorted = df.sort_values(['contract_id', 'date'])
        df_sorted['y_lag1'] = df_sorted.groupby('contract_id')[outcome_var].shift(1)
        autocorr = df_sorted[[outcome_var, 'y_lag1']].corr().iloc[0,1]
        print(f"一阶自相关: {autocorr:.4f}")
    
    return y

def analyze_lp_irf_results():
    """分析LP-IRF结果"""
    print("\n=== LP-IRF结果分析 ===")
    
    try:
        lp_df = pd.read_csv('output/tables/lp_irf_results_baseline.csv')
        
        print(f"LP-IRF结果数量: {len(lp_df)}")
        
        # 分析置信区间
        lp_df['ci_width'] = lp_df['conf_high'] - lp_df['conf_low']
        print(f"\n置信区间分析:")
        print(f"平均宽度: {lp_df['ci_width'].mean():.4f}")
        print(f"最大宽度: {lp_df['ci_width'].max():.4f}")
        print(f"最小宽度: {lp_df['ci_width'].min():.4f}")
        
        # 分析标准误
        print(f"\n标准误分析:")
        print(f"平均标准误: {lp_df['stderr'].mean():.4f}")
        print(f"最大标准误: {lp_df['stderr'].max():.4f}")
        print(f"最小标准误: {lp_df['stderr'].min():.4f}")
        
        # 分析系数
        print(f"\n系数分析:")
        print(f"平均系数绝对值: {lp_df['coeff'].abs().mean():.4f}")
        print(f"最大系数绝对值: {lp_df['coeff'].abs().max():.4f}")
        print(f"最小系数绝对值: {lp_df['coeff'].abs().min():.4f}")
        
        # 信噪比
        lp_df['signal_noise_ratio'] = lp_df['coeff'].abs() / lp_df['stderr']
        print(f"\n信噪比分析:")
        print(f"平均信噪比: {lp_df['signal_noise_ratio'].mean():.4f}")
        print(f"最大信噪比: {lp_df['signal_noise_ratio'].max():.4f}")
        print(f"最小信噪比: {lp_df['signal_noise_ratio'].min():.4f}")
        
        # 显著性
        significant = lp_df['pval'] < 0.05
        print(f"\n显著性分析:")
        print(f"显著系数数量: {significant.sum()}")
        print(f"总系数数量: {len(lp_df)}")
        print(f"显著比例: {significant.mean():.4f}")
        
        # 按事件类型分析
        print(f"\n按事件类型分析:")
        for shock_type in lp_df['shock_type'].unique():
            subset = lp_df[lp_df['shock_type'] == shock_type]
            sig_subset = subset['pval'] < 0.05
            print(f"{shock_type}: {sig_subset.sum()}/{len(subset)} 显著 ({sig_subset.mean():.2f})")
        
        return lp_df
        
    except FileNotFoundError:
        print("LP-IRF结果文件未找到")
        return None

def diagnose_problems(main_df, margin_events, lp_df):
    """诊断可能的问题"""
    print("\n=== 问题诊断 ===")
    
    problems = []
    
    # 1. 检查事件频率
    event_freq = (main_df['dlog_margin_rate'] != 0).mean()
    if event_freq < 0.01:  # 少于1%
        problems.append(f"事件频率过低 ({event_freq:.4f})")
    
    # 2. 检查事件幅度
    if len(margin_events) > 0:
        avg_magnitude = margin_events.abs().mean()
        if avg_magnitude < 0.01:  # 平均幅度小于1%
            problems.append(f"事件幅度过小 (平均{avg_magnitude:.4f})")
    
    # 3. 检查样本量
    if len(main_df) < 1000:
        problems.append(f"样本量过小 ({len(main_df)})")
    
    # 4. 检查面板平衡性
    contract_obs = main_df.groupby('contract_id').size()
    if contract_obs.std() / contract_obs.mean() > 0.5:
        problems.append("面板数据严重不平衡")
    
    # 5. 检查LP-IRF结果
    if lp_df is not None:
        avg_stderr = lp_df['stderr'].mean()
        avg_coeff = lp_df['coeff'].abs().mean()
        if avg_stderr > avg_coeff:
            problems.append("标准误大于系数，信号微弱")
        
        sig_ratio = (lp_df['pval'] < 0.05).mean()
        if sig_ratio < 0.1:
            problems.append(f"显著系数比例过低 ({sig_ratio:.2f})")
    
    if problems:
        print("发现的问题:")
        for i, problem in enumerate(problems, 1):
            print(f"{i}. {problem}")
    else:
        print("未发现明显问题")
    
    return problems

def suggest_solutions(problems):
    """建议解决方案"""
    print("\n=== 建议解决方案 ===")
    
    solutions = []
    
    if any("事件频率过低" in p for p in problems):
        solutions.append("1. 降低事件识别阈值")
        solutions.append("2. 使用更长的时间序列数据")
        solutions.append("3. 考虑使用连续的保证金率变化而非二元事件")
    
    if any("事件幅度过小" in p for p in problems):
        solutions.append("4. 检查保证金率数据的单位和计算方法")
        solutions.append("5. 考虑使用累积效应或更长的事件窗口")
    
    if any("样本量过小" in p for p in problems):
        solutions.append("6. 增加更多合约或更长时间序列")
        solutions.append("7. 考虑使用更高频率的数据")
    
    if any("面板数据严重不平衡" in p for p in problems):
        solutions.append("8. 使用平衡面板数据子集")
        solutions.append("9. 在回归中加入固定效应")
    
    if any("标准误大于系数" in p for p in problems):
        solutions.append("10. 增加控制变量减少噪音")
        solutions.append("11. 使用聚类标准误")
        solutions.append("12. 考虑使用工具变量方法")
    
    if any("显著系数比例过低" in p for p in problems):
        solutions.append("13. 检查模型设定是否正确")
        solutions.append("14. 考虑非线性效应")
        solutions.append("15. 使用更稳健的估计方法")
    
    if solutions:
        print("建议的解决方案:")
        for solution in solutions:
            print(f"  {solution}")
    else:
        print("数据质量良好，可能需要调整模型设定")

def main():
    """主诊断函数"""
    print("LP-IRF置信区间过宽问题诊断")
    print("=" * 50)
    
    try:
        # 分析数据特征
        main_df = analyze_data_characteristics()
        
        # 分析保证金事件
        margin_events = analyze_margin_events(main_df)
        
        # 分析结果变量
        outcome_var = analyze_outcome_variable(main_df)
        
        # 分析LP-IRF结果
        lp_df = analyze_lp_irf_results()
        
        # 诊断问题
        problems = diagnose_problems(main_df, margin_events, lp_df)
        
        # 建议解决方案
        suggest_solutions(problems)
        
        print("\n" + "=" * 50)
        print("诊断完成")
        
    except Exception as e:
        print(f"诊断过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
