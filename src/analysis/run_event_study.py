# -*- coding: utf-8 -*-
"""
事件研究分析主执行脚本

整合事件研究方法的完整分析流程，包括：
1. 事件研究分析
2. 结果可视化
3. 与LP-IRF结果的比较分析
4. 稳健性检验
"""

import pandas as pd
import numpy as np
import os
import logging
import sys
from datetime import datetime

# 项目配置和模块导入
try:
    from src import config
    from src.analysis.event_study_analysis import run_event_study_analysis_core
    from src.visualization.plot_event_study_results import generate_event_study_plots
    from src.analysis.comparative_analysis import run_comparative_analysis_core
except ImportError:
    # 如果直接运行脚本，调整路径
    PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PACKAGE_DIR)
    from src import config
    from src.analysis.event_study_analysis import run_event_study_analysis_core
    from src.visualization.plot_event_study_results import generate_event_study_plots
    from src.analysis.comparative_analysis import run_comparative_analysis_core

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.PATH_OUTPUT, 'logs', 'event_study_analysis.log')),
        logging.StreamHandler()
    ]
)

def run_main_event_study_analysis():
    """
    执行主要的事件研究分析
    """
    logging.info("=== 开始主要事件研究分析 ===")
    
    # 加载主数据
    try:
        if not os.path.exists(config.PANEL_DATA_FILEPATH):
            logging.error(f"主面板数据文件未找到: {config.PANEL_DATA_FILEPATH}")
            logging.error("请先运行 src/data_processing/build_features.py")
            return False
        
        main_df = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        main_df['date'] = pd.to_datetime(main_df['date'])
        logging.info(f"主数据加载成功: {main_df.shape}")
        
        # 执行事件研究分析
        logging.info("--- 执行主要事件研究分析 ---")
        results = run_event_study_analysis_core(
            data=main_df,
            outcome_var='log_gk_volatility',
            output_table_dir=config.PATH_OUTPUT_TABLES,
            output_suffix=""  # 主分析无后缀
        )
        
        if results:
            logging.info("主要事件研究分析成功完成")
            
            # 生成可视化图表
            logging.info("--- 生成事件研究可视化图表 ---")
            generate_event_study_plots(
                results_dict=results,
                outcome_var='log_gk_volatility',
                output_dir=config.PATH_OUTPUT_FIGURES,
                suffix=""
            )
            
            return True
        else:
            logging.error("主要事件研究分析失败")
            return False
    
    except Exception as e:
        logging.error(f"主要事件研究分析时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_robustness_checks():
    """
    执行稳健性检验
    """
    logging.info("=== 开始事件研究稳健性检验 ===")
    
    try:
        main_df = pd.read_parquet(config.PANEL_DATA_FILEPATH)
        main_df['date'] = pd.to_datetime(main_df['date'])
        
        # 1. 使用替代波动率指标
        logging.info("--- 稳健性检验1: 使用Parkinson波动率 ---")
        if 'log_parkinson_volatility' in main_df.columns:
            results_parkinson = run_event_study_analysis_core(
                data=main_df,
                outcome_var='log_parkinson_volatility',
                output_table_dir=config.PATH_OUTPUT_TABLES,
                output_suffix="_parkinson_vol"
            )
            
            if results_parkinson:
                generate_event_study_plots(
                    results_dict=results_parkinson,
                    outcome_var='log_parkinson_volatility',
                    output_dir=config.PATH_OUTPUT_FIGURES,
                    suffix="_parkinson_vol"
                )
                logging.info("Parkinson波动率稳健性检验完成")
        else:
            logging.warning("数据中缺少log_parkinson_volatility列，跳过此稳健性检验")
        
        # 2. 不同事件阈值
        logging.info("--- 稳健性检验2: 使用更严格的事件识别阈值 ---")
        results_strict = run_event_study_analysis_core(
            data=main_df,
            outcome_var='log_gk_volatility',
            output_table_dir=config.PATH_OUTPUT_TABLES,
            output_suffix="_strict_threshold",
            threshold=1e-4  # 更严格的阈值
        )
        
        if results_strict:
            generate_event_study_plots(
                results_dict=results_strict,
                outcome_var='log_gk_volatility',
                output_dir=config.PATH_OUTPUT_FIGURES,
                suffix="_strict_threshold"
            )
            logging.info("严格阈值稳健性检验完成")
        
        # 3. 排除极端事件
        logging.info("--- 稳健性检验3: 排除极端保证金调整事件 ---")
        if 'dlog_margin_rate' in main_df.columns:
            # 排除最极端的1%事件
            abs_margin_change = main_df['dlog_margin_rate'].abs()
            threshold_99 = abs_margin_change.quantile(0.99)
            
            filtered_df = main_df[abs_margin_change <= threshold_99].copy()
            logging.info(f"排除极端事件后数据量: {len(filtered_df)} (原始: {len(main_df)})")
            
            results_filtered = run_event_study_analysis_core(
                data=filtered_df,
                outcome_var='log_gk_volatility',
                output_table_dir=config.PATH_OUTPUT_TABLES,
                output_suffix="_filtered_extreme"
            )
            
            if results_filtered:
                generate_event_study_plots(
                    results_dict=results_filtered,
                    outcome_var='log_gk_volatility',
                    output_dir=config.PATH_OUTPUT_FIGURES,
                    suffix="_filtered_extreme"
                )
                logging.info("排除极端事件稳健性检验完成")
        
        # 4. 按品种类型分组分析
        logging.info("--- 稳健性检验4: 按品种类型分组分析 ---")
        if 'variety' in main_df.columns:
            variety_counts = main_df['variety'].value_counts()
            major_varieties = variety_counts[variety_counts >= 1000].index  # 至少1000个观测
            
            for variety in major_varieties[:3]:  # 分析前3个主要品种
                logging.info(f"分析品种: {variety}")
                variety_df = main_df[main_df['variety'] == variety].copy()
                
                # 清理品种名称用作文件名
                safe_variety = "".join(filter(str.isalnum, str(variety)))
                
                results_variety = run_event_study_analysis_core(
                    data=variety_df,
                    outcome_var='log_gk_volatility',
                    output_table_dir=config.PATH_OUTPUT_TABLES,
                    output_suffix=f"_variety_{safe_variety}"
                )
                
                if results_variety:
                    generate_event_study_plots(
                        results_dict=results_variety,
                        outcome_var='log_gk_volatility',
                        output_dir=config.PATH_OUTPUT_FIGURES,
                        suffix=f"_variety_{safe_variety}"
                    )
        
        logging.info("事件研究稳健性检验完成")
        return True
    
    except Exception as e:
        logging.error(f"稳健性检验时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comparative_analysis():
    """
    执行与LP-IRF结果的比较分析
    """
    logging.info("=== 开始LP-IRF与事件研究比较分析 ===")
    
    try:
        # 主分析比较
        logging.info("--- 主分析比较 ---")
        comparison_results = run_comparative_analysis_core(
            lp_irf_results_dir=config.PATH_OUTPUT_TABLES,
            event_study_results_dir=config.PATH_OUTPUT_TABLES,
            output_dir=config.PATH_OUTPUT_FIGURES,
            suffix=""
        )
        
        if comparison_results:
            logging.info("主分析比较完成")
        
        # 稳健性检验比较 (如果存在相应的LP-IRF结果)
        robustness_suffixes = ["_parkinson_vol", "_strict_threshold", "_filtered_extreme"]
        
        for suffix in robustness_suffixes:
            lp_file = f"lp_irf_results_baseline{suffix}.csv"
            es_file = f"event_study_aar_increase{suffix}.csv"
            
            lp_path = os.path.join(config.PATH_OUTPUT_TABLES, lp_file)
            es_path = os.path.join(config.PATH_OUTPUT_TABLES, es_file)
            
            if os.path.exists(lp_path) and os.path.exists(es_path):
                logging.info(f"--- 比较分析{suffix} ---")
                comparison_results_robust = run_comparative_analysis_core(
                    lp_irf_results_dir=config.PATH_OUTPUT_TABLES,
                    event_study_results_dir=config.PATH_OUTPUT_TABLES,
                    output_dir=config.PATH_OUTPUT_FIGURES,
                    suffix=suffix
                )
                
                if comparison_results_robust:
                    logging.info(f"比较分析{suffix}完成")
            else:
                logging.info(f"跳过比较分析{suffix} - 缺少必要文件")
        
        logging.info("LP-IRF与事件研究比较分析完成")
        return True
    
    except Exception as e:
        logging.error(f"比较分析时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_summary_report():
    """
    生成事件研究分析总结报告
    """
    logging.info("=== 生成事件研究分析总结报告 ===")
    
    try:
        report_lines = []
        report_lines.append("# 事件研究分析总结报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 检查主要结果文件
        main_files = [
            "event_study_aar_increase.csv",
            "event_study_aar_decrease.csv",
            "event_study_events_increase.csv",
            "event_study_events_decrease.csv"
        ]
        
        report_lines.append("## 主要分析结果文件")
        for file in main_files:
            file_path = os.path.join(config.PATH_OUTPUT_TABLES, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                report_lines.append(f"- {file}: {len(df)} 行数据")
            else:
                report_lines.append(f"- {file}: 文件不存在")
        
        report_lines.append("")
        
        # 检查图表文件
        figure_files = [f for f in os.listdir(config.PATH_OUTPUT_FIGURES) 
                       if f.startswith('event_study_') and f.endswith('.png')]
        
        report_lines.append("## 生成的图表文件")
        for file in sorted(figure_files):
            report_lines.append(f"- {file}")
        
        report_lines.append("")
        
        # 检查比较分析文件
        comparison_files = [f for f in os.listdir(config.PATH_OUTPUT_FIGURES) 
                           if f.startswith('lp_irf_vs_event_study_') and f.endswith('.png')]
        
        report_lines.append("## 比较分析图表文件")
        for file in sorted(comparison_files):
            report_lines.append(f"- {file}")
        
        # 保存报告
        report_path = os.path.join(config.PATH_OUTPUT, "event_study_analysis_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logging.info(f"总结报告已保存: {report_path}")
        return True
    
    except Exception as e:
        logging.error(f"生成总结报告时出错: {e}")
        return False

def main():
    """
    主函数 - 执行完整的事件研究分析流程
    """
    logging.info("开始执行完整的事件研究分析流程")
    
    # 确保输出目录存在
    os.makedirs(os.path.join(config.PATH_OUTPUT, 'logs'), exist_ok=True)
    
    success_count = 0
    total_steps = 4
    
    # 1. 主要事件研究分析
    if run_main_event_study_analysis():
        success_count += 1
    
    # 2. 稳健性检验
    if run_robustness_checks():
        success_count += 1
    
    # 3. 比较分析
    if run_comparative_analysis():
        success_count += 1
    
    # 4. 生成总结报告
    if generate_summary_report():
        success_count += 1
    
    # 输出最终结果
    logging.info(f"事件研究分析流程完成: {success_count}/{total_steps} 步骤成功")
    
    if success_count == total_steps:
        logging.info("所有分析步骤均成功完成！")
        print("\n" + "="*60)
        print("事件研究分析完成！")
        print(f"结果保存在: {config.PATH_OUTPUT}")
        print("主要输出文件:")
        print(f"- 表格: {config.PATH_OUTPUT_TABLES}")
        print(f"- 图表: {config.PATH_OUTPUT_FIGURES}")
        print(f"- 报告: {os.path.join(config.PATH_OUTPUT, 'event_study_analysis_report.md')}")
        print("="*60)
    else:
        logging.warning("部分分析步骤未成功完成，请检查日志")

if __name__ == "__main__":
    main()
