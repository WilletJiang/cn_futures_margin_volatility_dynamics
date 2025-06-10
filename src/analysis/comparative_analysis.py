# -*- coding: utf-8 -*-
"""
LP-IRF与事件研究比较分析

对比局部投影脉冲响应函数(LP-IRF)和事件研究方法的结果，
提供互补的证据来理解保证金调整对期货市场动态的影响。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy import stats
from scipy.stats import pearsonr, spearmanr

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

class ComparativeAnalyzer:
    """
    LP-IRF与事件研究比较分析器
    """
    
    def __init__(self, lp_irf_results_dir, event_study_results_dir, output_dir):
        """
        初始化比较分析器
        
        Args:
            lp_irf_results_dir (str): LP-IRF结果目录
            event_study_results_dir (str): 事件研究结果目录
            output_dir (str): 输出目录
        """
        self.lp_irf_results_dir = lp_irf_results_dir
        self.event_study_results_dir = event_study_results_dir
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        logging.info("比较分析器初始化完成")
    
    def load_lp_irf_results(self, suffix=""):
        """
        加载LP-IRF分析结果
        
        Args:
            suffix (str): 文件名后缀
        
        Returns:
            dict: LP-IRF结果字典
        """
        lp_results = {}
        
        # 加载基准结果
        baseline_file = f"lp_irf_results_baseline{suffix}.csv"
        baseline_path = os.path.join(self.lp_irf_results_dir, baseline_file)
        
        if os.path.exists(baseline_path):
            lp_results['baseline'] = pd.read_csv(baseline_path)
            logging.info(f"LP-IRF基准结果加载成功: {baseline_path}")
        else:
            logging.warning(f"LP-IRF基准结果文件未找到: {baseline_path}")
        
        # 加载状态依赖结果
        statedep_file = f"lp_irf_results_statedep{suffix}.csv"
        statedep_path = os.path.join(self.lp_irf_results_dir, statedep_file)
        
        if os.path.exists(statedep_path):
            lp_results['statedep'] = pd.read_csv(statedep_path)
            logging.info(f"LP-IRF状态依赖结果加载成功: {statedep_path}")
        else:
            logging.warning(f"LP-IRF状态依赖结果文件未找到: {statedep_path}")
        
        return lp_results
    
    def load_event_study_results(self, suffix=""):
        """
        加载事件研究结果
        
        Args:
            suffix (str): 文件名后缀
        
        Returns:
            dict: 事件研究结果字典
        """
        es_results = {}
        
        for event_type in ['increase', 'decrease']:
            aar_file = f"event_study_aar_{event_type}{suffix}.csv"
            aar_path = os.path.join(self.event_study_results_dir, aar_file)
            
            if os.path.exists(aar_path):
                es_results[event_type] = pd.read_csv(aar_path)
                logging.info(f"事件研究AAR结果加载成功: {aar_path}")
            else:
                logging.warning(f"事件研究AAR结果文件未找到: {aar_path}")
        
        return es_results
    
    def align_time_horizons(self, lp_data, es_data, max_horizon=10):
        """
        对齐LP-IRF和事件研究的时间范围
        
        Args:
            lp_data (pd.DataFrame): LP-IRF数据
            es_data (pd.DataFrame): 事件研究数据
            max_horizon (int): 最大时间范围
        
        Returns:
            tuple: 对齐后的(lp_aligned, es_aligned)
        """
        # LP-IRF数据对齐 (horizon对应relative_time)
        lp_aligned = lp_data[lp_data['horizon'] <= max_horizon].copy()
        lp_aligned['time'] = lp_aligned['horizon']
        
        # 事件研究数据对齐
        es_aligned = es_data[
            (es_data['relative_time'] >= 0) & 
            (es_data['relative_time'] <= max_horizon)
        ].copy()
        es_aligned['time'] = es_aligned['relative_time']
        
        return lp_aligned, es_aligned
    
    def calculate_correlation(self, lp_data, es_data, lp_coeff_col='coeff', es_coeff_col='AAR'):
        """
        计算LP-IRF和事件研究结果的相关性
        
        Args:
            lp_data (pd.DataFrame): LP-IRF数据
            es_data (pd.DataFrame): 事件研究数据
            lp_coeff_col (str): LP-IRF系数列名
            es_coeff_col (str): 事件研究系数列名
        
        Returns:
            dict: 相关性分析结果
        """
        # 合并数据
        merged = pd.merge(
            lp_data[['time', lp_coeff_col]],
            es_data[['time', es_coeff_col]],
            on='time',
            how='inner'
        )
        
        if merged.empty:
            logging.warning("无法合并LP-IRF和事件研究数据进行相关性分析")
            return {}
        
        # 计算相关系数
        pearson_corr, pearson_p = pearsonr(merged[lp_coeff_col], merged[es_coeff_col])
        spearman_corr, spearman_p = spearmanr(merged[lp_coeff_col], merged[es_coeff_col])
        
        results = {
            'n_observations': len(merged),
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'merged_data': merged
        }
        
        logging.info(f"相关性分析完成 - Pearson: {pearson_corr:.3f} (p={pearson_p:.3f}), "
                    f"Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")
        
        return results
    
    def plot_comparison(self, lp_data, es_data, event_type, suffix=""):
        """
        绘制LP-IRF与事件研究结果的比较图
        
        Args:
            lp_data (pd.DataFrame): LP-IRF数据
            es_data (pd.DataFrame): 事件研究数据
            event_type (str): 事件类型
            suffix (str): 文件名后缀
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 上图：系数比较
        if not lp_data.empty:
            ax1.plot(lp_data['time'], lp_data['coeff'], 'b-o', linewidth=2,
                    markersize=6, label='LP-IRF Coefficients')
            ax1.fill_between(lp_data['time'], lp_data['conf_low'], lp_data['conf_high'],
                           alpha=0.3, color='blue', label='LP-IRF 95% CI')

        if not es_data.empty:
            ax1.plot(es_data['time'], es_data['AAR'], 'r-s', linewidth=2,
                    markersize=6, label='Event Study AAR')
            ax1.fill_between(es_data['time'], es_data['CI_lower'], es_data['CI_upper'],
                           alpha=0.3, color='red', label='Event Study 95% CI')

        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time (Days)')
        ax1.set_ylabel('Coefficients/Abnormal Returns')
        ax1.set_title(f'LP-IRF vs Event Study - {event_type.upper()} Events')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：散点图和相关性
        correlation_results = self.calculate_correlation(lp_data, es_data)
        
        if correlation_results and not correlation_results['merged_data'].empty:
            merged = correlation_results['merged_data']
            ax2.scatter(merged['coeff'], merged['AAR'], alpha=0.7, s=60)
            
            # 添加拟合线
            z = np.polyfit(merged['coeff'], merged['AAR'], 1)
            p = np.poly1d(z)
            ax2.plot(merged['coeff'], p(merged['coeff']), "r--", alpha=0.8)
            
            # 添加相关系数信息
            pearson_r = correlation_results['pearson_correlation']
            pearson_p = correlation_results['pearson_p_value']
            ax2.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}\np = {pearson_p:.3f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_xlabel('LP-IRF Coefficients')
        ax2.set_ylabel('Event Study AAR')
        ax2.set_title('LP-IRF vs Event Study Correlation')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"lp_irf_vs_event_study_{event_type}{suffix}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"比较图已保存: {filepath}")
        
        return correlation_results
    
    def generate_comparison_table(self, correlation_results, event_type, suffix=""):
        """
        生成比较分析结果表格
        
        Args:
            correlation_results (dict): 相关性分析结果
            event_type (str): 事件类型
            suffix (str): 文件名后缀
        """
        if not correlation_results:
            logging.warning(f"无相关性结果可生成表格: {event_type}")
            return
        
        # 创建结果表格
        results_table = pd.DataFrame({
            '指标': ['观测数量', 'Pearson相关系数', 'Pearson p值', 'Spearman相关系数', 'Spearman p值'],
            '数值': [
                correlation_results['n_observations'],
                f"{correlation_results['pearson_correlation']:.4f}",
                f"{correlation_results['pearson_p_value']:.4f}",
                f"{correlation_results['spearman_correlation']:.4f}",
                f"{correlation_results['spearman_p_value']:.4f}"
            ]
        })
        
        # 保存表格
        filename = f"lp_irf_vs_event_study_correlation_{event_type}{suffix}.csv"
        filepath = os.path.join(self.output_dir, filename)
        results_table.to_csv(filepath, index=False)
        
        logging.info(f"比较分析表格已保存: {filepath}")
    
    def run_comparative_analysis(self, suffix=""):
        """
        执行完整的比较分析
        
        Args:
            suffix (str): 文件名后缀
        
        Returns:
            dict: 比较分析结果
        """
        logging.info("开始执行LP-IRF与事件研究比较分析")
        
        # 加载数据
        lp_results = self.load_lp_irf_results(suffix)
        es_results = self.load_event_study_results(suffix)
        
        if not lp_results or not es_results:
            logging.error("无法加载必要的分析结果文件")
            return {}
        
        comparison_results = {}
        
        # 对每种事件类型进行比较
        for event_type in ['increase', 'decrease']:
            if event_type not in es_results:
                logging.warning(f"事件研究结果中缺少 {event_type} 类型")
                continue
            
            # 获取对应的LP-IRF数据
            if 'baseline' in lp_results:
                lp_data = lp_results['baseline'][
                    lp_results['baseline']['shock_type'] == event_type
                ].copy()
            else:
                logging.warning(f"LP-IRF基准结果中缺少 {event_type} 类型")
                continue
            
            es_data = es_results[event_type]
            
            # 对齐时间范围
            lp_aligned, es_aligned = self.align_time_horizons(lp_data, es_data)
            
            if lp_aligned.empty or es_aligned.empty:
                logging.warning(f"无法对齐 {event_type} 事件的时间范围")
                continue
            
            # 绘制比较图并计算相关性
            correlation_results = self.plot_comparison(lp_aligned, es_aligned, event_type, suffix)
            
            # 生成比较表格
            self.generate_comparison_table(correlation_results, event_type, suffix)
            
            comparison_results[event_type] = {
                'lp_data': lp_aligned,
                'es_data': es_aligned,
                'correlation': correlation_results
            }
        
        logging.info("LP-IRF与事件研究比较分析完成")
        
        return comparison_results


def run_comparative_analysis_core(lp_irf_results_dir=None, event_study_results_dir=None,
                                 output_dir=None, suffix=""):
    """
    执行比较分析的核心函数 (可被其他脚本调用)
    
    Args:
        lp_irf_results_dir (str): LP-IRF结果目录
        event_study_results_dir (str): 事件研究结果目录
        output_dir (str): 输出目录
        suffix (str): 文件名后缀
    
    Returns:
        dict: 比较分析结果
    """
    # 使用默认目录
    if lp_irf_results_dir is None:
        lp_irf_results_dir = config.PATH_OUTPUT_TABLES
    if event_study_results_dir is None:
        event_study_results_dir = config.PATH_OUTPUT_TABLES
    if output_dir is None:
        output_dir = config.PATH_OUTPUT_FIGURES
    
    # 创建比较分析器
    analyzer = ComparativeAnalyzer(
        lp_irf_results_dir=lp_irf_results_dir,
        event_study_results_dir=event_study_results_dir,
        output_dir=output_dir
    )
    
    # 执行分析
    results = analyzer.run_comparative_analysis(suffix=suffix)
    
    return results


if __name__ == "__main__":
    logging.info("直接运行比较分析脚本...")
    
    try:
        # 执行默认比较分析
        results = run_comparative_analysis_core()
        
        if results:
            logging.info("比较分析成功完成。")
        else:
            logging.error("比较分析执行过程中出现错误。")
    
    except Exception as e:
        logging.error(f"直接运行比较分析时出错: {e}")
        import traceback
        traceback.print_exc()
