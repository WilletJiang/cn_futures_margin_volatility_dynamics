#!/bin/bash
# 主执行脚本 - 运行完整分析流程

# 设置工作目录为项目根目录
cd "$(dirname "$0")"

# 创建日志目录
mkdir -p output/logs

# 设置日志文件
LOG_FILE="output/logs/analysis_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "开始执行期货保证金调整与价格波动动态关系分析"
echo "开始时间: $(date)"
echo "=========================================="

# 1. 数据处理阶段
echo "1. 开始数据处理..."
python -m src.data_processing.build_features
echo "数据处理完成!"

# 2. 差分法(DID)与横截面分析
echo "2. 开始差分法分析..."
python -m src.analysis.did_cs_analysis
echo "差分法分析完成!"

# 3. 局部投影脉冲响应函数(LP-IRF)分析
echo "3. 开始局部投影脉冲响应函数分析..."
python -m src.analysis.lp_irf_analysis
echo "局部投影分析完成!"

# 4. 稳健性检验
echo "4. 开始稳健性检验..."
python -m src.robustness.run_robustness
echo "稳健性检验完成!"

# 5. 生成图表结果
echo "5. 开始生成结果图表..."
python -m src.visualization.plot_did_results
python -m src.visualization.plot_lp_irf_results
python -m src.visualization.generate_tables
echo "结果图表生成完成!"

echo "=========================================="
echo "分析流程全部完成!"
echo "结束时间: $(date)"
echo "输出结果位于: $(pwd)/output/"
echo "日志文件: $LOG_FILE"
echo "=========================================="
