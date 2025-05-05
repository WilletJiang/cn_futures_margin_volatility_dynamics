#!/bin/bash

# 主执行脚本
# 按顺序运行数据处理、分析、可视化和稳健性检验

echo "========================================"
echo "开始执行期货保证金波动率动态响应分析..."
echo "========================================"

# 步骤 1: 数据处理与特征构建
echo "\n[步骤 1/5] 运行数据处理与特征构建 (build_features.py)..."
python src/data_processing/build_features.py
if [ $? -ne 0 ]; then
    echo "错误：数据处理失败，脚本终止。"
    exit 1
fi
echo "数据处理完成。"

# 步骤 2: 生成描述性统计表格
echo "\n[步骤 2/5] 生成描述性统计表格 (generate_tables.py)..."
python src/visualization/generate_tables.py
if [ $? -ne 0 ]; then
    echo "警告：生成描述性统计表格失败。"
    # 选择不退出，因为这可能不是关键步骤
fi
echo "描述性统计表格生成完成。"

# 步骤 3: 运行核心分析 (DID 和 LP-IRF)
echo "\n[步骤 3/5] 运行核心分析..."
echo "  - 运行 DID 分析 (did_cs_analysis.py)..."
python src/analysis/did_cs_analysis.py
if [ $? -ne 0 ]; then
    echo "错误：DID 分析失败，脚本终止。"
    exit 1
fi
echo "  - 运行 LP-IRF 分析 (lp_irf_analysis.py)..."
python src/analysis/lp_irf_analysis.py
if [ $? -ne 0 ]; then
    echo "错误：LP-IRF 分析失败，脚本终止。"
    exit 1
fi
echo "核心分析完成。"

# 步骤 4: 生成核心分析结果的可视化图表
echo "\n[步骤 4/5] 生成可视化图表..."
echo "  - 生成 DID 事件研究图 (plot_did_results.py)..."
python src/visualization/plot_did_results.py
if [ $? -ne 0 ]; then
    echo "警告：生成 DID 图表失败。"
fi
echo "  - 生成 LP-IRF 图 (plot_lp_irf_results.py)..."
python src/visualization/plot_lp_irf_results.py
if [ $? -ne 0 ]; then
    echo "警告：生成 LP-IRF 图表失败。"
fi
echo "可视化图表生成完成。"

# 步骤 5: 运行稳健性检验
echo "\n[步骤 5/5] 运行稳健性检验 (run_robustness.py)..."
# 默认运行所有已实现的检验 (目前主要是 alt_vol)
python src/robustness/run_robustness.py --test all
if [ $? -ne 0 ]; then
    echo "警告：运行稳健性检验时出现错误。"
fi
echo "稳健性检验运行完成。"


echo "\n========================================"
echo "分析流程执行完毕。"
echo "请检查 'output/' 目录下的表格和图表。"
echo "注意：部分稳健性检验需要进一步实现。"
echo "========================================"
