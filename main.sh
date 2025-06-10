#!/bin/bash

# 主执行脚本
# 按顺序运行数据处理、LP-IRF分析、事件研究分析、可视化和稳健性检验
#
# 更新日期: 2024-12-10
# 新增功能: 事件研究方法、英文图表、比较分析
#
# 使用方法:
#   ./main.sh           # 运行完整流程
#   ./main.sh --quick   # 快速模式(跳过稳健性检验)
#   ./main.sh --help    # 显示帮助信息

# 解析命令行参数
QUICK_MODE=false
SHOW_HELP=false

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "未知参数: $arg"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 显示帮助信息
if [ "$SHOW_HELP" = true ]; then
    echo "期货保证金波动率动态响应分析 - 主执行脚本"
    echo ""
    echo "使用方法:"
    echo "  ./main.sh           运行完整分析流程"
    echo "  ./main.sh --quick   快速模式(跳过传统稳健性检验)"
    echo "  ./main.sh --help    显示此帮助信息"
    echo ""
    echo "完整流程包含:"
    echo "  1. 数据处理与特征构建"
    echo "  2. 描述性统计表格生成"
    echo "  3. LP-IRF分析"
    echo "  4. 事件研究分析(含稳健性检验)"
    echo "  5. 可视化图表生成(英文版)"
    echo "  6. 传统稳健性检验"
    echo ""
    echo "输出文件位置:"
    echo "  - 表格: output/tables/"
    echo "  - 图表: output/figures/"
    echo "  - 报告: output/event_study_analysis_report.md"
    exit 0
fi

echo "========================================"
echo "期货保证金波动率动态响应分析 - 完整流程"
echo "========================================"
echo "包含: LP-IRF分析 + 事件研究方法 + 比较分析"
echo "输出: 英文图表 + 中文注释代码"
if [ "$QUICK_MODE" = true ]; then
    echo "模式: 快速模式 (跳过传统稳健性检验)"
else
    echo "模式: 完整模式"
fi
echo "========================================"

# 记录开始时间
START_TIME=$(date +%s)

# 创建输出目录
echo "创建输出目录..."
mkdir -p output/tables output/figures output/logs
echo "输出目录准备完成。"

# 步骤 1: 数据处理与特征构建
echo "\n[步骤 1/6] 运行数据处理与特征构建..."
echo "  - 执行 build_features.py"
python src/data_processing/build_features.py
if [ $? -ne 0 ]; then
    echo "错误：数据处理失败，脚本终止。"
    exit 1
fi
echo "数据处理完成。"

# 步骤 2: 生成描述性统计表格
echo "\n[步骤 2/6] 生成描述性统计表格..."
echo "  - 执行 generate_tables.py"
python src/visualization/generate_tables.py
if [ $? -ne 0 ]; then
    echo "警告：生成描述性统计表格失败。"
    # 选择不退出，因为这可能不是关键步骤
fi
echo "描述性统计表格生成完成。"

# 步骤 3: 运行LP-IRF分析
echo "\n[步骤 3/6] 运行LP-IRF分析..."
echo "  - 执行 lp_irf_analysis.py"
python src/analysis/lp_irf_analysis.py
if [ $? -ne 0 ]; then
    echo "错误：LP-IRF 分析失败，脚本终止。"
    exit 1
fi
echo "LP-IRF分析完成。"

# 步骤 4: 运行事件研究分析
echo "\n[步骤 4/6] 运行事件研究分析..."
echo "  - 执行完整的事件研究分析流程"
echo "  - 包含主要分析、稳健性检验、比较分析"
python src/analysis/run_event_study.py
if [ $? -ne 0 ]; then
    echo "错误：事件研究分析失败，脚本终止。"
    exit 1
fi
echo "事件研究分析完成。"

# 步骤 5: 生成可视化图表
echo "\n[步骤 5/6] 生成可视化图表..."
echo "  - 生成LP-IRF图表 (英文版)"
python src/visualization/plot_lp_irf_results.py
if [ $? -ne 0 ]; then
    echo "警告：生成LP-IRF图表失败。"
fi
echo "所有图表生成完成。"

# 步骤 6: 运行传统稳健性检验 (可选)
if [ "$QUICK_MODE" = false ]; then
    echo "\n[步骤 6/6] 运行传统稳健性检验..."
    echo "  - 执行 run_robustness.py (LP-IRF稳健性检验)"
    # 默认运行所有已实现的检验 (目前主要是 alt_vol)
    python src/robustness/run_robustness.py --test all
    if [ $? -ne 0 ]; then
        echo "⚠️  警告：运行传统稳健性检验时出现错误。"
    fi
    echo "传统稳健性检验完成。"
else
    echo "\n[步骤 6/6] 跳过传统稳健性检验 (快速模式)"
    echo "  - 注意：事件研究的稳健性检验已在步骤4中完成"
fi

# 计算执行时间
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))
MINUTES=$((EXECUTION_TIME / 60))
SECONDS=$((EXECUTION_TIME % 60))

echo "\n========================================"
echo "完整分析流程执行完毕！"
echo "========================================"
echo "总执行时间: ${MINUTES}分${SECONDS}秒"
echo ""
echo "生成的结果文件:"
echo "  - 数据表格: output/tables/"
echo "  - 图表文件: output/figures/"
echo "  - 分析报告: output/event_study_analysis_report.md"
echo ""
echo "主要分析结果:"
echo "  - LP-IRF分析: 局部投影脉冲响应函数"
echo "  - 事件研究: 平均异常收益率(AAR)和累积异常收益率(CAAR)"
echo "  - 比较分析: LP-IRF vs 事件研究方法比较"
if [ "$QUICK_MODE" = false ]; then
    echo "  - 稳健性检验: 多种替代指标和方法验证"
else
    echo "  - 稳健性检验: 事件研究稳健性检验已完成"
fi
echo ""
echo "关键文件:"
echo "  - LP-IRF结果: output/tables/lp_irf_results_baseline.csv"
echo "  - 事件研究结果: output/tables/event_study_aar_*.csv"
echo "  - 主要图表: output/figures/lp_irf_baseline_*.png"
echo "  - 事件研究图表: output/figures/event_study_*.png"
echo "  - 比较分析图表: output/figures/lp_irf_vs_event_study_*.png"
echo ""
echo ""
echo "下一步建议:"
echo "  1. 查看分析报告: cat output/event_study_analysis_report.md"
echo "  2. 检查主要图表: ls output/figures/*.png"
echo "  3. 分析结果数据: head output/tables/event_study_aar_*.csv"
echo "  4. 如需重新运行: ./main.sh --quick (快速模式)"
echo "========================================"
