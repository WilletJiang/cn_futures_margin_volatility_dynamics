# 快速开始指南

## 🚀 一键运行完整分析

### 方法1：完整模式（推荐）
```bash
./main.sh
```
**包含所有分析步骤，预计运行时间：5-10分钟**

### 方法2：快速模式
```bash
./main.sh --quick
```
**跳过传统稳健性检验，预计运行时间：3-5分钟**

### 方法3：查看帮助
```bash
./main.sh --help
```

## 📋 分析流程概览

### 完整流程包含6个步骤：

1. **数据处理与特征构建** (30秒)
   - 加载原始数据
   - 构建波动率指标
   - 识别保证金调整事件

2. **描述性统计表格生成** (15秒)
   - 生成数据摘要统计
   - 创建变量描述表格

3. **LP-IRF分析** (1-2分钟)
   - 局部投影脉冲响应函数估计
   - 基准分析和状态依赖分析

4. **事件研究分析** (2-3分钟)
   - 事件识别和筛选
   - 异常收益率计算
   - 多种稳健性检验
   - 与LP-IRF结果比较

5. **可视化图表生成** (30秒)
   - 生成所有英文版图表
   - LP-IRF图表和事件研究图表

6. **传统稳健性检验** (1-2分钟，快速模式跳过)
   - LP-IRF方法的额外稳健性检验

## 📊 输出结果

### 主要输出文件位置：

#### 📈 图表文件 (`output/figures/`)
- **LP-IRF图表**:
  - `lp_irf_baseline_increase.png` - 保证金增加的基准IRF
  - `lp_irf_baseline_decrease.png` - 保证金减少的基准IRF
  - `lp_irf_statedep_*.png` - 状态依赖分析图表

- **事件研究图表**:
  - `event_study_aar_increase.png` - 保证金增加事件AAR图
  - `event_study_aar_decrease.png` - 保证金减少事件AAR图
  - `event_study_summary.png` - 事件研究结果汇总
  - `event_study_distribution_*.png` - 事件分布图

- **比较分析图表**:
  - `lp_irf_vs_event_study_increase.png` - 方法比较图(增加)
  - `lp_irf_vs_event_study_decrease.png` - 方法比较图(减少)

#### 📋 数据表格 (`output/tables/`)
- **LP-IRF结果**:
  - `lp_irf_results_baseline.csv` - 基准分析结果
  - `lp_irf_results_statedep.csv` - 状态依赖分析结果

- **事件研究结果**:
  - `event_study_aar_increase.csv` - 保证金增加事件AAR
  - `event_study_aar_decrease.csv` - 保证金减少事件AAR
  - `event_study_events_*.csv` - 识别的事件列表

- **比较分析结果**:
  - `lp_irf_vs_event_study_correlation_*.csv` - 相关性分析

#### 📄 分析报告
- `output/event_study_analysis_report.md` - 完整分析报告

## 🔍 关键结果解读

### LP-IRF结果
- **系数 (coeff)**: 保证金调整对波动率的影响大小
- **置信区间**: conf_low 到 conf_high
- **显著性**: p值 < 0.05 表示统计显著

### 事件研究结果
- **AAR**: 平均异常收益率，衡量事件的平均影响
- **CAAR**: 累积平均异常收益率，衡量累积影响
- **显著性**: significant列标记统计显著的时间点

### 比较分析
- **相关性**: Pearson和Spearman相关系数
- **一致性**: 两种方法结果的一致程度

## ⚡ 快速检查结果

### 1. 查看分析报告
```bash
cat output/event_study_analysis_report.md
```

### 2. 检查主要图表
```bash
ls output/figures/*.png | head -10
```

### 3. 查看关键数据
```bash
head output/tables/event_study_aar_increase.csv
head output/tables/lp_irf_results_baseline.csv
```

### 4. 统计生成文件数量
```bash
echo "表格文件数量: $(ls output/tables/*.csv | wc -l)"
echo "图表文件数量: $(ls output/figures/*.png | wc -l)"
```

## 🛠 故障排除

### 常见问题及解决方案：

#### 1. 权限问题
```bash
chmod +x main.sh
```

#### 2. Python模块缺失
```bash
pip install pandas numpy matplotlib seaborn scipy
```

#### 3. 数据文件不存在
确保原始数据文件在 `data/raw/` 目录下

#### 4. 内存不足
使用快速模式：
```bash
./main.sh --quick
```

#### 5. 查看详细错误信息
检查日志文件：
```bash
ls output/logs/
```

## 📚 进阶使用

### 单独运行特定分析：

#### 只运行LP-IRF分析
```bash
python src/analysis/lp_irf_analysis.py
python src/visualization/plot_lp_irf_results.py
```

#### 只运行事件研究
```bash
python src/analysis/run_event_study.py
```

#### 只运行比较分析
```bash
python src/analysis/comparative_analysis.py
```

### 自定义参数：
编辑 `src/config.py` 文件调整分析参数

## 🎯 下一步建议

1. **查看结果**: 先查看生成的图表和报告
2. **理解发现**: 分析LP-IRF和事件研究的一致性
3. **撰写论文**: 使用英文图表和统计结果
4. **进一步分析**: 根据需要调整参数重新运行

## 📞 技术支持

如遇问题：
1. 查看 `output/logs/` 中的日志文件
2. 运行测试脚本：`python simple_test.py`
3. 查看详细文档：`docs/event_study_methodology.md`

---

**🎉 恭喜！您现在可以开始使用这个强大的期货保证金波动率动态分析框架了！**
