#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

# 加载第一部分处理的数据
cat("加载第一部分处理的数据...\n")

# 确定项目根目录
if (dir.exists("../data")) {
  project_root <- ".."
} else {
  project_root <- "."
}
output_dir <- file.path(project_root, "output/did_analysis_improved")

load(file.path(output_dir, "part1_data.RData"))

# 重新加载必要的包
library(dplyr)
library(tidyr)
library(ggplot2)
library(data.table)
library(zoo)

#-----------------------------------------------------------------------------
# 5. 处理缺失的保证金率
#-----------------------------------------------------------------------------
cat("填充缺失的保证金率数据...\n")
processed_data_filled <- processed_data %>%
  dplyr::group_by(contract_id) %>%
  dplyr::arrange(date) %>%
  # 尝试使用前期值填充缺失的保证金率
  dplyr::mutate(
    margin_rate_filled = zoo::na.locf(margin_rate, na.rm = FALSE),
    # 对于开头就是NA的情况，使用后向填充
    margin_rate_filled = zoo::na.locf(margin_rate_filled, fromLast = TRUE, na.rm = FALSE)
  ) %>%
  dplyr::ungroup()

# 检查填充效果
margin_filled_check <- sum(is.na(processed_data_filled$margin_rate_filled))
cat(sprintf("填充后保证金率仍然缺失的观测数: %d (%.1f%%)\n", 
            margin_filled_check,
            margin_filled_check / nrow(processed_data_filled) * 100))

#-----------------------------------------------------------------------------
# 6. 识别保证金调整事件
#-----------------------------------------------------------------------------
cat("识别保证金调整事件...\n")

# 按合约分组，识别保证金率变化
margin_changes <- processed_data_filled %>%
  dplyr::group_by(contract_id) %>%
  dplyr::arrange(date) %>%
  dplyr::mutate(
    # 标识保证金率变化
    margin_rate_lag = dplyr::lag(margin_rate_filled),
    margin_changed = !is.na(margin_rate_filled) & 
                     !is.na(margin_rate_lag) & 
                     abs(margin_rate_filled - margin_rate_lag) > 0.01, # 最小变化阈值为0.01
    # 第一条记录设为FALSE
    margin_changed = ifelse(is.na(margin_changed), FALSE, margin_changed)
  ) %>%
  dplyr::ungroup()

# 汇总保证金调整情况
margin_summary <- margin_changes %>%
  dplyr::filter(margin_changed == TRUE) %>%
  dplyr::group_by(contract_id) %>%
  dplyr::summarise(
    total_changes = n(),
    first_change_date = min(date, na.rm = TRUE),
    min_margin = min(margin_rate_filled, na.rm = TRUE),
    max_margin = max(margin_rate_filled, na.rm = TRUE),
    .groups = "drop"
  )

# 检查保证金调整事件数量
cat(sprintf("检测到 %d 个保证金调整事件，涉及 %d 个不同合约\n", 
            sum(margin_changes$margin_changed, na.rm = TRUE),
            nrow(margin_summary)))

# 保存保证金调整摘要
tables_dir <- file.path(output_dir, "tables")
write.csv(margin_summary, file.path(tables_dir, "margin_changes_summary.csv"), row.names = FALSE)

#-----------------------------------------------------------------------------
# 保存进度
#-----------------------------------------------------------------------------
cat("保证金调整事件识别完成。部分2结束。\n")
# 保存数据以便后续分析
save(margin_changes, margin_summary, project_root, output_dir, tables_dir, plots_dir,
     file = file.path(output_dir, "part2_data.RData"))
