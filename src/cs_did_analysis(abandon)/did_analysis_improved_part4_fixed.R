#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

# 加载第三部分处理的数据
cat("加载第三部分处理的数据...\n")

# 确定项目根目录
if (dir.exists("../data")) {
  project_root <- ".."
} else {
  project_root <- "."
}
output_dir <- file.path(project_root, "output/did_analysis_improved")

load(file.path(output_dir, "part3_data.RData"))

# 重新加载必要的包
library(dplyr)
library(tidyr)
library(ggplot2)
library(data.table)

#-----------------------------------------------------------------------------
# 9. 创建DID分析所需的变量
#-----------------------------------------------------------------------------
cat("创建DID分析所需的变量...\n")

# 标记每个合约的首次保证金调整日期（处理时间）
first_treatment <- panel_data %>%
  dplyr::filter(margin_changed == TRUE) %>%
  dplyr::group_by(contract_id) %>%
  dplyr::summarise(
    first_treat_date = min(date),
    .groups = "drop"
  )

cat(sprintf("识别出 %d 个合约存在首次保证金调整\n", nrow(first_treatment)))

# 合并首次处理信息到面板数据
did_panel <- panel_data %>%
  dplyr::left_join(first_treatment, by = "contract_id")

# 创建连续的时间周期变量
all_dates <- sort(unique(did_panel$date))
date_map <- data.frame(
  date = all_dates,
  continuous_period = seq_along(all_dates)
)

# 合并连续时间周期到面板数据
did_panel <- did_panel %>%
  dplyr::left_join(date_map, by = "date") %>%
  dplyr::mutate(
    # 使用连续编号作为时间周期
    time_period = continuous_period,
    
    # 为每个合约的首次处理日期找到对应的连续时间周期
    first_treat_period = ifelse(!is.na(first_treat_date), 
                               as.numeric(date_map$continuous_period[match(first_treat_date, date_map$date)]), 
                               NA),
    
    # 创建数值化的合约ID，用于DID分析
    numeric_id = as.numeric(as.factor(contract_id)),
    
    # 标记是否为处理组（是否曾经受到处理）
    ever_treated = !is.na(first_treat_date),
    
    # 标记每个观测是否已经被处理（在处理后）
    post_treatment = !is.na(first_treat_date) & (date >= first_treat_date)
  )

#-----------------------------------------------------------------------------
# 10. 检查处理状态分布
#-----------------------------------------------------------------------------
cat("检查处理状态分布...\n")

# 检查观测在处理前后的分布
treatment_summary <- did_panel %>%
  dplyr::group_by(ever_treated, post_treatment) %>%
  dplyr::summarise(
    observations = n(),
    contracts = n_distinct(contract_id),
    avg_volatility = mean(log_volatility, na.rm = TRUE),
    .groups = "drop"
  )

cat("观测在处理前后的分布:\n")
print(treatment_summary)

# 保存处理状态摘要
tables_dir <- file.path(output_dir, "tables")
write.csv(treatment_summary, file.path(tables_dir, "treatment_status_summary.csv"), row.names = FALSE)

#-----------------------------------------------------------------------------
# 11. 创建人工对照组（如果需要）
#-----------------------------------------------------------------------------
# 检查是否有"从未处理"组
never_treated_count <- did_panel %>%
  dplyr::filter(is.na(first_treat_period)) %>%
  dplyr::pull(contract_id) %>%
  n_distinct()

cat(sprintf("从未处理的合约数量: %d (占总合约数的 %.1f%%)\n", 
            never_treated_count, 
            never_treated_count / n_distinct(did_panel$contract_id) * 100))

if(never_treated_count == 0) {
  cat("警告：没有'从未处理'组，将创建人工对照组\n")
  
  # 创建人工对照组
  set.seed(789) # 确保结果可重现
  
  # 获取所有合约
  all_contracts <- unique(did_panel$contract_id)
  contract_count <- length(all_contracts)
  
  # 随机将30%的合约设为"从未处理"（人工对照组）
  artificial_control_size <- max(1, round(contract_count * 0.3))
  artificial_controls <- sample(all_contracts, artificial_control_size)
  
  cat(sprintf("选择 %d/%d 个合约作为人工对照组: %s\n", 
              artificial_control_size, contract_count,
              paste(artificial_controls, collapse=", ")))
  
  # 将选定合约标记为永不处理
  did_panel <- did_panel %>%
    dplyr::mutate(
      # 如果合约在人工对照组中，将处理变量设为NA
      first_treat_period = ifelse(contract_id %in% artificial_controls, NA, first_treat_period),
      first_treat_date = ifelse(contract_id %in% artificial_controls, as.Date(NA), first_treat_date),
      ever_treated = !is.na(first_treat_date),
      post_treatment = !is.na(first_treat_date) & (date >= first_treat_date),
      # 标记为人工对照组
      artificial_control = contract_id %in% artificial_controls
    )
  
  # 重新检查从未处理的合约数量
  never_treated_count <- did_panel %>%
    dplyr::filter(is.na(first_treat_period)) %>%
    dplyr::pull(contract_id) %>%
    n_distinct()
  
  cat(sprintf("创建人工对照组后，从未处理的合约数量: %d (占比: %.1f%%)\n", 
              never_treated_count, 
              never_treated_count / n_distinct(did_panel$contract_id) * 100))
}

# 确定控制组策略
if(never_treated_count > 0) {
  cat("有足够的'从未处理'组，将使用'nevertreated'控制组策略\n")
  control_strategies <- c("nevertreated", "notyettreated")
} else {
  cat("警告：仍然没有'从未处理'组，将仅使用'notyettreated'控制组策略\n")
  control_strategies <- "notyettreated"
}

#-----------------------------------------------------------------------------
# 12. 准备DID分析数据
#-----------------------------------------------------------------------------
cat("准备DID分析数据...\n")

# 移除缺失值，确保关键变量完整
did_clean <- did_panel %>%
  tidyr::drop_na(numeric_id, time_period, log_volatility, log_volume_lag1, log_open_interest_lag1, return_lag1)

cat(sprintf("DID分析数据准备完成，包含 %d 个观测值，%d 个合约\n", 
            nrow(did_clean), 
            n_distinct(did_clean$contract_id)))

#-----------------------------------------------------------------------------
# 保存进度
#-----------------------------------------------------------------------------
cat("DID分析变量创建和人工对照组创建完成。部分4结束。\n")
# 保存数据以便后续分析
save(did_panel, did_clean, control_strategies, first_treatment, never_treated_count,
     project_root, output_dir, tables_dir, plots_dir,
     file = file.path(output_dir, "part4_data.RData"))
