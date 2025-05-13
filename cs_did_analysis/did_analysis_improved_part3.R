#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

# 加载第二部分处理的数据
cat("加载第二部分处理的数据...\n")

# 确定项目根目录
if (dir.exists("../data")) {
  project_root <- ".."
} else {
  project_root <- "."
}
output_dir <- file.path(project_root, "output/did_analysis_improved")

load(file.path(output_dir, "part2_data.RData"))

# 重新加载必要的包
library(dplyr)
library(tidyr)
library(ggplot2)
library(data.table)

#-----------------------------------------------------------------------------
# 7. 计算波动率和其他分析变量
#-----------------------------------------------------------------------------
cat("计算波动率和其他分析变量...\n")

# 转换数值变量
panel_data <- margin_changes %>%
  dplyr::mutate(
    # 转换价格相关变量为数值型
    high_price = as.numeric(high_price),
    low_price = as.numeric(low_price),
    trading_volume = as.numeric(trading_volume),
    open_interest = as.numeric(open_interest),
    close_price = as.numeric(close_price),
    open_price = as.numeric(open_price)
  )

# 检查转换后是否有缺失值
missing_prices <- sum(is.na(panel_data$high_price) | is.na(panel_data$low_price))
if(missing_prices > 0) {
  cat(sprintf("警告：价格数据中有 %d 个缺失值或无法转换为数值的记录\n", missing_prices))
}

# 计算波动率和其他变量
panel_data <- panel_data %>%
  dplyr::mutate(
    # 计算Garman-Klass波动率
    volatility = ifelse(
      !is.na(high_price) & !is.na(low_price) & !is.na(close_price) & !is.na(open_price) &
      high_price > 0 & low_price > 0 & close_price > 0 & open_price > 0,
      sqrt(0.5 * (log(high_price/low_price))^2 - (2*log(2)-1) * (log(close_price/open_price))^2),
      NA
    ),
    
    # 对数波动率
    log_volatility = ifelse(!is.na(volatility) & volatility > 0, 
                           log(pmax(volatility, 1e-10)), 
                           NA),
    
    # 对数成交量
    log_volume = ifelse(!is.na(trading_volume) & trading_volume > 0, 
                       log(trading_volume), 
                       NA),
    
    # 对数持仓量
    log_open_interest = ifelse(!is.na(open_interest) & open_interest > 0, 
                              log(open_interest), 
                              NA)
  )

# 计算滞后变量
panel_data <- panel_data %>%
  dplyr::group_by(contract_id) %>%
  dplyr::arrange(date) %>%
  dplyr::mutate(
    # 收益率 = 今日收盘价/昨日收盘价的对数差
    return = c(NA, diff(log(close_price))),
    # 计算各变量的滞后值，用于控制变量
    log_volatility_lag1 = dplyr::lag(log_volatility),
    log_volume_lag1 = dplyr::lag(log_volume),
    log_open_interest_lag1 = dplyr::lag(log_open_interest),
    return_lag1 = dplyr::lag(return)
  ) %>%
  dplyr::ungroup()

#-----------------------------------------------------------------------------
# 8. 检查分析变量的统计特性
#-----------------------------------------------------------------------------
cat("检查分析变量的统计特性...\n")

# 计算变量统计摘要
var_summary <- panel_data %>%
  dplyr::summarise(
    volatility_mean = mean(volatility, na.rm = TRUE),
    volatility_sd = sd(volatility, na.rm = TRUE),
    log_volatility_mean = mean(log_volatility, na.rm = TRUE),
    log_volatility_sd = sd(log_volatility, na.rm = TRUE),
    return_mean = mean(return, na.rm = TRUE),
    return_sd = sd(return, na.rm = TRUE),
    volume_mean = mean(trading_volume, na.rm = TRUE),
    volume_sd = sd(trading_volume, na.rm = TRUE)
  )

cat("分析变量统计摘要:\n")
print(var_summary)

# 保存变量统计摘要
tables_dir <- file.path(output_dir, "tables")
write.csv(var_summary, file.path(tables_dir, "analysis_variables_summary.csv"), row.names = FALSE)

#-----------------------------------------------------------------------------
# 保存进度
#-----------------------------------------------------------------------------
cat("波动率和分析变量计算完成。部分3结束。\n")
# 保存数据以便后续分析
save(panel_data, project_root, output_dir, tables_dir, plots_dir,
     file = file.path(output_dir, "part3_data.RData"))
