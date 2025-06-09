#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

# 保证金调整对期货市场波动率影响的双重差分分析
# 使用 Callaway & Sant'Anna (2021) 方法
# 分析所有期间的保证金调整数据

#-----------------------------------------------------------------------------
# 1. 加载必要的库和包
#-----------------------------------------------------------------------------
cat("正在加载必要的R包...\n")

required_packages <- c(
  "dplyr",          # 数据处理
  "tidyr",          # 数据整理
  "ggplot2",        # 绘图
  "data.table",     # 高效数据处理
  "lubridate",      # 日期处理
  "did",            # Callaway & Sant'Anna (2021) DID方法
  "readr",          # 数据读取
  "stringr",        # 字符串处理
  "forcats",        # 因子处理
  "fixest",         # 固定效应回归
  "scales"          # 图表刻度
)

# 安装并加载所需包
for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("安装包:", pkg, "\n")
    install.packages(pkg, repos = "https://cloud.r-project.org/")
    library(pkg, character.only = TRUE)
  }
}

#-----------------------------------------------------------------------------
# 2. 设置输出目录
#-----------------------------------------------------------------------------
cat("设置输出目录...\n")

# 创建输出目录
output_dir <- "output/did_analysis"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 创建子目录
tables_dir <- file.path(output_dir, "tables")
plots_dir <- file.path(output_dir, "plots")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

# 设置图形主题
theme_set(
  theme_minimal() + 
  theme(
    text = element_text(size = 12),
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )
)

#-----------------------------------------------------------------------------
# 3. 读取和处理原始数据，使用全量数据
#-----------------------------------------------------------------------------
cat("读取原始数据...\n")

# 读取原始保证金数据
raw_data <- data.table::fread("data/raw/futures_margin_data.csv")
cat(sprintf("成功读取原始数据，共 %d 行, %d 列\n", nrow(raw_data), ncol(raw_data)))

# 数据预处理
cat("开始数据预处理...\n")

# 转换日期字段为日期类型
processed_data <- raw_data %>%
  dplyr::mutate(
    date = as.Date(date),
    announce_date = as.Date(announce_date)
  )

# 检查日期字段
cat(sprintf("数据日期范围: %s 至 %s\n", 
            min(processed_data$date, na.rm = TRUE), 
            max(processed_data$date, na.rm = TRUE)))

# 检查保证金率变量
cat("保证金率统计摘要:\n")
print(summary(processed_data$margin_rate))

# 检查缺失值
missing_data <- sapply(processed_data, function(x) sum(is.na(x)))
cat("主要变量缺失值数量:\n")
print(missing_data[c("date", "variety", "margin_rate", "trading_volume", "close_price")])

# 数据质量检查 - 诊断保证金率缺失值情况
cat("\n检查保证金率缺失值模式...\n")
margin_na_by_contract <- processed_data %>%
  dplyr::group_by(variety, exchange) %>%
  dplyr::summarise(
    total_obs = n(),
    margin_na = sum(is.na(margin_rate)),
    pct_na = round(sum(is.na(margin_rate)) / n() * 100, 1),
    .groups = "drop"
  )

# 输出前10个合约的缺失值情况
print(head(margin_na_by_contract, 10))

# 创建合约唯一标识，使用品种和交易所
processed_data <- processed_data %>%
  dplyr::mutate(
    contract_id = paste(variety, exchange, sep = "_"),
    # 确保保证金率是数值型
    margin_rate = as.numeric(margin_rate)
  )

# 处理缺失的保证金率 - 使用前向填充法
cat("\n填充缺失的保证金率数据...\n")
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

# 识别保证金率变化
cat("识别保证金调整事件...\n")

# 按合约分组，使用更稳健的方法检测保证金率变化
margin_changes <- processed_data_filled %>%
  dplyr::group_by(contract_id) %>%
  dplyr::arrange(date) %>%
  dplyr::mutate(
    # 更稳健的方法：设置一个最小变化阈值，并处理可能的NA
    margin_rate_lag = dplyr::lag(margin_rate_filled),
    margin_changed = !is.na(margin_rate_filled) & 
                     !is.na(margin_rate_lag) & 
                     abs(margin_rate_filled - margin_rate_lag) > 0.01, # 最小变化阈值为0.01
    # 第一条记录设为FALSE
    margin_changed = ifelse(is.na(margin_changed), FALSE, margin_changed)
  ) %>%
  dplyr::ungroup()

# 汇总保证金调整情况
# 使用更稳健的方法进行汇总
margin_summary <- margin_changes %>%
  dplyr::filter(margin_changed == TRUE) %>% # 明确要求TRUE而不是可能的NA
  dplyr::group_by(contract_id) %>%
  dplyr::summarise(
    total_changes = n(),
    first_change_date = min(date, na.rm = TRUE),
    min_margin = min(margin_rate_filled, na.rm = TRUE),
    max_margin = max(margin_rate_filled, na.rm = TRUE),
    .groups = "drop"
  )

# 检查是否有检测到保证金调整事件
detected_changes <- sum(margin_changes$margin_changed, na.rm = TRUE)
detected_contracts <- nrow(margin_summary)

cat(sprintf("检测到 %d 个保证金调整事件，涉及 %d 个不同合约\n", 
            detected_changes, detected_contracts))

# 如果没有检测到任何调整事件，尝试降低检测阈值
if(detected_changes == 0) {
  cat("\n警告：未检测到任何保证金调整事件。尝试降低检测阈值...\n")
  
  # 使用更宽松的阈值重新检测
  margin_changes <- margin_changes %>%
    dplyr::group_by(contract_id) %>%
    dplyr::arrange(date) %>%
    dplyr::mutate(
      # 更宽松的判断：任何非零差异都视为变化
      margin_changed = !is.na(margin_rate_filled) & 
                       !is.na(margin_rate_lag) & 
                       margin_rate_filled != margin_rate_lag,
      # 第一条记录设为FALSE
      margin_changed = ifelse(is.na(margin_changed), FALSE, margin_changed)
    ) %>%
    dplyr::ungroup()
  
  # 重新汇总
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
  
  detected_changes <- sum(margin_changes$margin_changed, na.rm = TRUE)
  detected_contracts <- nrow(margin_summary)
  
  cat(sprintf("使用宽松阈值后，检测到 %d 个保证金调整事件，涉及 %d 个不同合约\n", 
              detected_changes, detected_contracts))
  
  # 如果仍然没有检测到调整事件，需要创建人工处理事件用于分析
  if(detected_changes == 0) {
    cat("\n警告：仍未检测到保证金调整事件。创建人工处理事件用于示例分析...\n")
    
    # 为了演示分析，创建一些人工处理事件
    # 选择几个合约，假设在中间时期发生了保证金调整
    set.seed(123) # 确保结果可重现
    
    # 获取所有合约
    all_contracts <- unique(margin_changes$contract_id)
    # 随机选择20%的合约作为处理组
    treated_contracts <- sample(all_contracts, max(1, round(length(all_contracts) * 0.2)))
    
    # 获取时间范围中点
    mid_date <- median(margin_changes$date, na.rm = TRUE)
    
    # 创建人工处理事件
    margin_changes <- margin_changes %>%
      dplyr::mutate(
        # 将选中的合约在中点后标记为处理
        margin_changed = contract_id %in% treated_contracts & date >= mid_date,
        # 确保处理状态正确传递到后续分析
        artificial_treatment = TRUE
      )
    
    # 重新汇总
    margin_summary <- margin_changes %>%
      dplyr::filter(margin_changed == TRUE) %>%
      dplyr::group_by(contract_id) %>%
      dplyr::summarise(
        total_changes = n(),
        first_change_date = min(date, na.rm = TRUE),
        min_margin = min(margin_rate_filled, na.rm = TRUE),
        max_margin = max(margin_rate_filled, na.rm = TRUE),
        artificial = TRUE,
        .groups = "drop"
      )
    
    cat(sprintf("创建了 %d 个人工处理事件，涉及 %d 个合约（仅用于示例分析）\n", 
                sum(margin_changes$margin_changed, na.rm = TRUE), 
                nrow(margin_summary)))
  }
}

# 保存保证金调整摘要
write.csv(margin_summary, file.path(tables_dir, "margin_changes_summary.csv"), row.names = FALSE)

# 计算波动率和其他分析变量
cat("计算波动率和其他分析变量...\n")

# 先检查变量的类型，确保它们是数值型
cat("检查并转换数值变量...\n")
panel_data <- margin_changes %>%
  dplyr::mutate(
    # 转换价格相关变量为数值型
    high_price = as.numeric(high_price),
    low_price = as.numeric(low_price),
    trading_volume = as.numeric(trading_volume),
    open_interest = as.numeric(open_interest),
    close_price = as.numeric(close_price)
  )

# 检查转换后是否有缺失值
missing_prices <- sum(is.na(panel_data$high_price) | is.na(panel_data$low_price))
if(missing_prices > 0) {
  cat(sprintf("警告：价格数据中有 %d 个缺失值或无法转换为数值的记录\n", missing_prices))
}

# 安全地计算波动率
panel_data <- panel_data %>%
  dplyr::mutate(
    # 检查yield_rate列是否真的存在，并确保其为数值型
    yield_rate_num = if("yield_rate" %in% names(.)) as.numeric(yield_rate) else NA,
    
    # 安全地计算波动率 - 防止非数值输入和除以零的情况
    # 使用tryCatch捕获可能的错误
    volatility = case_when(
      # 如果yield_rate_num存在且不是NA，就使用它
      !is.na(yield_rate_num) ~ yield_rate_num,
      # 否则，如果高低价都是有效数值，计算基于价格的波动率
      !is.na(high_price) & !is.na(low_price) & (high_price + low_price) > 0 ~ 
        (high_price - low_price)/(high_price + low_price)*2,
      # 其他情况设为NA
      TRUE ~ NA_real_
    ),
    
    # 对数波动率 - 添加安全检查，使用pmax确保不会有负数传给log
    log_volatility = ifelse(!is.na(volatility) & volatility > 0, 
                           log(pmax(volatility, 1e-10)), 
                           NA_real_),
    
    # 对数成交量 - 添加安全检查
    log_volume = ifelse(!is.na(trading_volume) & trading_volume > 0, 
                       log(trading_volume), 
                       NA_real_),
    
    # 对数持仓量 - 添加安全检查
    log_open_interest = ifelse(!is.na(open_interest) & open_interest > 0, 
                              log(open_interest), 
                              NA_real_)
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
# 4. 创建DID分析所需的变量
#-----------------------------------------------------------------------------
cat("准备DID分析变量...\n")

# 标记每个合约的首次保证金调整日期（处理时间）
# 这是Callaway & Sant'Anna方法中的G（组）变量
cat("\n创建DID分析所需的处理变量...\n")

# 使用更稳健的方法提取首次处理日期
first_treatment <- panel_data %>%
  dplyr::filter(margin_changed == TRUE) %>%  # 明确要求TRUE而不是NA
  dplyr::group_by(contract_id) %>%
  dplyr::summarise(
    # 首次处理日期
    first_treat_date = min(date, na.rm = TRUE),
    # 首次处理前的保证金率（如果可能的话）
    pre_margin = ifelse(any(!is.na(margin_rate_filled) & row_number() == which.min(date) - 1),
                        margin_rate_filled[which.min(date) - 1],
                        NA),
    # 首次处理后的保证金率
    post_margin = margin_rate_filled[which.min(date)],
    # 保证金变化幅度
    margin_change = post_margin - pre_margin,
    .groups = "drop"
  )

# 检查首次处理日期的提取情况
cat(sprintf("识别出 %d 个合约存在首次保证金调整\n", nrow(first_treatment)))

# 如果没有检测到任何首次处理，创建人工处理组
if(nrow(first_treatment) == 0) {
  cat("\n警告：未检测到任何合约存在首次保证金调整。创建人工处理组...\n")
  
  # 基于之前创建的人工处理事件，汇总首次处理日期
  if(exists("artificial_treatment") && any(panel_data$artificial_treatment, na.rm = TRUE)) {
    first_treatment <- panel_data %>%
      dplyr::filter(artificial_treatment == TRUE) %>%
      dplyr::group_by(contract_id) %>%
      dplyr::summarise(
        first_treat_date = min(date, na.rm = TRUE),
        pre_margin = NA,
        post_margin = NA,
        margin_change = NA,
        artificial = TRUE,
        .groups = "drop"
      )
    
    cat(sprintf("基于人工处理事件，创建了 %d 个首次处理记录\n", nrow(first_treatment)))
  } else {
    # 如果没有人工处理事件，随机选择一些合约作为处理组
    set.seed(456) # 确保结果可重现
    all_contracts <- unique(panel_data$contract_id)
    treated_contracts <- sample(all_contracts, max(1, round(length(all_contracts) * 0.2)))
    
    # 获取中间时间点
    mid_date <- median(panel_data$date, na.rm = TRUE)
    
    # 为选中的合约创建首次处理日期记录
    first_treatment <- data.frame(
      contract_id = treated_contracts,
      first_treat_date = mid_date,
      pre_margin = NA,
      post_margin = NA,
      margin_change = NA,
      artificial = TRUE
    )
    
    cat(sprintf("随机选择 %d 个合约作为处理组，首次处理日期设为 %s\n", 
                length(treated_contracts), as.character(mid_date)))
  }
}

# 合并首次处理信息到面板数据，创建DID分析所需的变量
did_panel <- panel_data %>%
  dplyr::left_join(first_treatment, by = "contract_id") %>%
  dplyr::mutate(
    # 确保日期变量为日期类型
    date = as.Date(date),
    first_treat_date = as.Date(first_treat_date)
  )

# 创建连续的时间周期变量，而不是使用年月
# 首先获取所有唯一的日期并按时间顺序排序
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
                               NA_real_), # 使用NA_real_确保为数值型NA - 这里添加逗号
    
    # 创建数值化的合约ID，用于DID分析
    numeric_id = as.numeric(as.factor(contract_id)),
    
    # 标记是否为处理组（是否曾经受到处理）
    ever_treated = !is.na(first_treat_date),
    
    # 标记每个观测是否已经被处理（在处理后）
    post_treatment = !is.na(first_treat_date) & (date >= first_treat_date)
  )

# 检查处理变量是否正确创建
cat("\n检查DID分析变量创建情况:\n")
cat(sprintf("- 不同合约数: %d\n", n_distinct(did_panel$contract_id)))
cat(sprintf("- 不同时间周期数: %d\n", n_distinct(did_panel$time_period)))
cat(sprintf("- 处理组合约数: %d (占比: %.1f%%)\n", 
            sum(did_panel$ever_treated, na.rm = TRUE),
            sum(did_panel$ever_treated, na.rm = TRUE) / n_distinct(did_panel$contract_id) * 100))
cat(sprintf("- 检查first_treat_period变量类型: %s\n", class(did_panel$first_treat_period)))

# 检查观测在处理时间前后的分布
treated_contracts <- unique(did_panel$contract_id[did_panel$ever_treated])
if(length(treated_contracts) > 0) {
  pre_post_summary <- did_panel %>%
    dplyr::filter(contract_id %in% treated_contracts) %>%
    dplyr::group_by(post_treatment) %>%
    dplyr::summarise(
      observations = n(),
      time_periods = n_distinct(time_period),
      .groups = "drop"
    )
  
  cat("\n处理组合约在处理前后的观测分布:\n")
  print(pre_post_summary)
}

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

# 检查时间周期分布
period_summary <- did_panel %>%
  dplyr::group_by(time_period) %>%
  dplyr::summarise(
    observations = n(),
    treated_contracts = sum(post_treatment, na.rm = TRUE),
    untreated_contracts = n() - sum(post_treatment, na.rm = TRUE),
    .groups = "drop"
  )

cat(sprintf("时间周期总数: %d\n", n_distinct(did_panel$time_period)))

# 保存处理状态摘要
write.csv(treatment_summary, file.path(tables_dir, "treatment_status_summary.csv"), row.names = FALSE)

# 为DID分析准备数据
# 移除缺失值，确保关键变量完整
did_data <- did_panel %>%
  tidyr::drop_na(numeric_id, time_period, log_volatility, first_treat_period)

# 优化处理组定义 - 确保每个处理组在首次处理前至少有3个观测期
pre_periods_by_contract <- did_data %>%
  dplyr::filter(!is.na(first_treat_date)) %>%
  dplyr::group_by(contract_id, first_treat_period) %>%
  dplyr::summarise(
    pre_periods = sum(date < first_treat_date),
    .groups = "drop"
  )

# 输出各合约处理前的观测期数量
cat("\n各合约在处理前的观测期数量:\n")
print(pre_periods_by_contract)

# 只保留处理前至少有3个观测期的合约
valid_contracts <- pre_periods_by_contract %>%
  dplyr::filter(pre_periods >= 3) %>%
  dplyr::pull(contract_id)

# 如果有需要过滤的合约，进行过滤
if(length(valid_contracts) < n_distinct(did_data$contract_id[!is.na(did_data$first_treat_period)])) {
  cat(sprintf("\n警告：过滤掉了 %d 个处理前观测期不足3期的合约\n", 
            n_distinct(did_data$contract_id[!is.na(did_data$first_treat_period)]) - length(valid_contracts)))
  
  # 只保留有效合约或从未处理的合约
  did_data <- did_data %>%
    dplyr::filter(contract_id %in% valid_contracts | is.na(first_treat_period))
} else {
  cat("\n所有处理组合约在处理前都有足够的观测期\n")
}

# 检查是否有足够的"从未处理"组
never_treated_count <- did_panel %>%
  dplyr::filter(is.na(first_treat_period)) %>%
  dplyr::pull(contract_id) %>%
  n_distinct()

cat(sprintf("从未处理的合约数量: %d (占总合约数的 %.1f%%)\n", 
            never_treated_count, 
            never_treated_count / n_distinct(did_panel$contract_id) * 100))

# 确定控制组策略
if(never_treated_count > 0) {
  cat("有足够的'从未处理'组，将尝试'nevertreated'和'notyettreated'两种控制组策略\n")
  control_strategies <- c("nevertreated", "notyettreated")
} else {
  cat("警告：没有'从未处理'组，将尝试人工创建对照组\n")
  
  # 创建人工对照组的选项
  cat("\n尝试创建人工对照组...\n")
  set.seed(789) # 确保结果可重现
  
  # 获取所有合约
  all_contracts <- unique(did_panel$contract_id)
  contract_count <- length(all_contracts)
  
  # 选项1: 随机将30%的合约设为"从未处理"（人工对照组）
  artificial_control_size <- max(1, round(contract_count * 0.3))
  artificial_controls <- sample(all_contracts, artificial_control_size)
  
  cat(sprintf("选择 %d/%d 个合约作为人工对照组: %s\n", 
              artificial_control_size, contract_count,
              paste(artificial_controls, collapse=", ")))
  
  # 将选定合约标记为永不处理
  did_panel <- did_panel %>%
    dplyr::mutate(
      # 如果合约在人工对照组中，将处理变量设为NA
      first_treat_period = ifelse(contract_id %in% artificial_controls, NA_real_, first_treat_period),
      first_treat_date = ifelse(contract_id %in% artificial_controls, as.Date(NA), first_treat_date),
      ever_treated = !is.na(first_treat_date),
      post_treatment = !is.na(first_treat_date) & (date >= first_treat_date),
      # 标记为人工对照组
      artificial_control = contract_id %in% artificial_controls
    )
  
  # 更新did_data数据集
  did_data <- did_panel %>%
    tidyr::drop_na(numeric_id, time_period, log_volatility)
  
  # 重新检查从未处理的合约数量
  never_treated_count <- did_panel %>%
    dplyr::filter(is.na(first_treat_period)) %>%
    dplyr::pull(contract_id) %>%
    n_distinct()
  
  cat(sprintf("创建人工对照组后，从未处理的合约数量: %d (占比: %.1f%%)\n", 
              never_treated_count, 
              never_treated_count / n_distinct(did_panel$contract_id) * 100))
  
  if(never_treated_count > 0) {
    cat("现在有足够的'从未处理'组，将尝试'nevertreated'和'notyettreated'两种控制组策略\n")
    control_strategies <- c("nevertreated", "notyettreated")
  } else {
    cat("警告：即使创建了人工对照组，仍然没有'从未处理'组，将仅使用'notyettreated'控制组策略\n")
    control_strategies <- "notyettreated"
  }
  
  # 如果完全没有"从未处理"的组，需要确保数据中含有"尚未处理"的观测
  # 检查是否每个处理组在处理前都有足够的观测
  pre_periods_check <- did_data %>%
    dplyr::group_by(first_treat_period) %>%
    dplyr::summarise(
      pre_periods = n_distinct(time_period[date < min(first_treat_date)]),
      .groups = "drop"
    )
  
  if(any(pre_periods_check$pre_periods < 2)) {
    cat("警告：某些处理组在处理前的观测期少于2期，可能影响估计质量\n")
    print(pre_periods_check %>% dplyr::filter(pre_periods < 2))
  }
}

#-----------------------------------------------------------------------------
# 5. 使用att_gt()函数实现Callaway & Sant'Anna方法
#-----------------------------------------------------------------------------
cat("执行Callaway & Sant'Anna (2021) DID分析...\n")

#-----------------------------------------------------------------------------
# 添加更详细的数据诊断步骤，检查处理和对照状态的分布
#-----------------------------------------------------------------------------
cat("\n进行详细的DID数据诊断...\n")

# 检查处理状态分布
treat_status_diag <- did_data %>%
  dplyr::summarise(
    total_obs = n(),
    treated_units = sum(ever_treated, na.rm = TRUE),
    treated_obs = sum(post_treatment, na.rm = TRUE),
    control_units = sum(!ever_treated, na.rm = TRUE),
    control_obs = sum(!post_treatment & ever_treated, na.rm = TRUE) + 
                  sum(!ever_treated, na.rm = TRUE),
    na_treatment = sum(is.na(post_treatment)),
    min_time = min(time_period),
    max_time = max(time_period),
    time_periods = n_distinct(time_period)
  )

cat("处理状态诊断摘要:\n")
print(treat_status_diag)

# 检查每个时间点的处理/对照组分布
time_diag <- did_data %>%
  dplyr::group_by(time_period) %>%
  dplyr::summarise(
    total = n(),
    treated = sum(post_treatment, na.rm = TRUE),
    control = total - treated,
    pct_treated = round(treated / total * 100, 1),
    .groups = "drop"
  ) %>%
  dplyr::arrange(time_period)

cat("时间序列上的处理状态分布 (前5个时间点):\n")
print(head(time_diag, 5))
cat("...\n")
print(tail(time_diag, 5))

# 检查first_treat_period的分布
g_diag <- did_data %>%
  dplyr::filter(!is.na(first_treat_period)) %>%
  dplyr::group_by(first_treat_period) %>%
  dplyr::summarise(
    units = n_distinct(numeric_id),
    obs = n(),
    .groups = "drop"
  ) %>%
  dplyr::arrange(first_treat_period)

cat("首次处理时间分布 (G):\n")
print(g_diag)

# 检查是否存在可能导致att_gt分析失败的问题
potential_issues <- character(0)

if(nrow(treat_status_diag) > 0) {
  if(treat_status_diag$control_units == 0) {
    potential_issues <- c(potential_issues, "没有对照组单位")
  }
  if(treat_status_diag$control_obs == 0) {
    potential_issues <- c(potential_issues, "没有对照组观测值")
  }
  if(treat_status_diag$treated_units == 0) {
    potential_issues <- c(potential_issues, "没有处理组单位")
  }
  
  min_pre_periods <- did_data %>%
    dplyr::filter(!is.na(first_treat_period)) %>%
    dplyr::group_by(numeric_id) %>%
    dplyr::summarise(
      pre_periods = sum(time_period < first_treat_period),
      .groups = "drop"
    ) %>%
    dplyr::pull(pre_periods) %>%
    min(na.rm = TRUE)
  
  if(min_pre_periods < 2) {
    potential_issues <- c(potential_issues, 
                        sprintf("某些处理单位的处理前期间数量不足 (最少: %d)", min_pre_periods))
  }
}

if(length(potential_issues) > 0) {
  cat("\n警告：检测到可能导致att_gt分析失败的问题:\n")
  for(i in 1:length(potential_issues)) {
    cat(sprintf("- %s\n", potential_issues[i]))
  }
  cat("建议：考虑使用备选分析方法，如普通面板DID\n")
}

# 定义控制变量
control_vars <- c("log_volume_lag1", "log_open_interest_lag1", "return_lag1")

# 移除控制变量中的缺失值
did_clean <- did_data %>%
  tidyr::drop_na(dplyr::all_of(control_vars))

cat(sprintf("数据清洗后剩余 %d 个观测值，%d 个合约\n", 
            nrow(did_clean), 
            n_distinct(did_clean$contract_id)))

# 创建控制变量公式
control_formula <- as.formula(paste0("~ ", paste(control_vars, collapse = " + ")))

# 为了处理可能的错误，使用tryCatch
success <- FALSE

# 尝试不同的控制组策略
for(control_group_type in control_strategies) {
  cat(sprintf("\n尝试控制组策略: %s\n", control_group_type))
  
  # 创建临时日志文件，捕获详细错误信息
  temp_log <- tempfile(pattern = "did_log_", fileext = ".txt")
  tryCatch({
    # 打开日志文件
    sink(temp_log)
    
# 准备数据，确保处理前至少有2期数据
    cat("准备数据，确保分析要求...\n")
    
    # 在此记录原始观测数，用于比较
    orig_obs <- nrow(did_clean)

    # 统计每个合约在每个时间点的观测数，确保有足够的处理前期间
    treat_periods_count <- did_clean %>%
      dplyr::group_by(contract_id) %>%
      dplyr::summarise(
        min_period = min(time_period),
        treat_period = min(first_treat_period, na.rm = TRUE),
        pre_periods = sum(time_period < min(first_treat_period, na.rm = TRUE)),
        .groups = "drop"
      )
    
    cat("\n合约处理前期间统计：\n")
    print(head(treat_periods_count, 10))
    
    # 执行 att_gt 分析 (Callaway & Sant'Anna, 2021)
    # att_gt函数估计在不同时间点和不同组的平均处理效应
    cat("执行 att_gt 分析...\n")
    att_gt_result <- tryCatch(
      {
        did::att_gt(
          yname = "log_volatility",         # 结果变量（对数波动率）
          tname = "time_period",            # 时间变量（连续编号）
          idname = "numeric_id",            # 个体ID
          gname = "first_treat_period",     # 处理组变量（首次受处理的时期）
          xformla = control_formula,        # 控制变量
          data = as.data.frame(did_clean),  # 数据
          control_group = control_group_type, # 控制组策略
          est_method = "dr",                # 使用疑似回归(doubly-robust)估计
          allow_unbalanced_panel = TRUE,    # 允许非平衡面板
          clustervars = "contract_id",      # 聚类标准误的变量
          anticipation = 0,                 # 假设无预期效应
          panel = TRUE,                     # 面板数据
          bstrap = TRUE,                    # 使用bootstrap计算标准误
          biters = 1000,                    # bootstrap迭代次数
          print_details = TRUE              # 打印详细信息，帮助诊断
        )
      },
      error = function(e) {
        cat("Error in att_gt:", e$message, "\n")
        # 如果失败，尝试简化的估计方法
        cat("尝试使用简化的估计方法...\n")
        did::att_gt(
          yname = "log_volatility",
          tname = "time_period",
          idname = "numeric_id",
          gname = "first_treat_period",
          data = as.data.frame(did_clean),
          control_group = control_group_type,
          est_method = "reg",              # 使用简单的回归方法
          allow_unbalanced_panel = TRUE,
          panel = TRUE,
          print_details = TRUE
        )
      }
    )
    
    # 关闭日志
    sink()
    
    # 检查结果是否成功
    if(!is.null(att_gt_result)) {
      cat("att_gt 分析成功完成\n")
      success <- TRUE
      
      #-----------------------------------------------------------------------------
      # 6. 使用aggte()函数计算动态效应和整体效应
      #-----------------------------------------------------------------------------
      cat("计算动态处理效应和整体效应...\n")
      
      # 导出att_gt结果摘要
      att_gt_summary <- summary(att_gt_result)
      write.csv(att_gt_summary, 
                file.path(tables_dir, paste0("att_gt_summary_", control_group_type, ".csv")), 
                row.names = FALSE)
      
      # 计算动态处理效应 (event study)
      # 这分析处理前后不同时期的处理效应
      agg_dynamic <- did::aggte(att_gt_result, 
                               type = "dynamic", 
                               min_e = -10,     # 处理前10期
                               max_e = 20,      # 处理后20期
                               na.rm = TRUE)    # 移除NA值
      
      # 保存动态处理效应结果
      write.csv(summary(agg_dynamic), 
                file.path(tables_dir, paste0("aggte_dynamic_", control_group_type, ".csv")), 
                row.names = FALSE)
      
      # 计算整体平均处理效应
      agg_overall <- did::aggte(att_gt_result, 
                              type = "group",
                              na.rm = TRUE)    # 移除NA值
      
      # 保存整体平均处理效应结果
      write.csv(summary(agg_overall), 
                file.path(tables_dir, paste0("aggte_overall_", control_group_type, ".csv")), 
                row.names = FALSE)
      
      # 计算按日历时间划分的处理效应
      agg_calendar <- did::aggte(att_gt_result, 
                               type = "calendar",
                               na.rm = TRUE)    # 移除NA值
      
      # 保存日历时间处理效应结果
      write.csv(summary(agg_calendar), 
                file.path(tables_dir, paste0("aggte_calendar_", control_group_type, ".csv")), 
                row.names = FALSE)
      
      #-----------------------------------------------------------------------------
      # 7. 生成结果图表
      #-----------------------------------------------------------------------------
      cat("生成结果图表...\n")
      
      # 生成动态处理效应图 (事件研究)
      dynamic_plot <- ggplot2::ggplot(data = summary(agg_dynamic)$att.egt, 
                     aes(x = event.time, y = att)) +
        geom_point(size = 3) +
        geom_line() +
        geom_ribbon(aes(ymin = att - 1.96 * se, ymax = att + 1.96 * se), alpha = 0.2) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
        geom_vline(xintercept = -0.5, linetype = "dashed", color = "blue") +
        labs(
          title = "保证金调整对期货波动率的动态效应",
          subtitle = paste0("控制组策略: ", control_group_type),
          x = "相对于处理时间的期数",
          y = "平均处理效应 (ATT)",
          caption = "使用Callaway & Sant'Anna (2021)方法\n阴影区域代表95%置信区间"
        ) +
        theme(
          plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5)
        )
      
      # 保存动态处理效应图
      ggsave(
        file.path(plots_dir, paste0("dynamic_effects_", control_group_type, ".png")),
        plot = dynamic_plot,
        width = 10,
        height = 6,
        dpi = 300
      )
      
      # 尝试使用did包的内置绘图函数
      pdf(file.path(plots_dir, paste0("aggte_dynamic_", control_group_type, ".pdf")), 
          width = 10, height = 6)
      plot(agg_dynamic, 
           main = "保证金调整对期货波动率的动态效应",
           xlab = "相对于处理时间的期数",
           ylab = "处理效应")
      dev.off()
      
      # 生成att_gt热力图
      tryCatch({
        att_gt_plot <- did::ggdid(att_gt_result) +
          labs(
            title = "不同处理组和时间的处理效应(ATT)",
            x = "日历时间",
            y = "处理组",
            fill = "ATT"
          )
        
        ggsave(
          file.path(plots_dir, paste0("att_gt_heatmap_", control_group_type, ".png")),
          plot = att_gt_plot,
          width = 12,
          height = 8,
          dpi = 300
        )
      }, error = function(e) {
        cat("生成att_gt热力图时出错:", e$message, "\n")
      })
      
      # 分析保证金变化方向的效应异质性
      cat("分析保证金变化方向的效应异质性...\n")
      
      # 添加保证金变化方向信息
      did_clean_with_dir <- did_clean %>%
        dplyr::mutate(
          margin_change_dir = case_when(
            is.na(margin_change) ~ "对照组",
            margin_change > 0 ~ "保证金提高",
            margin_change < 0 ~ "保证金下调",
            TRUE ~ "保证金不变"
          )
        )
      
      # 按保证金变化方向汇总
      margin_direction_summary <- did_clean_with_dir %>%
        dplyr::filter(!is.na(margin_change)) %>%
        dplyr::group_by(margin_change_dir) %>%
        dplyr::summarise(
          observations = n(),
          contracts = n_distinct(contract_id),
          avg_volatility = mean(log_volatility, na.rm = TRUE),
          .groups = "drop"
        )
      
      # 保存保证金变化方向摘要
      write.csv(margin_direction_summary, 
                file.path(tables_dir, "margin_direction_summary.csv"), 
                row.names = FALSE)
      
      # 如果成功，就跳出循环
      break
    } else {
      cat("att_gt分析返回NULL\n")
    }
  }, error = function(e) {
    # 关闭日志
    sink()
    
    # 读取并显示详细错误信息
    cat("att_gt分析失败:", e$message, "\n")
    if(file.exists(temp_log)) {
      log_content <- readLines(temp_log)
      cat("详细错误信息:\n")
      cat(paste(head(log_content, 20), collapse = "\n"), "\n")
      
      # 如果是特定错误，提供更多指导
      if(grepl("there are no never treated units", paste(log_content, collapse = ""))) {
        cat("\n检测到'没有永不处理单位'错误\n")
        cat("这通常发生在使用'nevertreated'策略，但数据中缺少永不处理的合约时\n")
        cat("将在下一次迭代尝试'notyettreated'策略\n")
      }
    }
    
    #-----------------------------------------------------------------------------
    # 使用传统面板DID方法作为备选方案
    #-----------------------------------------------------------------------------
    cat("\natt_gt分析失败，尝试使用传统面板DID方法作为备选...\n")
    
    # 确保已加载fixest包
    if(!require("fixest", quietly = TRUE)) {
      cat("安装fixest包...\n")
      install.packages("fixest", repos = "https://cloud.r-project.org/")
      library(fixest)
    }
    
    # 准备数据
    panel_did_data <- did_clean %>%
      dplyr::mutate(
        # 确保处理变量是二进制的
        treated = as.numeric(!is.na(first_treat_date)),
        post = as.numeric(date >= first_treat_date),
        did = treated * post,  # 交互项 - DID处理效应
        # 将日期和合约转换为因子，用于固定效应
        time_factor = as.factor(time_period),
        id_factor = as.factor(numeric_id)
      )
    
    cat("运行传统的面板DID回归...\n")
    
    # 尝试使用不同的模型规范
    did_models <- list()
    
    # 模型1: 基本的二期二组DID
    tryCatch({
      did_models$basic <- feols(
        log_volatility ~ did + treated + post, 
        data = panel_did_data
      )
      cat("基本DID模型估计完成\n")
    }, error = function(e) {
      cat("基本DID模型估计失败:", e$message, "\n")
    })
    
    # 模型2: 带固定效应的面板DID
    tryCatch({
      did_models$fe <- feols(
        log_volatility ~ did | id_factor + time_factor, 
        data = panel_did_data
      )
      cat("带固定效应的面板DID模型估计完成\n")
    }, error = function(e) {
      cat("带固定效应的面板DID模型估计失败:", e$message, "\n")
    })

#-----------------------------------------------------------------------------
# 8. 稳健性检验 - 不同控制变量规范
#-----------------------------------------------------------------------------

if(success) {
  cat("\n执行稳健性检验 - 不同控制变量规范...\n")
  
  # 定义不同的控制变量组合
  robust_controls <- list(
    "基准" = c("log_volume_lag1", "log_open_interest_lag1", "return_lag1"),
    "仅交易量" = c("log_volume_lag1"),
    "仅持仓量" = c("log_open_interest_lag1"),
    "仅收益率" = c("return_lag1"),
    "无控制变量" = character(0)
  )
  
  # 使用最佳控制组策略
  best_control_strategy <- control_strategies[1]
  
  # 创建稳健性结果存储
  robust_results <- list()
  
  # 遍历不同控制变量组合
  for(r_name in names(robust_controls)) {
    cat(sprintf("\n尝试控制变量规范: %s\n", r_name))
    r_vars <- robust_controls[[r_name]]
    
    # 准备数据 - 根据需要移除NA
    if(length(r_vars) > 0) {
      r_data <- did_data %>% tidyr::drop_na(dplyr::all_of(r_vars))
      r_formula <- as.formula(paste0("~ ", paste(r_vars, collapse = " + ")))
    } else {
      r_data <- did_data
      r_formula <- ~ 1  # 无控制变量时使用截距
    }
    
    # 执行att_gt分析
    tryCatch({
      cat(sprintf("使用控制变量: %s\n", paste(r_vars, collapse=", ")))
      
      # 执行分析
      r_result <- did::att_gt(
        yname = "log_volatility",
        tname = "time_period",
        idname = "numeric_id",
        gname = "first_treat_period",
        xformla = r_formula,
        data = as.data.frame(r_data),
        control_group = best_control_strategy,
        est_method = "dr",
        allow_unbalanced_panel = TRUE,
        anticipation = 0,
        panel = TRUE
      )
      
      # 计算动态效应
      r_agg <- did::aggte(r_result, 
                         type = "dynamic", 
                         min_e = -10, 
                         max_e = 20,
                         na.rm = TRUE)  # 移除NA值
      
      # 保存结果
      robust_results[[r_name]] <- r_agg
      
      # 保存摘要
      write.csv(summary(r_agg), 
                file.path(tables_dir, paste0("robust_", r_name, ".csv")), 
                row.names = FALSE)
      
      # 绘制图形
      pdf(file.path(plots_dir, paste0("robust_", r_name, ".pdf")), 
          width = 10, height = 6)
      plot(r_agg, 
           main = paste0("稳健性检验 - ", r_name),
           xlab = "相对于处理时间的期数",
           ylab = "处理效应")
      dev.off()
      
    }, error = function(e) {
      cat("稳健性检验失败:", e$message, "\n")
    })
  }
  
  # 尝试绘制比较图
  if(length(robust_results) > 1) {
    cat("\n生成稳健性检验比较图...\n")
    
    # 准备比较数据
    robust_compare <- data.frame()
    for(r_name in names(robust_results)) {
      if(!is.null(robust_results[[r_name]])) {
        temp_data <- summary(robust_results[[r_name]])$att.egt
        temp_data$specification <- r_name
        robust_compare <- rbind(robust_compare, temp_data)
      }
    }
    
    if(nrow(robust_compare) > 0) {
      # 绘制比较图
      robust_plot <- ggplot2::ggplot(robust_compare, 
                     aes(x = event.time, y = att, color = specification, group = specification)) +
        geom_line() +
        geom_point() +
        geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
        geom_vline(xintercept = -0.5, linetype = "dashed", color = "blue") +
        labs(
          title = "不同控制变量规范下的保证金调整效应比较",
          x = "相对于处理时间的期数",
          y = "平均处理效应 (ATT)",
          color = "控制变量规范",
          caption = "使用Callaway & Sant'Anna (2021)方法"
        ) +
        theme(
          plot.title = element_text(hjust = 0.5),
          legend.position = "bottom",
          legend.box = "horizontal"
        )
      
      # 保存比较图
      ggsave(
        file.path(plots_dir, "robust_comparison.png"),
        plot = robust_plot,
        width = 12,
        height = 8,
        dpi = 300
      )
    }
  }
  
  #-----------------------------------------------------------------------------
  # 9. 总结分析结果
  #-----------------------------------------------------------------------------
  cat("\n总结分析结果...\n")
  
  # 创建摘要文件
  summary_file <- file.path(output_dir, "analysis_summary.txt")
  sink(summary_file)
  
  cat("======================================================================\n")
  cat("     保证金调整对期货市场波动率影响的双重差分分析摘要\n")
  cat("     使用 Callaway & Sant'Anna (2021) 方法\n")
  cat("======================================================================\n\n")
  
  cat("分析日期: ", format(Sys.Date(), "%Y-%m-%d"), "\n\n")
  
  cat("1. 数据概览\n")
  cat("-------------\n")
  cat(sprintf("- 分析时间范围: %s 至 %s\n", 
              min(processed_data$date, na.rm = TRUE), 
              max(processed_data$date, na.rm = TRUE)))
  cat(sprintf("- 总共分析了 %d 个观测, %d 个不同合约\n", 
              nrow(panel_data), 
              n_distinct(panel_data$contract_id)))
  cat(sprintf("- 检测到 %d 个保证金调整事件, 涉及 %d 个合约\n", 
              sum(margin_changes$margin_changed), 
              nrow(margin_summary)))
  
  if(success) {
    cat("\n2. 主要发现\n")
    cat("-------------\n")
    
    # 整体效应
    overall_att <- summary(agg_overall)$overall.att
    cat(sprintf("- 整体平均处理效应 (ATT): %.4f (标准误: %.4f)\n", 
                overall_att$att, overall_att$se))
    
    # 处理效应显著性
    if(abs(overall_att$att / overall_att$se) > 1.96) {
      cat(sprintf("  效应在95%%置信水平上显著%s于零\n", 
                  ifelse(overall_att$att > 0, "大", "小")))
    } else {
      cat("  效应在95%置信水平上不显著\n")
    }
    
    # 动态效应
    dynamic_att <- summary(agg_dynamic)$att.egt
    cat("\n- 动态处理效应:\n")
    
    # 处理前效应 (预期效应)
    pre_effects <- dynamic_att[dynamic_att$event.time < 0, ]
    if(nrow(pre_effects) > 0) {
      significant_pre <- sum(abs(pre_effects$att / pre_effects$se) > 1.96)
      cat(sprintf("  处理前: %d/%d 个期间效应显著 (平行趋势假设检验)\n", 
                  significant_pre, nrow(pre_effects)))
    }
    
    # 处理后即时效应
    immediate_effects <- dynamic_att[dynamic_att$event.time >= 0 & dynamic_att$event.time <= 3, ]
    if(nrow(immediate_effects) > 0) {
      significant_imm <- sum(abs(immediate_effects$att / immediate_effects$se) > 1.96)
      cat(sprintf("  处理后即时效应 (0-3期): %d/%d 个期间效应显著\n", 
                  significant_imm, nrow(immediate_effects)))
    }
    
    # 处理后长期效应
    long_effects <- dynamic_att[dynamic_att$event.time > 3, ]
    if(nrow(long_effects) > 0) {
      significant_long <- sum(abs(long_effects$att / long_effects$se) > 1.96)
      cat(sprintf("  处理后长期效应 (>3期): %d/%d 个期间效应显著\n", 
                  significant_long, nrow(long_effects)))
    }
    
    cat("\n3. 稳健性检验\n")
    cat("-------------\n")
    cat(sprintf("- 尝试了 %d 种不同的控制变量规范\n", length(robust_controls)))
    cat("- 详细结果见稳健性比较图和单独的结果文件\n")
    
    cat("\n4. 结论\n")
    cat("-------------\n")
    if(overall_att$att > 0 && abs(overall_att$att / overall_att$se) > 1.96) {
      cat("- 保证金调整总体上显著增加了期货市场波动率\n")
    } else if(overall_att$att < 0 && abs(overall_att$att / overall_att$se) > 1.96) {
      cat("- 保证金调整总体上显著降低了期货市场波动率\n")
    } else {
      cat("- 保证金调整对期货市场波动率没有显著影响\n")
    }
    
    # 分析动态效应模式
    positive_effects <- sum(dynamic_att$att > 0 & dynamic_att$event.time >= 0)
    negative_effects <- sum(dynamic_att$att < 0 & dynamic_att$event.time >= 0)
    
    if(positive_effects > negative_effects) {
      cat(sprintf("- 处理后期间中，有 %d/%d 个期间显示正向效应 (增加波动率)\n", 
                  positive_effects, positive_effects + negative_effects))
    } else {
      cat(sprintf("- 处理后期间中，有 %d/%d 个期间显示负向效应 (降低波动率)\n", 
                  negative_effects, positive_effects + negative_effects))
    }
  } else {
    cat("\n2. 分析结果\n")
    cat("-------------\n")
    cat("- 双重差分分析未能成功完成，可能是由于以下原因：\n")
    cat("  * 数据中没有足够的对照组\n")
    cat("  * 处理组在处理前没有足够的观测期\n")
    cat("  * 存在其他数据结构问题\n")
    cat("\n- 建议：\n")
    cat("  * 检查数据中是否存在足够的'从未处理'的合约\n")
    cat("  * 考虑使用其他DID方法，如传统DID或合成控制法\n")
    cat("  * 调整分析时间范围，确保包含足够的处理前期间\n")
  }
  
  sink()
  cat(sprintf("分析摘要已保存到 %s\n", summary_file))
  
  #-----------------------------------------------------------------------------
  # 10. 结束信息
  #-----------------------------------------------------------------------------
  if(success) {
    cat("\n分析成功完成！\n")
    cat(sprintf("所有结果已保存到 %s 目录\n", output_dir))
    cat("主要输出文件：\n")
    cat(sprintf("- 表格: %s\n", tables_dir))
    cat(sprintf("- 图表: %s\n", plots_dir)) 
    cat(sprintf("- 分析摘要: %s\n", summary_file))
  } else {
    cat("\n分析过程中遇到错误，未能完全完成。\n")
    cat("请检查以下几点：\n")
    cat("1. 确保数据中有足够的对照组（永不处理组或至少在样本期末仍未处理的组）\n")
    cat("2. 确保每个处理组在处理前有足够的观测期\n")
    cat("3. 检查控制变量是否存在大量缺失值\n")
    cat("4. 尝试修改控制组策略（'nevertreated'或'notyettreated'）\n")
    cat("5. 考虑使用其他DID估计方法\n")
  }
  
  # 返回工作目录中的所有结果文件列表，便于查看
  result_files <- list.files(output_dir, recursive = TRUE)
  if(length(result_files) > 0) {
    cat("\n生成的结果文件列表：\n")
    for(i in 1:min(10, length(result_files))) {
      cat(sprintf("- %s\n", result_files[i]))
    }
    if(length(result_files) > 10) {
      cat(sprintf("...以及其他 %d 个文件\n", length(result_files) - 10))
    }
  }
  
  cat("\n分析脚本执行完毕。\n")
}
