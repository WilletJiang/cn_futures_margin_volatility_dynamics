# -----------------------------------------------------------------------------
# did_analysis_improved_part5_manual.R
# 
# 目的：手动计算处理效应，避开did::aggte()函数的问题
# 作者：
# 创建日期：2025-05-13
# -----------------------------------------------------------------------------

# 1. 加载必要的库和数据
# -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(did)
library(fixest)
library(data.table)
library(purrr)
library(tidyr)
library(stringr)
library(lubridate)
library(scales)

# 设置工作目录和输出目录
output_dir <- "../output"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# 加载之前处理好的数据
tryCatch({
  load("part4_data.RData")
  cat("成功加载part4_data.RData数据\n")
}, error = function(e) {
  stop("加载数据失败: ", e$message)
})

# 2. 检查并预处理数据
# -----------------------------------------------------------------------------
cat("开始检查数据...\n")

# 检查first_treat_period中的NA值
na_count_first_treat <- sum(is.na(panel_data$first_treat_period))
cat("first_treat_period中的NA值数量:", na_count_first_treat, "占比:", 
    round(na_count_first_treat/nrow(panel_data)*100, 2), "%\n")

# 检查其他关键变量是否有NA值
cat("ID中的NA值数量:", sum(is.na(panel_data$ID)), "\n")
cat("time中的NA值数量:", sum(is.na(panel_data$time)), "\n")
cat("volatility_log中的NA值数量:", sum(is.na(panel_data$volatility_log)), "\n")

# 检查unique的first_treat_period值
unique_treat_periods <- unique(panel_data$first_treat_period[!is.na(panel_data$first_treat_period)])
cat("不同的首次处理时期数量:", length(unique_treat_periods), "\n")
cat("首次处理时期值:", paste(sort(unique_treat_periods), collapse=", "), "\n")

# 创建一个新的数据集，处理NA值问题
# 策略：将first_treat_period为NA的观测视为"从未处理"的控制组
panel_data_processed <- panel_data %>%
  mutate(
    # 将NA的first_treat_period标记为一个非常大的值（表示永不处理）
    first_treat_period_fixed = ifelse(is.na(first_treat_period), 9999, first_treat_period)
  )

# 检查处理后的变量
cat("处理后first_treat_period_fixed中的NA值数量:", 
    sum(is.na(panel_data_processed$first_treat_period_fixed)), "\n")

# 3. 使用不同的控制组策略重新进行att_gt估计
# -----------------------------------------------------------------------------
cat("开始使用不同的控制组策略进行att_gt估计...\n")

# 尝试nevertreated策略
tryCatch({
  att_gt_never <- att_gt(
    yname = "volatility_log",
    tname = "time",
    idname = "ID",
    gname = "first_treat_period_fixed", 
    control_group = "nevertreated",
    data = panel_data_processed,
    anticipation = 0,
    allow_unbalanced_panel = TRUE,
    bstrap = TRUE,
    cband = TRUE,
    clustervars = "ID",
    cores = 1
  )
  cat("nevertreated策略的att_gt估计完成\n")
  
  # 检查结果
  summary_never <- summary(att_gt_never)
  cat("nevertreated策略的结果摘要:\n")
  print(summary_never)
  
  # 检查是否有有效的att估计
  na_count_att_never <- sum(is.na(att_gt_never$att))
  cat("nevertreated策略中的NA att数量:", na_count_att_never, 
      "占比:", round(na_count_att_never/length(att_gt_never$att)*100, 2), "%\n")
  
}, error = function(e) {
  cat("nevertreated策略失败:", e$message, "\n")
})

# 尝试notyettreated策略
tryCatch({
  att_gt_notyet <- att_gt(
    yname = "volatility_log",
    tname = "time",
    idname = "ID",
    gname = "first_treat_period_fixed", 
    control_group = "notyettreated",
    data = panel_data_processed,
    anticipation = 0,
    allow_unbalanced_panel = TRUE,
    bstrap = TRUE,
    cband = TRUE,
    clustervars = "ID",
    cores = 1
  )
  cat("notyettreated策略的att_gt估计完成\n")
  
  # 检查结果
  summary_notyet <- summary(att_gt_notyet)
  cat("notyettreated策略的结果摘要:\n")
  print(summary_notyet)
  
  # 检查是否有有效的att估计
  na_count_att_notyet <- sum(is.na(att_gt_notyet$att))
  cat("notyettreated策略中的NA att数量:", na_count_att_notyet, 
      "占比:", round(na_count_att_notyet/length(att_gt_notyet$att)*100, 2), "%\n")
  
}, error = function(e) {
  cat("notyettreated策略失败:", e$message, "\n")
})

# 选择最佳的att_gt结果
if (exists("att_gt_never") && sum(!is.na(att_gt_never$att)) > 0) {
  att_gt_best <- att_gt_never
  control_strategy <- "nevertreated"
} else if (exists("att_gt_notyet") && sum(!is.na(att_gt_notyet$att)) > 0) {
  att_gt_best <- att_gt_notyet
  control_strategy <- "notyettreated"
} else {
  cat("两种策略都无法产生有效的att估计，将进行手动计算\n")
  control_strategy <- "manual"
}

# 4. 开发手动方法计算动态处理效应
# -----------------------------------------------------------------------------
cat("开始手动计算动态处理效应...\n")

# 函数：计算特定相对时间的平均处理效应
calculate_dynamic_att <- function(data, rel_time) {
  # 数据应该包含以下列：
  # - group: 处理组
  # - t: 时间
  # - rel_time: 相对于处理的时间
  # - att: 处理效应
  # - se: 标准误
  
  filtered_data <- data %>% 
    filter(rel_time == !!rel_time & !is.na(att))
  
  if (nrow(filtered_data) == 0) {
    return(list(att = NA, se = NA, n = 0))
  }
  
  # 计算加权平均值
  # 我们使用标准误的倒数的平方作为权重，以让精确度更高的估计有更大的权重
  filtered_data <- filtered_data %>%
    mutate(weight = 1 / (se^2))
  
  sum_weights <- sum(filtered_data$weight, na.rm = TRUE)
  
  if (sum_weights > 0) {
    weighted_att <- sum(filtered_data$att * filtered_data$weight, na.rm = TRUE) / sum_weights
    # 计算加权标准误
    weighted_se <- sqrt(1 / sum_weights)
  } else {
    weighted_att <- mean(filtered_data$att, na.rm = TRUE)
    weighted_se <- NA
  }
  
  return(list(
    att = weighted_att,
    se = weighted_se,
    n = nrow(filtered_data)
  ))
}

# 如果aggte失败，则使用手动计算的方法
manual_calculation <- function(att_gt_result) {
  # 提取att_gt_result中的基本信息
  data_for_plot <- data.frame(
    group = att_gt_result$group,
    t = att_gt_result$t,
    att = att_gt_result$att,
    se = att_gt_result$se
  )
  
  # 计算相对于处理时间
  data_for_plot <- data_for_plot %>%
    mutate(rel_time = t - group) %>%
    filter(!is.na(att))  # 去除NA的att
  
  # 查看不同的rel_time
  rel_times <- sort(unique(data_for_plot$rel_time))
  
  # 对每个相对时间计算平均处理效应
  dynamic_effects <- lapply(rel_times, function(rt) {
    result <- calculate_dynamic_att(data_for_plot, rt)
    result$rel_time <- rt
    return(result)
  })
  
  # 转换为数据框
  dynamic_effects_df <- bind_rows(dynamic_effects)
  
  return(dynamic_effects_df)
}

# 根据前面的结果决定使用哪种方法
if (control_strategy %in% c("nevertreated", "notyettreated")) {
  # 如果att_gt产生了有效结果，尝试用标准的aggte函数
  tryCatch({
    cat("尝试使用aggte函数计算动态处理效应...\n")
    dynamic_att <- aggte(att_gt_best, type = "dynamic", na.rm = TRUE)
    manual_needed <- FALSE
    cat("成功使用aggte函数计算动态处理效应\n")
  }, error = function(e) {
    cat("aggte函数失败:", e$message, "\n")
    cat("将使用手动计算方法\n")
    manual_needed <- TRUE
  })
  
  if (exists("manual_needed") && manual_needed) {
    dynamic_results <- manual_calculation(att_gt_best)
  }
} else {
  # 如果两种控制策略都失败，则使用手动方法
  # 这里我们需要创建一个简化的面板数据集，直接计算处理效应
  cat("使用手动方法直接从面板数据计算处理效应...\n")
  
  # 创建一个包含处理状态的数据框
  panel_simplified <- panel_data_processed %>%
    mutate(
      treated = !is.na(first_treat_period) & (time >= first_treat_period_fixed),
      rel_time = time - first_treat_period_fixed
    ) %>%
    filter(!is.na(rel_time) & rel_time >= -10 & rel_time <= 10)  # 限制相对时间范围
  
  # 计算每个相对时间的处理效应
  rel_times <- sort(unique(panel_simplified$rel_time))
  
  # 使用固定效应模型计算每个相对时间的处理效应
  tryCatch({
    # 创建相对时间哑变量
    panel_with_dummies <- panel_simplified %>%
      mutate(rel_time_factor = factor(rel_time))
    
    # 使用fixest运行面板回归
    model <- feols(
      volatility_log ~ i(rel_time_factor, ref = -1) | ID + time,
      data = panel_with_dummies,
      cluster = "ID"
    )
    
    # 提取系数和标准误
    coefs <- coef(model)
    ses <- se(model)
    
    # 创建动态效应数据框
    dynamic_results <- data.frame(
      rel_time = as.numeric(gsub("rel_time_factor::", "", names(coefs))),
      att = as.numeric(coefs),
      se = as.numeric(ses),
      n = NA
    )
    
    # 移除参考类别（通常是-1）
    dynamic_results <- dynamic_results %>%
      filter(!is.na(rel_time))
    
    cat("成功使用固定效应模型计算动态处理效应\n")
  }, error = function(e) {
    cat("固定效应模型失败:", e$message, "\n")
    cat("将使用简化的差分计算方法\n")
    
    # 简化方法：计算处理组与控制组在各相对时间点的均值差异
    dynamic_results_list <- lapply(rel_times, function(rt) {
      temp_data <- panel_simplified %>% 
        filter(rel_time == rt)
      
      treated_mean <- mean(temp_data$volatility_log[temp_data$treated], na.rm = TRUE)
      control_mean <- mean(temp_data$volatility_log[!temp_data$treated], na.rm = TRUE)
      
      att_estimate <- treated_mean - control_mean
      
      # 简单计算标准误
      treated_var <- var(temp_data$volatility_log[temp_data$treated], na.rm = TRUE)
      control_var <- var(temp_data$volatility_log[!temp_data$treated], na.rm = TRUE)
      
      treated_n <- sum(!is.na(temp_data$volatility_log[temp_data$treated]))
      control_n <- sum(!is.na(temp_data$volatility_log[!temp_data$treated]))
      
      if (treated_n > 0 && control_n > 0) {
        se_estimate <- sqrt(treated_var/treated_n + control_var/control_n)
      } else {
        se_estimate <- NA
      }
      
      return(list(
        rel_time = rt,
        att = att_estimate,
        se = se_estimate,
        n_treated = treated_n,
        n_control = control_n
      ))
    })
    
    # 合并结果为数据框
    dynamic_results <- bind_rows(dynamic_results_list)
  })
}

# 5. 绘制处理效应图表
# -----------------------------------------------------------------------------
cat("开始绘制处理效应图表...\n")

# 确保dynamic_results存在
if (!exists("dynamic_results")) {
  if (exists("dynamic_att")) {
    # 如果使用的是aggte的结果
    dynamic_results <- data.frame(
      rel_time = dynamic_att$egt,
      att = dynamic_att$att.egt,
      se = dynamic_att$se.egt,
      n = NA
    )
  } else {
    stop("未能生成动态处理效应结果")
  }
}

# 计算置信区间
dynamic_results <- dynamic_results %>%
  mutate(
    ci_lower = att - 1.96 * se,
    ci_upper = att + 1.96 * se
  )

# 绘制动态处理效应图表
dynamic_plot <- ggplot(dynamic_results, aes(x = rel_time, y = att)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "blue", size = 3) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2, fill = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "darkgreen") +
  labs(
    title = "Dynamic Treatment Effects of Margin Rate Changes",
    subtitle = "Effect on Logarithm of Volatility",
    x = "Time Relative to Treatment (Months)",
    y = "Treatment Effect",
    caption = "Note: Shaded area represents 95% confidence intervals"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10),
    legend.position = "bottom",
    legend.title = element_blank()
  )

# 显示图表
print(dynamic_plot)

# 保存图表
ggsave(
  file.path(output_dir, "dynamic_treatment_effects.png"),
  plot = dynamic_plot,
  width = 10,
  height = 7,
  dpi = 300
)

# 6. 为论文准备辅助图表
# -----------------------------------------------------------------------------
cat("创建辅助图表...\n")

# 绘制处理前后的波动率分布图
if (exists("panel_simplified")) {
  # 为图表创建一个新的处理状态变量
  panel_for_plot <- panel_simplified %>%
    mutate(
      treatment_status = case_when(
        !is.na(first_treat_period) & rel_time < 0 ~ "Pre-Treatment",
        !is.na(first_treat_period) & rel_time >= 0 ~ "Post-Treatment",
        TRUE ~ "Control Group"
      ),
      # 转换为因子并设置顺序
      treatment_status = factor(
        treatment_status,
        levels = c("Control Group", "Pre-Treatment", "Post-Treatment")
      )
    )
  
  # 绘制波动率分布图
  volatility_dist_plot <- ggplot(panel_for_plot, aes(x = volatility_log, fill = treatment_status)) +
    geom_density(alpha = 0.6) +
    scale_fill_manual(
      values = c("Control Group" = "grey", "Pre-Treatment" = "blue", "Post-Treatment" = "red"),
      name = "Treatment Status"
    ) +
    labs(
      title = "Distribution of Log Volatility by Treatment Status",
      x = "Log Volatility",
      y = "Density"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10),
      legend.position = "bottom"
    )
  
  # 显示图表
  print(volatility_dist_plot)
  
  # 保存图表
  ggsave(
    file.path(output_dir, "volatility_distribution.png"),
    plot = volatility_dist_plot,
    width = 10,
    height = 7,
    dpi = 300
  )
}

# 7. 保存结果
# -----------------------------------------------------------------------------
cat("保存分析结果...\n")

# 保存dynamic_results为CSV文件
write.csv(
  dynamic_results,
  file.path(output_dir, "dynamic_treatment_effects.csv"),
  row.names = FALSE
)

# 保存处理效应的摘要结果
summary_results <- dynamic_results %>%
  summarise(
    average_att = mean(att, na.rm = TRUE),
    min_att = min(att, na.rm = TRUE),
    max_att = max(att, na.rm = TRUE),
    pre_treatment_att = mean(att[rel_time < 0], na.rm = TRUE),
    post_treatment_att = mean(att[rel_time >= 0], na.rm = TRUE),
    significant_effects = sum(ci_lower > 0 | ci_upper < 0, na.rm = TRUE),
    total_effects = sum(!is.na(att))
  )

# 保存摘要结果
write.csv(
  summary_results,
  file.path(output_dir, "treatment_effects_summary.csv"),
  row.names = FALSE
)

# 8. 保存工作区以备后续分析
# -----------------------------------------------------------------------------
cat("保存工作区...\n")

# 创建一个包含关键结果的列表
results_list <- list(
  dynamic_results = dynamic_results,
  summary_results = summary_results
)

# 如果有att_gt结果，也保存它们
if (exists("att_gt_best")) {
  results_list$att_gt_best <- att_gt_best
  results_list$control_strategy <- control_strategy
}

if (exists("dynamic_att")) {
  results_list$dynamic_att <- dynamic_att
}

# 保存结果
save(
  results_list,
  file = file.path(output_dir, "did_manual_results.RData")
)

# 9. 总结结果
# -----------------------------------------------------------------------------
cat("\n========== 分析结果总结 ==========\n")
cat("完成了手动计算的差分分析\n")

if (exists("control_strategy")) {
  cat("使用的控制组策略:", control_strategy, "\n")
} else {
  cat("使用了简化的差分计算方法\n")
}

cat("\n平均处理效应:", round(summary_results$average_att, 4), "\n")
cat("处理前平均效应:", round(summary_results$pre_treatment_att, 4), "\n")
cat("处理后平均效应:", round(summary_results$post_treatment_att, 4), "\n")
cat("显著的处理效应数量:", summary_results$significant_effects, 
    "占比:", round(summary_results$significant_effects/summary_results$total_effects*100, 2), "%\n")

cat("\n动态处理效应图表和数据已保存到输出目录:", output_dir, "\n")

cat("\n分析完成!\n")
