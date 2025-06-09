#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

# 诊断脚本：分析 att_gt_result 结构

# 确定项目根目录
if (dir.exists("../data")) {
  project_root <- ".."
} else {
  project_root <- "."
}
output_dir <- file.path(project_root, "output/did_analysis_improved")

# 加载第4部分数据
cat("加载第4部分处理的数据...\n")
load(file.path(output_dir, "part4_data.RData"))

# 加载必要的包
library(dplyr)
library(did)
library(tidyr)

# 输出详细的诊断信息
cat("------------- 诊断信息开始 -------------\n")
cat("R 版本:", R.version$version.string, "\n")
cat("did 包版本:", as.character(packageVersion("did")), "\n")

# 定义控制变量公式
control_vars <- c("log_volume_lag1", "log_open_interest_lag1", "return_lag1")
control_formula <- as.formula(paste0("~ ", paste(control_vars, collapse = " + ")))

# 检查是否有人工控制组
has_artificial_control <- "artificial_control" %in% names(did_clean) &&
                          any(did_clean$artificial_control, na.rm = TRUE)

# 确定控制组策略
control_strategies <- if(has_artificial_control) "notyettreated" else c("nevertreated", "notyettreated")

# 执行 att_gt 分析
cat("\n尝试运行 att_gt...\n")
tryCatch({
  # 执行 att_gt 分析 (Callaway & Sant'Anna, 2021)
  att_gt_result <- did::att_gt(
    yname = "log_volatility",         # 结果变量（对数波动率）
    tname = "time_period",            # 时间变量（连续编号）
    idname = "numeric_id",            # 个体ID
    gname = "first_treat_period",     # 处理组变量（首次受处理的时期）
    xformla = control_formula,        # 控制变量
    data = as.data.frame(did_clean),  # 数据
    control_group = control_strategies[1], # 控制组策略
    est_method = "dr",                # 使用疑似回归(doubly-robust)估计
    allow_unbalanced_panel = TRUE,    # 允许非平衡面板
    clustervars = "contract_id",      # 聚类标准误的变量
    anticipation = 0,                 # 假设无预期效应
    panel = TRUE                      # 面板数据
  )
  
  # 检查结果是否成功
  if(!is.null(att_gt_result)) {
    cat("att_gt 分析成功完成\n")
    
    # 打印 att_gt_result 的基本信息
    cat("\natt_gt_result 基本信息:\n")
    cat("类型:", class(att_gt_result), "\n")
    cat("长度:", length(att_gt_result), "\n")
    cat("名称:", paste(names(att_gt_result), collapse=", "), "\n")
    
    # 打印 att_gt_result 的结构
    cat("\natt_gt_result 详细结构:\n")
    print(str(att_gt_result, max.level = 2))
    
    # 检查关键组件
    cat("\n检查 att_gt_result 的关键组件:\n")
    
    # 检查处理组(group)
    groups <- unique(att_gt_result$group)
    cat("处理组数量:", length(groups), "\n")
    cat("处理组值:", paste(head(groups, 10), collapse=", "), if(length(groups) > 10) "..." else "", "\n")
    
    # 检查事件时间(event.time)
    event_times <- unique(att_gt_result$event.time)
    cat("事件时间数量:", length(event_times), "\n")
    cat("事件时间值:", paste(head(event_times, 10), collapse=", "), if(length(event_times) > 10) "..." else "", "\n")
    cat("是否存在NA事件时间:", any(is.na(event_times)), "\n")
    
    # 检查处理效应(att)
    att_values <- att_gt_result$att
    cat("处理效应数量:", length(att_values), "\n")
    cat("处理效应摘要:\n")
    print(summary(att_values))
    cat("处理效应中NA值数量:", sum(is.na(att_values)), "\n")
    
    # 检查标准误(se)
    se_values <- att_gt_result$se
    cat("标准误数量:", length(se_values), "\n")
    cat("标准误摘要:\n")
    print(summary(se_values))
    cat("标准误中NA值数量:", sum(is.na(se_values)), "\n")
    
    # 检查数据框中的关键变量
    cat("\n检查数据框中的关键变量:\n")
    did_clean_summary <- did_clean %>%
      summarise(
        样本总数 = n(),
        首次处理时期NA数量 = sum(is.na(first_treat_period)),
        时间变量NA数量 = sum(is.na(time_period)),
        个体ID变量NA数量 = sum(is.na(numeric_id)),
        结果变量NA数量 = sum(is.na(log_volatility))
      )
    print(did_clean_summary)
    
    # 尝试使用不同参数组合调用 aggte 函数
    cat("\n尝试多种 aggte 参数组合:\n")
    
    # 尝试方法1：默认参数加上 na.rm=TRUE
    cat("\n尝试方法1：默认参数 + na.rm=TRUE\n")
    tryCatch({
      agg_results_1 <- did::aggte(att_gt_result, na.rm = TRUE)
      cat("成功! agg_results_1 结构:\n")
      print(str(agg_results_1))
    }, error = function(e) {
      cat("失败:", e$message, "\n")
    })
    
    # 尝试方法2：动态处理效应 + 严格参数检查
    cat("\n尝试方法2：动态处理效应 + 严格参数\n")
    tryCatch({
      agg_results_2 <- did::aggte(
        att_gt_result,
        type = "dynamic",
        min_e = min(event_times[!is.na(event_times)]),
        max_e = max(event_times[!is.na(event_times)]),
        na.rm = TRUE,
        balance_e = FALSE
      )
      cat("成功! agg_results_2 结构:\n")
      print(str(agg_results_2))
    }, error = function(e) {
      cat("失败:", e$message, "\n")
      
      # 检查类型错误的具体原因
      if(grepl("invalid 'type'", e$message)) {
        cat("检查type参数问题...\n")
        tryCatch({
          # 打印type参数的实际类型
          type_param <- "dynamic"
          cat("type参数类型:", class(type_param), "\n")
          cat("type参数值:", type_param, "\n")
          
          # 尝试不同的type参数形式
          cat("尝试使用字符串'dynamic'...\n")
          agg_results_2a <- did::aggte(
            att_gt_result,
            type = "dynamic", 
            na.rm = TRUE
          )
          cat("成功!\n")
        }, error = function(e2) {
          cat("仍然失败:", e2$message, "\n")
        })
      }
    })
    
    # 尝试方法3：手动检查 did::aggte 函数源代码
    cat("\n尝试方法3：手动检查 did::aggte 函数源代码\n")
    tryCatch({
      aggte_src <- deparse(did::aggte)
      cat("aggte 函数源代码前10行:\n")
      cat(paste(head(aggte_src, 10), collapse="\n"), "\n...\n")
      
      # 检查did包的版本要求
      cat("检查did包版本兼容性...\n")
      did_description <- packageDescription("did")
      cat("did包版本:", did_description$Version, "\n")
      cat("did包依赖项:", did_description$Depends, "\n")
      cat("did包导入项:", did_description$Imports, "\n")
    }, error = function(e) {
      cat("无法获取源代码:", e$message, "\n")
    })
    
    # 尝试方法4：手动构建处理效应数据的更健壮版本
    cat("\n尝试方法4：手动构建处理效应数据的更健壮版本\n")
    tryCatch({
      # 过滤有效的事件时间
      valid_event_times <- event_times[!is.na(event_times)]
      
      # 创建结果数据框
      dynamic_data <- data.frame(
        event.time = numeric(),
        att = numeric(),
        se = numeric()
      )
      
      # 输出att_gt_result的内部结构信息
      cat("att_gt_result的组数:", length(unique(att_gt_result$group)), "\n")
      
      # 对每个有效的事件时间
      for(e in valid_event_times) {
        # 找到对应事件时间的处理效应
        idx <- which(att_gt_result$event.time == e)
        if(length(idx) > 0) {
          att_values_e <- att_gt_result$att[idx]
          se_values_e <- att_gt_result$se[idx]
          
          # 移除NA值
          valid_idx <- !is.na(att_values_e) & !is.na(se_values_e)
          att_values_e <- att_values_e[valid_idx]
          se_values_e <- se_values_e[valid_idx]
          
          if(length(att_values_e) > 0) {
            # 计算均值
            mean_att <- mean(att_values_e)
            mean_se <- mean(se_values_e)
            
            # 添加到结果
            dynamic_data <- rbind(dynamic_data, data.frame(
              event.time = e,
              att = mean_att,
              se = mean_se
            ))
          }
        }
      }
      
      # 输出结果
      if(nrow(dynamic_data) > 0) {
        cat("手动构建成功，提取了", nrow(dynamic_data), "个事件时间点\n")
        print(dynamic_data)
        
        # 尝试绘制简单图形
        cat("\n尝试生成简单的动态处理效应图...\n")
        # 保存为PDF文件
        pdf(file.path(output_dir, "manual_dynamic_effects.pdf"), width=8, height=6)
        # 计算置信区间
        dynamic_data$ci_lower <- dynamic_data$att - 1.96 * dynamic_data$se
        dynamic_data$ci_upper <- dynamic_data$att + 1.96 * dynamic_data$se
        # 绘制图形
        plot(dynamic_data$event.time, dynamic_data$att, type="b", pch=19,
             xlab="事件时间", ylab="处理效应", main="手动计算的动态处理效应",
             ylim=c(min(dynamic_data$ci_lower), max(dynamic_data$ci_upper)))
        # 添加置信区间
        for(i in 1:nrow(dynamic_data)) {
          lines(c(dynamic_data$event.time[i], dynamic_data$event.time[i]),
                c(dynamic_data$ci_lower[i], dynamic_data$ci_upper[i]))
        }
        # 添加0线
        abline(h=0, lty=2, col="red")
        # 添加事件时间0线
        abline(v=0, lty=2, col="blue")
        # 关闭设备
        dev.off()
        cat("图形已保存到", file.path(output_dir, "manual_dynamic_effects.pdf"), "\n")
        
        # 保存手动计算的数据
        save(dynamic_data, file=file.path(output_dir, "manual_dynamic_effects.RData"))
        cat("数据已保存到", file.path(output_dir, "manual_dynamic_effects.RData"), "\n")
      } else {
        cat("未能从 att_gt_result 提取数据\n")
      }
    }, error = function(e) {
      cat("手动构建失败:", e$message, "\n")
    })
  } else {
    cat("att_gt 分析结果为空\n")
  }
}, error = function(e) {
  cat("att_gt 分析失败:", e$message, "\n")
})

cat("------------- 诊断信息结束 -------------\n")

