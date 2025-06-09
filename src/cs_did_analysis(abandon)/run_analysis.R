#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# 保证金调整对期货市场波动率影响的双重差分分析 - 主运行脚本
# 使用 Callaway & Sant'Anna (2021) 方法
# 分析所有期间的保证金调整数据

# 设置工作目录到脚本所在位置
tryCatch({
  script_dir <- dirname(normalizePath(commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))][1]))
  if (!is.na(script_dir) && script_dir != "") {
    setwd(script_dir)
    cat(sprintf("工作目录设置为: %s\n", getwd()))
  }
}, error = function(e) {
  cat("无法确定脚本目录，将使用当前目录\n")
})

cat("===============================================================\n")
cat("  保证金调整对期货市场波动率影响的双重差分分析\n")
cat("  使用 Callaway & Sant'Anna (2021) 方法\n")
cat("===============================================================\n\n")

# 记录开始时间
start_time <- Sys.time()
cat(sprintf("分析开始时间: %s\n\n", start_time))

# 检查必要文件是否存在
files_to_check <- c(
  "did_analysis_improved_part1.R",
  "did_analysis_improved_part2.R",
  "did_analysis_improved_part3.R",
  "did_analysis_improved_part4_fixed.R",
  "did_analysis_improved_part5_with_plots.R"
)

all_files_exist <- TRUE
for (file in files_to_check) {
  if (!file.exists(file)) {
    cat(sprintf("错误: 找不到文件 '%s'\n", file))
    all_files_exist <- FALSE
  }
}

if (!all_files_exist) {
  cat("请确保所有必要的脚本文件都在cs_did_analysis目录中，并从该目录运行此脚本\n")
  quit(status = 1)
}

# 执行各部分分析
cat("1. 执行数据加载和基本处理...\n")
source("did_analysis_improved_part1.R")
cat("第1部分完成\n\n")

cat("2. 执行保证金率缺失值处理和保证金调整事件识别...\n")
source("did_analysis_improved_part2.R")
cat("第2部分完成\n\n")

cat("3. 执行波动率计算和其他分析变量创建...\n")
source("did_analysis_improved_part3.R")
cat("第3部分完成\n\n")

cat("4. 执行DID分析变量创建和人工对照组创建...\n")
source("did_analysis_improved_part4_fixed.R")
cat("第4部分完成\n\n")

cat("5. 执行Callaway & Sant'Anna DID分析和结果可视化...\n")
source("did_analysis_improved_part5_with_plots.R")
cat("第5部分完成\n\n")

# 记录结束时间
end_time <- Sys.time()
execution_time <- difftime(end_time, start_time, units = "mins")

cat("===============================================================\n")
cat(sprintf("分析结束时间: %s\n", end_time))
cat(sprintf("总执行时间: %.2f 分钟\n", as.numeric(execution_time)))
cat("===============================================================\n\n")

# 输出结果目录
output_dir <- "../output/did_analysis_improved"
if (dir.exists(output_dir)) {
  result_files <- list.files(output_dir, recursive = TRUE)
  if(length(result_files) > 0) {
    cat("生成的结果文件列表：\n")
    for(file in result_files) {
      cat(sprintf("- %s\n", file))
    }
  }
}

cat("\n分析完成！详细结果请参阅 output/did_analysis_improved 目录\n")
