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
  "scales",         # 图表刻度
  "zoo"             # 用于na.locf函数
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

# 确定项目根目录
if (dir.exists("../data")) {
  project_root <- ".."
} else {
  project_root <- "."
}

# 创建输出目录
output_dir <- file.path(project_root, "output/did_analysis_improved")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 创建子目录
tables_dir <- file.path(output_dir, "tables")
plots_dir <- file.path(output_dir, "plots")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

# 设置图形主题
theme_set(theme_minimal())

#-----------------------------------------------------------------------------
# 3. 读取原始数据
#-----------------------------------------------------------------------------
cat("读取原始数据...\n")
raw_data_path <- file.path(project_root, "data/raw/futures_margin_data.csv")
if (!file.exists(raw_data_path)) {
  stop(paste("数据文件不存在:", raw_data_path))
}
raw_data <- data.table::fread(raw_data_path)
cat(sprintf("成功读取原始数据，共 %d 行, %d 列\n", nrow(raw_data), ncol(raw_data)))

#-----------------------------------------------------------------------------
# 4. 基本数据处理
#-----------------------------------------------------------------------------
# 转换日期和创建合约ID
processed_data <- raw_data %>%
  dplyr::mutate(
    date = as.Date(date),
    contract_id = paste(variety, exchange, sep = "_"),
    # 确保保证金率是数值型
    margin_rate = as.numeric(margin_rate)
  )

# 检查保证金率缺失值
margin_na_by_contract <- processed_data %>%
  dplyr::group_by(variety, exchange) %>%
  dplyr::summarise(
    total_obs = n(),
    margin_na = sum(is.na(margin_rate)),
    pct_na = round(sum(is.na(margin_rate)) / n() * 100, 1),
    .groups = "drop"
  )

cat("保证金率缺失值情况:\n")
print(head(margin_na_by_contract, 10))

#-----------------------------------------------------------------------------
# 保存进度
#-----------------------------------------------------------------------------
cat("基本数据处理完成。部分1结束。\n")
# 保存数据以便后续分析
save(processed_data, project_root, output_dir, tables_dir, plots_dir,
     file = file.path(output_dir, "part1_data.RData"))
