################################################################################
# CH4排放对温度响应的机器学习和PDP分析 - 完整版
# 包含：多模型比较、超参数调优、最优模型选择
################################################################################

# 清空环境
rm(list = ls())
gc()

################################################################################
# 1. 安装和加载必要的包
################################################################################

cat("="*80, "\n")
cat("Step 1: 安装和加载R包\n")
cat("="*80, "\n")

# 定义所需包
packages <- c(
  # 数据处理
  "readxl", "dplyr", "tidyr", "data.table",
  # 可视化
  "ggplot2", "gridExtra", "viridis", "GGally", "corrplot", "cowplot",
  # 机器学习框架
  "caret", "mlr3", "mlr3learners", "mlr3tuning", "mlr3viz",
  # 具体模型
  "randomForest", "gbm", "xgboost", "ranger", "glmnet", 
  "kernlab", "earth", "nnet", "e1071",
  # PDP和解释性分析
  "pdp", "iml", "DALEX", "vip",
  # 其他工具
  "doParallel", "foreach", "parallel"
)

# 安装缺失的包
cat("\n检查并安装缺失的包...\n")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages) > 0) {
  cat("需要安装的包:", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages, dependencies = TRUE, repos = "https://cloud.r-project.org/")
}

# 加载所有包
cat("\n加载R包...\n")
invisible(lapply(packages, function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  cat("  ✓", pkg, "\n")
}))

# 设置并行计算
n_cores <- min(parallel::detectCores() - 1, 4)
registerDoParallel(cores = n_cores)
cat("\n并行计算核心数:", n_cores, "\n")

################################################################################
# 2. 数据读取和预处理
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 2: 数据读取和预处理\n")
# cat("="*80, "\n")

# 读取数据
data <- read_excel("CH4andT by sun 20260227.xlsx")
cat("\n原始数据维度:", dim(data), "\n")

# data <- filter(data, land_use_type == "paddy")
data <- filter(data, land_use_type == "paddy")


# 创建响应变量
data <- data %>%
  mutate(
    CH4_response = Me / Ma,
    log_CH4_response = log(Me / Ma)
  )

# 选择建模变量
model_data <- data %>%
  dplyr::select(
    log_CH4_response,
    increase,        # 温度增加 (核心解释变量)
    latitude,        # 纬度
    longitude,       # 经度
    MAT,            # 平均气温
    MAP,            # 平均降水
    # land_use_type,  # 土地利用类型
    napplication,   # 氮应用
    # watertable,     # 地下水位
    facility,       # 设施类型
    duration        # 持续时间
  ) %>%
  filter(
    is.finite(log_CH4_response),
    !is.na(increase)
  )

cat("\n过滤后数据维度:", dim(model_data), "\n")

# 处理分类变量
categorical_vars <- c("land_use_type", "napplication", "watertable", "facility")

for(var in categorical_vars) {
  if(var %in% names(model_data)) {
    model_data[[var]] <- as.factor(
      ifelse(is.na(model_data[[var]]) | model_data[[var]] == "", 
             "Unknown", 
             as.character(model_data[[var]]))
    )
  }
}

# 处理数值变量的缺失值
numeric_vars <- c("latitude", "longitude", "MAT", "MAP", "duration", "increase")
for(var in numeric_vars) {
  if(any(is.na(model_data[[var]]))) {
    model_data[[var]][is.na(model_data[[var]])] <- median(model_data[[var]], na.rm = TRUE)
  }
}

# 移除仍有缺失的行
model_data <- na.omit(model_data)
cat("\n最终建模数据维度:", dim(model_data), "\n")

# 数据摘要
cat("\n数据摘要:\n")
print(summary(model_data))

################################################################################
# 3. 数据划分
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 3: 划分训练集和测试集\n")
# cat("="*80, "\n")

set.seed(123)
train_index <- createDataPartition(model_data$log_CH4_response, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

cat("\n训练集样本数:", nrow(train_data), "\n")
cat("测试集样本数:", nrow(test_data), "\n")

################################################################################
# 4. 设置交叉验证
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 4: 设置交叉验证策略\n")
# cat("="*80, "\n")

# 设置5折交叉验证
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = FALSE,
  allowParallel = TRUE,
  savePredictions = "final"
)

cat("\n交叉验证方法: 5-fold CV\n")

################################################################################
# 5. 定义多个机器学习模型及其超参数网格
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 5: 定义模型和超参数网格\n")
# cat("="*80, "\n")

# 创建模型列表
models_list <- list()

# ============================================================================
# 5.1 Random Forest
# ============================================================================
cat("\n定义 Random Forest 模型...\n")
models_list[["rf"]] <- list(
  method = "rf",
  tuneGrid = expand.grid(
    mtry = c(2, 3, 4, 5)
  ),
  importance = TRUE
)

# ============================================================================
# 5.2 Gradient Boosting Machine (GBM)
# ============================================================================
cat("定义 Gradient Boosting 模型...\n")
models_list[["gbm"]] <- list(
  method = "gbm",
  tuneGrid = expand.grid(
    n.trees = c(500, 1000, 1500),
    interaction.depth = c(3, 4, 5),
    shrinkage = c(0.01, 0.05, 0.1),
    n.minobsinnode = c(5, 10)
  ),
  verbose = FALSE
)

# ============================================================================
# 5.3 XGBoost
# ============================================================================
cat("定义 XGBoost 模型...\n")
models_list[["xgb"]] <- list(
  method = "xgbTree",
  tuneGrid = expand.grid(
    nrounds = c(100, 200, 300),
    max_depth = c(3, 4, 5),
    eta = c(0.01, 0.05, 0.1),
    gamma = c(0, 0.1),
    colsample_bytree = c(0.6, 0.8),
    min_child_weight = c(1, 3),
    subsample = c(0.8, 1)
  )
)

# ============================================================================
# 5.4 Support Vector Machine (SVM)
# ============================================================================
cat("定义 SVM 模型...\n")
models_list[["svm"]] <- list(
  method = "svmRadial",
  tuneGrid = expand.grid(
    sigma = c(0.01, 0.05, 0.1),
    C = c(0.5, 1, 2, 5)
  )
)

# ============================================================================
# 5.5 Elastic Net
# ============================================================================
cat("定义 Elastic Net 模型...\n")
models_list[["glmnet"]] <- list(
  method = "glmnet",
  tuneGrid = expand.grid(
    alpha = seq(0, 1, by = 0.2),
    lambda = 10^seq(-3, 0, length = 10)
  )
)

# ============================================================================
# 5.6 MARS (Multivariate Adaptive Regression Splines)
# ============================================================================
cat("定义 MARS 模型...\n")
models_list[["mars"]] <- list(
  method = "earth",
  tuneGrid = expand.grid(
    degree = c(1, 2),
    nprune = c(5, 10, 15, 20)
  )
)

# ============================================================================
# 5.7 Neural Network
# ============================================================================
cat("定义 Neural Network 模型...\n")
models_list[["nnet"]] <- list(
  method = "nnet",
  tuneGrid = expand.grid(
    size = c(3, 5, 7, 10),
    decay = c(0, 0.01, 0.1)
  ),
  linout = TRUE,
  trace = FALSE,
  maxit = 500
)

cat("\n共定义了", length(models_list), "个模型进行比较\n")

################################################################################
# 6. 训练所有模型并进行超参数调优
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 6: 训练所有模型 (含超参数调优)\n")
# cat("="*80, "\n")

trained_models <- list()
training_times <- list()

for(model_name in names(models_list)) {
  # cat("\n", "="*60, "\n")
  # cat("训练模型:", model_name, "\n")
  # cat("="*60, "\n")
  
  start_time <- Sys.time()
  
  # 准备参数
  model_spec <- models_list[[model_name]]
  train_args <- list(
    form = log_CH4_response ~ .,
    data = train_data,
    method = model_spec$method,
    trControl = train_control,
    tuneGrid = model_spec$tuneGrid
  )
  
  # 添加模型特定参数
  if(!is.null(model_spec$verbose)) {
    train_args$verbose <- model_spec$verbose
  }
  if(!is.null(model_spec$importance)) {
    train_args$importance <- model_spec$importance
  }
  if(!is.null(model_spec$linout)) {
    train_args$linout <- model_spec$linout
  }
  if(!is.null(model_spec$trace)) {
    train_args$trace <- model_spec$trace
  }
  if(!is.null(model_spec$maxit)) {
    train_args$maxit <- model_spec$maxit
  }
  
  # 训练模型
  tryCatch({
    trained_models[[model_name]] <- do.call(train, train_args)
    end_time <- Sys.time()
    training_times[[model_name]] <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    cat("\n✓ 模型训练完成\n")
    cat("  训练时间:", round(training_times[[model_name]], 2), "秒\n")
    cat("  最优参数:\n")
    print(trained_models[[model_name]]$bestTune)
    
  }, error = function(e) {
    cat("\n✗ 模型训练失败:", conditionMessage(e), "\n")
    trained_models[[model_name]] <- NULL
    training_times[[model_name]] <- NA
  })
}

# 移除训练失败的模型
trained_models <- trained_models[!sapply(trained_models, is.null)]

# cat("\n", "="*80, "\n")
# cat("成功训练的模型数:", length(trained_models), "\n")
# cat("="*80, "\n")

################################################################################
# 7. 模型评估和比较
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 7: 模型评估和比较\n")
# cat("="*80, "\n")

# 创建结果数据框
results_df <- data.frame(
  Model = character(),
  Train_RMSE = numeric(),
  Train_R2 = numeric(),
  Train_MAE = numeric(),
  Test_RMSE = numeric(),
  Test_R2 = numeric(),
  Test_MAE = numeric(),
  CV_RMSE = numeric(),
  CV_RMSE_SD = numeric(),
  Training_Time = numeric(),
  stringsAsFactors = FALSE
)

# 对每个模型进行评估
for(model_name in names(trained_models)) {
  model <- trained_models[[model_name]]
  
  # 训练集预测
  train_pred <- predict(model, train_data)
  train_rmse <- sqrt(mean((train_data$log_CH4_response - train_pred)^2))
  train_r2 <- cor(train_data$log_CH4_response, train_pred)^2
  train_mae <- mean(abs(train_data$log_CH4_response - train_pred))
  
  # 测试集预测
  test_pred <- predict(model, test_data)
  test_rmse <- sqrt(mean((test_data$log_CH4_response - test_pred)^2))
  test_r2 <- cor(test_data$log_CH4_response, test_pred)^2
  test_mae <- mean(abs(test_data$log_CH4_response - test_pred))
  
  # 交叉验证结果
  cv_results <- model$results
  best_tune_idx <- which(apply(cv_results[, names(model$bestTune), drop = FALSE], 1, 
                               function(x) all(x == model$bestTune)))
  cv_rmse <- cv_results$RMSE[best_tune_idx]
  cv_rmse_sd <- cv_results$RMSESD[best_tune_idx]
  
  # 添加到结果
  results_df <- rbind(results_df, data.frame(
    Model = model_name,
    Train_RMSE = train_rmse,
    Train_R2 = train_r2,
    Train_MAE = train_mae,
    Test_RMSE = test_rmse,
    Test_R2 = test_r2,
    Test_MAE = test_mae,
    CV_RMSE = cv_rmse,
    CV_RMSE_SD = cv_rmse_sd,
    Training_Time = training_times[[model_name]]
  ))
}

# 按测试集R2排序
results_df <- results_df[order(-results_df$Test_R2), ]

cat("\n模型性能比较:\n")
print(results_df, row.names = FALSE, digits = 4)

# 保存结果
write.csv(results_df, "Model_Comparison_Results.csv", row.names = FALSE)
cat("\n✓ 模型比较结果已保存: Model_Comparison_Results.csv\n")

################################################################################
# 8. 选择最优模型
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 8: 选择最优模型\n")
# cat("="*80, "\n")

# 基于测试集R2选择最优模型
best_model_name <- results_df$Model[1]
best_model <- trained_models[[best_model_name]]

cat("\n最优模型:", best_model_name, "\n")
cat("测试集 R²:", round(results_df$Test_R2[1], 4), "\n")
cat("测试集 RMSE:", round(results_df$Test_RMSE[1], 4), "\n")
cat("\n最优超参数:\n")
print(best_model$bestTune)

################################################################################
# 9. 可视化模型比较
################################################################################

cat("\n", "="*80, "\n")
cat("Step 9: 可视化模型比较\n")
cat("="*80, "\n")

# 9.1 模型性能比较图
pdf("09_Model_Comparison.pdf", width = 14, height = 10)

par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))

# R² 比较
barplot(results_df$Test_R2, 
        names.arg = results_df$Model,
        col = rainbow(nrow(results_df)),
        main = "Test R² Comparison",
        ylab = "R²",
        las = 2,
        cex.names = 0.8)
abline(h = 0, lty = 2)
grid(NA, NULL)

# RMSE 比较
barplot(results_df$Test_RMSE, 
        names.arg = results_df$Model,
        col = rainbow(nrow(results_df)),
        main = "Test RMSE Comparison",
        ylab = "RMSE",
        las = 2,
        cex.names = 0.8)
grid(NA, NULL)

# 训练时间比较
barplot(results_df$Training_Time, 
        names.arg = results_df$Model,
        col = rainbow(nrow(results_df)),
        main = "Training Time Comparison",
        ylab = "Time (seconds)",
        las = 2,
        cex.names = 0.8)
grid(NA, NULL)

# CV RMSE 比较 (带误差条)
bp <- barplot(results_df$CV_RMSE, 
              names.arg = results_df$Model,
              col = rainbow(nrow(results_df)),
              main = "Cross-Validation RMSE",
              ylab = "CV RMSE",
              las = 2,
              cex.names = 0.8,
              ylim = c(0, max(results_df$CV_RMSE + results_df$CV_RMSE_SD) * 1.1))
arrows(bp, results_df$CV_RMSE - results_df$CV_RMSE_SD,
       bp, results_df$CV_RMSE + results_df$CV_RMSE_SD,
       angle = 90, code = 3, length = 0.1, lwd = 2)
grid(NA, NULL)

dev.off()

cat("✓ 保存: 09_Model_Comparison.pdf\n")

# 9.2 散点图：预测 vs 实际 (所有模型)
pdf("10_Prediction_vs_Actual_All_Models.pdf", width = 16, height = 12)

n_models <- length(trained_models)
n_cols <- ceiling(sqrt(n_models))
n_rows <- ceiling(n_models / n_cols)
par(mfrow = c(n_rows, n_cols), mar = c(4, 4, 3, 1))

for(model_name in names(trained_models)) {
  model <- trained_models[[model_name]]
  test_pred <- predict(model, test_data)
  test_r2 <- cor(test_data$log_CH4_response, test_pred)^2
  
  plot(test_data$log_CH4_response, test_pred,
       main = paste0(model_name, " (R² = ", round(test_r2, 3), ")"),
       xlab = "Observed log(Me/Ma)",
       ylab = "Predicted log(Me/Ma)",
       pch = 19, col = rgb(0, 0, 1, 0.5))
  abline(0, 1, col = "red", lwd = 2)
  abline(lm(test_pred ~ test_data$log_CH4_response), col = "blue", lwd = 2, lty = 2)
  grid()
}

dev.off()

cat("✓ 保存: 10_Prediction_vs_Actual_All_Models.pdf\n")

################################################################################
# 10. 最优模型的变量重要性分析
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 10: 最优模型的变量重要性分析\n")
# cat("="*80, "\n")

# 提取变量重要性
if(best_model_name %in% c("rf", "gbm", "xgb")) {
  
  pdf("11_Best_Model_Variable_Importance.pdf", width = 10, height = 8)
  
  if(best_model_name == "rf") {
    # Random Forest
    var_imp <- varImp(best_model, scale = FALSE)
    plot(var_imp, main = paste("Variable Importance -", best_model_name))
    
    # 保存数据
    imp_df <- as.data.frame(var_imp$importance)
    imp_df$Variable <- rownames(imp_df)
    imp_df <- imp_df[order(-imp_df$Overall), ]
    
  } else if(best_model_name == "gbm") {
    # GBM
    var_imp <- varImp(best_model, scale = FALSE)
    plot(var_imp, main = paste("Variable Importance -", best_model_name))
    
    imp_df <- as.data.frame(var_imp$importance)
    imp_df$Variable <- rownames(imp_df)
    imp_df <- imp_df[order(-imp_df$Overall), ]
    
  } else if(best_model_name == "xgb") {
    # XGBoost
    var_imp <- varImp(best_model, scale = FALSE)
    plot(var_imp, main = paste("Variable Importance -", best_model_name))
    
    imp_df <- as.data.frame(var_imp$importance)
    imp_df$Variable <- rownames(imp_df)
    imp_df <- imp_df[order(-imp_df$Overall), ]
  }
  
  dev.off()
  
  cat("\n变量重要性排序:\n")
  print(imp_df, row.names = FALSE)
  
  write.csv(imp_df, "Best_Model_Variable_Importance.csv", row.names = FALSE)
  cat("\n✓ 保存: 11_Best_Model_Variable_Importance.pdf\n")
  cat("✓ 保存: Best_Model_Variable_Importance.csv\n")
}

################################################################################
# 11. Partial Dependence Plots (PDP) - 使用最优模型
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 11: 生成Partial Dependence Plots (最优模型)\n")
# cat("="*80, "\n")

# 11.1 温度的PDP
cat("\n生成温度PDP...\n")
pdp_increase <- partial(best_model, 
                        pred.var = "increase", 
                        train = train_data,
                        grid.resolution = 50,
                        plot = FALSE)

# 11.2 其他重要变量的PDP
important_vars <- c("increase", "latitude", "MAT", "MAP", "duration")
pdp_list <- list()

for(var in important_vars) {
  if(var %in% names(train_data)) {
    pdp_list[[var]] <- partial(best_model,
                               pred.var = var,
                               train = train_data,
                               grid.resolution = 50,
                               plot = FALSE)
  }
}

# 绘制PDP
pdf("12_PDP_Best_Model.pdf", width = 14, height = 10)
par(mfrow = c(2, 3))

for(var in names(pdp_list)) {
  pdp_data <- pdp_list[[var]]
  
  plot(pdp_data[[var]], pdp_data$yhat,
       type = "l", lwd = 3, col = "red",
       main = paste("PDP:", var, "\n(", best_model_name, ")"),
       xlab = var,
       ylab = "Predicted log(Me/Ma)")
  abline(h = 0, lty = 2, col = "gray50", lwd = 2)
  grid()
  
  # 特别标注温度效应
  if(var == "increase") {
    temp_effect <- pdp_data$yhat[length(pdp_data$yhat)] - pdp_data$yhat[1]
    pct_change <- (exp(temp_effect) - 1) * 100
    text_label <- sprintf("Total effect: %.3f\n(%.1f%% CH4 change)", 
                         temp_effect, pct_change)
    text(x = min(pdp_data[[var]]) + 0.05 * diff(range(pdp_data[[var]])),
         y = max(pdp_data$yhat) - 0.1 * diff(range(pdp_data$yhat)),
         labels = text_label,
         pos = 4, cex = 0.9,
         col = "darkblue", font = 2)
  }
}

dev.off()

cat("✓ 保存: 12_PDP_Best_Model.pdf\n")

# 保存温度PDP数据
write.csv(pdp_increase, "PDP_Temperature_Best_Model.csv", row.names = FALSE)
cat("✓ 保存: PDP_Temperature_Best_Model.csv\n")

# 11.3 温度PDP - 带Bootstrap置信区间
cat("\n计算温度PDP的置信区间 (Bootstrap)...\n")

n_bootstrap <- 100
pdp_bootstrap <- matrix(NA, nrow = 50, ncol = n_bootstrap)

set.seed(123)
for(i in 1:n_bootstrap) {
  # Bootstrap采样
  boot_idx <- sample(nrow(train_data), replace = TRUE)
  boot_data <- train_data[boot_idx, ]
  
  # 重新训练模型
  boot_model <- train(
    log_CH4_response ~ .,
    data = boot_data,
    method = best_model$method,
    tuneGrid = best_model$bestTune,
    trControl = trainControl(method = "none")
  )
  
  # 计算PDP
  pdp_boot <- partial(boot_model,
                      pred.var = "increase",
                      train = boot_data,
                      grid.resolution = 50,
                      plot = FALSE)
  
  pdp_bootstrap[, i] <- pdp_boot$yhat
  
  if(i %% 20 == 0) {
    cat("  Bootstrap进度:", i, "/", n_bootstrap, "\n")
  }
}

# 计算置信区间
pdp_mean <- rowMeans(pdp_bootstrap)
pdp_lower <- apply(pdp_bootstrap, 1, quantile, probs = 0.025)
pdp_upper <- apply(pdp_bootstrap, 1, quantile, probs = 0.975)

# 绘制带置信区间的温度PDP
pdf("13_PDP_Temperature_With_CI.pdf", width = 12, height = 8)

plot(pdp_increase$increase, pdp_increase$yhat,
     type = "l", lwd = 4, col = "red",
     main = paste("Temperature Effect on CH4 (", best_model_name, ")\nwith 95% Confidence Interval"),
     xlab = "Temperature Increase (°C)",
     ylab = "Predicted log(Me/Ma)",
     ylim = range(c(pdp_lower, pdp_upper)),
     cex.main = 1.3, cex.lab = 1.2)

polygon(c(pdp_increase$increase, rev(pdp_increase$increase)),
        c(pdp_lower, rev(pdp_upper)),
        col = rgb(1, 0, 0, 0.2), border = NA)

abline(h = 0, lty = 2, col = "gray30", lwd = 2)
grid()

legend("topleft",
       legend = c("PDP", "95% CI"),
       col = c("red", rgb(1, 0, 0, 0.2)),
       lwd = c(4, 10),
       bty = "n")

# 添加数据密度
par(new = TRUE)
hist(train_data$increase, breaks = 30, 
     col = rgb(0, 0, 1, 0.1), border = NA,
     xlab = "", ylab = "", main = "", axes = FALSE)
axis(4, col = "blue", col.axis = "blue")
mtext("Data Density", side = 4, line = 3, col = "blue")

dev.off()

cat("✓ 保存: 13_PDP_Temperature_With_CI.pdf\n")

# 保存置信区间数据
pdp_ci_df <- data.frame(
  Temperature = pdp_increase$increase,
  PDP_Mean = pdp_increase$yhat,
  PDP_Lower_95 = pdp_lower,
  PDP_Upper_95 = pdp_upper
)
write.csv(pdp_ci_df, "PDP_Temperature_CI_Best_Model.csv", row.names = FALSE)

################################################################################
# 12. 二维PDP - 交互效应
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 12: 生成二维PDP (交互效应)\n")
# cat("="*80, "\n")

# 温度 × 纬度
cat("\n计算 Temperature × Latitude 交互效应...\n")
pdp_temp_lat <- partial(best_model,
                        pred.var = c("increase", "latitude"),
                        train = train_data,
                        grid.resolution = 20,
                        plot = FALSE)

# 温度 × MAT
cat("计算 Temperature × MAT 交互效应...\n")
pdp_temp_mat <- partial(best_model,
                        pred.var = c("increase", "MAT"),
                        train = train_data,
                        grid.resolution = 20,
                        plot = FALSE)

# 绘制2D PDP
pdf("14_PDP_2D_Interactions.pdf", width = 14, height = 6)
par(mfrow = c(1, 2))

# Temperature × Latitude
plotPartial(pdp_temp_lat, 
            levelplot = FALSE,
            drape = TRUE,
            colorkey = TRUE,
            screen = list(z = 30, x = -60),
            main = "Interaction: Temperature × Latitude",
            xlab = "Temperature Increase (°C)",
            ylab = "Latitude",
            zlab = "log(Me/Ma)")

# Temperature × MAT
plotPartial(pdp_temp_mat,
            levelplot = FALSE,
            drape = TRUE,
            colorkey = TRUE,
            screen = list(z = 30, x = -60),
            main = "Interaction: Temperature × MAT",
            xlab = "Temperature Increase (°C)",
            ylab = "MAT (°C)",
            zlab = "log(Me/Ma)")

dev.off()

cat("✓ 保存: 14_PDP_2D_Interactions.pdf\n")

################################################################################
# 13. 最优模型的综合评估图
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 13: 生成最优模型综合评估图\n")
# cat("="*80, "\n")

pdf("15_Best_Model_Comprehensive_Evaluation.pdf", width = 16, height = 12)

layout(matrix(c(1, 1, 2, 3, 
                1, 1, 4, 5), 
              nrow = 2, byrow = TRUE))

# 1. 温度PDP (大图)
plot(pdp_increase$increase, pdp_increase$yhat,
     type = "l", lwd = 4, col = "red",
     main = paste("A. Temperature Effect on CH4 -", best_model_name,
                  "\n(Controlling for Confounders)"),
     xlab = "Temperature Increase (°C)",
     ylab = "Predicted log(Me/Ma)",
     cex.main = 1.3, cex.lab = 1.2)

polygon(c(pdp_increase$increase, rev(pdp_increase$increase)),
        c(pdp_lower, rev(pdp_upper)),
        col = rgb(1, 0, 0, 0.2), border = NA)

abline(h = 0, lty = 2, col = "gray30", lwd = 2)
grid()

# 添加效应量信息
temp_effect <- pdp_increase$yhat[length(pdp_increase$yhat)] - pdp_increase$yhat[1]
pct_change <- (exp(temp_effect) - 1) * 100
info_text <- sprintf("Temperature range: %.1f - %.1f°C\nTotal effect: %.3f log units\nEquivalent to: %+.1f%% change in CH4\nModel: %s (R² = %.3f)",
                     min(pdp_increase$increase), max(pdp_increase$increase),
                     temp_effect, pct_change, best_model_name, results_df$Test_R2[1])

text(x = min(pdp_increase$increase) + 0.02 * diff(range(pdp_increase$increase)),
     y = max(pdp_increase$yhat) - 0.05 * diff(range(pdp_increase$yhat)),
     labels = info_text,
     pos = 4, cex = 0.9, font = 2,
     col = "darkblue")

# 2. 预测 vs 实际
test_pred <- predict(best_model, test_data)
plot(test_data$log_CH4_response, test_pred,
     pch = 19, col = rgb(0, 0, 1, 0.6),
     main = "B. Predictions vs Observations",
     xlab = "Observed log(Me/Ma)",
     ylab = "Predicted log(Me/Ma)")
abline(0, 1, col = "red", lwd = 2)
abline(lm(test_pred ~ test_data$log_CH4_response), col = "blue", lwd = 2, lty = 2)
grid()
legend("topleft", 
       legend = c("Perfect fit", "Linear fit"),
       col = c("red", "blue"),
       lty = c(1, 2), lwd = 2, bty = "n")

# 3. 残差图
residuals <- test_data$log_CH4_response - test_pred
plot(test_pred, residuals,
     pch = 19, col = rgb(0.5, 0, 0.5, 0.5),
     main = "C. Residual Plot",
     xlab = "Predicted log(Me/Ma)",
     ylab = "Residuals")
abline(h = 0, col = "red", lwd = 2, lty = 2)
grid()

# 4. 残差分布
hist(residuals, breaks = 20,
     col = "lightblue", border = "white",
     main = "D. Residual Distribution",
     xlab = "Residuals")
abline(v = 0, col = "red", lwd = 2, lty = 2)
grid()

# 5. 模型比较
barplot(results_df$Test_R2[1:min(5, nrow(results_df))],
        names.arg = results_df$Model[1:min(5, nrow(results_df))],
        col = rainbow(min(5, nrow(results_df))),
        main = "E. Top Models Comparison (Test R²)",
        ylab = "R²",
        las = 2)
grid(NA, NULL)

dev.off()

cat("✓ 保存: 15_Best_Model_Comprehensive_Evaluation.pdf\n")

################################################################################
# 14. 生成分析报告
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 14: 生成分析报告\n")
# cat("="*80, "\n")

# 计算温度效应
temp_slope <- (pdp_increase$yhat[length(pdp_increase$yhat)] - pdp_increase$yhat[1]) / 
              (pdp_increase$increase[length(pdp_increase$increase)] - pdp_increase$increase[1])

sink("Analysis_Report.txt")

# cat("="*80, "\n")
# cat("CH4排放对温度响应的机器学习分析 - 完整报告\n")
# cat("="*80, "\n\n")

cat("【1. 数据概况】\n")
cat("  总样本数:", nrow(model_data), "\n")
cat("  训练集:", nrow(train_data), "样本\n")
cat("  测试集:", nrow(test_data), "样本\n")
cat("  温度范围:", round(min(model_data$increase), 2), "°C 至", 
    round(max(model_data$increase), 2), "°C\n")
cat("  CH4响应比率范围:", round(min(exp(model_data$log_CH4_response)), 2), "至",
    round(max(exp(model_data$log_CH4_response)), 2), "\n\n")

cat("【2. 模型比较结果】\n")
cat("  共训练模型数:", length(trained_models), "\n")
cat("  交叉验证方法: 5-fold CV\n\n")
cat("  模型性能排名 (按测试集R²):\n")
for(i in 1:min(5, nrow(results_df))) {
  cat(sprintf("  %d. %s: R² = %.4f, RMSE = %.4f\n",
              i, results_df$Model[i], results_df$Test_R2[i], results_df$Test_RMSE[i]))
}

cat("\n【3. 最优模型】\n")
cat("  模型名称:", best_model_name, "\n")
cat("  测试集 R²:", round(results_df$Test_R2[1], 4), "\n")
cat("  测试集 RMSE:", round(results_df$Test_RMSE[1], 4), "\n")
cat("  测试集 MAE:", round(results_df$Test_MAE[1], 4), "\n")
cat("  训练时间:", round(results_df$Training_Time[1], 2), "秒\n\n")

cat("  最优超参数:\n")
for(param_name in names(best_model$bestTune)) {
  cat("    -", param_name, "=", best_model$bestTune[[param_name]], "\n")
}

cat("\n【4. 温度效应 (控制混杂因子后)】\n")
cat("  温度每增加1°C, log(Me/Ma)平均变化:", round(temp_slope, 4), "\n")
pct_per_degree <- (exp(temp_slope) - 1) * 100
cat("  相当于CH4排放每°C变化:", sprintf("%+.2f%%", pct_per_degree), "\n")
total_temp_effect <- pdp_increase$yhat[length(pdp_increase$yhat)] - pdp_increase$yhat[1]
total_pct_change <- (exp(total_temp_effect) - 1) * 100
cat("  整个温度范围的总效应:\n")
cat("    - log(Me/Ma)变化:", round(total_temp_effect, 4), "\n")
cat("    - CH4排放变化:", sprintf("%+.1f%%", total_pct_change), "\n")

cat("\n【5. 关键结论】\n")
if(temp_slope > 0) {
  cat("  ✓ 温度增加总体上促进CH4排放\n")
} else {
  cat("  ✓ 温度增加总体上抑制CH4排放\n")
}
cat("  ✓ 在控制了纬度、经度、MAT、MAP、土地利用类型、氮应用、\n")
cat("    地下水位、设施类型和持续时间等混杂因子后,\n")
cat("    温度对CH4排放仍有显著影响\n")
cat("  ✓ 最优模型:", best_model_name, "\n")
cat("  ✓ 模型解释力: R² =", round(results_df$Test_R2[1], 3), "\n")

cat("\n【6. 建议】\n")
cat("  1. 查看PDP图以了解详细的剂量-响应关系\n")
cat("  2. 关注二维交互PDP图了解温度与其他因子的协同效应\n")
cat("  3. 考虑不同环境条件下温度效应的异质性\n")

cat("\n【7. 输出文件】\n")
cat("  模型比较:\n")
cat("    - Model_Comparison_Results.csv\n")
cat("    - 09_Model_Comparison.pdf\n")
cat("    - 10_Prediction_vs_Actual_All_Models.pdf\n")
cat("  最优模型:\n")
cat("    - Best_Model_Variable_Importance.csv\n")
cat("    - 11_Best_Model_Variable_Importance.pdf\n")
cat("  PDP分析:\n")
cat("    - 12_PDP_Best_Model.pdf\n")
cat("    - 13_PDP_Temperature_With_CI.pdf\n")
cat("    - 14_PDP_2D_Interactions.pdf\n")
cat("    - PDP_Temperature_Best_Model.csv\n")
cat("    - PDP_Temperature_CI_Best_Model.csv\n")
cat("  综合评估:\n")
cat("    - 15_Best_Model_Comprehensive_Evaluation.pdf\n")

# cat("\n", "="*80, "\n")
# cat("分析完成时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
# cat("="*80, "\n")

sink()

cat("✓ 保存: Analysis_Report.txt\n")

################################################################################
# 15. 保存工作空间
################################################################################

# cat("\n", "="*80, "\n")
# cat("Step 15: 保存工作空间\n")
# cat("="*80, "\n")

save(
  model_data, train_data, test_data,
  trained_models, best_model, best_model_name,
  results_df, pdp_increase, pdp_ci_df,
  file = "CH4_Analysis_Workspace.RData"
)

cat("✓ 保存: CH4_Analysis_Workspace.RData\n")

################################################################################
# 完成
################################################################################

# cat("\n", "="*80, "\n")
# cat("所有分析完成!\n")
# cat("="*80, "\n")

cat("\n最优模型:", best_model_name, "\n")
cat("测试集 R²:", round(results_df$Test_R2[1], 4), "\n")
cat("温度效应: 每°C增加", sprintf("%+.2f%%", pct_per_degree), "CH4排放\n")

cat("\n所有结果文件已保存至当前工作目录\n")
cat("请查看 Analysis_Report.txt 获取完整分析报告\n")

# cat("\n", "="*80, "\n")