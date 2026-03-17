# TemperatureResponseYield

本仓库用于评估增温对稻田（paddy）与传统湿地（wetland）甲烷排放响应（LnRR）的线性/非线性关系、阈值特征与空间外推。

## 数据

- `paddy.xlsx`：稻田观测数据。
- `wetland.xlsx`：湿地观测数据。

核心字段：`Ma/SDa/Na`、`Me/SDe/Ne`、经纬度、气候变量、管理变量、增温变量等。

## 代码结构

- `analysis_pipeline.R`：统一工作流函数（数据增强、Meta 分析、多模型建模、阈值识别、空间预测）。
- `CH4 response to temperature claude paddy.r`：调用统一流程分析稻田。
- `CH4 response to temperature claude wetland.r`：调用统一流程分析湿地。

## 工作流内容（对应项目部署）

1. **土壤属性补充**：基于经纬度从公开全球土壤接口（`geodata::soil_world`，SoilGrids/HWSD 同类数据源）提取黏粒、SOC、TN、pH，并新增 SOM 与分级变量。
2. **数据探索**：输出缺失值统计、LnRR 分布图、正态性检验。
3. **传统 Meta 分析**：计算 LnRR (`yi`) 与采样方差 (`vi`)，做总体效应与分组调节变量分析（稻田/湿地分别用各自管理变量）。
4. **预测模型构建与可解释性分析**：
   - 线性：`lm`
   - 混合线性：`lmer`
   - 混合非线性：`gam`
   - 贝叶斯：`brms`（可选，环境支持时自动启用）
   - 机器学习：`ranger`、`xgboost`
   - 深度学习近似：`nnet`
   统一交叉验证与测试集评估（RMSE/R²/MAE），并导出最优模型。机器学习与深度学习模型采用显式超参数网格搜索（RF/XGBoost/NN/GAM）。
   最优模型阶段输出变量重要性、偏依赖关系（PDP）；在 `fastshap` 可用时，额外输出 SHAP 值及 SHAP 重要性。
5. **阈值识别**：通过分段拟合网格搜索估计增温响应拐点。
6. **中国区域空间预测**：使用 WorldClim 当前气候和 +3°C 情景生成中国域 LnRR 当前、未来和差值栅格（GeoTIFF）。

## 运行方法

在 R 环境中执行：

```r
source("CH4 response to temperature claude paddy.r")
source("CH4 response to temperature claude wetland.r")
```

## 输出目录

执行后会生成：

- `outputs/paddy/...`
- `outputs/wetland/...`

每个生态系统包含：
- `*_with_soil.csv`：补充土壤属性后的数据。
- `*_effectsize.csv`：LnRR 与方差。
- `01_data_quality/`：缺失统计、分布图、正态性检验。
- `02_meta/`：总体与分组 Meta 结果。
- `03_models/`：模型评估、最优模型、最优超参数（`best_hyperparameters.csv`）、变量重要性、偏依赖结果（`best_model_partial_dependence.*`）、SHAP 结果（可选）、增温响应曲线、阈值估计。
- `04_spatial/`：中国范围当前/未来/差值 GeoTIFF。

## 说明

- `brms`、`vip`、`fastshap` 被设为可选依赖；环境缺失时自动跳过，不影响主流程。
- 空间预测中若某些因子型变量缺少全国覆盖栅格，当前版本采用常量占位，建议后续替换为真实土地利用/管理情景图层。
