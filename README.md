# 电信客户流失预测与留存策略分析

> 基于 IBM Telco Customer Churn 公开数据集，完成从数据清洗、SQL 分组分析、Python EDA、流失预测建模、阈值与成本敏感分析，到优先挽留名单、ROI 测算和挽留策略 A/B 验证的完整数据分析项目。  
> 项目目标：识别高流失风险客户，解释流失关键因素，输出可落地的客户挽留策略，并量化潜在收益。

---

## 一、项目背景

电信、金融科技、SaaS 等行业中，客户流失直接侵蚀收入。相比事后挽回，提前识别高风险客户并提前干预，能显著降低获客成本、提升 LTV。

本项目重点回答：

1. 总体流失率是多少？
2. 哪些合同类型、在网时长、月费区间的客户更容易流失？
3. 互联网服务、支付方式、在线安全、技术支持如何影响流失？
4. 逻辑回归和随机森林哪个更适合流失预测？
5. 哪些特征对流失影响最大？
6. 如何选择预测阈值，使挽留成本最低？
7. 哪些客户应优先挽留？潜在 ROI 是多少？
8. 如果发券挽留，A/B 实验能否验证效果？

---

## 二、数据集与字段说明

- 数据集：IBM Telco Customer Churn
- 下载地址：https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- 数据量：7,043 条客户记录，21 个特征
- 目标变量：`Churn`（Yes / No）
- 主要字段：

| 字段类型   | 字段                                                         |
| ---------- | ------------------------------------------------------------ |
| 客户 ID    | customerID                                                   |
| 人口统计   | gender, SeniorCitizen, Partner, Dependents                   |
| 服务订阅   | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| 合同与账单 | Contract, PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges |
| 标签       | Churn                                                        |

数据清洗要点：

- `TotalCharges` 原为字符串，存在空格；
- 转换为数值后产生缺失值，共 11 个；
- 这些通常是 `tenure=0` 的新客户，缺失值填充为 0；
- 清洗后数据保存为 `telco_clean.csv`，并导入 MySQL 表 `customer_churn`。

---

## 三、技术栈

| 模块     | 工具                                                |
| -------- | --------------------------------------------------- |
| 数据存储 | MySQL                                               |
| 数据清洗 | Pandas                                              |
| SQL 分析 | GROUP BY、CASE WHEN、聚合、交叉分析                 |
| 可视化   | Matplotlib、Seaborn                                 |
| 建模     | Scikit-learn：逻辑回归、随机森林                    |
| 模型评估 | AUC、Accuracy、Precision、Recall、F1、混淆矩阵、ROC |
| 阈值分析 | Precision / Recall / F1 / 成本敏感分析              |
| A/B 实验 | 样本量计算、卡方检验、单侧 Z 检验、置信区间、ROI    |

---

## 四、目录结构

```text
telco-churn-analysis/
├── images/
│   └── churn_count.png
│   └── contract_churn.png
│   └── tenure_churn.png
│   └── monthly_churn.png
│   └── corr_heatmap.png
│   └── InternetService_churn.png
│   └── PaymentMethod_churn.png
│   └── OnlineSecurity_churn.png
│   └── TechSupport_churn.png
│   └── roc_curve.png
│   └── lr_coef.png
│   └── rf_importance.png
│   └── threshold_analysis.png
├── outputs/
│   └── model_results.json
│   └── priority_customers.csv
│   └── roi_estimate.json
│   └── ab_test/
│       ├── ab_test_results.json
│       ├── ab_test_results.png
│       └── experiment_groups.csv
├── sql/
│   └── telco-churn-analysis.sql
├── python/
│   ├── step1_prepare_data.py
│   ├── step2_eda.py
│   ├── step3_model.py
│   ├── step4_roi.py
│   └── step5_ab_test.py
└── README.md
```

---

## 五、分析框架

### 5.1 数据清洗

- 读取原始 CSV；
- `TotalCharges` 转数值，缺失填 0；
- 保存清洗后 CSV；
- 导入 MySQL 表 `customer_churn`。

### 5.2 SQL 分析

`telco-churn-analysis.sql` 包含：

1. 总体流失率；
2. 按合同类型统计流失率；
3. 按月费区间统计流失率；
4. 按在网时长统计流失率；
5. 按互联网服务 + 支付方式交叉统计流失率；
6. 客户级特征宽表 `customer_features`；
7. 多维度交叉流失率分析。

### 5.3 Python EDA

`step2_eda.py` 包含：

- 流失 / 非流失客户数量；
- 合同类型 vs 流失率；
- 在网时长 vs 流失；
- 月费 vs 流失；
- 数值变量相关性热力图；
- 互联网服务、支付方式、在线安全、技术支持流失率。

### 5.4 建模与解释

`step3_model.py` 包含：

- 逻辑回归 + 随机森林；
- 数值标准化 + 类别独热编码；
- 训练集 / 测试集 80/20，分层抽样；
- AUC、Accuracy、Precision、Recall、F1、混淆矩阵；
- ROC 曲线对比；
- 逻辑回归特征系数 Top15；
- 随机森林特征重要性 Top15；
- 阈值分析：Precision / Recall / F1 / 总成本；
- 输出高风险 + 高价值优先挽留客户名单。

### 5.5 ROI 测算

`step4_roi.py` 包含：

- 优先挽留客户数；
- 假设挽留成功率、多留存月数；
- 平均月费；
- 年化潜在留存收益。

### 5.6 A/B 验证

`step5_ab_test.py` 包含：

- 实验背景与统计假设；
- 样本量计算；
- 基于优先挽留名单随机分组；
- 均衡性检验；
- 模拟发券干预；
- 卡方检验、单侧 Z 检验、95% 置信区间；
- 业务收益测算；
- 输出 JSON、CSV、可视化。

---

## 六、核心结论

### 6.1 总体流失情况

<img src="images\churn_count.png" alt="churn_count" style="zoom:50%;" />

| 指标         |   数值 |
| ------------ | -----: |
| 总客户数     |  7,043 |
| 流失客户数   |  1,869 |
| 非流失客户数 |  5,174 |
| 总体流失率   | 26.54% |

结论：约每 4 个客户中有 1 个流失，流失问题需要重点治理。

---

### 6.2 合同类型流失率

<img src="images\contract_churn.png" alt="contract_churn" style="zoom:50%;" />

| 合同类型       | 客户数 | 流失数 | 流失率 |
| -------------- | -----: | -----: | -----: |
| Month-to-month |  3,875 |  1,655 | 42.71% |
| One year       |  1,473 |    166 | 11.27% |
| Two year       |  1,695 |     48 |  2.83% |

结论：

- 月付合同流失率远高于年付和两年合同；
- 合同类型是流失的最强预测特征之一；
- 推动月付客户转年付，是降低流失的核心策略。

---

### 6.3 月费区间流失率

| 月费区间        | 客户数 | 流失数 | 流失率 |
| --------------- | -----: | -----: | -----: |
| 高月费（>=70）  |  3,591 |  1,274 | 35.48% |
| 中月费（35-70） |  1,721 |    407 | 23.65% |
| 低月费（<35）   |  1,731 |    188 | 10.86% |

结论：

- 高月费客户流失率约 35.48%，显著高于低月费客户；
- 高月费客户对价格、服务体验更敏感；
- 需对高月费客户提供差异化权益，避免单纯涨价。

---

### 6.4 在网时长流失率

| 在网时长    | 客户数 | 流失数 | 流失率 |
| ----------- | -----: | -----: | -----: |
| 0-12 个月   |  2,069 |    999 | 48.28% |
| 12-24 个月  |  1,047 |    309 | 29.51% |
| 24-48 个月  |  1,624 |    339 | 20.87% |
| 48 个月以上 |  2,303 |    222 |  9.64% |

结论：

- 新客户（0-12 个月）流失率最高，约 48.28%；
- 在网时长越长，流失率越低；
- 前 12 个月是客户留存的关键窗口。

<img src="images\tenure_churn.png" alt="tenure_churn" style="zoom:50%;" />

---

### 6.5 月费与流失

<img src="images\monthly_churn.png" alt="monthly_churn" style="zoom:50%;" />

- 流失客户月费整体高于非流失客户；
- 高月费客户流失风险更大；
- 需结合合同类型、在网时长综合制定挽留策略。

---

### 6.6 互联网服务 + 支付方式交叉流失率

| InternetService | PaymentMethod             | 客户数 | 流失率 |
| --------------- | ------------------------- | -----: | -----: |
| Fiber optic     | Electronic check          |  1,595 | 53.23% |
| Fiber optic     | Mailed check              |    258 | 42.64% |
| DSL             | Electronic check          |    648 | 31.94% |
| Fiber optic     | Bank transfer (automatic) |    646 | 28.95% |
| Fiber optic     | Credit card (automatic)   |    597 | 25.29% |
| DSL             | Mailed check              |    613 | 20.72% |
| No              | Electronic check          |    122 | 12.30% |
| DSL             | Credit card (automatic)   |    594 | 12.12% |
| No              | Mailed check              |    741 |  9.58% |
| DSL             | Bank transfer (automatic) |    566 |  9.36% |
| No              | Bank transfer (automatic) |    332 |  5.42% |

结论：

- 光纤 + 电子支票组合流失率最高，约 53.23%；
- 电子支票用户流失率整体偏高；
- 自动扣款用户流失率相对较低；
- 推动电子支票用户改为自动扣款，可降低流失。

<img src="images\InternetService_churn.png" alt="InternetService_churn" style="zoom:50%;" />

<img src="images\PaymentMethod_churn.png" alt="PaymentMethod_churn" style="zoom:50%;" />

---

### 6.7 在线安全与技术支持

<img src="images\OnlineSecurity_churn.png" alt="OnlineSecurity_churn" style="zoom:50%;" />

<img src="images\TechSupport_churn.png" alt="TechSupport_churn" style="zoom:50%;" />

| 服务           | 取值                | 流失率（约） |
| -------------- | ------------------- | -----------: |
| OnlineSecurity | No                  |        41.8% |
| OnlineSecurity | No internet service |         7.4% |
| OnlineSecurity | Yes                 |        14.6% |
| TechSupport    | No                  |        41.6% |
| TechSupport    | No internet service |         7.4% |
| TechSupport    | Yes                 |        15.2% |

结论：

- 未开通在线安全、技术支持的用户流失率显著更高；
- 捆绑安全与支持服务，可作为提升留存的手段。

---

### 6.8 多维度交叉：客户分群流失率

| tenure_segment | charge_segment | Contract       | 客户数 | 流失率 |
| -------------- | -------------- | -------------- | -----: | -----: |
| 新客户         | 高月费         | Month-to-month |    818 | 69.68% |
| 成长期         | 高月费         | Month-to-month |    438 | 50.23% |
| 新客户         | 中月费         | Month-to-month |    603 | 45.77% |
| 成熟期         | 高月费         | Month-to-month |    565 | 42.65% |
| 新客户         | 低月费         | Month-to-month |    487 | 29.77% |
| 忠诚客户       | 高月费         | Month-to-month |    283 | 28.27% |
| 成长期         | 中月费         | Month-to-month |    200 | 26.50% |
| 新客户         | 高月费         | One year       |      8 | 25.00% |
| 成长期         | 高月费         | One year       |     42 | 21.43% |
| 忠诚客户       | 高月费         | One year       |    445 | 17.30% |
| 成熟期         | 高月费         | One year       |    223 | 17.04% |

结论：

- “新客户 + 高月费 + 月付合同”流失率最高，约 69.68%；
- 月付合同叠加高月费、短在网时长，是流失重灾区；
- 年付合同能显著降低流失率。

---

### 6.9 数值变量相关性

<img src="images\corr_heatmap.png" alt="corr_heatmap" style="zoom:50%;" />

| 变量对                        | 相关系数 |
| ----------------------------- | -------: |
| tenure - TotalCharges         |     0.83 |
| MonthlyCharges - TotalCharges |     0.65 |
| tenure - MonthlyCharges       |     0.25 |

说明：

- 在网时长与累计费用高度相关；
- 月费与累计费用中度相关；
- 建模时需注意多重共线性，但树模型影响较小。

---

### 6.10 模型评估对比

<img src="images\roc_curve.png" alt="roc_curve" style="zoom:50%;" />

| 指标      |                逻辑回归 |                 随机森林 |
| --------- | ----------------------: | -----------------------: |
| AUC       |                  0.8415 |                   0.8226 |
| Accuracy  |                  0.7381 |                   0.7665 |
| Precision |                  0.5043 |                   0.5529 |
| Recall    |                  0.7834 |                   0.6283 |
| F1        |                  0.6136 |                   0.5882 |
| 混淆矩阵  | [[747, 288], [81, 293]] | [[845, 190], [139, 235]] |

结论：

- 逻辑回归 AUC 更高，Recall 更高，更适合“尽量找出流失客户”的挽留场景；
- 随机森林 Accuracy、Precision 更高，但漏报更多；
- 若挽留成本低、流失损失高，优先选逻辑回归；
- 若希望减少误伤，可选随机森林或调整阈值。

---

### 6.11 逻辑回归特征系数 Top10

<img src="images\lr_coef.png" alt="lr_coef" style="zoom:50%;" />

| 特征                                      |    系数 | 方向                                               |
| ----------------------------------------- | ------: | -------------------------------------------------- |
| num__tenure                               | -1.1411 | 在网越久，流失越低                                 |
| cat__Contract_Two year                    | -0.7664 | 两年合同，流失低                                   |
| cat__InternetService_Fiber optic          |  0.7158 | 光纤，流失高                                       |
| num__MonthlyCharges                       | -0.6773 | 月费越高，流失越高（标准化后方向需结合共线性解释） |
| cat__Contract_Month-to-month              |  0.6630 | 月付，流失高                                       |
| cat__InternetService_DSL                  | -0.6177 | DSL，流失低                                        |
| num__TotalCharges                         |  0.4744 | 累计费用越高，流失越高（与 tenure 相关）           |
| cat__StreamingMovies_Yes                  |  0.2826 | 流媒体电影，流失略高                               |
| cat__DeviceProtection_No internet service | -0.2703 | 无互联网服务，流失低                               |
| cat__OnlineSecurity_No internet service   | -0.2703 | 无互联网服务，流失低                               |

解释：

- `tenure` 越长，流失概率越低；
- 月付合同、光纤服务显著提高流失概率；
- 两年合同、DSL 显著降低流失概率。

---

### 6.12 随机森林特征重要性 Top10

<img src="images\rf_importance.png" alt="rf_importance" style="zoom:50%;" />

| 特征                                | 重要性 |
| ----------------------------------- | -----: |
| num__TotalCharges                   | 0.1404 |
| num__tenure                         | 0.1290 |
| num__MonthlyCharges                 | 0.1228 |
| cat__Contract_Month-to-month        | 0.0687 |
| cat__OnlineSecurity_No              | 0.0393 |
| cat__Contract_Two year              | 0.0347 |
| cat__TechSupport_No                 | 0.0314 |
| cat__PaymentMethod_Electronic check | 0.0282 |
| cat__InternetService_Fiber optic    | 0.0249 |
| cat__OnlineBackup_No                | 0.0171 |

结论：

- 累计费用、在网时长、月费是最重要数值特征；
- 月付合同、无在线安全、无技术支持、电子支票、光纤服务是重要类别特征；
- 业务上应重点关注“月付 + 光纤 + 电子支票 + 无安全/支持”的组合。

---

### 6.13 阈值选择与成本敏感分析

<img src="images\threshold_analysis.png" alt="threshold_analysis" style="zoom:50%;" />

| 阈值 | Precision | Recall |     F1 |   TP |   FP |   FN |   TN |
| ---: | --------: | -----: | -----: | ---: | ---: | ---: | ---: |
| 0.10 |    0.3416 | 0.9893 | 0.5079 |  370 |  713 |    4 |  322 |
| 0.15 |    0.3672 | 0.9759 | 0.5336 |  365 |  629 |    9 |  406 |
| 0.20 |    0.3883 | 0.9626 | 0.5534 |  360 |  567 |   14 |  468 |
| 0.25 |    0.4067 | 0.9385 | 0.5675 |  351 |  512 |   23 |  523 |
| 0.30 |    0.4295 | 0.9278 | 0.5871 |  347 |  461 |   27 |  574 |
| 0.35 |    0.4471 | 0.9037 | 0.5982 |  338 |  418 |   36 |  617 |
| 0.40 |    0.4669 | 0.8663 | 0.6067 |  324 |  370 |   50 |  665 |
| 0.45 |    0.4831 | 0.8396 | 0.6133 |  314 |  336 |   60 |  699 |
| 0.50 |    0.5043 | 0.7834 | 0.6136 |  293 |  288 |   81 |  747 |
| 0.55 |    0.5252 | 0.7513 | 0.6183 |  281 |  254 |   93 |  781 |
| 0.60 |    0.5399 | 0.7059 | 0.6118 |  264 |  225 |  110 |  810 |
| 0.65 |    0.5672 | 0.6658 | 0.6125 |  249 |  190 |  125 |  845 |
| 0.70 |    0.6032 | 0.6016 | 0.6024 |  225 |  148 |  149 |  887 |
| 0.75 |    0.6574 | 0.5080 | 0.5732 |  190 |   99 |  184 |  936 |
| 0.80 |    0.7062 | 0.3984 | 0.5094 |  149 |   62 |  225 |  973 |
| 0.85 |    0.7373 | 0.2326 | 0.3537 |   87 |   31 |  287 | 1004 |

成本敏感分析假设：

- 假阳性（FP）成本 = 50 元；
- 假阴性（FN）成本 = 500 元。

结果：

- 成本最优阈值：**0.20**；
- 对应总成本：**35,350**；
- Precision：**0.3883**；
- Recall：**0.9626**。

业务建议：

- 若挽留成本低、流失损失高，选低阈值，尽量覆盖高风险客户；
- 若挽留资源有限，选中等阈值，优先覆盖高概率客户；
- 最终阈值应结合预算和 ROI 调整。

---

### 6.14 优先挽留客户与 ROI 测算

根据 `priority_customers.csv` 和 `roi_estimate.json`：

| 指标                                 |            数值 |
| ------------------------------------ | --------------: |
| 优先挽留客户数                       |           1,769 |
| 平均月费                             |      87.69 美元 |
| 假设挽留成功率                       |             15% |
| 假设多留存月数                       |         12 个月 |
| 年化潜在留存收益（美元）             |      279,214.83 |
| 年化潜在留存收益（人民币，按 1:7.2） | 2,010,346.78 元 |
| 约合人民币                           |  约 201.03 万元 |

优先挽留定义：

- 高风险：模型预测流失概率 >= 阈值；
- 高价值：月费处于高价值区间；
- 排序：按 ChurnProb 降序。

优先挽留客户示例：

| customerID | Contract       | tenure | MonthlyCharges | ChurnProb | RiskLevel | ValueLevel |
| ---------- | -------------- | -----: | -------------: | --------: | --------- | ---------- |
| 9497-QCMMS | Month-to-month |      1 |          93.55 |  0.938959 | 高风险    | 高价值     |
| 9300-AGZNL | Month-to-month |      1 |          94.00 |  0.938382 | 高风险    | 高价值     |
| 5178-LMXOP | Month-to-month |      1 |          95.10 |  0.937978 | 高风险    | 高价值     |
| 4424-TKOPW | Month-to-month |      2 |          93.85 |  0.937116 | 高风险    | 高价值     |
| 2720-WGKHP | Month-to-month |      2 |          94.00 |  0.936732 | 高风险    | 高价值     |

业务建议：

- 对优先名单客户，推送年付折扣券 + 免费技术支持；
- 对电子支票用户，引导自动扣款；
- 对光纤高月费用户，提供专属客服或套餐优化。

---

### 6.15 挽留策略 A/B 验证

<img src="outputs\ab_test\ab_test_results.png" alt="ab_test_results" style="zoom:50%;" />

根据 `ab_test_results.json`：

#### 实验设计

| 指标                |   数值 |
| ------------------- | -----: |
| 基线流失率          |    42% |
| 预期相对下降        |    20% |
| 预期实验组流失率    |  33.6% |
| 效应量（Cohen's h） | 0.1735 |
| 每组最小样本量      |    522 |
| 两组共需            |  1,044 |
| 实验池总人数        |  1,769 |
| 实际对照组人数      |    884 |
| 实际实验组人数      |    885 |

#### 分组均衡性检查

| 指标         | 对照组 A | 实验组 B |   t 值 |   p 值 | 结论                                 |
| ------------ | -------: | -------: | -----: | -----: | ------------------------------------ |
| 平均月费     |    87.18 |    88.19 | -1.993 | 0.0464 | 略不均衡，可用协变量调整或换随机种子 |
| 平均在网时长 |    17.67 |    18.13 | -0.612 | 0.5409 | 均衡                                 |
| 平均流失概率 |     0.79 |     0.79 |  0.403 | 0.6872 | 均衡                                 |

#### 模拟实验结果

| 指标              |            数值 |
| ----------------- | --------------: |
| 对照组流失人数    |       371 / 884 |
| 实验组流失人数    |       306 / 885 |
| 对照组流失率      |          41.97% |
| 实验组流失率      |          34.58% |
| 绝对下降          |   7.39 个百分点 |
| 相对下降          |          17.61% |
| 卡方统计量        |         10.2292 |
| 卡方 P 值（双侧） |        0.001382 |
| Z 统计量（单侧）  |         -3.1983 |
| 单侧 P 值         |        0.000691 |
| 95% CI            | [2.88%, 11.91%] |
| 统计显著          |              是 |

#### 业务测算

| 指标         |      数值 |
| ------------ | --------: |
| 挽留成功人数 |        65 |
| 挽回收入     | 54,600 元 |
| 实验成本     | 44,250 元 |
| 净收益       | 10,350 元 |
| ROI          |     23.4% |

结论：

- 发券 + 技术支持可显著降低流失率；
- 流失率从 41.97% 降至 34.58%；
- 单侧 P 值约 0.0007，统计显著；
- 95% CI 不包含 0；
- 模拟 ROI 约 23.4%，实际需线上 A/B 验证。

---

## 七、业务建议

1. **月付转年付**  
   对月付合同客户推送年付折扣，重点覆盖在网 3-12 个月客户。

2. **高月费光纤客户专属运营**  
   光纤用户流失率最高，需优化套餐、提升客服响应、提供价格保护。

3. **电子支票转自动扣款**  
   电子支票流失率约 45% 以上，引导自动扣款可提升留存。

4. **捆绑在线安全与技术支持**  
   未开通安全/支持的用户流失率约 42%，可打包优惠提升粘性。

5. **新客户 onboarding**  
   在网时长 < 12 个月客户流失率约 48.28%，前 3 个月加强使用指导与关怀。

6. **优先挽留名单**  
   对高风险 + 高价值客户，按 ChurnProb 降序触达，预算有限时优先覆盖。

7. **A/B 实验常态化**  
   对优惠券面额、权益组合、触达渠道、触达时机做多组实验，持续优化 ROI。

---

## 八、项目亮点

1. **完整链路**：清洗 → SQL → EDA → 建模 → 解释 → 阈值 → ROI → A/B。
2. **SQL 能力**：分组、交叉、CASE WHEN、特征宽表。
3. **Python 能力**：Pandas、Seaborn、Scikit-learn。
4. **建模能力**：逻辑回归 + 随机森林，ROC、混淆矩阵、特征重要性。
5. **业务闭环**：从流失因素到优先名单，再到挽留策略和收益测算。
6. **实验意识**：A/B 样本量、均衡性、显著性、置信区间、ROI。
7. **成本敏感**：阈值分析结合误伤成本和漏报成本，贴近真实业务。

---

## 九、运行说明

### 9.1 环境依赖

```bash
pip install pandas numpy matplotlib seaborn scikit-learn sqlalchemy pymysql scipy statsmodels
```

### 9.2 修改路径

在 `step1_prepare_data.py`、`step2_eda.py`、`step3_model.py`、`step4_roi.py`、`step5_ab_test.py` 中修改：

- 原始 CSV 路径；
- 清洗后 CSV 路径；
- 输出目录路径；
- MySQL 连接。

### 9.3 运行顺序

1. `step1_prepare_data.py`  
   清洗数据，保存 `telco_clean.csv`，导入 MySQL。

2. 在 MySQL 中运行 `telco-churn-analysis.sql`  
   获取总体流失率、分组流失率、特征宽表。

3. `step2_eda.py`  
   生成 EDA 图表。

4. `step3_model.py`  
   训练模型，输出 `model_results.json`、`priority_customers.csv`、阈值分析图。

5. `step4_roi.py`  
   输出 `roi_estimate.json`。

6. `step5_ab_test.py`  
   输出 `ab_test_results.json`、`experiment_groups.csv`、A/B 图。

---

## 十、局限性说明

1. 数据集为公开数据，非真实线上数据；
2. 流失预测为相关性建模，不代表因果；
3. A/B 实验为方案设计 + 模拟执行，无真实发券干预；
4. ROI 测算基于假设：挽留成功率 15%、多留存 12 个月、汇率 1:7.2；
5. 实际效果需线上 A/B 实验校准；
6. 模型阈值需结合预算、挽留成本和流失损失动态调整；
