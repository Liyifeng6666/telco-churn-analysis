-- 总体流失率
SELECT
    COUNT(*) AS 总客户数,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS 流失客户数,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS 流失率
FROM customer_churn;

-- 按合同类型统计流失率
SELECT
    Contract AS 合同类型,
    COUNT(*) AS 客户数,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS 流失数,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS 流失率
FROM customer_churn
GROUP BY Contract
ORDER BY 流失率 DESC;

-- 按月费区间统计流失率
SELECT
    CASE
        WHEN MonthlyCharges < 35 THEN '低月费(<35)'
        WHEN MonthlyCharges < 70 THEN '中月费(35-70)'
        ELSE '高月费(>=70)'
    END AS 月费区间,
    COUNT(*) AS 客户数,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS 流失数,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS 流失率
FROM customer_churn
GROUP BY 月费区间
ORDER BY 流失率 DESC;

-- 按在网时长统计流失率
SELECT
    CASE
        WHEN tenure < 12 THEN '0-12个月'
        WHEN tenure < 24 THEN '12-24个月'
        WHEN tenure < 48 THEN '24-48个月'
        ELSE '48个月以上'
    END AS 在网时长,
    COUNT(*) AS 客户数,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS 流失数,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS 流失率
FROM customer_churn
GROUP BY 在网时长
ORDER BY 流失率 DESC;

-- 按互联网服务和支付方式统计
SELECT
    InternetService,
    PaymentMethod,
    COUNT(*) AS 客户数,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS 流失率
FROM customer_churn
GROUP BY InternetService, PaymentMethod
ORDER BY 流失率 DESC;

-- 客户级特征宽表（一行一个客户，聚合所有关键特征）
CREATE TABLE customer_features AS
SELECT
    customerID,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    tenure,
    MonthlyCharges,
    TotalCharges,
    ROUND(TotalCharges / NULLIF(tenure, 0), 2) AS avg_monthly_actual,
    CASE WHEN tenure < 12 THEN '新客户' 
         WHEN tenure < 24 THEN '成长期' 
         WHEN tenure < 48 THEN '成熟期' 
         ELSE '忠诚客户' END AS tenure_segment,
    CASE WHEN MonthlyCharges < 35 THEN '低月费'
         WHEN MonthlyCharges < 70 THEN '中月费'
         ELSE '高月费' END AS charge_segment,
    (CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN OnlineBackup = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN StreamingTV = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END) AS num_services,
    Churn
FROM customer_churn;

-- 用宽表做流失率交叉分析
SELECT
    tenure_segment,
    charge_segment,
    Contract,
    COUNT(*) AS 客户数,
    ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS 流失率
FROM customer_features
GROUP BY tenure_segment, charge_segment, Contract
ORDER BY 流失率 DESC
LIMIT 20;