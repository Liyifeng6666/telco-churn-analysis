import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取清洗后的数据
df = pd.read_csv(r'C:\telco_customer_churn\data\clean\telco_clean.csv')

# 2. 目标变量编码：Yes=1，No=0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. 特征与目标
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

# SeniorCitizen 虽然是数字，但它是类别特征，转成字符串
X['SeniorCitizen'] = X['SeniorCitizen'].astype(str)

# 数值特征和类别特征
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = [c for c in X.columns if c not in num_features]

# 4. 预处理：数值标准化，类别独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. 逻辑回归
lr_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
lr_pipe.fit(X_train, y_train)
y_pred_proba_lr = lr_pipe.predict_proba(X_test)[:, 1]
y_pred_lr = lr_pipe.predict(X_test)

print("=" * 50)
print("逻辑回归结果")
print("AUC:", roc_auc_score(y_test, y_pred_proba_lr))
print(classification_report(y_test, y_pred_lr))
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred_lr))

# 7. 随机森林
rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'))
])
rf_pipe.fit(X_train, y_train)
y_pred_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
y_pred_rf = rf_pipe.predict(X_test)

print("=" * 50)
print("随机森林结果")
print("AUC:", roc_auc_score(y_test, y_pred_proba_rf))
print(classification_report(y_test, y_pred_rf))
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred_rf))

# 8. ROC曲线对比
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(lr_pipe, X_test, y_test, name='逻辑回归')
RocCurveDisplay.from_estimator(rf_pipe, X_test, y_test, name='随机森林')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC曲线对比')
plt.savefig(r'C:\telco_customer_churn\data\outputs\roc_curve.png', dpi=150)
plt.show()

# 9. 逻辑回归特征系数（Top15）
feature_names = lr_pipe.named_steps['prep'].get_feature_names_out()
coefs = lr_pipe.named_steps['clf'].coef_[0]
coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
coef_df['abs_coef'] = coef_df['coef'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='coef', y='feature', data=coef_df, palette='coolwarm')
plt.title('逻辑回归特征系数 Top15（正=增加流失，负=降低流失）')
plt.tight_layout()
plt.savefig(r'C:\telco_customer_churn\data\outputs\lr_coef.png', dpi=150)
plt.show()

# 10. 随机森林特征重要性 Top15
rf_importances = rf_pipe.named_steps['clf'].feature_importances_
rf_df = pd.DataFrame({'feature': feature_names, 'importance': rf_importances})
rf_df = rf_df.sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=rf_df, palette='viridis')
plt.title('随机森林特征重要性 Top15')
plt.tight_layout()
plt.savefig(r'C:\telco_customer_churn\data\outputs\rf_importance.png', dpi=150)
plt.show()

# ========== 6.1 固化模型评估结果 ==========
# 逻辑回归评估
lr_results = {
    'AUC': round(roc_auc_score(y_test, y_pred_proba_lr), 4),
    'Accuracy': round(accuracy_score(y_test, y_pred_lr), 4),
    'Precision': round(precision_score(y_test, y_pred_lr), 4),
    'Recall': round(recall_score(y_test, y_pred_lr), 4),
    'F1': round(f1_score(y_test, y_pred_lr), 4),
    'ConfusionMatrix': confusion_matrix(y_test, y_pred_lr).tolist()
}

rf_results = {
    'AUC': round(roc_auc_score(y_test, y_pred_proba_rf), 4),
    'Accuracy': round(accuracy_score(y_test, y_pred_rf), 4),
    'Precision': round(precision_score(y_test, y_pred_rf), 4),
    'Recall': round(recall_score(y_test, y_pred_rf), 4),
    'F1': round(f1_score(y_test, y_pred_rf), 4),
    'ConfusionMatrix': confusion_matrix(y_test, y_pred_rf).tolist()
}

print("=" * 60)
print("逻辑回归完整评估：")
print(json.dumps(lr_results, indent=2, ensure_ascii=False))
print("=" * 60)
print("随机森林完整评估：")
print(json.dumps(rf_results, indent=2, ensure_ascii=False))

# 保存 Top10 特征重要性
top_lr = coef_df.head(10)[['feature', 'coef']].to_dict('records')
top_rf = rf_df.head(10)[['feature', 'importance']].to_dict('records')

with open(r'C:\telco_customer_churn\data\outputs\model_results.json', 'w', encoding='utf-8') as f:
    json.dump({
        'logistic_regression': lr_results,
        'random_forest': rf_results,
        'lr_top10_features': top_lr,
        'rf_top10_features': top_rf
    }, f, ensure_ascii=False, indent=2)

print("\n模型结果已保存到 outputs/model_results.json")

# ========== 6.2 阈值选择与成本敏感评估 ==========

from sklearn.metrics import precision_recall_curve, f1_score

# 获取逻辑回归预测概率
y_proba = y_pred_proba_lr

# 1. 遍历不同阈值，计算 Precision / Recall / F1
thresholds = np.arange(0.1, 0.9, 0.05)
results = []

for th in thresholds:
    y_pred_th = (y_proba >= th).astype(int)
    tp = ((y_pred_th == 1) & (y_test == 1)).sum()
    fp = ((y_pred_th == 1) & (y_test == 0)).sum()
    fn = ((y_pred_th == 0) & (y_test == 1)).sum()
    tn = ((y_pred_th == 0) & (y_test == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results.append({
        'threshold': round(th, 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    })

results_df = pd.DataFrame(results)
print(results_df)

# 2. 成本敏感分析
# 假设：
# - 挽留一个客户的成本 = 50 元（优惠券 + 人工成本）
# - 流失一个客户损失 = 500 元（客户生命周期价值）
# - 假阳性（FP）成本 = 50 元（错误挽留）
# - 假阴性（FN）成本 = 500 元（漏掉流失客户）
COST_FP = 50
COST_FN = 500

results_df['total_cost'] = results_df['fp'] * COST_FP + results_df['fn'] * COST_FN

# 找总成本最低的阈值
best_threshold = results_df.loc[results_df['total_cost'].idxmin()]
print(f"\n成本最优阈值：{best_threshold['threshold']}")
print(f"对应总成本：{best_threshold['total_cost']}")
print(f"Precision：{best_threshold['precision']}, Recall：{best_threshold['recall']}")

# 3. 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(results_df['threshold'], results_df['precision'], marker='o', label='Precision')
plt.plot(results_df['threshold'], results_df['recall'], marker='s', label='Recall')
plt.plot(results_df['threshold'], results_df['f1'], marker='^', label='F1')
plt.xlabel('阈值')
plt.ylabel('分数')
plt.title('不同阈值下的 Precision / Recall / F1')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results_df['threshold'], results_df['total_cost'], marker='o', color='red')
plt.axvline(best_threshold['threshold'], color='green', linestyle='--',
            label=f"最优阈值={best_threshold['threshold']}")
plt.xlabel('阈值')
plt.ylabel('总成本')
plt.title('不同阈值下的总成本')
plt.legend()
plt.tight_layout()
plt.savefig(r'C:\telco_customer_churn\data\outputs\threshold_analysis.png', dpi=150)
plt.show()

# 11. 输出高风险客户名单
X_all = df.drop(columns=['customerID', 'Churn'])
X_all['SeniorCitizen'] = X_all['SeniorCitizen'].astype(str)
df['ChurnProb'] = lr_pipe.predict_proba(X_all)[:, 1]

result = df[['customerID', 'Contract', 'tenure', 'MonthlyCharges',
             'TotalCharges', 'Churn', 'ChurnProb']].copy()
result['Churn'] = result['Churn'].map({1: 'Yes', 0: 'No'})

# 风险分层
result['RiskLevel'] = pd.cut(
    result['ChurnProb'],
    bins=[0, 0.3, 0.6, 1],
    labels=['低风险', '中风险', '高风险']
)

# 价值分层：按月费
result['ValueLevel'] = pd.cut(
    result['MonthlyCharges'],
    bins=[0, 35, 70, 200],
    labels=['低价值', '中价值', '高价值']
)

# 优先挽留：高风险 + 高价值
priority = result[(result['RiskLevel'] == '高风险') & (result['ValueLevel'] == '高价值')]
priority = priority.sort_values('ChurnProb', ascending=False)

priority.to_csv(r'C:\telco_customer_churn\data\outputs\priority_customers.csv', index=False)

print("\n优先挽留客户数量：", len(priority))
print(priority.head(10))