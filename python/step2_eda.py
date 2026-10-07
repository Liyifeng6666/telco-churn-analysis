import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'C:\telco_customer_churn\data\clean\telco_clean.csv')

# 1. 总体流失率
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('流失与非流失客户数量')
plt.savefig(r'C:\telco_customer_churn\data\outputs\churn_count.png', dpi=150)
plt.show()

# 2. 合同类型 vs 流失率
contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
contract_churn.columns = ['Contract', 'ChurnRate']
plt.figure(figsize=(8, 5))
sns.barplot(x='Contract', y='ChurnRate', data=contract_churn, palette='Reds')
plt.title('不同合同类型的流失率')
plt.ylabel('流失率')
plt.savefig(r'C:\telco_customer_churn\data\outputs\contract_churn.png', dpi=150)
plt.show()

# 3. 在网时长 vs 流失
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='tenure', data=df, palette='Set2')
plt.title('流失与非流失客户的在网时长对比')
plt.savefig(r'C:\telco_customer_churn\data\outputs\tenure_churn.png', dpi=150)
plt.show()

# 4. 月费 vs 流失
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
plt.title('流失与非流失客户的月费对比')
plt.savefig(r'C:\telco_customer_churn\data\outputs\monthly_churn.png', dpi=150)
plt.show()

# 5. 数值变量相关性热力图
plt.figure(figsize=(6, 4))
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(), annot=True, cmap='coolwarm')
plt.title('数值变量相关性')
plt.savefig(r'C:\telco_customer_churn\data\outputs\corr_heatmap.png', dpi=150)
plt.show()

# 6. 类别特征流失率：互联网服务、支付方式、在线安全、技术支持
cat_cols = ['InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport']
for col in cat_cols:
    temp = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
    temp.columns = [col, 'ChurnRate']
    plt.figure(figsize=(8, 5))
    sns.barplot(x=col, y='ChurnRate', data=temp, palette='Oranges')
    plt.title(f'{col} 的流失率')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(rf'C:\telco_customer_churn\data\outputs\{col}_churn.png', dpi=150)
    plt.show()