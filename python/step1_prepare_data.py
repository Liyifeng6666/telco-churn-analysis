import pandas as pd
from sqlalchemy import create_engine

# 1. 读取数据
df = pd.read_csv(r'C:\telco_customer_churn\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("数据形状：", df.shape)
print(df.head())
print("\n各列数据类型：")
print(df.dtypes)

# 2. 清洗 TotalCharges
# 这个字段原本是字符串，里面有空格，需要转成数值
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("\nTotalCharges 缺失值数量：", df['TotalCharges'].isnull().sum())

# 这些缺失值通常是 tenure=0 的新客户，还没产生费用，填 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# 3. 保存清洗后的 CSV，方便后面 Python 分析
df.to_csv(r'C:\telco_customer_churn\data\clean\telco_clean.csv', index=False)

# 4. 导入 MySQL
engine = create_engine(
    'mysql+pymysql://root:123456@localhost:3306/telco_churn?charset=utf8mb4'
)

df.to_sql('customer_churn', engine, if_exists='replace', index=False, chunksize=1000)

print("\n数据已成功导入 MySQL！")