# ========== 7.3 潜在留存收益测算（严谨版） ==========
import pandas as pd

priority_df = pd.read_csv(r'C:\telco_customer_churn\data\outputs\priority_customers.csv')

n_priority = len(priority_df)

# 假设1：挽留成功率15%
# 来源：参考电信行业公开挽留活动效果数据（通常10%-25%），保守取15%
churn_rate_drop = 0.15

# 假设2：挽留后多留存12个月
expected_retention_months = 12

# 用月费计算未来收益，而不是用TotalCharges（历史累计）
avg_monthly = priority_df['MonthlyCharges'].mean()

# 潜在年化留存收益 = 优先挽留客户数 × 挽留成功率 × 月费 × 多留存月数
annual_saving_usd = n_priority * churn_rate_drop * avg_monthly * expected_retention_months

# 换算成人民币（假设1美元≈7.2元人民币）
usd_to_cny = 7.2
annual_saving_cny = annual_saving_usd * usd_to_cny

print(f"优先挽留客户数：{n_priority}")
print(f"假设挽留成功率：{churn_rate_drop*100:.0f}%（参考行业数据，保守取值）")
print(f"假设多留存月数：{expected_retention_months}个月")
print(f"平均月费：{avg_monthly:.2f} 美元")
print(f"\n年化潜在留存收益（美元）：{annual_saving_usd:.2f}")
print(f"年化潜在留存收益（人民币，按1:7.2）：{annual_saving_cny:.2f} 元")
print(f"约合：{annual_saving_cny/10000:.2f} 万元")

print("\n【假设来源说明】")
print("1. 挽留成功率15%：参考电信行业公开挽留活动效果，保守取值。")
print("2. 多留存12个月：假设挽留后客户至少多留存1年。")
print("3. 汇率1:7.2：近似值，实际需按当期汇率。")
print("4. 实际收益需通过A/B实验校准，本测算为量级估算。")