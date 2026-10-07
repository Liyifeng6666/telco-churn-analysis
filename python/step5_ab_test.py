# ============================================================
# 项目二：挽留策略 A/B 验证方案（完整版 · 修复版）
# 说明：本代码包含 A/B 实验的完整设计 + 模拟执行 + 统计检验
# 数据来源：IBM Telco Customer Churn（公开数据集，无真实干预）
# 结论定性：方案设计 + 模拟验证，不代表真实因果
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest
import json
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs(r'C:\telco_customer_churn\data\outputs\ab_test', exist_ok=True)


# ============================================================
# 工具函数：把 numpy 类型转成 Python 原生类型，供 json.dump 使用
# ============================================================
def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ============================================================
# 模块1：实验背景与假设定义
# ============================================================
print("=" * 60)
print("模块1：实验背景与假设定义")
print("=" * 60)

experiment_config = {
    "实验名称": "高风险流失客户挽留策略A/B实验",
    "实验对象": "模型预测流失概率>=阈值的客户",
    "对照组A": "不进行任何挽留动作，仅观察",
    "实验组B": "发送年付折扣券 + 免费3个月技术支持",
    "主要指标": "客户实际流失率（Churn Rate）",
    "次要指标": ["ARPU变化", "客户满意度", "年付转化率"],
    "实验周期": "4周",
    "显著性水平": 0.05,
    "统计功效": 0.80
}

for k, v in experiment_config.items():
    print(f"  {k}：{v}")

print("\n【统计假设】")
print("  H0（原假设）：实验组与对照组的流失率无显著差异")
print("  H1（备择假设）：实验组的流失率显著低于对照组")
print("  检验类型：单侧检验（我们只关心'挽留是否降低流失率'）")


# ============================================================
# 模块2：样本量计算（统计功效分析）
# ============================================================
print("\n" + "=" * 60)
print("模块2：样本量计算")
print("=" * 60)

baseline_churn_rate = 0.42
expected_relative_lift = 0.20
expected_churn_rate = baseline_churn_rate * (1 - expected_relative_lift)

print(f"基线流失率：{baseline_churn_rate*100:.1f}%")
print(f"预期相对下降：{expected_relative_lift*100:.0f}%")
print(f"预期实验组流失率：{expected_churn_rate*100:.1f}%")

effect_size = proportion_effectsize(baseline_churn_rate, expected_churn_rate)
print(f"效应量（Cohen's h）：{effect_size:.4f}")

power_analysis = NormalIndPower()
required_n = power_analysis.solve_power(
    effect_size=effect_size,
    alpha=0.05,
    power=0.80,
    ratio=1.0,
    alternative='two-sided'
)

required_n = int(np.ceil(required_n))
print(f"\n每组所需最小样本量：{required_n} 人")
print(f"两组共计需要：{required_n * 2} 人")

# 注意：actual_available 会在模块3读取名单后动态更新
actual_available = None


# ============================================================
# 模块3：实验分组（基于真实高风险客户）
# ============================================================
print("\n" + "=" * 60)
print("模块3：实验分组")
print("=" * 60)

try:
    priority_df = pd.read_csv(r'C:\telco_customer_churn\data\outputs\priority_customers.csv')
    print(f"读取优先挽留客户名单成功，共 {len(priority_df)} 人")
except FileNotFoundError:
    print("未找到 priority_customers.csv，使用模拟数据代替")
    np.random.seed(42)
    priority_df = pd.DataFrame({
        'customerID': [f'C{i:05d}' for i in range(1500)],
        'MonthlyCharges': np.random.uniform(70, 110, 1500),
        'tenure': np.random.randint(1, 12, 1500),
        'ChurnProb': np.random.uniform(0.6, 0.95, 1500)
    })

experiment_pool = priority_df.copy()
n_pool = len(experiment_pool)
actual_available = n_pool
print(f"实验池总人数：{n_pool}")

# 用实际人数反推可检测的最小效应（如果不够）
if actual_available < required_n * 2:
    print(f"\n【提醒】实际可触达 {actual_available} 人 < 两组共需 {required_n * 2} 人")
    print("方案1：延长实验周期，积累更多样本")
    print("方案2：降低预期效应，接受更小的提升")
    print("方案3：放宽显著性水平（不推荐）")
else:
    print(f"\n【样本量充足】实际可触达 {actual_available} 人 >= 两组共需 {required_n * 2} 人")

# 随机分组（固定种子，保证可复现）
np.random.seed(2024)
shuffled_idx = np.random.permutation(n_pool)
mid = n_pool // 2

control_idx = shuffled_idx[:mid]
treatment_idx = shuffled_idx[mid:]

experiment_pool = experiment_pool.reset_index(drop=True)
experiment_pool['group'] = 'unassigned'
experiment_pool.loc[control_idx, 'group'] = 'A_control'
experiment_pool.loc[treatment_idx, 'group'] = 'B_treatment'

print(f"\n分组结果：")
print(f"  对照组A（不发券）：{len(control_idx)} 人")
print(f"  实验组B（发券）：{len(treatment_idx)} 人")

print("\n【分组均衡性检查】")
balance_check = experiment_pool.groupby('group').agg(
    平均月费=('MonthlyCharges', 'mean'),
    平均在网时长=('tenure', 'mean'),
    平均流失概率=('ChurnProb', 'mean'),
    人数=('customerID', 'count')
).round(2)
print(balance_check)

for col in ['MonthlyCharges', 'tenure', 'ChurnProb']:
    a_vals = experiment_pool[experiment_pool['group'] == 'A_control'][col]
    b_vals = experiment_pool[experiment_pool['group'] == 'B_treatment'][col]
    t_stat, p_val = ttest_ind(a_vals, b_vals)
    status = "均衡" if p_val >= 0.05 else "略不均衡（可用协变量调整或换随机种子）"
    print(f"  {col}: t={t_stat:.3f}, p={p_val:.4f} → {status}")


# ============================================================
# 模块4：模拟实验执行（对B组施加"发券"干预）
# ============================================================
print("\n" + "=" * 60)
print("模块4：模拟实验执行")
print("=" * 60)

np.random.seed(2024)

control_df = experiment_pool[experiment_pool['group'] == 'A_control'].copy()
treatment_df = experiment_pool[experiment_pool['group'] == 'B_treatment'].copy()

control_df['churn_after'] = np.random.binomial(1, baseline_churn_rate, len(control_df))
treatment_df['churn_after'] = np.random.binomial(1, expected_churn_rate, len(treatment_df))

result_df = pd.concat([control_df, treatment_df], ignore_index=True)

print(f"模拟实验完成")
print(f"  对照组A流失人数：{control_df['churn_after'].sum()}/{len(control_df)}")
print(f"  实验组B流失人数：{treatment_df['churn_after'].sum()}/{len(treatment_df)}")


# ============================================================
# 模块5：统计检验（卡方检验 + 单侧比例Z检验 + 置信区间）
# ============================================================
print("\n" + "=" * 60)
print("模块5：统计检验")
print("=" * 60)

# 显式转成 Python 原生类型，避免后续 JSON 报错
a_total = int(len(control_df))
a_churn = int(control_df['churn_after'].sum())
a_retain = int(a_total - a_churn)
a_churn_rate = float(a_churn / a_total)

b_total = int(len(treatment_df))
b_churn = int(treatment_df['churn_after'].sum())
b_retain = int(b_total - b_churn)
b_churn_rate = float(b_churn / b_total)

print(f"对照组A：流失率 {a_churn_rate*100:.2f}%（{a_churn}/{a_total}）")
print(f"实验组B：流失率 {b_churn_rate*100:.2f}%（{b_churn}/{b_total}）")
print(f"绝对下降：{(a_churn_rate - b_churn_rate)*100:.2f} 个百分点")
print(f"相对下降：{(a_churn_rate - b_churn_rate)/a_churn_rate*100:.2f}%")

# 卡方检验（双侧）
contingency_table = np.array([
    [a_churn, a_retain],
    [b_churn, b_retain]
])

chi2, p_value, dof, expected = stats.chi2_contingency(
    contingency_table, correction=False
)
chi2 = float(chi2)
p_value = float(p_value)

print(f"\n【卡方检验（双侧）】")
print(f"  列联表：\n{contingency_table}")
print(f"  卡方统计量：{chi2:.4f}")
print(f"  自由度：{dof}")
print(f"  P值：{p_value:.6f}")

# 单侧比例 Z 检验（与实验假设一致：B 组流失率 < A 组）
z_stat, p_one_sided = proportions_ztest(
    count=[b_churn, a_churn],
    nobs=[b_total, a_total],
    alternative='smaller'
)
z_stat = float(z_stat)
p_one_sided = float(p_one_sided)

print(f"\n【单侧比例Z检验（B < A）】")
print(f"  Z统计量：{z_stat:.4f}")
print(f"  单侧P值：{p_one_sided:.6f}")

if p_one_sided < 0.05:
    print(f"  结论：P<0.05，拒绝原假设，实验组流失率显著低于对照组")
else:
    print(f"  结论：P>=0.05，不能拒绝原假设，两组差异不显著")

# 计算流失率差异的95%置信区间
p1 = a_churn_rate
p2 = b_churn_rate
se = np.sqrt(p1*(1-p1)/a_total + p2*(1-p2)/b_total)
diff = float(p1 - p2)
ci_lower = float(diff - 1.96 * se)
ci_upper = float(diff + 1.96 * se)

print(f"\n【95%置信区间】")
print(f"  流失率绝对下降：{diff*100:.2f} 个百分点")
print(f"  95% CI：[{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
if ci_lower > 0:
    print(f"  结论：置信区间不包含0，差异显著")


# ============================================================
# 模块6：业务收益测算
# ============================================================
print("\n" + "=" * 60)
print("模块6：业务收益测算")
print("=" * 60)

cost_per_coupon = 50
avg_monthly_fee_cny = 70
retention_months = 12

saved_customers = int(a_churn - b_churn)
revenue_saved = float(saved_customers * avg_monthly_fee_cny * retention_months)
total_cost = float(b_total * cost_per_coupon)
net_profit = float(revenue_saved - total_cost)
roi = float(net_profit / total_cost) if total_cost > 0 else 0.0

print(f"假设条件：")
print(f"  每张优惠券成本：{cost_per_coupon} 元")
print(f"  平均月费：{avg_monthly_fee_cny} 元/月")
print(f"  挽留后多留存：{retention_months} 个月")
print(f"\n测算结果：")
print(f"  挽留成功客户数：{saved_customers} 人")
print(f"  挽回收入：{revenue_saved:,.0f} 元")
print(f"  实验成本：{total_cost:,.0f} 元")
print(f"  净收益：{net_profit:,.0f} 元")
print(f"  ROI：{roi*100:.1f}%")

print("\n【说明】以上为模拟实验下的收益测算，实际投放需通过线上A/B实验校准")


# ============================================================
# 模块7：可视化
# ============================================================
print("\n" + "=" * 60)
print("模块7：可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：两组流失率对比
ax1 = axes[0, 0]
bars = ax1.bar(['对照组A\n(不发券)', '实验组B\n(发券)'],
               [a_churn_rate*100, b_churn_rate*100],
               color=['#C44E52', '#55A868'], width=0.5)
ax1.set_ylabel('流失率 (%)')
ax1.set_title(f'两组流失率对比 (单侧P={p_one_sided:.4f})')
for bar, val in zip(bars, [a_churn_rate*100, b_churn_rate*100]):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5,
             f'{val:.2f}%', ha='center', fontsize=12, fontweight='normal')
ax1.set_ylim(0, max(a_churn_rate, b_churn_rate)*100 * 1.3)

# 图2：留存/流失人数堆叠柱状图
ax2 = axes[0, 1]
categories = ['对照组A', '实验组B']
retained = [a_retain, b_retain]
churned = [a_churn, b_churn]
x = np.arange(len(categories))
ax2.bar(x, retained, label='留存', color='#55A868')
ax2.bar(x, churned, bottom=retained, label='流失', color='#C44E52')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.set_ylabel('人数')
ax2.set_title('两组留存/流失人数对比')
ax2.legend()

# 图3：置信区间图
ax3 = axes[1, 0]
ax3.errorbar([0], [diff*100],
             yerr=[[(diff-ci_lower)*100], [(ci_upper-diff)*100]],
             fmt='o', color='steelblue', capsize=10, markersize=12)
ax3.axhline(0, color='red', linestyle='--', label='无差异线')
ax3.set_xlim(-0.5, 0.5)
ax3.set_xticks([0])
ax3.set_xticklabels(['流失率下降'])
ax3.set_ylabel('百分点')
ax3.set_title(f'流失率下降的95%置信区间\n[{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]')
ax3.legend()

# 图4：收益测算瀑布图
ax4 = axes[1, 1]
labels = ['挽回收入', '实验成本', '净收益']
values = [revenue_saved, -total_cost, net_profit]
colors = ['#55A868', '#C44E52', '#4C72B0']
bars = ax4.bar(labels, values, color=colors)
ax4.axhline(0, color='black', linewidth=0.8)
ax4.set_ylabel('金额（元）')
ax4.set_title('挽留策略收益测算')
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2,
             val + (5000 if val > 0 else -8000),
             f'{val:,.0f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(r'C:\telco_customer_churn\data\outputs\ab_test/ab_test_results.png', dpi=150)
plt.show()


# ============================================================
# 模块8：结果固化（保存JSON + CSV）
# ============================================================
print("\n" + "=" * 60)
print("模块8：结果固化")
print("=" * 60)

ab_results = {
    "实验设计": {
        "基线流失率": round(baseline_churn_rate, 4),
        "预期相对下降": float(expected_relative_lift),
        "每组最小样本量": int(required_n),
        "实际实验组人数": int(b_total),
        "实际对照组人数": int(a_total)
    },
    "实验结果": {
        "对照组流失率": round(a_churn_rate, 4),
        "实验组流失率": round(b_churn_rate, 4),
        "绝对下降": round(diff, 4),
        "相对下降": round(diff / a_churn_rate, 4),
        "卡方统计量": round(chi2, 4),
        "卡方P值_双侧": round(p_value, 6),
        "Z统计量_单侧": round(z_stat, 4),
        "单侧P值": round(p_one_sided, 6),
        "95%CI下限": round(ci_lower, 4),
        "95%CI上限": round(ci_upper, 4),
        "统计显著": bool(p_one_sided < 0.05)
    },
    "业务测算": {
        "挽留成功人数": int(saved_customers),
        "挽回收入": round(revenue_saved, 2),
        "实验成本": round(total_cost, 2),
        "净收益": round(net_profit, 2),
        "ROI": round(roi, 4)
    },
    "重要说明": "本实验为方案设计+模拟执行，数据来自公开数据集，"
                "无真实发券干预。结论不代表真实因果，实际效果需线上实验验证。"
}

with open(
    r'C:\telco_customer_churn\data\outputs\ab_test/ab_test_results.json',
    'w',
    encoding='utf-8'
) as f:
    json.dump(
        ab_results,
        f,
        ensure_ascii=False,
        indent=2,
        default=json_default
    )

result_df.to_csv(
    r'C:\telco_customer_churn\data\outputs\ab_test/experiment_groups.csv',
    index=False
)

print("结果已保存：")
print("  outputs/ab_test/ab_test_results.json")
print("  outputs/ab_test/experiment_groups.csv")
print("  outputs/ab_test/ab_test_results.png")

print("\n" + "=" * 60)
print("全部完成！")
print("=" * 60)