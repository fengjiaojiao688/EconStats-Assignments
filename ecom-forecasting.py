import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel('电子商务分析.xlsx', sheet_name='Sheet1', index_col=0)

# 数据预处理
df = df.dropna(how='all')
df = df.apply(pd.to_numeric, errors='coerce')
df.index = df.index.astype(str)  # 确保索引为字符串类型

print("=" * 60)
print("电子商务销售额时间趋势分析")
print("=" * 60)

# 1. 总体趋势分析
print("\n1. 电子商务总体销售额趋势:")
total_sales = df.sum(axis=1)
print(total_sales)

# 计算年度增长率
total_growth = total_sales.pct_change() * 100
print("\n年度增长率:")
for year, growth in total_growth.items():
    if not pd.isna(growth):
        print(f"{year}: {growth:.2f}%")

# 2. 各行业趋势分析
print("\n2. 各行业复合年均增长率(CAGR) 2014-2023:")
cagr_results = {}
for industry in df.columns:
    start_val = df[industry].iloc[-1]  # 2014年
    end_val = df[industry].iloc[0]     # 2023年
    years = 9  # 2014到2023共9年
    if start_val > 0:
        cagr = (end_val / start_val) ** (1/years) - 1
        cagr_results[industry] = cagr * 100
    else:
        cagr_results[industry] = np.nan

# 按CAGR排序
cagr_sorted = {k: v for k, v in sorted(cagr_results.items(),
                                      key=lambda item: item[1] if not pd.isna(item[1]) else -999,
                                      reverse=True)}

for industry, cagr in cagr_sorted.items():
    if not pd.isna(cagr):
        print(f"{industry}: {cagr:.2f}%")

# 3. 行业占比变化分析
print("\n3. 主要行业占比变化 (2014 vs 2023):")
industry_shares_2014 = (df.iloc[-1] / df.iloc[-1].sum() * 100).sort_values(ascending=False)
industry_shares_2023 = (df.iloc[0] / df.iloc[0].sum() * 100).sort_values(ascending=False)

print("\n2014年行业占比前5:")
for i, (industry, share) in enumerate(industry_shares_2014.head().items()):
    print(f"{i+1}. {industry}: {share:.1f}%")

print("\n2023年行业占比前5:")
for i, (industry, share) in enumerate(industry_shares_2023.head().items()):
    print(f"{i+1}. {industry}: {share:.1f}%")

# 4. 趋势稳定性分析（计算R²）
print("\n4. 各行业趋势稳定性（线性趋势R²）:")
trend_stability = {}
years = np.arange(len(df)).reshape(-1, 1)  # 年份作为自变量

for industry in df.columns:
    sales = df[industry].values.reshape(-1, 1)
    if not np.isnan(sales).any():
        model = LinearRegression()
        model.fit(years, sales)
        r_squared = model.score(years, sales)
        trend_stability[industry] = (r_squared, model.coef_[0][0])  # R²和斜率

# 按R²排序
stability_sorted = {k: v for k, v in sorted(trend_stability.items(),
                                           key=lambda item: item[1][0], reverse=True)}

for industry, (r2, slope) in stability_sorted.items():
    trend = "上升" if slope > 0 else "下降"
    print(f"{industry}: R²={r2:.3f} ({trend}趋势)")

# 5. 移动平均分析（平滑趋势）
print("\n5. 三年移动平均增长率分析:")
ma_3y = df.rolling(window=3, min_periods=1).mean()
ma_growth = ma_3y.pct_change() * 100

# 找出增长最快的年份和行业
max_growth = ma_growth.max().max()
max_growth_year = ma_growth.max(axis=1).idxmax()
max_growth_industry = ma_growth.max().idxmax()

print(f"最大三年移动平均增长率: {max_growth:.2f}%")
print(f"发生在: {max_growth_year}年, 行业: {max_growth_industry}")

# 6. 创建趋势分析汇总表
trend_summary = pd.DataFrame({
    '2014年销售额(亿元)': df.iloc[-1],
    '2023年销售额(亿元)': df.iloc[0],
    '绝对增长(亿元)': df.iloc[0] - df.iloc[-1],
    '增长倍数': df.iloc[0] / df.iloc[-1],
    'CAGR(%)': [cagr_results[col] for col in df.columns],
    '趋势稳定性(R²)': [trend_stability.get(col, (np.nan, np.nan))[0] for col in df.columns],
    '平均年度增长率(%)': df.pct_change().mean() * 100
})

print("\n6. 趋势分析汇总表:")
print(trend_summary.round(3))

# 7. 识别高增长行业
print("\n7. 高增长行业识别 (CAGR > 20%):")
high_growth = trend_summary[trend_summary['CAGR(%)'] > 20]
if not high_growth.empty:
    for industry, row in high_growth.iterrows():
        print(f"{industry}: CAGR={row['CAGR(%)']:.1f}%, 增长倍数={row['增长倍数']:.1f}x")
else:
    print("没有CAGR超过20%的行业")

# 8. 可视化准备数据
print("\n8. 可视化数据已准备完成")
print("可进一步使用以下数据进行可视化:")
print("- 各行业时间序列数据")
print("- 行业占比变化数据")
print("- 移动平均数据")

# 保存分析结果到Excel
output_filename = "电子商务趋势分析结果.xlsx"
with pd.ExcelWriter(output_filename) as writer:
    df.to_excel(writer, sheet_name='原始数据')
    trend_summary.to_excel(writer, sheet_name='趋势分析汇总')
    ma_3y.to_excel(writer, sheet_name='三年移动平均')
    industry_shares_2014.to_excel(writer, sheet_name='2014年行业占比')
    industry_shares_2023.to_excel(writer, sheet_name='2023年行业占比')

print(f"\n分析结果已保存到: {output_filename}")