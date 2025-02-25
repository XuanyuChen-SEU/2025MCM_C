import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
programs_data = pd.read_csv("programs_long.csv")
medal_data = pd.read_csv("olympic_medal_summary.csv")

# 2. 数据清理
programs_data["Year"] = programs_data["Year"].astype(int)
medal_data["Year"] = medal_data["Year"].astype(int)

# 3. 数据合并
merged_data = pd.merge(programs_data, medal_data, on="Year", how="inner")

# 4. 分析赛事数量与奖牌数量的相关性
event_count_by_year = merged_data.groupby("Year")["Event_Count"].sum().reset_index()
merged_data_with_total_events = pd.merge(merged_data, event_count_by_year, on="Year", how="left", suffixes=("_Individual", "_Total"))
correlation = merged_data_with_total_events["Event_Count_Total"].corr(merged_data_with_total_events["Total"])
print(f"Total Events vs. Total Medals Correlation: {correlation:.2f}")

# 绘制散点图
plt.scatter(merged_data_with_total_events["Event_Count_Total"], merged_data_with_total_events["Total"])

# 进行线性拟合
x = merged_data_with_total_events["Event_Count_Total"]
y = merged_data_with_total_events["Total"]
coefficients = np.polyfit(x, y, 1)
polynomial = np.poly1d(coefficients)
xs = np.linspace(x.min(), x.max(), 100)
ys = polynomial(xs)

# 绘制拟合线
plt.plot(xs, ys, color='red', label='Linear Fit')

plt.title("Total Events vs. Total Medals")
plt.xlabel("Total Events")
plt.ylabel("Total Medals")
plt.legend()
plt.show()
