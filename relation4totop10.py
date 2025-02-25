import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据文件
programs_file = "programs_long.csv"  # 替换为实际文件路径
programs_data = pd.read_csv(programs_file)

# 计算每年的总赛事数量
programs_summary = programs_data.groupby('Year', as_index=False)['Event_Count'].sum()
programs_summary = programs_summary.rename(columns={'Event_Count': 'Total_Events'})

# 按运动计算每年的赛事占比
sport_importance = programs_data.groupby(['Sport', 'Year'], as_index=False)['Event_Count'].sum()
sport_importance['Event_Share'] = sport_importance['Event_Count'] / sport_importance.groupby('Year')['Event_Count'].transform('sum')

# 计算每个运动的平均赛事占比
average_importance = sport_importance.groupby('Sport', as_index=False)['Event_Share'].mean()
average_importance = average_importance.sort_values(by='Event_Share', ascending=False)

# 输出分析结果
print("按运动的平均赛事占比排序：")
print(average_importance)

# 保存分析结果到文件
average_importance.to_csv("sport_importance_analysis.csv", index=False)

# 选择前10个项目
top_10_sports = average_importance.head(10)['Sport'].tolist()

# 过滤掉不需要的行
filtered_sport_importance = sport_importance[sport_importance['Sport'].isin(top_10_sports)]

# 热力图表示赛事类型与总赛事数的关系（仅限前10运动）
plt.figure(figsize=(10, 6))
sport_pivot = filtered_sport_importance.pivot(index="Year", columns="Sport", values="Event_Share")
sns.heatmap(sport_pivot, cmap="YlGnBu", cbar=True, annot=False)
plt.xlabel("Sport")
plt.ylabel("Year")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 可视化每年赛事总数的变化
plt.figure(figsize=(10, 6))
sns.lineplot(data=programs_summary, x='Year', y='Total_Events', marker="o")
plt.title("Total Events by Year")
plt.xlabel("Year")
plt.ylabel("Total Events")
plt.grid(True)
plt.tight_layout()
plt.show()
