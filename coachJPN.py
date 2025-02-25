import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 1. 数据读取
try:
    df = pd.read_csv("summerOly_athletes.csv")
except FileNotFoundError:
    print("Error: File 'summerOly_athletes.csv' not found.")
    exit()

# 2. 数据筛选
df['Year'] = df['Year'].astype(int)
df_JPN = df[(df['Year'] >= 2000) & (df['NOCC'] == 'JPN')]

# 3. 数据分析
# 计算参与次数
participation = df_JPN.groupby('Sport')['Name'].count().reset_index()
participation.rename(columns={'Name': 'Participation_Count'}, inplace=True)

# 计算奖牌数
def medal_to_score(medal):
    if medal == "Gold":
        return 3
    elif medal == "Silver":
        return 2
    elif medal == "Bronze":
        return 1
    else:
        return 0
df_JPN['Medal_Score'] = df_JPN['Medal'].apply(medal_to_score)

medal_count = df_JPN.groupby('Sport')['Medal_Score'].sum().reset_index()
medal_count.rename(columns={'Medal_Score': 'Medal_Count'}, inplace=True)

# 合并参与次数和奖牌数
merged_data = pd.merge(participation, medal_count, on='Sport', how='outer').fillna(0)

# 4. 结果展示
print("美国在2000年之后各个项目上的参与情况和获奖情况:")
print(merged_data.sort_values(by=['Medal_Count'],ascending=False))
import matplotlib.pyplot as plt

# 筛选出参与人员大于等于50的项目
filtered_data = merged_data[merged_data['Participation_Count'] >= 50]

# 对数据进行排序，以便更好地展示
filtered_data.sort_values(by=['Medal_Count'], ascending=False, inplace=True)

# # 创建一个新的图形
# plt.figure(figsize=(12, 6))

# # 绘制参与次数的柱状图
# plt.bar(filtered_data['Sport'], filtered_data['Participation_Count'], label='Participation Count')

# # 绘制奖牌数的柱状图
# plt.bar(filtered_data['Sport'], filtered_data['Medal_Count'], label='Medal Count')

# # 添加标题和标签
# plt.title('Participation and Medal Count of JPN in Each Sport Since 2000')
# plt.xlabel('Sport')
# plt.ylabel('Count')

# # 旋转x轴标签，防止重叠
# plt.xticks(rotation=45)

# # 添加图例
# plt.legend()

# # 显示图形
# plt.show()

# 计算奖牌数与参赛数的比值
filtered_data['Medal_Ratio'] = filtered_data['Medal_Count'] / filtered_data['Participation_Count']

# 对数据进行排序，以便更好地展示
filtered_data.sort_values(by=['Medal_Ratio'], ascending=False, inplace=True)

# 创建一个新的图形
plt.figure(figsize=(12, 6))

# 绘制热力图
sns.heatmap(filtered_data.pivot_table(index='Sport', values='Medal_Ratio').sort_values(by='Medal_Ratio', ascending=False), cmap='YlGnBu', annot=True, fmt=".2f")

# 添加标题和标签
plt.title('Medal Ratio of JPN in Each Sport Since 2000')

# 显示图形
plt.show()

