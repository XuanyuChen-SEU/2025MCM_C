import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据预处理
df = pd.read_csv("summerOly_athletes.csv")

# 过滤1984年以后的数据
df = df[df['Year'] >= 1984]

# 处理奖牌列，将其转化为数值
def medal_to_score(medal):
    if medal == "Gold":
        return 3
    elif medal == "Silver":
        return 2
    elif medal == "Bronze":
        return 1
    else:
        return 0

df['Medal_Score'] = df['Medal'].apply(medal_to_score)

# 2. 特征工程
# 按国家和项目分组，计算总得分
country_sport_score = df.groupby(['NOCC', 'Sport'])['Medal_Score'].sum().reset_index()

# 过滤国家
country_total_score = country_sport_score.groupby('NOCC')['Medal_Score'].sum()
countries_to_keep = country_total_score[country_total_score >= 50].index
filtered_df = country_sport_score[country_sport_score['NOCC'].isin(countries_to_keep)]

# 过滤项目
sport_total_count = filtered_df.groupby('Sport')['Medal_Score'].count()
sports_to_keep = sport_total_count[sport_total_count >= 50].index
filtered_df = filtered_df[filtered_df['Sport'].isin(sports_to_keep)]

# 重新计算pivot table
country_sport_score_pivot = filtered_df.pivot(index='NOCC', columns='Sport', values='Medal_Score').fillna(0)

# 计算每个项目的总分
sport_total_score = filtered_df.groupby('Sport')['Medal_Score'].sum()

# 将每个项目的得分除以总分，得到比值
for sport in country_sport_score_pivot.columns:
    country_sport_score_pivot[sport] = country_sport_score_pivot[sport] / sport_total_score[sport]

# 3. 结果可视化
plt.figure(figsize=(15, 15))
sns.heatmap(
    country_sport_score_pivot, 
    annot=False, 
    cmap='coolwarm',  # 调整为高对比度配色方案
    cbar_kws={'label': 'Normalized Medal Score'},  # 添加颜色条标签
    linewidths=0.5,  # 增加单元格边框
    linecolor='black',  # 设置单元格边框颜色
    vmin=0, vmax=0.5  # 明确设定颜色范围
)
plt.title('Country - Sport Score Heatmap (After Screening)', fontsize=18)
plt.xlabel('Sport', fontsize=14)
plt.ylabel('NOCC', fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
