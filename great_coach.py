import pandas as pd  # 导入 pandas 库，用于数据处理
import statsmodels.api as sm  # 导入 statsmodels 库，用于统计建模
import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图
import seaborn as sns  # 导入 seaborn 库，用于美化绘图

try:
    df = pd.read_csv("great_coach_data.csv") # 读取 CSV 文件，如果文件不存在，则会引发 FileNotFoundError 异常
except FileNotFoundError:
    print("Error: File 'great_coach_data.csv' not found.") # 打印错误信息
    exit() # 退出程序

#  为中国女排准备数据
X_chn_volleyball = df[['LangpingCHN']]  #  选择 "LangpingCHN" 列作为自变量
y_chn_volleyball = df['CHN_volleyball']  #  选择 "CHN_volleyball" 列作为因变量
X_chn_volleyball = sm.add_constant(X_chn_volleyball)  # 添加常数项到自变量中

#  为美国女排准备数据
X_usa_volleyball = df[['LangpingUSA']] # 选择 "LangpingUSA" 列作为自变量
y_usa_volleyball = df['USA_volleyball'] # 选择 "USA_volleyball" 列作为因变量
X_usa_volleyball = sm.add_constant(X_usa_volleyball) # 添加常数项到自变量中

#  为罗马尼亚体操队准备数据
X_rou_gym = df[['bela_karolyiROU']] # 选择 "bela_karolyiROU" 列作为自变量
y_rou_gym = df['ROU_gym'] # 选择 "ROU_gym" 列作为因变量
X_rou_gym = sm.add_constant(X_rou_gym) # 添加常数项到自变量中

#  为美国体操队准备数据
X_usa_gym = df[['bela_karolyiUSA']] # 选择 "bela_karolyiUSA" 列作为自变量
y_usa_gym = df['USA_gym'] # 选择 "USA_gym" 列作为因变量
X_usa_gym = sm.add_constant(X_usa_gym) # 添加常数项到自变量中

# 创建一个字典来存储模型，其中包含每个队伍的自变量、因变量和教练列的名称
models = {
    "CHN Volleyball": {"X":X_chn_volleyball, "y":y_chn_volleyball, "coach_col":"LangpingCHN"}, # 中国女排
    "USA Volleyball": {"X":X_usa_volleyball, "y":y_usa_volleyball, "coach_col":"LangpingUSA"}, # 美国女排
    "ROU Gymnastics": {"X":X_rou_gym, "y":y_rou_gym, "coach_col":"bela_karolyiROU"}, # 罗马尼亚体操队
    "USA Gymnastics": {"X":X_usa_gym, "y":y_usa_gym, "coach_col":"bela_karolyiUSA"} # 美国体操队
}

coach_coefs = {} # 创建一个字典来存储每个队伍的教练效应系数

for team, data in models.items(): # 循环处理每个队伍的数据
    model = sm.OLS(data['y'], data['X']).fit() # 使用 OLS 模型拟合数据
    coef = model.params[data['coach_col']] # 提取教练效应的系数
    coach_coefs[team] = coef # 将系数存储到字典中

coef_df = pd.DataFrame.from_dict(coach_coefs, orient='index', columns=['Coefficient']) # 将字典转换为 DataFrame
coef_df['Team'] = coef_df.index # 添加 "Team" 列，存储队伍名称

plt.figure(figsize=(10, 6)) # 设置图表大小
sns.barplot(x='Team', y='Coefficient', data=coef_df, palette='viridis') # 绘制柱状图
plt.title('Coaching Effect Coefficients') # 设置图表标题
plt.xlabel('Team') # 设置 x 轴标签
plt.ylabel('Coefficient') # 设置 y 轴标签
plt.xticks(rotation=45, ha='right')  # 旋转 x 轴标签，使其不重叠
plt.tight_layout() # 调整图表布局
plt.show() # 显示图表