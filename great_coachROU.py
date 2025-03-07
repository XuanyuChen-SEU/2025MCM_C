import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 假设你已经有了一个 DataFrame，名为 'data'，包含了你的数据
data = pd.read_csv('langping_data.csv')

# 定义自变量和Scores
X = data[['Coach_Flag', 'Year']]  # 使用 'Coach_Flag' 和 'Year' 作为自变量
y = data['Scores']  # 替换为你的Scores列名

# 添加常数项
X = sm.add_constant(X)

# 拟合多元回归模型
ols_model = sm.OLS(y, X).fit()

# 打印模型摘要
print(ols_model.summary())

# 提取中国和美国的数据
CHN_data = data[data['NOC'] == 'CHN']
us_data = data[data['NOC'] == 'USA']

# 预测中国和美国的数据
CHN_predictions = ols_model.predict(sm.add_constant(CHN_data[['Coach_Flag', 'Year']]))
us_predictions = ols_model.predict(sm.add_constant(us_data[['Coach_Flag', 'Year']]))

# 绘制折线图
plt.figure(figsize=(10, 6))

# 中国数据
plt.plot(CHN_data['Year'], CHN_data['Scores'], label='Actual (CHN)', marker='o')
plt.plot(CHN_data['Year'], CHN_predictions, label='Predicted (CHN)', linestyle='--', marker='x')

# 美国数据
plt.plot(us_data['Year'], us_data['Scores'], label='Actual (USA)', marker='o')
plt.plot(us_data['Year'], us_predictions, label='Predicted (USA)', linestyle='--', marker='x')

plt.title('Model vs Actual Data for CHN and USA')
plt.xlabel('Year')
plt.ylabel('Scores')
plt.legend()
plt.show()
