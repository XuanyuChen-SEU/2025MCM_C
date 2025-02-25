import statsmodels.api as sm
import pandas as pd

# 读取你的数据
data = pd.read_csv("medal_counts.csv")

# 定义自变量和因变量 (根据你的实际情况调整)
X = data[["Host_Flag", "Total_Events_Per_Year", "Athlete_Count", "Historical_Total"]] 
y = data["Total"]

# 添加常数项
X = sm.add_constant(X)

# 拟合负二项回归模型
model = sm.NegativeBinomialP(y, X).fit()

# 打印模型结果，包括 AIC 和 BIC
print(model.summary())

# 从结果中提取 AIC 和 BIC 的值，你可以直接使用结果中的值，也可以手动提取。
print("AIC:", model.aic)
print("BIC:", model.bic)