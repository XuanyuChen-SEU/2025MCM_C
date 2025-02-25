import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# 1. 加载数据
medal_data = pd.read_csv('2025_Problem_C_Data/summerOly_medal_counts.csv')

# 2. 数据预处理
def prepare_data(df, country):
    country_data = df[df['NOC'] == country].sort_values(by='Year')
    country_data = country_data[['Year', 'Gold']]
    country_data = country_data.set_index('Year')
    return country_data

country_to_predict = 'China'
country_data = prepare_data(medal_data, country_to_predict)
print(country_data.head())



# 3. 数据探索
plt.figure(figsize=(10, 6))
plt.plot(country_data['Gold'])
plt.title(f'{country_to_predict} Gold Medals Over Time')
plt.xlabel('Year')
plt.ylabel('Gold Medals')
plt.show()

plot_acf(country_data['Gold'], lags=10)
plt.title(f'Autocorrelation of {country_to_predict} Gold Medals')
plt.show()

plot_pacf(country_data['Gold'], lags=4)
plt.title(f'Partial Autocorrelation of {country_to_predict} Gold Medals')
plt.show()


# 4. 选择 ARIMA 模型参数 (p, d, q)
p = 1
d = 1
q = 1

# 5. 训练 ARIMA 模型
model = ARIMA(country_data['Gold'], order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())

# 6. 模型预测
last_year = country_data.index[-1]
prediction_year = 2028
prediction = model_fit.predict(start = last_year + 4, end = prediction_year)
print(f'Predicted Gold Medals for {country_to_predict} in {prediction_year}: {prediction[prediction_year]}')

# 7. 模型评估
train_size = int(len(country_data) * 0.8)
train, test = country_data[:train_size], country_data[train_size:]

model_train = ARIMA(train['Gold'], order=(p,d,q))
model_train_fit = model_train.fit()

test_start_year = test.index[0]
test_prediction = model_train_fit.predict(start = test_start_year, end = test.index[-1])

rmse = np.sqrt(mean_squared_error(test['Gold'], test_prediction))
print(f'Root Mean Squared Error (RMSE): {rmse}')

plt.figure(figsize=(10,6))
plt.plot(train['Gold'], label='Train Data')
plt.plot(test['Gold'], label='Test Data')
plt.plot(test.index, test_prediction, label='Prediction', color='red')
plt.legend()
plt.title(f'{country_to_predict} Gold Medals Prediction')
plt.xlabel('Year')
plt.ylabel('Gold Medals')
plt.show()