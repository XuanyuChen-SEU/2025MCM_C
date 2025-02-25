# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端以保存图表为文件
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import chardet
import re
import warnings
from scipy import stats

# 过滤警告信息
warnings.filterwarnings('ignore')

# 设置中文字体，确保图表中可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置Seaborn的样式
sns.set(style="whitegrid")

# 定义 95% 置信区间的 z-score
z = 1.96


# 定义检测文件编码的函数
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))  # 读取前100KB
    print(f"检测到的文件编码 {file_path}: {result['encoding']}")
    return result['encoding']


# 定义文件路径（请根据实际路径进行调整）
medal_counts_path = r'summerOly_medal_counts.csv'
hosts_path = r'summerOly_hosts.csv'
programs_path = r'summerOly_programs.csv'
athletes_path = r'summerOly_athletes.csv'
data_dictionary_path = r'data_dictionary.csv'

# 检测并读取CSV文件
def read_csv_with_encoding(file_path):
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        print(f"读取 {file_path} 时遇到 UnicodeDecodeError，尝试使用 'latin1' 编码。")
        df = pd.read_csv(file_path, encoding='latin1')
    return df


# 读取所有数据
medal_counts = read_csv_with_encoding(medal_counts_path)
hosts = read_csv_with_encoding(hosts_path)
programs = read_csv_with_encoding(programs_path)
athletes = read_csv_with_encoding(athletes_path)
data_dictionary = read_csv_with_encoding(data_dictionary_path)

# 显示各数据集的前几行
print("\n奖牌统计数据 (Medal Counts):")
print(medal_counts.head())

print("\n主办城市数据 (Hosts):")
print(hosts.head())

print("\n项目数据 (Programs):")
print(programs.head())

print("\n运动员数据 (Athletes):")
print(athletes.head())

# -------------------------------
# 2. 数据预处理
# -------------------------------

# 2.1. 清理 summerOly_programs.csv
# 替换特殊符号为 NaN 并清理注释
programs.replace({'': np.nan, '': np.nan}, inplace=True)
programs['Sports Governing Body'] = programs['Sports Governing Body'].str.replace(r'\[.*?\]', '',
                                                                                  regex=True).str.strip()

# 将项目数据从宽格式转换为长格式
years = [str(year) for year in range(1896, 2033, 4)]  # 包括2028和2032
id_vars = ['Sport', 'Discipline', 'Code', 'Sports Governing Body']
existing_years = [year for year in years if year in programs.columns]
missing_years = set(years) - set(programs.columns)
if missing_years:
    print(f"\n警告: 以下年份在项目数据中缺失，将从 melt 中排除: {sorted(missing_years)}")

value_vars = existing_years
programs_long = programs.melt(id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Event_Count')

# 将 Year 转换为整数并处理缺失值
programs_long['Year'] = pd.to_numeric(programs_long['Year'], errors='coerce')
programs_long['Event_Count'] = programs_long['Event_Count'].astype(str).str.strip()

# 使用正则表达式移除 'Event_Count' 中的非数字字符
programs_long['Event_Count'] = programs_long['Event_Count'].apply(
    lambda x: re.findall(r'\d+', x)[0] if re.findall(r'\d+', x) else '0')

# 将 'Event_Count' 转换为整数
programs_long['Event_Count'] = programs_long['Event_Count'].fillna('0').astype(int)

# 确认 'Sport' 列存在且清理
print("\nprograms_long['Sport'] 样本:")
print(programs_long['Sport'].head())

# 去除 'Sport' 列的前后空格
programs_long['Sport'] = programs_long['Sport'].str.strip()

# 2.2. 合并 summerOly_medal_counts.csv 与 summerOly_hosts.csv
# 识别取消的奥运年份
# 通常取消的年份会在 'Host' 列中包含 'Cancelled' 字样
cancelled_years = hosts[hosts['Host'].str.contains('Cancelled', na=False)]['Year'].tolist()
print(f"\n取消的奥运年份 (Cancelled Olympic Years): {cancelled_years}")

# 仅保留非取消的年份
medal_counts = medal_counts[~medal_counts['Year'].isin(cancelled_years)]

# 创建一个国家名称到 NOC 代码的映射（根据实际数据补充）
country_to_noc = {
    'United States': 'USA',
    'Greece': 'GRE',
    'Germany': 'GER',
    'France': 'FRA',
    'Great Britain': 'GBR',
    'Hungary': 'HUN',
    'Austria': 'AUT',
    'Australia': 'AUS',
    'Denmark': 'DEN',
    'Switzerland': 'SUI',
    'Mixed team': 'MIX',
    'Belgium': 'BEL',
    'Italy': 'ITA',
    'Cuba': 'CUB',
    'Canada': 'CAN',
    'Spain': 'ESP',
    'Luxembourg': 'LUX',
    'Norway': 'NOR',
    'Netherlands': 'NED',
    'Sweden': 'SWE',
    'West Germany': 'GER',  # 处理西德
    'Soviet Union': 'URS',
    'Soviet Union/Russia': 'RUS',
    'South Korea': 'KOR',
    'Japan': 'JPN',
    'Brazil': 'BRA',
    'Finland': 'FIN',
    'China': 'CHN',
    'Czech Republic': 'CZE',
    'Czechoslovakia': 'TCH',
    'India': 'IND',
    'New Zealand': 'NZL',
    # 请根据实际数据继续添加其他国家的映射
}


# 函数：从 Host 列提取国家名称，并映射到 NOC 代码
def extract_noc(host_entry):
    if pd.isna(host_entry):
        return 'UNK'
    parts = host_entry.split(',')
    if len(parts) < 2:
        return 'UNK'
    country = parts[1].strip().replace('\xa0', ' ')
    return country_to_noc.get(country, 'UNK')


# 应用函数提取 Host_NOC
hosts['Host_NOC'] = hosts['Host'].apply(extract_noc)

# 合并 Host_NOC 到 medal_counts
medal_counts = pd.merge(medal_counts, hosts[['Year', 'Host_NOC']], on='Year', how='left')

# 创建 Host_Flag：如果 NOC 是 Host_NOC，则为1，否则为0
medal_counts['Host_Flag'] = (medal_counts['NOC'].map(country_to_noc) == medal_counts['Host_NOC']).astype(int)

# 2.3. 合并 summerOly_programs.csv 到 medal_counts
# 按年份聚合每年的总比赛项目数
total_events_per_year = programs_long.groupby(['Year'])['Event_Count'].sum().reset_index().rename(
    columns={'Event_Count': 'Total_Events_Per_Year'})

print("\n合并前 medal_counts 的列:", medal_counts.columns.tolist())
print("Total_Events_Per_Year 的列:", total_events_per_year.columns.tolist())

# 合并 Total_Events_Per_Year 到 medal_counts
medal_counts = pd.merge(medal_counts, total_events_per_year, on='Year', how='left').fillna(0)

# 检查合并后的 DataFrame
print("\n合并后 medal_counts 的部分数据:")
print(medal_counts[['Year', 'Total_Events_Per_Year']].head())

# 2.4. 合并 summerOly_athletes.csv 到 medal_counts
# 按 NOC 和 Year 计算 Athlete_Count 和 Sport_Count
required_athlete_columns = {'NOC', 'Year', 'Name', 'Sport', 'Medal'}
if required_athlete_columns.issubset(athletes.columns):
    athletes_grouped = athletes.groupby(['NOC', 'Year']).agg({
        'Name': 'nunique',  # 独立运动员数量
        'Sport': 'nunique',  # 独立项目数量
        'Medal': lambda x: (x != 'No medal').sum()  # 奖牌数量
    }).reset_index().rename(columns={'Name': 'Athlete_Count', 'Sport': 'Sport_Count', 'Medal': 'Medal_Count'})

    # 合并到 medal_counts
    medal_counts = pd.merge(medal_counts, athletes_grouped[['NOC', 'Year', 'Athlete_Count', 'Sport_Count']],
                            on=['NOC', 'Year'], how='left').fillna(0)
else:
    print("\nAthletes 数据不包含所需的列。")
    medal_counts['Athlete_Count'] = 0
    medal_counts['Sport_Count'] = 0

# 2.5. 特征工程

# 按 NOC 和 Year 排序以进行累积计算
medal_counts = medal_counts.sort_values(['NOC', 'Year'])

# # 计算 Historical_Gold 和 Historical_Total
# medal_counts['Historical_Gold'] = medal_counts.groupby('NOC')['Gold'].cumsum().shift(1).fillna(0)
# medal_counts['Historical_Total'] = medal_counts.groupby('NOC')['Total'].cumsum().shift(1).fillna(0)


# 计算每个国家最近三次奥运会的历史奖牌总数
medal_counts['Historical_Total'] = medal_counts.groupby('NOC')['Total'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
medal_counts['Historical_Gold'] = medal_counts.groupby('NOC')['Gold'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)

# 填充缺失值
medal_counts['Historical_Total'].fillna(0, inplace=True)
medal_counts['Historical_Gold'].fillna(0, inplace=True)


# # Assign Olympic_Number based on sorted unique years
# unique_years = sorted(medal_counts['Year'].unique())
# year_to_number = {year: idx + 1 for idx, year in enumerate(unique_years)}
# medal_counts['Olympic_Number'] = medal_counts['Year'].map(year_to_number)

# # Assign GreatCoach_Proxy as Historical_Total
# medal_counts['GreatCoach_Proxy'] = medal_counts['Historical_Total']

# 显示处理后的 medal_counts 数据框的部分内容
print("\n处理后的奖牌统计数据 (Processed Medal Counts):")
print(medal_counts.head())
medal_counts.to_csv('medal_counts.csv', index=False)
# -------------------------------
# 3. 添加“从未获奖”的国家
# -------------------------------
# 添加5个假国家，确保它们在2024年及之前没有获奖记录
fake_nocs = ['Utopia1', 'Utopia2', 'Utopia3', 'Atlantis1', 'Atlantis2']
fake_rows = []
for i, noc in enumerate(fake_nocs, start=1):
    fake_rows.append({
        'Rank': 9998 + i,
        'NOC': noc,
        'Gold': 0,
        'Silver': 0,
        'Bronze': 0,
        'Total': 0,
        'Year': 2024,  # 假设在2024年也没有获奖
        'Host_NOC': 'UNK',
        'Host_Flag': 0,
        'Total_Events_Per_Year': 600,  # 与平均值相同
        'Athlete_Count': 5,  # 假设有5名运动员
        'Sport_Count': 10,  # 增加Sport_Count以提高获奖概率
        'Historical_Gold': 0,
        'Historical_Total': 0,
    })

fake_df = pd.DataFrame(fake_rows)
medal_counts = pd.concat([medal_counts, fake_df], ignore_index=True)

# 重新排序并重置索引
medal_counts = medal_counts.sort_values(['NOC', 'Year']).reset_index(drop=True)

# -------------------------------
# 4. 构建和训练奖牌计数模型
# -------------------------------

# 4.1. 构建金牌数预测模型
predictors_gold = ['Historical_Gold', 'Sport_Count', 'Total_Events_Per_Year']
model_data_gold = medal_counts.dropna(subset=predictors_gold + ['Gold']).copy()
X_gold = model_data_gold[predictors_gold].copy()
y_gold = model_data_gold['Gold'].copy()

# 标准化预测变量
scaler_g = StandardScaler()
X_gold_scaled = scaler_g.fit_transform(X_gold)
X_gold_scaled = pd.DataFrame(X_gold_scaled, index=X_gold.index, columns=X_gold.columns)
X_gold_scaled = sm.add_constant(X_gold_scaled)

# 检查预测变量的方差
print("\n预测变量的方差 (Gold Medals):")
print(X_gold.var())

# 识别低方差特征（方差 < 0.1）
low_variance_features = X_gold.columns[X_gold.var() < 0.1].tolist()
print(f"\n低方差或零方差的变量: {low_variance_features}")

# 移除低方差特征
X_gold_cleaned = X_gold.drop(columns=low_variance_features)
print(f"清理后的预测变量: {X_gold_cleaned.columns.tolist()}")

# 标准化清理后的预测变量
X_gold_final_scaled = scaler_g.transform(X_gold_cleaned)
X_gold_final_scaled = pd.DataFrame(X_gold_final_scaled, index=X_gold_cleaned.index, columns=X_gold_cleaned.columns)
X_gold_final_scaled = sm.add_constant(X_gold_final_scaled)

# 拟合金牌数的负二项回归模型
model_gold = None
avg_gold = y_gold.mean()  # 预设平均值
try:
    model_gold = NegativeBinomial(y_gold, X_gold_final_scaled).fit()
    print("\n金牌数的负二项回归模型 (Negative Binomial Regression for Gold Medals):")
    print(model_gold.summary())

    # 特征重要性分析（随机森林）
    rf_gold = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_gold.fit(X_gold_cleaned, y_gold)
    importances_gold = rf_gold.feature_importances_
    feature_importance_gold = pd.DataFrame({
        'Feature': X_gold_cleaned.columns,
        'Importance': importances_gold
    }).sort_values(by='Importance', ascending=False)
except Exception as e:
    print(f"\n拟合金牌数的负二项回归模型时出错: {e}")
    print("由于模型拟合失败，使用历史平均值进行预测。")
    model_gold = None

# 4.2. 构建总奖牌数预测模型
predictors_total = ['Historical_Gold', 'Sport_Count', 'Total_Events_Per_Year']
model_data_total = medal_counts.dropna(subset=predictors_total + ['Total']).copy()
X_total = model_data_total[predictors_total].copy()
y_total = model_data_total['Total'].copy()

# 标准化预测变量
scaler_t = StandardScaler()
X_total_scaled = scaler_t.fit_transform(X_total)
X_total_scaled = pd.DataFrame(X_total_scaled, index=X_total.index, columns=X_total.columns)
X_total_scaled = sm.add_constant(X_total_scaled)

# 检查预测变量的方差
print("\n预测变量的方差 (Total Medals):")
print(X_total.var())

# 识别低方差特征（方差 < 0.1）
low_variance_features_total = X_total.columns[X_total.var() < 0.1].tolist()
print(f"\n低方差或零方差的变量 (Total Medals): {low_variance_features_total}")

# 移除低方差特征
X_total_cleaned = X_total.drop(columns=low_variance_features_total)
print(f"清理后的预测变量 (Total Medals): {X_total_cleaned.columns.tolist()}")

# 标准化清理后的预测变量
X_total_final_scaled = scaler_t.transform(X_total_cleaned)
X_total_final_scaled = pd.DataFrame(X_total_final_scaled, index=X_total_cleaned.index, columns=X_total_cleaned.columns)
X_total_final_scaled = sm.add_constant(X_total_final_scaled)

# 拟合总奖牌数的负二项回归模型
model_total = None
avg_total = y_total.mean()  # 预设平均值
try:
    model_total = NegativeBinomial(y_total, X_total_final_scaled).fit()
    print("\n总奖牌数的负二项回归模型 (Negative Binomial Regression for Total Medals):")
    print(model_total.summary())

    # 特征重要性分析（随机森林）
    rf_total = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_total.fit(X_total_cleaned, y_total)
    importances_total = rf_total.feature_importances_
    feature_importance_total = pd.DataFrame({
        'Feature': X_total_cleaned.columns,
        'Importance': importances_total
    }).sort_values(by='Importance', ascending=False)
except Exception as e:
    print(f"\n拟合总奖牌数的负二项回归模型时出错: {e}")
    print("由于模型拟合失败，使用历史平均值进行预测。")
    model_total = None

# -------------------------------
# 5. 预测2028年洛杉矶夏季奥运会奖牌榜
# -------------------------------

# 5.1. 准备预测数据

# 获取所有独特的 NOC
all_nocs = medal_counts['NOC'].unique()

# 计算截至2024年的 Historical_Gold 和 Historical_Total
historical_gold = medal_counts[(medal_counts['Year'] >= 2016) & (medal_counts['Year'] <= 2024)].groupby('NOC')['Gold'].sum().reset_index().rename(
    columns={'Gold': 'Historical_Gold'})
historical_total = medal_counts[(medal_counts['Year'] >= 2016) & (medal_counts['Year'] <= 2024)].groupby('NOC')['Total'].sum().reset_index().rename(
    columns={'Total': 'Historical_Total'})

# 计算每个国家或地区在最近三次奥运会上的平均运动项目数量
avg_sport_count = medal_counts.groupby('NOC')['Sport_Count'].rolling(window=3, min_periods=1).mean().reset_index().rename(columns={'Sport_Count': 'Avg_Sport_Count'})

# 合并历史数据
df_2028 = pd.DataFrame({'NOC': all_nocs})
df_2028 = df_2028.merge(historical_gold, on='NOC', how='left')
df_2028 = df_2028.merge(historical_total, on='NOC', how='left')
df_2028 = df_2028.merge(avg_sport_count, on='NOC', how='left')

# 填充缺失值为0
df_2028.fillna(0, inplace=True)

# 重命名 'Avg_Sport_Count' 为 'Sport_Count' 以匹配模型预测变量
df_2028.rename(columns={'Avg_Sport_Count': 'Sport_Count'}, inplace=True)

# 删除 'Avg_Athlete_Count' 列（如果存在）
df_2028.drop(columns=['Avg_Athlete_Count'], inplace=True, errors='ignore')

# 打印列名以确认
print("\ndf_2028 处理后的列:")
print(df_2028.columns.tolist())

# 设置 Host_Flag：2028年由 USA 主办
df_2028['Host_Flag'] = df_2028['NOC'].apply(lambda x: 1 if x.upper() == 'USA' else 0)

# 设置 Total_Events_Per_Year 为历史数据的平均值
average_total_events = medal_counts['Total_Events_Per_Year'].mean()
df_2028['Total_Events_Per_Year'] = average_total_events



# -------------------------------
# 5.2. 预测金牌数和总奖牌数
# -------------------------------

# 预测金牌数
if model_gold is not None:
    try:
        # 定义用于2028年的预测变量
        predictors_2028_gold = ['Historical_Gold', 'Sport_Count', 'Total_Events_Per_Year']
        X_2028_gold = df_2028[predictors_2028_gold].copy()

        # 标准化使用与训练相同的 scaler_g
        X_2028_gold_scaled = scaler_g.transform(X_2028_gold)
        X_2028_gold_scaled = pd.DataFrame(X_2028_gold_scaled, index=df_2028.index, columns=predictors_2028_gold)
        X_2028_gold_scaled = sm.add_constant(X_2028_gold_scaled)

        # 预测金牌数
        df_2028['Predicted_Gold'] = model_gold.predict(X_2028_gold_scaled)

        # 获取负二项回归模型的 alpha 参数
        alpha_gold = model_gold.params.get('alpha', 1e-6)
        alpha_gold = max(alpha_gold, 1e-6)

        # 近似方差计算：Variance = mu + mu^2 / alpha
        df_2028['Gold_SE'] = np.sqrt(df_2028['Predicted_Gold'] + (df_2028['Predicted_Gold'] ** 2) / alpha_gold)

        # 计算置信区间
        df_2028['Gold_Lower'] = df_2028['Predicted_Gold'] - z * df_2028['Gold_SE']
        df_2028['Gold_Upper'] = df_2028['Predicted_Gold'] + z * df_2028['Gold_SE']

        # 确保预测值不为负
        df_2028['Gold_Lower'] = df_2028['Gold_Lower'].apply(lambda x: max(x, 0))
        df_2028['Gold_Upper'] = df_2028['Gold_Upper'].apply(lambda x: max(x, 0))

        # 限制金牌数的上限为100
        df_2028['Predicted_Gold'] = df_2028['Predicted_Gold'].apply(lambda x: min(x, 100))
        df_2028['Gold_Lower'] = df_2028['Gold_Lower'].apply(lambda x: min(x, 100))
        df_2028['Gold_Upper'] = df_2028['Gold_Upper'].apply(lambda x: min(x, 100))

    except Exception as e:
        print(f"\n预测2028年金牌数时出错: {e}")
        # 使用预设的平均值进行预测
        df_2028['Predicted_Gold'] = avg_gold
        df_2028['Gold_SE'] = 0  # 预设标准误差
        df_2028['Gold_Lower'] = avg_gold
        df_2028['Gold_Upper'] = avg_gold
else:
    print("\n金牌数模型未成功拟合。使用历史平均值进行预测。")
    # 使用预设的平均值进行预测
    df_2028['Predicted_Gold'] = y_gold.mean()
    df_2028['Gold_SE'] = 0  # 预设标准误差
    df_2028['Gold_Lower'] = y_gold.mean()
    df_2028['Gold_Upper'] = y_gold.mean()

# 预测总奖牌数
if model_total is not None:
    try:
        # 定义用于总奖牌数预测的预测变量
        predictors_2028_total = ['Historical_Gold', 'Sport_Count', 'Total_Events_Per_Year']
        X_2028_total = df_2028[predictors_2028_total].copy()

        # 标准化使用与训练相同的 scaler_t
        X_2028_total_scaled = scaler_t.transform(X_2028_total)
        X_2028_total_scaled = pd.DataFrame(X_2028_total_scaled, index=df_2028.index, columns=predictors_2028_total)
        X_2028_total_scaled = sm.add_constant(X_2028_total_scaled)

        # 预测总奖牌数
        df_2028['Predicted_Total'] = model_total.predict(X_2028_total_scaled)

        # 获取负二项回归模型的 alpha 参数
        alpha_total = model_total.params.get('alpha', 1e-6)
        alpha_total = max(alpha_total, 1e-6)

        # 近似方差计算：Variance = mu + mu^2 / alpha
        df_2028['Total_SE'] = np.sqrt(df_2028['Predicted_Total'] + (df_2028['Predicted_Total'] ** 2) / alpha_total)

        # 计算置信区间
        df_2028['Total_Lower'] = df_2028['Predicted_Total'] - z * df_2028['Total_SE']
        df_2028['Total_Upper'] = df_2028['Predicted_Total'] + z * df_2028['Total_SE']

        # 确保预测值不为负
        df_2028['Total_Lower'] = df_2028['Total_Lower'].apply(lambda x: max(x, 0))
        df_2028['Total_Upper'] = df_2028['Total_Upper'].apply(lambda x: max(x, 0))

        # 限制总奖牌数的上限为500
        df_2028['Predicted_Total'] = df_2028['Predicted_Total'].apply(lambda x: min(x, 500))
        df_2028['Total_Lower'] = df_2028['Total_Lower'].apply(lambda x: min(x, 500))
        df_2028['Total_Upper'] = df_2028['Total_Upper'].apply(lambda x: min(x, 500))

    except Exception as e:
        print(f"\n预测2028年总奖牌数时出错: {e}")
        # 使用预设的平均值进行预测
        df_2028['Predicted_Total'] = avg_total
        df_2028['Total_SE'] = 0  # 预设标准误差
        df_2028['Total_Lower'] = avg_total
        df_2028['Total_Upper'] = avg_total
else:
    print("\n总奖牌数模型未成功拟合。使用历史平均值进行预测。")
    # 使用预设的平均值进行预测
    df_2028['Predicted_Total'] = y_total.mean()
    df_2028['Total_SE'] = 0  # 预设标准误差
    df_2028['Total_Lower'] = y_total.mean()
    df_2028['Total_Upper'] = y_total.mean()

# -------------------------------
# 6. 预测首次获得奖牌的国家数量
# -------------------------------

# 6.1. 识别截至2024年从未获得奖牌的NOC
award_won_nocs = medal_counts[medal_counts['Total'] > 0]['NOC'].unique()
all_nocs = df_2028['NOC'].unique()
no_medal_nocs = [noc for noc in all_nocs if noc not in award_won_nocs]
print(f"\n截至2024年从未获得奖牌的NOC数量: {len(no_medal_nocs)}")

# 6.2. 准备逻辑回归的训练数据
# 筛选已获得奖牌的NOC
medal_counts_won = medal_counts[medal_counts['Total'] > 0].copy()

# 创建二元变量 'First_Medal'，标记每个NOC的首次获奖事件
medal_counts_won['First_Medal'] = 0
for noc, grp in medal_counts_won.groupby('NOC'):
    first_index = grp[grp['Total'] > 0].index[0]
    medal_counts_won.loc[first_index, 'First_Medal'] = 1

# 使用所有 'First_Medal' 样本进行训练（包括 0 和 1）
train_first_medal = medal_counts_won.copy()
y_train_first = train_first_medal['First_Medal']

# 检查 'First_Medal' 的类别分布
print("\n'First_Medal' 类别分布:")
print(y_train_first.value_counts())

if y_train_first.nunique() < 2:
    print("无法训练逻辑回归模型，因为目标变量仅包含一个类别。")
    expected_first_medals = 0
    ci_low = 0
    ci_high = 0
else:
    # 6.3. 训练逻辑回归模型
    predictors_logistic = ['Historical_Gold', 'Sport_Count', 'Total_Events_Per_Year']
    X_train_first = train_first_medal[predictors_logistic].copy()

    # 标准化预测变量
    scaler_first = StandardScaler()
    X_train_first_scaled = scaler_first.fit_transform(X_train_first)

    # 训练逻辑回归模型
    log_reg = LogisticRegression()
    log_reg.fit(X_train_first_scaled, y_train_first)
    print("\n逻辑回归模型已成功训练（首次获奖预测）。")

    # 6.4. 准备需要预测首次获奖的国家数据
    if len(no_medal_nocs) > 0:
        df_first_medal = df_2028[df_2028['NOC'].isin(no_medal_nocs)].copy()
        X_2028_first = df_first_medal[predictors_logistic].copy()

        # 检查预测数据中是否包含所有预测变量
        print("\ndf_first_medal 的列:")
        print(X_2028_first.columns.tolist())

        if 'Sport_Count' not in X_2028_first.columns or 'Total_Events_Per_Year' not in X_2028_first.columns:
            print("错误: 'Sport_Count' 或 'Total_Events_Per_Year' 在预测数据中缺失。")
            expected_first_medals = 0
            ci_low = 0
            ci_high = 0
        else:
            # 标准化预测数据
            X_2028_first_scaled = scaler_first.transform(X_2028_first)

            # 6.5. 预测首次获奖的概率
            df_first_medal['Medal_Prob'] = log_reg.predict_proba(X_2028_first_scaled)[:, 1]

            # 6.6. 计算预计首次获奖国家数量（整数）
            # 通过伯努利试验决定每个国家是否首次获奖
            np.random.seed(42)  # 固定随机种子
            df_first_medal['Medal_Trial'] = np.random.binomial(1, df_first_medal['Medal_Prob']).astype(int)
            expected_first_medals = df_first_medal['Medal_Trial'].sum()
            print(f"\n预计2028年首次获奖的国家数量: {expected_first_medals}")

            # 6.7. 计算置信区间（基于二项分布）
            if len(no_medal_nocs) > 0:
                # 使用 Wilson score interval
                n = len(no_medal_nocs)
                p_hat = expected_first_medals / n
                ci_low, ci_high = stats.binom.interval(0.95, n, p_hat, loc=0)
                ci_low = max(ci_low, 0)
                ci_high = min(ci_high, n)
                print(f"95%置信区间：[{ci_low}, {ci_high}]")
            else:
                ci_low = ci_high = expected_first_medals

            # 6.8. 计算每个NOC的赔率
            df_first_medal['Odds'] = df_first_medal['Medal_Prob'] / (1 - df_first_medal['Medal_Prob'])
            df_first_medal['Odds'] = df_first_medal['Odds'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # 显示概率最高的前10个NOC
            top_first_medal_probs = df_first_medal.sort_values(by='Medal_Prob', ascending=False).head(10)
            print("\n概率最高的前10个NOC:")
            print(top_first_medal_probs[['NOC', 'Medal_Prob', 'Odds']])

            # 6.9. 绘制首次获奖概率的水平条形图
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Medal_Prob', y='NOC', data=top_first_medal_probs, palette='coolwarm')
            plt.title('2028年首次获奖国家的获奖概率')
            plt.xlabel('获奖概率')
            plt.ylabel('国家（NOC）')
            plt.xlim(0, 1)  # 设定 x 轴范围为0到1
            plt.tight_layout()
            plt.savefig('first_time_medalist_probs.png')  # 保存图表到文件
            plt.close()
            print("首次获奖概率图已保存为 'first_time_medalist_probs.png'")
    else:
        print("没有国家需要预测首次获奖。")
        expected_first_medals = 0
        ci_low = 0
        ci_high = 0

# -------------------------------
# 7. 可视化
# -------------------------------

# 7.1. 绘制金牌数预测的水平条形图及置信区间（仅前10个）
if 'Predicted_Gold' in df_2028.columns:
    try:
        top_gold_plot = df_2028.sort_values('Predicted_Gold', ascending=False).head(10)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Predicted_Gold', y='NOC', data=top_gold_plot, palette='viridis')
        plt.errorbar(x=top_gold_plot['Predicted_Gold'], y=range(len(top_gold_plot)),
                     xerr=z * top_gold_plot['Gold_SE'], fmt='none', c='black', capsize=5)
        plt.title('2028年洛杉矶夏季奥运会各国预测金牌数 (前10名)')
        plt.xlabel('预测金牌数')
        plt.ylabel('国家（NOC）')
        plt.xlim(0, 100)  # 设置 x 轴范围为0到100
        plt.tight_layout()
        plt.savefig('predicted_gold_medals.png')  # 保存图表到文件
        plt.close()
        print("\n金牌数预测图已保存为 'predicted_gold_medals.png'")
    except Exception as e:
        print(f"\n绘制金牌数预测图时出错: {e}")
else:
    print("\n缺少 'Predicted_Gold' 列。跳过金牌数预测图绘制。")

# 7.2. 绘制总奖牌数预测的水平条形图及置信区间（仅前10个）
if 'Predicted_Total' in df_2028.columns:
    try:
        top_total_plot = df_2028.sort_values('Predicted_Total', ascending=False).head(10)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Predicted_Total', y='NOC', data=top_total_plot, palette='magma')
        plt.errorbar(x=top_total_plot['Predicted_Total'], y=range(len(top_total_plot)),
                     xerr=z * top_total_plot['Total_SE'], fmt='none', c='black', capsize=5)
        plt.title('2028年洛杉矶夏季奥运会各国预测总奖牌数 (前10名)')
        plt.xlabel('预测总奖牌数')
        plt.ylabel('国家（NOC）')
        plt.xlim(0, 500)  # 设置 x 轴范围为0到500
        plt.tight_layout()
        plt.savefig('predicted_total_medals.png')  # 保存图表到文件
        plt.close()
        print("总奖牌数预测图已保存为 'predicted_total_medals.png'")
    except Exception as e:
        print(f"\n绘制总奖牌数预测图时出错: {e}")
else:
    print("\n缺少 'Predicted_Total' 列。跳过总奖牌数预测图绘制。")

# 7.3. 绘制金牌数预测的特征重要性水平条形图
if 'feature_importance_gold' in locals():
    try:
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_gold, palette='viridis')
        plt.title('金牌数预测的特征重要性 (Random Forest)')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.savefig('feature_importance_gold_plot.png')  # 保存图表到文件
        plt.close()
        print("金牌数预测的特征重要性图已保存为 'feature_importance_gold_plot.png'")
    except Exception as e:
        print(f"\n绘制金牌数预测的特征重要性图时出错: {e}")
else:
    print("\n金牌数模型未成功拟合，无法分析 'GreatCoach_Proxy' 特征重要性。")

# 7.4. 绘制总奖牌数预测的特征重要性水平条形图
if 'feature_importance_total' in locals():
    try:
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_total, palette='magma')
        plt.title('总奖牌数预测的特征重要性 (Random Forest)')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.savefig('feature_importance_total_plot.png')  # 保存图表到文件
        plt.close()
        print("总奖牌数预测的特征重要性图已保存为 'feature_importance_total_plot.png'")
    except Exception as e:
        print(f"\n绘制总奖牌数预测的特征重要性图时出错: {e}")
else:
    print("\n总奖牌数模型未成功拟合，无法分析 'GreatCoach_Proxy' 特征重要性。")

# 7.5. 绘制项目数量与奖牌数的相关性热图
try:
    plt.figure(figsize=(8, 6))
    correlation = medal_counts[['Total_Events_Per_Year', 'Gold', 'Total']].corr()
    sns.heatmap(correlation, annot=True, cmap='Blues')
    plt.title('项目数量与奖牌数的相关性')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')  # 保存图表到文件
    plt.close()
    print("项目数量与奖牌数的相关性热图已保存为 'correlation_heatmap.png'")
except Exception as e:
    print(f"\n绘制项目数量与奖牌数的相关性热图时出错: {e}")

# 7.6. 绘制项目数量与金牌数的关系散点图
try:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Total_Events_Per_Year', y='Gold', data=medal_counts, alpha=0.5)
    plt.title('项目数量与金牌数的关系')
    plt.xlabel('每年总比赛项目数')
    plt.ylabel('金牌数')
    plt.tight_layout()
    plt.savefig('scatter_gold_medals.png')  # 保存图表到文件
    plt.close()
    print("项目数量与金牌数的关系散点图已保存为 'scatter_gold_medals.png'")
except Exception as e:
    print(f"\n绘制项目数量与金牌数的关系散点图时出错: {e}")

# 7.7. 绘制项目数量与总奖牌数的关系散点图
try:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Total_Events_Per_Year', y='Total', data=medal_counts, alpha=0.5, color='green')
    plt.title('项目数量与总奖牌数的关系')
    plt.xlabel('每年总比赛项目数')
    plt.ylabel('总奖牌数')
    plt.tight_layout()
    plt.savefig('scatter_total_medals.png')  # 保存图表到文件
    plt.close()
    print("项目数量与总奖牌数的关系散点图已保存为 'scatter_total_medals.png'")
except Exception as e:
    print(f"\n绘制项目数量与总奖牌数的关系散点图时出错: {e}")

# 7.8. 绘制“伟大教练”与金牌数的关系散点图
try:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='GreatCoach_Proxy', y='Gold', data=medal_counts, alpha=0.5)
    plt.title('“伟大教练”与金牌数的关系')
    plt.xlabel('GreatCoach_Proxy')
    plt.ylabel('金牌数')
    plt.tight_layout()
    plt.savefig('great_coach_vs_gold.png')  # 保存图表到文件
    plt.close()
    print("“伟大教练”与金牌数的关系散点图已保存为 'great_coach_vs_gold.png'")
except Exception as e:
    print(f"\n绘制“伟大教练”与金牌数的关系散点图时出错: {e}")

# 7.9. 绘制“伟大教练”与总奖牌数的关系散点图
try:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='GreatCoach_Proxy', y='Total', data=medal_counts, alpha=0.5, color='green')
    plt.title('“伟大教练”与总奖牌数的关系')
    plt.xlabel('GreatCoach_Proxy')
    plt.ylabel('总奖牌数')
    plt.tight_layout()
    plt.savefig('great_coach_vs_total.png')  # 保存图表到文件
    plt.close()
    print("“伟大教练”与总奖牌数的关系散点图已保存为 'great_coach_vs_total.png'")
except Exception as e:
    print(f"\n绘制“伟大教练”与总奖牌数的关系散点图时出错: {e}")

# -------------------------------
# 8. 预测结果汇总
# -------------------------------

# 8.1. 显示预测的前10个金牌数
if 'Predicted_Gold' in df_2028.columns:
    try:
        top_gold = df_2028.sort_values(by='Predicted_Gold', ascending=False).head(10)
        print("\n2028年预测金牌数最高的前10个国家:")
        print(top_gold[['NOC', 'Predicted_Gold', 'Gold_Lower', 'Gold_Upper']])
    except Exception as e:
        print(f"\n显示前10个预测金牌数时出错: {e}")
else:
    print("\n缺少 'Predicted_Gold' 列。跳过前10个预测金牌数的显示。")

# 8.2. 显示预测的前10个总奖牌数
if 'Predicted_Total' in df_2028.columns:
    try:
        top_total = df_2028.sort_values(by='Predicted_Total', ascending=False).head(10)
        print("\n2028年预测总奖牌数最高的前10个国家:")
        print(top_total[['NOC', 'Predicted_Total', 'Total_Lower', 'Total_Upper']])
    except Exception as e:
        print(f"\n显示前10个预测总奖牌数时出错: {e}")
else:
    print("\n缺少 'Predicted_Total' 列。跳过总奖牌数预测图绘制。")

# 8.3. 显示预计的首次获奖国家数量及置信区间
if 'expected_first_medals' in locals():
    try:
        print(f"\n预计2028年首次获奖的国家数量：{int(expected_first_medals)}")
        print(f"95%置信区间：[{int(ci_low)}, {int(ci_high)}]")
    except Exception as e:
        print(f"\n显示预计首次获奖国家数量时出错: {e}")
else:
    print("无法计算首次获奖国家数量，因为没有需要预测的国家。")

print("\n脚本运行完毕。所有结果已输出，图表已保存。")
