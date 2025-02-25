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
programs_long.to_csv('programs_long.csv', index=False)
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

# 计算每个国家最近三次奥运会的历史奖牌总数
medal_counts['Historical_Total'] = medal_counts.groupby('NOC')['Total'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
medal_counts['Historical_Gold'] = medal_counts.groupby('NOC')['Gold'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)

# 填充缺失值
medal_counts['Historical_Total'].fillna(0, inplace=True)
medal_counts['Historical_Gold'].fillna(0, inplace=True)

# 保存处理后的数据
print("\n处理后的奖牌统计数据 (Processed Medal Counts):")
print(medal_counts.head())
medal_counts.to_csv('medal_counts.csv', index=False)