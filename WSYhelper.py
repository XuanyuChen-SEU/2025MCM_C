import pandas as pd
import re

def extract_country(host_string):
    if pd.isna(host_string) or not isinstance(host_string, str): #处理空值和非字符串
      return None
    match = re.search(r',([A-Za-z ]+)$', host_string) # 使用正则表达式匹配最后一个逗号后面的字符串
    if match:
        return match.group(1).strip() # 去除两端空格
    else:
        return host_string # 如果没有匹配项，则返回原始字符串

# 1. 读取 CSV 文件
try:
    df = pd.read_csv("summerOly_hosts.csv") # 替换为你的实际文件名
except FileNotFoundError:
    print("Error: File 'summerOly_hosts.csv' not found.")
    exit()

# 2. 应用函数提取国家
df['Host'] = df['Host'].apply(extract_country)
# 3. 输出处理后的数据
print(df)
# 你可以选择将结果保存到新的 CSV 文件
df.to_csv('output_file.csv', index=False)
print("results saved to 'output_file.csv'")