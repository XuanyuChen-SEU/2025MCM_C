import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取 CSV 文件
df = pd.read_csv("predictions_4_gold-2.csv")  # 将 "your_data.csv" 替换为你的文件名

# 2. 按照 "new_gold" 列降序排序
df_sorted = df.sort_values(by="new_gold", ascending=False)

# 3. 选择前十行
df_top10 = df_sorted.head(10)

# 4. 设置条形高度
bar_height = 0.35

# 5. 创建条形图
plt.figure(figsize=(10, 8))  # 设置图像大小 (交换宽高)

# 6. 设置 Y 轴位置（注意反转顺序）
y = np.arange(len(df_top10["noc"]))[::-1]  # 反转 Y 轴顺序

# 7. 绘制 "new_gold" 的条形 (水平)
gold_bars = plt.barh(y - bar_height/2, df_top10["new_gold"], bar_height, label="Gold Medals (new_gold)", color="gold")

# 8. 绘制 "new_total" 的条形 (水平)
total_bars = plt.barh(y + bar_height/2, df_top10["new_total"], bar_height, label="Total Medals (new_total)", color="red")

# 9. 添加标题和轴标签
plt.title("Top 10 Countries by Gold Medals and Total Medals")  # 添加标题
plt.ylabel("Country (NOC)")  # 添加 Y 轴标签 (交换标签)
plt.xlabel("Medals")  # 添加 X 轴标签 (交换标签)

# 10. 设置 Y 轴标签（注意反转顺序）
plt.yticks(y, df_top10["noc"])


# 11. 添加数值标签
def add_value_labels(bars):
    for bar in bars:
        width = bar.get_width()
        plt.text(width+0.1, bar.get_y() + bar.get_height()/2, str(round(width, 2)), va='center', ha='left')

add_value_labels(gold_bars)
add_value_labels(total_bars)

# 12. 添加图例
plt.legend()

# 13. 调整布局
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

# 14. 显示条形图
plt.show()