import matplotlib.pyplot as plt
import numpy as np

# 数据定义
subjects = [7, 11, 22, 40, 45]  # 横坐标Subject ID

# 第一张图数据（未使用数据增强） - 基于描述，使用代理值
# 注：原始第一张图数据对应Subject ID 5、10、20、40、45，这里映射到指定ID
no_augmentation = [0.746, 0.871, 0.767, 0.758, 0.887]  # 对应subjects顺序

# 第二张图数据（使用数据增强） - 基于描述，使用代理值
# 注：原始第二张图数据对应Subject ID 7、11、22、40、45
with_augmentation = [0.783, 0.854, 0.783, 0.783, 0.896]  # 对应subjects顺序

# 绘图设置
x = np.arange(len(subjects))  # x轴位置
width = 0.35  # 柱状图宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bars1 = ax.bar(x - width/2, no_augmentation, width, label='Without Data Augmentation', color='lightblue', edgecolor='black')
bars2 = ax.bar(x + width/2, with_augmentation, width, label='With Data Augmentation', color='lightcoral', edgecolor='black')

# 设置图表标签和标题（全部使用英语）
ax.set_xlabel('Subject ID', fontsize=12)
ax.set_ylabel('Average Validation Accuracy', fontsize=12)
ax.set_title('Final Model Performance by Subject', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylim(0.0, 1.0)  # Y轴范围
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格线

# 添加图例
ax.legend()

# 在柱状图上方显示数值
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(bars1)
autolabel(bars2)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()