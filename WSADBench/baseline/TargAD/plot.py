import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取Excel文件
df = pd.read_excel('results/tabular/summary/TargAD_summary1.xlsx')

# 2. 按照seed排序（可选，保证横坐标有序）
df = df.sort_values('seed')

# 3. 绘制折线图
plt.figure(figsize=(8, 5))
plt.plot(df['seed'], df['aucroc'], marker='o', label='AUCROC')
plt.plot(df['seed'], df['aucpr'], marker='s', label='AUCPR')

plt.xlabel('Seed', fontsize=12)
plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
plt.ylim(0, 1.0)
plt.title('TargAD perform', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
plt.savefig('results/tabular/summary/result1.png')
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.ndimage import gaussian_filter1d

# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

# df = pd.read_excel('results/tabular/summary/TargAD_summary1.xlsx')
# print(df.head)
# plt.figure(figsize=(8,5))
# sns.set(style="whitegrid", font_scale=1.15)

# # 平滑数据（可选）
# x = df['seed']
# y1 = gaussian_filter1d(df['aucroc'], sigma=1)
# y2 = gaussian_filter1d(df['aucpr'], sigma=1)

# plt.plot(x, y1, marker='o', label='AUCROC', linewidth=2)
# plt.plot(x, y2, marker='s', label='AUCPR', linewidth=2, linestyle='--', color='orange')

# # 标注数据点数值
# for a, b in zip(x, y1):
#     plt.text(a, b, f'{b:.3f}', ha='center', va='bottom', fontsize=10, color='blue')
# for a, b in zip(x, y2):
#     plt.text(a, b, f'{b:.3f}', ha='center', va='top', fontsize=10, color='orange')

# plt.title('TargAD perform', fontsize=16)
# plt.ylabel('AUC(0.0~1.0)')
# plt.xlabel('Number of seed')
# plt.ylim(0.0, 1.0)
# plt.xticks(df['seed'])
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig('results/tabular/summary/result1.png')