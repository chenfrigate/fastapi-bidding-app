import numpy as np
import sys
import os

# 添加当前目录到Python路径以便导入main模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从main.py中导入核心计算函数
from main import calc_K_vectorized, score_single

# 用户提供的报价数据
prices = np.array([453.73, 455.93, 100, 800, 1200, 958.9])

# 转换为2D数组格式以符合calc_K_vectorized函数的输入要求
price_matrix = prices.reshape(1, -1)

# 计算K值
K_value = calc_K_vectorized(price_matrix)[0]

# 计算每个报价的得分（win_thr通常为80）
win_thr = 80
scores = score_single(prices, K_value, win_thr)

# 输出结果
print(f"报价数据: {prices}")
print(f"K值 (评标基准价): {K_value:.2f}")
print("每个报价的得分:")
for price, score in zip(prices, scores):
    print(f"  报价 {price}: 得分 {score}")