import numpy as np
import sys
import os

# 添加当前目录到Python路径以便导入main模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从main.py中导入核心计算函数
from main import calc_K_vectorized

# 定义不同情况下的测试数据
test_cases = [
    # 测试情况1: 3家报价（n≤3）
    {
        'name': '3家报价 (n≤3)',
        'prices': np.array([100, 200, 300]),
        'expected': (100 + 200 + 300) / 3  # 所有价格的平均值
    },
    # 测试情况2: 4家报价（4≤n≤6）
    {
        'name': '4家报价 (4≤n≤6)',
        'prices': np.array([100, 200, 300, 400]),
        'expected': (100 + 200 + 300) / 3  # 去掉最高价400后的平均值
    },
    # 测试情况3: 6家报价（4≤n≤6）
    {
        'name': '6家报价 (4≤n≤6)',
        'prices': np.array([100, 200, 300, 400, 500, 600]),
        'expected': (100 + 200 + 300 + 400 + 500) / 5  # 去掉最高价600后的平均值
    },
    # 测试情况4: 7家报价（n≥7）
    {
        'name': '7家报价 (n≥7)',
        'prices': np.array([100, 200, 300, 400, 500, 600, 700]),
        'expected': (200 + 300 + 400 + 500 + 600) / 5  # 去掉最高价700和最低价100后的平均值
    },
    # 测试情况5: 8家报价（n≥7）
    {
        'name': '8家报价 (n≥7)',
        'prices': np.array([100, 200, 300, 400, 500, 600, 700, 800]),
        'expected': (200 + 300 + 400 + 500 + 600 + 700) / 6  # 去掉最高价800和最低价100后的平均值
    }
]

# 运行测试
print("验证calc_K_vectorized函数是否符合评标基准价计算方法")
print("=" * 70)

for i, test_case in enumerate(test_cases):
    # 准备输入数据（转换为2D数组）
    price_matrix = test_case['prices'].reshape(1, -1)
    
    # 计算K值
    k_value = calc_K_vectorized(price_matrix)[0]
    
    # 获取预期值
    expected = test_case['expected']
    
    # 验证结果
    is_correct = np.isclose(k_value, expected)
    
    # 输出结果
    print(f"测试 {i+1}: {test_case['name']}")
    print(f"  报价: {test_case['prices']}")
    print(f"  计算K值: {k_value:.2f}")
    print(f"  预期K值: {expected:.2f}")
    print(f"  结果: {'✓ 符合要求' if is_correct else '✗ 不符合要求'}")
    print()

# 总体结论
print("结论:")
all_passed = all(np.isclose(calc_K_vectorized(tc['prices'].reshape(1, -1))[0], tc['expected']) for tc in test_cases)
if all_passed:
    print("✅ calc_K_vectorized函数的实现完全符合用户提供的评标基准价计算方法。")
    print("   - 当n≤3时，计算所有报价的平均值")
    print("   - 当4≤n≤6时，去掉最高价后计算平均值")
    print("   - 当n≥7时，去掉最高价和最低价后计算平均值")
else:
    print("❌ calc_K_vectorized函数的实现不完全符合要求，请检查。")