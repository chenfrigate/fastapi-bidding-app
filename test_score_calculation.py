import numpy as np
from main import calc_K_vectorized, score_single

# 测试 K 值计算函数
def test_calc_K_vectorized():
    print("\n===== 测试 K 值计算函数 =====")
    
    # 测试用例1：3家投标单位（不排除）
    prices1 = np.array([[900.0, 950.0, 1000.0]])
    K1 = calc_K_vectorized(prices1)
    expected1 = 950.0
    print(f"测试用例1 (n=3): K = {K1[0]}, 期望 = {expected1}, {'✓ 通过' if abs(K1[0] - expected1) < 0.001 else '✗ 失败'}")
    
    # 测试用例2：4家投标单位（排除最高价）
    prices2 = np.array([[800.0, 900.0, 1000.0, 1100.0]])
    K2 = calc_K_vectorized(prices2)
    expected2 = 900.0  # (800+900+1000)/3 = 900
    print(f"测试用例2 (n=4): K = {K2[0]}, 期望 = {expected2}, {'✓ 通过' if abs(K2[0] - expected2) < 0.001 else '✗ 失败'}")
    
    # 测试用例3：7家投标单位（排除最高和最低价）
    prices3 = np.array([[800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0]])
    K3 = calc_K_vectorized(prices3)
    expected3 = 950.0  # (850+900+950+1000+1050)/5 = 950
    print(f"测试用例3 (n=7): K = {K3[0]}, 期望 = {expected3}, {'✓ 通过' if abs(K3[0] - expected3) < 0.001 else '✗ 失败'}")
    
    # 测试用例4：1家投标单位
    prices4 = np.array([[950.0]])
    K4 = calc_K_vectorized(prices4)
    expected4 = 950.0
    print(f"测试用例4 (n=1): K = {K4[0]}, 期望 = {expected4}, {'✓ 通过' if abs(K4[0] - expected4) < 0.001 else '✗ 失败'}")
    
    # 测试用例5：6家投标单位（排除最高价）
    prices5 = np.array([[800.0, 850.0, 900.0, 950.0, 1000.0, 1100.0]])
    K5 = calc_K_vectorized(prices5)
    expected5 = 900.0  # (800+850+900+950+1000)/5 = 900
    print(f"测试用例5 (n=6): K = {K5[0]}, 期望 = {expected5}, {'✓ 通过' if abs(K5[0] - expected5) < 0.001 else '✗ 失败'}")

# 测试 报价分 计算函数
def test_score_single():
    print("\n===== 测试 报价分 计算函数 =====")
    K = 1000.0  # 基准价
    
    # 转换标量为NumPy数组以避免类型问题
    def get_score(P):
        # 将价格和基准价转换为2D数组以匹配函数的向量化要求
        P_arr = np.array([[P]])
        K_arr = np.array([[K]])
        result = score_single(P_arr, K_arr, 80)
        return result[0, 0]  # 提取标量结果
    
    # 测试用例1：评标价等于基准价
    P1 = 1000.0
    score1 = get_score(P1)
    expected1 = 80.0
    print(f"测试用例1 (P=K): 得分 = {score1}, 期望 = {expected1}, {'✓ 通过' if abs(score1 - expected1) < 0.001 else '✗ 失败'}")
    
    # 测试用例2：评标价高于基准价1%
    P2 = 1010.0  # K + 1%
    score2 = get_score(P2)
    expected2 = 79.0
    print(f"测试用例2 (P>K 1%): 得分 = {score2}, 期望 = {expected2}, {'✓ 通过' if abs(score2 - expected2) < 0.001 else '✗ 失败'}")
    
    # 测试用例3：评标价高于基准价25%（低于最低分）
    P3 = 1250.0  # K + 25%
    score3 = get_score(P3)
    expected3 = 60.0
    print(f"测试用例3 (P>K 25%): 得分 = {score3}, 期望 = {expected3}, {'✓ 通过' if abs(score3 - expected3) < 0.001 else '✗ 失败'}")
    
    # 测试用例4：评标价低于基准价10%
    P4 = 900.0  # K - 10%
    score4 = get_score(P4)
    expected4 = 90.0
    print(f"测试用例4 (P<K 10%): 得分 = {score4}, 期望 = {expected4}, {'✓ 通过' if abs(score4 - expected4) < 0.001 else '✗ 失败'}")
    
    # 测试用例5：评标价低于基准价20%（刚好达到最高分）
    P5 = 800.0  # K - 20%
    score5 = get_score(P5)
    expected5 = 100.0
    print(f"测试用例5 (P<K 20%): 得分 = {score5}, 期望 = {expected5}, {'✓ 通过' if abs(score5 - expected5) < 0.001 else '✗ 失败'}")
    
    # 测试用例6：评标价低于基准价30%（超过20%但低于40%）
    P6 = 700.0  # K - 30%
    score6 = get_score(P6)
    expected6 = 90.0  # 100 - (30-20) = 90
    print(f"测试用例6 (P<K 30%): 得分 = {score6}, 期望 = {expected6}, {'✓ 通过' if abs(score6 - expected6) < 0.001 else '✗ 失败'}")
    
    # 测试用例7：评标价低于基准价40%（刚好到边界）
    P7 = 600.0  # K - 40%
    score7 = get_score(P7)
    expected7 = 80.0  # 100 - (40-20) = 80
    print(f"测试用例7 (P<K 40%): 得分 = {score7}, 期望 = {expected7}, {'✓ 通过' if abs(score7 - expected7) < 0.001 else '✗ 失败'}")
    
    # 测试用例8：评标价低于基准价45%（低于40%以上）
    P8 = 550.0  # K - 45%
    score8 = get_score(P8)
    expected8 = 80.0
    print(f"测试用例8 (P<K 45%): 得分 = {score8}, 期望 = {expected8}, {'✓ 通过' if abs(score8 - expected8) < 0.001 else '✗ 失败'}")
    
    # 测试用例9：插值计算 - 评标价低于基准价15.5%
    P9 = 845.0  # K - 15.5%
    score9 = get_score(P9)
    expected9 = 95.5  # 80 + 15.5 = 95.5
    print(f"测试用例9 (P<K 15.5%): 得分 = {score9}, 期望 = {expected9}, {'✓ 通过' if abs(score9 - expected9) < 0.001 else '✗ 失败'}")
    
    # 测试用例10：向量化输入测试
    P_vec = np.array([[1000.0, 1050.0, 900.0, 700.0, 500.0]])
    K_vec = np.array([[1000.0, 1000.0, 1000.0, 1000.0, 1000.0]])
    scores_vec = score_single(P_vec, K_vec, 80)
    expected_vec = np.array([[80.0, 75.0, 90.0, 90.0, 80.0]])
    passed = np.all(np.abs(scores_vec - expected_vec) < 0.001)
    print(f"测试用例10 (向量化输入): {'✓ 通过' if passed else '✗ 失败'}")

# 运行测试
if __name__ == "__main__":
    test_calc_K_vectorized()
    test_score_single()