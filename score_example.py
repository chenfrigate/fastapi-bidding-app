import numpy as np

# 从main.py导入score函数和calc_K函数
from main import score, calc_K

# 计算基准价K的函数，直接调用main.py中的calc_K函数
# 该函数根据有效评标价数量应用不同规则：
# 1. <=3家：所有有效评标价的平均值
# 2. 4-6家：去掉最高价后的平均值
# 3. >=7家：去掉最高价和最低价后的平均值
def calculate_K(bid_prices):
    # 为了适应calc_K函数的输入要求，需要将一维数组转换为二维数组
    # 并确保正确的形状 (1, n)，其中n是投标价格数量
    if bid_prices.ndim == 1:
        bid_prices_2d = bid_prices.reshape(1, -1)
        return calc_K(bid_prices_2d)[0]  # 取结果的第一个元素
    else:
        return calc_K(bid_prices)

# 创建示例数据
def demonstrate_multi_company_bidding():
    print("=== 多投标单位报价分计算演示 ===\n")
    
    # 投标单位名称列表
    companies = ["公司A", "公司B", "公司C", "公司D", "公司E", "公司F"]
    
    # 场景1: 单一项目，多个投标单位
    print("【场景1: 单一项目，多个投标单位】")
    # 6家公司的投标报价（万元）
    bid_prices_project1 = np.array([102.5, 98.0, 105.0, 85.0, 110.0, 75.0])
    
    # 计算该项目的基准价K
    K_project1 = calculate_K(bid_prices_project1)
    print(f"  基准价(K): {K_project1:.2f}万元")
    
    # 计算各公司的得分
    scores_project1 = score(bid_prices_project1, np.full_like(bid_prices_project1, K_project1))
    
    # 显示结果
    print("  投标单位报价及得分情况:")
    print("  +--------+---------+---------+-------------+")
    print("  | 公司   | 投标价格| 价格差异| 报价得分    |")
    print("  |        | (万元)  |  (%)    | (100分制)   |")
    print("  +--------+---------+---------+-------------+")
    for i, company in enumerate(companies):
        price = bid_prices_project1[i]
        diff = (price - K_project1) / K_project1 * 100
        company_score = scores_project1[i]
        print(f"  | {company:6}| {price:9.2f}| {diff:9.2f}| {company_score:11.2f}|" )
    print("  +--------+---------+---------+-------------+\n")
    
    # 场景2: 多个包件，多个投标单位
    print("【场景2: 多个包件，多个投标单位】")
    # 假设有3个包件，每个包件有不同的基准价
    # 6家公司对3个包件的投标报价（万元）
    # 格式: [公司A包件1, 公司A包件2, 公司A包件3, 公司B包件1, ...]
    bid_prices_projects = np.array([
        [102.5, 85.0, 120.0],  # 公司A
        [98.0, 82.0, 118.0],   # 公司B
        [105.0, 88.0, 125.0],  # 公司C
        [85.0, 75.0, 105.0],   # 公司D
        [110.0, 90.0, 130.0],  # 公司E
        [75.0, 70.0, 95.0]     # 公司F
    ])
    
    # 计算每个包件的基准价K
    K_projects = np.array([
        calculate_K(bid_prices_projects[:, 0]),  # 包件1基准价
        calculate_K(bid_prices_projects[:, 1]),  # 包件2基准价
        calculate_K(bid_prices_projects[:, 2])   # 包件3基准价
    ])
    
    print("  各包件基准价:")
    for i, K in enumerate(K_projects):
        print(f"  包件{i+1}: {K:.2f}万元")
    
    # 计算各公司各包件的得分
    scores_projects = np.zeros_like(bid_prices_projects)
    for i in range(3):  # 遍历3个包件
        # 为每个包件计算所有公司的得分
        scores_projects[:, i] = score(bid_prices_projects[:, i], np.full(6, K_projects[i]))
    
    # 显示结果
    print("\n  各公司各包件报价及得分情况:")
    for i, company in enumerate(companies):
        print(f"\n  {company}:")
        print("    +--------+---------+---------+-------------+")
        print("    | 包件   | 投标价格| 价格差异| 报价得分    |")
        print("    |        | (万元)  |  (%)    | (100分制)   |")
        print("    +--------+---------+---------+-------------+")
        for j in range(3):
            package = f"包件{j+1}"
            price = bid_prices_projects[i, j]
            K = K_projects[j]
            diff = (price - K) / K * 100
            company_score = scores_projects[i, j]
            print(f"    | {package:6}| {price:9.2f}| {diff:9.2f}| {company_score:11.2f}|" )
        print("    +--------+---------+---------+-------------+")
        
        # 计算该公司的平均得分
        avg_score = np.mean(scores_projects[i, :])
        print(f"    平均得分: {avg_score:.2f}分")

if __name__ == "__main__":
    demonstrate_multi_company_bidding()