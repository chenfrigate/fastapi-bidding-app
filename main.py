import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse    # ← 这里加上 JSONResponse
import traceback                                            # ← 加上 traceback
import logging                    # ← 新增
import threading
import numpy as np
from skopt import Optimizer
from skopt.space import Real
from statsmodels.stats.proportion import proportion_confint

# —— 日志配置 —— 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI(title="报价决策评估系统")

# —— HTML 页面（内嵌） ——
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>报价决策评估系统</title>
  <style>
    body{font-family:sans-serif; max-width:800px; margin:2em auto;}
    label{display:block; margin:0.5em 0;}
    input,select{width:100%; padding:0.5em;}
    button{padding:0.5em 1em;}
    pre{background:#f5f5f5; padding:1em;}

    /* 放在 <style> 里，或单独的 .css 文件 */
    .range-row {
    display: flex;
    align-items: center;
    gap: 8px;              /* 两个框之间留一点间距 */
    }

    .range-row input {
    flex: 1;               /* 两个框等分父容器 */
    padding: 0.5em;
    box-sizing: border-box;
    }

    /* 给第二个 input 加左边粗线，模拟分隔条 */
    .range-row input + input {
    border-left: 4px solid #333;  /* 粗度和颜色可按需调整 */
    padding-left: 0.75em;         /* 给内容留出一点空间 */
    }

    table { border-collapse: collapse; width: 100%; }
    th, td {
        border: 1px solid #ccc;
        padding: 6px;
        text-align: center;    /* 居中 */
    }
    th { background: #f0f0f0; }
    .highlight {
        background-color: rgba(255, 0, 0, 0.2);
    }


  </style>
</head>
<body>
  <h1>报价决策评估系统</h1>
  <!-- 输入表单 -->
  <h2>公共参数</h2>
  <label>我的报价 P:
    <input id="P" type="number" step="0.01" value="958.9">
  </label>
  <label>对手数量:
    <select id="rivals">
      <option>0</option><option>1</option><option selected>2</option><option>3</option>
      <option>4</option><option>5</option><option>6</option><option>7</option><option>8</option>
    </select>
  </label>
  <label>对手报价区间:
    <div class="range-row">
        <input id="low_opp" type="number" step="0.01" placeholder="下限" value="450">
        <input id="high_opp" type="number" step="0.01" placeholder="上限" value="470">
    </div>
  </label>
  <h2>计算胜率最高的伙伴报价</h2>
  <label>伙伴报价搜索区间:
    <div class="range-row">
        <input id="low_pat"  type="number" step="0.01" placeholder="最低价 L下限" value="450">
        <input id="high_pat" type="number" step="0.01" placeholder="最高价 H上限" value="1200">
    </div>
  </label>
    <label>胜利阈值（分）≥
    <input id="win_thr" type="number" value="80">
  </label>
  <label>Monte-Carlo 抽样次数 N_MC:
    <input id="n_mc" type="number" value="500000">
  </label>
  <label>优化器类型:
  <select id="estimator">
    <option value="RF" selected>随机森林</option>
    <option value="GP">高斯过程</option>
  </select>
</label>
<label>采集函数（acq_func）:
  <select id="acq_func">
    <option value="EI" selected>期望改进（EI）</option>
    <option value="PI">概率改进（PI）</option>
    <option value="LCB">下置信界（LCB）</option>
  </select>
</label>
  <label>贝叶斯总调用次数 n_calls:
    <input id="n_calls" type="number" value="50">
  </label>
  <label>初始随机点次数 n_init:
    <input id="n_init" type="number" value="15">
  </label>
  <button id="btn">计算最优报价</button>


<h2>评估指定的伙伴报价的胜率</h2>
  <label>伙伴报价 L, M, H:
    <div class="range-row">
      <input id="eval-L" type="number" step="0.01" placeholder="L" value="650">
      <input id="eval-M" type="number" step="0.01" placeholder="M" value="800">
      <input id="eval-H" type="number" step="0.01" placeholder="H" value="1200">
    </div>
  </label>
  <label>胜利阈值（分）≥
    <input id="eval-win_thr" type="number" value="80">
  </label>
  <label>Monte-Carlo 抽样次数 N_MC:
    <input id="eval-n_mc" type="number" value="500000">
  </label>
  <button id="eval-btn">计算报价胜率</button>




  <h2>结果</h2>
  <pre id="out">请点击“计算最优报价”或”计算报价胜率“</pre>

  <h3>采样结果</h3>
  <div id="sample-panel" style="overflow-x:auto; max-width:800px;">
    <!-- table 会被 JS 插入到这里 -->
  </div>
  
  <!-- 用户说明链接 -->
  <div style="margin-top: 20px;">
    <a href="/user_guide" target="_blank">查看用户说明</a>
  </div>

<script>
document.getElementById('btn').onclick = async () => {
  const out   = document.getElementById('out');
  const panel = document.getElementById('sample-panel');
  panel.innerHTML = '';       // 清空旧表格
  out.textContent  = '评估中…';

  // 构造请求参数
  const params = new URLSearchParams({
    P:       document.getElementById('P').value,
    rivals:  document.getElementById('rivals').value,
    low_opp:  document.getElementById('low_opp').value,
    high_opp: document.getElementById('high_opp').value,
    low_pat:  document.getElementById('low_pat').value,
    high_pat: document.getElementById('high_pat').value,
    n_mc:     document.getElementById('n_mc').value,
    n_calls:  document.getElementById('n_calls').value,
    n_init:   document.getElementById('n_init').value,
    win_thr:  document.getElementById('win_thr').value,
    estimator: document.getElementById('estimator').value,
    acq_func: document.getElementById('acq_func').value
  });

  try {
    const res = await fetch('/optimize?' + params.toString());
    if (!res.ok) {
      const txt = await res.text();
      out.textContent = '后端出错：\\n' + txt;
      return;
    }
    const data = await res.json();


    // ——— 无对手分支：把提示和伙伴报价都放到 `out` 里 ———
    if (data["提示"]) {
      let text = data["提示"] + "\\n\\n";

      if (data["K_used"] !== undefined) {
        text += `使用 K = ${data["K_used"]}\\n`;
      }
      if (data["有效伙伴报价"]) {
        const eff = data["有效伙伴报价"];
        text += `有效伙伴报价：L1 = ${eff.L1}，L2 = ${eff.L2}\\n`;
      }
      if (data["最高价伙伴报价"] !== undefined) {
        text += `最高价伙伴报价：${data["最高价伙伴报价"]}`;
      }

      out.textContent = text;
      return;
    }

      // ——— 错误分支 ———
      if (data["错误"]) {
        out.textContent = data["错误"];
        return;
      }

    // 1. 展示优化结果到 pre
    out.textContent = JSON.stringify(data["优化结果"], null, 2);

    // 2. 构建采样结果表格
    const samples = data["采样结果"] || [];
    if (samples.length === 0) {
      panel.textContent = '没有采样结果';
      return;
    }

    // 创建表格元素
    const table = document.createElement('table');

    // —— 构建表头 —— 
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
     "K 值", "我方分", "对手最高分",
     "≥阈值", ">对手最高分", "胜利"]
      .forEach(text => {
        const th = document.createElement('th');
        th.textContent = text;
        headerRow.appendChild(th);
      });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // —— 构建表体 —— 
    const tbody = document.createElement('tbody');
    samples.forEach(item => {
      const row = document.createElement('tr');
      const bids = item["对手出价"]; // 固定为长度 8 的数组（不足以 "--" 占位）
      const oppScores = item["对手得分列表"];
      const maxIdx = item["最高得分对手索引"];
      // 按列插入 R1…R5
      bids.forEach((b,idx) => {
        const td = document.createElement('td');
        td.textContent = b;
        if (idx === maxIdx) {
          td.classList.add('highlight');
        }
        row.appendChild(td);
      });
      // 插入剩余列
      [
        item["K值"],
        item["我方得分"],
        item["对手最高得分"],
        item["我方得分>=阈值"],
        item["我方得分>对手最高得分"],
        item["胜利"]
      ].forEach((val,idx) => {
        const td = document.createElement('td');
        td.textContent = val;
        if (idx === 2) {
            td.classList.add('highlight');
        }
        // 当胜利值为true时，设置该行字体颜色为绿色
        if (idx === 5 && val === true) {
            row.style.color = 'green';
        }
        row.appendChild(td);
      });
      tbody.appendChild(row);
    });
    table.appendChild(tbody);

    // 把表格插入面板
    panel.appendChild(table);

  } catch (e) {
    out.textContent = '出错：' + e;
  }
};

document.getElementById('eval-btn').onclick = async () => {
  const out   = document.getElementById('out');
  const panel = document.getElementById('sample-panel');
  panel.innerHTML = '';    // 清空旧表格
  out.textContent = '评估中…';

  const params = new URLSearchParams({
    P:      document.getElementById('P').value,
    rivals: document.getElementById('rivals').value,
    low_opp:  document.getElementById('low_opp').value,
    high_opp: document.getElementById('high_opp').value,
    L:      document.getElementById('eval-L').value,
    M:      document.getElementById('eval-M').value,
    H:      document.getElementById('eval-H').value,
    win_thr: document.getElementById('eval-win_thr').value,
    n_mc:   document.getElementById('eval-n_mc').value
  });

  try {
    const res  = await fetch('/evaluate?' + params);
    const data = await res.json();
    if (data.error) {
      out.textContent = '错误：' + data.error;
      return;
    }

    // 1. 先把伙伴报价、胜率和置信区间写到 out 区
// 拼伙伴报价
let text = `伙伴报价: [${data['伙伴报价'].join(', ')}]`;

// 如果有胜率字段，就拼胜率和置信区间
if (data['胜率'] !== undefined) {
  text += `\n胜率: ${data['胜率']}` +
          `\n95%置信区间: [${data['95%置信区间'].join(', ')}]`;
}
// 否则就拼我方得分
else {
  text += `\n我方得分: ${data['我方得分']}` +
          `\n伙伴得分: [${data['伙伴得分'].join(', ')}]`;
}

out.textContent = text;




    // 2. 再把 data['采样结果'] 渲染成表格 —— 与 btn 一致
    const samples = data['采样结果'] || [];
    if (!samples.length) {
      panel.textContent = '没有采样结果';
      return;
    }
    const table = document.createElement('table');
    // 表头
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ["R1","R2","R3","R4","R5","R6","R7","R8","K 值","我方分","对手最高分","≥阈值",">对手最高分","胜利"]
      .forEach(t => {
        const th = document.createElement('th');
        th.textContent = t;
        headerRow.appendChild(th);
      });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // 表体
    const tbody = document.createElement('tbody');
    samples.forEach(item => {
      const row = document.createElement('tr');
      item["对手出价"].forEach((b,i) => {
        const td = document.createElement('td');
        td.textContent = b;
        if (i === item["最高得分对手索引"]) td.classList.add('highlight');
        row.appendChild(td);
      });
      [
        item["K值"],
        item["我方得分"],
        item["对手最高得分"],
        item["我方得分>=阈值"],
        item["我方得分>对手最高得分"],
        item["胜利"]
      ].forEach((v, idx) => {
        const td = document.createElement('td');
        td.textContent = v;
        if (idx === 2) td.classList.add('highlight');
        // 当胜利值为true时，设置该行字体颜色为绿色
        if (idx === 5 && v === true) {
            row.style.color = 'green';
        }
        row.appendChild(td);
      });
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    panel.appendChild(table);

  } catch (e) {
    out.textContent = '出错：' + e;
  }
};
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def homepage():
    return HTML_PAGE

@app.get("/user_guide", response_class=HTMLResponse)
def user_guide():
    with open("user_guide.html", "r", encoding="utf-8") as f:
        return f.read()

# 核心函数：计算 K（向量化版本） 新的评分逻辑
# 评标基准价(K)计算方法:
# 1.若有效评标价为小于等于 3 家时，则各投标人有效评标价的平均值 K 为评标基准价。【基准价计算公式:K= { (P1+P2+P3+...+Pn)/n)};(n≤3)】
# 2.若有效评标价为 4-6 家时，则去掉最高有效评标价后其余各投标人有效评标价的平均值 K 为评标基准价。【基准价计算公式:K= {(P1+P2+P3+... +Pn-1)/(n-1)};(4≤n≤6)】
# 3.若有效评标价为 7 家(含 7 家)以上时，则去掉最高有效评标价和最低有效评标价后其余各投标人有效评标价的平均值 K 为评标基准价。【基准价计算公式:K= {(P2+P3+P4+...+Pn-1)/(n-2)};(n≥ 7)】
def calc_K(prices: np.ndarray) -> np.ndarray:
    n = prices.shape[1]  # 当前有效评标价数量
    idx_max = prices.argmax(axis=1)  # 找出最高价的索引
    idx_min = prices.argmin(axis=1)  # 找出最低价的索引
    mask = np.ones_like(prices, dtype=bool)  # 创建掩码数组
    
    # 根据不同的区间应用不同的计算规则
    if n <= 3:
        # 有效评标价小于等于3家时，不排除任何价格
        denom = n
    elif 4 <= n <= 6:
        # 有效评标价为4-6家时，只去掉最高价
        mask[np.arange(prices.shape[0]), idx_max] = False
        denom = n - 1
    else:  # n >= 7
        # 有效评标价为7家及以上时，去掉最高价和最低价
        mask[np.arange(prices.shape[0]), idx_max] = False
        mask[np.arange(prices.shape[0]), idx_min] = False
        denom = n - 2
    
    # 计算加权平均值作为评标基准价K
    return (prices * mask).sum(axis=1) / denom


# 核心函数：按分数规则打分
# 报价分(100 分)计算办法:
# (1)若评标价 P 等于 K 时，得 80 分(基准分);
# (2)评标价 P 每高于评标基准价 1%在基准分基础上 扣 1 分，最低得 60 分;
# (3)评标价 P 每低于评标基准价 1%在基准分基础上 加 1 分，最高为 100 分;
# (4)若评标价 P 低于评标基准价 20%以上时，每再 低 1%在 100 分基础上扣 1 分;
# (5)若评标价 P 低于评标基准价 40%以上时，得 80 分;
# (6)中间采用插值法计算，分值计算保留小数点 后两位，小数点后第三位“四舍五入”。
def score(pd: np.ndarray, K: np.ndarray, thr: float = 0.0) -> np.ndarray:
    # 计算价格差异率（百分比）
    diff_ratio = (pd - K) / K * 100
    
    # 创建一个与输入形状相同的数组用于存储分数
    scores = np.zeros_like(diff_ratio, dtype=np.float64)
    
    # 条件1: 评标价等于基准价，得80分
    mask_eq = (pd == K)
    scores[mask_eq] = 80.0
    
    # 条件2: 评标价高于基准价
    mask_gt = (pd > K)
    # 每高1%扣1分，最低60分
    scores[mask_gt] = np.maximum(80.0 - diff_ratio[mask_gt], 60.0)
    
    # 条件3-5: 评标价低于基准价
    mask_lt = (pd < K)
    diff_ratio_lt = -diff_ratio[mask_lt]  # 转换为正数以便计算
    
    # 条件5: 低于40%以上，得80分
    mask_lt_40 = (diff_ratio_lt > 40)
    scores_lt_40 = np.full_like(diff_ratio_lt[mask_lt_40], 80.0)
    
    # 条件4: 低于20%以上但不超过40%，每再低1%在100分基础上扣1分
    mask_lt_20_40 = (diff_ratio_lt > 20) & (diff_ratio_lt <= 40)
    # 先加20分到达100分，然后每超过20%的部分扣1分
    scores_lt_20_40 = np.maximum(100.0 - (diff_ratio_lt[mask_lt_20_40] - 20), 80.0)
    
    # 条件3: 低于20%以内（含20%），每低1%加1分，最高100分
    mask_lt_20 = (diff_ratio_lt <= 20)
    scores_lt_20 = np.minimum(80.0 + diff_ratio_lt[mask_lt_20], 100.0)
    
    # 将各条件下的分数赋值回原数组
    scores_lt = np.zeros_like(diff_ratio_lt)
    scores_lt[mask_lt_40] = scores_lt_40
    scores_lt[mask_lt_20_40] = scores_lt_20_40
    scores_lt[mask_lt_20] = scores_lt_20
    
    # 将计算结果放回到总分数组中
    scores[mask_lt] = scores_lt
    
    # 四舍五入到两位小数
    scores = np.round(scores, 2)
    
    return scores



# Monte Carlo 估计胜率
def estimate_win_rate(
    P: float, L: float, M: float, H: float,
    rivals: int, low_opp: float, high_opp: float,
    win_thr: float, n_mc: int, rng: np.random.Generator
) -> float:
    # 模拟对手报价
    opp = rng.uniform(low_opp, high_opp, size=(n_mc, rivals))
    # 组装报价矩阵：P、L、M、H、对手
    prices = np.column_stack([
        np.full(n_mc, P),
        np.full(n_mc, L),
        np.full(n_mc, M),
        np.full(n_mc, H),
        opp
    ])
    K = calc_K(prices)
    my_sc = score(P, K, win_thr)
    opp_sc = score(opp, K[:, None], win_thr)
    win_flag = (my_sc >= win_thr) | (my_sc[:, None] > opp_sc).all(axis=1)
    return win_flag.mean()

@app.get("/evaluate")
def evaluate(
    P: float = Query(...),
    rivals: int = Query(...),
    low_opp: float = Query(...),
    high_opp: float = Query(...),
    L: float = Query(...),
    M: float = Query(...),
    H: float = Query(...),
    win_thr: float = Query(...),
    n_mc: int = Query(100000)
):
    """
    根据指定的伙伴报价 L, M, H 估计胜率。
    """
    try:

        if rivals == 0:
            # 1) 只用 P, L, M, H 计算一个 K
            prices = np.array([[P, L, M, H]])
            K_val  = float(calc_K_vectorized(prices)[0])

            # 2) 计算我方和三位伙伴的得分
            sc_P = float(score_single(P,   K_val, win_thr))
            sc_L = float(score_single(L,   K_val, win_thr))
            sc_M = float(score_single(M,   K_val, win_thr))
            sc_H = float(score_single(H,   K_val, win_thr))

            # 3) 返回时把伙伴的得分也带上
            return JSONResponse({
                "伙伴报价":   [L,    M,    H],
                "K值":       round(K_val, 2),
                "我方得分":   round(sc_P,   2),
                "伙伴得分": [
                    round(sc_L, 2),
                    round(sc_M, 2),
                    round(sc_H, 2)
                ]
            })

        rng = np.random.default_rng()
        rate = estimate_win_rate(P, L, M, H, rivals, low_opp, high_opp, win_thr, n_mc, rng)
        succ = int(rate * n_mc)
        ci_low, ci_high = proportion_confint(succ, n_mc, alpha=0.05, method="beta")
        # 2) 再追加 10 次单次采样，构造和 optimize 接口一致的 sample_results —— <<< 新增
        sample_results = []
        for _ in range(10):
            single_rivals = np.sort(rng.uniform(low_opp, high_opp, size=(rivals,)))
            price_vec = np.concatenate(([P, L, M, H], single_rivals))
            K_val     = float(calc_K_vectorized(price_vec[None, :])[0])
            my_sc     = float(score_single(P, K_val, win_thr))
            opp_scores= score_single(single_rivals, K_val, win_thr)
            opp_list  = list(np.round(opp_scores,2))
            opp_max   = float(np.max(opp_scores))
            max_idx   = int(np.argmax(opp_scores))
            ge_thr    = my_sc >= win_thr
            beat_opp  = my_sc > opp_max
            victory   = bool(ge_thr or beat_opp)
            bids      = list(np.round(single_rivals,2)) + ["--"]*(8-rivals)

            sample_results.append({
                "对手出价": bids,
                "对手得分列表": opp_list,
                "K值": round(K_val,2),
                "我方得分": round(my_sc,2),
                "对手最高得分": round(opp_max,2),
                "最高得分对手索引": max_idx,
                "我方得分>=阈值": ge_thr,
                "我方得分>对手最高得分": beat_opp,
                "胜利": victory
            })
        # <<< 新增完毕



        return JSONResponse({
            "伙伴报价": [L, M, H],
            "胜率": round(rate, 4),
            "95%置信区间": [round(ci_low,4), round(ci_high,4)],
            "采样结果": sample_results
        })
    except Exception as e:
        logging.error("evaluate error", exc_info=True)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})











# —— 核心计算函数 —— 
# def calc_K_vectorized(matrix):
#     n_total = matrix.shape[1]
#     idx_max = matrix.argmax(axis=1)
#     idx_min = matrix.argmin(axis=1)
#     mask = np.ones_like(matrix, dtype=bool)
#     mask[np.arange(matrix.shape[0]), idx_max] = False
#     if n_total >= 7:
#         mask[np.arange(matrix.shape[0]), idx_min] = False
#     denom = n_total - 1 if n_total <= 6 else n_total - 2
#     return (matrix * mask).sum(axis=1) / denom


def calc_K_vectorized(matrix):
    n_total = matrix.shape[1]  # 当前有效评标价数量
    idx_max = matrix.argmax(axis=1)  # 找出最高价的索引
    idx_min = matrix.argmin(axis=1)  # 找出最低价的索引
    mask = np.ones_like(matrix, dtype=bool)  # 创建掩码数组
    
    # 根据不同的区间应用不同的计算规则
    if n_total <= 3:
        # 有效评标价小于等于3家时，不排除任何价格
        denom = n_total
    elif 4 <= n_total <= 6:
        # 有效评标价为4-6家时，只去掉最高价
        mask[np.arange(matrix.shape[0]), idx_max] = False
        denom = n_total - 1
    else:  # n >= 7
        # 有效评标价为7家及以上时，去掉最高价和最低价
        mask[np.arange(matrix.shape[0]), idx_max] = False
        mask[np.arange(matrix.shape[0]), idx_min] = False
        denom = n_total - 2
    
    # 计算加权平均值作为评标基准价K
    return (matrix * mask).sum(axis=1) / denom


# def score_single(price_vec, K_vec, win_thr):
#     diff = (K_vec - price_vec) / K_vec * 100.0
#     return np.where(
#         diff < 0,
#         np.maximum(win_thr + diff, 60),
#         np.where(
#             diff <= 20,
#             win_thr + diff,
#             np.maximum(120 - diff, 60)
#         )
#     )

def score_single(price_vec, K_vec, win_thr):
    # 确保输入是NumPy数组，支持标量输入
    price_arr = np.array(price_vec)
    K_arr = np.array(K_vec)
    
    # 计算价格差异率（百分比）
    diff_ratio = (price_arr - K_arr) / K_arr * 100
    
    # 创建一个与输入形状相同的数组用于存储分数
    scores = np.zeros_like(diff_ratio, dtype=np.float64)
    
    # 条件1: 评标价等于基准价，得80分
    mask_eq = (price_arr == K_arr)
    scores[mask_eq] = 80.0
    
    # 条件2: 评标价高于基准价
    mask_gt = (price_arr > K_arr)
    # 每高1%扣1分，最低60分
    scores[mask_gt] = np.maximum(80.0 - diff_ratio[mask_gt], 60.0)
    
    # 条件3-5: 评标价低于基准价
    mask_lt = (price_arr < K_arr)
    diff_ratio_lt = -diff_ratio[mask_lt]  # 转换为正数以便计算
    
    # 条件5: 低于40%以上，得80分
    mask_lt_40 = (diff_ratio_lt > 40)
    scores_lt_40 = np.full_like(diff_ratio_lt[mask_lt_40], 80.0)
    
    # 条件4: 低于20%以上但不超过40%，每再低1%在100分基础上扣1分
    mask_lt_20_40 = (diff_ratio_lt > 20) & (diff_ratio_lt <= 40)
    # 先加20分到达100分，然后每超过20%的部分扣1分
    scores_lt_20_40 = np.maximum(100.0 - (diff_ratio_lt[mask_lt_20_40] - 20), 80.0)
    
    # 条件3: 低于20%以内（含20%），每低1%加1分，最高100分
    mask_lt_20 = (diff_ratio_lt <= 20)
    scores_lt_20 = np.minimum(80.0 + diff_ratio_lt[mask_lt_20], 100.0)
    
    # 将各条件下的分数赋值回原数组
    scores_lt = np.zeros_like(diff_ratio_lt)
    scores_lt[mask_lt_40] = scores_lt_40
    scores_lt[mask_lt_20_40] = scores_lt_20_40
    scores_lt[mask_lt_20] = scores_lt_20
    
    # 将计算结果放回到总分数组中
    scores[mask_lt] = scores_lt
    
    # 四舍五入到两位小数
    scores = np.round(scores, 2)
    
    return scores



@app.get("/optimize")
def optimize(
    P:        float = Query(942.86),
    rivals:   int   = Query(5, ge=0, le=20),
    low_opp:  float = Query(650.0),
    high_opp: float = Query(700.0),
    low_pat:  float = Query(600.0),
    high_pat: float = Query(1200.0),
    n_mc:     int   = Query(160000),
    n_calls:  int   = Query(200),
    n_init:   int   = Query(50),
    win_thr:  float = Query(80.0),
    estimator: str  = Query("RF"),
    acq_func: str  = Query("EI")    
):
    try:
        # —— 第一步：打印入参 —— 
        logging.info(f"optimize called with P={P}, rivals={rivals}, "
                     f"low_opp={low_opp}, high_opp={high_opp}, "
                     f"low_pat={low_pat}, high_pat={high_pat}, "
                     f"n_mc={n_mc}, n_calls={n_calls}, n_init={n_init}, "
                     f"win_thr={win_thr}")
        
        rng = np.random.default_rng()
        if rivals == 0:
            TARGET = 100.0
            # 1. 计算 K（由 S=80+(K-P)/K*100 且 S=TARGET 推出）
            dA = TARGET - 80.0
            K = P / (1 - dA / 100)

            # 2. 在 [low_pat, min(high_pat, P)) 区间内找 L1，使得 L2=3K-P-L1 也合法
            total = 3 * K - P
            for _ in range(500):
                L1 = rng.uniform(P, high_pat)
                L2 = total - L1
                # 要求 L2 也在 [low_pat, high_pat] 且至少有一个 < P
                if low_pat <= L1 <= high_pat and low_pat <= L2 <= high_pat:
                    L1, L2 = round(L1, 2), round(L2, 2)
                    break
            else:
                return JSONResponse({
                    "错误": f"无法在区间 [{low_pat}, {high_pat}] 内找到满足 TARGET={TARGET} 的两位伙伴报价。"
                })

            # 3. 随机生成第三个报价 L3，使其不低于前两者且 ≤ high_pat
            min_third = max(L1, L2)
            if min_third > high_pat:
                return JSONResponse({
                    "错误": f"有效报价最大值 {min_third} 超过 high_pat={high_pat}，请调整搜索区间。"
                })
            L3 = round(rng.uniform(min_third, high_pat), 2)

            # 返回
            return JSONResponse({
                "提示": f"无对手参与，满足我方得分={TARGET} 的要求，随机生成三位伙伴报价：",
                "K_used": round(K, 2),
                "有效伙伴报价": {"L1": L1, "L2": L2},
                "最高价伙伴报价": L3
            })




        def mc_loss(L, M, H):
            rivals_mat = rng.uniform(low_opp, high_opp, size=(n_mc, rivals))
            prices = np.column_stack([
                np.full(n_mc, P),
                np.full(n_mc, L),
                np.full(n_mc, M),
                np.full(n_mc, H),
                rivals_mat
            ])
            K = calc_K_vectorized(prices)
            my_sc  = score_single(P,         K,          win_thr)
            opp_sc = score_single(rivals_mat, K[:, None], win_thr)
            cond   = (my_sc >= win_thr) | (my_sc[:, None] > opp_sc).all(axis=1)
            return 1.0 - cond.mean()

        # 贝叶斯优化
        space = [
            Real(low_pat, high_pat, name="L"),
            Real(low_pat, high_pat, name="M"),
            Real(low_pat, high_pat, name="H"),
        ]
        opt = Optimizer(
            dimensions=space,
            base_estimator=estimator,
            acq_func=acq_func,
            random_state=42,
            n_initial_points=n_init
        )

        best_loss, best_x = 1.0, None
        for i in range(1, n_calls+1):
            x = opt.ask()
            loss = mc_loss(*x)
            opt.tell(x, loss)
            # —— 打印每次迭代的 L,M,H 和 对应 loss —— 
            logging.info(f"Iteration {i}/{n_calls}: x={x}, loss={loss:.4f}")
            if loss < best_loss:
                best_loss, best_x = loss, x
                logging.info(f"  → New best found: loss={best_loss:.4f}, x={best_x}")

        # 添加检查，确保best_x不为None
        if best_x is None:
            return JSONResponse({
                "错误": "优化过程中未能找到有效解，请增加迭代次数或调整参数范围。",
                "n_calls": n_calls,
                "n_mc": n_mc,
                "low_pat": low_pat,
                "high_pat": high_pat
            })
        
        L_opt, M_opt, H_opt = best_x
        win_prob = 1 - best_loss
        succ     = int(win_prob * n_mc)
        ci_low, ci_high = proportion_confint(succ, n_mc, alpha=0.05, method="beta")

        # —— 新增：对最优解再做 10 次“单次”采样并计算得分 —— 
        sample_results = []
        for _ in range(10):
            # 1) 随机生成一个对手出价向量（长度 = rivals）
            single_rivals = rng.uniform(low_opp, high_opp, size=(rivals,))
            # —— 按从小到大排序，保证 R1…R5 分别是最低到最高 —— 
            single_rivals = np.sort(single_rivals)

            # 2) 组装本次报价向量：P, L_opt, M_opt, H_opt, *single_rivals
            price_vec = np.concatenate(([P, L_opt, M_opt, H_opt], single_rivals))
            # 3) 计算 K（注意输入要扩成 2D）
            K_val = float(calc_K_vectorized(price_vec[None, :])[0])
            # 4) 计算得分
            my_score  = float(score_single(P, K_val, win_thr))
            # opp_scores = score_single(single_rivals, K_val, win_thr)
            # opp_max   = float(np.max(opp_scores))
            # 4) 计算每个对手的得分列表，并取最大
            opp_scores = score_single(single_rivals, K_val, win_thr)
            opp_scores_list = list(np.round(opp_scores, 2))
            opp_max   = float(np.max(opp_scores))
            # 最高分的对手索引，用于前端高亮
            max_idx   = int(np.argmax(opp_scores))
            # 5) 胜利判定
            ge_thr    = my_score >= win_thr
            beat_opp  = my_score > opp_max
            victory   = bool(ge_thr or beat_opp)

            # 6) 对手出价列固定 8 列
            bids = list(np.round(single_rivals, 2))
            bids += ["--"] * (8 - len(bids))

            sample_results.append({
                "对手出价": bids,
                "对手得分列表": opp_scores_list,    # 新增
                "K值": round(K_val, 2),
                "我方得分": round(my_score, 2),
                "对手最高得分": round(opp_max, 2),
                "最高得分对手索引": max_idx,        # 新增
                "我方得分>=阈值": ge_thr,
                "我方得分>对手最高得分": beat_opp,
                "胜利": victory
            })


        # 构造返回 JSON
        result = {
            "系统名称": "报价决策评估系统",
            # "参数概览": {
            #     "P": P,
            #     "对手数量": rivals,
            #     "对手区间": [low_opp, high_opp],
            #     "伙伴搜索区间": [low_pat, high_pat],
            #     "Monte_Carlo": n_mc,
            #     "贝叶斯总调用": n_calls,
            #     "初始随机点": n_init,
            #     "胜利阈值": win_thr
            # },
            "优化结果": {
                "最优_L": round(L_opt, 2),
                "最优_M": round(M_opt, 2),
                "最优_H": round(H_opt, 2),
                "胜率": round(win_prob, 4),
                "95%_置信区间": [round(ci_low, 4), round(ci_high, 4)]
            },
            "采样结果": sample_results
        }
        return JSONResponse(result)

    except Exception as e:
        logging.error("optimize error", exc_info=True)
        # —— except 与 try 保持相同缩进（4 个空格） —— 
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
