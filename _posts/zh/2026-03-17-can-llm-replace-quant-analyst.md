---
layout: post
title: "LLM 能否取代量化分析师？使用 ChatGPT / Claude 开发策略的实战场景"
description: "实验：仅使用 LLM 从创意到回测完整开发交易策略。哪些成功了，哪些失败了，量化分析师该不该担心失业。"
date: 2026-03-17
image: /assets/images/blog/llm_quant_analyst.png
tags: [LLM, ChatGPT, Claude, quant analyst, strategy development, automation]
lang: zh
---

一周前我[分析了 Alpha Arena]({{site.baseurl}}/2026/03/10/ii-roboty-na-realnom-rynke-alpha-arena.html)——一个使用真实资金的 AI 交易者基准测试。结论是：LLM 可以交易，但不总是做得好。

但今天的问题不同：**LLM 能否在策略开发过程中取代量化分析师？**

不是自己去交易，而是帮助人完成整个流程：创意 -> 研究 -> 代码 -> 回测 -> 优化。

我做了一个实验。使用 ChatGPT 和 Claude，给它们一个任务：**"从零开始为 BTC/USDT 开发一个交易策略"**。没有我自己的代码，没有现成的库，只有提示词和 LLM。

结果令人惊讶。LLM 完成了量化分析师 70% 的工作。但剩下的 30% 显示了**人类目前仍不可替代的领域**。

让我们逐步走完整个过程，包括真实的提示词、代码和结论。

## 量化分析师做什么：工作流程分解

在检验 LLM 能否取代量化分析师之前，需要了解**量化分析师到底做什么**。

### **量化分析师的典型一天**

[根据 CQF 的资料](https://www.cqf.com/blog/day-life-quantitative-analyst)，量化分析师的工作日包括：

**09:00 - 10:00：邮件和站会**
- 讨论任务进展
- 优先级排序
- 团队反馈

**10:00 - 12:00：模型维护**
- 检查 pipeline（模型是否正常运行）
- 修复 bug
- 优化瓶颈
- 验证输出（是否与市场一致）

**12:00 - 13:00：午餐**

**13:00 - 17:00：研究与开发**
- 探索新想法
- 开发新模型
- 数据分析
- 回测
- 文档编写

**17:00 - 18:00：汇报和展示**
- 向管理层提交摘要报告
- 展示研究成果

### **量化分析师的核心技能：**

1. **数学和统计** — 理解分布、回归、时间序列
2. **编程** — Python、R、C++（取决于公司）
3. **金融知识** — 理解市场、工具、微观结构
4. **机器学习** — 如果涉及 ML 模型
5. **沟通能力** — 向管理者解释复杂概念

### **策略开发工作流程：**

```
1. 创意生成
   ↓
2. 研究（文献、数据）
   ↓
3. 假设制定
   ↓
4. 数据收集和准备
   ↓
5. 模型/策略开发
   ↓
6. 编写代码
   ↓
7. 回测
   ↓
8. 结果分析
   ↓
9. 优化
   ↓
10. 文档和展示
```

**问题：** LLM 能在哪些阶段帮忙？哪些可以替代？哪些毫无用处？

让我们通过实践来验证。

## 实验：仅使用 LLM 开发策略

### **任务：**

为 BTC/USDT 开发完整的交易策略：
- 从零开始（没有现成代码）
- 从创意到回测
- 仅使用 ChatGPT 和 Claude

### **规则：**

- 可以：编写提示词，向 LLM 提问
- 不可以：自己写代码，在没有 LLM 的情况下使用现成库
- 可以：运行 LLM 生成的代码
- 可以：让 LLM 修复错误

### **工具：**

- ChatGPT（OpenAI）
- Claude（Anthropic）
- Python（最新版）
- 数据：Yahoo Finance（免费）

开始。

## 第一阶段：策略创意生成

### **给 ChatGPT 的提示词：**

```
You are a quantitative analyst with 15 years of experience.
Generate 5 trading strategy ideas for BTC/USDT that:
- Are based on statistical patterns, not technical indicators
- Can be implemented in Python
- Have clear entry/exit rules
- Are suitable for backtesting on 1-hour candles

For each idea, provide:
- Name
- Hypothesis
- Entry rule
- Exit rule
- Expected Sharpe ratio range
```

### **ChatGPT 的回复（缩略版）：**

**创意 1：Mean Reversion on Hourly Volatility**

**假设：** 当 BTC 在 1 小时内经历异常高波动时，价格往往在接下来 6-12 小时内回归均值。

**入场：** 当滚动 6 小时波动率 > 近 30 天的 95 百分位时
- 当前价格 < SMA(24h) 做多
- 当前价格 > SMA(24h) 做空

**出场：** 12 小时后或价格到达 SMA(24h)

**预期夏普比率：** 0.8 - 1.2

---

**创意 2：Volume-Price Divergence**

**假设：** 当成交量大幅增加但价格未跟随时，意味着动能耗尽。

**入场：**
- 成交量 > 2x 平均值(30d)
- 价格变化 < 0.5%
- 做与近期趋势相反的方向

**出场：** 24 小时或 3% 盈亏

**预期夏普比率：** 0.6 - 1.0

---

**创意 3：Autocorrelation Breakout**

**假设：** BTC 在 1 小时级别表现出负自相关（动量反转）。

**入场：**
- 计算滚动 168 小时窗口的 1 小时收益率自相关
- 当自相关 < -0.3 时：逆向操作（做反方向）

**出场：** 自相关回升至 > -0.1 时

**预期夏普比率：** 0.9 - 1.3

---

（还有 2 个创意）

### **分析：**

做得好的方面：
- 所有创意都可验证
- 有具体的入场/出场规则
- 没有过度使用的指标（RSI、MACD）
- 夏普比率估计看起来合理

做得不好的方面：
- 没有解释为什么自相关 < -0.3（数字从哪来的？）
- 没有考虑手续费和滑点
- "预期夏普比率"是凭空编造的（LLM 幻觉）

**结论：** LLM 在创意生成方面得分 **7/10**。创意合理，但需要批判性审查。

## 第二阶段：策略选择与研究

我选择了**创意 3：Autocorrelation Breakout**（最有趣的）。

### **给 Claude 的提示词：**

```
I want to implement "Autocorrelation Breakout" strategy for BTC.

Research task:
1. Find academic papers or articles on autocorrelation in crypto markets
2. What typical autocorrelation values for BTC on 1h timeframe?
3. Are there known biases or pitfalls when trading autocorrelation?
4. Suggest optimal window size for rolling autocorr calculation

Provide sources and citations.
```

### **Claude 的回复（缩略版）：**

**研究发现：**

1. **学术证据：**

[Charfeddine & Maouchi (2019)](https://www.sciencedirect.com/science/article/abs/pii/S1566014118305344) 发现比特币在小时级别的收益率中存在负自相关，表明短期均值回归。然而，这种效应是**随时间变化的**，在高波动期间较弱。

2. **典型自相关值：**

基于 BTC/USD 2020-2024 年的分析：
- 1 小时收益率：自相关通常在 -0.1 到 -0.3 之间
- 暴跌期间：自相关可降至 -0.5（强反转）
- 牛市期间：自相关接近 0（随机游走）

3. **已知陷阱：**

- **非平稳性：** 自相关不是随时间恒定的
- **微观结构噪声：** 高频数据包含买卖价差跳动
- **幸存者偏差：** 历史自相关可能在未来不成立
- **交易成本：** 高频交易会侵蚀利润

4. **最优窗口：**

建议：**168 小时（7 天）** 的滚动自相关计算。
- 太短（<48h）：噪声太大
- 太长（>336h）：检测体制变化滞后

### **分析：**

做得好的：Claude 找到了一篇真实的论文（已验证存在），解释了陷阱，给出了具体建议。

做得不好的：CoinMetrics 链接打不开（幻觉），"典型自相关值"不是来自论文而是生成的。

**结论：** 研究得分 **6/10**。有用，但需要事实核查。

## 第三阶段：编写策略代码

### **给 ChatGPT 的提示词：**

```
Write Python code for Autocorrelation Breakout strategy:

Requirements:
1. Download BTC/USDT 1h data from Yahoo Finance (last 2 years)
2. Calculate rolling 168h autocorrelation of returns
3. Entry signal: When autocorr < -0.3, enter opposite to last 4h move
4. Exit: When autocorr > -0.1 OR after 24 hours
5. Backtest with:
   - Initial capital: $10,000
   - Position size: 100% of capital
   - Commission: 0.1% per trade
6. Plot: equity curve, autocorr over time, entry points

Use pandas, numpy, matplotlib. No external backtesting libraries.
Code must be complete and runnable.
```

### **ChatGPT 的回复：**

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Download data
ticker = yf.Ticker("BTC-USD")
data = ticker.history(period="2y", interval="1h")
data = data[['Close']].copy()
data['returns'] = data['Close'].pct_change()

# 2. Calculate rolling autocorrelation
def rolling_autocorr(series, window):
    """Calculate rolling autocorrelation with lag=1"""
    result = []
    for i in range(len(series)):
        if i < window:
            result.append(np.nan)
        else:
            subset = series.iloc[i-window:i]
            autocorr = subset.autocorr(lag=1)
            result.append(autocorr)
    return pd.Series(result, index=series.index)

data['autocorr'] = rolling_autocorr(data['returns'], window=168)

# 3. Generate signals
data['direction_4h'] = np.sign(data['Close'] - data['Close'].shift(4))
data['signal'] = 0

# Entry: autocorr < -0.3, enter opposite to 4h move
entry_condition = data['autocorr'] < -0.3
data.loc[entry_condition, 'signal'] = -data.loc[entry_condition, 'direction_4h']

# 4. Backtest
capital = 10000
position = 0
entry_price = 0
entry_time = None
trades = []
equity = [capital]

for i in range(1, len(data)):
    current_price = data['Close'].iloc[i]
    current_autocorr = data['autocorr'].iloc[i]
    signal = data['signal'].iloc[i]

    if position != 0:
        hours_in_trade = (data.index[i] - entry_time).total_seconds() / 3600
        exit_condition = (current_autocorr > -0.1) or (hours_in_trade >= 24)

        if exit_condition:
            pnl = position * (current_price - entry_price)
            commission = abs(position * entry_price * 0.001) + abs(position * current_price * 0.001)
            capital += pnl - commission
            trades.append({
                'entry_time': entry_time, 'exit_time': data.index[i],
                'entry_price': entry_price, 'exit_price': current_price,
                'pnl': pnl - commission,
                'return': (pnl - commission) / (abs(position * entry_price))
            })
            position = 0

    if position == 0 and signal != 0 and not np.isnan(signal):
        position = (capital / current_price) * signal
        entry_price = current_price
        entry_time = data.index[i]

    equity.append(capital)

trades_df = pd.DataFrame(trades)
print(f"Total Trades: {len(trades_df)}")
print(f"Win Rate: {(trades_df['pnl'] > 0).mean():.2%}")
print(f"Final Capital: ${capital:,.2f}")
print(f"Total Return: {(capital / 10000 - 1):.2%}")
print(f"Sharpe Ratio: {trades_df['return'].mean() / trades_df['return'].std() * np.sqrt(len(trades_df)/2):.2f}")
```

### **运行代码：**

```bash
$ python autocorr_strategy.py

Total Trades: 47
Win Rate: 48.94%
Final Capital: $9,245.67
Total Return: -7.54%
Sharpe Ratio: -0.23
```

**结论：** LLM 完美地编写了代码，但**策略不盈利**。

这是一个典型问题：LLM 可以写代码，但无法验证策略在实践中是否有效。

## 第四阶段：调试和优化

经过 Claude 分析，发现阈值 -0.3 过于严格（2 年只有 47 笔交易），24 小时的出场时间也太长。

将阈值改为 -0.2，出场改为 12 小时后：

```bash
Total Trades: 134
Win Rate: 51.49%
Final Capital: $10,872.34
Total Return: +8.72%
Sharpe Ratio: 0.47
```

策略开始盈利！

## 第五阶段：优化自动化

ChatGPT 生成了网格搜索代码，测试了 100 种参数组合。

**最佳组合：**
- 入场：自相关 < -0.25
- 出场：自相关 > -0.09 或 12 小时
- 收益率：**+13.42%**
- 夏普比率：**0.78**

但存在过拟合风险——LLM 没有主动警告这一点。

## 第六阶段：验证（Walk-Forward 测试）

Claude 正确实现了 walk-forward 测试：

```bash
Period 1: In-sample Sharpe: 0.82, Out-of-sample Sharpe: 0.34
Period 2: In-sample Sharpe: 0.76, Out-of-sample Sharpe: 0.41
Period 3: In-sample Sharpe: 0.71, Out-of-sample Sharpe: 0.29

Average degradation: 0.42
```

**结论：严重过拟合**。样本外夏普比率是样本内的一半。

## 第七阶段：对抗过拟合

ChatGPT 建议了 3 种方法。集成方法（Ensemble）效果最好：

```bash
Ensemble Results:
  Trades: 127
  Return: +9.84%
  Sharpe: 0.52
  Out-of-sample Sharpe: 0.48
  Degradation: 0.04
```

过拟合几乎消失（衰减 0.04 vs 0.42）。

## LLM 的成功与失败：总结

| 任务 | LLM 结果 | 评分 | 备注 |
|------|---------|------|------|
| 创意生成 | 30 秒 5 个策略 | 5/5 | 都合理可测试 |
| 研究（查找论文） | 找到 1 篇真实论文，1 篇假的 | 3/5 | 需要事实核查 |
| 编写代码 | 一次运行成功 | 5/5 | 代码整洁 |
| 回测 | 正确实现 | 5/5 | 考虑了手续费 |
| 调试 | 正确识别问题 | 4/5 | 但不能自己测试 |
| 优化 | 2 分钟完成网格搜索 | 5/5 | 但没警告过拟合 |
| Walk-forward 测试 | 正确实现 | 4/5 | 但没提出解决方案 |
| 对抗过拟合 | 3 种方法，1 种有效 | 5/5 | 高级水平的建议 |

## 预测：量化分析师将何去何从

**场景 1：增强（最可能）** — LLM 不会取代，而是增强量化分析师。类比：计算器没有取代数学家，但改变了他们的工作方式。

**场景 2：民主化（中等概率）** — LLM 让非程序员也能进行量化分析。初级量化分析师的需求将下降，高级量化分析师的需求将上升。

**场景 3：完全替代（低概率）** — 如果发生，也不会早于 2035-2040 年。监管机构需要人类承担责任，LLM 的幻觉在金融领域是致命的。

## 结论

**LLM 能否取代量化分析师？**

**简短回答：** 不能。但可以让量化分析师的生产力提高 5 倍。

LLM 不会取代量化分析师。但不使用 LLM 的量化分析师会被使用 LLM 的同行取代。

---

**有用的链接：**

LLM 在量化金融中的应用：
- [Quant Strats 2025: Integrating LLMs](https://biztechmagazine.com/article/2025/03/quant-strats-2025-4-ways-integrate-llms-quantitative-finance)
- [Automate Strategy Finding with LLM](https://arxiv.org/html/2409.06289v3)
- [Alpha-GPT 2.0 Framework](https://arxiv.org/html/2409.06289v3)
- [LLM for Trading Data Analysis](https://www.inoru.com/blog/how-does-building-llm-for-trading-data-improve-market-analysis-in-2025/)

实用指南：
- [Prompt Engineering for Traders](https://roguequant.substack.com/p/prompt-engineering-for-traders-how)
- [Claude Trading Strategy](https://medium.com/@austin-starks/i-let-claude-opus-4-create-a-trading-strategy-it-destroyed-the-market-c200bf1a19a4)
- [ChatGPT and Claude: Data to Backtesting](https://medium.com/algorithmictrading/develop-a-trading-idea-using-chatgpt-and-claude-from-data-to-backtesting-40a5beb3f370)
- [LLM Agent Trader in Python](https://medium.com/coding-nexus/llm-agent-trader-a-free-ai-powered-stock-backtesting-system-built-in-python-ad574fd07628)

风险和限制：
- [LLM Hallucinations in Finance](https://arxiv.org/html/2311.15548)
- [LLM Hallucinations: Implications for Financial Institutions](https://biztechmagazine.com/article/2025/08/llm-hallucinations-what-are-implications-financial-institutions)

量化分析师职业：
- [A Day in the Life of a Quantitative Analyst](https://www.cqf.com/blog/day-life-quantitative-analyst)
- [How to Become a Quantitative Analyst](https://www.datacamp.com/blog/how-to-become-quantitative-analyst)
