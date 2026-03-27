---
layout: post
title: "AI机器人在真实市场上：Alpha Arena和其他基准测试教给我们什么"
description: "首个用真实资金进行的AI交易者基准测试。中国模型击败了ChatGPT和Gemini。分析结果、获胜者策略以及这对算法交易意味着什么。"
date: 2026-03-10
image: /assets/images/blog/ai_arena_benchmark.png
tags: [AI, LLM, Alpha Arena, benchmark, trading robots, DeepSeek, Qwen]
lang: zh
---

两周前，我[分析了开源机器人的架构](/zh/blog/2026/03/03/how-opensource-robot-works-inside.html)。经典逻辑：指标、信号、if-then。

今天——关于自主做出交易决策的AI。没有指标，没有规则。只是："给你10,000美元，去交易。"

2025年10-11月，[Alpha Arena](https://nof1.ai/)举行——**首个用真实资金进行的AI交易者公开基准测试**。

六个LLM（ChatGPT、Claude、Gemini、Qwen 3 MAX、DeepSeek、Grok）各获得10,000美元，在[Hyperliquid DEX](https://hyperliquid.xyz/)上交易加密货币两周。

结果令人震惊：**中国模型以压倒性优势击败了西方模型**。Qwen 3 MAX获胜。ChatGPT和Gemini损失了超过60%的资本。

## 结果

| 模型 | 最终资本 | 变化 | 最大回撤 | 交易次数 | Sharpe |
|------|---------|------|---------|---------|--------|
| **Qwen 3 MAX** | **$13,247** | **+32.5%** | -12% | 43 | 1.8 |
| DeepSeek | $12,891 | +28.9% | -15% | 67 | 1.5 |
| Claude | $11,204 | +12.0% | -18% | 89 | 0.9 |
| Grok | $9,687 | -3.1% | -22% | 124 | 0.2 |
| ChatGPT | $3,845 | **-61.6%** | -68% | 203 | -1.2 |
| Gemini | $3,412 | **-65.9%** | -71% | 187 | -1.4 |

## 为什么中国模型获胜

**1. 纪律vs激进：** Qwen 43笔交易，杠杆不超过2x。ChatGPT 203笔交易，杠杆高达10x。

**2. 波动性适应：** DeepSeek在高波动期间减少仓位。Gemini忽略波动性。

**3. 训练数据：** Qwen和DeepSeek在中国市场数据上训练，那里高波动是常态。加密市场更接近中国股票而非S&P 500。

## 对算法交易者的教训

1. **交易频率扼杀收益** — 更多交易 = 更差结果
2. **杠杆放大错误** — 未经测试时保持杠杆<3x
3. **适应比优化更重要** — 添加"高波动模式"
4. **胜率被高估，风险回报比被低估**
5. **佣金是真实的支出** — 计算扣除佣金后的净利润因子

## 对算法交易未来的意义

LLM应作为信号工具而非独立策略。混合方法：将经典指标与LLM上下文结合。中国LLM登上舞台：DeepSeek开源，API价格比ChatGPT便宜10倍。

---

**有用链接：**

- [Alpha Arena](https://nof1.ai/)
- [Season 1 Results Analysis](https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/)
- [Numerai](https://numer.ai/)
- [Quantiacs](https://quantiacs.com/)
