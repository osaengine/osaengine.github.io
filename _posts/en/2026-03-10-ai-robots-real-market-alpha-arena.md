---
layout: post
title: "AI Robots on the Real Market: What Alpha Arena and Other Benchmarks Teach Us"
description: "The first benchmark of AI traders with real money. Chinese models beat ChatGPT and Gemini. We analyze the results, winner strategies, and what this means for algotrading."
date: 2026-03-10
image: /assets/images/blog/ai_arena_benchmark.png
tags: [AI, LLM, Alpha Arena, benchmark, trading robots, DeepSeek, Qwen]
lang: en
---

Two weeks ago, I [analyzed the architecture of open-source robots](/en/blog/2026/03/03/how-opensource-robot-works-inside.html). Classic logic: indicators, signals, if-then.

Today — about AI that makes trading decisions on its own. No indicators. No rules. Just: "here's $10,000, trade."

And this isn't theory. In October-November 2025, [Alpha Arena](https://nof1.ai/) took place — **the first public benchmark of AI traders with real money**.

Six LLMs (ChatGPT, Claude, Gemini, Qwen 3 MAX, DeepSeek, Grok) each received $10,000 and traded cryptocurrency on [Hyperliquid DEX](https://hyperliquid.xyz/) for two weeks.

The results were shocking: **Chinese models crushed Western ones**. Qwen 3 MAX won. ChatGPT and Gemini lost over 60% of their capital.

## Why Alpha Arena Is a Breakthrough

Before Alpha Arena, LLM benchmarks measured knowledge and logic, not the ability to make money. Simulations suffer from overfitting, look-ahead bias, and absence of real slippage.

Alpha Arena uses **live money, live market, public audit trail**: $10,000 per model, real exchange with real liquidity, on-chain transparency, 17 days of live trading, no human intervention.

## Results: Shock and Awe

| Model | Final Capital | Change | Max Drawdown | Trades | Sharpe |
|-------|--------------|--------|--------------|--------|--------|
| **Qwen 3 MAX** | **$13,247** | **+32.5%** | -12% | 43 | 1.8 |
| DeepSeek | $12,891 | +28.9% | -15% | 67 | 1.5 |
| Claude | $11,204 | +12.0% | -18% | 89 | 0.9 |
| Grok | $9,687 | -3.1% | -22% | 124 | 0.2 |
| ChatGPT | $3,845 | **-61.6%** | -68% | 203 | -1.2 |
| Gemini | $3,412 | **-65.9%** | -71% | 187 | -1.4 |

**Key takeaways:** Chinese models took 1st and 2nd place. ChatGPT and Gemini lost >60%. More trades correlated with bigger losses. Claude was the only profitable Western model.

## Why Chinese Models Won

**1. Discipline vs. Aggression:** Qwen made 43 trades (2.5/day), never used leverage >2x. ChatGPT made 203 trades (12/day), used leverage up to 10x.

**2. Volatility Adaptation:** DeepSeek reduced position sizes in volatile periods. Gemini ignored volatility with fixed stop-losses.

**3. Training Data:** Qwen and DeepSeek trained on Chinese market data where high volatility is the norm. Crypto is closer to Chinese stocks than to the S&P 500.

## ChatGPT and Gemini's Failure

**Overconfidence:** ChatGPT used 5-10x leverage, turning correct directional calls into catastrophic losses when timing was off.

**FOMO:** Gemini opened positions on every 2%+ move, resulting in negative expected value per trade.

**Ignoring commissions:** ChatGPT paid ~10% of capital in commissions alone (203 trades at 0.05% each).

## Lessons for Algotraders

1. **Trading frequency kills** — more trades = worse results
2. **Leverage amplifies mistakes** — if untested, keep leverage <3x
3. **Adaptation beats optimization** — add a "high volatility mode"
4. **Win rate is overrated, R/R is underrated** — even 40% win rate profits with 1:3 R/R
5. **Commissions are real** — calculate Net Profit Factor after commissions

## What This Means for Algotrading's Future

**LLMs as signals, not strategies:** Use for sentiment analysis, pattern recognition, strategy generation — but not as autonomous traders.

**Hybrid approach:** Combine classical indicators with LLM context (market regime classification).

**Chinese LLMs enter the stage:** DeepSeek is open-source, 10x cheaper than ChatGPT API, and potentially better for volatile markets.

## Criticism of Alpha Arena

17 days and 6 models isn't statistically significant. Only crypto on one exchange. Prompts aren't disclosed. $10,000 is small capital where luck and leverage can dominate.

## Other AI Trading Benchmarks

- [Numerai](https://numer.ai/) — crowdsourced hedge fund with weekly prediction tournaments
- [Quantiacs](https://quantiacs.com/) — real money on best Python strategies
- [Kaggle](https://www.kaggle.com/) — financial prediction competitions

## Conclusions

Alpha Arena showed three important things:

1. **LLMs can trade** — but not all equally well
2. **Discipline beats intelligence** — fewer trades, less leverage, volatility adaptation
3. **Chinese models are competitive** — and in some tasks better than Western ones

For algotraders: don't rely on LLMs as autonomous traders, use them as tools (sentiment, ideas, debugging), study winner strategies (Qwen, DeepSeek), try Chinese LLM APIs (cheaper, sometimes better).

---

**Useful links:**

- [Alpha Arena Official Website](https://nof1.ai/)
- [Season 1 Results Analysis](https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/)
- [AI Trading Showdown Breakdown](https://www.iweaver.ai/blog/alpha-arena-ai-trader-showdown/)
- [Numerai](https://numer.ai/)
- [Quantiacs](https://quantiacs.com/)
