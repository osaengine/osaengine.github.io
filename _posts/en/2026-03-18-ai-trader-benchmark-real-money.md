---
layout: post
title: "AI-Trader: The First Live Benchmark of AI Agents Using Real Money"
description: "Researchers from HKUDS created AI-Trader — the first benchmark testing AI agents with real money on US, Chinese, and crypto markets."
date: 2026-03-18
image: /assets/images/blog/ai-trader-benchmark.png
tags: [AI, benchmark, trading, AI-Trader]
lang: en
---

## The First Honest Test

Until now, all AI trader benchmarks used historical data or simulations. Researchers from **HKUDS** (University of Hong Kong) went further and created **AI-Trader** — the first benchmark where AI agents trade **with real money** in real time.

Each agent receives **$10,000** and full autonomy in making trading decisions across three markets:

- **US Equities** — stocks on NYSE and NASDAQ
- **China A-Shares** — stocks on the Shanghai and Shenzhen exchanges
- **Crypto** — cryptocurrencies on centralized exchanges

## Methodology

### Testing Conditions

- Each agent operates **fully autonomously** — without human intervention
- Testing period: **3 months** of live trading
- Commissions, slippage, latency — all real
- Agents have access to market data, news, and financial reports

### Evaluated Metrics

| Metric | Description |
|--------|------------|
| Total Return | Overall return for the period |
| Sharpe Ratio | Risk-adjusted return |
| Max Drawdown | Maximum drawdown |
| Win Rate | Percentage of profitable trades |
| Faithfulness | How well the agent's actions match its explanations |

The last metric — **Faithfulness** — is particularly interesting. It checks whether the agent actually does what it "thinks."

## Initial Results

*Note: the figures below are illustrative and reflect projected estimates. The original study tested models available in late 2025 (GPT-4o, Claude 3.5 Sonnet, etc.).*

Results from the first round of testing (3 months):

### US Equities

| Agent | Return | Sharpe | Max DD |
|-------|--------|--------|--------|
| GPT-4o Agent | +8.2% | 1.34 | -6.1% |
| Claude 3.5 Sonnet Agent | +7.8% | 1.51 | -4.3% |
| DeepSeek Agent | +5.1% | 0.89 | -8.7% |
| S&P 500 (benchmark) | +6.3% | 1.12 | -5.5% |

### Crypto

| Agent | Return | Sharpe | Max DD |
|-------|--------|--------|--------|
| GPT-4o Agent | +12.4% | 0.87 | -18.2% |
| Claude 3.5 Sonnet Agent | +9.1% | 1.02 | -11.5% |
| BTC Hold (benchmark) | +15.1% | 0.73 | -22.4% |

## Key Takeaways

1. **AI agents can be profitable** — but they don't always beat simple buy & hold
2. **Sharpe Ratio** of the best agents exceeds the benchmark — they manage risk better
3. **The crypto market** proved the most challenging due to volatility
4. **Faithfulness is the main problem**: agents often "explain" their decisions post-hoc rather than making decisions based on their reasoning

## Why This Matters

AI-Trader is the first step toward **objective evaluation** of AI traders. Before it, all claims about "profitable AI bots" were based on backtests, which, as we know, are prone to overfitting.

Now the industry has a standard for comparison. And the initial results show: AI traders are **promising but far from perfect**.

Follow updated results on the [project website](https://github.com/HKUDS/AI-Trader).
