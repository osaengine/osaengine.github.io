---
lang: en
layout: faq_article
title: "What is backtesting and forward testing?"
section: practice
order: 3
---

Backtesting and forward testing are key stages of trading robot testing that allow you to verify their effectiveness and stability before deploying with real money.

## Backtesting:

1. **What is it?**
   Backtesting is testing a trading strategy on historical data to evaluate its past behavior.

2. **How does it work?**
   - The robot applies the algorithm to historical data as if it were operating in real time.
   - Key metrics are analyzed: profitability, maximum drawdown, risk-reward ratio.

3. **Backtesting tools:**
   - **[StockSharp Designer](https://stocksharp.com/):** Provides a convenient interface for visual backtesting and result analysis.
   - **[MetaTrader](https://www.metatrader4.com/):** Integrated strategy tester.
   - **[QuantConnect](https://www.quantconnect.com/):** Supports testing on large data volumes.

## Forward testing:

1. **What is it?**
   Forward testing is testing a strategy on real market data in real time, but without using real capital.

2. **How does it work?**
   - The robot operates on a demo account or in test mode.
   - It verifies how the algorithm reacts to current market conditions, delays, spreads, and other factors.

## Why is this important?

- Backtesting helps identify strategy weaknesses based on historical data.
- Forward testing shows how the robot performs in real market conditions without the risk of losses.

## Tips:

- Use high-quality historical data for backtesting.
- Conduct forward testing for at least 1-2 weeks to confirm strategy stability.
- Compare the results of both tests to assess algorithm reliability.
