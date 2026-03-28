---
lang: en
layout: faq_article
title: "What metrics should be used to evaluate a trading robot's performance?"
section: practice
order: 6
---

Various metrics are used to analyze a trading robot's performance, helping to evaluate the effectiveness of the strategy and its resilience to market changes.

## Key metrics:

1. **Total profitability:**
   - Total profit earned over a specific period.
   - Helps determine how successful the strategy is overall.

2. **Maximum drawdown:**
   - The difference between the local maximum and minimum of the account balance.
   - Allows you to assess the risks associated with using the strategy.

3. **Sharpe Ratio:**
   - The ratio of average profit to the standard deviation of profit.
   - The higher the value, the more stable the strategy.

4. **Win/loss ratio:**
   - The percentage of successful trades out of the total number.
   - Important to consider together with the risk-reward ratio.

5. **Execution speed:**
   - The time between sending an order and its execution.
   - Critical for high-frequency strategies.

## Analysis tools:

- **MetaTrader:** The built-in strategy analyzer provides detailed metrics.
- **QuantConnect:** Allows evaluating strategies in a cloud environment.
- **StockSharp Designer:** Suitable for comprehensive analysis with result visualization.
- **TSLab:** Provides a convenient interface for risk and return analysis.

## Tips:

- Focus not only on profit but also on strategy stability.
- Choose metrics depending on your goals: for long-term trading, drawdown is important; for short-term trading, execution speed matters.
- Regularly compare the robot's results with benchmarks such as market indices.
