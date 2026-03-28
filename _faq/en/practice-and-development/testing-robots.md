---
lang: en
layout: faq_article
title: "How to test a trading robot before launching?"
section: practice
order: 2
---

Testing a trading robot before launching it in live trading is a key step that helps avoid errors and minimize risks.

## Testing stages:

1. **Backtesting:**
   - Verifying the strategy on historical data.
   - Performance indicators are evaluated: profitability, drawdown, risk-reward ratio.

2. **Forward testing:**
   - Testing the robot in real time on a demo account.
   - Checking how the algorithm behaves in current market conditions.

3. **Performance monitoring:**
   - Measuring data processing speed and order submission time.
   - Verifying the stability of the exchange connection.

4. **Error analysis:**
   - Logging the robot's actions to identify issues.
   - Making adjustments to the strategy and code.

## Testing tools:

- **[StockSharp Designer](https://stocksharp.ru/):** A universal tool for visual strategy testing, backtesting, and robot performance analysis.
- **[MetaTrader](https://www.metatrader4.com/):** Built-in features for backtesting and strategy optimization.
- **[QuantConnect](https://www.quantconnect.com/):** A platform for cloud-based algorithm testing.
- **[TradingView](https://www.tradingview.com/):** Simple data visualization and strategy testing.

## Tips:

- Use as much data as possible for backtesting to account for different market phases.
- Do not over-optimize the algorithm to avoid overfitting.
- After successful testing on a demo account, start with a small amount of real capital.
