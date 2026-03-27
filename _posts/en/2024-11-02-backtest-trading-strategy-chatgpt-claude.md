---
layout: post
title: "Finding a Trading Idea with ChatGPT and Claude: From Data to Backtest"
description: "We explore how AI can be used for data analysis, finding inefficiencies, and creating a trading strategy using cryptocurrency minute-level data as an example."
date: 2024-11-02
image: /assets/images/blog/ai-trading-strategy-preview.png
tags: [ChatGPT, Claude]
lang: en
---

In this article, I decided to compare two popular services — [ChatGPT](https://chatgpt.com/) and [Claude.ai](https://claude.ai/) — and see how they handle the task of finding trading inefficiencies as of November 2024. I evaluated their functionality and ease of use to determine which one is better suited for data analysis and developing a profitable trading strategy.

To simplify data collection, I used **[Hydra](https://stocksharp.ru/store/%D1%81%D0%BA%D0%B0%D1%87%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BC%D0%B0%D1%80%D0%BA%D0%B5%D1%82-%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85/)** — arguably the best free tool for downloading market data.

I downloaded minute-level BTCUSDT data for 2024, which amounted to approximately 25 MB, and exported it to a CSV file.

![](/assets/images/blog/hydra_2.png)

![](/assets/images/blog/hydra_3.png)

Hydra has its own built-in analytics, but you'll see below how outdated it is compared to AI capabilities, where you don't even need to write code yourself:

![](/assets/images/blog/hydra_4.png)

However, the main part of my work was not data collection but rather analysis and the search for strategy ideas. Instead of manually looking for approaches, I decided to trust the AI and find out what strategies it would suggest, what patterns and inefficiencies it could identify in the data, and how to optimize parameters for testing. With the help of **ChatGPT**, I was able to not only conduct a detailed analysis but also run a backtest of the strategy on the data.

---

### Data Preparation

After receiving the minute-level data, I loaded it into Python (the AI wrote the code — I just typed in plain text what I needed from it) and started with preprocessing. This included assigning names to each column and merging the date and time into a single column to simplify time series analysis.

---

### Finding Inefficiencies with AI

After preprocessing the data, I asked the AI about possible inefficiencies and patterns that could be useful for strategy development. ChatGPT suggested several approaches:

1. **Volatility Clusters** — Hours with high volatility could be suitable for a momentum strategy.
2. **Mean Reversion Tendency** — When the price deviates from the average level, a mean reversion strategy could be applied.
3. **Momentum Patterns** — During certain hours, sustained price movements were observed, which could serve as signals for a trend-following strategy.

![](/assets/images/blog/volatility-clusters.png)

---

### Strategy Development

Based on the AI's suggestions, I chose two strategies for testing:

1. **Mean Reversion**: Opening a short position when the price deviates significantly above the average and a long position when it deviates below. Position is closed when the price returns to the mean.

2. **Momentum Strategy**: Opening a position in the direction of the trend during periods of elevated volatility. If the return is positive and above the threshold, a buy position is opened; if negative and below the threshold — a sell position.

For each strategy, basic entry and exit rules were defined, along with stop-losses for risk management.

![](/assets/images/blog/hourly-returns.png)

---

### Backtesting the Strategies

With the help of ChatGPT, I was also able to backtest both strategies to see how they would have performed on historical data. The test results showed the equity curve for the mean reversion strategy (see the chart below).

The chart shows how the portfolio's capitalization could have changed when following the strategy. You can see that the strategy demonstrated steady growth during certain periods, but there were also moments of drawdown. This confirms the importance of parameter tuning and risk management.

![](/assets/images/blog/mean-reversion-equity-curve.png)

---

### Claude.ai

During my work, I also tried using **Claude Sonnet** by Anthropic, which had recently announced its data analysis feature (more details [here](https://www.anthropic.com/news/analysis-tool)). The idea seemed promising: upload a 25 MB file for Claude to help with analysis.

![](/assets/images/blog/claude_analytics.png)

However, I ran into a number of difficulties. Unfortunately, the feature turned out to be rough and unfinished — my file wouldn't even upload. I ended up splitting it into smaller parts, but due to previous errors, I quickly hit the request limit. All I managed to get was an error when trying to build a chart.

![](/assets/images/blog/claude_error_1.png)

Although I love working with Claude, I hope the project's engineers will refine this feature and significantly expand the data upload window. This would enable more efficient analysis of large files and open up new possibilities for working with large volumes of data.

![](/assets/images/blog/claude_error_2.png)

---

### Conclusion

Using ChatGPT allowed me not just to analyze data but also to ask the AI questions about suitable methods for strategy creation. This approach not only generated new ideas but also helped quickly test hypotheses and get recommendations that might have gone unnoticed with a traditional approach. The method where AI helps discover ideas and strategy parameters opens up new possibilities for flexible and adaptive development of trading strategies.
