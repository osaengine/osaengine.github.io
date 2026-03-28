---
lang: zh
layout: faq_article
title: "交易机器人需要哪些数据？"
section: technical
order: 2
---

交易机器人的运行需要访问各种类型的数据。这些数据确保算法正确决策和精确执行。

## 数据类型：

1. **市场数据：**
   - 当前价格（报价）和交易量。
   - 订单簿（市场深度）用于流动性分析。

2. **历史数据：**
   - 用于策略测试（回测）。
   - 包括过去时期的价格、交易量和市场事件。

3. **基本面数据：**
   - 公司财务报告、新闻、宏观经济统计数据。
   - 对长期策略很重要。

4. **事件数据：**
   - 公司事件信息，如股息、并购等。

## 从哪里获取数据？

- **[AlphaVantage](https://alphavantage.co/)：** 各交易所的历史数据。
- **[Yahoo Finance](https://finance.yahoo.com/)：** 免费的股票和指数分析数据。
- **[Quandl](https://www.quandl.com/)：** 用于分析的基本面和市场数据。
- **[Interactive Brokers](https://www.interactivebrokers.com/)：** 通过API访问实时市场数据。

## 建议：

- 使用前检查数据质量。
- 对于高频交易，选择延迟最小的数据源。
- 归档数据以便分析和比较结果。
