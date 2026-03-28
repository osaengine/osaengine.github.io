---
lang: zh
layout: faq_article
title: "在哪里获取用于策略测试的历史数据？"
section: tools
order: 2
---

历史数据是测试交易策略和评估其有效性的基础。获取高质量的数据可以进行回测并改进算法。

## 数据类型：

1. **价格数据：**
   - 开盘价、收盘价、最高价和最低价（OHLC）。
   - 逐笔数据用于详细分析。

2. **交易量：**
   - 市场交易量信息。

3. **订单簿：**
   - 市场深度用于流动性分析。

## 数据来源：

- **[Quandl](https://www.quandl.com/)：** 股票、期货和指数的历史数据。
- **[Yahoo Finance](https://finance.yahoo.com/)：** 免费的回测数据。
- **[Interactive Brokers](https://www.interactivebrokers.com/)：** 通过API访问历史和实时数据。
- **MetaTrader：** 支持下载外汇交易的历史数据。

## 建议：

- 使用前检查数据的完整性和质量。
- 使用缺失值最少的数据以获得准确结果。
- 归档下载的数据以供后续分析。
