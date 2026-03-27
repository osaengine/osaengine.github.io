---
layout: post
title: "AI 能读懂股票图表吗？一个 DistilBERT 实验"
description: "一位开发者教语言模型通过图表的文字描述来预测价格走势。在莫斯科交易所 200 多只股票上的测试显示 AUC 为 0.53。"
date: 2025-10-14
image: /assets/images/blog/llm_stock_charts.png
tags: [machine learning, Moscow Exchange, experiment]
lang: zh
---

Mikhail Shardin 进行了一项实验：如果用文字描述图表，语言模型能预测价格吗？

## 想法

模型接收的不是原始报价，而是自然语言描述：价格强势上涨、成交量增加、接近阻力位。

DistilBERT 模型被训练来预测次日价格上涨。

## 结果

在莫斯科交易所 200 多只股票上进行了测试：

- 平均 AUC：0.53（略好于随机）
- 最佳表现：AFLT（0.72）、RTSB（0.70）、PIKK（0.70）
- 最差表现：PLZL（0.33）、VJGZP（0.33）

就交易目的而言，结果较弱，但模型在没有直接访问数字的情况下捕捉到了一些规律 -- 这本身就很有趣。

## 技术栈

Python + PyTorch + Hugging Face + Docker。前向验证，通过 pandas 进行向量化处理。整个过程可复现。

**GitHub 代码：** [github.com/empenoso/llm-stock-market-predictor](https://github.com/empenoso/llm-stock-market-predictor)

---

**来源：** [Habr](https://habr.com/ru/articles/955612/) | **作者：** Mikhail Shardin
