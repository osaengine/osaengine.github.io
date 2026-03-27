---
layout: post
title: "ChatGPT生成想法，构建器组装机器人：算法交易的未来还是炒作？"
description: "我花了一个月测试AI+可视化构建器组合。通过ChatGPT/Claude生成策略，在TSLab中组装。结果如下。"
date: 2026-01-13
image: /assets/images/blog/ai_visual_builders.png
tags: [AI, ChatGPT, Claude, builders, automation]
lang: zh
---

"描述一个基于EMA交叉加RSI过滤器的策略。"

ChatGPT在10秒内给出逻辑。我打开TSLab，组装模块。15分钟后——一个现成的机器人。

听起来像梦想。但在实践中有效吗？

过去一个月我一直在测试这个组合：用AI生成想法，用可视化构建器组装。以下是现实。

## 实验：ChatGPT的10个策略→TSLab

10个策略中：3个在回测中盈利（年化>20%），5个接近零（+5%到-5%），2个亏损（-10%和-15%）。

## 问题 #1：AI不理解市场背景

ChatGPT生成逻辑正确的策略。但它不了解：标的物特性、当前市场状态、你的交易风格。AI需要非常精确的指导。

## 问题 #2：构建器限制复杂性

[Claude可以生成复杂策略](https://medium.com/@austin-starks/i-let-claude-opus-4-create-a-trading-strategy-it-destroyed-the-market-c200bf1a19a4)。但可视化构建器不支持。AI可以生成比构建器能组装的更复杂的策略。

## 问题 #3：AI会"幻觉"指标

ChatGPT有时建议构建器中不存在的指标。需要知道你的构建器有哪些指标。

## 什么有效：正确的提示词

**差的提示词：** "想出一个交易策略"

**好的提示词：** "为小时级EUR/USD K线建议一个策略（外汇）。只使用这些指标：SMA、EMA、RSI、MACD。平均波动50点/天。目标：每周3-5笔交易。止损30点以内。"

## 未来还是炒作？

**这不是未来。是一个工具。**

AI+构建器不会取代量化程序员。但会加速工作。对初学者有用，降低入门门槛。但不是万能药。想深入理解——学编程。

---

**有用链接：**
- [Medium: Claude Trading Strategy](https://medium.com/@austin-starks/i-let-claude-opus-4-create-a-trading-strategy-it-destroyed-the-market-c200bf1a19a4)
- [PickMyTrade: Claude for Trading Guide](https://blog.pickmytrade.trade/claude-4-1-for-trading-guide/)
