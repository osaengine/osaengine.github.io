---
layout: post
title: "用 ChatGPT 和 Claude 寻找交易思路：从数据到回测"
description: "探讨如何利用 AI 进行数据分析、寻找市场低效性，并以加密货币分钟级数据为例创建交易策略。"
date: 2024-11-02
image: /assets/images/blog/ai-trading-strategy-preview.png
tags: [ChatGPT, Claude]
lang: zh
---

在这篇文章中，我决定比较两个热门服务——[ChatGPT](https://chatgpt.com/) 和 [Claude.ai](https://claude.ai/)——看看它们在 2024 年 11 月时如何处理寻找交易低效性的任务。我评估了它们的功能和易用性，以确定哪个更适合数据分析和开发盈利交易策略。

为了简化数据收集，我使用了 **[Hydra](https://stocksharp.ru/store/%D1%81%D0%BA%D0%B0%D1%87%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BC%D0%B0%D1%80%D0%BA%D0%B5%D1%82-%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85/)**——可以说是最好的免费市场数据下载工具。

我下载了 2024 年的 BTCUSDT 分钟级数据，大约 25 MB，并导出为 CSV 文件。

![](/assets/images/blog/hydra_2.png)

![](/assets/images/blog/hydra_3.png)

Hydra 有自己的内置分析功能，但接下来你会看到，与 AI 的能力相比它有多落后——你甚至不需要自己写代码：

![](/assets/images/blog/hydra_4.png)

然而，我工作的主要部分不是数据收集，而是分析和寻找策略思路。我没有手动寻找方法，而是决定信任 AI，看看它会建议什么策略，能在数据中发现什么模式和低效性，以及如何优化测试参数。借助 **ChatGPT**，我不仅进行了详细分析，还对策略进行了回测。

---

### 数据准备

拿到分钟级数据后，我将其加载到 Python 中（代码由 AI 编写——我只是用文字描述我需要什么），并开始预处理。这包括为每列命名，以及将日期和时间合并为一列，以简化时间序列分析。

---

### 利用 AI 寻找低效性

数据预处理后，我询问 AI 关于可能的低效性和可用于策略开发的模式。ChatGPT 提出了几种方法：

1. **波动率集群** — 高波动率时段可能适合动量策略。
2. **均值回归倾向** — 当价格偏离均值水平时，可以使用均值回归策略。
3. **动量模式** — 在某些时段观察到持续的价格运动，可以作为趋势跟踪策略的信号。

![](/assets/images/blog/volatility-clusters.png)

---

### 策略开发

基于 AI 的建议，我选择了两种策略进行测试：

1. **均值回归（Mean Reversion）**：当价格大幅高于均值时开空仓，大幅低于均值时开多仓。当价格回归均值时平仓。

2. **动量策略（Momentum）**：在波动率升高期间沿趋势方向开仓。如果收益为正且高于阈值，则做多；如果为负且低于阈值，则做空。

为每种策略设定了基本的入场和出场规则，以及用于风险管理的止损。

![](/assets/images/blog/hourly-returns.png)

---

### 策略回测

借助 ChatGPT，我还对两种策略进行了回测，以查看它们在历史数据上的表现。测试结果显示了均值回归策略的权益曲线（见下图）。

图表显示了遵循该策略时投资组合资本可能的变化情况。可以看到策略在某些时期表现出稳定增长，但也存在回撤时期。这证实了参数调优和风险管理的重要性。

![](/assets/images/blog/mean-reversion-equity-curve.png)

---

### Claude.ai

在工作过程中，我还尝试使用了 Anthropic 的 **Claude Sonnet**，它最近发布了数据分析功能（详情见[这里](https://www.anthropic.com/news/analysis-tool)）。这个想法看起来很有前景：上传一个 25 MB 的文件让 Claude 帮助分析。

![](/assets/images/blog/claude_analytics.png)

然而，我遇到了不少困难。遗憾的是，该功能还不够成熟——我的文件甚至无法上传。最后我把文件分成了小块，但由于之前的错误，很快就达到了请求限制。我唯一得到的就是在尝试生成图表时的一个错误。

![](/assets/images/blog/claude_error_1.png)

虽然我很喜欢使用 Claude，但我希望项目的工程师们能完善这个功能并大幅扩展数据上传的限制。这将使大文件的分析更加高效，并为处理大量数据开辟新的可能性。

![](/assets/images/blog/claude_error_2.png)

---

### 总结

使用 ChatGPT 让我不仅能够分析数据，还能向 AI 询问适合的策略创建方法。这种方法不仅产生了新的想法，还帮助快速测试假设，获得了传统方法可能忽略的建议。AI 帮助发现策略思路和参数的方法，为灵活且自适应的交易策略开发开辟了新的可能性。
