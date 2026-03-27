---
layout: post
title: "从流程图到代码：我如何在3个月内从可视化构建器转向编程"
description: "从可视化构建器过渡到完整编程的真实故事。计划、错误、工具，以及为什么这比你想象的更容易。"
date: 2025-12-30
image: /assets/images/blog/visual_to_code.png
tags: [learning, programming, Python, transition, builders]
lang: zh
---

一年前我在TSLab中组装策略。流程图，拖放操作，没有代码。一切正常。直到我碰到了天花板。

我需要一个自定义指标。需要实时交易统计。需要与外部API集成。

构建器无法处理这些。

我决定学习编程。三个月前我写下了第一行Python代码。今天我的机器人在交易，所有代码都是我的。

这不是"我是编程天才"的故事。这是"任何人都可以做到，只要知道从哪里开始"的故事。

## 为什么我决定学习编程

**触发因素 #1：碰到构建器的限制**

我想添加基于ATR（平均真实波幅）的自适应止损。TSLab有ATR模块。有止损模块。

但没有"每根K线根据ATR动态调整止损"的模块。

我可以在TSLab内编写C#脚本。但如果要学C#——为什么不直接用Python写，不依赖任何平台呢？

**触发因素 #2：供应商锁定**

我在TSLab中构建的一切只存在于TSLab中。如果平台关闭、更新、崩溃——我的策略就死了。

Python代码是一个文件。它永远属于我。我可以在任何地方运行。

**触发因素 #3：好奇心**

我理解策略逻辑。我看到模块之间的连接。但*内部*发生了什么？

构建器隐藏了复杂性。但当出现问题时——我不理解*为什么*。

代码给予控制。完全的控制。

## 我的起点：选择语言

有三个选择：

### Python

**优点：**
- [语法直观](https://spreadbet.ai/python-or-c-trading-bots/)，对初学者友好
- 大量算法交易库（Backtrader, LEAN, ccxt）
- [机器学习和数据分析的最佳选择](https://blog.traderize.com/posts/top-languages-trading-bots/)

**缺点：**
- 比C#/C++慢
- 不适合高频交易

### C#

**优点：**
- [比Python快](https://www.quantconnect.com/forum/discussion/3163/what-039-s-your-preference-c-or-python-and-why/)
- 用于StockSharp, LEAN, NinjaTrader
- 与.NET生态系统良好集成

**缺点：**
- 对初学者语法更复杂
- 算法交易学习材料较少

### MQL5 (MetaTrader)

**优点：**
- [语法类似C#](https://www.mql5.com/en/book)
- 直接在MetaTrader中运行
- 庞大的外汇交易者社区

**缺点：**
- 绑定MetaTrader（又是供应商锁定）
- [复杂逻辑能力有限](https://forums.babypips.com/t/should-i-upgrade-from-mql4-to-python/527073)

**我的选择：Python**

我选择了Python。因为：
1. 更容易起步
2. 初学者材料更多
3. 可以快速验证想法
4. 不需要高频交易（我在小时图上交易）

[如果需要速度——C#更好](https://aesircrypto.com/blog/best-programming-language-for-building-a-crypto-trading-bot/)。但对于日线/小时线的散户交易者，Python足够了。

## 路线图：3个月从零到工作机器人

以下是我做的。按周计划。

### **第1-4周：Python基础**

**学了什么：**
- 变量、数据类型（int, float, string, list, dict）
- 条件语句（if, else, elif）
- 循环（for, while）
- 函数
- 文件处理

**在哪里学的：**
- [Habr上的"6个月学习Python"](https://habr.com/ru/articles/709102/)——学习计划
- [Skillbox免费课程](https://skillbox.ru/media/code/kak-izuchit-python-samostoyatelno-i-besplatno/)
- Codecademy（前几节免费）

**花费时间：**
每天1-2小时，每周5天。[规律性比时长更重要](https://pythonru.com/baza-znanij/python-obuchenie-s-nulya)。

**第一个成果：**
月末我写了一个脚本：
1. 读取行情CSV文件
2. 计算移动平均线
3. 当SMA(20)穿过SMA(50)时打印信号

最简单的逻辑。但这是**我的**代码。

### **第5-8周：数据分析库**

**学了什么：**
- **Pandas**：表格处理（DataFrame）
- **NumPy**：数学运算
- **Matplotlib**：绑图表

**为什么需要：**
几乎所有算法交易都是处理行情表格（日期、开盘、最高、最低、收盘、成交量）。

Pandas让这变得简单。

**成果：**
写了计算任何指标的函数：

```python
import pandas as pd

def sma(data, period):
    return data['Close'].rolling(window=period).mean()

def rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

现在我可以实现**任何**指标逻辑。没有构建器的限制。

### **第9-12周：Backtrader——第一个交易系统**

**做了什么：**
学习了[Backtrader](https://www.backtrader.com/)库——用于回测策略的框架。

**我的第一个策略：**

```python
import backtrader as bt

class SmaCross(bt.Strategy):
    params = (('fast', 20), ('slow', 50),)

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(period=self.params.fast)
        self.sma_slow = bt.indicators.SMA(period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if self.crossover > 0:
            if not self.position:
                self.buy()
        elif self.crossover < 0:
            if self.position:
                self.sell()
```

与TSLab中相同的逻辑。但**我控制每一行**。

## 我犯的错误

### 错误 #1：试图一次学习所有东西

前两周我下载了10个课程、5本书，订阅了20个YouTube频道。

结果：信息过载。什么都记不住。

**有用的方法：** 一次一个来源。完成一个课程，然后下一个。

### 错误 #2：只读不写代码

我看视频、读教程。心想"懂了，很简单"。

当我坐下来写的时候——想不起来语法。

**有用的方法：** 规则：每一小时理论——一小时实践。看完课程→手写代码。

### 错误 #3：不做项目

我学语法。解练习题。但不应用。

**有用的方法：** [设定目标：3个月结束时——Backtrader上有一个工作策略](https://algotrading101.com/learn/quantitative-trader-guide/)。这给了焦点。

### 错误 #4：害怕提问

卡在问题上——搜索几个小时，不好意思问。

**有用的方法：** Stack Overflow、Reddit（r/algotrading）、算法交易Telegram群。只要问题表述得当，人们会帮忙。

## 当我意识到自己准备好了

顿悟时刻在第10周到来。

我打开TSLab中的旧策略。流程图看起来像意大利面。我试图回忆它做什么。

然后打开同样逻辑的Python代码。**一读就懂了**。

代码比流程图**更易读**。

在那一刻我意识到：我会编程了。

## 给想重复我这条路的人的计划

如果你现在在用构建器，觉得"学编程又长又难"，这是一个现实的计划。

### **步骤1：Python基础（4-6周）**

**任务：**
- 完成[Python基础课程](https://skillbox.ru/media/code/kak-izuchit-python-samostoyatelno-i-besplatno/)
- 在[Codewars](https://www.codewars.com/)或[LeetCode Easy](https://leetcode.com/)上解50-100个简单题

**准备就绪标准：** 你能写一个函数，接受价格列表，返回移动平均线。

### **步骤2：Pandas + NumPy（2-4周）**

**准备就绪标准：** 你能加载行情CSV，添加指标列，绑图表。

### **步骤3：Backtrader上的第一个策略（4-6周）**

**准备就绪标准：** 策略运行，结果接近构建器回测（考虑佣金和滑点）。

### **步骤4：真实市场集成（4-6周）**

**准备就绪标准：** 策略在模拟账户上至少交易一个月且无严重错误。

**总计：14-22周（3-5个月）**

[以每天1-2小时、每周5天的节奏](https://habr.com/ru/articles/709102/)。

这不是"成为高级开发者"。而是"编写一个工作的交易机器人"。

## 转到代码后发生了什么变化

### **优点：**

**1. 完全控制** — 任何逻辑、任何指标、任何集成。没有限制。

**2. 平台独立** — 我的代码永远属于我。不绑定TSLab、Designer、NinjaTrader。

**3. 免费** — Python、Backtrader、VS Code——全部免费。

**4. 理解** — 我知道每一步发生了什么。如果有错误——我能看到确切位置。

### **缺点：**

**1. 没有可视化** — 在TSLab中流程图直观。在代码中——是文本。

**2. 初始时间更多** — TSLab中的简单策略——15分钟。Python第一次——2-3小时。

**3. 调试更难** — 在构建器中错误会高亮。在代码中——需要阅读traceback，设置断点。

## 总结：值得吗？

一年前我想："编程是给IT人的。我只是个交易者。"

今天我理解了：编程是一种工具。像Excel。像TradingView。

我没有成为开发者。我写了500行代码，做我需要的事情。

这就**足够了**。

编程用于算法交易不是"成为程序员"。而是"不受限制地自动化你的想法"。

而且这比你想象的更容易。

---

**有用链接：**

- [Should I Use C# Or Python To Build Trading Bots?](https://spreadbet.ai/python-or-c-trading-bots/)
- [Top Languages for Building Custom Trading Bots](https://blog.traderize.com/posts/top-languages-trading-bots/)
- [AlgoTrading101: Quantitative Trader's Roadmap](https://algotrading101.com/learn/quantitative-trader-guide/)
- [Start Algorithmic Trading: Beginner's Roadmap](https://startalgorithmictrading.com/beginners-algo-trading-roadmap)
