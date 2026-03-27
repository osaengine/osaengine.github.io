---
layout: post
title: "典型开源机器人的内部架构：以简单策略为例"
description: "深入分析交易机器人的真实架构：从层次和组件到设计模式。Freqtrade、NautilusTrader、微服务、Event Sourcing——实践中什么有效。"
date: 2026-03-03
image: /assets/images/blog/opensource_robot_architecture.png
tags: [architecture, open-source, trading robots, Event Sourcing, microservices]
lang: zh
---

三个月前，我[对比了MOEX的开源框架](/zh/blog/2026/02/24/comparing-lean-stocksharp-backtrader.html)。搞清楚了选什么。但问题仍然存在：**这一切内部是如何运作的？**

今天不是关于选择平台，而是关于**引擎盖下发生了什么**。

我分析了四个流行开源机器人的架构：[Freqtrade](https://github.com/freqtrade/freqtrade)、[NautilusTrader](https://github.com/nautechsystems/nautilus_trader)、[Hummingbot](https://github.com/hummingbot/hummingbot)和微服务系统[MBATS](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System)。

结论：尽管语言和目标不同，**模式是重复的**。

## 分层架构：基础

大多数交易机器人遵循5层架构：

```
第5层：通信层      <- Telegram、Web UI、API
第4层：策略层      <- 交易逻辑
第3层：执行和风控  <- 订单、风险管理
第2层：数据处理    <- 指标、标准化
第1层：数据采集    <- 交易所连接
```

## 设计模式

### 1. Event Sourcing
将每个状态变化保存为事件。对审计、调试和合规至关重要。

### 2. CQRS
分离读写。Event Store处理写入；单独的数据库处理查询。

### 3. 微服务架构
MBATS将系统拆分为通过Kafka/RabbitMQ通信的独立服务。

### 4. Actor Model
NautilusTrader和Hummingbot使用它。每个Actor有独立状态，通过消息通信。无共享状态意味着无竞态条件。

## 设计清单

1. **从单体开始** — 不要过早跳到微服务
2. **从第一天起分离层次**
3. **为测试而设计** — 依赖注入
4. **从第一天起添加可观测性** — 日志、指标
5. **规划持久化** — 保存所有订单、持仓、设置
6. **何时转向微服务：** >100个工具、HFT、团队>3人

架构决定了你能走多远。意大利面代码能用一个月。正确的架构——能用好几年。

---

**有用链接：**

- [Freqtrade](https://github.com/freqtrade/freqtrade)
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader)
- [Hummingbot](https://github.com/hummingbot/hummingbot)
- [MBATS](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System)
