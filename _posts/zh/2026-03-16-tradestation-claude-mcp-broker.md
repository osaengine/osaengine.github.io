---
layout: post
title: "TradeStation + Claude：首家接入 AI 的 MCP 券商"
description: "TradeStation 成为首家发布官方 MCP 连接 Claude 的券商。现在可以通过 AI 助手管理交易。"
date: 2026-03-16
image: /assets/images/blog/tradestation-claude-mcp.png
tags: [AI, MCP, TradeStation, trading]
lang: zh
---

## TradeStation 创造历史

2026 年 1 月，**TradeStation** 成为全球首家发布官方 [MCP 服务器](https://www.tradestation.com/) 以连接 AI 模型的券商。这意味着 TradeStation 的客户现在可以通过 **Claude** 及其他兼容 MCP 的 AI 客户端与其券商账户进行交互。

## 通过 MCP 可以做什么

TradeStation MCP 服务器提供以下功能：

### 数据查看（Resources）
- 当前投资组合和持仓
- 账户余额和保证金要求
- 订单和交易历史
- 实时报价
- 图表和技术指标

### 交易操作（Tools）
- 提交市价单和限价单
- 设置止损和止盈
- 修改现有订单
- 取消订单
- 平仓

### 分析（Prompts）
- 投资组合指标分析（夏普比率、回撤）
- 与基准的业绩比较
- 持仓风险概览

## 实际操作是什么样的

Claude Desktop 中的会话示例：

```
用户：显示我的未平仓头寸和盈亏

Claude：正在通过 TradeStation 查询数据...

您的未平仓头寸：
┌────────┬─────────┬──────────┬─────────┐
│ 代码    │ 数量     │ 均价      │ 盈亏     │
├────────┼─────────┼──────────┼─────────┤
│ NVDA   │ 30      │ $142.50  │ +$1,245 │
│ MSFT   │ 25      │ $428.00  │ +$312   │
│ TSLA   │ 15      │ $385.20  │ -$89    │
└────────┴─────────┴──────────┴─────────┘

总未实现盈亏：+$1,468
建议：TSLA 持仓显示亏损，建议设置止损。
```

## 安全性和限制

TradeStation 实施了多层保护：

- 带双因素认证的 **OAuth 2.0**
- **订单规模限制** — 用户可自定义
- **标的白名单** — 只能交易许可的标的
- 默认**只读模式** — 交易需要明确激活
- **每笔订单确认** — AI 不能未经批准进行交易

## 市场反应

公告发布后：

- TradeStation 1 月新注册量增长 **23%**
- 竞争对手（Interactive Brokers、Charles Schwab）宣布开发自己的 MCP 方案
- TradeStation 股价（母公司 Monex Group）上涨 **8%**

## 适合谁

TradeStation 的 MCP 集成非常适合：

- **活跃交易者** — 希望通过语音/文字管理投资组合
- **开发者** — 构建基于 LLM 的交易机器人
- **投资组合经理** — 需要快速的分析界面

这是迈向 AI 原生交易的第一步，但绝非最后一步。
