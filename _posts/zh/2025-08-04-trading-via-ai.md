---
layout: post
title: "Alpaca 发布支持 AI 的 MCP 服务器：详细视频指南和开源代码"
description: "Alpaca Trading API 的官方 Model Context Protocol 服务器允许您通过 Claude AI、VS Code 和其他 IDE 交易股票和期权。该公司准备了五步视频教程，并在 GitHub 上发布了源代码。"
date: 2025-08-04
image: /assets/images/blog/alpaca_mcp_server_release.png
tags: [Alpaca, MCP, algorithmic trading, AI, GitHub]
lang: zh
---

七月底，Alpaca 推出了其 Trading API 的**官方 MCP 服务器**，并立即发布了一段**五分钟视频教程**，介绍如何在本地部署并连接到 Claude AI。根据开发者的设想，新服务器应该能简化使用自然语言创建交易策略的过程，缩短从想法到执行交易之间的时间。

## 视频展示内容

视频 ["How to Set Up Alpaca MCP Server to Trade with Claude AI"](https://www.youtube.com/watch?v=W9KkdTZEvGM) 的作者演示了五步设置流程：

1. 克隆仓库并创建虚拟环境
2. 使用 Alpaca API 密钥配置环境变量
3. 启动服务器（stdio 或 HTTP 传输）
4. 通过 `mcp.json` 连接到 Claude Desktop
5. 首次使用自然语言进行交易请求

这样，即使没有深入的 Python 知识，您也可以快速测试通过 AI 助手进行交易。

## MCP 服务器的主要功能

* **市场数据**：实时报价、历史 K 线、期权和 Greeks
* **账户管理**：余额、购买力、账户状态
* **持仓和订单**：开仓、平仓、交易历史
* **期权**：合约搜索、多腿策略
* **公司行动**：财报日历、拆股、股息
* **自选列表**和资产搜索

完整功能列表可在仓库 README 中查看。

## GitHub 仓库

该项目以 **MIT** 许可证开源，已经获得了 **170+ 星标和约 50 个 Fork**，社区持续提交 Pull Request。最新更新日期为 2025 年 7 月 31 日。

```bash
git clone https://github.com/alpacahq/alpaca-mcp-server.git
cd alpaca-mcp-server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python alpaca_mcp_server.py
```

## 为什么这很重要

MCP 服务器将 Alpaca API 变成了 AI 模型的"沙盒"：现在您可以**通过自然语言命令来对冲持仓、构建自选列表和下限价单**。对交易者而言，这意味着：

- 无需额外代码即可更快地进行**策略原型设计**
- 与 Claude AI、VS Code、Cursor 和其他开发工具集成
- 能够通过环境变量连接多个账户（模拟和实盘）

Alpaca 继续推进算法交易的民主化，社区已经在添加对新语言和传输方式的支持。如果您一直想尝试 AI 交易，现在是开始的好时机。

> **链接：**
> -- YouTube 视频指南：<https://www.youtube.com/watch?v=W9KkdTZEvGM>
> -- GitHub 仓库：<https://github.com/alpacahq/alpaca-mcp-server>
