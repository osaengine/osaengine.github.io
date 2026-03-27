---
layout: post
title: "Gemini 3.1 Pro：百万 token 上下文与 ARC-AGI-2 上的 77% 得分"
description: "Google 发布了 Gemini 3.1 Pro，拥有 100 万 token 的上下文窗口和 ARC-AGI-2 上 77% 的成绩。Google AI 的市场份额正在增长。"
date: 2026-03-14
image: /assets/images/blog/gemini-3-1-pro.png
tags: [AI, Gemini, Google, benchmarks]
lang: zh
---

## 百万 token：为何重要

Google 继续押注长上下文。**Gemini 3.1 Pro** 保持着 **100 万 token** 的创纪录上下文窗口——大约 70 万个单词，相当于几本完整的书籍。

在实际应用中，这意味着：

- 将中型项目的**整个代码库**一次性加载到单个请求中
- 分析**年度财务报告**而不丢失上下文
- 处理**长对话历史**和文档
- 处理数小时的**会议和通话录音**

## ARC-AGI-2：抽象思维测试

[ARC-AGI-2](https://arcprize.org/) 基准测试检验模型的抽象推理能力——这些任务对孩子来说轻而易举，但对大多数 AI 系统却是难题。

Gemini 3.1 Pro 在 ARC-AGI-2 上取得了 **77%** 的成绩，是商业模型中最佳结果之一：

| 模型 | ARC-AGI-2 |
|------|-----------|
| Claude Sonnet 4.6 | 79.2% |
| **Gemini 3.1 Pro** | **77.0%** |
| GPT-5.3 | 74.5% |
| DeepSeek V3 | 71.3% |

## 市场份额增长

据分析师统计，Google 在 LLM API 市场的份额在过去六个月从 12% 增长到 18%。主要原因：

### 定价策略

Google 提供极具竞争力的价格：

- **输入**：$3.50 / 百万 token
- **输出**：$10.50 / 百万 token
- 超过 128K 的上下文：$7 / $21 每百万

### Google Cloud 生态系统

与 **Vertex AI**、**BigQuery** 及其他 Google Cloud 服务的集成，使 Gemini 对已在使用 Google 云基础设施的企业客户极具吸引力。

### 多模态

Gemini 3.1 Pro 原生支持：

- 文本、图像、音频和视频
- 代码生成和分析
- 表格和结构化数据处理

## 对交易者的意义

100 万 token 的长上下文开辟了有趣的可能性：

1. **加载长期的完整交易历史**用于模式分析
2. **同时分析**多家公司的财务报告
3. **处理一整天的新闻流**而不遗漏重要细节

不过需要注意，长上下文开头和结尾处的信息处理质量可能存在差异——即所谓的"中间丢失"问题。

Google 正稳步巩固其在 AI 市场三强中的地位，Gemini 3.1 Pro 是这场竞赛中的有力证明。
