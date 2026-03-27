---
layout: post
title: "GPT-5.3 Codex：OpenAI 更新旗舰模型"
description: "OpenAI 发布了 GPT-5.3 Codex——旗舰模型的更新版本，编程能力得到提升。我们将其与 Claude 4.6 Opus 进行对比。"
date: 2026-03-13
image: /assets/images/blog/gpt-5-3-codex.png
tags: [AI, GPT-5, OpenAI, coding]
lang: zh
---

## GPT-5.3 Codex：有哪些新变化

OpenAI 继续发展 GPT-5 系列。新版本 **GPT-5.3 Codex** 被定位为该公司最强大的编程模型。据 OpenAI 称，该模型在以下方面取得了显著提升：

- **代码生成**——覆盖所有主流编程语言
- **调试与重构**——针对现有代码库
- **代码解释**——模型更善于"理解"项目架构
- **测试生成**——单元测试生成的准确度明显提高

## 基准测试

独立测试结果：

| 测试 | GPT-5.3 Codex | Claude Opus 4.6 | Claude Sonnet 4.6 |
|------|---------------|-----------------|-----------------|
| SWE-Bench Verified | 78.4% | 76.1% | 82.1% |
| HumanEval+ | 95.8% | 94.3% | 96.2% |
| MBPP+ | 88.2% | 87.1% | 89.7% |
| Codeforces Rating | 1847 | 1792 | 1801 |

GPT-5.3 Codex 在编程任务上明显超越 Claude Opus 4.6，但在 SWE-Bench 上仍然不及 Claude Sonnet 4.6。

## 关键改进

### 扩展的代码上下文

GPT-5.3 Codex 拥有 **128K token 的上下文窗口**，并针对代码文件进行了优化。OpenAI 声称该模型能够在"记忆"中保持数百个文件组成的项目结构。

### 改进的函数调用

对于使用 API 的开发者来说，**函数调用**变得更加可靠。模型能更准确地生成 JSON 调用模式，并且更少"编造"不存在的参数。

### Codex Agent 模式

OpenAI 推出了 **Codex Agent** 模式，在该模式下模型可以：

- 在终端中顺序执行命令
- 读取和修改文件
- 运行测试并根据结果进行迭代

这是对 Anthropic 的 **Claude Code** 及类似智能体产品的直接回应。

## 定价

GPT-5.3 Codex 通过 API 提供，定价如下：

- **输入**：$8 / 百万 token
- **输出**：$24 / 百万 token
- **缓存输入**：$2 / 百万 token

这使得该模型处于中等价格区间——比 DeepSeek 贵，但比 Claude Opus 便宜。

## 交易机器人应该选择哪个？

对于算法交易系统的开发者，GPT-5.3 和 Claude 之间的选择取决于具体任务：

- **从零编写策略**——Claude Sonnet 4.6 表现最佳
- **与现有 API 集成**——GPT-5.3 Codex 凭借精准的函数调用胜出
- **市场数据分析**——两者表现都不错，但 GPT-5.3 在流式生成方面更快

模型之间的竞争只会愈演愈烈，这对最终用户来说是个好消息。
