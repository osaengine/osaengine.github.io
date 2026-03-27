---
layout: post
title: "GPT-5.3 Codex: OpenAI Updates Its Flagship Model"
description: "OpenAI has released GPT-5.3 Codex — an updated version of its flagship model with improved coding capabilities. We compare it with Claude 4.6 Opus."
date: 2026-03-13
image: /assets/images/blog/gpt-5-3-codex.png
tags: [AI, GPT-5, OpenAI, coding]
lang: en
---

## GPT-5.3 Codex: What's New

OpenAI continues to develop the GPT-5 lineup. The new **GPT-5.3 Codex** is positioned as the company's best model for programming tasks. According to OpenAI, the model shows significant improvements in:

- **Code generation** across all popular languages
- **Debugging and refactoring** existing codebases
- **Code explanation** — the model better "sees" project architecture
- **Test generation** — unit test generation has become noticeably more accurate

## Benchmarks

Independent test results:

| Test | GPT-5.3 Codex | Claude Opus 4.6 | Claude Sonnet 4.6 |
|------|---------------|-----------------|-----------------|
| SWE-Bench Verified | 78.4% | 76.1% | 82.1% |
| HumanEval+ | 95.8% | 94.3% | 96.2% |
| MBPP+ | 88.2% | 87.1% | 89.7% |
| Codeforces Rating | 1847 | 1792 | 1801 |

GPT-5.3 Codex confidently outperforms Claude Opus 4.6 in coding tasks but still falls short of Claude Sonnet 4.6 on SWE-Bench.

## Key Improvements

### Expanded Code Context

GPT-5.3 Codex features **128K tokens of context** optimized for code files. OpenAI claims the model can hold the structure of a project with several hundred files in "memory."

### Improved Function Calling

For developers using the API, **function calling** has become more reliable. The model generates JSON call schemas more accurately and less frequently "invents" nonexistent parameters.

### Codex Agent Mode

OpenAI introduced the **Codex Agent** mode, in which the model can:

- Execute commands sequentially in a terminal
- Read and modify files
- Run tests and iterate on results

This is a direct response to **Claude Code** from Anthropic and similar agent products.

## Pricing

GPT-5.3 Codex is available via API at the following prices:

- **Input**: $8 / 1M tokens
- **Output**: $24 / 1M tokens
- **Cached input**: $2 / 1M tokens

This places the model in the mid-price segment — more expensive than DeepSeek but cheaper than Claude Opus.

## What to Choose for Trading Bots?

For developers of algorithmic trading systems, the choice between GPT-5.3 and Claude depends on the task:

- **For writing strategies from scratch** — Claude Sonnet 4.6 shows the best results
- **For integration with existing APIs** — GPT-5.3 Codex wins due to accurate function calling
- **For market data analysis** — both options work well, but GPT-5.3 is faster at streaming generation

Competition between models is only intensifying, and that's great news for end users.
