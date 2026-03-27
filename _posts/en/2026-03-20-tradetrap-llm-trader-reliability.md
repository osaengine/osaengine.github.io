---
layout: post
title: "TradeTrap: How Reliable Are LLM Traders Really?"
description: "The TradeTrap study revealed serious issues with reliability and faithfulness of LLM traders. We explore why AI bots make decisions differently from how they explain them."
date: 2026-03-20
image: /assets/images/blog/tradetrap-llm-reliability.png
tags: [AI, LLM, trading, reliability, research]
lang: en
---

## The Faithfulness Problem

When an LLM trader explains its decision -- "I bought AAPL because RSI shows oversold conditions and earnings beat expectations" -- did it actually rely on those factors? Or is the explanation a **post-hoc rationalization**, while the real "decision" was driven by entirely different reasons?

The **TradeTrap** study by a team of researchers examined precisely this question.

## Research Methodology

The researchers created a controlled environment in which:

1. LLM agents received **market data and news** for making trading decisions
2. Some data contained **intentional traps** -- false signals that looked convincing
3. Agents had to make decisions and **explain** them
4. Researchers compared **stated** reasons with **actual** triggers

### Types of Traps

- **Anchoring trap** -- a random "target price" was inserted into the context, not based on any analysis
- **Recency trap** -- recent data was worse than the average, but the trend remained positive
- **Authority trap** -- fake quotes from "well-known analysts" with incorrect forecasts
- **Confirmation trap** -- data confirming the model's existing bias

## Results

### Trap Hit Rate

*Note: the tables use models that were available at the time of the study (late 2025).*

| Model | Anchoring | Recency | Authority | Confirmation |
|-------|-----------|---------|-----------|-------------|
| GPT-4o | 34% | 41% | 28% | 52% |
| Claude 3.5 Sonnet | 22% | 35% | 19% | 44% |
| DeepSeek V3 | 39% | 48% | 33% | 57% |
| Gemini 2.0 Flash | 31% | 38% | 25% | 49% |

### Faithfulness Score

How well the model's explanations match the actual reasons behind its decisions:

| Model | Faithfulness |
|-------|-------------|
| Claude 3.5 Sonnet | 67% |
| GPT-4o | 61% |
| Gemini 2.0 Flash | 58% |
| DeepSeek V3 | 54% |

This means that in **33-46% of cases**, LLM trader explanations **do not match** the actual reasons behind their decisions.

## Key Takeaways

### 1. Confirmation Bias Is the Biggest Problem

All models showed the greatest vulnerability to **confirming their own preconceptions**. If a model "decided" to buy an asset, it finds data supporting that decision, even when objective data says otherwise.

### 2. Chain-of-Thought Does Not Help

Even reasoning models with detailed Chain-of-Thought are susceptible to traps. Moreover, a long reasoning chain sometimes **masks** unreliable decisions, creating the illusion of deep analysis.

### 3. The Cost of Errors Grows with Autonomy

The more autonomy an LLM trader has, the more expensive each faithfulness error becomes. If the agent automatically places orders based on flawed reasoning, the consequences can be severe.

## Practical Recommendations

The study authors suggest:

- **Do not trust the explanations** of LLM traders -- verify decisions independently
- **Use ensemble** approaches -- multiple models vote on a decision
- **Limit autonomy** -- human-in-the-loop for large trades
- **Test on adversarial data** -- check how the agent reacts to traps
- **Log all intermediate steps** -- for post-mortem analysis of errors

## What This Means for the Industry

TradeTrap is an important signal for everyone building AI trading systems. **A high benchmark on SWE-Bench or MMLU does not mean reliability in trading.** Specialized tests that account for cognitive traps and faithfulness are needed.

The full text of the study is available on [arXiv](https://arxiv.org/abs/2512.02261).
