---
layout: post
title: "AI vs Open Source: What Actually Changed and Where the Line Is"
description: "A detailed analysis of how modern code models have shifted the balance between generation and ready-made libraries in algorithmic trading."
date: 2025-09-03
image: /assets/images/blog/ai_vs_oss.png
tags: [AI, Open Source, algorithmic trading, development]
lang: en
---

I published a new article on Habr: **["AI vs Open Source: What Actually Changed and Where the Line Is"](https://habr.com/ru/articles/943670/)**

With the emergence of working code models, a more down-to-earth development path has appeared: formulate a requirement, write tests, and get a small, understandable module with no unnecessary dependencies. This isn't a war against OSS -- it's a shift in the equilibrium point.

## Main Takeaways:

### What Changed
- **Before**: "library first." Search for a library, accept transitive dependencies, read documentation.
- **Now**: "description -> tests -> implementation." Small, testable modules instead of monolithic "combines."

### Where AI Already Replaces Libraries
1. **Mini-implementations**: indicators (EMA/SMA/RSI), statistics, risk rules
2. **Narrow integrations**: REST/WebSocket clients with just 2-3 needed methods
3. **Skeleton generation**: backtest scaffolds, data schemas
4. **Adapters**: mapping between exchanges, code migrations

### Where AI Should NOT Replace OSS
- Cryptography and secure protocols
- Binary protocols (FIX/ITCH/OUCH/FAST)
- Database engines, compilers, runtimes
- Numerical solvers and optimizers

### Practical Advice
- Keep modules small
- Describe behavior in simple words
- Do minimal checks for confident merges
- Generate without external dependencies

In algorithmic trading, this is especially relevant: fewer dependencies means lower risks, more compact artifacts, easier audits, and faster iterations.

**Key Takeaway**: Choose your tool based on context. A narrow task that's easy to describe and verify is a candidate for generation. Everything else -- go with proven OSS.
