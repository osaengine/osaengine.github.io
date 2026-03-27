---
layout: post
title: "SaaS vs Open-Source: When It Makes Sense to Build Your Own Algotrading Engine"
description: "A detailed breakdown of total cost of ownership, hidden expenses, and the break-even point. When TSLab is cheaper than your own stack, and when it's the other way around."
date: 2026-02-17
image: /assets/images/blog/saas_vs_opensource.png
tags: [SaaS, open-source, TCO, infrastructure, platform choice]
lang: en
---

"Build your own stack or pay for a ready-made SaaS?"

A year ago, I switched from TSLab (60k/year) to my own stack (Python + Backtrader + Docker).

I thought I'd save money. Turns out — it's not that simple.

Over the past year, I calculated the **real** total cost of ownership (TCO) for both approaches. I factored in not just money, but time, risks, and hidden expenses.

Here's what I learned.

## The Illusion of Free Open-Source

**Myth:** Open-source is free. Python, Backtrader, Docker — all free.

**Reality:** The software is free. Time, infrastructure, support — are not.

### My Story

**2024:** TSLab: 60k/year + MOEX AlgoPack: 55k/year + VPS: 12k/year = **127k/year total.**

I thought: "Why pay? I'll build on open-source. Save 115k/year."

**2025 (reality):** Built a stack: Python + Backtrader + TimescaleDB + Docker. Infrastructure: 54k/year. Time: 40 hours setup + 120 hours maintenance.

**First year total: 534k** (counting my time at developer rate).

**TSLab + AlgoPack: 127k/year.** I lost **407,000** in the first year.

## TCO: Total Cost of Ownership

[TCO](https://www.xelent.ru/blog/chto-takoe-sovokupnaya-stoimost-vladeniya-tco/) isn't just the license fee. It's CapEx, OpEx, and hidden costs.

**TCO for SaaS (TSLab):** First year 147k, subsequent years 127k.

**TCO for open-source (Python stack):** First year 644k, second year 474k.

Open-source is **4.4x more expensive** in the first year.

## When SaaS Is Cheaper

1. **You're not a programmer** — learning Python, Docker, SQL costs 200+ hours
2. **Simple strategies** — SaaS handles them out of the box
3. **Testing an idea** — 5 hours in TSLab vs 40 hours in open-source
4. **Capital <5M rubles** — platform cost exceeds capital

## When Open-Source Is Cheaper

1. **You're a programmer** — setup and maintenance times halve
2. **Complex strategies (ML, arbitrage)** — SaaS can't handle them
3. **Capital >10M rubles** — flexibility yields additional returns
4. **Scale (5+ users)** — SaaS licenses multiply, open-source doesn't
5. **HFT** — open-source with direct WebSocket/FIX API: 1-5ms vs TSLab's 10-30ms

## Platform Comparison

### SaaS for the Russian Market

| Platform | Cost (rub/year) | Pros | Cons |
|----------|----------------|------|------|
| TSLab | 60,000 | Visual builder, backtester, support | Vendor lock-in, no ML |
| MetaTrader 5 | 0 | Free, simple | Limited functionality |
| TradingView | 15-60k | Charts, Pine Script | No full backtesting |

### Open-Source Frameworks

| Framework | Language | Pros | Cons |
|-----------|---------|------|------|
| Backtrader | Python | Simple, flexible | Slow, unsupported |
| LEAN | C#/Python | Professional, active development | Complex setup |
| StockSharp | C# | 90+ exchanges, GUI Designer | Steep learning curve |

## Checklist: SaaS or Open-Source?

1. **Are you a programmer?** Yes -> Open-source. No -> SaaS.
2. **Simple strategy?** Yes -> SaaS. No (ML, arbitrage) -> Open-source.
3. **Capital?** <5M -> SaaS. >10M -> Open-source.
4. **Time is money?** Yes -> SaaS. No (hobby) -> Open-source.
5. **Vendor lock-in critical?** Yes -> Open-source. No -> SaaS.
6. **HFT?** Yes -> Open-source. No -> SaaS.

## My Opinion

**SaaS (TSLab)** — if you're not a programmer, strategy is simple, capital <5M, time costs more than money.

**Open-source** — if you're a programmer, strategy is complex, capital >10M, you need full independence.

**Hybrid approach** — best of both worlds.

**My personal recommendation:** If you're a beginner, **start with SaaS**. Verify that algotrading is for you. After 6-12 months, when you hit platform boundaries, switch to open-source.

**Calculate TCO honestly.** Factor in time. If time is money, SaaS is almost always more cost-effective.

---

**Useful links:**

- [Xelent: What is TCO](https://www.xelent.ru/blog/chto-takoe-sovokupnaya-stoimost-vladeniya-tco/)
- [QuantConnect Pricing](https://www.quantconnect.com/pricing/)
- [Backtrader](https://www.backtrader.com/)
- [LEAN (QuantConnect)](https://github.com/QuantConnect/Lean)
- [StockSharp](https://github.com/StockSharp/StockSharp)
