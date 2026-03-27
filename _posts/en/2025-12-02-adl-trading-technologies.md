---
layout: post
title: "ADL by Trading Technologies: When a Strategy Builder Costs as Much as an Apartment"
description: "I wanted to test ADL (Algo Design Lab) by Trading Technologies — a visual builder for professionals. But I ran into a problem: it's an enterprise solution with 'contact us' pricing. Here's what I found out."
date: 2025-12-02
image: /assets/images/blog/adl_trading_tech.png
tags: [ADL, Trading Technologies, no-code, institutional trading, futures]
lang: en
---

Over the past few weeks, I've been testing visual strategy builders. TSLab for 60,000 rubles a year, StockSharp Designer for free, NinjaTrader for 150,000, fxDreema for 10,000. It made sense to add ADL (Algo Design Lab) by Trading Technologies to the list — a builder for "professionals."

I went to the Trading Technologies website. Found the ADL page. Nice screenshots, marketing about drag-and-drop, visual programming, integration with an institutional-grade platform.

**Question:** Where's the "Download" button, or at least "Try It"?

Answer: there isn't one.

## How to Get Access to ADL (Spoiler: You Can't)

ADL isn't a standalone product. It's a module within TT Platform Pro by Trading Technologies. To get ADL, you first need access to TT Platform.

I started looking into how to do that.

**Option 1:** Register on the TT website and download the platform.

Tried it. The website has a "Contact Us" form. Filled it out. Mentioned I wanted to test ADL for a review. A day later I got a response: "Thank you for your interest. We will contact you to discuss your trading needs."

Two weeks passed. Nobody contacted me.

**Option 2:** Find a broker that provides access to TT Platform.

Googled it. AMP Futures, Optimus Futures, Discount Trading — several American brokers offer TT Platform. But everywhere it's the same: "Pricing on request," "Depends on trading volume," "Contact us for a custom quote."

I reached out to one of the brokers. Asked about access to TT Platform + ADL.

Response: "The minimum subscription to TT Platform Pro starts at $1,500 per month. Plus trade commissions. Plus market data fees. ADL is included free if you have TT Platform Pro."

$1,500 per month. **$18,000 per year.** In rubles, that's roughly **1.8 million**.

For a visual strategy builder.

## What I Could Learn Without Access

Since I couldn't actually test ADL, I had to piece together information: TT documentation, YouTube videos, forums, trader reviews.

**What ADL is:**

It's a visual algorithm builder embedded in TT Platform. Drag-and-drop interface, blocks for conditions and actions, backtesting on historical data. Conceptually similar to TSLab or StockSharp Designer.

**Key difference:** ADL lives inside a professional trading platform. TT Platform is used by hedge funds, prop traders, institutional players. This is not a retail product.

**What it can do (according to documentation):**

- Visual algorithm building through blocks
- Backtesting on historical data
- Real-time market simulation
- Integration with Order Management System (OMS)
- Direct algorithm execution into the order book
- Real-time performance monitoring

**What it CANNOT do:**

- Work outside TT Platform (no code export)
- Work for free or even cheaply
- Be accessible to an ordinary retail trader

## Who Is This Even For?

I thought about this for several days. Here's what I concluded.

ADL is not for retail traders. It's not even for active individual traders. It's for institutional players:

- Prop trading firms
- Hedge funds
- Market makers
- Large asset management companies

People who trade millions of dollars a day. For them, $1,500 per month for a platform is pocket change compared to their volumes.

**The paradox:** ADL is positioned as "a builder anyone can use to create algorithms." But to get access, you need to pay like an institutional player.

## Comparison With What I Actually Tested

Over the past few weeks, I actually worked with four visual builders:

**TSLab** — 60,000 rubles per year. Flowcharts, Russian market, Russian language. It works, but expensive for what it offers.

**StockSharp Designer** — free. Open-source, flowcharts, code export. Russian + international markets. Less mature, but functionally close to TSLab.

**NinjaTrader Strategy Builder** — 150,000 rubles lifetime or 120,000 per year. Tabular interface (not blocks), international markets only. Mature product, but for a narrow niche.

**fxDreema** — 10,000 rubles per year. Browser-based flowcharts, MetaTrader only. A side project by enthusiasts. It works, but there's a risk it could shut down.

**ADL** — 1.8 million rubles per year (minimum). Visual builder inside a professional platform. Couldn't test it, but based on reviews — a solid tool for those who truly need it.

The price difference — 30x compared to TSLab and 180x compared to fxDreema.

## Honest Conclusion: I Couldn't Test It

Usually in my articles, I write about real experience. Installed, tried, ran into problems, drew conclusions.

With ADL, that didn't happen.

**The reason is simple:** it's an enterprise solution. They don't have a demo version. No trial period. Not even public pricing. Everything is "contact us," "custom offer," "depends on volume."

I could have written an article based on TT's marketing materials. But that wouldn't have been my article — it would have been a retelling of someone else's advertising.

Instead, I decided to write honestly: **ADL looks like a powerful tool, but it's not for ordinary traders.**

If you trade millions of dollars through American futures, work at a prop firm or hedge fund, and need a visual builder with institutional-grade infrastructure — ADL might be a good choice.

But if you're an individual trader who wants to build a robot for the Moscow Exchange or just try algorithmic trading — forget about ADL. Too expensive. Too hard to get access. Too tailored for the institutional level.

## What I Did Instead

Unable to get access to ADL, I went back to what I had already tested:

- **StockSharp Designer** — free, works with Russian brokers, open-source
- **fxDreema** — 10,000 per year, if you trade through MetaTrader
- **TSLab** — 60,000 per year, if you want a ready-made solution with support

All three offer visual programming. All three are actually accessible. All three can be tested in 20 minutes.

**My conclusion:** For 99% of traders, ADL is a pretty picture on the Trading Technologies website. Inaccessible, expensive, institutional.

Maybe someday I'll have access to TT Platform. Then I'll write a full ADL review with real tests and screenshots.

For now — this is a story about a platform I couldn't test. But one that perfectly illustrates the difference between retail and institutional algorithmic trading.

Institutional players pay millions for infrastructure. Retail traders build robots from free open-source libraries.

Two different universes. ADL is from the one where the minimum platform fee costs as much as a nice car.

---

**Useful links:**

- [ADL official page](https://tradingtechnologies.com/trading/algo-trading/adl/)
- [TT Platform documentation](https://library.tradingtechnologies.com/)
