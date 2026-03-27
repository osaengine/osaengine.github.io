---
layout: post
title: "fxDreema: When You Have MetaTrader but Don't Want to Learn MQL"
description: "MetaTrader is installed, broker is connected, but writing MQL is painful. I found fxDreema -- a visual bot builder for MT4/MT5. Here's what happened and whether it's worth $100 a year."
date: 2025-11-25
image: /assets/images/blog/fxdreema_builder.png
tags: [fxDreema, MetaTrader, MT4, MT5, no-code, forex, visual builder]
lang: en
---

I had a problem. MetaTrader is installed, broker is connected, I have a strategy in my head. But to implement it, I need to learn MQL4 or MQL5. And that's a language with its own quirks, half-English documentation, and forums where half the answers are "read the manual."

I googled "MetaTrader without programming" and stumbled upon fxDreema. A visual bot builder. Drag blocks -- get an expert advisor. Sounds simple. I decided to check it out.

## How It Works (or Should Work)

fxDreema is a web application. **Important point:** this is not a product from MetaQuotes (the developers of MetaTrader). It's a third-party tool made by enthusiasts. A small team building a constructor for someone else's platform.

You go to the website, register, start building a strategy. No installations, no IDEs. Everything in the browser.

**The idea:** take blocks (indicators, conditions, actions), connect them with arrows, like a flowchart. The program generates MQL code. Download the file, drop it into MetaTrader -- bot ready.

In theory, it's beautiful. In theory.

I registered (free), opened the editor. And indeed -- visual blocks, like in Scratch or Node-RED. Drag, connect. There's a library of ready-made blocks: indicators, price checks, orders.

Built a simple strategy: if RSI is below 30 -- buy, if above 70 -- sell. Classic. Pressed "Generate Code," downloaded the .mq4 file. Dropped it into MetaTrader.

**It launched.** No errors. The bot trades.

First reaction: "Wow, this actually works."

## Then the Nuances Began

Simple strategies assemble easily. Moving average crossovers, RSI, MACD -- all available as ready-made blocks. 15-20 minutes and the bot is ready.

But I wanted to add a trailing stop. And that's when I discovered that the free version has a limit: **maximum 10 "connections"** between blocks.

10 connections is roughly 5-6 blocks with conditions. Enough for simple strategies. For anything more complex -- you hit the limit.

Okay, I thought, I'll buy the full version. Went to check prices.

**$95 per year.** Or $33 for 3 months.

I thought about it. $95 isn't outrageous. But the question is: what do I get for that money?

- Removal of the 10-connection limit
- MQL4 to MQL5 conversion (and vice versa)
- That seems to be it

No support. No indicator library updates. Just removing an artificial limitation.

## Trying to Build Something More Complex

I decided not to buy right away, but to see what I could squeeze out of the free version. Simplified the strategy, removed unnecessary checks, fit within 10 connections.

Generated the code. Ran it in MetaTrader on a demo account.

**Problem number one:** Visually everything looks clear -- blocks, arrows. But when the strategy starts losing money, debugging it in fxDreema is painful. You need to open the browser, look at the diagram, change blocks, regenerate code, drop it into MetaTrader, restart.

In normal code (in MQL or Python) you open the file, change a couple of lines, save. Here -- a whole cycle.

**Problem number two:** The generated MQL code looks... strange. Variables with auto-generated names, logic spread across functions, comments in English (if present at all). Hard to read. Even harder to modify manually.

So if fxDreema can't build what you need -- you're stuck. The code gets generated, but working with it like normal code won't fly.

## Comparison with What I've Already Tested

Over the past weeks I've tried various visual builders. Here's what emerges:

**TSLab/StockSharp Designer** -- flowcharts, you can see the logic, exportable to C#. Works with Russian brokers.

**NinjaTrader** -- table interface (not blocks), built for American futures. $1,500 for a license.

**fxDreema** -- flowcharts like Designer, but only for MetaTrader. $95 a year. And the free version has a hard complexity limit.

fxDreema has one advantage: it works in the browser. Nothing to install. Visit, build, download, run.

But that's also a disadvantage. Everything is online. If the site goes down -- you're without a tool.

**And here's the interesting part:** fxDreema is not an official MetaQuotes product. It's a third-party service that generates code for someone else's platform. A small team, the project lives on user subscriptions.

What happens if tomorrow MetaQuotes changes something in MQL and the code stops compiling? Or if fxDreema's developers shut down the project? Your diagrams will remain on their servers. The generated code is also tied to their architecture.

With official platforms (TSLab, NinjaTrader) at least it's clear they won't shut down next year. Here -- there's risk.

## Who This Is Really For

I thought about this for several days. Here's my conclusion.

fxDreema is suitable if:

- You already have MetaTrader (MT4 or MT5) and a broker
- You trade forex or CFDs through MetaTrader
- You need a simple indicator strategy (crossovers, levels, RSI/MACD)
- You don't want to learn MQL
- You're willing to pay ~$95/year for convenience

fxDreema is NOT suitable if:

- You trade Russian markets (MOEX, Russian futures)
- You need complex logic with many conditions
- You want a free solution (the 10-connection limit runs out very quickly)
- You plan to modify code manually (generated MQL is unreadable)
- You want stability and guarantees (it's a third-party service, not an official product)

## What I Did in the End

I didn't buy the subscription. Built one simple strategy in the free version, downloaded the code, dropped it into MetaTrader. Works.

But for my next strategy I simply opened an MQL5 textbook and wrote the code by hand. An hour to learn basic syntax, another hour to write -- and I have a working expert advisor. Without limits. Without subscriptions. With full control.

**The paradox:** fxDreema was created to eliminate the need to learn MQL. But when you hit the visual builder's limits, you end up deciding it would've been easier to just learn the language.

Paying $95 a year to a third-party service that could shut down at any moment, for a tool that saves a couple hours of learning? Everyone decides for themselves. For me, it didn't add up.

## Honest Conclusion

fxDreema is not a bad tool. It actually works. Flowcharts assemble easily, code generates, bots launch.

But it's a tool with a very narrow range of use:

- MetaTrader only (MT4/MT5)
- Simple strategies only (in the free version)
- Only if you're willing to pay to remove limits

If you already trade through MetaTrader, want to automate a simple indicator strategy, and don't want to deal with programming -- try the free version. Maybe 10 connections will be enough for you.

But if you plan to seriously pursue algorithmic trading -- learn MQL or switch to something more flexible. Visual builders sooner or later hit their limits. And then you'll have to code anyway.

I spent two days on fxDreema. Built three strategies, ran them in the tester, looked at results. Ended up going back to code.

Maybe it's just not for me. Or maybe visual builders are always a compromise between simplicity and control.

---

**Useful links:**

- [fxDreema official site](https://fxdreema.com/)
- [Documentation and examples](https://fxdreema.com/forum/)
