---
layout: post
title: "Why Your Builder-Made Robot Loses on Live Trading: 5 Debugging Mistakes No One Talks About"
description: "Backtest showed +300% annual returns. On live trading, minus 15% in a month. We break down the typical pitfalls of debugging visual strategies and how to avoid them."
date: 2025-12-23
image: /assets/images/blog/debug_visual_strategies.png
tags: [debugging, backtesting, mistakes, visual builders, testing]
lang: en
---

Two weeks ago I received a message from a reader. He had assembled a strategy in TSLab. A three-year historical backtest showed fantastic results: +280% annual returns, maximum drawdown of 8%.

He deployed the strategy to a demo account. After one month, the result: minus 12%.

What went wrong? The problem wasn't the builder. The problem was **how he tested**.

This is a classic story. Visual builders make assembling a strategy easy. But they don't make it **correct**. And most mistakes happen not during assembly, but during testing.

In this article — five pitfalls that 90% of builder beginners step on. And how to avoid them.

## Mistake #1: Over-Optimization (Curve Fitting)

**What it is:**

You take a strategy, run parameter optimization. Try SMA from 10 to 100 with step 1. Try RSI from 20 to 80 with step 5. Find the combination that gives the best result on historical data.

Congratulations: you've just created a strategy that works **only** on that specific historical period.

**Why it's dangerous:**

[Curve fitting is when a strategy adapts to historical data so strongly](https://www.quantifiedstrategies.com/curve-fitting-trading/) that it stops working on new data. You found not a market pattern, but random noise.

**Real example:**

SMA cross optimization on 2020-2023 data. Best result: SMA(37) and SMA(83). Return +180% per year.

Run on 2024: minus 5%.

Why? Because the 37/83 combination has no logical basis. It's fitting to noise.

**How to recognize it:**

- Too many parameters (more than 3-4)
- Perfect historical results (200%+ annual without drawdowns)
- [Parameters look random](https://www.buildalpha.com/3-simple-ways-to-reduce-the-risk-of-curve-fitting/) (37, 83 instead of round numbers like 20, 50)
- Results drop sharply when changing a parameter by 1-2 units

**How to avoid it:**

### 1. Limit the number of parameters

[For classical testing, use no more than 2 optimizable parameters](https://empirix.ru/pereoptimizacziya-strategij/). The fewer — the better.

A simple strategy lives longer. A complex one dies quickly.

### 2. Out-of-Sample testing

Split the history into two parts:
- **In-Sample** (70%): Parameter optimization
- **Out-of-Sample** (30%): Results verification

If Out-of-Sample results are significantly worse — over-optimization.

In TSLab: Optimize on 2020-2022, test on 2023.

In Designer: Same logic, manually change the period.

### 3. Walk-Forward Analysis

Even more reliable: [run a sliding window](https://eatradingacademy.com/help/strategy-builders/expert-advisor-studio/strategy-tools-optimization/walk-forward-testing/).

Example:
- Optimize on 2020-2021, test on 2022
- Optimize on 2021-2022, test on 2023
- Optimize on 2022-2023, test on 2024

If the strategy holds across all periods — it's robust.

### 4. Check parameter stability

Build a heat map of optimization results.

If the best result is a single "hot spot" amid a sea of red — that's over-optimization.

If there's a wide "plateau" of good results — the strategy is stable to parameter changes. That's good.

TSLab and NinjaTrader show 3D optimization charts. Use them.

## Mistake #2: Look-Ahead Bias

**What it is:**

Your strategy accidentally uses information that **wasn't yet available** at the time of decision-making.

**Classic example:**

You use an indicator on the bar's **close**, but the signal is generated at the **open** of the next one.

Problem: when a bar closes, you already know its High/Low/Close. In real trading — you don't.

**Where it occurs:**

### In TSLab:

[TSLab counts bar time as the start time](https://doc.tslab.pro/tslab/rabota-s-programmoi/torgovlya-agentami-robotami/rabota-agenta-i-osobye-situacii). If you don't account for this — it's easy to create look-ahead.

Example: The "Close Price" block on candle N returns a value that will only be known **after** that candle closes.

If you generate a signal based on Close[0] — that's look-ahead. You should use Close[1].

### In Designer:

Same thing. Designer works on closed candles. If your logic is built on the current candle — check if that data is available in real time.

### In NinjaTrader:

Strategy Builder has an option "Calculate on bar close". If disabled — signals are generated on every tick, including unclosed candles. If enabled — only on close.

For most strategies, you need "Calculate on bar close = true".

**How to avoid it:**

1. **Use only closed candles**
   - If strategy is on H1, signal appears only after the hourly candle closes
   - Don't use current candle data for signal generation

2. **Check data delays**
   - Macroeconomic data is published with delays
   - News doesn't appear instantly
   - [Financial reports get revised](https://analystprep.com/study-notes/cfa-level-2/problems-in-backtesting/)

3. **Run on demo before live testing**
   - If backtest shows 100 trades per month, and demo shows 10, the problem is look-ahead

## Mistake #3: Survivorship Bias

**What it is:**

You test a strategy on stocks that **exist today**. But over three years, some companies went bankrupt, were delisted, or acquired.

They're not in your backtest. But they existed in real trading.

**Real example:**

Strategy on Russian stocks. Backtest for 2020-2023. The list of tested stocks includes:
- Sberbank ✅
- Gazprom ✅
- Yandex ✅
- TCS Holding ✅

But missing:
- Rusal (delisted in 2022) ❌
- Moscow Exchange (temporary delisting 2022) ❌
- Stocks that dropped 90% and disappeared from radar ❌

Your strategy "forgot" about losses on those instruments. [Survivorship bias inflates returns by 1-4% per year](https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained/).

**Where it occurs:**

### In TSLab and Designer:

If you load stock lists through a broker connection — you only get **current** stocks. Delisted ones aren't there.

### In NinjaTrader:

Same issue with futures. Expired contracts often don't make it into backtests.

**How to avoid it:**

1. **Use databases with delisted securities**
   - [QuantConnect, Norgate Data](https://medium.com/auquan/backtesting-biases-and-how-to-avoid-them-776180378335) provide survivorship-bias-free data
   - For the Russian market — harder, few such databases exist

2. **Test on an index, not cherry-picked stocks**
   - If strategy is on MOEX stocks — take the entire MOEX index, not just top 10

3. **Check how many securities disappeared during the test period**
   - If testing 3 years and the stock list hasn't changed — there's a problem

4. **Add liquidity filters**
   - Strategy shouldn't trade stocks with daily volume under 10 million rubles
   - This reduces the risk of getting into stocks before delisting

## Mistake #4: Ignoring Commissions, Slippage, and Execution Realities

**What it is:**

Backtests assume: you always buy at your desired price. Orders execute instantly. Commission = 0.

Reality: commissions, slippage, delays, partial fills.

**Real example:**

Strategy on minute bars. 200 trades per month. Average profit per trade: 0.15%.

Broker commission: 0.05% entry, 0.05% exit. Total 0.1% round-trip.

**Net profit:** 0.15% - 0.1% = 0.05% per trade.

200 trades * 0.05% = 10% per month. Seems fine.

But add slippage of 0.03% per trade. Now: 0.15% - 0.1% - 0.03% = **0.02%**.

200 trades * 0.02% = **4% per month**. Not so impressive anymore.

And if the spread is wide (illiquid stock), slippage of 0.1%? The strategy is **unprofitable**.

**How to avoid it:**

### 1. Configure commissions in the builder

**TSLab:**
Settings → Trading → Commissions. Enter your broker's actual commissions (typically 0.03-0.05%).

**Designer:**
The backtest window has a "Commission" field. Set it in absolute values (dollars/rubles) or percentages.

**NinjaTrader:**
Strategy → Properties → Commission. Enter commission per contract.

**fxDreema:**
In the generated MQL code, you need to add spread checks manually.

### 2. Add slippage

TSLab and NinjaTrader allow configuring slippage separately. For a retail trader on liquid stocks: 1-3 ticks.

For illiquid ones: 5-10 ticks or more.

### 3. Test on real spread

If the strategy trades inside the spread (scalping) — check if profit covers the spread size.

Simple formula:
```
Profit per trade > Commission * 2 + Average Spread + Slippage
```

If not — the strategy won't survive live trading.

### 4. Check trade count

[The more trades, the stronger the impact of commissions](https://www.quantifiedstrategies.com/survivorship-bias-in-backtesting/).

100 trades per year — commissions aren't critical.

1000 trades per year — commissions can eat all the profit.

**Rule:** If strategy yields <0.5% per trade after commissions — it's on the edge. The slightest market deterioration will kill it.

## Mistake #5: No Forward Testing

**What it is:**

A backtest is a test on the past. A forward test is a test on the future (but without real money).

[Forward testing shows how a strategy performs on data it has never seen](https://academy.ftmo.com/lesson/forward-testing-of-trading-strategies/).

**Why it matters:**

Suppose you optimized a strategy on 2020-2023. Results are excellent. You launch it live in 2024.

Problem: the market in 2024 may behave differently. Volatility changed. Correlations broke.

Forward testing on a demo account lets you verify this **before** you lose money.

**How to do Forward Testing:**

### 1. Run on a demo account

**Minimum duration:** [3-6 months](https://academy.ftmo.com/lesson/forward-testing-of-trading-strategies/).

Why so long? Because:
- You need to catch different market regimes (trend, range, volatility)
- You need a minimum of 50-100 trades
- You need to check psychological resilience (yes, even on demo)

### 2. Keep a trade journal

Record:
- Entry/exit
- Trade reason (which block generated the signal)
- Deviation from backtest (if any)

If demo results are **significantly** worse than backtest — something broke. Go back to debugging.

### 3. Compare metrics

| Metric | Backtest | Forward Test |
|--------|----------|--------------|
| Win Rate | 65% | ? |
| Average Profit | 1.2% | ? |
| Average Loss | -0.8% | ? |
| Maximum Drawdown | 12% | ? |
| Trades/month | 20 | ? |

If deviation exceeds 20-30% — there's a problem.

### 4. Use paper trading on platforms

**TradingView:** [Free paper trading](https://wundertrading.com/journal/en/learn/article/paper-trading-tradingview) via virtual account.

**AlgoTest:** [Paper trading with detailed analytics](https://docs.algotest.in/strategy-builder/paper-trading-analysing/).

**TSLab/Designer:** Running on simulation with real broker connection (but without sending orders).

### 5. Don't rush

The most common mistake: test for a week on demo, see profit, deploy to live.

A week is nothing. You need at least 2-3 months to understand how the strategy behaves in different conditions.

## Checklist Before Launching a Strategy Live

Before pressing "Start" on a live account, go through this list:

### Testing

- [ ] Strategy tested on at least 2 years of history
- [ ] Out-of-sample test conducted (30% of history)
- [ ] Number of parameters ≤ 3
- [ ] Parameters are logically justified (not noise fitting)
- [ ] Results are stable when changing parameters by ±10%

### Biases

- [ ] Verified no look-ahead bias (closed candles only)
- [ ] Survivorship bias accounted for (or minimized with filters)
- [ ] Realistic commissions added (0.03-0.05%)
- [ ] Slippage added (1-3 ticks for liquid instruments)
- [ ] Strategy is profitable after commissions and slippage

### Forward Testing

- [ ] Strategy tested on demo account for at least 3 months
- [ ] At least 50 trades accumulated
- [ ] Demo results close to backtest (deviation <30%)
- [ ] Trade journal maintained
- [ ] Tested in different market regimes (trend, range, volatility)

### Risk Management

- [ ] Maximum risk per trade ≤ 2% of account
- [ ] Maximum backtest drawdown ≤ 20%
- [ ] Action plan exists for drawdown >15%
- [ ] Position size calculated based on instrument volatility

If even one item is not met — don't go live.

## Debugging Tools in Builders

### TSLab

**Pros:**
- Built-in debugger with step-by-step execution
- Trade visualization on charts
- Detailed report for each trade
- [3D optimization visualization](https://vc.ru/u/715109-tslab/204062-optimizaciya-mehanicheskih-torgovyh-sistem)

**Cons:**
- [No automatic out-of-sample test](http://forum.tslab.ru/ubb/ubbthreads.php?ubb=showflat&Number=86791)
- Issues with tick data

### StockSharp Designer

**Pros:**
- Flexible commission and slippage settings
- Support for tick and order book data
- Export to C# for deep debugging

**Cons:**
- Less debugging documentation
- Visualization weaker than TSLab

### NinjaTrader Strategy Builder

**Pros:**
- Visual Studio integration for code debugging
- Detailed execution logs
- Market Replay for step-by-step testing

**Cons:**
- Harder to set up for beginners
- Expensive ($1,500 for lifetime)

### fxDreema

**Pros:**
- Generates MQL code that can be debugged in MetaEditor
- MetaTrader visual tester

**Cons:**
- Free version limitations (10 connections between blocks)
- Need MQL knowledge for deep debugging

## Conclusions

Visual builders make strategy creation easy. But debugging remains hard.

**Five main mistakes:**

1. **Over-optimization** — fitting to historical noise
2. **Look-ahead bias** — using future data
3. **Survivorship bias** — ignoring delisted securities
4. **Ignoring commissions** — unrealistic execution assumptions
5. **No forward testing** — going live without demo verification

**What to do:**

- Limit parameters (≤3)
- Do out-of-sample tests
- Check for look-ahead bias
- Add realistic commissions and slippage
- Test on demo for at least 3 months

[A correct backtest](https://www.morpher.com/ru/blog/backtesting-trading-strategies) isn't about pretty equity curves. It's about an honest answer to the question: "Will this work live?"

If a backtest shows 300% annual returns — most likely there's an error somewhere. Realistic returns for retail algo trading: 20-50% annually with 10-20% drawdown.

If your results are much better — go back to the points above. You missed something.

---

**Useful links:**

Research and resources:
- [TradingView: How to Debug Pine Script](https://trading-strategies.academy/archives/401)
- [FTMO Academy: Forward Testing of Trading Strategies](https://academy.ftmo.com/lesson/forward-testing-of-trading-strategies/)
- [AlgoTest: Paper Trading Guide](https://docs.algotest.in/strategy-builder/paper-trading-analysing/)
- [QuantifiedStrategies: Curve Fitting in Trading](https://www.quantifiedstrategies.com/curve-fitting-trading/)
- [Build Alpha: 3 Ways to Reduce Curve-Fitting Risk](https://www.buildalpha.com/3-simple-ways-to-reduce-the-risk-of-curve-fitting/)
- [AlgoTrading101: What is Overfitting in Trading?](https://algotrading101.com/learn/what-is-overfitting-in-trading/)
- [Auquan: Backtesting Biases and How To Avoid Them](https://medium.com/auquan/backtesting-biases-and-how-to-avoid-them-776180378335)
- [LuxAlgo: Survivorship Bias Explained](https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained/)
- [Empirix: Strategy Over-Optimization](https://empirix.ru/pereoptimizacziya-strategij/)
- [LONG/SHORT: Backtesting Strategies on Historical Data](https://long-short.pro/uspeshnaya-proverka-algoritmicheskih-torgovyh-strategih-na-istoricheskih-dannyh-chast-1-oshibki-okazyvayuschie-vliyanie-309/)
- [TSLab Documentation: Agent Operation and Special Situations](https://doc.tslab.pro/tslab/rabota-s-programmoi/torgovlya-agentami-robotami/rabota-agenta-i-osobye-situacii)
- [EA Trading Academy: Walk Forward Testing](https://eatradingacademy.com/help/strategy-builders/expert-advisor-studio/strategy-tools-optimization/walk-forward-testing/)
