---
layout: post
title: "PLUTUS: A New Reproducibility Standard for Trading Strategies"
description: "The open-source PLUTUS framework standardizes the description and testing of algorithmic trading strategies. We explore why this matters."
date: 2026-03-26
image: /assets/images/blog/plutus-framework.png
tags: [open-source, algo trading, PLUTUS, standards]
lang: en
---

## The Reproducibility Problem

Algorithmic trading has a fundamental problem: when someone publishes a "profitable strategy," reproducing the results is virtually impossible. The reasons:

- **Unspecified parameters** -- the author forgot to mention key settings
- **Different data** -- data sources give slightly different prices
- **Hidden assumptions** -- commissions, slippage, execution time
- **Platform differences** -- the same algorithm yields different results on different backtesting engines

The **PLUTUS** framework was created to solve this problem.

## What Is PLUTUS

**PLUTUS** is an open-source framework for the standardized description, testing, and publication of trading strategies.

Developed by an international research group and published on [GitHub](https://github.com/algotrade-plutus) under the MIT license.

## Architecture

PLUTUS defines four standardized components:

### 1. Strategy Specification

A formal description of the strategy in YAML/JSON format:

```yaml
strategy:
  name: "Mean Reversion RSI"
  version: "1.0"
  author: "researcher@university.edu"

  signals:
    entry_long:
      condition: "RSI(14) < 30 AND SMA(50) > SMA(200)"
    exit_long:
      condition: "RSI(14) > 70 OR stop_loss(-2%)"

  parameters:
    rsi_period: 14
    sma_fast: 50
    sma_slow: 200
    stop_loss_pct: -2.0

  universe:
    type: "equity"
    market: "US"
    filter: "S&P 500 constituents"

  execution:
    order_type: "market"
    slippage_model: "fixed_bps(5)"
    commission_model: "per_share(0.005)"
```

### 2. Data Specification

Standardized data description:

- Source (Yahoo Finance, Polygon, MOEX)
- Period (start, end)
- Frequency (1 minute, 1 hour, 1 day)
- Processing (adjusted/unadjusted, fill method)
- Data hash for verification

### 3. Backtest Engine

A standardized backtesting engine with:

- Defined order processing logic
- Fixed intra-bar calculation order
- Transparent slippage model
- Report with 50+ metrics

### 4. Report Format

A unified report format that includes:

- Equity curve
- All metrics (Sharpe, Sortino, Max DD, Calmar, etc.)
- Trade distribution
- Time-period analysis
- Walk-forward results

## Why This Matters

### For Researchers

Publishing a strategy in the PLUTUS format allows other researchers to **exactly reproduce** the results. This is what the scientific world has long had for experiments, but what algorithmic trading has been missing.

### For Practitioners

A standardized format simplifies:

- **Strategy comparison** -- all metrics are calculated the same way
- **Auditing** -- every parameter can be verified
- **Portability** -- transferring a strategy between platforms

### For AI Agents

PLUTUS is especially useful for LLM agents that generate trading strategies. The standardized format enables:

- Automatic validation of the specification
- Running backtests without manual setup
- Comparing results against a benchmark

## Current Status

- **Version**: 0.8 (beta)
- **Languages**: Python (primary), adapters for C# and Java
- **Supported markets**: US, EU, China, Crypto
- **Integrations**: Backtrader, Zipline, VectorBT, QuantConnect

PLUTUS is a step toward making algo trading more transparent and scientific. If you are developing trading strategies, it is worth paying attention to.
