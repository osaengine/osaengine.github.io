---
lang: en
layout: faq_article
title: "What are exchange APIs and how do you work with them?"
section: technical
order: 3
---

Exchange APIs (Application Programming Interfaces) provide programmatic access to the functions of exchanges and brokers, allowing trading robots to receive data and send trading commands.

## Main API capabilities:

1. **Receiving market data:**
   - Real-time quotes.
   - Market depth (order book).
   - Historical data.

2. **Trading operations:**
   - Creating, modifying, and canceling orders.
   - Tracking the status of open positions.

3. **Account monitoring:**
   - Balance and margin requirements.
   - Trade history.

## Popular exchange APIs:

1. **[Interactive Brokers API](https://www.interactivebrokers.com/):**
   - Supports many instruments (stocks, forex, options).
   - Powerful tools for algorithmic trading.

2. **[Binance API](https://www.binance.com/):**
   - Suitable for cryptocurrency trading.
   - Simple integration and broad functionality.

3. **[Alpaca API](https://alpaca.markets/):**
   - Focused on the stock market.
   - Free access to data and testing.

## How to start working with APIs:

1. **Obtain access keys:**
   - Register with a broker or exchange.
   - Generate API keys for access.

2. **Connect through a library:**
   - Choose a suitable programming language (Python, C#, Java).
   - Install the library for working with the API.

3. **Test:**
   - Verify the algorithm on a test server.
   - Ensure correct data processing and command execution.

## Tips:

- Use sandbox environments for testing.
- Keep API keys secure.
- Monitor the robot's performance, especially during high-frequency trading.
