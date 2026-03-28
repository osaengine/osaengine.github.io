---
lang: en
layout: faq_article
title: "Can a robot be launched on multiple exchanges or markets simultaneously?"
section: advanced
order: 1
---

A trading robot can be configured to operate on multiple exchanges or markets simultaneously. This allows you to diversify risks, exploit arbitrage opportunities, and increase potential profits.

## How to implement this:

1. **Support for multiple APIs:**
   - The robot must be connected to exchanges through their APIs. Most platforms, such as **[StockSharp](https://stocksharp.ru/)** or **[QuantConnect](https://www.quantconnect.com/)**, support connections to multiple markets.

2. **Data management:**
   - Each market provides its own data (quotes, order books), which the robot must process correctly.
   - Use data structures that allow you to separate information by exchange.

3. **Time synchronization:**
   - Different exchanges operate in different time zones. Make sure the robot is properly synchronized with their trading sessions.

4. **Arbitrage strategies:**
   - Use the robot to identify price discrepancies between exchanges.
   - Example: buying on one exchange and selling on another to profit from the price difference.

## Tips:

- Make sure your robot is optimized to process large amounts of data in real time.
- Start with a small number of exchanges to test the robot's performance.
- Regularly update API keys and monitor changes in exchange conditions.

## Software for multi-market operation:

- **[StockSharp](https://stocksharp.ru/):** A universal platform with support for multiple connections.
- **[QuantConnect](https://www.quantconnect.com/):** A cloud platform with support for multiple markets.
- **TSLab:** Suitable for automating work with multiple exchanges, but requires preliminary configuration.
