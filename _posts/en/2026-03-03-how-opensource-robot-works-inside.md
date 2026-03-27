---
layout: post
title: "How a Typical Open-Source Robot Works Inside: Architecture of a Simple Strategy"
description: "We break down the real architecture of trading robots: from layers and components to design patterns. Freqtrade, NautilusTrader, microservices, Event Sourcing — what works in practice."
date: 2026-03-03
image: /assets/images/blog/opensource_robot_architecture.png
tags: [architecture, open-source, trading robots, Event Sourcing, microservices]
lang: en
---

Three months ago, I [compared open-source frameworks for MOEX](/en/blog/2026/02/24/comparing-lean-stocksharp-backtrader.html). Figured out what to choose. But the question remained: **how does it all work inside?**

You can read the docs. You can study the API. But real understanding comes only when you open the code and see the architecture.

So today isn't about choosing a platform. It's about **what happens under the hood**.

I analyzed the architecture of four popular open-source robots: [Freqtrade](https://github.com/freqtrade/freqtrade), [NautilusTrader](https://github.com/nautechsystems/nautilus_trader), [Hummingbot](https://github.com/hummingbot/hummingbot), and the microservices-based [MBATS](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System).

Conclusion: despite different languages and goals, **patterns repeat**. If you understand them, you can design your own system right from the start.

## Layered Architecture: The Foundation

Most trading robots follow a 5-layer architecture:

```
Layer 5: Communication Layer     <- Telegram, Web UI, API
Layer 4: Strategy Layer          <- Trading logic
Layer 3: Execution & Risk        <- Orders, risk management
Layer 2: Data Processing         <- Indicators, normalization
Layer 1: Data Ingestion          <- Exchange connections
```

## Layer 1: Data Ingestion

**Task:** Get data from the exchange and deliver it to the system.

Freqtrade uses [ccxt](https://github.com/ccxt/ccxt) to support 100+ exchanges without writing separate connectors. NautilusTrader has its core in Rust with streaming up to 5 million rows per second.

**Takeaways:** Use adapters, unify interfaces, separate data retrieval from processing.

## Layer 2: Data Processing

**Task:** Transform raw prices into indicators, ML features, signals.

NautilusTrader writes all indicators in Rust with bounded memory. FreqAI integrates ML (LightGBM, PyTorch, Reinforcement Learning) directly into Freqtrade.

**Takeaways:** Separate feature generation from inference, cache indicator calculations, use bounded buffers.

## Layer 3: Execution & Risk Management

**Task:** Take a "buy" decision and turn it into a real order with risk controls.

Freqtrade includes dynamic position sizing and overtrading protection. Microservices architectures use Smart Order Routing (SOR) as a separate service.

## Layer 4: Strategy Layer

**Task:** Determine when to buy and sell.

Freqtrade completely separates strategy from execution. NautilusTrader uses the Actor Model for event-driven strategies with microsecond reactions.

## Layer 5: Communication & Monitoring

Freqtrade has built-in Telegram bot, REST API, and Web UI. Professional systems export metrics to Prometheus and visualize in Grafana.

## Design Patterns for Trading Robots

### 1. Event Sourcing

Saves every state change as an event. Crucial for auditing, debugging, and regulatory compliance.

### 2. CQRS (Command Query Responsibility Segregation)

Separates reads from writes. The Event Store handles writes; a separate database with pre-computed aggregates handles queries.

### 3. Microservices Architecture

MBATS splits the system into independent services communicating via Kafka/RabbitMQ. Each service scales independently. Different languages for different tasks.

### 4. Actor Model

NautilusTrader and Hummingbot use it. Each actor has independent state, communicates via messages. No shared state means no race conditions.

## Checklist: Designing Your Own Robot

1. **Start with a monolith** — don't jump to microservices prematurely
2. **Separate layers from day one** — even in a monolith
3. **Design for testing** — dependency injection, mocks, abstractions
4. **Add observability from day one** — logs, metrics, tracing
5. **Plan persistence** — save all orders, positions, settings, events
6. **Switch to microservices** only when: >100 instruments, HFT, team >3 developers

## Common Mistakes

1. **Everything in one file** — impossible to test, reuse, or maintain
2. **No exchange abstractions** — adding a second exchange means changing code in 50 places
3. **No logging or metrics** — you won't understand why the robot traded at 3 AM
4. **Synchronous code** — use async (aiohttp, not requests) for exchange APIs
5. **Ignoring backpressure** — use bounded queues to prevent memory growth

## Conclusions

**Key principles:**

1. Separate layers — data, processing, strategy, execution, communication
2. Use adapters — don't write exchange logic in strategies
3. Design for testing — dependency injection
4. Add observability — logs, metrics, tracing
5. Start with a monolith — don't over-engineer prematurely

Architecture determines how far you'll go. Spaghetti code works for a month. Proper architecture — for years.

---

**Useful links:**

- [Freqtrade](https://github.com/freqtrade/freqtrade)
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader)
- [Hummingbot](https://github.com/hummingbot/hummingbot)
- [MBATS Microservices Trading System](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System)
- [Architectural Design Patterns for HFT Bots](https://vocal.media/education/architectural-design-patterns-for-high-frequency-algo-trading-bots)
