---
layout: post
title: "Wie ein typischer Open-Source-Roboter von innen funktioniert: Architektur einer einfachen Strategie"
description: "Wir analysieren die echte Architektur von Trading-Robotern: von Schichten und Komponenten bis zu Entwurfsmustern. Freqtrade, NautilusTrader, Microservices, Event Sourcing — was in der Praxis funktioniert."
date: 2026-03-03
image: /assets/images/blog/opensource_robot_architecture.png
tags: [architecture, open-source, trading robots, Event Sourcing, microservices]
lang: de
---

Vor drei Monaten habe ich [Open-Source-Frameworks fuer MOEX verglichen](/de/blog/2026/02/24/comparing-lean-stocksharp-backtrader.html). Herausgefunden, was man waehlen sollte. Aber die Frage blieb: **Wie funktioniert das alles von innen?**

Heute geht es nicht um die Plattformwahl. Sondern darum, **was unter der Haube passiert**.

Ich analysierte die Architektur von vier populaeren Open-Source-Robotern: [Freqtrade](https://github.com/freqtrade/freqtrade), [NautilusTrader](https://github.com/nautechsystems/nautilus_trader), [Hummingbot](https://github.com/hummingbot/hummingbot) und das Microservices-basierte [MBATS](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System).

Fazit: Trotz verschiedener Sprachen und Ziele **wiederholen sich die Muster**.

## Schichtarchitektur: Das Fundament

Die meisten Trading-Roboter folgen einer 5-Schichten-Architektur:

```
Schicht 5: Kommunikation    <- Telegram, Web UI, API
Schicht 4: Strategie        <- Handelslogik
Schicht 3: Ausfuehrung & Risiko <- Orders, Risikomanagement
Schicht 2: Datenverarbeitung <- Indikatoren, Normalisierung
Schicht 1: Datenerfassung   <- Boersenanbindungen
```

## Entwurfsmuster

### 1. Event Sourcing
Speichert jede Zustandsaenderung als Ereignis. Entscheidend fuer Audit und Debugging.

### 2. CQRS
Trennt Lesen von Schreiben.

### 3. Microservices-Architektur
MBATS teilt das System in unabhaengige Services, die ueber Kafka/RabbitMQ kommunizieren.

### 4. Actor Model
Jeder Actor hat unabhaengigen Zustand, kommuniziert ueber Nachrichten. Kein geteilter Zustand = keine Race Conditions.

## Design-Checkliste

1. **Beginnen Sie mit einem Monolithen**
2. **Trennen Sie Schichten vom ersten Tag an**
3. **Entwerfen Sie fuer Tests** — Dependency Injection
4. **Fuegen Sie Observability vom ersten Tag hinzu**
5. **Planen Sie Persistenz**
6. **Microservices nur wenn:** >100 Instrumente, HFT, Team >3 Personen

Die Architektur bestimmt, wie weit Sie kommen. Spaghetti-Code funktioniert einen Monat. Richtige Architektur — Jahre.

---

**Nuetzliche Links:**

- [Freqtrade](https://github.com/freqtrade/freqtrade)
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader)
- [Hummingbot](https://github.com/hummingbot/hummingbot)
- [MBATS](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System)
