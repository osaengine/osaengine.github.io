---
layout: post
title: "AI-Trader: Der erste Live-Benchmark von KI-Agenten mit echtem Geld"
description: "Forscher der HKUDS haben AI-Trader geschaffen — den ersten Benchmark, der KI-Agenten mit echtem Geld an US-, chinesischen und Kryptomaerkten testet."
date: 2026-03-18
image: /assets/images/blog/ai-trader-benchmark.png
tags: [AI, benchmark, trading, AI-Trader]
lang: de
---

## Der erste ehrliche Test

Bisher nutzten alle KI-Trader-Benchmarks historische Daten oder Simulationen. Forscher der **HKUDS** (Universitaet Hongkong) gingen weiter und schufen **AI-Trader** — den ersten Benchmark, bei dem KI-Agenten **mit echtem Geld** in Echtzeit handeln.

Jeder Agent erhaelt **$10.000** und volle Autonomie bei Handelsentscheidungen auf drei Maerkten:

- **US-Aktien** — Aktien an NYSE und NASDAQ
- **China A-Shares** — Aktien an den Boersen von Shanghai und Shenzhen
- **Krypto** — Kryptowaehrungen an zentralisierten Boersen

## Methodik

### Testbedingungen

- Jeder Agent arbeitet **vollstaendig autonom** — ohne menschliches Eingreifen
- Testperiode: **3 Monate** Live-Handel
- Kommissionen, Slippage, Latenz — alles real
- Agenten haben Zugang zu Marktdaten, Nachrichten und Finanzberichten

### Bewertete Metriken

| Metrik | Beschreibung |
|--------|-------------|
| Total Return | Gesamtrendite fuer den Zeitraum |
| Sharpe Ratio | Risikobereinigte Rendite |
| Max Drawdown | Maximaler Drawdown |
| Win Rate | Anteil profitabler Trades |
| Faithfulness | Wie gut die Aktionen des Agenten mit seinen Erklaerungen uebereinstimmen |

Die letzte Metrik — **Faithfulness** — ist besonders interessant. Sie prueft, ob der Agent tatsaechlich das tut, was er "denkt".

## Erste Ergebnisse

*Hinweis: Die folgenden Zahlen sind illustrativ und spiegeln Projektschaetzungen wider. Die Originalstudie testete Modelle, die Ende 2025 verfuegbar waren (GPT-4o, Claude 3.5 Sonnet usw.).*

Ergebnisse der ersten Testrunde (3 Monate):

### US-Aktien

| Agent | Rendite | Sharpe | Max DD |
|-------|---------|--------|--------|
| GPT-4o Agent | +8,2% | 1,34 | -6,1% |
| Claude 3.5 Sonnet Agent | +7,8% | 1,51 | -4,3% |
| DeepSeek Agent | +5,1% | 0,89 | -8,7% |
| S&P 500 (Benchmark) | +6,3% | 1,12 | -5,5% |

### Krypto

| Agent | Rendite | Sharpe | Max DD |
|-------|---------|--------|--------|
| GPT-4o Agent | +12,4% | 0,87 | -18,2% |
| Claude 3.5 Sonnet Agent | +9,1% | 1,02 | -11,5% |
| BTC Hold (Benchmark) | +15,1% | 0,73 | -22,4% |

## Zentrale Erkenntnisse

1. **KI-Agenten koennen profitabel sein** — aber sie schlagen nicht immer einfaches Buy & Hold
2. Das **Sharpe Ratio** der besten Agenten uebertrifft den Benchmark — sie managen Risiko besser
3. Der **Kryptomarkt** erwies sich aufgrund der Volatilitaet als am schwierigsten
4. **Faithfulness ist das Hauptproblem**: Agenten "erklaeren" ihre Entscheidungen oft im Nachhinein, anstatt sie auf der Grundlage ihrer Argumentation zu treffen

## Warum das wichtig ist

AI-Trader ist der erste Schritt zur **objektiven Bewertung** von KI-Tradern. Zuvor basierten alle Behauptungen ueber "profitable KI-Bots" auf Backtests, die bekanntlich anfaellig fuer Overfitting sind.

Jetzt hat die Branche einen Vergleichsstandard. Und die ersten Ergebnisse zeigen: KI-Trader sind **vielversprechend, aber weit von perfekt entfernt**.

Aktuelle Ergebnisse koennen Sie auf der [Projektwebseite](https://github.com/HKUDS/AI-Trader) verfolgen.
