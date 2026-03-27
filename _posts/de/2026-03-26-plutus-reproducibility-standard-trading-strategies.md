---
layout: post
title: "PLUTUS: Ein neuer Reproduzierbarkeitsstandard für Handelsstrategien"
description: "Das Open-Source-Framework PLUTUS standardisiert die Beschreibung und das Testen algorithmischer Handelsstrategien. Wir erklären, warum das wichtig ist."
date: 2026-03-26
image: /assets/images/blog/plutus-framework.png
tags: [open-source, algo trading, PLUTUS, standards]
lang: de
---

## Das Problem der Reproduzierbarkeit

Im algorithmischen Handel gibt es ein grundlegendes Problem: Wenn jemand eine "profitable Strategie" veröffentlicht, ist es praktisch unmöglich, die Ergebnisse zu reproduzieren. Die Gründe:

- **Nicht angegebene Parameter** -- der Autor hat wichtige Einstellungen nicht erwähnt
- **Unterschiedliche Daten** -- Datenquellen liefern leicht unterschiedliche Kurse
- **Versteckte Annahmen** -- Gebühren, Slippage, Ausführungszeit
- **Plattformunterschiede** -- derselbe Algorithmus liefert auf verschiedenen Backtesting-Engines unterschiedliche Ergebnisse

Das **PLUTUS**-Framework wurde geschaffen, um dieses Problem zu lösen.

## Was ist PLUTUS

**PLUTUS** ist ein Open-Source-Framework zur standardisierten Beschreibung, zum Testen und zur Veröffentlichung von Handelsstrategien.

Entwickelt von einer internationalen Forschergruppe und auf [GitHub](https://github.com/algotrade-plutus) unter MIT-Lizenz veröffentlicht.

## Architektur

PLUTUS definiert vier standardisierte Komponenten:

### 1. Strategy Specification (Strategiespezifikation)

Eine formale Beschreibung der Strategie im YAML/JSON-Format:

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

### 2. Data Specification (Datenspezifikation)

Standardisierte Datenbeschreibung:

- Quelle (Yahoo Finance, Polygon, MOEX)
- Zeitraum (Anfang, Ende)
- Frequenz (1 Minute, 1 Stunde, 1 Tag)
- Verarbeitung (adjustiert/nicht adjustiert, Füllmethode)
- Daten-Hash zur Verifizierung

### 3. Backtest Engine (Backtesting-Engine)

Eine standardisierte Backtesting-Engine mit:

- Definierter Orderverarbeitungslogik
- Fester Berechnungsreihenfolge innerhalb eines Bars
- Transparentem Slippage-Modell
- Bericht mit über 50 Kennzahlen

### 4. Report Format (Berichtsformat)

Ein einheitliches Berichtsformat, das umfasst:

- Equity-Kurve
- Alle Kennzahlen (Sharpe, Sortino, Max DD, Calmar usw.)
- Handelsverteilung
- Zeitperiodenanalyse
- Walk-Forward-Ergebnisse

## Warum das wichtig ist

### Für Forscher

Die Veröffentlichung einer Strategie im PLUTUS-Format ermöglicht es anderen Forschern, die Ergebnisse **exakt zu reproduzieren**. Das ist das, was die Wissenschaft für Experimente längst hat, dem algorithmischen Handel aber bisher fehlte.

### Für Praktiker

Das standardisierte Format vereinfacht:

- **Strategievergleich** -- alle Kennzahlen werden gleich berechnet
- **Audit** -- jeder Parameter kann überprüft werden
- **Portabilität** -- Übertragung einer Strategie zwischen Plattformen

### Für KI-Agenten

PLUTUS ist besonders nützlich für LLM-Agenten, die Handelsstrategien generieren. Das standardisierte Format ermöglicht:

- Automatische Validierung der Spezifikation
- Durchführung von Backtests ohne manuelle Einrichtung
- Vergleich der Ergebnisse mit einem Benchmark

## Aktueller Status

- **Version**: 0.8 (Beta)
- **Sprachen**: Python (primär), Adapter für C# und Java
- **Unterstützte Märkte**: USA, EU, China, Krypto
- **Integrationen**: Backtrader, Zipline, VectorBT, QuantConnect

PLUTUS ist ein Schritt hin zu einem transparenteren und wissenschaftlicheren algorithmischen Handel. Wenn Sie Handelsstrategien entwickeln, lohnt sich ein Blick darauf.
