---
layout: post
title: "Von Flussdiagrammen zu Code: Wie ich in 3 Monaten von Baukastensystemen zur Programmierung wechselte"
description: "Eine wahre Geschichte des Uebergangs von visuellen Baukastensystemen zur vollstaendigen Programmierung. Der Plan, Fehler, Werkzeuge und warum es einfacher ist, als man denkt."
date: 2025-12-30
image: /assets/images/blog/visual_to_code.png
tags: [learning, programming, Python, transition, builders]
lang: de
---

Vor einem Jahr baute ich Strategien in TSLab zusammen. Flussdiagramme, Drag-and-Drop, kein Code. Es funktionierte. Bis ich an die Grenzen stiess.

Ich brauchte einen benutzerdefinierten Indikator. Brauchte Echtzeit-Handelsstatistiken. Brauchte Integration mit einer externen API.

Der Baukasten konnte das nicht bewerkstelligen.

Ich beschloss, Programmierung zu lernen. Vor drei Monaten schrieb ich meine erste Zeile Python. Heute handelt mein Roboter, und der gesamte Code gehoert mir.

Das ist keine Geschichte eines "Programmiergenies." Es ist die Geschichte: "Jeder kann es schaffen, wenn er weiss, wo er anfangen soll."

## Warum ich mich entschied, Code zu lernen

**Ausloeser #1: An die Grenzen des Baukastens gestossen**

Ich wollte einen adaptiven Stop-Loss basierend auf ATR hinzufuegen. TSLab hat einen ATR-Block. Hat einen Stop-Loss-Block. Aber keinen Block fuer "Stop-Loss dynamisch pro Kerze basierend auf ATR anpassen."

**Ausloeser #2: Vendor Lock-In**

Alles, was ich in TSLab gebaut hatte, lebt nur in TSLab. Wenn die Plattform schliesst, aktualisiert, kaputtgeht — sind meine Strategien tot. Code in Python ist eine Datei. Sie gehoert mir fuer immer.

**Ausloeser #3: Neugier**

Ich verstand die Strategielogik. Sah die Verbindungen zwischen Bloecken. Aber was passiert *drinnen*? Der Baukasten verbarg die Komplexitaet. Wenn etwas nicht funktionierte, verstand ich nicht *warum*. Code gibt Kontrolle. Volle Kontrolle.

## Roadmap: 3 Monate von Null zum funktionierenden Roboter

### **Wochen 1-4: Python-Grundlagen**

Variablen, Datentypen, Bedingungen, Schleifen, Funktionen, Dateiverarbeitung. 1-2 Stunden pro Tag, 5 Tage die Woche.

**Erstes Ergebnis:** Ein Skript, das eine CSV-Datei mit Kursen liest, einen gleitenden Durchschnitt berechnet und ausgibt, wann SMA(20) SMA(50) kreuzt.

### **Wochen 5-8: Bibliotheken fuer Datenanalyse**

**Pandas**, **NumPy**, **Matplotlib**. Funktionen zur Berechnung jedes Indikators:

```python
import pandas as pd

def sma(data, period):
    return data['Close'].rolling(window=period).mean()

def rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### **Wochen 9-12: Backtrader — erstes Handelssystem**

```python
import backtrader as bt

class SmaCross(bt.Strategy):
    params = (('fast', 20), ('slow', 50),)

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(period=self.params.fast)
        self.sma_slow = bt.indicators.SMA(period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if self.crossover > 0:
            if not self.position:
                self.buy()
        elif self.crossover < 0:
            if self.position:
                self.sell()
```

Dieselbe Logik wie in TSLab. Aber **ich kontrolliere jede Zeile**.

## Fehler, die ich gemacht habe

1. **Alles auf einmal lernen wollen** — Informationsueberflutung. Loesung: Eine Quelle nach der anderen.
2. **Gelesen aber keinen Code geschrieben** — Regel: Fuer jede Stunde Theorie eine Stunde Praxis.
3. **Keine Projekte gemacht** — Ziel: Funktionierende Strategie auf Backtrader am Ende von 3 Monaten.
4. **Angst vor Fragen** — Stack Overflow, Reddit (r/algotrading). Menschen helfen, wenn die Frage gut formuliert ist.

## Wann Programmierung lernen Sinn macht und wann nicht

### **Lernen Sie Programmierung wenn:**
1. Sie an die Grenzen des Baukastens gestossen sind
2. Sie benutzerdefinierte Logik brauchen (ML, Arbitrage, Portfolios)
3. Sie Algotrading langfristig ernsthaft betreiben wollen

### **Lernen Sie keine Programmierung wenn:**
1. Ihre Strategie in die Baukasten-Bloecke passt und funktioniert
2. Sie keine Zeit haben (mind. 1-2 Stunden taeglich fuer 3 Monate)
3. Programmierung Abneigung hervorruft

## Was sich nach dem Wechsel zu Code aenderte

**Vorteile:** Volle Kontrolle, Plattformunabhaengigkeit, kostenlos, tiefes Verstaendnis, riesige Community.

**Nachteile:** Keine Visualisierung, mehr Zeit am Anfang, Debugging schwieriger, Lernaufwand.

## Fazit: Hat es sich gelohnt?

Vor einem Jahr dachte ich: "Programmierung ist fuer IT-Leute. Ich bin nur ein Trader." Heute verstehe ich: Programmierung ist ein Werkzeug. Wie Excel. Wie TradingView. Ich wurde kein Entwickler. Ich schrieb 500 Zeilen Code, die das tun, was ich brauche. Und das ist **genug**.

Programmierung fuer Algotrading bedeutet nicht "Programmierer werden." Es bedeutet "Ihre Idee ohne Einschraenkungen automatisieren." Und es ist einfacher, als Sie denken.

---

**Nuetzliche Links:**

- [Should I Use C# Or Python To Build Trading Bots?](https://spreadbet.ai/python-or-c-trading-bots/)
- [Top Languages for Building Custom Trading Bots](https://blog.traderize.com/posts/top-languages-trading-bots/)
- [AlgoTrading101: Quantitative Trader's Roadmap](https://algotrading101.com/learn/quantitative-trader-guide/)
- [Start Algorithmic Trading: Beginner's Roadmap](https://startalgorithmictrading.com/beginners-algo-trading-roadmap)
