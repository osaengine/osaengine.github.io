---
layout: post
title: "Ich habe 5 Trading-Roboter-Baukaesten getestet. Das wuerde ich fuer mich waehlen"
description: "Ein Monat Testen visueller Baukaesten: TSLab, StockSharp Designer, NinjaTrader, fxDreema und der Versuch, ADL zu erreichen. Ehrlicher Vergleich ohne Tabellen und Marketing."
date: 2025-12-09
image: /assets/images/blog/constructors_comparison.png
tags: [comparison, TSLab, StockSharp, NinjaTrader, fxDreema, ADL, visual builders]
lang: de
---

Vor einem Monat startete ich ein Experiment: alle populaeren visuellen Trading-Strategie-Baukaesten testen. Die Idee war einfach — verstehen, ob man 2025 wirklich algorithmischen Handel ohne Programmierung betreiben kann.

Ich ging durch fuenf Plattformen. Auf einigen baute ich funktionierende Roboter. Bei anderen stiess ich an Grenzen. Bei einer konnte ich nicht einmal Zugang bekommen.

Dies wird keine Vergleichstabelle mit Haekchen. Dies ist die Geschichte dessen, was ich gelernt habe, womit ich konfrontiert wurde und was ich jetzt fuer mich waehlen wuerde.

## Wie ich getestet habe

**Die Kriterien waren einfach:**

1. Kann ich Zugang bekommen? (Demo, Trial, kostenlose Version)
2. Kann ich in einer Stunde eine einfache Strategie bauen?
3. Laeuft sie auf einer echten/Demo-Plattform?
4. Was passiert, wenn die Strategie komplexer wird?
5. Was kostet es wirklich?

**Plattformen:** TSLab, StockSharp Designer, NinjaTrader Strategy Builder, fxDreema, ADL von Trading Technologies.

## TSLab: Wenn man eine fertige Loesung will

**Was es ist:** Russische Plattform mit visuellem Baukasten. Flussdiagramme, Drag-and-Drop, Integration mit russischen Brokern.

Erste Strategie in 20 Minuten gebaut. Aber bei zunehmender Komplexitaet werden die Diagramme zum Labyrinth. Akzeptable Backtest-Leistung. Hervorragende Integration mit russischen Brokern.

Preis: 60.000 Rubel/Jahr. Kein Code-Export. Klassischer Vendor Lock-in.

**Hauptnachteil:** Preis, kein Code-Export, Vendor Lock-in.

**Hauptvorteil:** Fertige Loesung mit Support und einfacher russischer Broker-Integration.

## StockSharp Designer: Professionelle Plattform ohne Kosten

**Was es ist:** [Professionelle Algotrading-Plattform](https://stocksharp.com/) mit visuellem Designer-Baukasten. Ueber 90 Boersenanbindungen weltweit, kostenlos fuer Privatpersonen. Strategieexport in C#-Code.

Die Schluesselfunktion: visuell gebaute Strategie in ein vollstaendiges, unabhaengiges C#-Projekt exportieren. Sauberer, lesbarer Code in Visual Studio. Funktioniert auf VPS ohne GUI.

[Unterstuetzt ueber 90 Verbindungen](https://doc.stocksharp.ru/): russische Broker (QUIK, Transaq, Tinkoff, Alor), internationale (Interactive Brokers, LMAX), Kryptowaehrungen (Binance, Bitfinex).

Leistung: fast doppelt so schnell wie TSLab. Kostenlos fuer Privatpersonen ohne Einschraenkungen.

**Hauptnachteil:** Erfordert Lernzeit.

**Hauptvorteil:** Kostenlos, C#-Code-Export, 90+ Konnektoren, Leistung, kein Vendor Lock-in.

## NinjaTrader Strategy Builder: Der amerikanische Standard

Amerikanische Plattform fuer Futures. Tabelleninterface (keine Bloecke). Keine Unterstuetzung russischer Broker. Lebenslange Lizenz: $1.499. Exzellenter Backtester und riesige Community.

**Hauptnachteil:** Keine Unterstuetzung des russischen Marktes.

**Hauptvorteil:** Professioneller Backtester, grosse Community.

## fxDreema: MetaTrader im Browser

Web-Anwendung zur Erstellung von MetaTrader-EAs. Kostenlose Version auf 10 Verbindungen begrenzt. Pro: $99/Jahr.

**Hauptnachteil:** Abhaengigkeit von Drittanbieter-Service, Risiko der Projektschliessung.

**Hauptvorteil:** Schneller Einstieg, guenstig.

## ADL: Enterprise-Loesung

Algo Design Lab von Trading Technologies. Mindestens $1.500/Monat. $18.000/Jahr. Fuer einzelne Trader nicht zugaenglich.

## Was ich fuer mich waehlen wuerde

### Wenn ich den russischen Markt handle

**StockSharp Designer.** Kostenlos, C#-Export, 90+ Konnektoren, professionelle Leistung.

### Wenn ich amerikanische Futures handle

NinjaTrader oder StockSharp.

### Wenn ich ueber MetaTrader handle

fxDreema fuer sehr einfache Strategien. Aber besser ein Wochenende MQL lernen.

## Ehrliches Schlussfazit

Visuelle Baukaesten funktionieren, aber sie **stossen irgendwann an Grenzen.** Einfache Strategien lassen sich leicht bauen. Aber bei zunehmender Komplexitaet: Flussdiagramme werden zu Spaghetti, kostenlose Versionen stossen an Limits, die Leistung leidet.

**Das Paradox:** Baukaesten wurden geschaffen, um Programmierung zu vermeiden. Aber fuer ernsthafte Arbeit muss man trotzdem programmieren.

Meine persoenliche Entscheidung: StockSharp Designer zum Prototyping, Export in C#-Code fuer die Produktion.

| Kriterium | StockSharp | TSLab | NinjaTrader | fxDreema | ADL |
|-----------|-----------|-------|-------------|----------|-----|
| **Preis/Jahr** | 0 | 60k | 100-150k | 10k | ~1,8M |
| **Russischer Markt** | 5/5 | 4/5 | 0/5 | 2/5 | 1/5 |
| **Einfachheit** | 4/5 | 5/5 | 3/5 | 4/5 | ? |
| **Flexibilitaet** | 5/5 | 3/5 | 3/5 | 2/5 | 5/5 |
| **Leistung** | 5/5 | 3/5 | 4/5 | 2/5 | 5/5 |
| **Vendor Lock-in** | Minimal | Hoch | Mittel | Niedrig | Hoch |

---

**Nuetzliche Links:**

- [StockSharp offizielle Website](https://stocksharp.com/)
- [TSLab offizielle Website](https://www.tslab.pro/)
- [NinjaTrader](https://ninjatrader.com/)
- [fxDreema](https://fxdreema.com/)
- [Trading Technologies ADL](https://tradingtechnologies.com/trading/algo-trading/adl/)
