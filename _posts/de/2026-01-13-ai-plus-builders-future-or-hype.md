---
layout: post
title: "ChatGPT generiert die Idee, der Baukasten baut den Roboter: Zukunft des Algotradings oder voruebergehender Hype?"
description: "Ich habe einen Monat lang das Combo KI + visuelle Baukastensysteme getestet. Strategien ueber ChatGPT/Claude generiert, in TSLab zusammengebaut. Das ist passiert."
date: 2026-01-13
image: /assets/images/blog/ai_visual_builders.png
tags: [AI, ChatGPT, Claude, builders, automation]
lang: de
---

"Beschreibe mir eine Strategie basierend auf EMA-Kreuzung mit RSI-Filter."

ChatGPT liefert die Logik in 10 Sekunden. Ich oeffne TSLab, baue die Bloecke zusammen. In 15 Minuten — ein fertiger Roboter.

Klingt wie ein Traum. Aber funktioniert es in der Praxis?

Den letzten Monat habe ich das Combo getestet: KI fuer Ideengenerierung, visuelle Baukastensysteme fuer den Zusammenbau. Hier ist die Realitaet.

## Experiment: 10 Strategien von ChatGPT → TSLab

Von 10 Strategien: 3 zeigten Gewinn im Backtest (>20% jaehrlich), 5 waren nahe Null, 2 verlustbringend.

## Problem #1: KI versteht den Marktkontext nicht

ChatGPT generiert logisch korrekte Strategien. Kennt aber nicht: Instrumentenspezifika, aktuelles Marktregime, Ihren Handelsstil. KI braucht sehr praezise Anweisungen.

## Problem #2: Baukastensysteme begrenzen die Komplexitaet

[Claude kann komplexe Strategien generieren](https://medium.com/@austin-starks/i-let-claude-opus-4-create-a-trading-strategy-it-destroyed-the-market-c200bf1a19a4). Aber der visuelle Baukasten unterstuetzt das nicht. KI kann komplexere Strategien generieren, als der Baukasten zusammenbauen kann.

## Problem #3: KI halluziniert Indikatoren

ChatGPT schlaegt manchmal Indikatoren vor, die im Baukasten nicht existieren. Man muss wissen, welche Indikatoren der eigene Baukasten hat.

## Was funktioniert: Die richtigen Prompts

**Schlechter Prompt:** "Erfinde eine Handelsstrategie"

**Guter Prompt:** "Schlage eine Strategie fuer stuendliche EUR/USD-Kerzen (Forex) vor. Verwende nur diese Indikatoren: SMA, EMA, RSI, MACD. Durchschnittliche Volatilitaet 50 Pips/Tag. Ziel: 3-5 Trades pro Woche. Stop-Loss bis 30 Pips."

## Zukunft oder Hype?

**Das ist nicht die Zukunft. Es ist ein Werkzeug.**

KI + Baukastensysteme werden den Quant-Programmierer nicht ersetzen. Aber die Arbeit beschleunigen. Nuetzlich fuer Anfaenger, senkt die Einstiegshuerde. Aber kein Allheilmittel. Wenn Sie tiefes Verstaendnis wollen — lernen Sie Programmierung.

---

**Nuetzliche Links:**
- [Medium: Claude Trading Strategy](https://medium.com/@austin-starks/i-let-claude-opus-4-create-a-trading-strategy-it-destroyed-the-market-c200bf1a19a4)
- [PickMyTrade: Claude for Trading Guide](https://blog.pickmytrade.trade/claude-4-1-for-trading-guide/)
