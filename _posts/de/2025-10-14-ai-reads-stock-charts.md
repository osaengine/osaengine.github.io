---
layout: post
title: "Kann KI Boersendiagramme lesen? Ein Experiment mit DistilBERT"
description: "Ein Entwickler brachte einem Sprachmodell bei, Preisbewegungen durch textuelle Beschreibungen von Charts vorherzusagen. Der Test mit ueber 200 Aktien der Moskauer Boerse ergab einen AUC von 0,53."
date: 2025-10-14
image: /assets/images/blog/llm_stock_charts.png
tags: [machine learning, Moscow Exchange, experiment]
lang: de
---

Mikhail Shardin fuehrte ein Experiment durch: Kann ein Sprachmodell Preise vorhersagen, wenn Charts in Textform beschrieben werden?

## Die Idee

Anstelle von Rohkursen erhielt das Modell Beschreibungen in natuerlicher Sprache: Preis steigt stark, Volumen nimmt zu, nahe am Widerstand.

Das DistilBERT-Modell wurde trainiert, um Kursanstiege am naechsten Tag vorherzusagen.

## Ergebnisse

Getestet an ueber 200 Aktien der Moskauer Boerse:

- Durchschnittlicher AUC: 0,53 (etwas besser als Zufall)
- Beste Ergebnisse: AFLT (0,72), RTSB (0,70), PIKK (0,70)
- Schlechteste Ergebnisse: PLZL (0,33), VJGZP (0,33)

Fuer Handelszwecke ist das Ergebnis schwach, aber das Modell hat Muster erkannt, ohne direkten Zugang zu Zahlen zu haben -- das allein ist schon interessant.

## Technologie

Python + PyTorch + Hugging Face + Docker. Walk-Forward-Validierung, vektorisierte Verarbeitung ueber pandas. Der gesamte Prozess ist reproduzierbar.

**Code auf GitHub:** [github.com/empenoso/llm-stock-market-predictor](https://github.com/empenoso/llm-stock-market-predictor)

---

**Quelle:** [Habr](https://habr.com/ru/articles/955612/) | **Autor:** Mikhail Shardin
