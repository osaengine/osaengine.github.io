---
layout: post
title: "TradeTrap: Wie zuverlässig sind LLM-Trader wirklich?"
description: "Die TradeTrap-Studie deckte ernsthafte Probleme mit der Zuverlässigkeit und Faithfulness von LLM-Tradern auf. Wir untersuchen, warum KI-Bots anders entscheiden als sie erklären."
date: 2026-03-20
image: /assets/images/blog/tradetrap-llm-reliability.png
tags: [AI, LLM, trading, reliability, research]
lang: de
---

## Das Faithfulness-Problem

Wenn ein LLM-Trader seine Entscheidung erklärt -- "Ich habe AAPL gekauft, weil der RSI Überverkauft anzeigt und die Quartalszahlen über den Erwartungen lagen" -- hat er sich tatsächlich von diesen Faktoren leiten lassen? Oder ist die Erklärung eine **nachträgliche Rationalisierung**, während die eigentliche "Entscheidung" aus völlig anderen Gründen getroffen wurde?

Die **TradeTrap**-Studie eines Forscherteams hat genau diese Frage untersucht.

## Forschungsmethodik

Die Forscher schufen eine kontrollierte Umgebung, in der:

1. LLM-Agenten **Marktdaten und Nachrichten** für Handelsentscheidungen erhielten
2. Ein Teil der Daten **absichtliche Fallen** (Traps) enthielt -- falsche Signale, die überzeugend aussahen
3. Die Agenten Entscheidungen treffen und diese **erklären** mussten
4. Die Forscher die **angegebenen** Gründe mit den **tatsächlichen** Auslösern verglichen

### Arten von Fallen

- **Anker-Falle** -- ein zufälliger "Zielkurs" wurde in den Kontext eingefügt, ohne analytische Grundlage
- **Rezenz-Falle** -- aktuelle Daten waren schlechter als der Durchschnitt, aber der Trend blieb positiv
- **Autoritäts-Falle** -- gefälschte Zitate "bekannter Analysten" mit falschen Prognosen
- **Bestätigungs-Falle** -- Daten, die die bestehende Voreingenommenheit des Modells bestätigten

## Ergebnisse

### Fallenquote

*Hinweis: Die Tabellen verwenden Modelle, die zum Zeitpunkt der Studie (Ende 2025) verfügbar waren.*

| Modell | Ankerung | Rezenz | Autorität | Bestätigung |
|--------|----------|--------|-----------|-------------|
| GPT-4o | 34 % | 41 % | 28 % | 52 % |
| Claude 3.5 Sonnet | 22 % | 35 % | 19 % | 44 % |
| DeepSeek V3 | 39 % | 48 % | 33 % | 57 % |
| Gemini 2.0 Flash | 31 % | 38 % | 25 % | 49 % |

### Faithfulness-Score

Wie gut die Erklärungen des Modells mit den tatsächlichen Entscheidungsgründen übereinstimmen:

| Modell | Faithfulness |
|--------|-------------|
| Claude 3.5 Sonnet | 67 % |
| GPT-4o | 61 % |
| Gemini 2.0 Flash | 58 % |
| DeepSeek V3 | 54 % |

Das bedeutet, dass in **33-46 % der Fälle** die Erklärungen der LLM-Trader **nicht mit den tatsächlichen Gründen** ihrer Entscheidungen übereinstimmen.

## Wichtigste Erkenntnisse

### 1. Bestätigungsfehler ist das größte Problem

Alle Modelle zeigten die größte Anfälligkeit für die **Bestätigung eigener Vorurteile**. Wenn ein Modell "beschlossen" hat, einen Vermögenswert zu kaufen, findet es Daten, die diese Entscheidung stützen, selbst wenn objektive Daten das Gegenteil sagen.

### 2. Chain-of-Thought hilft nicht

Selbst Reasoning-Modelle mit ausführlicher Argumentationskette (Chain-of-Thought) sind anfällig für Fallen. Mehr noch: Eine lange Argumentationskette **maskiert** manchmal unzuverlässige Entscheidungen und erzeugt die Illusion einer tiefgehenden Analyse.

### 3. Die Fehlerkosten steigen mit der Autonomie

Je mehr Autonomie ein LLM-Trader hat, desto teurer wird jeder Faithfulness-Fehler. Wenn der Agent automatisch Orders auf Basis fehlerhafter Begründungen platziert, können die Konsequenzen schwerwiegend sein.

## Praktische Empfehlungen

Die Autoren der Studie empfehlen:

- **Vertrauen Sie den Erklärungen** von LLM-Tradern nicht -- überprüfen Sie Entscheidungen unabhängig
- **Verwenden Sie Ensemble**-Ansätze -- mehrere Modelle stimmen über eine Entscheidung ab
- **Begrenzen Sie die Autonomie** -- Mensch-in-der-Schleife für große Trades
- **Testen Sie mit adversarialen Daten** -- prüfen Sie, wie der Agent auf Fallen reagiert
- **Protokollieren Sie alle Zwischenschritte** -- für die Post-Mortem-Analyse von Fehlern

## Was das für die Branche bedeutet

TradeTrap ist ein wichtiges Signal für alle, die KI-Handelssysteme bauen. **Ein hoher Benchmark auf SWE-Bench oder MMLU bedeutet nicht Zuverlässigkeit im Handel.** Es werden spezialisierte Tests benötigt, die kognitive Fallen und Faithfulness berücksichtigen.

Der vollständige Studientext ist auf [arXiv](https://arxiv.org/abs/2512.02261) verfügbar.
