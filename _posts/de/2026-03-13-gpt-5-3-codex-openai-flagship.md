---
layout: post
title: "GPT-5.3 Codex: OpenAI aktualisiert sein Flaggschiffmodell"
description: "OpenAI hat GPT-5.3 Codex veroeffentlicht — eine aktualisierte Version des Flaggschiffmodells mit verbesserten Programmierfaehigkeiten. Wir vergleichen es mit Claude 4.6 Opus."
date: 2026-03-13
image: /assets/images/blog/gpt-5-3-codex.png
tags: [AI, GPT-5, OpenAI, coding]
lang: de
---

## GPT-5.3 Codex: Was ist neu

OpenAI entwickelt die GPT-5-Reihe weiter. Die neue Version **GPT-5.3 Codex** wird als das beste Modell des Unternehmens fuer Programmieraufgaben positioniert. Laut OpenAI zeigt das Modell erhebliche Verbesserungen bei:

- **Code-Generierung** in allen gaengigen Programmiersprachen
- **Debugging und Refactoring** bestehender Codebasen
- **Code-Erklaerung** — das Modell "erkennt" Projektarchitekturen besser
- **Test-Generierung** — die Erstellung von Unit-Tests ist deutlich praeziser geworden

## Benchmarks

Ergebnisse unabhaengiger Tests:

| Test | GPT-5.3 Codex | Claude Opus 4.6 | Claude Sonnet 4.6 |
|------|---------------|-----------------|-----------------|
| SWE-Bench Verified | 78.4% | 76.1% | 82.1% |
| HumanEval+ | 95.8% | 94.3% | 96.2% |
| MBPP+ | 88.2% | 87.1% | 89.7% |
| Codeforces Rating | 1847 | 1792 | 1801 |

GPT-5.3 Codex uebertrifft Claude Opus 4.6 bei Programmieraufgaben deutlich, bleibt aber hinter Claude Sonnet 4.6 bei SWE-Bench zurueck.

## Wichtige Verbesserungen

### Erweiterter Code-Kontext

GPT-5.3 Codex verfuegt ueber **128K Token Kontext**, optimiert fuer Code-Dateien. OpenAI behauptet, das Modell koenne die Struktur eines Projekts mit mehreren hundert Dateien im "Gedaechtnis" behalten.

### Verbessertes Function Calling

Fuer Entwickler, die die API nutzen, ist das **Function Calling** zuverlaessiger geworden. Das Modell erstellt JSON-Aufruf-Schemata praeziser und "erfindet" seltener nicht existierende Parameter.

### Codex Agent-Modus

OpenAI hat den **Codex Agent**-Modus vorgestellt, in dem das Modell:

- Befehle sequentiell im Terminal ausfuehren kann
- Dateien lesen und aendern kann
- Tests ausfuehren und anhand der Ergebnisse iterieren kann

Dies ist eine direkte Antwort auf **Claude Code** von Anthropic und aehnliche Agentenprodukte.

## Preise

GPT-5.3 Codex ist ueber die API zu folgenden Preisen verfuegbar:

- **Input**: $8 / 1M Token
- **Output**: $24 / 1M Token
- **Gecachter Input**: $2 / 1M Token

Damit liegt das Modell im mittleren Preissegment — teurer als DeepSeek, aber guenstiger als Claude Opus.

## Was fuer Trading-Bots waehlen?

Fuer Entwickler algorithmischer Handelssysteme haengt die Wahl zwischen GPT-5.3 und Claude von der Aufgabe ab:

- **Fuer das Schreiben von Strategien von Grund auf** — Claude Sonnet 4.6 zeigt die besten Ergebnisse
- **Fuer die Integration mit bestehenden APIs** — GPT-5.3 Codex gewinnt durch praezises Function Calling
- **Fuer die Analyse von Marktdaten** — beide Optionen funktionieren gut, aber GPT-5.3 ist bei Streaming-Generierung schneller

Der Wettbewerb zwischen den Modellen wird immer staerker, und das sind grossartige Nachrichten fuer die Endnutzer.
