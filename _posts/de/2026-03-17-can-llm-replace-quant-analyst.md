---
layout: post
title: "Kann ein LLM einen Quant-Analysten ersetzen? Praktisches Szenario zur Strategieentwicklung mit ChatGPT / Claude"
description: "Experiment: Wir entwickeln eine Handelsstrategie von der Idee bis zum Backtest nur mit LLMs. Was funktionierte, wo sie versagten, und ob Quants sich Sorgen machen sollten."
date: 2026-03-17
image: /assets/images/blog/llm_quant_analyst.png
tags: [LLM, ChatGPT, Claude, quant analyst, strategy development, automation]
lang: de
---

Vor einer Woche habe ich [Alpha Arena analysiert]({{site.baseurl}}/2026/03/10/ii-roboty-na-realnom-rynke-alpha-arena.html) — einen Benchmark fuer KI-Trader mit echtem Geld. Fazit: LLMs koennen handeln, aber nicht immer gut.

Heute ist die Frage eine andere: **Kann ein LLM einen Quant-Analysten im Strategieentwicklungsprozess ersetzen?**

Nicht selbst handeln. Sondern einem Menschen helfen, den Weg zu gehen: Idee -> Recherche -> Code -> Backtest -> Optimierung.

Ich habe ein Experiment durchgefuehrt. ChatGPT und Claude bekamen die Aufgabe: **"Entwickle eine Handelsstrategie fuer BTC/USDT von Grund auf."** Kein Code von mir. Keine fertigen Bibliotheken. Nur Prompts und LLMs.

Das Ergebnis war ueberraschend. Das LLM bewältigte 70% der Quant-Aufgaben. Aber die restlichen 30% zeigten, **wo Menschen noch unersetzlich sind**.

Gehen wir den gesamten Prozess Schritt fuer Schritt durch, mit echten Prompts, Code und Schlussfolgerungen.

## Was ein Quant-Analyst tut: Workflow-Zerlegung

Bevor wir pruefen, ob ein LLM einen Quant ersetzen kann, muessen wir verstehen, **was ein Quant eigentlich tut**.

### **Ein typischer Tag eines Quant-Analysten**

[Laut CQF](https://www.cqf.com/blog/day-life-quantitative-analyst) besteht der Arbeitstag eines Quants aus:

**09:00 - 10:00:** E-Mails und Standup
**10:00 - 12:00:** Modellwartung (Pipelines, Bugs, Optimierung)
**12:00 - 13:00:** Mittagspause
**13:00 - 17:00:** Forschung und Entwicklung (neue Ideen, Modelle, Backtesting)
**17:00 - 18:00:** Praesentationen und Berichte

### **Workflow der Strategieentwicklung:**

```
1. Ideengenerierung
   ↓
2. Recherche (Literatur, Daten)
   ↓
3. Hypothesenformulierung
   ↓
4. Datensammlung und -aufbereitung
   ↓
5. Modell-/Strategieentwicklung
   ↓
6. Code schreiben
   ↓
7. Backtesting
   ↓
8. Ergebnisanalyse
   ↓
9. Optimierung
   ↓
10. Dokumentation und Praesentation
```

## Das Experiment: Strategieentwicklung nur mit LLMs

**Aufgabe:** Eine vollstaendige Strategie fuer BTC/USDT von Grund auf entwickeln, nur mit ChatGPT und Claude.

## Phase 1: Ideengenerierung

ChatGPT generierte 5 statistische Strategieideen. Gewaehlt wurde **Autocorrelation Breakout** — BTC zeigt negative Autokorrelation auf 1-Stunden-Basis (Momentum-Umkehrungen).

**Bewertung:** LLM erreichte **7/10** bei der Ideengenerierung. Logische Ideen, aber kritische Ueberpruefung noetig.

## Phase 2: Recherche

Claude fand eine echte akademische Arbeit von Charfeddine & Maouchi (2019) ueber Autokorrelation bei Kryptowaehrungen und empfahl ein 168-Stunden-Fenster. Allerdings war eine Referenz falsch (Halluzination).

**Bewertung:** Recherche **6/10**. Nuetzlich, aber erfordert Faktencheck.

## Phase 3: Strategiecode

ChatGPT schrieb vollstaendigen Python-Code mit Backtest inklusive Kommissionen. Der Code lief fehlerfrei, aber die Strategie verlor Geld (-7,54%, Sharpe -0,23).

**Fazit:** Das LLM schrieb perfekten Code, aber **die Strategie funktioniert nicht**.

## Phase 4: Debugging und Optimierung

Claude identifizierte korrekt, dass der Schwellenwert -0,3 zu streng war. Mit Schwellenwert -0,2 und 12-Stunden-Exit: +8,72%, Sharpe 0,47.

## Phase 5: Optimierungsautomatisierung

ChatGPT generierte Grid-Search-Code in 2 Minuten. Beste Kombination: Einstieg < -0,25, Ausstieg > -0,09 oder 12 Stunden, Rendite +13,42%, Sharpe 0,78.

**Problem:** Das LLM warnte nicht vor Overfitting.

## Phase 6: Walk-Forward-Validierung

Claude implementierte den Walk-Forward-Test korrekt. Ergebnis: **starkes Overfitting** (Sharpe faellt von 0,78 auf 0,35, durchschnittliche Degradation 0,42).

## Phase 7: Kampf gegen Overfitting

ChatGPT schlug 3 Methoden vor. Der Ensemble-Ansatz funktionierte am besten:

```bash
Ensemble-Ergebnisse:
  Trades: 127
  Rendite: +9,84%
  Sharpe: 0,52
  Out-of-Sample Sharpe: 0,48
  Degradation: 0,04
```

Overfitting nahezu beseitigt.

## Zusammenfassung: Wo das LLM erfolgreich war und wo es versagte

| Aufgabe | Ergebnis | Bewertung | Kommentar |
|---------|----------|-----------|-----------|
| Ideengenerierung | 5 Strategien in 30 Sek | 5/5 | Alle logisch und testbar |
| Recherche | 1 echte Arbeit, 1 falsche | 3/5 | Faktencheck noetig |
| Code schreiben | Funktioniert beim ersten Versuch | 5/5 | Sauberer Code |
| Backtesting | Korrekte Implementierung | 5/5 | Kommissionen beruecksichtigt |
| Debugging | Problem korrekt identifiziert | 4/5 | Kann aber nicht selbst testen |
| Optimierung | Grid Search in 2 Min | 5/5 | Warnte nicht vor Overfitting |
| Walk-Forward | Korrekte Implementierung | 4/5 | Schlug keine Loesung vor |
| Anti-Overfitting | 3 Methoden, 1 funktionierte | 5/5 | Senior-Level |

## Prognose: Was wird mit Quant-Analysten geschehen

**Szenario 1: Augmentation (am wahrscheinlichsten)** — LLMs werden Quants nicht ersetzen, sondern verstaerken. Analogie: Taschenrechner haben Mathematiker nicht ersetzt, aber ihre Arbeit veraendert.

**Szenario 2: Demokratisierung (mittlere Wahrscheinlichkeit)** — LLMs machen quantitative Analyse fuer Nicht-Programmierer zugaenglich. Nachfrage nach Junior-Quants sinkt; nach Senior-Quants steigt.

**Szenario 3: Vollstaendiger Ersatz (geringe Wahrscheinlichkeit)** — Falls ueberhaupt, nicht vor 2035-2040.

## Fazit

**Kann ein LLM einen Quant-Analysten ersetzen?**

**Kurze Antwort:** Nein. Aber es kann ihn 5-mal produktiver machen.

LLMs werden Quants nicht ersetzen. Aber Quants, die keine LLMs nutzen, werden von denen ersetzt, die es tun.

---

**Nuetzliche Links:**

- [Quant Strats 2025: Integrating LLMs](https://biztechmagazine.com/article/2025/03/quant-strats-2025-4-ways-integrate-llms-quantitative-finance)
- [Automate Strategy Finding with LLM](https://arxiv.org/html/2409.06289v3)
- [Prompt Engineering for Traders](https://roguequant.substack.com/p/prompt-engineering-for-traders-how)
- [LLM Hallucinations in Finance](https://arxiv.org/html/2311.15548)
- [A Day in the Life of a Quantitative Analyst](https://www.cqf.com/blog/day-life-quantitative-analyst)
