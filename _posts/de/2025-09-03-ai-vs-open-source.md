---
layout: post
title: "KI vs Open Source: Was sich wirklich geaendert hat und wo die Grenze verlaeuft"
description: "Eine detaillierte Analyse, wie moderne Code-Modelle das Gleichgewicht zwischen Generierung und fertigen Bibliotheken im algorithmischen Handel veraendert haben."
date: 2025-09-03
image: /assets/images/blog/ai_vs_oss.png
tags: [AI, Open Source, algorithmic trading, development]
lang: de
---

Ich habe einen neuen Artikel auf Habr veroeffentlicht: **["KI vs Open Source: Was sich wirklich geaendert hat und wo die Grenze verlaeuft"](https://habr.com/ru/articles/943670/)**

Mit dem Erscheinen funktionierender Code-Modelle ist ein pragmatischerer Entwicklungsweg entstanden: Anforderung formulieren, Tests schreiben und ein kleines, verstaendliches Modul ohne ueberfluessige Abhaengigkeiten erhalten. Das ist kein Krieg gegen OSS -- es ist eine Verschiebung des Gleichgewichtspunkts.

## Kernaussagen des Artikels:

### Was sich geaendert hat
- **Frueher**: "Zuerst die Bibliothek." Bibliothek suchen, transitive Abhaengigkeiten akzeptieren, Dokumentation lesen.
- **Jetzt**: "Beschreibung -> Tests -> Implementierung." Kleine, testbare Module statt monolithischer "Kombiloesungen."

### Wo KI bereits Bibliotheken ersetzt
1. **Mini-Implementierungen**: Indikatoren (EMA/SMA/RSI), Statistiken, Risikoregeln
2. **Schmale Integrationen**: REST/WebSocket-Clients mit nur 2-3 benoetigten Methoden
3. **Geruest-Generierung**: Backtest-Grundgerueste, Datenschemata
4. **Adapter**: Mapping zwischen Boersen, Code-Migrationen

### Wo KI OSS NICHT ersetzen sollte
- Kryptographie und sichere Protokolle
- Binaere Protokolle (FIX/ITCH/OUCH/FAST)
- Datenbank-Engines, Compiler, Laufzeitumgebungen
- Numerische Solver und Optimierer

### Praktische Tipps
- Module klein halten
- Verhalten in einfachen Worten beschreiben
- Minimale Pruefungen fuer sichere Merges durchfuehren
- Ohne externe Abhaengigkeiten generieren

Im algorithmischen Handel ist das besonders relevant: Weniger Abhaengigkeiten bedeuten niedrigere Risiken, kompaktere Artefakte, einfachere Audits und schnellere Iterationen.

**Wichtigste Erkenntnis**: Waehlen Sie das Werkzeug passend zum Kontext. Eine eng gefasste Aufgabe, die leicht zu beschreiben und zu ueberpruefen ist, eignet sich fuer die Generierung. Alles andere -- dafuer besser bewaehrte OSS nutzen.
