---
layout: post
title: "Trading-Ideen finden mit ChatGPT und Claude: Von den Daten zum Backtest"
description: "Wir untersuchen, wie KI für Datenanalyse, das Finden von Ineffizienzen und die Erstellung einer Handelsstrategie am Beispiel von Kryptowährungs-Minutendaten genutzt werden kann."
date: 2024-11-02
image: /assets/images/blog/ai-trading-strategy-preview.png
tags: [ChatGPT, Claude]
lang: de
---

In diesem Artikel habe ich mich entschieden, zwei beliebte Dienste zu vergleichen — [ChatGPT](https://chatgpt.com/) und [Claude.ai](https://claude.ai/) — und zu sehen, wie sie mit der Aufgabe umgehen, Handels-Ineffizienzen im November 2024 zu finden. Ich habe ihre Funktionalität und Benutzerfreundlichkeit bewertet, um herauszufinden, welcher besser für Datenanalyse und die Entwicklung einer profitablen Handelsstrategie geeignet ist.

Um die Datenerfassung zu vereinfachen, habe ich **[Hydra](https://stocksharp.ru/store/%D1%81%D0%BA%D0%B0%D1%87%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BC%D0%B0%D1%80%D0%BA%D0%B5%D1%82-%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85/)** verwendet — wohl das beste kostenlose Tool zum Herunterladen von Marktdaten.

Ich habe BTCUSDT-Minutendaten für 2024 heruntergeladen, die etwa 25 MB umfassten, und sie in eine CSV-Datei exportiert.

![](/assets/images/blog/hydra_2.png)

![](/assets/images/blog/hydra_3.png)

Hydra hat zwar eigene Analysefunktionen, aber weiter unten werdet ihr sehen, wie veraltet diese im Vergleich zu den KI-Fähigkeiten sind, bei denen man nicht einmal selbst Code schreiben muss:

![](/assets/images/blog/hydra_4.png)

Der Hauptteil meiner Arbeit war jedoch nicht die Datenerfassung, sondern deren Analyse und die Suche nach Strategie-Ideen. Anstatt manuell nach Ansätzen zu suchen, beschloss ich, der KI zu vertrauen und herauszufinden, welche Strategien sie vorschlagen würde, welche Muster und Ineffizienzen sie in den Daten identifizieren könnte und wie Parameter für Tests optimiert werden können. Mit Hilfe von **ChatGPT** konnte ich nicht nur eine detaillierte Analyse durchführen, sondern auch einen Backtest der Strategie mit den Daten durchführen.

---

### Datenvorbereitung

Nachdem ich die Minutendaten erhalten hatte, lud ich sie in Python (den Code schrieb die KI — ich tippte nur in Klartext ein, was ich brauchte) und begann mit der Vorverarbeitung. Dazu gehörte die Benennung jeder Spalte und das Zusammenführen von Datum und Uhrzeit in eine einzige Spalte zur Vereinfachung der Zeitreihenanalyse.

---

### Ineffizienzen finden mit KI

Nach der Datenvorverarbeitung fragte ich die KI nach möglichen Ineffizienzen und Mustern, die für die Strategieentwicklung nützlich sein könnten. ChatGPT schlug mehrere Ansätze vor:

1. **Volatilitäts-Cluster** — Stunden mit hoher Volatilität könnten für eine Momentum-Strategie geeignet sein.
2. **Tendenz zur Mean Reversion** — Bei Abweichungen des Preises vom Durchschnittsniveau könnte eine Mean-Reversion-Strategie angewendet werden.
3. **Momentum-Muster** — Zu bestimmten Stunden wurden nachhaltige Preisbewegungen beobachtet, die als Signale für eine Trendfolge-Strategie dienen könnten.

![](/assets/images/blog/volatility-clusters.png)

---

### Strategieentwicklung

Basierend auf den Vorschlägen der KI wählte ich zwei Strategien zum Testen:

1. **Mean Reversion**: Eröffnung einer Short-Position bei starker Preisabweichung nach oben vom Durchschnitt und einer Long-Position bei Abweichung nach unten. Die Position wird geschlossen, wenn der Preis zum Mittelwert zurückkehrt.

2. **Momentum-Strategie**: Eröffnung einer Position in Trendrichtung bei erhöhter Volatilität. Wenn die Rendite positiv und über der Schwelle liegt, wird eine Kaufposition eröffnet; wenn negativ und unter der Schwelle — eine Verkaufsposition.

Für jede Strategie wurden grundlegende Ein- und Ausstiegsregeln sowie Stop-Losses für das Risikomanagement definiert.

![](/assets/images/blog/hourly-returns.png)

---

### Backtesting der Strategien

Mit Hilfe von ChatGPT konnte ich beide Strategien auch backtesten, um zu sehen, wie sie auf historischen Daten abgeschnitten hätten. Die Testergebnisse zeigten die Equity-Kurve für die Mean-Reversion-Strategie (siehe Diagramm unten).

Das Diagramm zeigt, wie sich die Portfoliokapitalisierung bei Befolgung der Strategie hätte entwickeln können. Man kann sehen, dass die Strategie in bestimmten Zeiträumen ein stetiges Wachstum zeigte, es aber auch Phasen des Drawdowns gab. Dies bestätigt die Bedeutung der Parameteroptimierung und des Risikomanagements.

![](/assets/images/blog/mean-reversion-equity-curve.png)

---

### Claude.ai

Während meiner Arbeit versuchte ich auch **Claude Sonnet** von Anthropic zu nutzen, das kürzlich seine Datenanalysefunktion angekündigt hatte (mehr Details [hier](https://www.anthropic.com/news/analysis-tool)). Die Idee klang vielversprechend: eine 25-MB-Datei hochladen, damit Claude bei der Analyse helfen kann.

![](/assets/images/blog/claude_analytics.png)

Allerdings stieß ich auf einige Schwierigkeiten. Leider erwies sich die Funktion als unausgereift — meine Datei ließ sich nicht einmal hochladen. Am Ende teilte ich sie in kleinere Teile auf, aber aufgrund vorheriger Fehler erreichte ich schnell das Anfragelimit. Alles, was ich bekam, war ein Fehler beim Versuch, ein Diagramm zu erstellen.

![](/assets/images/blog/claude_error_1.png)

Obwohl ich gerne mit Claude arbeite, hoffe ich, dass die Ingenieure des Projekts diese Funktion verfeinern und das Daten-Upload-Fenster erheblich erweitern werden. Das würde eine effizientere Analyse großer Dateien ermöglichen und neue Möglichkeiten für die Arbeit mit großen Datenmengen eröffnen.

![](/assets/images/blog/claude_error_2.png)

---

### Fazit

Die Nutzung von ChatGPT ermöglichte es mir, nicht nur Daten zu analysieren, sondern der KI auch Fragen zu geeigneten Methoden für die Strategieerstellung zu stellen. Dieser Ansatz generierte nicht nur neue Ideen, sondern half auch, Hypothesen schnell zu testen und Empfehlungen zu erhalten, die bei einem traditionellen Ansatz möglicherweise unbemerkt geblieben wären. Der Ansatz, bei dem KI hilft, Ideen und Strategieparameter zu entdecken, eröffnet neue Möglichkeiten für eine flexible und adaptive Entwicklung von Handelsstrategien.
