---
layout: post
title: "NinjaTrader Strategy Builder - fast ein visueller Baukasten"
description: "Ich habe eine Woche mit dem NinjaTrader Strategy Builder verbracht, um herauszufinden, ob er $1.500 wert ist. Spoiler: Wenn Sie russische Maerkte handeln -- nein. Wenn amerikanische Futures -- auch fraglich."
date: 2025-11-18
image: /assets/images/blog/ninjatrader_strategy_builder.png
tags: [NinjaTrader, Strategy Builder, no-code, futures, international markets]
lang: de
---

Als ich vom NinjaTrader Strategy Builder hoerte, klangen die Versprechen grossartig: ein visueller Bot-Baukasten, kein Code, eine riesige Community, ein professionelles Werkzeug. Ich beschloss herauszufinden, ob es wirklich funktioniert oder ob es nur eine schoene Verpackung fuer ein teures Produkt ist. Spoiler: naja.

## Erster Eindruck: Wo sind die Flussdiagramme?

NinjaTrader ist eine amerikanische Plattform fuer Futures. E-mini S&P 500, Nasdaq, Oel, Gold -- alles serioees, alles professionell. Sie haben einen Strategy Builder -- einen "visuellen" Baukasten.

Nur ist er nur in einem sehr weiten Sinne visuell.

Wenn Sie TSLab oder StockSharp Designer gesehen haben, dort gibt es echte visuelle Flussdiagramme: Sie ziehen Bloecke, verbinden sie mit Pfeilen und erhalten ein Diagramm.

**Bei NinjaTrader ist alles anders.** Die Oberflaeche ist wie Excel: eine Tabelle mit Spalten und Zeilen. Sie erstellen Bedingungen wie Filter:
- Zeile 1: Indikator SMA(50) > SMA(200)
- Zeile 2: RSI < 30
- Aktion: Kaufen

Keine Bloecke. Keine Pfeile. Nur eine Tabelle mit Bedingungen.

Ehrlich? Die ersten 10 Minuten habe ich versucht zu finden, wo man den "normalen" visuellen Modus aktiviert. Es stellte sich heraus -- das IST der visuelle Modus.

**Aber es gibt einen Haken.** NinjaTrader ist fuer internationale Maerkte gebaut. Die russische MOEX? Vergessen Sie es. Man kann ueber Umwege und FIX API verbinden, aber das ist so muehsam, dass man besser gleich ein anderes Werkzeug waehlt.

![NinjaTrader Strategy Builder Oberflaeche]({{site.baseurl}}/assets/images/blog/ninjatrader_strategy.png)

## Was versprochen wird vs. was man bekommt

**Im Marketing klingt alles fantastisch:**

Visueller Baukasten! Backtesting! Optimierung! Indikatorbibliothek! Broker-Integration! NinjaScript in C#!

Ich lud die Demo-Version herunter. Versuchte auf den Strategy Builder zuzugreifen. Erste Ueberraschung: **Die kostenlose Version gibt keinen Zugang zum Baukasten**. Man muss den Support anschreiben und um eine "Simulationslizenz" bitten. Okay, habe ich gemacht. Am naechsten Tag bekam ich sie.

**Begann eine einfache Strategie zu bauen:** Kreuzung zweier gleitender Durchschnitte.

Die Tabellenoberflaeche erwies sich als ziemlich logisch. Bedingung hinzugefuegt, Indikator gewaehlt, Parameter gesetzt. Strategie in 20 Minuten gebaut. Backtest mit E-mini S&P 500 Daten gestartet.

**Funktioniert.** Charts, Statistiken, Win Rate -- alles da.

Aber dann versuchte ich etwas Komplexeres. Einen Volumenfilter hinzufuegen. Die Handelszeit pruefen. Verschachtelte AND/OR-Bedingungen hinzufuegen.

Und da begann die Verwirrung. Im Tabellenformat ist es schwer, die Logik zu verfolgen: Welche Bedingung ist mit welcher verknuepft, wo ist AND, wo OR. In TSLab/Designer ist das visuell im Diagramm klar -- Bloecke, Pfeile, man sieht die ganze Struktur. Hier muss man die Tabelle wie Code lesen.

**Erste Erkenntnis:** NinjaTraders Tabellenoberflaeche funktioniert fuer einfache Strategien. Aber sie ist weniger anschaulich als Flussdiagramme der russischen Gegenstuecke. Fuer komplexe Strategien wechselt man sowieso zu NinjaScript (C#-Code).

## Was kostet das Vergnuegen

Hier wird es interessant.

**Kostenlos kann man:**
- Charts ansehen
- Backtests durchfuehren
- Strategien im Baukasten erstellen (aber nur zum Testen!)
- Handel simulieren

**Aber um einen Bot mit echtem Geld zu starten:**
- **Monatlich:** ~$100/Monat (~$1.200/Jahr)
- **Dauerhaft:** ~$1.500 einmalig

Ich starrte lange auf diese Zahlen. $1.500. Fuer eine Handelsplattform. Die nur mit internationalen Maerkten funktioniert. Wo die Dokumentation nur auf Englisch ist. Wo der Support in einem Tag antwortet.

**Realitaetscheck:** Fuer $1.500 koennte man einen vernuenftigen Programmierer engagieren, der eine Strategie in Python oder C# nach Ihren spezifischen Beduerfnissen schreibt. Mit Quellcode. Mit Dokumentation. Ohne Plattformbindung.

Oder fuer das gleiche Geld ein jaehrliches Datenabo kaufen, einen VPS mieten, und es bleibt noch uebrig.

## Versuch, russische Maerkte anzubinden

Ich gab nicht auf. Googelte "NinjaTrader MOEX." Fand einige Forumsthreads. Leute versuchen sich ueber FIX API zu verbinden. Manche schreiben behelfsmäßige Konnektoren.

**Selbst versucht.**

NinjaTraders Dokumentation fuer benutzerdefinierte Konnektoren ist schmerzhaft. Man muss in C# schreiben, ihre Architektur verstehen, testen, debuggen. Am Ende wurde mir klar: **Es ist einfacher, einen Bot von Grund auf zu schreiben**, als zu versuchen, einen russischen Broker in NinjaTrader zu integrieren.

Die Frage: Wozu ein visueller Baukasten, wenn die Verbindung zum eigenen Broker trotzdem Programmierung erfordert?

**Zweite Erkenntnis:** NinjaTrader ist fuer amerikanische Futures. Punkt. Wenn Sie MOEX handeln -- vergessen Sie diese Plattform.

## Was wirklich funktioniert und was nicht

**Funktioniert:**

Einfache Indikatorstrategien lassen sich schnell zusammenbauen. Gleitende-Durchschnitt-Kreuzung in 15 Minuten. Backtesting mit historischen Daten -- auch in Ordnung. Schoene Charts, detaillierte Statistiken.

**Funktioniert nicht (oder nur schmerzhaft):**

1. **Komplexe Strategien.** Sobald mehr als 5-7 Bedingungen hinzukommen, wird die Tabellenoberflaeche unleserlich. Im Gegensatz zu Flussdiagrammen (TSLab/Designer), wo man die visuelle Struktur mit Bloecken und Verbindungen sieht, muss man hier die Tabelle durchlesen. Unleserlich. Nicht debugbar. Man wechselt zum Code.

2. **Russische Broker.** Anbindung moeglich. Ueber Umwege, FIX API und mehrere Tage Qual. Frage: Wozu?

3. **Dokumentation.** Alles auf Englisch. Foren auf Englisch. Beispiele auf Englisch. Wenn Sie kein Englisch lesen, wird es sehr frustrierend.

4. **Support.** Antwortet langsam. Ich schrieb wegen des Zugangs zur Simulationslizenz -- Antwort nach 18 Stunden. In Foren oft komplette Stille.

**Das Gefuehl:** Die Plattform ist ordentlich, aber sie ist fuer eine enge Nische gebaut -- amerikanische Futures + englischsprachiges Publikum. Wenn Sie nicht in dieser Nische sind -- warum $1.500 bezahlen?

## Ehrliches Urteil: Lohnt es sich?

Ich habe eine Woche mit dem Testen von NinjaTrader verbracht. Mehrere Strategien gebaut, Backtests durchgefuehrt, versucht einen russischen Broker anzubinden, Foren gelesen.

**Mein Fazit:** Das ist keine Plattform fuer russische Trader.

**Wenn Sie nur MOEX handeln** -- schauen Sie nicht mal in Richtung NinjaTrader. Verbindung ueber Umwege, englischsprachiger Support, $1.500 fuer die Lizenz. Einfacher, ein kostenloses Tool zu nehmen, das russische Broker von Haus aus unterstuetzt.

**Wenn Sie amerikanische Futures handeln** -- NinjaTrader ergibt Sinn. Aber die Frage bleibt: Brauchen Sie einen visuellen Baukasten fuer $1.500? Oder ist es einfacher, einen Programmierer zu engagieren, der eine Strategie nach Ihren Beduerfnissen schreibt?

**Das Lustigste:** Strategy Builder generiert C#-Code. Das heisst, frueher oder spaeter kommen Sie sowieso beim Programmieren an. Die visuelle Oberflaeche ist nur eine Illusion der Einfachheit.

**Alternative:** Fuer die gleichen $1.500 koennen Sie:
- Einen Freelance-Programmierer engagieren
- Ein jaehrliches Datenabo kaufen
- Einen VPS fuer ein Jahr mieten
- Und es bleibt noch uebrig

$1.500 fuer eine schoene Oberflaeche und englischsprachigen Support bezahlen? Nicht ueberzeugend.

## Fallstricke (die ich gefunden habe)

**Ein-Klick-Ueberoptimierung.**

**Vendor Lock-in.**

Die Strategie lebt in NinjaTrader. Wollen Sie sie in ein anderes System verschieben -- von vorne umschreiben. Ja, Sie koennen nach NinjaScript (C#) exportieren, aber der Code ist spezifisch fuer ihre Architektur.

**Die Sprachbarriere ist ein echtes Problem.**

Ich lese Englisch. Aber als ich versuchte, benutzerdefinierte Indikatoren zu verstehen, verbrachte ich drei Stunden in der Dokumentation. Wenn Sie kein Englisch lesen -- multiplizieren Sie die Zeit mit drei.

Foren sind auch auf Englisch. Support antwortet auf Englisch. Codebeispiele haben englische Kommentare. Das ist keine Plattform fuer den russischen Markt, es ist ein amerikanisches Produkt fuer den amerikanischen Trader.

## Abschliessende Gedanken

Ich begann mit hohen Erwartungen. NinjaTrader positioniert sich als professionelles Werkzeug. Im Marketing ist alles schoen: visueller Baukasten, Tausende Nutzer, riesige Community.

**Was ich tatsaechlich bekam:**

- Einen "visuellen" Baukasten in Tabellenform (keine Flussdiagramme wie TSLab/Designer)
- Eine Plattform fuer $1.500, die den russischen Markt nicht unterstuetzt
- Englischsprachige Dokumentation und langsamen Support
- Die Notwendigkeit, C# zu lernen, wenn man mehr will als zwei Durchschnitte zu kreuzen

**Ehrlich:** Wenn Sie amerikanische Futures handeln, Englisch lesen und bereit sind zu bezahlen -- ist NinjaTrader eine gute Wahl. Die Plattform ist ausgereift, Bugs sind selten, die Funktionalitaet ist reich.

**Aber** wenn Sie ein russischer Trader sind, der MOEX handelt -- ist es rausgeworfenes Geld. Fuer die gleichen $1.500 koennen Sie einen vollstaendigen Algo-Trading-Stack aufbauen: Programmierer + Datenfeed + VPS. Mit Quellcode. Ohne Plattformbindung.

**Visuelle Baukästen sind eine Illusion.** Frueher oder spaeter kommen Sie sowieso zum Code. NinjaTrader generiert NinjaScript (C#), aber das ist nur ein verzoegerter Uebergang zum Programmieren. Die einzige Frage ist, wie viel Sie bereit sind, fuer diesen Aufschub zu bezahlen.

Ich habe die Lizenz nicht gekauft. Stattdessen schrieb ich am Wochenende eine Strategie in Python. Kostenlos. Mit voller Kontrolle. Ohne Vendor Lock-in.

---

**Nuetzliche Links:**

- [NinjaTrader offizielle Seite](https://ninjatrader.com/)
- [Strategy Builder Dokumentation](https://ninjatrader.com/support/helpguides/nt8/strategy_builder.htm)
