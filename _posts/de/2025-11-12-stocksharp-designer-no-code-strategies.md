---
layout: post
title: "StockSharp Designer: Kostenlose Trading-Bots ohne Code"
description: "StockSharp Designer ist ein visueller Trading-Bot-Baukasten. Voellig kostenlos, Open-Source, funktioniert mit jedem Broker. Klingt zu gut? Schauen wir, ob es einen Haken gibt."
date: 2025-11-12
image: /assets/images/blog/stocksharp_designer.png
tags: [StockSharp, Designer, no-code, open-source, algorithmic trading]
lang: de
---

StockSharp Designer -- das ist, wenn Sie einen Trading-Bot aus visuellen Bloecken per Mausklick zusammenbauen, voellig kostenlos, und den Quellcode der gesamten Plattform auf GitHub haben. Klingt wie ein Scherz? Nein, es ist ein echtes Produkt, und jetzt klaeren wir, warum es kostenlos ist und ob es einen Haken gibt.

## Was es ist

Designer ist ein visueller Strategiebaukasten von StockSharp. Sie setzen einen Trading-Bot buchstaeblich aus fertigen Bloecken zusammen: Indikator hineinziehen, mit einer Bedingung verbinden, Kaufsignal hinzufuegen -- fertig. Kein Code, kein if-else, keine Arrays.

**Das Hauptmerkmal:** Es ist voellig kostenlos und Open-Source.

Es gibt keine Bezahlversion. Keine 30-Tage-Testversion. Kein "Kaufen Sie die Vollversion fuer 600 Dollar pro Jahr." Einfach herunterladen, installieren, benutzen.

**Die natuerliche Frage:** Wenn es kostenlos ist, wo ist der Haken?

Der Haken ist, dass StockSharp mit Designer kein Geld verdient. Sie verkaufen Enterprise-Lizenzen an Unternehmen, Beratung und individuelle Entwicklung. Designer ist das Schaufenster ihres Frameworks. Wenn es Ihnen gefaellt, moechten Sie sie vielleicht spaeter fuer ein ernstes Projekt engagieren. Einfaches Geschaeftsmodell.

## Wie es funktioniert

Die Logik ist einfach:

Wollen Sie einen Bot auf Basis von gleitenden Durchschnitt-Kreuzungen? Nehmen Sie einen "Preis"-Block, zwei "SMA"-Bloecke mit verschiedenen Perioden, einen "Kreuzung"-Block, einen "Kaufen"-Block. Verbinden Sie sie mit Linien. Starten Sie einen Backtest. Sehen Sie die Ergebnisse.

Das alles in 20-30 Minuten ohne eine einzige Codezeile.

**Beispiel:**
```
Preis -> SMA(20) \
                   -> Kreuzung nach oben -> Kaufen
Preis -> SMA(50) /
```

Visuell sieht es aus wie ein Algorithmus-Flussdiagramm aus dem Informatik-Lehrbuch, nur dass statt "Anfang-Ende" Indikatoren und Handelssignale stehen.

![StockSharp Designer Oberflaeche]({{site.baseurl}}/assets/images/blog/designer_interface.png)

## Was es kann

**Ab Werk:**
- Jede Menge Indikatoren (SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic und etwa 60 weitere)
- Logikbloecke (UND, ODER, NICHT, Vergleiche, Bedingungen)
- Handelsaktionen (Kauf, Verkauf, Stop-Loss, Trailing-Stops)
- Backtesting mit historischen Daten
- Parameteroptimierung (Beste Werte finden)
- Broker-Verbindungen (nationale und internationale)

**Broker:**
- Russische: QUIK, Transaq (Finam), ALOR API, Tinkoff Invest, BCS
- Internationale: Interactive Brokers, Binance, BitMEX, Bybit
- Jeder Broker mit FIX API oder REST API (eigenen Konnektor schreiben moeglich)

**Daten:**
- CSV-Dateien (historische Kurse importieren)
- Finam Export (kostenlose Daten von Finam)
- Direkte Broker-Verbindung (Echtzeit-Kurse)

## Der zentrale Unterschied zu anderen Baukästen

Hier setzt sich Designer deutlich von Wettbewerbern wie TSLab ab.

**Die Strategie ist nicht an Designer gebunden.**

Sie bauen eine Strategie im visuellen Baukasten, exportieren sie in C#-Code, und koennen sie dann **ueberall** ausfuehren -- ohne Designer selbst, ohne GUI, ohne Windows.

So funktioniert es:

1. Strategie in Designer bauen (visuell, ohne Code)
2. Nach C# exportieren (ein Klick)
3. Eine Konsolenanwendung auf StockSharp API erhalten
4. Auf einem Linux-Server, in einem Docker-Container, auf einem VPS ausfuehren

**Die Konkurrenz kann das nicht.** TSLab ist fest an seine GUI gebunden. Die Strategie lebt nur innerhalb von TSLab und kann nur ueber die Programmoberflaeche ausgefuehrt werden.

Designer nutzt StockSharp API als Basis. Der visuelle Baukasten ist lediglich ein praktischer Wrapper fuer die Codegenerierung. Aber der resultierende Code ist gewoehnliches C#, das unabhaengig funktioniert.

**Praktische Bedeutung:**

- Strategie auf einem Server ohne GUI ausfuehren (Headless-Modus)
- Autostart ueber systemd (Linux) oder Aufgabenplanung (Windows) einrichten
- Ueberwachung ueber API oder Logs, ohne Designer geoeffnet zu halten
- Deployment in Docker fuer Isolation und Skalierung

Es ist wie LEAN von QuantConnect -- ein professioneller Ansatz. Entwicklung ueber GUI, Produktion ueber Konsole.

**Fuer den Heimtrader** ist diese Funktion uebertrieben. Aber wenn Sie eine ernsthafte Infrastruktur planen -- ist es ein entscheidender Vorteil.

## Praxiserfahrung

**Was schnell gelingt:**

Klassische Indikatorstrategien. SMA-Kreuzung, Bollinger-Bands-Abprall, RSI-Ueberkauft -- all das laesst sich in 15-20 Minuten zusammenbauen.

Backtesting funktioniert einfach: Daten laden, Test starten, Ergebnisse erhalten. Win Rate, Profit Factor, Drawdown, Equity-Chart -- alles auf dem Bildschirm.

Parameteroptimierung: Ein Klick -- Designer iteriert alle Kombinationen und zeigt die besten. Gefaehrlich, weil man leicht auf historische Daten ueberoptimiert.

**Wo Probleme beginnen:**

Wenn die Strategie komplex wird. Bei 5-7 Bedingungen -- kein Problem. Bei 20-30 -- wird das Diagramm zum Spaghetti. Linien zwischen Bloecken verheddern sich, die Logik ist schwer zu verstehen.

**Loesung:** Man kann eigene Bloecke in C# schreiben. Aber wenn man C# schreibt -- wozu braucht man den visuellen Baukasten?

**Weiteres Problem:** Die Dokumentation ist bescheiden. Sie existiert, aber ist nicht so ausfuehrlich wie gewuenscht. Man muss Dinge durch Versuch und Irrtum herausfinden.

Es gibt eine Community (Forum, Telegram), aber sie ist nicht riesig. Fragen werden beantwortet, aber nicht immer schnell.

## Fallstricke

**Ueberoptimierung ist die groesste Gefahr.**

Designer macht die Optimierung zu einfach. Sie legen einen Parameterbereich fest (z.B. SMA-Periode von 10 bis 50), druecken einen Knopf, und das Programm findet "ideale" Werte.

Auf historischen Daten zeigt die Strategie +40% pro Jahr. Sie starten sie freudig live, und sie vernichtet das Depot in einem Monat.

Warum? Weil die "idealen" Parameter einfach perfekt auf einen bestimmten historischen Zeitraum zugeschnitten sind. Das ist kein Muster -- das ist ein Artefakt.

**Wie Sie sich schuetzen:** Walk-Forward-Testing. Auf einem Zeitraum optimieren (In-Sample), auf einem anderen verifizieren (Out-of-Sample). Wenn die Ergebnisse stark abweichen -- Strategie verwerfen.

**Zweites Problem:** Portabilitaet auf andere Plattformen.

Wenn Sie die Strategie zu Backtrader, LEAN oder MetaTrader migrieren wollen -- muessen Sie umschreiben.

Aber im Gegensatz zu TSLab exportiert Designer die Strategie in C#-Code auf StockSharp API. Sie koennen sie ueberall ohne Designer ausfuehren -- auf dem Server, in Docker, auf Linux. Der Code ist nicht der schoenste, aber er ist unabhaengig.

**Drittes Problem:** Grenzen des visuellen Ansatzes.

Visuelle Bloecke eignen sich fuer einfache Logik. Aber sobald etwas Nicht-Standardmaessiges benoetigt wird (Spread-Trading, Arbitrage, News-Parsing, Machine Learning) -- werden visuelle Diagramme unhandlich.

Es entsteht ein Paradox: Fuer einfache Aufgaben ist Designer ueberdimensioniert (einfacher, 10 Zeilen Code zu schreiben), fuer komplexe -- nicht flexibel genug.

![Strategiebeispiel in Designer]({{site.baseurl}}/assets/images/blog/designer_strategy.png)

## Fuer wen Designer geeignet ist

**Definitiv geeignet fuer:**
- Trader, die wissen was funktioniert, aber nicht programmieren koennen
- Analysten, die schnell Hypothesen testen wollen
- Diejenigen, die an internationalen Boersen handeln (Binance, IB)
- Open-Source-Enthusiasten
- Diejenigen, die nicht fuer einen visuellen Baukasten bezahlen wollen

**Eher nicht geeignet fuer:**
- Programmierer (schneller, Code in Python zu schreiben)
- Diejenigen, die komplexe Multi-Instrument-Strategien planen
- Hochfrequenzhaendler (HFT)
- Diejenigen, die Machine Learning wollen (besser gleich Python + sklearn)

## Warum kostenlos und was ist mit Open-Source

Der gesamte StockSharp-Code liegt auf GitHub. Sie koennen nachsehen, wie jeder Indikator funktioniert, wie der Backtester implementiert ist, wie der Broker-Konnektor aufgebaut ist.

Wollen Sie eine eigene Funktion hinzufuegen? Repository forken, Code schreiben, Pull Request erstellen. Ihre Funktion koennte in den Hauptzweig aufgenommen werden.

**Open-Source-Vorteile:**
- Transparenz (Sie sehen, was im Inneren passiert)
- Sicherheit (Sie koennen pruefen, ob die Plattform Ihre API-Schluessel stiehlt)
- Erweiterbarkeit (Sie koennen alles hinzufuegen)
- Unabhaengigkeit (Strategie in Code exportieren und ohne Designer ausfuehren)

**Open-Source-Nachteile:**
- Niemand garantiert Support
- Wenn Sie einen Bug finden -- er koennte in einem Tag behoben werden, oder in einem Monat
- Dokumentation ist nicht immer aktuell

Aber fuer kostenlos -- ist das zu verkraften.

## Ehrliche Antwort: Lohnt es sich?

**Ja, wenn:**
- Sie nicht programmieren lernen wollen
- Sie schnell eine einfache Idee testen muessen
- Sie auf russischen oder internationalen Maerkten handeln
- Ihnen die Idee von kostenlosem Open-Source gefaellt
- Sie bereit sind, Dinge selbst herauszufinden (Dokumentation ist nicht perfekt)

**Nein, wenn:**
- Sie Python/C# koennen oder lernen wollen (dann einfach Code schreiben)
- Sie komplexe Logik brauchen (visuelle Diagramme skalieren nicht)
- Sie Hochfrequenzhandel wollen (visuelle Bloecke sind zu langsam)

## Alternativen

Wenn Designer nichts fuer Sie war, gibt es Optionen:

**Kostenpflichtige visuelle Baukästen:**
- TSLab (~600$/Jahr oder ~50$/Monat) -- ein russisches Pendant zu Designer, ausgereifter
- NinjaTrader Strategy Builder -- fuer internationale Maerkte
- fxDreema -- fuer MetaTrader 5

**Kostenlose Loesungen mit Code:**
- Backtrader (Python) -- erfordert Code-Schreiben, aber flexibler
- LEAN (C#/Python) -- professionelles Niveau, komplexer

**Broker-Plattformen:**
- QUIK (wenn Ihr Broker es unterstuetzt, mit Lua-Scripting)
- MetaTrader 5 (MQL5 fuer Strategien)

## Fazit

StockSharp Designer ist eine kostenlose Moeglichkeit, algorithmischen Handel ohne Programmierung auszuprobieren. Fuer einfache Indikatorstrategien funktioniert es gut. Fuer komplexe -- stossen Sie an die Grenzen des visuellen Ansatzes.

**Hauptvorteil:** Kostenlos und Open-Source. Man muss nicht Hunderte Dollar pro Jahr fuer eine Lizenz bezahlen.

**Hauptnachteil:** Dokumentation und Support sind nicht auf dem Niveau kommerzieller Produkte. Man muss Dinge selbst herausfinden.

**Abschliessender Gedanke:**

Visuelle Baukästen sind Kruecken. Bequeme Kruecken fuer diejenigen, die nicht programmieren lernen wollen. Aber wenn Sie es ernst meinen mit algorithmischem Handel, muessen Sie frueher oder spaeter Python oder C# lernen.

Designer (wie jeder visuelle Baukasten) ist grossartig fuer den **Einstieg**. Testen Sie ein paar Ideen, verstehen Sie die Backtesting-Logik, machen Sie sich mit Indikatoren vertraut. Danach -- entweder zum Code migrieren oder die Grenzen des visuellen Ansatzes akzeptieren.

Aber fuer die erste Begegnung mit algorithmischem Handel -- warum nicht. Besonders wenn es kostenlos ist.

---

**Nuetzliche Links:**

- [StockSharp (Hauptseite)](https://stocksharp.ru/store/%D0%B4%D0%B8%D0%B7%D0%B0%D0%B9%D0%BD%D0%B5%D1%80-%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D0%B9/)
- [StockSharp Designer](https://algodes.com/de/)
- [GitHub-Repository](https://github.com/StockSharp/StockSharp)
- [Dokumentation](https://doc.stocksharp.ru/)
- [StockSharp Forum](https://stocksharp.ru/forum/)
- [Telegram-Chat](https://t.me/stocksharp)

**Andere Artikel:**

- [TSLab: Trading-Bots ohne Code fuer 600 Dollar pro Jahr](/de/blog/tslab-no-code-strategies/) -- eine kostenpflichtige Alternative zu Designer

**Was kommt als Naechstes:** In den folgenden Artikeln werden wir weitere visuelle Baukästen (NinjaTrader, fxDreema) untersuchen und sie alle in einer Tabelle vergleichen.
