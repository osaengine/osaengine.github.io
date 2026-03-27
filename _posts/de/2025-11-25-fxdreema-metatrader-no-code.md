---
layout: post
title: "fxDreema: Wenn man MetaTrader hat, aber MQL nicht lernen will"
description: "MetaTrader ist installiert, Broker verbunden, aber MQL schreiben ist schmerzhaft. Ich habe fxDreema gefunden -- einen visuellen Bot-Baukasten fuer MT4/MT5. Was dabei herauskam und ob es $95 pro Jahr wert ist."
date: 2025-11-25
image: /assets/images/blog/fxdreema_builder.png
tags: [fxDreema, MetaTrader, MT4, MT5, no-code, forex, visual builder]
lang: de
---

Ich hatte ein Problem. MetaTrader ist installiert, Broker verbunden, Strategie im Kopf. Aber um sie umzusetzen, muss ich MQL4 oder MQL5 lernen. Und das ist eine Sprache mit eigenen Eigenheiten, halb-englischer Dokumentation und Foren, wo die Haelfte der Antworten "lies das Handbuch" lautet.

Ich googelte "MetaTrader ohne Programmierung" und stiess auf fxDreema. Ein visueller Bot-Baukasten. Bloecke ziehen -- Expert Advisor erhalten. Klingt einfach. Ich beschloss, es auszuprobieren.

## Wie es funktioniert (oder funktionieren sollte)

fxDreema ist eine Webanwendung. **Wichtiger Punkt:** Dies ist kein Produkt von MetaQuotes (den Entwicklern von MetaTrader). Es ist ein Drittanbieter-Tool von Enthusiasten. Ein kleines Team, das einen Baukasten fuer eine fremde Plattform baut.

Man geht auf die Website, registriert sich, beginnt eine Strategie zu bauen. Keine Installationen, keine IDEs. Alles im Browser.

**Die Idee:** Bloecke (Indikatoren, Bedingungen, Aktionen) nehmen, mit Pfeilen verbinden, wie ein Flussdiagramm. Das Programm generiert MQL-Code. Datei herunterladen, in MetaTrader ablegen -- Bot fertig.

In der Theorie schoen. In der Theorie.

Ich registrierte mich (kostenlos), oeffnete den Editor. Und tatsaechlich -- visuelle Bloecke, wie in Scratch oder Node-RED. Ziehen, verbinden. Es gibt eine Bibliothek fertiger Bloecke: Indikatoren, Preispruefungen, Orders.

Baute eine einfache Strategie: wenn RSI unter 30 -- kaufen, wenn ueber 70 -- verkaufen. Klassisch. Drueckte "Code generieren," lud die .mq4-Datei herunter. Legte sie in MetaTrader ab.

**Es startete.** Keine Fehler. Der Bot handelt.

Erste Reaktion: "Wow, das funktioniert tatsaechlich."

## Dann kamen die Nuancen

Einfache Strategien lassen sich leicht zusammenbauen. Durchschnittskreuzungen, RSI, MACD -- alles als fertige Bloecke vorhanden. 15-20 Minuten und der Bot ist fertig.

Aber ich wollte einen Trailing-Stop hinzufuegen. Und da entdeckte ich, dass die kostenlose Version ein Limit hat: **maximal 10 "Verbindungen"** zwischen Bloecken.

10 Verbindungen sind ungefaehr 5-6 Bloecke mit Bedingungen. Fuer einfache Strategien reicht es. Fuer Komplexeres -- stossen Sie ans Limit.

Okay, dachte ich, kaufe ich die Vollversion. Schaute mir die Preise an.

**$95 pro Jahr.** Oder $33 fuer 3 Monate.

Ich ueberlegte. $95 ist kein Vermoegen. Aber die Frage ist: Was bekomme ich fuer das Geld?

- Aufhebung des 10-Verbindungen-Limits
- MQL4-zu-MQL5-Konvertierung (und umgekehrt)
- Scheint alles zu sein

Kein Support. Keine Updates der Indikator-Bibliothek. Nur die Aufhebung einer kuenstlichen Einschraenkung.

## Versuch, etwas Komplexeres zu bauen

Ich entschied mich, nicht sofort zu kaufen, sondern zu sehen, was ich aus der kostenlosen Version herausholen kann. Strategie vereinfacht, ueberfluessige Pruefungen entfernt, in 10 Verbindungen gepasst.

Code generiert. In MetaTrader auf dem Demo-Konto ausgefuehrt.

**Problem Nummer eins:** Visuell sieht alles klar aus -- Bloecke, Pfeile. Aber wenn die Strategie Geld verliert, ist das Debugging in fxDreema schmerzhaft. Man muss den Browser oeffnen, das Diagramm anschauen, Bloecke aendern, Code neu generieren, in MetaTrader ablegen, neu starten.

In normalem Code (in MQL oder Python) oeffnet man die Datei, aendert ein paar Zeilen, speichert. Hier -- ein ganzer Zyklus.

**Problem Nummer zwei:** Der generierte MQL-Code sieht... seltsam aus. Variablen mit automatischen Namen, Logik ueber Funktionen verteilt, Kommentare auf Englisch (falls vorhanden). Schwer zu lesen. Noch schwerer, manuell zu modifizieren.

Das heisst, wenn fxDreema nicht bauen kann, was Sie brauchen -- stecken Sie fest. Der Code wird generiert, aber damit wie mit normalem Code zu arbeiten, funktioniert nicht.

## Vergleich mit dem, was ich bereits getestet habe

In den letzten Wochen habe ich verschiedene visuelle Baukästen ausprobiert. Hier ist, was sich ergibt:

**TSLab/StockSharp Designer** -- Flussdiagramme, Logik sichtbar, nach C# exportierbar. Funktioniert mit russischen Brokern.

**NinjaTrader** -- Tabellenoberflaeche (keine Bloecke), fuer amerikanische Futures gebaut. $1.500 fuer die Lizenz.

**fxDreema** -- Flussdiagramme wie Designer, aber nur fuer MetaTrader. $95 pro Jahr. Und die kostenlose Version hat ein striktes Komplexitaetslimit.

fxDreema hat einen Vorteil: Es funktioniert im Browser. Nichts zu installieren. Besuchen, bauen, herunterladen, starten.

Aber das ist auch ein Nachteil. Alles ist online. Wenn die Seite ausfaellt -- haben Sie kein Werkzeug.

**Und hier wird es interessant:** fxDreema ist kein offizielles MetaQuotes-Produkt. Es ist ein Drittanbieter-Service, der Code fuer eine fremde Plattform generiert. Kleines Team, das Projekt lebt von Benutzer-Abonnements.

Was passiert, wenn MetaQuotes morgen etwas an MQL aendert und der Code nicht mehr kompiliert? Oder wenn die fxDreema-Entwickler das Projekt schliessen? Ihre Diagramme bleiben auf deren Servern. Der generierte Code ist auch an deren Architektur gebunden.

Bei offiziellen Plattformen (TSLab, NinjaTrader) ist wenigstens klar, dass sie naechstes Jahr nicht dichtmachen. Hier -- gibt es ein Risiko.

## Fuer wen ist das wirklich

Ich habe mehrere Tage darueber nachgedacht. Hier ist mein Fazit.

fxDreema ist geeignet, wenn:

- Sie bereits MetaTrader (MT4 oder MT5) und einen Broker haben
- Sie Forex oder CFDs ueber MetaTrader handeln
- Sie eine einfache Indikatorstrategie brauchen (Kreuzungen, Levels, RSI/MACD)
- Sie MQL nicht lernen wollen
- Sie bereit sind, ~$95/Jahr fuer Bequemlichkeit zu zahlen

fxDreema ist NICHT geeignet, wenn:

- Sie russische Maerkte handeln (MOEX, russische Futures)
- Sie komplexe Logik mit vielen Bedingungen brauchen
- Sie eine kostenlose Loesung wollen (das 10-Verbindungen-Limit ist sehr schnell erschoepft)
- Sie planen, Code manuell zu bearbeiten (generiertes MQL ist unleserlich)
- Sie Stabilitaet und Garantien wollen (es ist ein Drittanbieter-Service, kein offizielles Produkt)

## Was ich letztendlich gemacht habe

Ich kaufte kein Abo. Baute eine einfache Strategie in der kostenlosen Version, lud den Code herunter, legte ihn in MetaTrader ab. Funktioniert.

Aber fuer die naechste Strategie oeffnete ich einfach ein MQL5-Lehrbuch und schrieb den Code per Hand. Eine Stunde fuer die Grundsyntax, noch eine Stunde zum Schreiben -- und ich habe einen funktionierenden Expert Advisor. Ohne Limits. Ohne Abonnements. Mit voller Kontrolle.

**Das Paradox:** fxDreema wurde geschaffen, um die Notwendigkeit zu beseitigen, MQL zu lernen. Aber wenn man an die Grenzen des visuellen Baukastens stoesst, kommt man zum Schluss, dass es einfacher gewesen waere, die Sprache zu lernen.

$95 pro Jahr an einen Drittanbieter-Service zahlen, der jederzeit schliessen koennte, fuer ein Werkzeug, das ein paar Stunden Lernzeit spart? Jeder entscheidet selbst. Fuer mich hat es nicht gepasst.

## Ehrliches Fazit

fxDreema ist kein schlechtes Werkzeug. Es funktioniert tatsaechlich. Flussdiagramme lassen sich leicht zusammenbauen, Code wird generiert, Bots starten.

Aber es ist ein Werkzeug mit einem sehr engen Anwendungsbereich:

- Nur MetaTrader (MT4/MT5)
- Nur einfache Strategien (in der kostenlosen Version)
- Nur wenn Sie bereit sind, fuer die Aufhebung der Limits zu zahlen

Wenn Sie bereits ueber MetaTrader handeln, eine einfache Indikatorstrategie automatisieren wollen und sich nicht mit Programmierung befassen moechten -- probieren Sie die kostenlose Version. Vielleicht reichen 10 Verbindungen fuer Sie.

Aber wenn Sie planen, sich ernsthaft mit algorithmischem Handel zu beschaeftigen -- lernen Sie MQL oder wechseln Sie zu etwas Flexiblerem. Visuelle Baukästen stossen frueher oder spaeter an ihre Grenzen. Und dann muessen Sie sowieso programmieren.

Ich habe zwei Tage mit fxDreema verbracht. Drei Strategien gebaut, im Tester ausgefuehrt, Ergebnisse angeschaut. Am Ende bin ich zum Code zurueckgekehrt.

Vielleicht ist es einfach nicht meins. Oder vielleicht sind visuelle Baukästen immer ein Kompromiss zwischen Einfachheit und Kontrolle.

---

**Nuetzliche Links:**

- [fxDreema offizielle Seite](https://fxdreema.com/)
- [Dokumentation und Beispiele](https://fxdreema.com/forum/)
