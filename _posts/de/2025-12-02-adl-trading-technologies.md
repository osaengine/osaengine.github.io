---
layout: post
title: "ADL von Trading Technologies: Wenn ein Strategie-Baukasten so viel kostet wie eine Wohnung"
description: "Ich wollte ADL (Algo Design Lab) von Trading Technologies testen — einen visuellen Baukasten fuer Profis. Aber ich stiess auf ein Problem: Es ist eine Enterprise-Loesung mit Preisen 'auf Anfrage'. Hier ist, was ich herausgefunden habe."
date: 2025-12-02
image: /assets/images/blog/adl_trading_tech.png
tags: [ADL, Trading Technologies, no-code, institutional trading, futures]
lang: de
---

In den letzten Wochen habe ich visuelle Strategie-Baukaesten getestet. TSLab fuer 60.000 Rubel pro Jahr, StockSharp Designer kostenlos, NinjaTrader fuer 150.000, fxDreema fuer 10.000. Es war logisch, ADL (Algo Design Lab) von Trading Technologies auf die Liste zu setzen — ein Baukasten fuer "Profis".

Ich ging auf die Website von Trading Technologies. Fand die ADL-Seite. Schoene Screenshots, Marketing ueber Drag-and-Drop, visuelle Programmierung, Integration mit einer institutionellen Plattform.

**Frage:** Wo ist der "Download"-Button oder wenigstens "Ausprobieren"?

Antwort: Es gibt keinen.

## Wie man Zugang zu ADL bekommt (Spoiler: gar nicht)

ADL ist kein eigenstaendiges Produkt. Es ist ein Modul innerhalb der TT Platform Pro von Trading Technologies. Um ADL zu bekommen, braucht man zuerst Zugang zur TT Platform.

Ich begann zu recherchieren, wie das geht.

**Option 1:** Sich auf der TT-Website registrieren und die Plattform herunterladen.

Versucht. Auf der Website gibt es ein "Contact Us"-Formular. Ausgefuellt. Angegeben, dass ich ADL fuer eine Rezension testen moechte. Einen Tag spaeter kam die Antwort: "Vielen Dank fuer Ihr Interesse. Wir werden uns mit Ihnen in Verbindung setzen, um Ihre Handelsbeduerfnisse zu besprechen."

Zwei Wochen vergingen. Niemand meldete sich.

**Option 2:** Einen Broker finden, der Zugang zur TT Platform bietet.

Gegoogelt. AMP Futures, Optimus Futures, Discount Trading — mehrere amerikanische Broker bieten TT Platform an. Aber ueberall dasselbe: "Preise auf Anfrage", "Abhaengig vom Handelsvolumen", "Kontaktieren Sie uns fuer ein individuelles Angebot".

Ich kontaktierte einen der Broker. Fragte nach Zugang zu TT Platform + ADL.

Antwort: "Das Mindestabonnement fuer TT Platform Pro beginnt bei $1.500 pro Monat. Plus Handelskommissionen. Plus Marktdatengebuehren. ADL ist kostenlos enthalten, wenn Sie TT Platform Pro haben."

$1.500 pro Monat. **$18.000 pro Jahr.** In Rubel sind das etwa **1,8 Millionen**.

Fuer einen visuellen Strategie-Baukasten.

## Was ich ohne Zugang herausfinden konnte

Da ich ADL nicht wirklich testen konnte, musste ich Informationen stueckweise zusammentragen: TT-Dokumentation, YouTube-Videos, Foren, Trader-Bewertungen.

**Was ADL ist:**

Ein visueller Algorithmen-Baukasten, eingebettet in die TT Platform. Drag-and-Drop-Oberflaeche, Bloecke fuer Bedingungen und Aktionen, Backtesting auf historischen Daten. Konzeptionell aehnlich wie TSLab oder StockSharp Designer.

**Hauptunterschied:** ADL lebt innerhalb einer professionellen Handelsplattform. Die TT Platform wird von Hedgefonds, Prop-Tradern und institutionellen Akteuren genutzt. Das ist kein Retail-Produkt.

**Was es kann (laut Dokumentation):**

- Visueller Aufbau von Algorithmen durch Bloecke
- Backtesting auf historischen Daten
- Echtzeit-Marktsimulation
- Integration mit Order Management System (OMS)
- Direkte Algorithmenausfuehrung im Orderbuch
- Echtzeit-Performance-Monitoring

**Was es NICHT kann:**

- Ausserhalb der TT Platform arbeiten (kein Code-Export)
- Kostenlos oder zumindest guenstig funktionieren
- Fuer gewoehnliche Retail-Trader zugaenglich sein

## Fuer wen ist das ueberhaupt?

Ich habe mehrere Tage darueber nachgedacht. Und hier ist mein Fazit.

ADL ist nicht fuer Retail-Trader. Es ist nicht einmal fuer aktive Privatanleger. Es ist fuer institutionelle Akteure:

- Prop-Trading-Firmen
- Hedgefonds
- Market Maker
- Grosse Vermoegensverwaltungen

Menschen, die Millionen Dollar pro Tag handeln. Fuer sie sind $1.500 pro Monat fuer eine Plattform Kleingeld im Vergleich zu ihren Volumina.

**Das Paradox:** ADL wird als "Baukasten positioniert, mit dem jeder Algorithmen erstellen kann". Aber um Zugang zu bekommen, muss man wie ein institutioneller Akteur zahlen.

## Vergleich mit dem, was ich getestet habe

In den letzten Wochen habe ich tatsaechlich mit vier visuellen Baukaesten gearbeitet:

**TSLab** — 60.000 Rubel pro Jahr. Flussdiagramme, russischer Markt, russische Sprache. Funktioniert, aber teuer fuer das, was es bietet.

**StockSharp Designer** — kostenlos. Open-Source, Flussdiagramme, Code-Export. Russische und internationale Maerkte. Weniger ausgereift, aber funktional nah an TSLab.

**NinjaTrader Strategy Builder** — 150.000 Rubel lebenslang oder 120.000 pro Jahr. Tabelleninterface (keine Bloecke), nur internationale Maerkte. Ausgereiftes Produkt, aber fuer eine enge Nische.

**fxDreema** — 10.000 Rubel pro Jahr. Flussdiagramme im Browser, nur MetaTrader. Ein Nebenprojekt von Enthusiasten. Funktioniert, aber es besteht das Risiko, dass es eingestellt wird.

**ADL** — 1,8 Millionen Rubel pro Jahr (Minimum). Visueller Baukasten innerhalb einer professionellen Plattform. Konnte es nicht testen, aber laut Bewertungen ein solides Tool fuer diejenigen, die es wirklich brauchen.

Der Preisunterschied — 30-fach im Vergleich zu TSLab und 180-fach im Vergleich zu fxDreema.

## Ehrliches Fazit: Ich konnte es nicht testen

Normalerweise schreibe ich in meinen Artikeln ueber echte Erfahrungen. Installiert, ausprobiert, auf Probleme gestossen, Schlussfolgerungen gezogen.

Bei ADL war das nicht moeglich.

**Der Grund ist einfach:** Es ist eine Enterprise-Loesung. Es gibt keine Demo-Version. Keine Testphase. Nicht einmal oeffentliche Preise. Alles laeuft ueber "kontaktieren Sie uns", "individuelles Angebot", "abhaengig vom Volumen".

Ich haette einen Artikel auf Basis der Marketing-Materialien von TT schreiben koennen. Aber das waere nicht mein Artikel gewesen, sondern eine Nacherzaehlung fremder Werbung.

Stattdessen entschied ich mich, ehrlich zu schreiben: **ADL sieht nach einem maaechtigen Tool aus, aber es ist nicht fuer gewoehnliche Trader.**

Wenn Sie Millionen Dollar ueber amerikanische Futures handeln, bei einer Prop-Firma oder einem Hedgefonds arbeiten und einen visuellen Baukasten mit institutioneller Infrastruktur benoetigen — koennte ADL eine gute Wahl sein.

Aber wenn Sie ein Privatanleger sind, der einen Roboter fuer die Moskauer Boerse bauen oder einfach algorithmischen Handel ausprobieren moechte — vergessen Sie ADL. Zu teuer. Zu schwer, Zugang zu bekommen. Zu stark auf das institutionelle Niveau ausgerichtet.

## Was ich stattdessen getan habe

Ohne Zugang zu ADL kehrte ich zu dem zurueck, was ich bereits getestet hatte:

- **StockSharp Designer** — kostenlos, funktioniert mit russischen Brokern, Open-Source
- **fxDreema** — 10.000 pro Jahr, wenn Sie ueber MetaTrader handeln
- **TSLab** — 60.000 pro Jahr, wenn Sie eine fertige Loesung mit Support wollen

Alle drei bieten visuelle Programmierung. Alle drei sind tatsaechlich zugaenglich. Alle drei koennen in 20 Minuten getestet werden.

**Mein Fazit:** Fuer 99% der Trader ist ADL ein huebsches Bild auf der Website von Trading Technologies. Unzugaenglich, teuer, institutionell.

Vielleicht habe ich eines Tages Zugang zur TT Platform. Dann schreibe ich eine vollstaendige ADL-Rezension mit echten Tests und Screenshots.

Vorerst ist das die Geschichte einer Plattform, die ich nicht testen konnte. Die aber perfekt den Unterschied zwischen Retail- und institutionellem algorithmischen Handel zeigt.

Institutionelle Akteure zahlen Millionen fuer Infrastruktur. Retail-Trader bauen Roboter aus kostenlosen Open-Source-Bibliotheken.

Zwei verschiedene Welten. ADL kommt aus der Welt, in der die Mindestgebuehr fuer eine Plattform so viel kostet wie ein gutes Auto.

---

**Nuetzliche Links:**

- [ADL offizielle Seite](https://tradingtechnologies.com/trading/algo-trading/adl/)
- [TT Platform Dokumentation](https://library.tradingtechnologies.com/)
