---
layout: post
title: "Warum 95 % der Retail-Algo-Trader Geld verlieren: Eine datenbasierte Analyse"
description: "Wir analysieren reale Verluststatistiken an der MOEX, versteckte Kosten des Algo-Tradings und warum Backtests lügen. Basierend auf Habr-Daten und Börsenstatistiken."
date: 2026-03-18
image: /assets/images/blog/ritejl-algotrejding-poteri.png
tags: [algo trading, MOEX, statistics, risks]
lang: de
---

## Die harte Wahrheit

Die Zahl "95 % der Trader verlieren Geld" ist längst zum Meme geworden, doch dahinter stecken reale Daten. Schauen wir uns an, warum algorithmischer Handel -- insbesondere für Privatanleger -- ein extrem schwieriges Unterfangen bleibt.

## MOEX-Statistiken

Laut Daten der Moskauer Börse und Analysen von [Habr](https://habr.com/):

- **76 % der aktiven Trader** auf der MOEX sind über ein Jahr hinweg unprofitabel
- Unter denjenigen, die algorithmischen Handel nutzen, sind etwa **~70 %** unprofitabel -- etwas besser, aber kein radikaler Unterschied
- Durchschnittlicher Verlust eines Retail-Algo-Traders: **-12 % jährlich** (nach Gebühren)
- Nur **3-5 %** erzielen über einen Zeitraum von 3+ Jahren beständig Gewinne

## Versteckte Kosten, die Strategien zerstören

### 1. Börsen- und Brokergebühren

Typische Gebühren für Privatanleger an der MOEX:

```
Börsengebühr (Aktienmarkt):
- Maker: 0,01 % des Handelsvolumens
- Taker: 0,015 % des Handelsvolumens

Brokergebühr:
- 0,03 % bis 0,06 % (je nach Broker und Tarif)

Gesamt für Roundtrip (Eröffnung + Schließung):
- Minimum: 0,08 % des Volumens
- Typisch: 0,12-0,15 % des Volumens
```

Bei 10 Trades pro Tag und einer durchschnittlichen Positionsgröße von 100.000 Rubel:

```
10 Trades × 0,12 % × 100.000 = 1.200 RUB/Tag
× 250 Handelstage = 300.000 RUB/Jahr
```

Das sind **300.000 Rubel pro Jahr** allein an Gebühren. Bei einem Depot von 1.000.000 Rubel sind das 30 % jährlich, die Sie erst verdienen müssen, um auf Null zu kommen.

### 2. Slippage

Slippage ist die Differenz zwischen dem Preis, zu dem die Strategie einsteigen "wollte", und dem Preis, zu dem die Order tatsächlich ausgeführt wurde:

- Bei liquiden Instrumenten (Sberbank, Gazprom): **0,01-0,05 %**
- Bei weniger liquiden: **0,1-0,5 %**
- Bei Nachrichtenereignissen: **1-5 %+**

### 3. Market Impact

Wenn Ihre Order im Verhältnis zum Orderbuch signifikant ist, bewegen Sie den Preis gegen sich selbst. Für Privatanleger bei liquiden Instrumenten selten, aber bei wenig gehandelten Wertpapieren ein ernstes Problem.

## Warum Backtests lügen

### Look-ahead Bias

Der häufigste Fehler: Nutzung von Daten, die zum Zeitpunkt der Entscheidung noch nicht verfügbar waren. Beispiele:

- Nutzung des Tagesschlusskurses für eine Entscheidung **am selben Tag**
- Nutzung adjustierter Daten, die rückwirkend verändert wurden

### Survivorship Bias

Ein Backtest auf S&P-500-Aktien berücksichtigt nur Unternehmen, **die überlebt haben**. Unternehmen, die in Konkurs gingen oder übernommen wurden, sind nicht in der Stichprobe enthalten, was die Illusion höherer Renditen erzeugt.

### Overfitting

Der heimtückischste Feind:

```
Je mehr Parameter eine Strategie hat,
desto besser funktioniert sie auf historischen Daten
und desto schlechter im realen Markt.
```

Wenn Ihre Strategie 10+ Parameter hat und im Backtest 200 % Jahresrendite zeigt, ist sie höchstwahrscheinlich überangepasst.

### Regime Change

Märkte verändern sich. Eine Strategie, die 2020-2023 funktioniert hat, kann 2024-2026 völlig versagen. Beispiele:

- Volatilitätsstrategien, die vor COVID entwickelt wurden, brachen in der Pandemie zusammen
- Momentum-Strategien, die auf einen Bullenmarkt abgestimmt sind, verlieren in Seitwärtsmärkten
- Arbitrage-Strategien "schließen sich", je mehr Teilnehmer sie kopieren

## Die realen Kosten des Algo-Tradings

Über die Handelskosten hinaus:

| Ausgabenposten | Jährliche Kosten |
|----------------|------------------|
| Server (VPS/Colocation) | 30.000 - 300.000 RUB |
| Daten (historisch + Echtzeit) | 10.000 - 100.000 RUB |
| Software (Plattform, Tools) | 0 - 50.000 RUB |
| Ihre eigene Zeit | unbezahlbar |

## Was tun, wenn Sie es trotzdem versuchen wollen

1. **Fangen Sie klein an** -- mit einem Depot, dessen Verlust Sie verkraften können
2. **Berücksichtigen Sie ALLE Kosten** im Backtest -- Gebühren, Slippage, Latenz
3. **Testen Sie auf Out-of-Sample-Daten** -- teilen Sie die Historie in Trainings- und Testdatensätze
4. **Begrenzen Sie die Parameteranzahl** -- je einfacher die Strategie, desto besser
5. **Nutzen Sie Walk-Forward-Analyse** -- überprüfen Sie Parameter regelmäßig
6. **Beginnen Sie mit Paper-Trading** -- testen Sie die Strategie in Echtzeit ohne Geld
7. **Diversifizieren Sie** -- setzen Sie nicht alles auf eine einzige Strategie

Algo-Trading ist kein "Geldknopf". Es ist ernsthafte ingenieurtechnische und analytische Arbeit, die Disziplin, Kapital und Ehrlichkeit gegenüber sich selbst erfordert.
