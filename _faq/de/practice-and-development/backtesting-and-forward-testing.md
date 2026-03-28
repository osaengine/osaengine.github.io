---
lang: de
layout: faq_article
title: "Was sind Backtesting und Forward-Testing?"
section: practice
order: 3
---
lang: de

Backtesting und Forward-Testing sind zentrale Phasen beim Testen von Trading-Robotern, die es ermoeglichen, deren Wirksamkeit und Stabilitaet vor dem Einsatz mit echtem Geld zu ueberpruefen.

## Backtesting:

1. **Was ist das?**
   Backtesting ist das Testen einer Handelsstrategie auf historischen Daten, um ihr Verhalten in der Vergangenheit zu bewerten.

2. **Wie funktioniert es?**
   - Der Roboter wendet den Algorithmus auf historische Daten an, als wuerde er in Echtzeit arbeiten.
   - Wichtige Kennzahlen werden analysiert: Rentabilitaet, maximaler Drawdown, Risiko-Ertrags-Verhaeltnis.

3. **Werkzeuge fuer Backtesting:**
   - **[StockSharp Designer](https://stocksharp.com/):** Bietet eine benutzerfreundliche Oberflaeche fuer visuelles Backtesting und Ergebnisanalyse.
   - **[MetaTrader](https://www.metatrader4.com/):** Integrierter Strategietester.
   - **[QuantConnect](https://www.quantconnect.com/):** Unterstuetzt Tests auf grossen Datenmengen.

## Forward-Testing:

1. **Was ist das?**
   Forward-Testing ist das Testen einer Strategie auf realen Marktdaten in Echtzeit, jedoch ohne den Einsatz von echtem Kapital.

2. **Wie funktioniert es?**
   - Der Roboter arbeitet auf einem Demokonto oder im Testmodus.
   - Es wird geprueft, wie der Algorithmus auf aktuelle Marktbedingungen, Verzoegerungen, Spreads und andere Faktoren reagiert.

## Warum ist das wichtig?

- Backtesting hilft, Schwachstellen der Strategie auf Basis historischer Daten zu identifizieren.
- Forward-Testing zeigt, wie der Roboter unter realen Marktbedingungen ohne Verlustrisiko arbeitet.

## Tipps:

- Verwenden Sie qualitativ hochwertige historische Daten fuer das Backtesting.
- Fuehren Sie Forward-Testing mindestens 1-2 Wochen durch, um die Stabilitaet der Strategie zu bestaetigen.
- Vergleichen Sie die Ergebnisse beider Tests, um die Zuverlaessigkeit des Algorithmus zu bewerten.
