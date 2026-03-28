---
lang: de
layout: faq_article
title: "Welche Daten werden fuer den Betrieb eines Trading-Roboters benoetigt?"
section: technical
order: 2
---
lang: de

Fuer den Betrieb eines Trading-Roboters ist der Zugang zu verschiedenen Datentypen erforderlich. Diese Daten gewaehrleisten eine korrekte Entscheidungsfindung und die Genauigkeit des Algorithmus.

## Datentypen:

1. **Marktdaten:**
   - Aktuelle Kurse (Notierungen) und Handelsvolumina.
   - Orderbuecher (Markttiefe) zur Liquiditaetsanalyse.

2. **Historische Daten:**
   - Werden fuer das Testen von Strategien (Backtesting) verwendet.
   - Umfassen Kurse, Volumina und Marktereignisse vergangener Zeitraeume.

3. **Fundamentaldaten:**
   - Finanzberichte von Unternehmen, Nachrichten, makrooekonomische Statistiken.
   - Wichtig fuer langfristige Strategien.

4. **Ereignisdaten:**
   - Informationen ueber Unternehmensereignisse wie Dividenden, Fusionen und Uebernahmen.

## Wo bekommt man die Daten?

- **[AlphaVantage](https://alphavantage.co/):** Historische Daten fuer verschiedene Boersen.
- **[Yahoo Finance](https://finance.yahoo.com/):** Kostenlose Daten zur Analyse von Aktien und Indizes.
- **[Quandl](https://www.quandl.com/):** Fundamental- und Marktdaten fuer die Analytik.
- **[Interactive Brokers](https://www.interactivebrokers.com/):** API fuer den Zugriff auf Echtzeit-Marktdaten.

## Tipps:

- Ueberpruefen Sie die Datenqualitaet vor der Verwendung.
- Fuer Hochfrequenzhandel waehlen Sie Quellen mit minimaler Latenz.
- Archivieren Sie Daten fuer Analyse und Ergebnisvergleich.
