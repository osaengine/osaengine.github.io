---
lang: de
layout: faq_article
title: "Kann man einen Roboter auf mehreren Boersen oder Maerkten gleichzeitig betreiben?"
section: advanced
order: 1
---
lang: de

Ein Trading-Roboter kann fuer die gleichzeitige Arbeit auf mehreren Boersen oder Maerkten konfiguriert werden. Dies ermoeglicht die Diversifizierung von Risiken, die Nutzung von Arbitragemoeglichkeiten und die Steigerung des potenziellen Gewinns.

## Wie setzt man das um:

1. **Unterstuetzung mehrerer APIs:**
   - Der Roboter muss ueber deren APIs mit den Boersen verbunden sein. Die meisten Plattformen wie **[StockSharp](https://stocksharp.ru/)** oder **[QuantConnect](https://www.quantconnect.com/)** unterstuetzen die Anbindung an mehrere Maerkte.

2. **Datenverwaltung:**
   - Jeder Markt liefert seine eigenen Daten (Kurse, Orderbuecher), die der Roboter korrekt verarbeiten muss.
   - Nutzen Sie Datenstrukturen, die eine Trennung der Informationen nach Boersen ermoeglichen.

3. **Zeitsynchronisation:**
   - Verschiedene Boersen arbeiten in unterschiedlichen Zeitzonen. Stellen Sie sicher, dass der Roboter korrekt mit ihren Handelssitzungen synchronisiert ist.

4. **Arbitragestrategien:**
   - Nutzen Sie den Roboter zur Erkennung von Preisabweichungen zwischen Boersen.
   - Beispiel: Kauf an einer Boerse und Verkauf an einer anderen mit Gewinn aus der Preisdifferenz.

## Tipps:

- Stellen Sie sicher, dass Ihr Roboter fuer die Verarbeitung grosser Datenmengen in Echtzeit optimiert ist.
- Beginnen Sie mit einer kleinen Anzahl von Boersen, um die Roboterleistung zu testen.
- Aktualisieren Sie regelmaessig die API-Schluessel und verfolgen Sie Aenderungen der Boersenbedingungen.

## Programme fuer die Arbeit:

- **[StockSharp](https://stocksharp.ru/):** Universelle Plattform mit Unterstuetzung mehrerer Verbindungen.
- **[QuantConnect](https://www.quantconnect.com/):** Cloud-Plattform mit Unterstuetzung mehrerer Maerkte.
- **TSLab:** Geeignet fuer die Automatisierung der Arbeit mit mehreren Boersen, erfordert jedoch Vorkonfiguration.
