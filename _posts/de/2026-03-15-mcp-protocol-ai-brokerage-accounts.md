---
layout: post
title: "MCP: Das Protokoll, das KI mit Brokerkonten verbindet"
description: "Das Model Context Protocol von Anthropic ermoeglicht KI-Modellen die direkte Interaktion mit externen Systemen, einschliesslich Broker-Plattformen. Wir erklaeren, wie es funktioniert."
date: 2026-03-15
image: /assets/images/blog/mcp-protocol-trading.png
tags: [AI, MCP, trading, brokers, Claude]
lang: de
---

## Was ist MCP

Das **Model Context Protocol (MCP)** ist ein offenes Protokoll, entwickelt von [Anthropic](https://www.anthropic.com/), das die Art und Weise standardisiert, wie KI-Modelle sich mit externen Datenquellen und Werkzeugen verbinden. Vereinfacht ausgedrueckt: MCP ist ein "USB-Anschluss" fuer KI.

Vor MCP erforderte jede Integration von KI mit einem externen Dienst das Schreiben von individuellem Code. MCP bietet einen einheitlichen Standard, der nach dem **Client-Server**-Modell funktioniert:

```
KI-Modell (Client) <-> MCP-Protokoll <-> MCP-Server <-> Externer Dienst
```

## Wie MCP mit Brokerkonten funktioniert

Die MCP-Architektur bietet drei Arten der Interaktion:

### 1. Resources (Ressourcen)
Datenabfrage: Kurse, Portfolio, Handelshistorie, Kontostand.

### 2. Tools (Werkzeuge)
Aktionen ausfuehren: Orders platzieren, Orders aendern, Positionen schliessen.

### 3. Prompts
Vordefinierte Interaktionsvorlagen: "Zeige mein Portfolio", "Analysiere meine Handelshistorie der letzten Woche".

## Interaktionsbeispiel

Stellen Sie sich ein Gespraech mit einem KI-Assistenten vor:

> **Sie**: Welche Positionen befinden sich derzeit in meinem Portfolio?
>
> **KI** *(fragt den Broker via MCP ab)*: Sie haben 3 Positionen in Ihrem Portfolio:
> - AAPL: 50 Aktien, +4,2%
> - GOOGL: 20 Aktien, -1,1%
> - SBER: 100 Aktien, +2,8%
>
> Gesamtrendite des Portfolios: +3,1%

Oder sogar:

> **Sie**: Setze einen Stop-Loss auf AAPL bei $210.
>
> **KI**: Stop-Order zum Verkauf von 50 Aktien AAPL bei $210 erstellt. Bestaetigen?

## Sicherheit

Sicherheit ist das zentrale Thema bei Finanzintegrationen. MCP bietet:

- **OAuth 2.0-Autorisierung** — Standardprotokoll fuer sicheren Zugriff
- **Granulare Berechtigungen** — nur Lesezugriff ohne Handelsberechtigung moeglich
- **Audit-Logs** — alle Aktionen werden protokolliert
- **Operationsbestaetigung** — kritische Aktionen erfordern ausdrueckliche Zustimmung des Nutzers

## Wer unterstuetzt MCP bereits

Stand Maerz 2026 sind MCP-Server verfuegbar fuer:

- **TradeStation** — der erste Broker mit vollstaendiger MCP-Integration
- **FactSet**, **S&P Global**, **MSCI** — ueber Anthropics Finanz-Plugins
- Weitere Broker und Plattformen erkunden Integrationsmoeglichkeiten

## Was das fuer den Algo-Handel bedeutet

MCP ebnet den Weg zum **KI-gesteuerten Handel**, bei dem das Modell nicht nur Signale generiert, sondern:

1. Eigenstaendig die Marktsituation analysieren kann
2. Handelsentscheidungen formulieren kann
3. Diese ueber den Broker ausfuehren kann
4. Ergebnisse ueberwachen und die Strategie anpassen kann

Das bedeutet nicht, dass man der KI die volle Kontrolle ueber das Konto geben sollte. Aber der **Human-in-the-Loop**-Ansatz — bei dem die KI vorschlaegt und der Mensch bestaetigt — ist bereits Realitaet.

## Wie man beginnt

Wenn Sie mit MCP experimentieren moechten:

1. Installieren Sie [Claude Desktop](https://claude.ai/download) oder einen anderen MCP-kompatiblen Client
2. Verbinden Sie den MCP-Server Ihres Brokers
3. Starten Sie im Nur-Lesen-Modus — nur Daten ansehen
4. Fuegen Sie schrittweise weitere Funktionen hinzu, wenn Ihr Vertrauen waechst

MCP ist keine Revolution ueber Nacht — es ist das Fundament fuer eine neue Aera der Mensch-KI-Interaktion im Finanzwesen.
