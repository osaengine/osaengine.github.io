---
layout: post
title: "TradeStation + Claude: Der erste Broker mit MCP-Anbindung an KI"
description: "TradeStation wurde der erste Broker mit offizieller MCP-Anbindung an Claude. Jetzt kann man den Handel ueber einen KI-Assistenten steuern."
date: 2026-03-16
image: /assets/images/blog/tradestation-claude-mcp.png
tags: [AI, MCP, TradeStation, trading]
lang: de
---

## TradeStation schreibt Geschichte

Im Januar 2026 wurde **TradeStation** zum weltweit ersten Broker, der einen offiziellen [MCP-Server](https://www.tradestation.com/) fuer die Anbindung an KI-Modelle veroeffentlichte. Das bedeutet, dass TradeStation-Kunden jetzt ueber **Claude** und andere MCP-kompatible KI-Clients mit ihrem Brokerkonto interagieren koennen.

## Was ueber MCP verfuegbar ist

Der TradeStation MCP-Server bietet folgende Moeglichkeiten:

### Datenanzeige (Resources)
- Aktuelles Portfolio und Positionen
- Kontostand und Margin-Anforderungen
- Order- und Handelshistorie
- Echtzeitkurse
- Charts und technische Indikatoren

### Handelsoperationen (Tools)
- Platzierung von Markt- und Limitorders
- Setzen von Stop-Loss und Take-Profit
- Aenderung bestehender Orders
- Stornierung von Orders
- Schliessen von Positionen

### Analytik (Prompts)
- Analyse von Portfolio-Kennzahlen (Sharpe, Drawdown)
- Leistungsvergleich mit Benchmarks
- Risikouebersicht nach Positionen

## Wie es in der Praxis aussieht

Beispielsitzung in Claude Desktop:

```
Benutzer: Zeige meine offenen Positionen und deren P&L

Claude: Frage Daten ueber TradeStation ab...

Ihre offenen Positionen:
┌────────┬─────────┬──────────┬─────────┐
│ Ticker │ Anz.    │ Durchschn│ P&L     │
├────────┼─────────┼──────────┼─────────┤
│ NVDA   │ 30      │ $142,50  │ +$1.245 │
│ MSFT   │ 25      │ $428,00  │ +$312   │
│ TSLA   │ 15      │ $385,20  │ -$89    │
└────────┴─────────┴──────────┴─────────┘

Gesamter unrealisierter P&L: +$1.468
Empfehlung: Die TSLA-Position zeigt einen Verlust,
erwaegen Sie einen Stop-Loss zu setzen.
```

## Sicherheit und Einschraenkungen

TradeStation hat mehrere Schutzebenen implementiert:

- **OAuth 2.0** mit Zwei-Faktor-Authentifizierung
- **Ordergroessen-Limits** — vom Benutzer konfigurierbar
- **Instrumenten-Whitelist** — Handel nur mit erlaubten Titeln
- **Nur-Lesen-Modus** standardmaessig — Handel erfordert explizite Aktivierung
- **Bestaetigung jeder Order** — KI kann ohne Genehmigung nicht handeln

## Marktreaktion

Nach der Ankuendigung:

- Neue Registrierungen bei TradeStation stiegen im Januar um **23%**
- Konkurrenten (Interactive Brokers, Charles Schwab) kuendigten die Entwicklung eigener MCP-Loesungen an
- Die TradeStation-Aktie (Muttergesellschaft Monex Group) stieg um **8%**

## Fuer wen ist das geeignet

Die MCP-Integration von TradeStation ist ideal fuer:

- **Aktive Trader**, die ihr Portfolio per Sprache/Text verwalten moechten
- **Entwickler**, die LLM-basierte Trading-Bots bauen
- **Portfoliomanager**, die eine schnelle analytische Oberflaeche benoetigen

Dies ist der erste, aber sicher nicht der letzte Schritt in Richtung KI-nativer Handel.
