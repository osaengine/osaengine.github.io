---
layout: post
title: "Alpaca veroeffentlicht MCP-Server mit KI-Unterstuetzung: Ausfuehrliche Video-Anleitung und Open-Source-Code"
description: "Der offizielle Model Context Protocol-Server fuer die Alpaca Trading API ermoeglicht den Handel mit Aktien und Optionen ueber Claude AI, VS Code und andere IDEs. Das Unternehmen hat ein fuenfstufiges Video-Tutorial vorbereitet und den Quellcode auf GitHub veroeffentlicht."
date: 2025-08-04
image: /assets/images/blog/alpaca_mcp_server_release.png
tags: [Alpaca, MCP, algorithmic trading, AI, GitHub]
lang: de
---

Ende Juli stellte Alpaca einen **offiziellen MCP-Server** fuer seine Trading API vor und veroeffentlichte sofort ein **fuenfminuetiges Video-Tutorial** darueber, wie man ihn lokal einrichtet und mit Claude AI verbindet. Nach den Vorstellungen der Entwickler soll der neue Server die Erstellung von Handelsstrategien in natuerlicher Sprache vereinfachen und die Zeit zwischen Idee und Handelsumsetzung verkuerzen.

## Was das Video zeigt

Der Autor des Videos ["How to Set Up Alpaca MCP Server to Trade with Claude AI"](https://www.youtube.com/watch?v=W9KkdTZEvGM) demonstriert die Einrichtung in fuenf Schritten:

1. Repository klonen und virtuelle Umgebung erstellen
2. Umgebungsvariablen mit Alpaca API-Schluesseln konfigurieren
3. Server starten (stdio- oder HTTP-Transport)
4. Verbindung zu Claude Desktop ueber `mcp.json` herstellen
5. Erste Handelsanfragen in natuerlicher Sprache

So koennen Sie auch ohne tiefgreifende Python-Kenntnisse schnell den Handel ueber einen KI-Assistenten testen.

## Hauptfunktionen des MCP-Servers

* **Marktdaten**: Echtzeitkurse, historische Bars, Optionen und Greeks
* **Kontoverwaltung**: Kontostand, Kaufkraft, Kontostatus
* **Positionen und Orders**: Eroeffnung, Liquidation, Handelshistorie
* **Optionen**: Kontraktsuche, Multi-Leg-Strategien
* **Unternehmensaktionen**: Berichtskalender, Splits, Dividenden
* **Watchlist** und Vermoegenswertsuche

Die vollstaendige Funktionsliste ist in der README des Repositories verfuegbar.

## GitHub-Repository

Das Projekt ist unter der **MIT**-Lizenz veroeffentlicht, hat bereits **170+ Sterne und ~50 Forks** gesammelt und erhaelt aktiv Pull Requests aus der Community. Das letzte Update stammt vom 31. Juli 2025.

```bash
git clone https://github.com/alpacahq/alpaca-mcp-server.git
cd alpaca-mcp-server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python alpaca_mcp_server.py
```

## Warum das wichtig ist

Der MCP-Server verwandelt die Alpaca API in eine "Sandbox" fuer KI-Modelle: Sie koennen jetzt **Positionen hedgen, Watchlists erstellen und Limit-Orders platzieren, indem Sie einfach Befehle in natuerlicher Sprache formulieren**. Fuer Trader bedeutet das:

- Schnelleres **Strategie-Prototyping** ohne zusaetzlichen Code
- Integration mit Claude AI, VS Code, Cursor und anderen Entwicklungstools
- Moeglichkeit, mehrere Konten (Paper und Live) ueber Umgebungsvariablen zu verbinden

Alpaca setzt seinen Kurs zur Demokratisierung des algorithmischen Handels fort, und die Community fuegt bereits Unterstuetzung fuer neue Sprachen und Transporte hinzu. Wenn Sie KI-gestuetztes Trading ausprobieren wollten, ist jetzt ein ausgezeichneter Zeitpunkt dafuer.

> **Links:**
> -- Video-Anleitung auf YouTube: <https://www.youtube.com/watch?v=W9KkdTZEvGM>
> -- GitHub-Repository: <https://github.com/alpacahq/alpaca-mcp-server>
