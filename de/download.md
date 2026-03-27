---
layout: default
title: "OS Engine herunterladen"
description: "Wie man OS Engine-Plattformen herunterlädt und installiert"
lang: "de"
permalink: /de/download/
---

{% assign lang = "de" %}
{% assign t = site.data.i18n[lang] %}

# OS Engine: Open-Source-Handelsplattformen

In der Welt des Tradings werden Open-Source-Engines (OS Engine) immer beliebter. Sie bieten Flexibilität, Transparenz und die Möglichkeit, Strategien an individuelle Bedürfnisse anzupassen.

## Was ist OS Engine?

OS Engine ist ein allgemeiner Name für Open-Source-Plattformen (daher der Name Open Source oder OSEngine), die im algorithmischen Handel eingesetzt werden. Diese Engines ermöglichen es Tradern, eigene Handelsstrategien ohne erhebliche finanzielle Kosten zu entwickeln und zu testen.

## Vorteile der Verwendung von OS Engine

1. Flexibilität: Anpassungsfähigkeit an verschiedene Handelsbedingungen.
2. Kostenersparnis: Kostenloser Zugang und keine Lizenzgebühren.
3. Transparenz: Zugang zum Code für Studium und Modifikation.
4. Aktive Community: Unterstützung durch Entwickler und andere Nutzer.

## Beliebte OS Engine-Plattformen

<div class="platforms-grid">
{% for project in site.data.de.projects %}
<div class="platform-section">
  <div class="platform-header">
    <img src="{{ project.image }}" alt="{{ project.name }}" class="platform-logo">
    <h3>{{ project.name }}</h3>
  </div>

  <div class="platform-content">
    <p>{{ project.description }}</p>

    <div class="platform-advantages">
      {% for advantage in project.advantages %}
      <span class="advantage-tag">{{ advantage }}</span>
      {% endfor %}
    </div>

    <div class="platform-details">
      <div class="detail-item">
        <strong>Systemanforderungen:</strong>
        <span>{{ project.system_requirements }}</span>
      </div>

      <div class="detail-item">
        <strong>Installation:</strong>
        <code>{{ project.install_command }}</code>
      </div>
    </div>

    <div class="platform-links">
      <a href="{{ project.download_url }}" class="platform-link download-link" target="_blank">
        <span class="link-icon">⬇️</span>
        <span class="link-text">Herunterladen</span>
      </a>
      <a href="{{ project.github }}" class="platform-link github-link" target="_blank">
        <span class="link-icon">📁</span>
        <span class="link-text">GitHub</span>
      </a>
      <a href="{{ project.website }}" class="platform-link website-link" target="_blank">
        <span class="link-icon">🌐</span>
        <span class="link-text">Webseite</span>
      </a>
      {% if project.documentation %}
      <a href="{{ project.documentation }}" class="platform-link docs-link" target="_blank">
        <span class="link-icon">📖</span>
        <span class="link-text">Dokumentation</span>
      </a>
      {% endif %}
      {% if project.telegram_chat %}
      <a href="{{ project.telegram_chat }}" class="platform-link chat-link" target="_blank">
        <span class="link-icon">💬</span>
        <span class="link-text">Chat</span>
      </a>
      {% endif %}
    </div>
  </div>
</div>
{% endfor %}
</div>

## Wie man eine Plattform auswählt

### Für Anfänger
- StockSharp: Grafische Oberfläche, No-Code-Strategie-Designer
- Backtrader: Einfache Python-API, gute Dokumentation

### Für Python-Entwickler
- LEAN: Professionelle Engine mit Cloud-Funktionen
- Backtrader: Einfachheit und Flexibilität
- CCXT: Spezialisierung auf Kryptowährungen

### Für C#-Entwickler
- StockSharp: Vollwertige Plattform

## Schritt-für-Schritt-Installationsanleitung

### Schritt 1: Plattform auswählen
Bestimmen Sie, welche Plattform am besten zu Ihren Bedürfnissen und Ihrem Erfahrungsniveau passt.

### Schritt 2: Entwicklungsumgebung vorbereiten
- Installieren Sie Git zum Klonen von Repositories
- Installieren Sie Python/C# je nach gewählter Plattform
- Richten Sie Ihre IDE ein (PyCharm, Visual Studio Code, Visual Studio)

### Schritt 3: Plattform installieren
Verwenden Sie die oben aufgeführten Installationsbefehle für jede Plattform.

### Schritt 4: Dokumentation studieren
Lesen Sie die offizielle Dokumentation und führen Sie Beispiele aus.

### Schritt 5: Der Community beitreten
Treten Sie Support-Chats bei, um Hilfe und Erfahrungsaustausch zu erhalten.

### Schritt 6: Mit der Entwicklung beginnen
Erstellen Sie Ihre eigenen Handelsstrategien, beginnend mit einfachen Algorithmen.
