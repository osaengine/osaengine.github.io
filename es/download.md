---
layout: default
title: "Descargar OS Engine"
description: "Como descargar e instalar las plataformas OS Engine"
lang: "es"
permalink: /es/download/
---

{% assign lang = "es" %}
{% assign t = site.data.i18n[lang] %}

# OS Engine: Plataformas de Trading Open Source

En el mundo del trading, los motores de codigo abierto (OS Engine) son cada vez mas populares. Ofrecen flexibilidad, transparencia y la capacidad de personalizar estrategias segun las necesidades individuales.

## Que es OS Engine?

OS Engine es un nombre general para plataformas de codigo abierto (de ahi el nombre, Open Source, u OSEngine) utilizadas en el trading algoritmico. Estos motores permiten a los traders desarrollar y probar sus propias estrategias de trading sin costes financieros significativos.

## Ventajas de Usar OS Engine

1. Flexibilidad: Capacidad de adaptacion a diversas condiciones de trading.
2. Ahorro de costes: Acceso gratuito y sin tarifas de licencia.
3. Transparencia: Acceso al codigo para estudio y modificacion.
4. Comunidad activa: Soporte de desarrolladores y otros usuarios.

## Plataformas OS Engine Populares

<div class="platforms-grid">
{% for project in site.data.es.projects %}
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
        <strong>Requisitos del sistema:</strong>
        <span>{{ project.system_requirements }}</span>
      </div>

      <div class="detail-item">
        <strong>Instalacion:</strong>
        <code>{{ project.install_command }}</code>
      </div>
    </div>

    <div class="platform-links">
      <a href="{{ project.download_url }}" class="platform-link download-link" target="_blank">
        <span class="link-icon">⬇️</span>
        <span class="link-text">Descargar</span>
      </a>
      <a href="{{ project.github }}" class="platform-link github-link" target="_blank">
        <span class="link-icon">📁</span>
        <span class="link-text">GitHub</span>
      </a>
      <a href="{{ project.website }}" class="platform-link website-link" target="_blank">
        <span class="link-icon">🌐</span>
        <span class="link-text">Sitio web</span>
      </a>
      {% if project.documentation %}
      <a href="{{ project.documentation }}" class="platform-link docs-link" target="_blank">
        <span class="link-icon">📖</span>
        <span class="link-text">Documentacion</span>
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

## Como Elegir una Plataforma

### Para Principiantes
- StockSharp: Interfaz grafica, disenador de estrategias sin codigo
- Backtrader: API Python sencilla, buena documentacion

### Para Desarrolladores Python
- LEAN: Motor profesional con capacidades en la nube
- Backtrader: Simplicidad y flexibilidad
- CCXT: Especializacion en criptomonedas

### Para Desarrolladores C#
- StockSharp: Plataforma completa

## Guia de Instalacion Paso a Paso

### Paso 1: Elegir una Plataforma
Determina que plataforma se adapta mejor a tus necesidades y nivel de experiencia.

### Paso 2: Preparar el Entorno de Desarrollo
- Instalar Git para clonar repositorios
- Instalar Python/C# segun la plataforma elegida
- Configurar tu IDE (PyCharm, Visual Studio Code, Visual Studio)

### Paso 3: Instalar la Plataforma
Usa los comandos de instalacion indicados para cada plataforma arriba.

### Paso 4: Estudiar la Documentacion
Lee la documentacion oficial y ejecuta los ejemplos.

### Paso 5: Unirse a la Comunidad
Unete a los chats de soporte para obtener ayuda y compartir experiencias.

### Paso 6: Comenzar a Desarrollar
Crea tus propias estrategias de trading, comenzando con algoritmos simples.
