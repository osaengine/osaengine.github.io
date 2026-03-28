---
layout: default
title: "Descargar OS Engine"
description: "Como descargar e instalar las plataformas OS Engine"
lang: "es"
permalink: /es/download/
---

{% assign lang = "es" %}
{% assign t = site.data.i18n[lang] %}

<section class="hero">
    <h2>{{ t.download_title | default: "Plataformas de Trading Open Source" }}</h2>
    <p>{{ t.download_description | default: "Plataformas gratuitas para trading algoritmico — descarga, instala y comienza a desarrollar estrategias." }}</p>
</section>

<div class="download-container">
{% for project in site.data.es.projects %}
<div class="download-card">
  <div class="download-card-header">
    <img src="{{ project.image }}" alt="{{ project.name }}" class="download-logo">
    <div>
      <h3>{{ project.name }}</h3>
      <p class="download-desc">{{ project.description }}</p>
    </div>
  </div>
  <div class="download-tags">
    {% for advantage in project.advantages %}
    <span class="tag">{{ advantage }}</span>
    {% endfor %}
  </div>
  <div class="download-links">
    {% if project.download_url %}<a href="{{ project.download_url }}" target="_blank">Descargar</a>{% endif %}
    {% if project.github %}<a href="{{ project.github }}" target="_blank">GitHub</a>{% endif %}
    {% if project.website %}<a href="{{ project.website }}" target="_blank">Sitio web</a>{% endif %}
    {% if project.documentation %}<a href="{{ project.documentation }}" target="_blank">Documentacion</a>{% endif %}
  </div>
</div>
{% endfor %}
</div>
