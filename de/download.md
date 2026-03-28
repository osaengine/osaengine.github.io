---
layout: default
title: "OS Engine herunterladen"
description: "Wie man OS Engine-Plattformen herunterlädt und installiert"
lang: "de"
permalink: /de/download/
---

{% assign lang = "de" %}
{% assign t = site.data.i18n[lang] %}

<section class="hero">
    <h2>{{ t.download_title | default: "Open-Source-Handelsplattformen" }}</h2>
    <p>{{ t.download_description | default: "Kostenlose Plattformen für algorithmischen Handel — herunterladen, installieren und mit der Strategieentwicklung beginnen." }}</p>
</section>

<div class="download-container">
{% for project in site.data.de.projects %}
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
    {% if project.download_url %}<a href="{{ project.download_url }}" target="_blank">Herunterladen</a>{% endif %}
    {% if project.github %}<a href="{{ project.github }}" target="_blank">GitHub</a>{% endif %}
    {% if project.website %}<a href="{{ project.website }}" target="_blank">Webseite</a>{% endif %}
    {% if project.documentation %}<a href="{{ project.documentation }}" target="_blank">Dokumentation</a>{% endif %}
  </div>
</div>
{% endfor %}
</div>
