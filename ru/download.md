---
layout: default
title: "Скачать OS Engine"
description: "Как скачивать и устанавливать платформы OS Engine"
lang: "ru"
permalink: /ru/download/
---

{% assign lang = "ru" %}
{% assign t = site.data.i18n[lang] %}

<section class="hero">
    <h2>{{ t.download_title | default: "Открытые платформы для трейдинга" }}</h2>
    <p>{{ t.download_description | default: "Бесплатные платформы для алгоритмической торговли — скачайте, установите и начните разработку стратегий." }}</p>
</section>

<div class="download-container">
{% for project in site.data.ru.projects %}
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
    {% if project.download_url %}<a href="{{ project.download_url }}" target="_blank">Скачать</a>{% endif %}
    {% if project.github %}<a href="{{ project.github }}" target="_blank">GitHub</a>{% endif %}
    {% if project.website %}<a href="{{ project.website }}" target="_blank">Сайт</a>{% endif %}
    {% if project.documentation %}<a href="{{ project.documentation }}" target="_blank">Документация</a>{% endif %}
  </div>
</div>
{% endfor %}
</div>
