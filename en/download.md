---
layout: default
title: "Download OS Engine"
description: "How to download and install OS Engine platforms"
lang: "en"
permalink: /en/download/
---

{% assign lang = "en" %}
{% assign t = site.data.i18n[lang] %}

<section class="hero">
    <h2>{{ t.download_title | default: "Open Source Trading Platforms" }}</h2>
    <p>{{ t.download_description | default: "Free platforms for algorithmic trading — download, install, and start developing strategies." }}</p>
</section>

<div class="download-container">
{% for project in site.data.en.projects %}
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
    {% if project.download_url %}<a href="{{ project.download_url }}" target="_blank">Download</a>{% endif %}
    {% if project.github %}<a href="{{ project.github }}" target="_blank">GitHub</a>{% endif %}
    {% if project.website %}<a href="{{ project.website }}" target="_blank">Website</a>{% endif %}
    {% if project.documentation %}<a href="{{ project.documentation }}" target="_blank">Docs</a>{% endif %}
  </div>
</div>
{% endfor %}
</div>
