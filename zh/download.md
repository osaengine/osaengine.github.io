---
layout: default
title: "下载 OS Engine"
description: "如何下载和安装 OS Engine 平台"
lang: "zh"
permalink: /zh/download/
---

{% assign lang = "zh" %}
{% assign t = site.data.i18n[lang] %}

<section class="hero">
    <h2>{{ t.download_title | default: "开源交易平台" }}</h2>
    <p>{{ t.download_description | default: "免费的算法交易平台 — 下载、安装并开始开发策略。" }}</p>
</section>

<div class="download-container">
{% for project in site.data.zh.projects %}
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
    {% if project.download_url %}<a href="{{ project.download_url }}" target="_blank">下载</a>{% endif %}
    {% if project.github %}<a href="{{ project.github }}" target="_blank">GitHub</a>{% endif %}
    {% if project.website %}<a href="{{ project.website }}" target="_blank">官网</a>{% endif %}
    {% if project.documentation %}<a href="{{ project.documentation }}" target="_blank">文档</a>{% endif %}
  </div>
</div>
{% endfor %}
</div>
