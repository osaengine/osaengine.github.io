---
layout: default
title: "Download OS Engine"
description: "How to download and install OS Engine platforms"
lang: "en"
permalink: /en/download/
---

{% assign lang = "en" %}
{% assign t = site.data.i18n[lang] %}

# OS Engine: Open Source Trading Platforms

In the trading world, open source engines (OS Engine) are becoming increasingly popular. They offer flexibility, transparency, and the ability to customize strategies to individual needs.

## What is OS Engine?

OS Engine is a general name for open source platforms (hence the name, Open Source, or OSEngine) used in algorithmic trading. These engines allow traders to develop and test their own trading strategies without significant financial costs.

## Advantages of Using OS Engine

1. Flexibility: Ability to adapt to various trading conditions.
2. Cost savings: Free access and no licensing fees.
3. Transparency: Access to code for study and modification.
4. Active community: Support from developers and other users.

## Popular OS Engine Platforms

<div class="platforms-grid">
{% for project in site.data.en.projects %}
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
        <strong>System Requirements:</strong>
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
        <span class="link-text">Download</span>
      </a>
      <a href="{{ project.github }}" class="platform-link github-link" target="_blank">
        <span class="link-icon">📁</span>
        <span class="link-text">GitHub</span>
      </a>
      <a href="{{ project.website }}" class="platform-link website-link" target="_blank">
        <span class="link-icon">🌐</span>
        <span class="link-text">Website</span>
      </a>
      {% if project.documentation %}
      <a href="{{ project.documentation }}" class="platform-link docs-link" target="_blank">
        <span class="link-icon">📖</span>
        <span class="link-text">Documentation</span>
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

## How to Choose a Platform

### For Beginners
- StockSharp: Graphical interface, no-code strategy designer
- Backtrader: Simple Python API, good documentation

### For Python Developers
- LEAN: Professional engine with cloud capabilities
- Backtrader: Simplicity and flexibility
- CCXT: Specialization in cryptocurrencies

### For C# Developers
- StockSharp: Full-featured platform

## Step-by-Step Installation Guide

### Step 1: Choose a Platform
Determine which platform best suits your needs and experience level.

### Step 2: Prepare Your Development Environment
- Install Git for cloning repositories
- Install Python/C# depending on the chosen platform
- Set up your IDE (PyCharm, Visual Studio Code, Visual Studio)

### Step 3: Install the Platform
Use the installation commands listed for each platform above.

### Step 4: Study the Documentation
Read the official documentation and run examples.

### Step 5: Join the Community
Join support chats for help and experience sharing.

### Step 6: Start Developing
Create your own trading strategies, starting with simple algorithms.
