---
layout: default
title: "下载 OS Engine"
description: "如何下载和安装 OS Engine 平台"
lang: "zh"
permalink: /zh/download/
---

{% assign lang = "zh" %}
{% assign t = site.data.i18n[lang] %}

# OS Engine：开源交易平台

在交易领域，开源引擎（OS Engine）正变得越来越受欢迎。它们提供了灵活性、透明度以及根据个人需求定制策略的能力。

## 什么是 OS Engine？

OS Engine 是用于算法交易的开源平台的统称（因此得名 Open Source，即 OSEngine）。这些引擎允许交易者在无需大量资金投入的情况下开发和测试自己的交易策略。

## 使用 OS Engine 的优势

1. 灵活性：能够适应各种交易条件。
2. 节省成本：免费访问，无需许可费用。
3. 透明度：可以访问代码进行研究和修改。
4. 活跃社区：来自开发者和其他用户的支持。

## 热门 OS Engine 平台

<div class="platforms-grid">
{% for project in site.data.zh.projects %}
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
        <strong>系统要求：</strong>
        <span>{{ project.system_requirements }}</span>
      </div>

      <div class="detail-item">
        <strong>安装命令：</strong>
        <code>{{ project.install_command }}</code>
      </div>
    </div>

    <div class="platform-links">
      <a href="{{ project.download_url }}" class="platform-link download-link" target="_blank">
        <span class="link-icon">⬇️</span>
        <span class="link-text">下载</span>
      </a>
      <a href="{{ project.github }}" class="platform-link github-link" target="_blank">
        <span class="link-icon">📁</span>
        <span class="link-text">GitHub</span>
      </a>
      <a href="{{ project.website }}" class="platform-link website-link" target="_blank">
        <span class="link-icon">🌐</span>
        <span class="link-text">官网</span>
      </a>
      {% if project.documentation %}
      <a href="{{ project.documentation }}" class="platform-link docs-link" target="_blank">
        <span class="link-icon">📖</span>
        <span class="link-text">文档</span>
      </a>
      {% endif %}
      {% if project.telegram_chat %}
      <a href="{{ project.telegram_chat }}" class="platform-link chat-link" target="_blank">
        <span class="link-icon">💬</span>
        <span class="link-text">聊天</span>
      </a>
      {% endif %}
    </div>
  </div>
</div>
{% endfor %}
</div>

## 如何选择平台

### 适合初学者
- StockSharp：图形界面，无代码策略设计器
- Backtrader：简单的 Python API，优秀的文档

### 适合 Python 开发者
- LEAN：具有云端功能的专业引擎
- Backtrader：简洁且灵活
- CCXT：专注于加密货币

### 适合 C# 开发者
- StockSharp：功能齐全的平台

## 分步安装指南

### 第一步：选择平台
确定哪个平台最适合您的需求和经验水平。

### 第二步：准备开发环境
- 安装 Git 用于克隆仓库
- 根据所选平台安装 Python/C#
- 设置您的 IDE（PyCharm、Visual Studio Code、Visual Studio）

### 第三步：安装平台
使用上面每个平台列出的安装命令。

### 第四步：学习文档
阅读官方文档并运行示例。

### 第五步：加入社区
加入支持聊天群以获取帮助和分享经验。

### 第六步：开始开发
创建您自己的交易策略，从简单的算法开始。
