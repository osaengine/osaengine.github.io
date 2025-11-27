---
layout: post
title: "Как мы собираем каталог open-source решений для алготрейдинга"
date: 2026-04-28
categories: [алготрейдинг, open-source]
tags: [OSA-Engine, каталог, open-source, Freqtrade, LEAN, StockSharp, Backtrader]
author: OSA Engine Team
excerpt: "История создания OSA Engine — каталога open-source платформ для алготрейдинга. Критерии отбора, методология оценки, структура данных и планы развития проекта."
image: /assets/images/blog/osa_engine_catalog.png
---

В предыдущих статьях мы разобрали технические аспекты алготрейдинга: от [архитектуры роботов]({{ site.baseurl }}/2026/03/03/kak-ustroen-opensource-robot-vnutri.html) до [типичных ошибок начинающих]({{ site.baseurl }}/2026/04/21/tipichnye-oshibki-nachinayushchih.html). Теперь расскажем, **как и зачем** мы создали OSA Engine — каталог open-source решений для алготрейдинга.

---

## Проблема: информационный хаос в мире open-source

Когда новичок хочет начать в алготрейдинге, он сталкивается с вопросами:
- Какую платформу выбрать: Freqtrade, LEAN, Backtrader, StockSharp?
- Какая поддерживает нужную мне биржу?
- Какая подходит для моего языка программирования?
- Какая активно развивается, а какая заброшена?

Информация разбросана по:
- GitHub репозиториям
- Форумам и Reddit
- Документации (часто устаревшей)
- Личным блогам

**OSA Engine** решает эту проблему: централизованный каталог с **объективными критериями оценки** каждой платформы.

---

## Методология: как мы отбираем и оценива

ем платформы

### Критерии включения в каталог

Платформа должна:
1. **Open-source** — исходный код доступен (MIT, Apache, GPL лицензии)
2. **Активная разработка** — минимум 1 коммит за последние 6 месяцев
3. **Документация** — есть README и базовые примеры
4. **Community** — минимум 100 GitHub stars ИЛИ активное комьюнити (форум/Discord)

### Параметры оценки

Для каждой платформы мы собираем:

```yaml
platform:
  name: "Freqtrade"
  language: "Python"
  license: "GPL-3.0"

  metrics:
    github_stars: 28500
    forks: 5800
    contributors: 420
    last_commit: "2025-03-15"
    first_release: "2017-05-18"

  features:
    backtesting: true
    live_trading: true
    paper_trading: true
    ml_support: true
    gui: true

  supported_exchanges:
    - Binance
    - Bybit
    - Kraken
    - (200+ via CCXT)

  supported_markets:
    - crypto

  learning_curve: "medium"  # easy, medium, hard

  strengths:
    - "Огромное комьюнити"
    - "Встроенная поддержка ML (FreqAI)"
    - "200+ бирж через CCXT"

  weaknesses:
    - "Только крипто"
    - "GPL лицензия (не подходит для коммерческих продуктов)"
```

### Источники данных

1. **GitHub API**
```python
import requests

def get_github_stats(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    response = requests.get(url)
    data = response.json()

    return {
        'stars': data['stargazers_count'],
        'forks': data['forks_count'],
        'watchers': data['subscribers_count'],
        'open_issues': data['open_issues_count'],
        'last_commit': data['pushed_at'],
        'created_at': data['created_at'],
        'language': data['language'],
        'license': data['license']['key'] if data.get('license') else None
    }

stats = get_github_stats('freqtrade', 'freqtrade')
print(stats)
```

2. **Документация** — ручной анализ features
3. **Community** — мониторинг форумов, Discord, Telegram
4. **Тестирование** — установка и пробный запуск

---

## Структура каталога OSA Engine

### 1. Категории платформ

- **Universal** (любые рынки): LEAN, Backtrader
- **Crypto-focused**: Freqtrade, Jesse, Hummingbot
- **Stock-focused**: StockSharp, QuantConnect
- **Frameworks**: Zipline, PyAlgoTrade
- **Infrastructure**: NautilusTrader, MBATS

### 2. Фильтры и поиск

```python
class PlatformFilter:
    def __init__(self, catalog):
        self.catalog = catalog

    def filter_by_language(self, language):
        """Фильтр по языку программирования"""
        return [p for p in self.catalog if p['language'] == language]

    def filter_by_exchange(self, exchange):
        """Фильтр по поддержке биржи"""
        return [p for p in self.catalog if exchange in p['supported_exchanges']]

    def filter_by_market(self, market):
        """Фильтр по типу рынка (crypto, stocks, forex)"""
        return [p for p in self.catalog if market in p['supported_markets']]

    def filter_active(self, months=6):
        """Только активно развивающиеся (коммит за последние N месяцев)"""
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=months*30)

        return [p for p in self.catalog
                if datetime.fromisoformat(p['metrics']['last_commit']) > cutoff]

# Использование
catalog = load_catalog()
filter = PlatformFilter(catalog)

# Найти Python-платформы для Binance
results = filter.filter_by_language('Python')
results = [p for p in results if 'Binance' in p['supported_exchanges']]

for platform in results:
    print(f"{platform['name']}: {platform['github_stars']} stars")
```

### 3. Рейтинговая система

```python
class PlatformRanker:
    def calculate_score(self, platform):
        """Рассчитываем общий score платформы"""
        score = 0

        # 1. Community (0-40 points)
        stars = platform['metrics']['github_stars']
        if stars > 10000:
            score += 40
        elif stars > 5000:
            score += 30
        elif stars > 1000:
            score += 20
        elif stars > 100:
            score += 10

        # 2. Activity (0-30 points)
        last_commit = datetime.fromisoformat(platform['metrics']['last_commit'])
        days_since_commit = (datetime.now() - last_commit).days

        if days_since_commit < 7:
            score += 30  # очень активная
        elif days_since_commit < 30:
            score += 25
        elif days_since_commit < 90:
            score += 15
        elif days_since_commit < 180:
            score += 5
        # иначе 0

        # 3. Features (0-30 points)
        features = platform['features']
        if features.get('backtesting'):
            score += 10
        if features.get('live_trading'):
            score += 10
        if features.get('ml_support'):
            score += 5
        if features.get('gui'):
            score += 5

        return score

    def rank_platforms(self, platforms):
        """Ранжируем платформы по score"""
        ranked = []
        for p in platforms:
            score = self.calculate_score(p)
            ranked.append((p, score))

        # Сортируем по убыванию score
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

ranker = PlatformRanker()
ranked = ranker.rank_platforms(catalog)

print("=== Top 10 Platforms ===")
for i, (platform, score) in enumerate(ranked[:10], 1):
    print(f"{i}. {platform['name']}: {score}/100 points")
```

---

## Автоматизация сбора данных

### GitHub Actions workflow

```yaml
# .github/workflows/update_catalog.yml
name: Update Catalog Data

on:
  schedule:
    - cron: '0 0 * * 0'  # Каждое воскресенье
  workflow_dispatch:  # Ручной запуск

jobs:
  update:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install requests pyyaml

      - name: Update GitHub metrics
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: python scripts/update_github_stats.py

      - name: Commit changes
        run: |
          git config user.name "OSA Bot"
          git config user.email "bot@osa-engine.org"
          git add data/platforms/*.yml
          git commit -m "Auto-update: GitHub metrics $(date +%Y-%m-%d)" || echo "No changes"
          git push
```

### Скрипт обновления

```python
# scripts/update_github_stats.py
import os
import yaml
import requests
from datetime import datetime
from pathlib import Path

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}

def update_platform_stats(platform_file):
    """Обновляем статистику платформы"""
    with open(platform_file, 'r') as f:
        platform = yaml.safe_load(f)

    repo = platform.get('github_repo')
    if not repo:
        return

    # Получаем данные из GitHub API
    url = f"https://api.github.com/repos/{repo}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Error fetching {repo}: {response.status_code}")
        return

    data = response.json()

    # Обновляем метрики
    platform['metrics']['github_stars'] = data['stargazers_count']
    platform['metrics']['forks'] = data['forks_count']
    platform['metrics']['contributors'] = data.get('contributors_count', 0)
    platform['metrics']['last_commit'] = data['pushed_at']
    platform['metrics']['open_issues'] = data['open_issues_count']
    platform['updated_at'] = datetime.now().isoformat()

    # Сохраняем
    with open(platform_file, 'w') as f:
        yaml.dump(platform, f, default_flow_style=False, allow_unicode=True)

    print(f"✓ Updated {platform['name']}")

# Обновляем все платформы
data_dir = Path('data/platforms')
for platform_file in data_dir.glob('*.yml'):
    update_platform_stats(platform_file)
```

---

## Web-интерфейс каталога

### Frontend (React)

```jsx
// components/PlatformCard.jsx
import React from 'react';

function PlatformCard({ platform }) {
  return (
    <div className="platform-card">
      <h3>{platform.name}</h3>

      <div className="metrics">
        <span>⭐ {platform.metrics.github_stars.toLocaleString()}</span>
        <span>🔀 {platform.metrics.forks.toLocaleString()}</span>
        <span>👥 {platform.metrics.contributors}</span>
      </div>

      <div className="tags">
        <span className="tag">{platform.language}</span>
        {platform.supported_markets.map(market => (
          <span key={market} className="tag tag-market">{market}</span>
        ))}
      </div>

      <p>{platform.description}</p>

      <div className="features">
        {platform.features.backtesting && <span>✓ Backtesting</span>}
        {platform.features.live_trading && <span>✓ Live Trading</span>}
        {platform.features.ml_support && <span>✓ ML Support</span>}
      </div>

      <a href={`https://github.com/${platform.github_repo}`}
         className="btn-primary">
        View on GitHub
      </a>
    </div>
  );
}

// components/PlatformList.jsx
function PlatformList() {
  const [platforms, setPlatforms] = useState([]);
  const [filters, setFilters] = useState({
    language: 'all',
    market: 'all',
    exchange: 'all'
  });

  useEffect(() => {
    // Загружаем данные
    fetch('/api/platforms')
      .then(res => res.json())
      .then(data => setPlatforms(data));
  }, []);

  const filteredPlatforms = platforms.filter(p => {
    if (filters.language !== 'all' && p.language !== filters.language)
      return false;
    if (filters.market !== 'all' && !p.supported_markets.includes(filters.market))
      return false;
    if (filters.exchange !== 'all' && !p.supported_exchanges.includes(filters.exchange))
      return false;
    return true;
  });

  return (
    <div className="platform-list">
      <div className="filters">
        <select onChange={e => setFilters({...filters, language: e.target.value})}>
          <option value="all">All Languages</option>
          <option value="Python">Python</option>
          <option value="C#">C#</option>
          <option value="Rust">Rust</option>
        </select>

        <select onChange={e => setFilters({...filters, market: e.target.value})}>
          <option value="all">All Markets</option>
          <option value="crypto">Crypto</option>
          <option value="stocks">Stocks</option>
          <option value="forex">Forex</option>
        </select>
      </div>

      <div className="platforms-grid">
        {filteredPlatforms.map(platform => (
          <PlatformCard key={platform.name} platform={platform} />
        ))}
      </div>
    </div>
  );
}
```

---

## Планы развития OSA Engine

### Текущий статус (2026)

- ✅ 50+ платформ в каталоге
- ✅ Автоматическое обновление метрик
- ✅ Web-интерфейс с фильтрами
- ✅ API для интеграций

### Roadmap 2026-2027

1. **Community Reviews** (Q2 2026)
   - Пользовательские отзывы и рейтинги
   - Реальный опыт использования

2. **Tutorials & Guides** (Q3 2026)
   - Гайды "Как начать с X"
   - Сравнительные обзоры

3. **Performance Benchmarks** (Q4 2026)
   - Единый набор тестов
   - Сравнение скорости бэктестинга
   - Замеры latency в live trading

4. **Integration Marketplace** (Q1 2027)
   - Каталог готовых стратегий
   - Плагины и расширения
   - Индикаторы и tools

---

## Как внести вклад в OSA Engine

### 1. Добавить новую платформу

```bash
# 1. Fork репозитория
git clone https://github.com/osa-engine/osa-engine.git

# 2. Создать файл платформы
cp data/platforms/template.yml data/platforms/my_platform.yml

# 3. Заполнить данные
nano data/platforms/my_platform.yml

# 4. Отправить Pull Request
git checkout -b add-my-platform
git add data/platforms/my_platform.yml
git commit -m "Add MyPlatform to catalog"
git push origin add-my-platform
```

### 2. Обновить существующую информацию

Заметили устаревшую информацию? Создайте Issue или Pull Request:

```yaml
# Было
features:
  ml_support: false

# Стало (платформа добавила ML в новой версии)
features:
  ml_support: true
  ml_frameworks:
    - TensorFlow
    - PyTorch
```

### 3. Написать tutorial

```markdown
# Getting Started with Freqtrade

## Installation
...

## First Strategy
...

## Backtesting
...
```

---

## Заключение

**OSA Engine** — это живой каталог, который развивается благодаря комьюнити. Наша цель: сделать выбор платформы для алготрейдинга **простым и обоснованным**.

Присоединяйтесь:
- GitHub: github.com/osa-engine/osa-engine
- Website: osa-engine.org
- Discord: discord.gg/osa-engine

В следующей статье: **практический гайд по OSA Engine для новичков** — как выбрать первую платформу и начать торговать.
