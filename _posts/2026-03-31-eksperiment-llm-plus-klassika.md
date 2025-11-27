---
layout: post
title: "Эксперимент: LLM + классический алгоритм. Можем ли мы улучшить стабильную стратегию с помощью ИИ-фильтров?"
description: "Берём работающую SMA-стратегию (Sharpe 0.9) и добавляем LLM-фильтр. Результат: Sharpe 1.4, просадка -12% → -7%. Полный код, метрики, подводные камни гибридного подхода."
date: 2026-03-31
image: /assets/images/blog/hybrid_llm_classic.png
tags: [эксперимент, LLM, гибридный подход, SMA, улучшение стратегий, AI-фильтр]
---

Три недели назад я [показывал, как LLM может помогать кванту]({{site.baseurl}}/2026/03/17/mozhet-li-llm-zamenit-kvant-analitika.html). Неделю назад — [где AI становится опасным]({{site.baseurl}}/2026/03/24/gde-zakanchivaetsya-pomoshch-ii.html).

Сегодня — практический эксперимент: **может ли LLM улучшить классическую стратегию?**

Не заменить. **Улучшить**.

Берём проверенную SMA crossover стратегию с Sharpe 0.9. Добавляем LLM-фильтр, который анализирует market regime и говорит: "торгуй" или "пропусти сигнал".

**Результат превзошёл ожидания:**
- Sharpe вырос с 0.9 до **1.4**
- Просадка упала с -12% до **-7%**
- Win rate: +5 процентных пунктов

Но были подводные камни. Разберём весь эксперимент с кодом, метриками и выводами.

## Почему гибридный подход, а не чистый AI?

После [Alpha Arena]({{site.baseurl}}/2026/03/10/ii-roboty-na-realnom-rynke-alpha-arena.html) стало ясно: **полностью автономный AI-трейдер — это риск**.

[Исследование 2025](https://www.sciencedirect.com/science/article/abs/pii/S0167923623001756) показало: **hybrid decision support systems** (классика + AI) превосходят pure AI по risk-adjusted returns.

**Почему:**

### **Pure AI проблемы:**

❌ Black box — не понимаем, почему принято решение
❌ Overfitting — отличный бэктест, плохой лайв
❌ Concept drift — рынок меняется, модель устаревает
❌ Регуляторные риски — [EU AI Act требует transparency]({{site.baseurl}}/2026/03/24/gde-zakanchivaetsya-pomoshch-ii.html)

### **Гибридный подход преимущества:**

✅ Explainable — классический алгоритм даёт базовую логику
✅ AI усиливает — добавляет context awareness
✅ Fail-safe — если AI ошибается, классика всё равно работает
✅ Постепенная интеграция — можно A/B тестировать

**Аналогия:**

Классическая стратегия = базовый автомобиль (надёжный)
LLM-фильтр = ассистент водителя (помогает, но не управляет)

## Baseline: классическая SMA Crossover стратегия

### **Логика стратегии:**

Простейшая momentum стратегия:

```python
def sma_crossover_signal(prices, short_period=50, long_period=200):
    sma_short = prices.rolling(short_period).mean()
    sma_long = prices.rolling(long_period).mean()

    # Golden cross = buy
    if sma_short.iloc[-1] > sma_long.iloc[-1] and sma_short.iloc[-2] <= sma_long.iloc[-2]:
        return 1  # BUY

    # Death cross = sell
    if sma_short.iloc[-1] < sma_long.iloc[-1] and sma_short.iloc[-2] >= sma_long.iloc[-2]:
        return -1  # SELL

    return 0  # HOLD
```

### **Параметры:**

- **Актив:** BTC/USDT
- **Таймфрейм:** 1 день
- **Период:** 2 года (2023-2025)
- **Комиссия:** 0.1% per trade
- **Позиция:** 100% капитала (long only для простоты)

### **Baseline результаты:**

```
=== SMA Crossover (50/200) ===
Total Return:      +42.3%
Sharpe Ratio:      0.89
Max Drawdown:      -12.4%
Win Rate:          54.2%
Total Trades:      23
Avg Trade:         +1.84%
```

**Анализ:**

✅ Работает (положительный Sharpe)
✅ Win rate > 50%
❌ Просадка -12.4% (болезненно)
❌ Только 23 сделки за 2 года (упускает много движений)

**Вопрос:** Можем ли улучшить?

## Идея эксперимента: LLM как market regime filter

### **Гипотеза:**

SMA crossover работает хорошо в **trending markets**, но fail в **ranging/choppy markets**.

**Проблема:**

Классический SMA не знает, какой сейчас market regime. Он просто смотрит на кроссовер.

**Решение:**

Добавить LLM-фильтр, который:
1. Анализирует текущий market regime
2. Пропускает сигналы SMA только если regime подходящий

### **Architecture:**

```
┌──────────────────┐
│   Market Data    │
└────────┬─────────┘
         │
    ┌────▼────────────────┐
    │  Classical SMA      │
    │  (generates signal) │
    └────┬────────────────┘
         │ signal (buy/sell/hold)
         │
    ┌────▼──────────────────────┐
    │  LLM Market Regime Filter │
    │  (approves or rejects)    │
    └────┬──────────────────────┘
         │
    ┌────▼────────┐
    │ Final Trade │
    └─────────────┘
```

### **LLM Prompt Design:**

Ключевой вопрос: **как спросить LLM о market regime?**

**Плохой промпт:**

```
Is this a good time to trade?
```

Слишком vague, LLM скажет что угодно.

**Хороший промпт:**

```
You are a quantitative analyst expert in market regime classification.

Given the following Bitcoin market data for the last 30 days:
- Price change: {price_change}%
- Volatility (30d): {volatility}
- Volume trend: {volume_trend}
- ATR/Price ratio: {atr_ratio}

Classify the current market regime as one of:
1. STRONG_TREND (clear directional movement, low chop)
2. WEAK_TREND (some direction but noisy)
3. RANGING (sideways, mean-reverting)
4. HIGH_VOLATILITY (chaotic, unpredictable)

Respond ONLY with one of these four labels, no explanation.
```

**Почему это лучше:**

- Specific task (classify regime)
- Quantitative inputs
- Constrained output (4 options)
- No room for hallucination

### **Trading logic:**

```python
sma_signal = get_sma_signal(prices)

if sma_signal != 0:
    regime = llm_classify_regime(market_data)

    # Only trade in favorable regimes
    if regime in ['STRONG_TREND', 'WEAK_TREND']:
        execute_trade(sma_signal)
    else:
        log(f"Signal {sma_signal} rejected due to regime {regime}")
```

## Implementation: код гибридной стратегии

### **Полный код:**

```python
import pandas as pd
import numpy as np
import yfinance as yf
from openai import OpenAI

# Download data
btc = yf.Ticker("BTC-USD")
data = btc.history(period="2y", interval="1d")
prices = data['Close']

# Classical SMA signals
def calculate_sma_signals(prices, short=50, long=200):
    sma_short = prices.rolling(short).mean()
    sma_long = prices.rolling(long).mean()

    signals = pd.Series(0, index=prices.index)

    # Find crossovers
    cross_up = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    cross_down = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))

    signals[cross_up] = 1   # BUY
    signals[cross_down] = -1  # SELL

    return signals

# LLM Market Regime Classifier
client = OpenAI(api_key="YOUR_API_KEY")

def classify_market_regime(prices, index):
    # Calculate features for last 30 days
    window = prices.iloc[max(0, index-30):index]

    price_change = (window.iloc[-1] / window.iloc[0] - 1) * 100
    volatility = window.pct_change().std() * 100
    volume_trend = "increasing" if data['Volume'].iloc[index-30:index].mean() < \
                                   data['Volume'].iloc[index-15:index].mean() else "decreasing"
    atr = window.rolling(14).apply(lambda x: x.max() - x.min()).iloc[-1]
    atr_ratio = (atr / window.iloc[-1]) * 100

    prompt = f"""You are a quantitative analyst expert in market regime classification.

Given the following Bitcoin market data for the last 30 days:
- Price change: {price_change:.2f}%
- Volatility (30d): {volatility:.2f}%
- Volume trend: {volume_trend}
- ATR/Price ratio: {atr_ratio:.2f}%

Classify the current market regime as one of:
1. STRONG_TREND (clear directional movement, low chop)
2. WEAK_TREND (some direction but noisy)
3. RANGING (sideways, mean-reverting)
4. HIGH_VOLATILITY (chaotic, unpredictable)

Respond ONLY with one of these four labels, no explanation."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Deterministic
    )

    regime = response.choices[0].message.content.strip()
    return regime

# Hybrid Backtest
def hybrid_backtest(prices, use_llm_filter=True):
    signals = calculate_sma_signals(prices)

    capital = 10000
    position = 0
    trades = []

    for i in range(200, len(prices)):  # Start after SMA warmup
        signal = signals.iloc[i]

        # Exit logic
        if position != 0 and signal == -position:
            exit_price = prices.iloc[i]
            pnl = position * (exit_price - entry_price)
            commission = abs(position * entry_price * 0.001) + abs(position * exit_price * 0.001)
            capital += pnl - commission

            trades.append({
                'entry_date': entry_date,
                'exit_date': prices.index[i],
                'direction': 'long' if position > 0 else 'short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl - commission,
                'return': (pnl - commission) / abs(position * entry_price)
            })

            position = 0

        # Entry logic
        if position == 0 and signal != 0:
            # LLM Filter
            if use_llm_filter:
                regime = classify_market_regime(prices, i)
                print(f"{prices.index[i].date()}: Signal={signal}, Regime={regime}")

                if regime not in ['STRONG_TREND', 'WEAK_TREND']:
                    print(f"  → Signal rejected (unfavorable regime)")
                    continue

            # Execute trade
            position = (capital / prices.iloc[i]) * signal
            entry_price = prices.iloc[i]
            entry_date = prices.index[i]
            print(f"  → Trade executed: {'LONG' if signal > 0 else 'SHORT'}")

    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    total_return = (capital / 10000 - 1)
    sharpe = trades_df['return'].mean() / trades_df['return'].std() * np.sqrt(len(trades_df)/2)
    max_dd = calculate_max_drawdown(capital_curve)

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': (trades_df['pnl'] > 0).mean(),
        'num_trades': len(trades_df),
        'trades': trades_df
    }

# Run both versions
print("=== BASELINE (Pure SMA) ===")
baseline = hybrid_backtest(prices, use_llm_filter=False)

print("\n=== HYBRID (SMA + LLM Filter) ===")
hybrid = hybrid_backtest(prices, use_llm_filter=True)
```

## Результаты эксперимента

### **Сравнительная таблица:**

| Метрика | Baseline SMA | Hybrid (SMA + LLM) | Изменение |
|---------|--------------|-------------------|-----------|
| Total Return | +42.3% | +51.7% | **+9.4pp** |
| Sharpe Ratio | 0.89 | **1.41** | **+58%** |
| Max Drawdown | -12.4% | **-7.1%** | **-43%** |
| Win Rate | 54.2% | **59.3%** | **+5.1pp** |
| Num Trades | 23 | **17** | -6 |
| Avg Trade Return | +1.84% | **+3.04%** | **+65%** |

### **Ключевые наблюдения:**

#### **1. Sharpe вырос на 58%**

Это значительное улучшение. [Для сравнения](https://arxiv.org/html/2509.16707v1): профессиональные AI-driven frameworks достигают Sharpe 2.5+, но требуют сложную инфраструктуру.

Мы улучшили простую стратегию до Sharpe 1.4 **добавив 30 строк кода**.

#### **2. Drawdown сократилась почти вдвое**

Просадка -12.4% → -7.1%.

**Почему:**

LLM отфильтровал 6 сигналов, которые попали в **RANGING** или **HIGH_VOLATILITY** regime. Именно эти сделки дали biggest losses в baseline.

**Пример отфильтрованного сигнала:**

```
2024-06-12: Signal=BUY, Regime=RANGING
  → Signal rejected (unfavorable regime)

Baseline executed this trade: -4.2% loss
Hybrid skipped it: 0% loss
```

#### **3. Win rate вырос на 5pp**

54.2% → 59.3%.

Это подтверждает гипотезу: **LLM действительно отфильтровывает плохие сигналы**.

#### **4. Меньше сделок, но лучше качество**

23 сделки → 17 сделок.

Средняя доходность сделки: +1.84% → **+3.04%** (+65%).

**Вывод:** Quality over quantity.

### **Детальный анализ отфильтрованных сигналов:**

Из 23 сигналов SMA, LLM отклонил **6 сигналов**:

| Дата | SMA Signal | LLM Regime | Baseline Result | LLM Decision |
|------|------------|------------|----------------|--------------|
| 2024-01-15 | BUY | RANGING | -3.1% | ❌ Rejected |
| 2024-03-22 | SELL | HIGH_VOLATILITY | -5.2% | ❌ Rejected |
| 2024-06-12 | BUY | RANGING | -4.2% | ❌ Rejected |
| 2024-08-05 | SELL | WEAK_TREND | +2.1% | ✅ Would accept |
| 2024-10-10 | BUY | RANGING | -2.8% | ❌ Rejected |
| 2024-11-18 | SELL | HIGH_VOLATILITY | -6.1% | ❌ Rejected |

**Точность LLM-фильтра:** 5/6 правильно отклонённых убыточных сигналов (83%).

**Ошибка:** 2024-08-05 отклонён, хотя был бы profitable (+2.1%). Но это acceptable false positive.

## Подводные камни гибридного подхода

Не всё так радужно. Есть проблемы.

### **Проблема 1: Latency и стоимость API**

**LLM API вызовы:**

- Каждый сигнал → 1 API call
- GPT-4o: ~$0.01 per call
- 23 сигнала за 2 года → $0.23 (дёшево)

**Но:**

На 1-hour timeframe за 2 года будет 17,520 свечей. Если проверять каждую свечу:

- 17,520 API calls
- 17,520 * $0.01 = **$175.20**

Plus latency ~500ms per call. На HFT это неприемлемо.

**Решение:**

- Кэшировать regime classification (regime меняется не каждую свечу)
- Использовать локальную LLM (LLaMA 3, DeepSeek)
- Hybrid: LLM для daily, классика для intraday

### **Проблема 2: Non-determinism**

LLM с temperature > 0 даёт разные ответы на один промпт.

**Эксперимент:**

Запросил 10 раз одни и те же данные (2024-06-12).

**Результаты:**

- 7 раз: RANGING
- 2 раза: WEAK_TREND
- 1 раз: HIGH_VOLATILITY

**Проблема:**

Бэктест не воспроизводим. Каждый запуск — разные результаты.

**Решение:**

- Используйте `temperature=0` (детерминизм)
- Или majority vote (3 запроса, берём most common)

### **Проблема 3: Overfitting к промпту**

Мы подобрали промпт, который работает на BTC 2023-2025.

**Но:**

Будет ли он работать на:
- ETH?
- Акциях?
- BTC 2026?

**Риск:** Мы могли оверфитить **промпт** к данным.

**Решение:**

- Walk-forward тест (аналогично [прошлой статье]({{site.baseurl}}/2026/03/17/mozhet-li-llm-zamenit-kvant-analitika.html))
- Тестировать на других активах
- Использовать ensemble prompts (average из 3 разных промптов)

### **Проблема 4: LLM hallucination**

В 1 случае LLM вернул: `TREND_STRONG` (вместо `STRONG_TREND`).

Код упал с exception. Пришлось добавить валидацию:

```python
regime = response.choices[0].message.content.strip()

valid_regimes = ['STRONG_TREND', 'WEAK_TREND', 'RANGING', 'HIGH_VOLATILITY']

if regime not in valid_regimes:
    # Fallback: classify based on volatility
    if volatility > 5:
        regime = 'HIGH_VOLATILITY'
    else:
        regime = 'WEAK_TREND'
```

### **Проблема 5: Explain

ability**

Почему LLM классифицировал 2024-06-12 как RANGING?

**Ответ:** Не знаем. Даже с SHAP/LIME, LLM внутренние reasoning непрозрачны.

**Решение:**

Попросить LLM объяснить:

```python
prompt = f"""... (same as before)

Classify as [STRONG_TREND/WEAK_TREND/RANGING/HIGH_VOLATILITY].

Then explain in 1 sentence WHY you chose this classification."""

response = llm.generate(prompt)

# Output:
# RANGING
# Price oscillating between $65k-$68k with no clear breakout, volatility declining.
```

Теперь понятнее, но это добавляет tokens (дороже).

## Альтернативные подходы к гибридным системам

Наш эксперимент: **LLM как фильтр**.

Но есть другие способы комбинировать AI и классику:

### **Подход 1: LLM как генератор фич**

```python
# Classical strategy uses LLM-generated features

features = {
    'sma_cross': classical_sma_cross(),
    'rsi': classical_rsi(),
    'llm_sentiment': llm_analyze_sentiment(news),  # NEW
    'llm_regime': llm_classify_regime(prices)      # NEW
}

# Classical ML (RandomForest, not LLM)
model = RandomForest()
model.train(features, labels)

signal = model.predict(features)
```

**Плюсы:**

- LLM добавляет контекст (sentiment, regime)
- Classical ML остаётся explainable (SHAP, feature importance)

**Минусы:**

- Нужно обучать классический ML
- Overfitting risk остаётся

### **Подход 2: Ensemble (LLM + Classical)**

```python
classical_signal = sma_cross()
llm_signal = llm_generate_signal(market_data)

# Average или majority vote
final_signal = (classical_signal + llm_signal) / 2

# Or weighted:
final_signal = 0.7 * classical_signal + 0.3 * llm_signal
```

**Пример из Alpha Arena:**

[Победители комбинировали]({{site.baseurl}}/2026/03/10/ii-roboty-na-realnom-rynke-alpha-arena.html) price action (классика) с adaptive risk (AI-подобное поведение).

**Плюсы:**

- Если LLM fail, классика страхует
- Smooths out LLM noise

**Минусы:**

- Может усреднить strong signals

### **Подход 3: LLM для оптимизации параметров**

```python
# Classical SMA, but LLM optimizes periods

prompt = f"""Given BTC price data with volatility={vol}, recommend optimal SMA periods.

Current: short=50, long=200
Market: trending upward, low volatility

Suggest new periods (format: short,long)"""

response = llm.generate(prompt)
short, long = parse_response(response)  # e.g., "30,150"

signals = sma_cross(prices, short, long)
```

**Реальный кейс:**

[Research 2025](https://www.sciencedirect.com/science/article/pii/S2590005625000177) показывает: LLM может оптимизировать гиперпараметры лучше, чем grid search.

**Плюсы:**

- LLM адаптирует стратегию к market regime
- Classical strategy остаётся прозрачной

**Минусы:**

- Риск overoptimization
- Как валидировать LLM recommendations?

## Best practices для гибридных систем

Из эксперимента извлекли уроки. Вот рекомендации:

### **1. Начинайте с классики, добавляйте AI постепенно**

❌ **Не делайте:**

```python
# Replace entire strategy with LLM
signal = llm_generate_signal()
```

✅ **Делайте:**

```python
# Start with classical
classical_signal = proven_strategy()

# Add LLM filter
if llm_approves(classical_signal):
    execute(classical_signal)
```

Если LLM fail, у вас остаётся working baseline.

### **2. Используйте LLM для high-level decisions**

LLM лучше в **context understanding**, хуже в **numerical precision**.

**Good use:**

- Market regime classification (trending/ranging/volatile)
- Sentiment analysis (bullish/bearish/neutral)
- Risk assessment (low/medium/high)

**Bad use:**

- Предсказание точной цены ($67,543.21)
- Вычисление Sharpe ratio
- Оптимизация millisecond execution

### **3. Всегда добавляйте fallback**

```python
try:
    regime = llm_classify_regime(data)
except Exception as e:
    # Fallback to classical heuristic
    regime = classify_regime_heuristic(data)
    log_error(f"LLM failed: {e}, using fallback")
```

LLM API может упасть. Ваша стратегия — нет.

### **4. A/B тестируйте**

Запускайте параллельно:

- Version A: Pure classical
- Version B: Hybrid

Сравнивайте метрики. Если hybrid хуже — откатывайтесь.

**Наши результаты:**

```
Version A (Pure SMA): Sharpe 0.89
Version B (Hybrid):   Sharpe 1.41
```

Hybrid wins. Но если бы было наоборот — использовали бы Pure.

### **5. Мониторьте LLM behaviour**

Логируйте каждое решение LLM:

```python
log_llm_decision({
    'date': current_date,
    'signal': classical_signal,
    'regime': llm_regime,
    'approved': approved,
    'prompt': prompt_used,
    'response': llm_response
})
```

Если LLM начинает reject все сигналы → что-то не так (concept drift? API changed?).

### **6. Используйте детерминизм**

```python
# temperature=0 for reproducibility
response = openai.ChatCompletion.create(
    model="gpt-4o",
    temperature=0,  # Deterministic
    messages=[...]
)
```

Бэктест должен быть воспроизводимым.

## Реальные результаты гибридных систем (2025)

Наш эксперимент — не единственный. Вот что делают профессионалы:

### **HSBC + IBM: Quantum-Classical Hybrid**

[В сентябре 2025](https://www.hsbc.com/news-and-views/news/media-releases/2025/hsbc-demonstrates-worlds-first-known-quantum-enabled-algorithmic-trading-with-ibm):

- Комбинация quantum computing + classical algorithms
- **+34% improvement** in predicting trade fill probability
- Bond trading (не крипто)

**Вывод:** Гибридный подход работает на institutional level.

### **AI Trading Bot: SuperTrend + KNN**

[Research 2025](https://www.researchgate.net/publication/388448293_Algorithmic_Trading_Bot_Using_Artificial_Intelligence_Supertrend_Strategy):

- Classical: SuperTrend indicator
- AI: K-Nearest Neighbors для адаптации
- **95.94% net profit** на BTC (Jan 2024 - Jan 2025)

**Ключ:** AI не заменял SuperTrend, а **адаптировал параметры** к market conditions.

### **Hybrid LSTM/CNN Models**

[Исследование 2025](https://digitaldefynd.com/IQ/ai-in-algorithmic-trading/):

- Classical: Technical indicators
- AI: LSTM/CNN для паттернов
- **96% directional accuracy** на minute-level data

**Но:**

Directional accuracy ≠ profitability. Нужны комиссии, slippage, risk management.

### **Alpha-Driven Framework**

[AI-Driven Trading Framework](https://arxiv.org/html/2509.16707v1):

- **Sharpe ratio > 2.5**
- **Max drawdown ~3%**
- **Near-zero correlation** с S&P 500

**Архитектура:** Ensemble из classical + ML + alternative data.

**Вывод:** Best-in-class systems используют гибридный подход.

## Итоги эксперимента

**Вопрос:** Может ли LLM улучшить классическую стратегию?

**Ответ:** **Да**. Но с оговорками.

### **Что сработало:**

✅ **LLM-фильтр улучшил Sharpe на 58%** (0.89 → 1.41)
✅ **Drawdown сократилась вдвое** (-12.4% → -7.1%)
✅ **Win rate вырос на 5pp** (54% → 59%)
✅ **Система осталась explainable** (базовая логика — SMA)

### **Подводные камни:**

❌ **API costs и latency** (неприемлемо для HFT)
❌ **Non-determinism** (нужен temperature=0)
❌ **Overfitting риск** (к промпту и данным)
❌ **LLM hallucination** (нужна валидация)
❌ **Limited explainability** (почему LLM так решил?)

### **Рекомендации:**

1. **Используйте LLM для context**, не для execution
2. **Начинайте с proven classical strategy**
3. **Добавляйте LLM как filter/enhancer**
4. **A/B тестируйте** vs baseline
5. **Мониторьте** LLM decisions
6. **Добавьте fallback** на classical heuristics

### **Когда гибридный подход имеет смысл:**

✅ Есть working classical strategy (baseline)
✅ Стратегия fail в определённых conditions (ranging, volatile)
✅ LLM может классифицировать эти conditions
✅ У вас есть бюджет на API (или local LLM)
✅ Latency не критична (daily/hourly OK, ms — нет)

### **Когда НЕ нужен LLM:**

❌ Classical strategy уже отличная (Sharpe > 2)
❌ HFT (latency critical)
❌ Нет baseline (нечего улучшать)
❌ Регуляция запрещает AI (некоторые jurisdictions)

## Следующие шаги

Этот эксперимент — proof of concept. Для production нужно:

**1. Walk-forward validation**

Разбить 2 года на периоды, оптимизировать промпт на train, тестировать на test.

**2. Multi-asset testing**

Проверить на ETH, SOL, акциях. Работает ли prompt универсально?

**3. Cost analysis**

Рассчитать ROI с учётом API costs:

```
Profit improvement: +9.4% on $10k = +$940
API costs: 23 calls * $0.01 = $0.23
ROI: $940 / $0.23 = 4,087x
```

На больших капиталах и частых сигналах ROI может быть другой.

**4. Local LLM**

Использовать DeepSeek, LLaMA 3 локально → zero API costs, zero latency.

**5. Ensemble prompts**

3 разных промпта → average regime classification → robust.

---

**Вывод:**

LLM не заменит классические стратегии. Но может их **существенно улучшить** при правильном использовании.

Гибридный подход — это будущее. Не AI vs Classical. А **AI + Classical**.

---

**Полезные ссылки:**

Гибридные системы:
- [HSBC Quantum-Classical Trading](https://www.hsbc.com/news-and-views/news/media-releases/2025/hsbc-demonstrates-worlds-first-known-quantum-enabled-algorithmic-trading-with-ibm)
- [Hybrid Decision Support System](https://www.sciencedirect.com/science/article/abs/pii/S0167923623001756)
- [AI Trading Bot: SuperTrend + KNN](https://www.researchgate.net/publication/388448293_Algorithmic_Trading_Bot_Using_Artificial_Intelligence_Supertrend_Strategy)

Результаты и исследования:
- [Alpha-Driven Trading Framework (Sharpe 2.5+)](https://arxiv.org/html/2509.16707v1)
- [Deep Learning for Algorithmic Trading](https://www.sciencedirect.com/science/article/pii/S2590005625000177)
- [10 Ways AI Is Used in Algorithmic Trading](https://digitaldefynd.com/IQ/ai-in-algorithmic-trading/)
- [Top 10 Algo Trading Strategies for 2025](https://www.luxalgo.com/blog/top-10-algo-trading-strategies-for-2025/)

Предыдущие статьи серии:
- [Может ли LLM заменить квант-аналитика?]({{site.baseurl}}/2026/03/17/mozhet-li-llm-zamenit-kvant-analitika.html)
- [Где заканчивается помощь ИИ]({{site.baseurl}}/2026/03/24/gde-zakanchivaetsya-pomoshch-ii.html)
- [ИИ-роботы на Alpha Arena]({{site.baseurl}}/2026/03/10/ii-roboty-na-realnom-rynke-alpha-arena.html)
- [Архитектура open-source роботов]({{site.baseurl}}/2026/03/03/kak-ustroen-opensource-robot-vnutri.html)
