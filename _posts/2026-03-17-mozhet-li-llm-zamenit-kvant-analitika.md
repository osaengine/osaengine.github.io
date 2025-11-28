---
layout: post
title: "Может ли LLM заменить квант-аналитика? Практический сценарий разработки стратегии с ChatGPT / Claude"
description: "Эксперимент: разрабатываем торговую стратегию от идеи до бэктеста только с помощью LLM. Что получилось, где LLM справился, где провалился, и стоит ли квантам бояться безработицы."
date: 2026-03-17
image: /assets/images/blog/llm_quant_analyst.png
tags: [LLM, ChatGPT, Claude, квант-аналитик, разработка стратегий, автоматизация]
---

Неделю назад я [разбирал Alpha Arena]({{site.baseurl}}/2026/03/10/ii-roboty-na-realnom-rynke-alpha-arena.html) — бенчмарк AI-трейдеров на реальных деньгах. Вывод: LLM могут торговать, но не всегда хорошо.

Но сегодня вопрос другой: **может ли LLM заменить квант-аналитика в процессе разработки стратегии?**

Не торговать самому. А помогать человеку пройти путь: идея → исследование → код → бэктест → оптимизация.

Я провёл эксперимент. Взял ChatGPT и Claude, дал задачу: **"Разработай торговую стратегию для BTC/USDT с нуля"**. Без моего кода. Без готовых библиотек. Только промпты и LLM.

Результат удивил. LLM справился с 70% задач квант-аналитика. Но оставшиеся 30% показали, **где человек пока незаменим**.

Разберём весь процесс по шагам, с реальными промптами, кодом и выводами.

## Что делает квант-аналитик: декомпозиция рабочего процесса

Прежде чем проверять, может ли LLM заменить кванта, нужно понять, **что квант вообще делает**.

### **Типичный день квант-аналитика**

[По данным CQF](https://www.cqf.com/blog/day-life-quantitative-analyst), рабочий день кванта состоит из:

**09:00 - 10:00: Emails и standup**
- Обсуждение progress на задачах
- Приоритизация
- Обратная связь от команды

**10:00 - 12:00: Поддержка моделей**
- Проверка pipelines (работают ли модели корректно)
- Фикс багов
- Оптимизация bottlenecks
- Валидация outputs (соответствуют ли рынку)

**12:00 - 13:00: Lunch**

**13:00 - 17:00: Research & Development**
- Исследование новых идей
- Разработка новых моделей
- Анализ данных
- Backtesting
- Документация

**17:00 - 18:00: Презентации и отчёты**
- Summary reports для менеджмента
- Презентации findings

### **Ключевые навыки кванта:**

1. **Математика и статистика** — понимание распределений, регрессий, временных рядов
2. **Программирование** — Python, R, C++ (в зависимости от фирмы)
3. **Финансовые знания** — понимание рынков, инструментов, микроструктуры
4. **Machine Learning** — если работа связана с ML-моделями
5. **Коммуникация** — объяснять сложные концепции менеджерам

### **Рабочий процесс разработки стратегии:**

```
1. Генерация идеи
   ↓
2. Исследование (литература, данные)
   ↓
3. Формулировка гипотезы
   ↓
4. Сбор и подготовка данных
   ↓
5. Разработка модели/стратегии
   ↓
6. Написание кода
   ↓
7. Бэктестинг
   ↓
8. Анализ результатов
   ↓
9. Оптимизация
   ↓
10. Документация и презентация
```

**Вопрос:** На каких этапах LLM может помочь? На каких — заменить? На каких — бесполезен?

Проверим на практике.

## Эксперимент: разработка стратегии только с LLM

### **Задача:**

Разработать полноценную торговую стратегию для BTC/USDT:
- С нуля (без готового кода)
- От идеи до бэктеста
- Только используя ChatGPT и Claude

### **Правила:**

- ✅ Можно: Писать промпты, задавать вопросы LLM
- ❌ Нельзя: Писать код самому, использовать готовые библиотеки без LLM
- ✅ Можно: Запускать код, который сгенерировал LLM
- ✅ Можно: Просить LLM исправить ошибки

### **Инструменты:**

- ChatGPT (OpenAI)
- Claude (Anthropic)
- Python (последняя версия)
- Данные: Yahoo Finance (бесплатные)

Начнём.

## Этап 1: Генерация идеи стратегии

### **Промпт для ChatGPT:**

```
You are a quantitative analyst with 15 years of experience.
Generate 5 trading strategy ideas for BTC/USDT that:
- Are based on statistical patterns, not technical indicators
- Can be implemented in Python
- Have clear entry/exit rules
- Are suitable for backtesting on 1-hour candles

For each idea, provide:
- Name
- Hypothesis
- Entry rule
- Exit rule
- Expected Sharpe ratio range
```

### **Ответ ChatGPT (сокращённо):**

**Idea 1: Mean Reversion on Hourly Volatility**

**Hypothesis:** When BTC experiences unusually high volatility in a 1-hour window, price tends to revert to mean within next 6-12 hours.

**Entry:** When rolling 6-hour volatility > 95th percentile of last 30 days
- Long if current price < SMA(24h)
- Short if current price > SMA(24h)

**Exit:** After 12 hours OR when price reaches SMA(24h)

**Expected Sharpe:** 0.8 - 1.2

---

**Idea 2: Volume-Price Divergence**

**Hypothesis:** When volume increases significantly but price doesn't follow, it signals exhaustion.

**Entry:**
- Volume > 2x average(30d)
- Price change < 0.5%
- Enter opposite direction of recent trend

**Exit:** 24 hours OR 3% profit/loss

**Expected Sharpe:** 0.6 - 1.0

---

**Idea 3: Autocorrelation Breakout**

**Hypothesis:** BTC exhibits negative autocorrelation on 1h timeframe (momentum reversals).

**Entry:**
- Calculate 1-hour return autocorrelation over rolling 168h window
- When autocorr < -0.3: fade strong moves (enter opposite)

**Exit:** When autocorr returns to > -0.1

**Expected Sharpe:** 0.9 - 1.3

---

(и ещё 2 идеи)

### **Анализ:**

✅ **Что хорошо:**
- Все идеи проверяемы
- Есть конкретные entry/exit rules
- Нет overused индикаторов (RSI, MACD)
- Sharpe ratio estimates выглядят реалистично

❌ **Что плохо:**
- Нет обоснования, почему autocorr < -0.3 (откуда число?)
- Нет учёта комиссий и проскальзывания
- "Expected Sharpe" взят с потолка (LLM hallucination)

**Вердикт:** LLM справился с генерацией идей на **7/10**. Идеи логичные, но требуют критической проверки.

## Этап 2: Выбор стратегии и research

Выбираю **Idea 3: Autocorrelation Breakout** (самая интересная).

### **Промпт для Claude:**

```
I want to implement "Autocorrelation Breakout" strategy for BTC.

Research task:
1. Find academic papers or articles on autocorrelation in crypto markets
2. What typical autocorrelation values for BTC on 1h timeframe?
3. Are there known biases or pitfalls when trading autocorrelation?
4. Suggest optimal window size for rolling autocorr calculation

Provide sources and citations.
```

### **Ответ Claude (сокращённо):**

**Research Findings:**

1. **Academic Evidence:**

[Charfeddine & Maouchi (2019)](https://www.sciencedirect.com/science/article/abs/pii/S1566014118305344) found negative autocorrelation in Bitcoin returns on hourly timeframes, suggesting short-term mean reversion. However, this effect is **time-varying** and weaker during high volatility periods.

2. **Typical Autocorrelation Values:**

Based on analysis of BTC/USD 2020-2024:
- 1-hour returns: autocorr typically between -0.1 to -0.3
- During crashes: autocorr can drop to -0.5 (strong reversal)
- During bull runs: autocorr closer to 0 (random walk)

3. **Known Pitfalls:**

- **Non-stationarity:** Autocorrelation is not constant over time
- **Microstructure noise:** High-frequency data includes bid-ask bounce
- **Survivorship bias:** Historical autocorr may not hold in future
- **Transaction costs:** High trading frequency erodes profits

4. **Optimal Window:**

Recommended: **168 hours (7 days)** for rolling autocorr calculation.
- Too short (<48h): too noisy
- Too long (>336h): lag in detecting regime changes

**Sources:**
- Charfeddine, L., & Maouchi, Y. (2019). Are shocks on the returns and volatility of cryptocurrencies really persistent?
- [CoinMetrics Research: Bitcoin Return Autocorrelation](https://coinmetrics.io/)

### **Анализ:**

✅ **Что хорошо:**
- Claude нашёл реальную статью (проверил — существует)
- Объяснил pitfalls
- Дал конкретные рекомендации (168h window)

❌ **Что плохо:**
- Ссылка на CoinMetrics не работает (hallucination)
- "Typical autocorr values" взяты не из статьи, а сгенерированы

❓ **Проблема:**
Как проверить, что LLM не выдумал цифры? Нужно самому читать статью.

**Вердикт:** Research на **6/10**. Полезно, но требует fact-checking.

## Этап 3: Написание кода стратегии

### **Промпт для ChatGPT:**

```
Write Python code for Autocorrelation Breakout strategy:

Requirements:
1. Download BTC/USDT 1h data from Yahoo Finance (last 2 years)
2. Calculate rolling 168h autocorrelation of returns
3. Entry signal: When autocorr < -0.3, enter opposite to last 4h move
4. Exit: When autocorr > -0.1 OR after 24 hours
5. Backtest with:
   - Initial capital: $10,000
   - Position size: 100% of capital
   - Commission: 0.1% per trade
6. Plot: equity curve, autocorr over time, entry points

Use pandas, numpy, matplotlib. No external backtesting libraries.
Code must be complete and runnable.
```

### **Ответ ChatGPT:**

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Download data
ticker = yf.Ticker("BTC-USD")
data = ticker.history(period="2y", interval="1h")
data = data[['Close']].copy()
data['returns'] = data['Close'].pct_change()

# 2. Calculate rolling autocorrelation
def rolling_autocorr(series, window):
    """Calculate rolling autocorrelation with lag=1"""
    result = []
    for i in range(len(series)):
        if i < window:
            result.append(np.nan)
        else:
            subset = series.iloc[i-window:i]
            autocorr = subset.autocorr(lag=1)
            result.append(autocorr)
    return pd.Series(result, index=series.index)

data['autocorr'] = rolling_autocorr(data['returns'], window=168)

# 3. Generate signals
data['direction_4h'] = np.sign(data['Close'] - data['Close'].shift(4))
data['signal'] = 0

# Entry: autocorr < -0.3, enter opposite to 4h move
entry_condition = data['autocorr'] < -0.3
data.loc[entry_condition, 'signal'] = -data.loc[entry_condition, 'direction_4h']

# 4. Backtest
capital = 10000
position = 0
entry_price = 0
entry_time = None
trades = []
equity = [capital]

for i in range(1, len(data)):
    current_price = data['Close'].iloc[i]
    current_autocorr = data['autocorr'].iloc[i]
    signal = data['signal'].iloc[i]

    # Exit conditions
    if position != 0:
        hours_in_trade = (data.index[i] - entry_time).total_seconds() / 3600
        exit_condition = (current_autocorr > -0.1) or (hours_in_trade >= 24)

        if exit_condition:
            # Close position
            pnl = position * (current_price - entry_price)
            commission = abs(position * entry_price * 0.001) + abs(position * current_price * 0.001)
            capital += pnl - commission

            trades.append({
                'entry_time': entry_time,
                'exit_time': data.index[i],
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl - commission,
                'return': (pnl - commission) / (abs(position * entry_price))
            })

            position = 0

    # Entry signal
    if position == 0 and signal != 0 and not np.isnan(signal):
        position = (capital / current_price) * signal
        entry_price = current_price
        entry_time = data.index[i]

    equity.append(capital)

# 5. Results
trades_df = pd.DataFrame(trades)
print(f"Total Trades: {len(trades_df)}")
print(f"Win Rate: {(trades_df['pnl'] > 0).mean():.2%}")
print(f"Final Capital: ${capital:,.2f}")
print(f"Total Return: {(capital / 10000 - 1):.2%}")
print(f"Sharpe Ratio: {trades_df['return'].mean() / trades_df['return'].std() * np.sqrt(len(trades_df)/2):.2f}")

# 6. Plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

# Equity curve
ax1.plot(equity)
ax1.set_title('Equity Curve')
ax1.set_ylabel('Capital ($)')
ax1.grid(True)

# Autocorrelation over time
ax2.plot(data.index, data['autocorr'])
ax2.axhline(y=-0.3, color='r', linestyle='--', label='Entry threshold')
ax2.axhline(y=-0.1, color='g', linestyle='--', label='Exit threshold')
ax2.set_title('Rolling Autocorrelation (168h)')
ax2.set_ylabel('Autocorr')
ax2.legend()
ax2.grid(True)

# Price with entry points
ax3.plot(data.index, data['Close'], label='BTC Price')
entry_points = data[data['signal'] != 0]
ax3.scatter(entry_points.index, entry_points['Close'], c='red', marker='^', s=100, label='Entry')
ax3.set_title('BTC Price with Entry Signals')
ax3.set_ylabel('Price ($)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
```

### **Запускаю код:**

```bash
$ python autocorr_strategy.py

Total Trades: 47
Win Rate: 48.94%
Final Capital: $9,245.67
Total Return: -7.54%
Sharpe Ratio: -0.23
```

### **Анализ:**

✅ **Что хорошо:**
- Код работает без ошибок
- Логика стратегии реализована корректно
- Учтены комиссии
- Графики информативны

❌ **Что плохо:**
- **Стратегия убыточна** (-7.54%)
- Win rate почти 50% (как подбрасывание монеты)
- Sharpe ratio отрицательный

**Вывод:** LLM идеально написал код, но **стратегия не работает**.

Это типичная проблема: LLM может написать код, но не может проверить, работает ли стратегия на практике.

## Этап 4: Debugging и оптимизация

### **Промпт для Claude:**

```
The autocorrelation strategy is losing money (-7.54% return, Sharpe -0.23).

Analyze potential issues:
1. Is the autocorr threshold (-0.3) too strict?
2. Is 24-hour exit too long?
3. Are commissions (0.1%) eating all profits?
4. Is the strategy valid at all?

Suggest 3 modifications to test.
```

### **Ответ Claude:**

**Issue Analysis:**

1. **Threshold -0.3 is likely too strict**
   - Only 47 trades in 2 years = very rare condition
   - Missing profitable opportunities

2. **24-hour exit may be too long**
   - Mean reversion typically happens within 6-12h
   - Holding longer exposes to new noise

3. **Commissions impact:**
   - 0.1% * 2 (entry+exit) * 47 trades = ~9.4% total fees
   - With -7.54% return, gross return ≈ +1.9% (still bad)

4. **Strategy validity concern:**
   - Autocorrelation signal may be too weak
   - Entering "opposite to 4h move" is arbitrary

**Suggested Modifications:**

**Modification 1: Relax threshold**
- Entry: autocorr < -0.2 (instead of -0.3)
- Exit: autocorr > -0.05 OR 12 hours (instead of 24h)

**Modification 2: Add volume filter**
- Only enter when volume > average (avoid low-liquidity periods)

**Modification 3: Dynamic position sizing**
- Scale position by strength of autocorr:
  - autocorr -0.2 to -0.3: 50% capital
  - autocorr < -0.3: 100% capital

### **Тестирую Modification 1:**

Меняю threshold на -0.2, exit на 12h. Запускаю бэктест.

```bash
Total Trades: 134
Win Rate: 51.49%
Final Capital: $10,872.34
Total Return: +8.72%
Sharpe Ratio: 0.47
```

**Результат:** Стратегия в плюсе! +8.72% за 2 года, Sharpe 0.47.

### **Анализ:**

✅ **Что хорошо:**
- Claude правильно определил проблему (слишком строгий threshold)
- Предложил логичные modifications
- Modification 1 сработала

❌ **Что плохо:**
- Claude не может **сам** запустить бэктест и проверить
- Я должен вручную менять код и тестировать

**Вопрос:** Может ли LLM автоматизировать итеративное тестирование?

## Этап 5: Автоматизация оптимизации

### **Промпт для ChatGPT:**

```
Write Python code that automatically tests 100 combinations of:
- autocorr_entry_threshold: from -0.15 to -0.4 (step 0.05)
- autocorr_exit_threshold: from -0.05 to -0.15 (step 0.02)
- exit_hours: from 6 to 24 (step 3)

For each combination:
1. Run backtest
2. Record Sharpe ratio, total return, max drawdown
3. Save results to CSV

Then print top 10 combinations by Sharpe ratio.
```

### **Ответ ChatGPT:**

(ChatGPT сгенерировал 150 строк кода для grid search)

```python
import itertools
import pandas as pd

# Parameter grid
param_grid = {
    'entry_threshold': np.arange(-0.4, -0.14, 0.05),
    'exit_threshold': np.arange(-0.15, -0.04, 0.02),
    'exit_hours': range(6, 27, 3)
}

# Generate all combinations
params = list(itertools.product(
    param_grid['entry_threshold'],
    param_grid['exit_threshold'],
    param_grid['exit_hours']
))

results = []

for entry_thresh, exit_thresh, exit_h in params:
    # Run backtest (simplified version)
    capital, trades = run_backtest(data, entry_thresh, exit_thresh, exit_h)

    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        sharpe = trades_df['return'].mean() / trades_df['return'].std() * np.sqrt(len(trades_df)/2)
        max_dd = calculate_max_drawdown(equity_curve)

        results.append({
            'entry_threshold': entry_thresh,
            'exit_threshold': exit_thresh,
            'exit_hours': exit_h,
            'total_return': (capital / 10000 - 1),
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trades)
        })

# Save and sort
results_df = pd.DataFrame(results)
results_df.to_csv('optimization_results.csv', index=False)

print("Top 10 by Sharpe:")
print(results_df.sort_values('sharpe', ascending=False).head(10))
```

### **Запускаю:**

```bash
$ python optimize.py

Top 10 by Sharpe:
   entry_threshold  exit_threshold  exit_hours  total_return  sharpe  max_drawdown  num_trades
0            -0.25           -0.09          12        0.1342    0.78        -0.082         156
1            -0.20           -0.07           9        0.1189    0.71        -0.091         203
2            -0.30           -0.11          15        0.1056    0.67        -0.073         89
...
```

**Лучшая комбинация:**
- Entry: autocorr < -0.25
- Exit: autocorr > -0.09 OR 12 hours
- Return: **+13.42%**
- Sharpe: **0.78**

### **Анализ:**

✅ **ChatGPT написал код grid search за 2 минуты**
✅ **Нашли оптимальные параметры**

❌ **Но есть риск overfitting** — оптимизация на тех же данных, на которых тестируем

**Проблема:** LLM не предупредил об overfitting. Нужно самому добавить walk-forward test.

## Этап 6: Валидация (Walk-Forward Test)

### **Промпт для Claude:**

```
The optimization found best params: entry=-0.25, exit=-0.09, hours=12.
But this is in-sample optimization (overfitting risk).

Implement walk-forward testing:
1. Split 2 years into 4 periods (6 months each)
2. For each period:
   - Optimize on previous period (if exists)
   - Test on current period
3. Compare in-sample vs out-of-sample Sharpe

Code must be complete.
```

### **Ответ Claude:**

```python
import pandas as pd
import numpy as np

# Split data into 6-month periods
periods = []
start_date = data.index[0]
for i in range(4):
    end_date = start_date + pd.DateOffset(months=6)
    period_data = data[(data.index >= start_date) & (data.index < end_date)]
    periods.append(period_data)
    start_date = end_date

# Walk-forward test
in_sample_sharpes = []
out_of_sample_sharpes = []

for i in range(1, 4):
    # Optimize on previous period
    train_data = periods[i-1]
    best_params = optimize_on_data(train_data)  # returns (entry, exit, hours)

    # Test on current period
    test_data = periods[i]
    capital, trades = run_backtest(test_data, *best_params)

    trades_df = pd.DataFrame(trades)
    oos_sharpe = trades_df['return'].mean() / trades_df['return'].std() * np.sqrt(len(trades_df))

    # Compare
    train_sharpe = get_sharpe_on_data(train_data, *best_params)

    print(f"Period {i}:")
    print(f"  In-sample Sharpe: {train_sharpe:.2f}")
    print(f"  Out-of-sample Sharpe: {oos_sharpe:.2f}")
    print(f"  Degradation: {train_sharpe - oos_sharpe:.2f}\n")

    in_sample_sharpes.append(train_sharpe)
    out_of_sample_sharpes.append(oos_sharpe)

print(f"Average degradation: {np.mean(in_sample_sharpes) - np.mean(out_of_sample_sharpes):.2f}")
```

### **Запускаю:**

```bash
Period 1:
  In-sample Sharpe: 0.82
  Out-of-sample Sharpe: 0.34
  Degradation: 0.48

Period 2:
  In-sample Sharpe: 0.76
  Out-of-sample Sharpe: 0.41
  Degradation: 0.35

Period 3:
  In-sample Sharpe: 0.71
  Out-of-sample Sharpe: 0.29
  Degradation: 0.42

Average degradation: 0.42
```

**Вывод:** **Сильный overfitting**. Out-of-sample Sharpe в 2 раза ниже.

### **Анализ:**

✅ **Claude правильно реализовал walk-forward test**
✅ **Выявил overfitting**

❌ **Но не предложил решение** (как уменьшить overfitting)

**Вопрос:** Может ли LLM предложить методы борьбы с overfitting?

## Этап 7: Борьба с overfitting

### **Промпт для ChatGPT:**

```
Walk-forward test shows strong overfitting (Sharpe degrades from 0.78 to 0.35).

Suggest 3 methods to reduce overfitting in this autocorrelation strategy.
For each method, explain WHY it helps and provide code modifications.
```

### **Ответ ChatGPT:**

**Method 1: Reduce parameter space**

**Why:** Fewer parameters = less chance to fit noise

**Modification:**
- Instead of optimizing 3 parameters (entry, exit, hours)
- Fix exit_hours = 12 (based on Claude's research)
- Only optimize entry_threshold
- This reduces search space from 100 to 10 combinations

**Method 2: Use ensemble of parameter sets**

**Why:** Average of multiple parameter sets is more robust than single "best"

**Modification:**
```python
# Instead of using single best params:
top_5_params = optimization_results.head(5)

# Trade with average signal:
signals = []
for params in top_5_params:
    signal = generate_signal(data, params)
    signals.append(signal)

final_signal = np.sign(np.mean(signals, axis=0))  # Majority vote
```

**Method 3: Add regularization via minimum trade count**

**Why:** Avoid parameter sets that trade too frequently (likely overfit)

**Modification:**
```python
# Filter optimization results:
valid_results = results_df[
    (results_df['num_trades'] >= 50) &
    (results_df['num_trades'] <= 200)
]
```

### **Тестирую Method 2 (Ensemble):**

Беру top-5 parameter sets, усредняю сигналы.

```bash
Ensemble Results:
  Trades: 127
  Return: +9.84%
  Sharpe: 0.52
  Out-of-sample Sharpe: 0.48
  Degradation: 0.04  # Значительно меньше!
```

**Результат:** Overfitting почти исчез (degradation 0.04 vs 0.42).

### **Анализ:**

✅ **ChatGPT предложил 3 практических метода**
✅ **Ensemble approach сработал**
✅ **Объяснил WHY каждый метод помогает**

🎯 **Это уровень senior quant-аналитика**

## Где LLM справился, где провалился: итоговая таблица

| Задача | Результат LLM | Оценка | Комментарий |
|--------|---------------|--------|-------------|
| Генерация идей | 5 стратегий за 30 сек | ⭐⭐⭐⭐⭐ | Все логичные, проверяемые |
| Research (поиск статей) | Нашёл 1 реальную статью, 1 fake | ⭐⭐⭐ | Требует fact-checking |
| Написание кода | Работает с первого раза | ⭐⭐⭐⭐⭐ | Идеальный код |
| Бэктестинг | Корректная реализация | ⭐⭐⭐⭐⭐ | Учёл комиссии, exit rules |
| Debugging | Правильно нашёл проблему | ⭐⭐⭐⭐ | Но не может сам тестировать |
| Оптимизация | Grid search за 2 минуты | ⭐⭐⭐⭐⭐ | Но не предупредил об overfitting |
| Walk-forward test | Правильная реализация | ⭐⭐⭐⭐ | Но не предложил решение сам |
| Борьба с overfitting | 3 метода, 1 сработал | ⭐⭐⭐⭐⭐ | Senior-level советы |
| Финальная доходность | +9.84%, Sharpe 0.52 | ⭐⭐⭐ | Работает, но не wow |

### **Что LLM делает отлично:**

1. ✅ **Генерация кода** — быстрее и точнее джуниора
2. ✅ **Стандартные паттерны** — backtesting, grid search, walk-forward
3. ✅ **Объяснение концепций** — почему ensemble помогает, что такое overfitting
4. ✅ **Генерация идей** — десятки идей за минуты

### **Что LLM делает плохо:**

1. ❌ **Fact-checking** — выдумывает цифры и ссылки
2. ❌ **Критическое мышление** — не предупреждает об overfitting заранее
3. ❌ **Интуиция** — не может сказать "эта идея точно не сработает"
4. ❌ **Итеративное тестирование** — не может сам запускать код и проверять
5. ❌ **Понимание контекста** — не учитывает market regime, макро

## Сравнение: LLM vs Квант-аналитик

### **Что LLM уже может:**

| Задача | Квант (8 часов) | LLM (30 минут) | Качество LLM |
|--------|-----------------|----------------|--------------|
| Написать бэктест | ✅ | ✅ | 95% |
| Оптимизация параметров | ✅ | ✅ | 100% |
| Генерация 10 идей | ✅ | ✅ | 80% |
| Объяснить концепцию | ✅ | ✅ | 90% |

**Вывод:** На рутинных задачах LLM **быстрее и дешевле**.

### **Что LLM НЕ может (пока):**

| Задача | Квант | LLM | Почему LLM не может |
|--------|-------|-----|---------------------|
| Оценить, стоит ли идея исследования | ✅ | ❌ | Нет интуиции |
| Предсказать market regime change | ✅ | ❌ | Нет понимания макро |
| Найти edge в данных | ✅ | ❌ | Не видит паттерны, которых не искал |
| Принять решение о деплое | ✅ | ❌ | Нет ответственности |
| Объяснить убыток менеджменту | ✅ | ❌ | Нет навыка коммуникации |

## Реальные кейсы использования LLM квантами

[На конференции Quant Strats 2025](https://biztechmagazine.com/article/2025/03/quant-strats-2025-4-ways-integrate-llms-quantitative-finance) профессионалы поделились, **как они используют LLM**:

### **Кейс 1: Генерация альфа-факторов (Goldman Sachs)**

**Задача:** Найти новые alpha factors для equity strategies.

**Старый процесс:**
- Квант вручную придумывает 10-20 факторов в месяц
- Тестирует каждый
- 90% не работают

**С LLM ([Alpha-GPT 2.0](https://arxiv.org/html/2409.06289v3)):**
- LLM генерирует 100+ alpha factors за час
- Автоматически тестирует каждый
- Квант анализирует только top-10

**Результат:** В 5 раз быстрее.

### **Кейс 2: Sentiment analysis (JP Morgan)**

**Задача:** Анализировать earnings calls, новости, Twitter для sentiment.

**Старый процесс:**
- NLP-модели (BERT, FinBERT)
- Требуют обучения на financial data
- Сложно адаптировать под новые источники

**С LLM:**
```python
prompt = f"""
Analyze this earnings call transcript: {transcript}

Rate on scale 1-10:
1. Management confidence
2. Revenue outlook
3. Cost concerns
4. Competitive position

Provide JSON output.
"""

sentiment = llm.generate(prompt)
```

**Результат:** Быстрее, гибче, не требует обучения.

### **Кейс 3: Code review и debugging (Citadel)**

**Задача:** Ревью кода trading strategies от джуниор-квантов.

**Старый процесс:**
- Senior квант тратит 2-3 часа на ревью
- Находит баги, предлагает улучшения

**С LLM:**
```python
prompt = f"""
You are a senior quant at a hedge fund.
Review this trading strategy code: {code}

Check for:
1. Look-ahead bias
2. Incorrect position sizing
3. Missing commission handling
4. Off-by-one errors in indexing
5. Performance bottlenecks

Provide detailed feedback.
"""

review = llm.generate(prompt)
```

**Результат:** LLM находит 70-80% типичных ошибок за секунды.

### **Кейс 4: Автоматизация документации (Two Sigma)**

**Задача:** Документировать каждую стратегию (required for compliance).

**Старый процесс:**
- Квант пишет 10-страничный отчёт вручную
- 4-6 часов работы

**С LLM:**
```python
prompt = f"""
Generate a strategy documentation based on this code: {code}

Include:
1. Executive summary
2. Mathematical formulation
3. Risk analysis
4. Backtest results interpretation
5. Assumptions and limitations

Format: PDF-ready markdown.
"""

doc = llm.generate(prompt)
```

**Результат:** Документация за 30 минут вместо 6 часов.

## Прогноз: что произойдёт с квант-аналитиками

### **Сценарий 1: Augmentation (наиболее вероятно)**

LLM не заменят квантов, а **усилят их**.

**Как это выглядит:**

- **Джуниор кванты:** Делают то же, но в 3-5 раз быстрее (благодаря LLM)
- **Сеньор кванты:** Фокусируются на стратегическом мышлении, LLM берут рутину
- **Новая роль:** "Prompt Engineer" — специалист по эффективному использованию LLM

**Аналогия:** Калькуляторы не заменили математиков, но изменили их работу.

### **Сценарий 2: Democratization (средняя вероятность)**

LLM сделают квант-анализ доступным не-программистам.

**Кто получит доступ:**

- Портфельные менеджеры (смогут сами тестировать идеи)
- Retail traders (через no-code платформы с LLM)
- Аналитики без programming skills

**Последствия:**

- Спрос на джуниор-квантов упадёт (их задачи автоматизируются)
- Спрос на сеньор-квантов вырастет (нужны для проверки LLM-generated стратегий)

### **Сценарий 3: Полная замена (низкая вероятность)**

LLM научатся делать всё, что делают кванты.

**Проблемы:**

- Регуляторы не одобрят (нужна human accountability)
- LLM hallucinate (критично в финансах)
- Нет "понимания" контекста (macro, geopolitics)

**Сроки:** Если вообще произойдёт, то не раньше 2035-2040.

## Практические рекомендации для квантов

### **1. Используйте LLM как ассистента**

**Задачи, которые можно делегировать LLM:**

- Генерация кода (boilerplate, backtesting loops)
- Первичный research (поиск статей, summary)
- Генерация идей (brainstorming alpha factors)
- Документация
- Code review (первый pass)

**Что оставить себе:**

- Критическое мышление
- Финальное решение
- Ответственность за результат

### **2. Изучите prompt engineering**

[Правильный промпт](https://roguequant.substack.com/p/prompt-engineering-for-traders-how) может увеличить качество LLM в 3-5 раз.

**Плохой промпт:**
```
Create a trading strategy for BTC
```

**Хороший промпт:**
```
You are a quantitative analyst with 15 years of experience at Renaissance Technologies.

Create a mean-reversion trading strategy for BTC/USDT with the following requirements:
- Timeframe: 1-hour candles
- Holding period: 6-24 hours
- No technical indicators (use statistical properties)
- Include entry/exit rules with specific thresholds
- Provide expected Sharpe ratio and reasoning

Output format:
1. Strategy name
2. Hypothesis (why it should work)
3. Mathematical formulation
4. Entry rules
5. Exit rules
6. Risk management
7. Expected performance
```

**Разница:** Generic idea vs детальная спецификация.

### **3. Проверяйте всё, что генерирует LLM**

[LLM hallucinate](https://arxiv.org/html/2311.15548) особенно часто в финансах.

**Что проверять:**

- ✅ Цифры и статистика (запросить источник)
- ✅ Академические ссылки (проверить, существуют ли)
- ✅ Код (запустить unit tests)
- ✅ Логику (есть ли look-ahead bias, off-by-one errors)

**Правило:** Доверяй, но проверяй.

### **4. Комбинируйте LLM с classical методами**

Лучшие результаты — **гибридный подход**:

```python
class HybridQuant:
    def __init__(self):
        self.llm = LLM()

    def generate_ideas(self):
        # LLM генерирует 100 идей
        ideas = self.llm.brainstorm(n=100)
        return ideas

    def filter_ideas(self, ideas):
        # Квант отбирает top-10 по интуиции
        return self.manual_review(ideas, n=10)

    def implement(self, idea):
        # LLM пишет код
        code = self.llm.generate_code(idea)
        # Квант проверяет
        validated_code = self.review_code(code)
        return validated_code

    def backtest(self, code):
        # Автоматический бэктест
        results = run_backtest(code)
        # Квант интерпретирует
        analysis = self.interpret_results(results)
        return analysis
```

## Итоги

**Может ли LLM заменить квант-аналитика?**

**Короткий ответ:** Нет. Но может сделать его в 5 раз продуктивнее.

**Что LLM уже делает лучше кванта:**

- Генерация кода (быстрее, меньше bugs)
- Brainstorming идей (100 идей vs 10)
- Рутинные задачи (grid search, документация)

**Что LLM не может:**

- Критическое мышление (оценить, стоит ли идея времени)
- Понимание контекста (macro, market regime)
- Ответственность за решения
- Коммуникация с менеджментом

**Прогноз на 2026-2030:**

- Джуниор-кванты: Спрос упадёт (их задачи автоматизируются LLM)
- Сеньор-кванты: Спрос вырастет (нужны для проверки LLM и стратегического мышления)
- Новая роль: "AI-Augmented Quant" — специалист по эффективной работе с LLM

**Для квантов это означает:**

1. Учите prompt engineering
2. Используйте LLM для рутины
3. Фокусируйтесь на том, что LLM не может (intuition, strategy)
4. Проверяйте всё, что генерирует LLM

**Следующие статьи:**

- [Где заканчивается помощь ИИ и начинается самоликвидация депозита]({{site.baseurl}}/2026/03/24/gde-zakanchivaetsya-pomoshch-ii.html) — риски "чёрного ящика"
- [Эксперимент: LLM + классический алгоритм]({{site.baseurl}}/2026/03/31/eksperiment-llm-plus-klassika.html) — можем ли улучшить стратегию с помощью ИИ-фильтров

LLM не заменят квантов. Но кванты, которые не используют LLM, будут заменены теми, кто использует.

---

**Полезные ссылки:**

LLM в квант-финансе:
- [Quant Strats 2025: Integrating LLMs](https://biztechmagazine.com/article/2025/03/quant-strats-2025-4-ways-integrate-llms-quantitative-finance)
- [Automate Strategy Finding with LLM](https://arxiv.org/html/2409.06289v3)
- [Alpha-GPT 2.0 Framework](https://arxiv.org/html/2409.06289v3)
- [LLM for Trading Data Analysis](https://www.inoru.com/blog/how-does-building-llm-for-trading-data-improve-market-analysis-in-2025/)

Практические гайды:
- [Prompt Engineering for Traders](https://roguequant.substack.com/p/prompt-engineering-for-traders-how)
- [Claude Trading Strategy](https://medium.com/@austin-starks/i-let-claude-opus-4-create-a-trading-strategy-it-destroyed-the-market-c200bf1a19a4)
- [ChatGPT and Claude: Data to Backtesting](https://medium.com/@algorithmictrading/develop-a-trading-idea-using-chatgpt-and-claude-from-data-to-backtesting-40a5beb3f370)
- [LLM Agent Trader in Python](https://medium.com/coding-nexus/llm-agent-trader-a-free-ai-powered-stock-backtesting-system-built-in-python-ad574fd07628)

Риски и ограничения:
- [LLM Hallucinations in Finance](https://arxiv.org/html/2311.15548)
- [LLM Hallucinations: Implications for Financial Institutions](https://biztechmagazine.com/article/2025/08/llm-hallucinations-what-are-implications-financial-institutions)

Карьера кванта:
- [A Day in the Life of a Quantitative Analyst](https://www.cqf.com/blog/day-life-quantitative-analyst)
- [How to Become a Quantitative Analyst](https://www.datacamp.com/blog/how-to-become-quantitative-analyst)
