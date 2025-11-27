---
layout: post
title: "Как я использую LLM как тимлида-кванта: постановка задач, ревью кода, разбор багов"
date: 2026-05-12
categories: [AI, алготрейдинг]
tags: [LLM, ChatGPT, Claude, разработка, code-review, debugging]
author: OSA Engine Team
excerpt: "Практический опыт использования LLM (ChatGPT, Claude) как помощника в разработке торговых стратегий: генерация кода, ревью, дебаг, документация. Реальные примеры и промпты."
image: /assets/images/blog/llm_quant_lead.png
---

В предыдущих статьях мы обсуждали, [может ли LLM заменить квант-аналитика]({{ site.baseurl }}/2026/03/17/mozhet-li-llm-zamenit-kvant-analitika.html) (спойлер: нет, но помочь может) и [где заканчивается помощь ИИ]({{ site.baseurl }}/2026/03/24/gde-zakanchivaetsya-pomoshch-ii.html). Теперь — **практический гайд**: как я использую LLM ежедневно в разработке торговых стратегий.

---

## Сценарий 1: Генерация базового кода стратегии

### Задача

Нужно быстро набросать скелет стратегии на mean reversion с Bollinger Bands.

### Промпт для LLM

```
You are an expert quantitative trader. Write a Python trading strategy using the Backtrader framework that implements a mean reversion approach with Bollinger Bands.

Requirements:
- Entry: When price touches lower Bollinger Band (oversold)
- Exit: When price reaches middle band (mean)
- Stop-loss: 2% below entry
- Position sizing: Risk 1% of capital per trade
- Bollinger Bands: 20-period SMA, 2 standard deviations

Include proper risk management and comments explaining the logic.
```

### Результат от Claude/ChatGPT

```python
import backtrader as bt

class BollingerMeanReversion(bt.Strategy):
    params = (
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('risk_pct', 0.01),  # Risk 1% per trade
        ('stop_loss_pct', 0.02),  # 2% stop-loss
    )

    def __init__(self):
        # Bollinger Bands indicator
        self.boll = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_dev
        )

        # Track order and entry price
        self.order = None
        self.entry_price = None

    def next(self):
        # Skip if we have a pending order
        if self.order:
            return

        # Entry logic: Price touches lower band
        if not self.position:
            if self.data.close[0] <= self.boll.lines.bot[0]:
                # Calculate position size based on risk
                risk_amount = self.broker.getvalue() * self.params.risk_pct
                stop_distance = self.data.close[0] * self.params.stop_loss_pct
                size = risk_amount / stop_distance

                # Buy
                self.order = self.buy(size=size)
                self.entry_price = self.data.close[0]
                self.log(f'BUY SIGNAL: Price {self.data.close[0]:.2f} <= BB Lower {self.boll.lines.bot[0]:.2f}')

        # Exit logic: Price reaches middle band OR stop-loss
        else:
            # Exit at middle band (mean)
            if self.data.close[0] >= self.boll.lines.mid[0]:
                self.order = self.close()
                self.log(f'SELL SIGNAL: Price {self.data.close[0]:.2f} >= BB Mid {self.boll.lines.mid[0]:.2f}')

            # Stop-loss
            elif self.entry_price and self.data.close[0] <= self.entry_price * (1 - self.params.stop_loss_pct):
                self.order = self.close()
                self.log(f'STOP-LOSS: Price {self.data.close[0]:.2f} <= Stop {self.entry_price * (1 - self.params.stop_loss_pct):.2f}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size:.2f}')

        self.order = None

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
```

### Оценка результата

**Что LLM сделал хорошо:**
✅ Правильная структура Backtrader стратегии
✅ Bollinger Bands индикатор подключен
✅ Risk-based position sizing
✅ Stop-loss логика
✅ Комментарии

**Что нужно доработать:**
❌ Position sizing может давать слишком большой размер (нет ограничения на max % капитала)
❌ Нет проверки на минимальный размер ордера
❌ Entry price сбрасывается неправильно (нужно обнулять после выхода)

### Доработка с помощью LLM

```
The code looks good, but I found a few issues:

1. Position sizing calculation doesn't limit the maximum position size (should be max 20% of capital)
2. entry_price is not reset after closing the position
3. No minimum order size check

Please fix these issues.
```

LLM исправляет код:

```python
def next(self):
    if self.order:
        return

    if not self.position:
        if self.data.close[0] <= self.boll.lines.bot[0]:
            # Calculate position size with max limit
            risk_amount = self.broker.getvalue() * self.params.risk_pct
            stop_distance = self.data.close[0] * self.params.stop_loss_pct
            size = risk_amount / stop_distance

            # Limit to max 20% of capital
            max_position_value = self.broker.getvalue() * 0.20
            max_size = max_position_value / self.data.close[0]
            size = min(size, max_size)

            # Minimum order size check (0.001)
            if size >= 0.001:
                self.order = self.buy(size=size)
                self.entry_price = self.data.close[0]
                self.log(f'BUY SIGNAL: Size {size:.4f}')
    else:
        if self.data.close[0] >= self.boll.lines.mid[0]:
            self.order = self.close()
            self.entry_price = None  # Reset entry price
            self.log(f'SELL SIGNAL')

        elif self.entry_price and self.data.close[0] <= self.entry_price * (1 - self.params.stop_loss_pct):
            self.order = self.close()
            self.entry_price = None  # Reset entry price
            self.log(f'STOP-LOSS')
```

**Экономия времени**: вместо 30-40 минут написания с нуля → 5 минут с LLM + 10 минут доработки = **15 минут**.

---

## Сценарий 2: Code Review

### Задача

Написал сложную стратегию, нужен code review на потенциальные баги.

### Промпт для LLM

```
You are a senior quantitative developer. Review this trading strategy code for potential bugs, performance issues, and best practices violations.

Pay special attention to:
- Look-ahead bias
- Off-by-one errors
- Memory leaks
- Race conditions
- Risk management issues

[ВСТАВЛЯЮ КОД СТРАТЕГИИ]

Provide detailed feedback with severity levels (Critical, High, Medium, Low).
```

### Пример ответа от LLM

```markdown
## Code Review: Trading Strategy

### CRITICAL Issues

**1. Look-Ahead Bias in SMA Calculation (Line 45)**
Severity: CRITICAL
```python
# WRONG:
sma = data['close'].rolling(20).mean().iloc[i]

# This recalculates the entire SMA series on each iteration,
# potentially accessing future data.

# CORRECT:
# Use incremental calculation or ensure you're only using data up to index i
sma = data['close'].iloc[:i+1].rolling(20).mean().iloc[-1]
```

**Impact**: Strategy will show unrealistic backtest results. In live trading, performance will degrade significantly.

**Recommendation**: Refactor to use event-driven architecture or ensure strict temporal ordering.

---

**2. Position Size Not Validated (Line 78)**
Severity: CRITICAL
```python
position_size = capital * 0.10 / price  # No validation!

# What if this results in size = 0.0001?
# Or size = 1000000 (if price is very low)?
```

**Impact**: Could send invalid orders to exchange, causing errors or unexpected exposure.

**Recommendation**: Add validation:
```python
position_size = max(MIN_ORDER_SIZE, min(position_size, MAX_POSITION_SIZE))
```

### HIGH Issues

**3. No Error Handling for Exchange API (Line 92)**
```python
order_id = exchange.create_order(...)  # What if this fails?
```

**Recommendation**:
```python
try:
    order_id = exchange.create_order(...)
except ExchangeError as e:
    logger.error(f"Order failed: {e}")
    # Implement retry logic or alert
```

### MEDIUM Issues

**4. Inefficient Loop (Line 120)**
```python
for i in range(len(data)):
    for j in range(len(data)):  # O(n²) complexity
        if data[i] correlates with data[j]:
            ...
```

**Impact**: Slow backtesting on large datasets.

**Recommendation**: Use vectorized operations with pandas/numpy.

### LOW Issues

**5. Magic Numbers (Throughout)**
```python
if rsi < 30:  # Why 30? Document this
if drawdown > 0.15:  # Why 15%?
```

**Recommendation**: Use named constants:
```python
RSI_OVERSOLD = 30
MAX_DRAWDOWN_PCT = 0.15
```
```

### Ценность

LLM находит:
- 80% очевидных багов
- 60% логических ошибок
- 90% style issues

Что **не находит**:
- Сложные race conditions
- Бизнес-логику (правильна ли стратегия?)
- Performance bottlenecks в специфичном окружении

---

## Сценарий 3: Debugging

### Задача

Стратегия падает с ошибкой `KeyError: 'close'`. Нужно понять почему.

### Промпт для LLM

```
I'm getting this error when running my trading strategy:

```
KeyError: 'close'
  File "strategy.py", line 45, in calculate_indicators
    sma = data['close'].rolling(20).mean()
```

Here's the relevant code:

```python
def calculate_indicators(self, data):
    sma = data['close'].rolling(20).mean()
    rsi = talib.RSI(data['close'], timeperiod=14)
    return sma, rsi

def on_bar(self, bar):
    indicators = self.calculate_indicators(bar)
```

The data comes from Binance API. What could be wrong?
```

### Ответ от LLM

```markdown
The issue is that you're passing a **single bar** (dict) to `calculate_indicators()`,
but the function expects a **DataFrame** with a 'close' column.

**Problem:**
```python
def on_bar(self, bar):
    # bar is a dict: {'timestamp': ..., 'open': ..., 'high': ..., 'low': ..., 'close': ...}
    indicators = self.calculate_indicators(bar)  # ERROR!
```

**Why it fails:**
- `bar['close']` would work (accessing dict key)
- But `data['close'].rolling(20)` expects a pandas Series, not a single value

**Solutions:**

**Option 1: Accumulate bars into DataFrame**
```python
class Strategy:
    def __init__(self):
        self.bars = []

    def on_bar(self, bar):
        # Append to history
        self.bars.append(bar)

        # Convert to DataFrame
        df = pd.DataFrame(self.bars)

        # Now calculate indicators
        if len(df) >= 20:  # Need at least 20 bars for SMA
            indicators = self.calculate_indicators(df)
```

**Option 2: Use incremental indicators**
```python
from collections import deque

class Strategy:
    def __init__(self):
        self.close_prices = deque(maxlen=20)

    def on_bar(self, bar):
        self.close_prices.append(bar['close'])

        if len(self.close_prices) == 20:
            sma = sum(self.close_prices) / 20
```

**Option 3: Check Binance API response format**
Sometimes the API returns different formats. Print `bar` to verify:
```python
def on_bar(self, bar):
    print(f"Bar type: {type(bar)}")
    print(f"Bar content: {bar}")
```

Most likely you need **Option 1** — accumulating bars into a DataFrame.
```

**Результат**: проблема решена за 2 минуты вместо 20-30 минут debugging.

---

## Сценарий 4: Оптимизация кода

### Промпт

```
This backtesting code is very slow (takes 5 minutes for 1 year of data).
Can you optimize it?

```python
for i in range(len(data)):
    for symbol in symbols:
        price = data[data['symbol'] == symbol]['close'].iloc[i]
        sma = data[data['symbol'] == symbol]['close'].iloc[:i+1].rolling(20).mean().iloc[-1]

        if price > sma:
            signals.append(('BUY', symbol, price))
```
```

### Ответ от LLM

```python
# OPTIMIZED VERSION

# Problem 1: data[data['symbol'] == symbol] is called multiple times per iteration (slow!)
# Problem 2: Rolling calculation is repeated for every bar (O(n²))
# Problem 3: iloc[:i+1] creates new DataFrame slice each time

# Solution: Vectorize and pre-calculate

# Pre-group by symbol (do once, not in loop)
grouped = data.groupby('symbol')

signals = []

for symbol, symbol_data in grouped:
    # Pre-calculate SMA for entire series (vectorized)
    symbol_data['sma'] = symbol_data['close'].rolling(20).mean()

    # Vectorized comparison (no loop!)
    buy_signals = symbol_data[symbol_data['close'] > symbol_data['sma']]

    # Append signals
    for idx, row in buy_signals.iterrows():
        signals.append(('BUY', symbol, row['close']))

# Performance improvement: 5 minutes → 3 seconds (100x faster!)
```

**Дополнительное объяснение от LLM:**

```markdown
### Why This is Faster

**Original O(n × m × k):**
- n = number of bars
- m = number of symbols
- k = rolling window size
- Total: ~1,000,000 operations

**Optimized O(n × m):**
- Pre-grouping: O(n × m)
- Vectorized rolling: O(n × m)
- Vectorized comparison: O(n × m)
- Total: ~50,000 operations

### Further Optimizations

If still slow, consider:

1. **Use Numba JIT compilation**:
```python
from numba import jit

@jit(nopython=True)
def calculate_sma(prices, period):
    # Fast native code
    ...
```

2. **Parallel processing**:
```python
from multiprocessing import Pool

with Pool(4) as p:
    results = p.map(process_symbol, symbols)
```

3. **Use specialized libraries**:
- `ta-lib` (C-based, very fast)
- `pandas-ta` (optimized for pandas)
```
```

---

## Сценарий 5: Документация и комментарии

### Промпт

```
Add comprehensive docstrings and comments to this strategy code:

[ВСТАВЛЯЮ КОД]

Follow Google Python Style Guide for docstrings.
```

### Результат

```python
class BollingerMeanReversion(bt.Strategy):
    """
    Mean Reversion strategy using Bollinger Bands.

    This strategy enters long positions when price touches the lower Bollinger Band
    (indicating oversold conditions) and exits when price returns to the middle band
    (mean reversion). Includes 2% stop-loss for risk management.

    Parameters:
        bb_period (int): Period for Bollinger Bands SMA calculation. Default: 20
        bb_dev (float): Number of standard deviations for bands. Default: 2.0
        risk_pct (float): Percentage of capital to risk per trade. Default: 0.01 (1%)
        stop_loss_pct (float): Stop-loss percentage below entry. Default: 0.02 (2%)

    Attributes:
        boll (BollingerBands): Bollinger Bands indicator instance
        order (Order): Currently pending order (None if no order)
        entry_price (float): Price at which current position was entered

    Example:
        >>> cerebro = bt.Cerebro()
        >>> cerebro.addstrategy(BollingerMeanReversion,
        ...                     bb_period=20,
        ...                     risk_pct=0.01)
        >>> cerebro.run()
    """

    params = (
        ('bb_period', 20),      # Bollinger Bands period
        ('bb_dev', 2.0),        # Standard deviations
        ('risk_pct', 0.01),     # Risk 1% per trade
        ('stop_loss_pct', 0.02),  # 2% stop-loss
    )

    def __init__(self):
        """Initialize strategy with Bollinger Bands indicator."""
        # Create Bollinger Bands indicator
        # Uses close price with specified period and deviation
        self.boll = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_dev
        )

        self.order = None  # Track pending orders
        self.entry_price = None  # Track entry price for stop-loss

    def next(self):
        """
        Execute strategy logic on each new bar.

        Entry Logic:
            - No open position
            - Current close <= Lower Bollinger Band (oversold)
            → Buy with risk-based position sizing

        Exit Logic:
            - Open position exists
            - Current close >= Middle Bollinger Band (mean reversion)
            OR
            - Current close <= entry_price * (1 - stop_loss_pct) (stop-loss)
            → Sell to close position

        Risk Management:
            - Position size calculated to risk exactly risk_pct% of capital
            - Maximum position size capped at 20% of capital
            - Minimum order size check (0.001)
        """
        # Skip if we have a pending order
        if self.order:
            return

        # === ENTRY LOGIC ===
        if not self.position:
            # Check if price touched lower band (oversold)
            if self.data.close[0] <= self.boll.lines.bot[0]:
                # Calculate position size based on risk
                # Risk Amount = Capital × Risk%
                # Position Size = Risk Amount / Stop Distance
                risk_amount = self.broker.getvalue() * self.params.risk_pct
                stop_distance = self.data.close[0] * self.params.stop_loss_pct
                size = risk_amount / stop_distance

                # Apply maximum position size limit (20% of capital)
                max_position_value = self.broker.getvalue() * 0.20
                max_size = max_position_value / self.data.close[0]
                size = min(size, max_size)

                # Minimum order size check
                if size >= 0.001:
                    self.order = self.buy(size=size)
                    self.entry_price = self.data.close[0]
                    self.log(f'BUY SIGNAL: Price {self.data.close[0]:.2f} <= BB Lower {self.boll.lines.bot[0]:.2f}, Size {size:.4f}')

        # === EXIT LOGIC ===
        else:
            # Exit at middle band (mean reversion target)
            if self.data.close[0] >= self.boll.lines.mid[0]:
                self.order = self.close()
                profit = (self.data.close[0] - self.entry_price) / self.entry_price * 100
                self.log(f'SELL SIGNAL (TARGET): Price {self.data.close[0]:.2f} >= BB Mid {self.boll.lines.mid[0]:.2f}, Profit {profit:.2f}%')
                self.entry_price = None

            # Stop-loss
            elif self.entry_price and self.data.close[0] <= self.entry_price * (1 - self.params.stop_loss_pct):
                self.order = self.close()
                loss = (self.data.close[0] - self.entry_price) / self.entry_price * 100
                self.log(f'STOP-LOSS: Price {self.data.close[0]:.2f} <= Stop {self.entry_price * (1 - self.params.stop_loss_pct):.2f}, Loss {loss:.2f}%')
                self.entry_price = None
```

---

## Лучшие практики работы с LLM

### 1. Будьте конкретны в промптах

**Плохо:**
```
Fix this code
```

**Хорошо:**
```
This code has a look-ahead bias in line 45 where we access future data.
Fix it by ensuring we only use data up to the current bar index.
Also add error handling for the exchange API call on line 92.
```

### 2. Проверяйте результат

LLM делает ошибки. ВСЕГДА:
- Тестируйте сгенерированный код
- Проверяйте логику
- Запускайте бэктесты

### 3. Итеративная доработка

Не ждите идеального результата с первого раза. Работайте итерациями:

```
1. Сгенерировать базовый код
2. Протестировать
3. Найти проблемы
4. Попросить LLM исправить
5. Повторить
```

### 4. Используйте для рутинных задач

LLM отлично справляется с:
- Boilerplate code
- Документацией
- Простыми алгоритмами
- Рефакторингом
- Code review

НЕ используйте для:
- Критической бизнес-логики без проверки
- Сложных математических моделей (проверяйте дважды)
- Production code без тестирования

---

## Заключение

LLM — это **ассистент**, а не замена разработчику. При правильном использовании он может:

✅ Ускорить разработку в 2-3 раза
✅ Найти 70-80% багов на code review
✅ Сократить время на документацию
✅ Помочь с debugging

❌ Но не может:
- Заменить понимание алготрейдинга
- Гарантировать правильность сложной логики
- Спроектировать архитектуру системы

**Мой workflow**:
1. Быстрый прототип с LLM (30% времени)
2. Code review с LLM (10% времени)
3. Ручная доработка и оптимизация (40% времени)
4. Тестирование и валидация (20% времени)

Общая экономия времени: **~30-40%** по сравнению с разработкой без LLM.

---

Это была последняя статья из серии. Надеюсь, материал был полезен!
