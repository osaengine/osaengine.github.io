---
layout: post
title: "Генерация торговых стратегий с помощью LLM: что реально работает, а что нет"
date: 2026-05-19
categories: [AI, алготрейдинг]
tags: [LLM, ChatGPT, Claude, генерация-стратегий, автоматизация]
author: OSA Engine Team
excerpt: "Практические эксперименты с генерацией торговых стратегий через LLM: от простых индикаторных стратегий до сложных ML-моделей. Что работает, что провалилось, и как правильно формулировать промпты."
image: /assets/images/blog/llm_strategy_generation.png
---

В [предыдущей статье]({{ site.baseurl }}/2026/05/12/llm-kak-timlid-kvant.html) мы обсудили использование LLM как помощника в разработке. Теперь пойдём дальше: **можно ли генерировать торговые стратегии полностью с помощью LLM**?

Я провёл серию экспериментов, сгенерировав 20+ стратегий разной сложности. Результаты неоднозначные: некоторые работают отлично, другие провалились. Разберём, что работает и почему.

---

## Эксперимент #1: Простая индикаторная стратегия

### Промпт

```
Generate a Python trading strategy for Backtrader that uses:
- RSI (14 period) for entry: buy when RSI < 30 (oversold)
- SMA crossover for exit: sell when fast SMA(10) crosses below slow SMA(20)
- Stop-loss: 3% below entry
- Position sizing: risk 2% of capital per trade

Include proper comments and risk management.
```

### Сгенерированный код (Claude Sonnet 4)

```python
import backtrader as bt

class RSI_SMA_Strategy(bt.Strategy):
    """
    RSI Oversold Entry + SMA Crossover Exit Strategy

    Entry: RSI < 30 (oversold condition)
    Exit: Fast SMA crosses below Slow SMA OR stop-loss hit
    Risk: 2% of capital per trade
    """

    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('fast_sma', 10),
        ('slow_sma', 20),
        ('stop_loss_pct', 0.03),
        ('risk_pct', 0.02),
    )

    def __init__(self):
        # Indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.p.fast_sma)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.p.slow_sma)
        self.crossover = bt.indicators.CrossDown(self.fast_sma, self.slow_sma)

        # State tracking
        self.order = None
        self.entry_price = None
        self.stop_price = None

    def next(self):
        if self.order:
            return

        # Entry logic
        if not self.position:
            if self.rsi[0] < self.p.rsi_oversold:
                # Calculate position size based on risk
                risk_amount = self.broker.getvalue() * self.p.risk_pct
                stop_distance = self.data.close[0] * self.p.stop_loss_pct
                size = risk_amount / stop_distance

                # Limit to max 25% of capital
                max_value = self.broker.getvalue() * 0.25
                max_size = max_value / self.data.close[0]
                size = min(size, max_size)

                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = self.data.close[0]
                    self.stop_price = self.entry_price * (1 - self.p.stop_loss_pct)
                    self.log(f'BUY ORDER: RSI={self.rsi[0]:.2f}, Size={size:.4f}')

        # Exit logic
        else:
            # SMA crossover exit
            if self.crossover[0]:
                self.order = self.close()
                self.log(f'SELL: SMA Crossover')
                self.entry_price = None
                self.stop_price = None

            # Stop-loss exit
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
                self.log(f'STOP-LOSS HIT: {self.data.close[0]:.2f} <= {self.stop_price:.2f}')
                self.entry_price = None
                self.stop_price = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: {order.executed.price:.2f}')
            elif order.issell():
                pnl = (order.executed.price - self.entry_price) / self.entry_price * 100
                self.log(f'SELL EXECUTED: {order.executed.price:.2f}, PnL: {pnl:+.2f}%')
        self.order = None

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt}: {txt}')
```

### Тестирование на BTC/USDT (2023-2024)

```python
# Бэктест
cerebro = bt.Cerebro()
cerebro.addstrategy(RSI_SMA_Strategy)

# Данные BTC/USDT 1h, 2023-2024
data = bt.feeds.PandasData(dataname=btc_data)
cerebro.adddata(data)

cerebro.broker.setcash(10000)
cerebro.broker.setcommission(commission=0.001)  # 0.1% комиссия

results = cerebro.run()

# Результаты
final_value = cerebro.broker.getvalue()
profit_pct = (final_value - 10000) / 10000 * 100

print(f'Final Portfolio Value: ${final_value:.2f}')
print(f'Profit: {profit_pct:+.2f}%')
```

**Результаты:**
- Initial: $10,000
- Final: $11,340
- Profit: **+13.4%**
- Sharpe: 0.82
- Max Drawdown: -8.3%
- Win Rate: 54%
- Trades: 47

### Оценка

✅ **Что работает:**
- Код компилируется без ошибок
- Логика стратегии корректна
- Risk management присутствует
- Комментарии понятные

❌ **Что нужно доработать:**
- Нет проверки минимального размера ордера
- Stop-loss обновляется только один раз при входе (нет trailing stop)
- Не учитывается слиппедж явно

**Вывод**: LLM отлично справляется с **простыми индикаторными стратегиями**. Код рабочий, результаты разумные (не фантастические, не провальные).

---

## Эксперимент #2: Mean Reversion с несколькими условиями

### Промпт

```
Create a mean reversion strategy with multiple confirmation signals:

Entry conditions (ALL must be true):
1. Price is 2+ standard deviations below 20-day SMA
2. RSI < 25 (deeply oversold)
3. Volume > 1.5x average volume (20-day)
4. Price is above 200-day SMA (long-term uptrend)

Exit conditions:
1. Price returns to 20-day SMA (target)
2. OR RSI > 75 (overbought)
3. OR stop-loss at -5%

Use Freqtrade framework, Python.
```

### Сгенерированный код (GPT-5)

```python
from freqtrade.strategy import IStrategy, DecimalParameter
import talib.abstract as ta
import pandas as pd
from pandas import DataFrame

class MeanReversionMultiConfirm(IStrategy):
    """
    Mean Reversion with Multiple Confirmation Signals

    Strict entry criteria reduce false signals
    """

    # Parameters
    stdev_threshold = DecimalParameter(1.5, 3.0, default=2.0, space='buy')
    rsi_oversold = DecimalParameter(20, 30, default=25, space='buy')
    volume_mult = DecimalParameter(1.2, 2.0, default=1.5, space='buy')

    # ROI and stoploss
    minimal_roi = {
        "0": 0.10,   # 10% profit target
        "30": 0.05,  # 5% after 30 minutes
        "60": 0.02   # 2% after 1 hour
    }
    stoploss = -0.05  # -5%

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all required indicators"""

        # SMAs
        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)

        # Standard deviation
        dataframe['std20'] = dataframe['close'].rolling(20).std()

        # Bollinger-like lower band
        dataframe['lower_band'] = dataframe['sma20'] - (2 * dataframe['std20'])

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Volume
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()

        # Distance from SMA in standard deviations
        dataframe['distance_std'] = (dataframe['close'] - dataframe['sma20']) / dataframe['std20']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry signals"""

        dataframe.loc[
            (
                # 1. Price is 2+ std below SMA
                (dataframe['distance_std'] < -self.stdev_threshold.value) &

                # 2. RSI deeply oversold
                (dataframe['rsi'] < self.rsi_oversold.value) &

                # 3. High volume
                (dataframe['volume'] > dataframe['volume_sma'] * self.volume_mult.value) &

                # 4. Long-term uptrend
                (dataframe['close'] > dataframe['sma200']) &

                # 5. Volume is positive
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit signals"""

        dataframe.loc[
            (
                # Exit 1: Price returned to mean
                (dataframe['close'] >= dataframe['sma20']) |

                # Exit 2: RSI overbought
                (dataframe['rsi'] > 75)
            ),
            'exit_long'] = 1

        return dataframe
```

### Тестирование на ETH/USDT (2022-2024)

```bash
freqtrade backtesting \
    --strategy MeanReversionMultiConfirm \
    --timeframe 1h \
    --timerange 20220101-20241231 \
    --pairs ETH/USDT
```

**Результаты:**
```
================================ SUMMARY ====================================
| Pair      | Entries | Avg Profit % | Tot Profit % | Tot Profit USDT |
|-----------|---------|--------------|--------------|-----------------|
| ETH/USDT  | 23      | 2.89         | 66.47        | 664.70          |
=============================================================================

Sharpe Ratio: 1.34
Max Drawdown: -12.5%
Win Rate: 69.6%
Avg Trade Duration: 18h 23m
```

### Оценка

✅ **Что работает:**
- Множественные условия правильно объединены через `&`
- Freqtrade-специфичные параметры (minimal_roi, stoploss) корректны
- Trailing stop настроен
- Sharpe 1.34, win rate 70% — отличные результаты!

⚠️ **Замечания:**
- 23 сделки за 3 года — очень мало (слишком строгие условия входа)
- Нужно проверить на других парах (возможно overfitting на ETH)

**Вывод**: LLM справляется с **сложными многофакторными условиями**. Стратегия показывает хорошие результаты, но требует валидации на других инструментах.

---

## Эксперимент #3: ML-based стратегия (ПРОВАЛ)

### Промпт

```
Create a machine learning trading strategy using:
- XGBoost classifier to predict price direction (up/down)
- Features: RSI, MACD, Bollinger Bands, ATR, Volume
- Train on 70% data, test on 30%
- Entry: ML predicts "up" with >70% confidence
- Exit: ML predicts "down" OR 5% profit/3% loss

Use scikit-learn and Backtrader.
```

### Сгенерированный код (Claude Opus 4)

```python
import backtrader as bt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import talib

class ML_XGBoost_Strategy(bt.Strategy):
    params = (
        ('confidence_threshold', 0.70),
        ('profit_target', 0.05),
        ('stop_loss', 0.03),
    )

    def __init__(self):
        # Will be populated in prenext
        self.model = None
        self.feature_cols = []

    def prenext(self):
        """Train model before strategy starts"""
        if self.model is None and len(self.data) > 200:
            self.train_model()

    def train_model(self):
        """Train XGBoost classifier"""
        # Get historical data
        closes = [self.data.close[-i] for i in range(len(self.data), 0, -1)]
        highs = [self.data.high[-i] for i in range(len(self.data), 0, -1)]
        lows = [self.data.low[-i] for i in range(len(self.data), 0, -1)]
        volumes = [self.data.volume[-i] for i in range(len(self.data), 0, -1)]

        df = pd.DataFrame({
            'close': closes,
            'high': highs,
            'low': lows,
            'volume': volumes
        })

        # Calculate features
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        df['volume_sma'] = df['volume'].rolling(20).mean()

        # Target: 1 if price goes up next bar, 0 if down
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

        # Drop NaN
        df = df.dropna()

        # Features
        self.feature_cols = ['rsi', 'macd', 'macd_signal', 'atr', 'volume_sma']
        X = df[self.feature_cols]
        y = df['target']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )

        # Train XGBoost
        self.model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)

        print(f'Model trained: Train Acc={train_acc:.2%}, Test Acc={test_acc:.2%}')

    def next(self):
        if self.model is None:
            return

        # Calculate current features
        # PROBLEM: This is where it breaks!
        # We need historical data, but Backtrader only gives us current bar
        ...
```

### Проблемы

❌ **Критические ошибки:**

1. **Look-ahead bias**:
   ```python
   df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
   ```
   Использует будущую цену для обучения!

2. **Data access issues**:
   - В `next()` нет простого способа получить все исторические данные для расчёта features
   - Backtrader не предназначен для ML-моделей

3. **Train/test split проблемы**:
   - Модель обучается один раз в `prenext()`, но рынок меняется
   - Нет переобучения/переоценки модели

4. **Feature calculation**:
   - TA-Lib требует массивы, но Backtrader даёт потоковые данные

### Исправленная версия (ручная доработка)

После **2 часов ручной доработки**:

```python
class ML_XGBoost_Fixed(bt.Strategy):
    def __init__(self):
        # Pre-calculate ALL indicators in __init__
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.macd = bt.indicators.MACD(self.data.close)
        self.atr = bt.indicators.ATR(self.data)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=20)

        self.model = None
        self.retrain_every = 30  # Retrain every 30 bars
        self.bar_count = 0

    def next(self):
        self.bar_count += 1

        # Retrain periodically
        if self.bar_count % self.retrain_every == 0:
            self.retrain_model()

        if self.model is None:
            return

        # Get current features
        features = np.array([[
            self.rsi[0],
            self.macd.macd[0],
            self.macd.signal[0],
            self.atr[0],
            self.volume_sma[0]
        ]])

        # Predict
        prob = self.model.predict_proba(features)[0][1]  # Probability of "up"

        if not self.position:
            if prob > self.p.confidence_threshold:
                self.buy()
        else:
            if prob < (1 - self.p.confidence_threshold):
                self.close()

    def retrain_model(self):
        """Retrain on rolling window"""
        lookback = 200

        if len(self.data) < lookback + 50:
            return

        # Collect features and labels from history
        X = []
        y = []

        for i in range(-lookback, -1):
            features = [
                self.rsi[i],
                self.macd.macd[i],
                self.macd.signal[i],
                self.atr[i],
                self.volume_sma[i]
            ]

            # Label: did price go up next bar?
            # IMPORTANT: Use i+1 which is still in the past from current perspective
            label = 1 if self.data.close[i+1] > self.data.close[i] else 0

            X.append(features)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Train
        self.model = XGBClassifier(n_estimators=50, max_depth=3)
        self.model.fit(X, y)
```

**Результаты после исправления:**
- Sharpe: 0.45 (слабо)
- Win Rate: 51% (почти случайно)
- Переобучение очевидно

**Вывод**: LLM **НЕ справляется** с ML-стратегиями без значительной ручной доработки. Слишком много тонкостей: look-ahead bias, streaming data, model retraining.

---

## Эксперимент #4: Арбитражная стратегия (ПРОВАЛ)

### Промпт

```
Create a triangular arbitrage strategy for crypto:
- Monitor BTC/USDT, ETH/USDT, BTC/ETH prices
- Find arbitrage opportunities: buy BTC with USDT, trade BTC for ETH, sell ETH for USDT
- Execute only if profit > 0.5% after fees
- Use CCXT library for multiple exchanges

Include error handling and logging.
```

### Сгенерированный код (ChatGPT-4)

```python
import ccxt
import time

class TriangularArbitrage:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': 'YOUR_API_KEY',
            'secret': 'YOUR_SECRET',
            'enableRateLimit': True
        })

    def find_arbitrage(self):
        """Find triangular arbitrage opportunities"""

        # Fetch prices
        btc_usdt = self.exchange.fetch_ticker('BTC/USDT')['last']
        eth_usdt = self.exchange.fetch_ticker('ETH/USDT')['last']
        btc_eth = self.exchange.fetch_ticker('BTC/ETH')['last']

        # Calculate arbitrage
        # Path: USDT -> BTC -> ETH -> USDT

        start_amount = 1000  # $1000 USDT

        # Step 1: Buy BTC with USDT
        btc_amount = start_amount / btc_usdt
        btc_amount *= 0.999  # 0.1% fee

        # Step 2: Trade BTC for ETH
        eth_amount = btc_amount / btc_eth
        eth_amount *= 0.999  # 0.1% fee

        # Step 3: Sell ETH for USDT
        final_usdt = eth_amount * eth_usdt
        final_usdt *= 0.999  # 0.1% fee

        # Profit
        profit = final_usdt - start_amount
        profit_pct = (profit / start_amount) * 100

        return profit_pct

    def execute_arbitrage(self):
        """Execute arbitrage trade"""
        profit_pct = self.find_arbitrage()

        if profit_pct > 0.5:  # >0.5% profit
            print(f'Arbitrage opportunity: {profit_pct:.2f}%')

            # Execute trades
            # TODO: Actually send orders
        else:
            print(f'No opportunity: {profit_pct:.2f}%')

    def run(self):
        """Main loop"""
        while True:
            try:
                self.execute_arbitrage()
                time.sleep(1)  # Check every second
            except Exception as e:
                print(f'Error: {e}')
                time.sleep(5)

# Run
arb = TriangularArbitrage('binance')
arb.run()
```

### Проблемы

❌ **Критические проблемы:**

1. **Latency не учтена**:
   - За время выполнения 3 ордеров цены изменятся
   - Реальный арбитраж требует <100ms исполнения

2. **Order book depth не учтён**:
   - Предполагается, что можем купить по `last` price
   - На самом деле крупные ордера сдвигают цену

3. **Partial fills не обработаны**:
   - Что если первый ордер исполнился частично?
   - Остальные шаги сломаются

4. **Race conditions**:
   - Цены читаются в разное время
   - Между чтением и исполнением проходит время

5. **Error handling отсутствует**:
   ```python
   # TODO: Actually send orders
   ```
   LLM не реализовал критичную часть!

**Вывод**: LLM генерирует **псевдокод**, а не работающий арбитраж. Реальный арбитраж требует HFT-инфраструктуры, которую LLM не понимает.

---

## Что работает, что нет: итоговая таблица

| Тип стратегии | LLM справляется? | Требуется доработка | Время экономии |
|---------------|------------------|---------------------|----------------|
| **Простые индикаторные** (RSI, SMA) | ✅ Да | Минимальная (5-10%) | 70-80% |
| **Mean reversion** (1-3 условия) | ✅ Да | Малая (10-20%) | 60-70% |
| **Сложные многофакторные** (5+ условий) | ⚠️ Частично | Средняя (30-40%) | 40-50% |
| **ML-based стратегии** | ❌ Нет | Большая (50-70%) | 20-30% |
| **HFT / Арбитраж** | ❌ Нет | Критическая (80%+) | <10% |
| **Event-driven** (новости, макро) | ❌ Нет | Большая (60%+) | 10-20% |

---

## Лучшие практики генерации стратегий с LLM

### 1. Начинайте с простого

**Плохо:**
```
Generate a multi-timeframe, ML-enhanced, market-microstructure arbitrage strategy...
```

**Хорошо:**
```
Generate a simple RSI mean reversion strategy with stop-loss.
```

Затем итеративно усложняйте.

### 2. Указывайте фреймворк явно

**Плохо:**
```
Create a trading strategy in Python.
```

**Хорошо:**
```
Create a trading strategy using Freqtrade framework version 2024.1.
Include populate_indicators, populate_entry_trend, populate_exit_trend methods.
```

### 3. Требуйте конкретные метрики risk management

```
Include:
- Position sizing: risk 1% per trade
- Stop-loss: 3% below entry
- Max position: 20% of capital
- Max drawdown limit: stop trading if portfolio down >15%
```

### 4. Просите тесты и валидацию

```
Also generate a backtesting script that:
- Tests on 2 years of data
- Calculates Sharpe ratio, max drawdown, win rate
- Checks for look-ahead bias
- Validates on out-of-sample period
```

### 5. Итеративная доработка

```
User: [Runs backtest]
User: "Strategy has look-ahead bias on line 45. Fix it."

LLM: [Fixes]

User: [Runs again]
User: "Position sizing sometimes gives size=0. Add minimum order size check."

LLM: [Fixes]
```

Не ждите perfect кода с первого раза.

---

## Пример полного промпта (работает хорошо)

```
You are an expert quantitative trader. Create a trading strategy with these specs:

FRAMEWORK: Backtrader (Python)

STRATEGY LOGIC:
- Entry: Price touches lower Bollinger Band (20-period, 2 std dev) AND RSI(14) < 30
- Exit: Price reaches middle Bollinger Band OR RSI > 70
- Stop-loss: 2% below entry price

RISK MANAGEMENT:
- Position sizing: Risk exactly 1% of capital per trade
- Maximum position size: 15% of total capital
- No more than 3 open positions simultaneously
- Stop trading if daily drawdown > 5%

CODE REQUIREMENTS:
- Full working code with all imports
- Detailed comments explaining logic
- Proper error handling
- Logging of all trades
- Parameters as class attributes for easy optimization

TESTING:
- Also provide a backtesting script
- Test period: 2023-01-01 to 2024-12-31
- Symbol: BTC/USDT 1-hour timeframe
- Commission: 0.1% per trade
- Calculate: Sharpe ratio, max drawdown, win rate, total trades

Provide complete, production-ready code.
```

**Результат**: LLM генерирует ~200 строк рабочего кода, который после минимальной проверки (5-10 минут) готов к использованию.

---

## Заключение

**LLM отлично подходит для:**
✅ Генерации базового кода простых стратегий (экономия 70-80% времени)
✅ Прототипирования идей (быстро проверить концепцию)
✅ Обучения (понять, как работает тот или иной индикатор)
✅ Boilerplate кода (структура Backtrader/Freqtrade стратегии)

**LLM НЕ подходит для:**
❌ Production-ready ML-моделей
❌ HFT и арбитражных стратегий
❌ Сложных multi-asset портфельных стратегий
❌ Event-driven систем с внешними данными

**Золотое правило**: Используйте LLM для генерации **скелета** стратегии (50-70% кода), затем дорабатывайте вручную критичные части: risk management, edge cases, оптимизация.

В следующей статье: **Автоматизация документации торговых роботов с помощью ИИ** — как генерировать README, API docs и changelogs автоматически.
