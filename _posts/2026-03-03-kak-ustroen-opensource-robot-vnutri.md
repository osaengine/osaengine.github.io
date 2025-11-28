---
layout: post
title: "Как устроен типичный open-source робот внутри: архитектура на примере простой стратегии"
description: "Разбираем реальную архитектуру торговых роботов: от слоёв и компонентов до выбора паттернов. Freqtrade, NautilusTrader, микросервисы, Event Sourcing — что работает на практике."
date: 2026-03-03
image: /assets/images/blog/opensource_robot_architecture.png
tags: [архитектура, open-source, торговые роботы, Event Sourcing, микросервисы]
---

Три месяца назад я [сравнивал open-source фреймворки для MOEX]({{site.baseurl}}/2026/02/24/sravnenie-lean-stocksharp-backtrader.html). Разобрался, что выбрать. Но вопрос остался: **а как это всё устроено внутри?**

Можно прочитать документацию. Можно изучить API. Но настоящее понимание приходит только когда открываешь код и видишь архитектуру.

Поэтому сегодня — не про выбор платформы. Про то, **что происходит под капотом**.

Я разобрал архитектуру четырёх популярных open-source роботов: [Freqtrade](https://github.com/freqtrade/freqtrade), [NautilusTrader](https://github.com/nautechsystems/nautilus_trader), [Hummingbot](https://github.com/hummingbot/hummingbot) и микросервисную систему [MBATS](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System).

Вывод: несмотря на разные языки и цели, **паттерны повторяются**. И если их понять — можно спроектировать свою систему правильно с первого раза.

## Зачем вообще изучать архитектуру чужих роботов?

**Проблема новичка:**

Вы пишете первого робота. Начинаете с простого: `if RSI > 70: sell()`. Через месяц код превращается в спагетти. Стратегия, подключение к бирже, риск-менеджмент, логи — всё в одном файле.

Добавить вторую стратегию? Скопировать файл и менять. Подключить вторую биржу? Переписать половину кода. Запустить бэктест? Хардкодить моки.

**Проблема профессионала:**

Вы пишете систему для продакшена. Нужна высокая доступность, аудит всех операций, масштабирование на несколько рынков. С чего начать?

**Решение:**

Изучить, как это сделали другие. Open-source даёт то, что не даст ни один курс — **реальную рабочую архитектуру**, которая пережила баги, рефакторинг и тысячи пользователей.

## Слоистая архитектура: фундамент любого торгового робота

### **Почему слои?**

[Freqtrade использует трёхслойную архитектуру](https://medium.com/@lufeiy/freqtrade-uncovered-how-machine-learning-powers-open-source-crypto-trading-25b1eab16ad9): Input Layer, Processing Layer, Output Layer.

[NautilusTrader делит систему на](https://nautilustrader.io/docs/latest/concepts/architecture/): Data Layer, Core Engine, Strategy Layer, Execution Layer.

Почему все так делают?

**Принцип разделения ответственности (Separation of Concerns):**

- Каждый слой решает свою задачу
- Изменения в одном слое не ломают другие
- Можно тестировать слои независимо

### **Типичная структура: 5 слоёв**

Вот как выглядит архитектура большинства торговых роботов:

```
┌─────────────────────────────────────┐
│   Layer 5: Communication Layer      │  ← Telegram, Web UI, API
├─────────────────────────────────────┤
│   Layer 4: Strategy Layer           │  ← Торговая логика
├─────────────────────────────────────┤
│   Layer 3: Execution & Risk         │  ← Ордера, риск-менеджмент
├─────────────────────────────────────┤
│   Layer 2: Data Processing          │  ← Индикаторы, нормализация
├─────────────────────────────────────┤
│   Layer 1: Data Ingestion           │  ← Подключение к биржам
└─────────────────────────────────────┘
```

Разберём каждый.

## Layer 1: Data Ingestion — откуда приходят данные

**Задача:** Получить данные с биржи и доставить их в систему.

### **Архитектура Freqtrade**

[Input Layer в Freqtrade](https://medium.com/aimonks/freqtrade-master-crypto-auto-trading-with-a-modular-bot-daf8e06a0533) включает:

**Exchange Interface:**
- Подключение к Binance, Kraken, Kucoin через ccxt
- Унификация API разных бирж

**Market Data Module:**
- Загрузка OHLCV данных
- Подписка на стримы в реальном времени

**Пример кода (упрощённо):**

```python
# freqtrade/exchange/exchange.py
class Exchange:
    def __init__(self, config):
        self.ccxt = ccxt.binance({
            'apiKey': config['api_key'],
            'secret': config['api_secret'],
        })

    def fetch_ohlcv(self, pair, timeframe):
        return self.ccxt.fetch_ohlcv(pair, timeframe)

    def create_order(self, pair, order_type, side, amount, price=None):
        return self.ccxt.create_order(pair, order_type, side, amount, price)
```

**Ключевая деталь:** Использование [ccxt](https://github.com/ccxt/ccxt) позволяет поддерживать 100+ бирж без написания отдельного коннектора для каждой.

### **Архитектура NautilusTrader**

[NautilusTrader пошёл дальше](https://nautilustrader.io/docs/latest/concepts/architecture/): **ядро написано на Rust**, Python-биндинги через PyO3.

**Почему Rust?**

- Асинхронная обработка через [tokio](https://tokio.rs/)
- [Стриминг до 5 млн строк в секунду](https://docs.rs/nautilus-trading)
- Безопасность на уровне компиляции

**Data Adapters:**

NautilusTrader абстрагирует источники данных через адаптеры:

```rust
// Упрощённый пример концепции
pub trait DataAdapter {
    async fn subscribe(&self, instrument: InstrumentId);
    async fn request_bars(&self, request: BarDataRequest) -> Vec<Bar>;
}
```

Это позволяет подключать не только биржи, но и исторические данные из CSV, Parquet, TimescaleDB.

### **Что можно взять для своей системы:**

1. **Используйте адаптеры** — не пишите логику работы с API прямо в стратегии
2. **Унифицируйте интерфейс** — разные биржи должны выглядеть одинаково для вашего кода
3. **Отделяйте получение данных от их обработки** — один модуль скачивает, другой нормализует

## Layer 2: Data Processing — от сырых данных к сигналам

**Задача:** Преобразовать сырые цены в индикаторы, фичи для ML, сигналы.

### **Индикаторы в NautilusTrader**

[Все индикаторы написаны на Rust](https://docs.rs/nautilus-indicators) с bounded memory usage (ограниченным потреблением памяти):

```rust
pub struct ExponentialMovingAverage {
    period: usize,
    alpha: f64,
    value: f64,
    initialized: bool,
}

impl Indicator for ExponentialMovingAverage {
    fn update(&mut self, price: f64) {
        if !self.initialized {
            self.value = price;
            self.initialized = true;
        } else {
            self.value = self.alpha * price + (1.0 - self.alpha) * self.value;
        }
    }
}
```

**Преимущества:**

- Циклические буферы вместо бесконечного роста массивов
- Работают в режиме реального времени без пересчёта с нуля

### **FreqAI: Machine Learning слой во Freqtrade**

[FreqAI — это подсистема для ML](https://www.freqtrade.io/en/stable/freqai/) внутри Freqtrade.

**Три компонента:**

1. **IFreqaiModel** — логика обучения и inference
2. **FreqaiDataKitchen** — препроцессинг фич
3. **FreqaiDataDrawer** — хранилище предсказаний и моделей

**Пример стратегии с ML:**

```python
class MyStrategy(IStrategy):
    def populate_any_indicators(self, metadata, pair, df, tf):
        # Генерируем фичи
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['ema_fast'] = ta.EMA(df['close'], timeperiod=12)
        df['ema_slow'] = ta.EMA(df['close'], timeperiod=26)
        return df

    def populate_predictions(self, df, metadata):
        # FreqAI автоматически обучает модель
        df['prediction'] = self.freqai.start(df, metadata)
        return df

    def populate_entry_trend(self, df, metadata):
        df.loc[(df['prediction'] > 0.6), 'enter_long'] = 1
        return df
```

FreqAI поддерживает:
- LightGBM, CatBoost, XGBoost
- PyTorch, TensorFlow
- Reinforcement Learning (PPO, DQN)

### **Что можно взять:**

1. **Разделяйте генерацию фич и inference** — не смешивайте в одной функции
2. **Кэшируйте вычисления индикаторов** — RSI за 100 свечей не должен пересчитываться каждый тик
3. **Используйте bounded buffers** — ограничьте потребление памяти для long-running процессов

## Layer 3: Execution & Risk Management — от сигнала к ордеру

**Задача:** Принять решение "купить" и превратить его в реальный ордер с учётом рисков.

### **Risk Management в Freqtrade**

[Processing Layer включает Risk Management Module](https://medium.com/@lufeiy/freqtrade-uncovered-how-machine-learning-powers-open-source-crypto-trading-25b1eab16ad9):

```python
# freqtrade/strategy/interface.py (упрощённо)
class IStrategy:
    def custom_stake_amount(self, pair, current_time, current_rate,
                           proposed_stake, **kwargs):
        # Динамический расчёт размера позиции
        if self.wallets.get_free('USDT') < 100:
            return None  # Не открываем позицию

        # Риск 2% от депозита
        risk_per_trade = self.wallets.get_total('USDT') * 0.02
        return risk_per_trade
```

**Защита от переторговли:**

```python
def confirm_trade_entry(self, pair, order_type, amount, rate, **kwargs):
    # Максимум 3 открытые позиции
    if len(self.trades) >= 3:
        return False

    # Не открываем, если волатильность слишком высокая
    if self.get_volatility(pair) > 5.0:
        return False

    return True
```

### **Smart Order Routing (SOR)**

[В микросервисных архитектурах](https://vocal.media/education/architectural-design-patterns-for-high-frequency-algo-trading-bots) SOR — это отдельный сервис:

**Задача SOR:**

- Разбить большой ордер на части (чтобы не двигать рынок)
- Выбрать лучшую биржу (по ликвидности и комиссиям)
- Использовать алгоритмы исполнения (TWAP, VWAP, Iceberg)

**Пример микросервисной архитектуры:**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Strategy   │────▶│     SOR      │────▶│   Exchange   │
│   Service    │     │   Service    │     │   Adapter    │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │
       │             ┌──────▼──────┐
       │             │ Risk Engine │
       └────────────▶│   Service   │
                     └─────────────┘
```

## Layer 4: Strategy Layer — где живёт ваша логика

**Задача:** Определить, когда покупать и когда продавать.

### **Архитектура стратегий в Freqtrade**

[Strategy Engine анализирует данные и генерирует сигналы](https://www.freqtrade.io/en/stable/):

```python
class SimpleMAStrategy(IStrategy):
    # Параметры стратегии
    buy_sma_short = 12
    buy_sma_long = 26

    def populate_indicators(self, dataframe, metadata):
        dataframe['sma_short'] = ta.SMA(dataframe, timeperiod=self.buy_sma_short)
        dataframe['sma_long'] = ta.SMA(dataframe, timeperiod=self.buy_sma_long)
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[
            (dataframe['sma_short'] > dataframe['sma_long']),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[
            (dataframe['sma_short'] < dataframe['sma_long']),
            'exit_long'
        ] = 1
        return dataframe
```

**Ключевое преимущество:** Стратегия полностью отделена от исполнения. Вы можете:
- Запустить бэктест без изменения кода
- Переключиться на другую биржу
- Добавить виртуальную торговлю (paper trading)

### **Event-Driven стратегии в NautilusTrader**

[NautilusTrader использует Actor Model](https://nautilustrader.io/docs/latest/concepts/architecture/):

```python
from nautilus_trader.trading.strategy import Strategy

class EMACross(Strategy):
    def on_start(self):
        # Подписываемся на данные
        self.subscribe_bars(bar_type=BarType.from_str("BTCUSDT.BINANCE-1-MINUTE-LAST"))

    def on_bar(self, bar):
        # Обрабатываем каждую свечу
        if self.ema_fast.value > self.ema_slow.value:
            if not self.portfolio.is_flat(bar.instrument_id):
                return
            self.buy(bar.instrument_id, quantity=Decimal("1.0"))
```

**Преимущества event-driven:**
- Реакция в микросекундах (благодаря Rust)
- Естественная поддержка асинхронности
- Легко тестировать (просто скармливаем события)

## Layer 5: Communication & Monitoring — как следить за роботом

**Задача:** Получать уведомления, управлять роботом, видеть метрики.

### **Telegram Bot в Freqtrade**

Freqtrade [встроил Telegram](https://www.freqtrade.io/en/stable/telegram-usage/):

```
/start - Запустить торговлю
/stop - Остановить
/status - Показать открытые позиции
/profit - Показать PnL
/forcebuy BTC/USDT - Принудительно открыть позицию
/forcesell 1 - Принудительно закрыть сделку #1
```

**Автоматические уведомления:**

- Сделка открыта/закрыта
- Стоп-лосс сработал
- Недостаточно баланса
- Ошибка подключения к бирже

### **Web UI и REST API**

[Freqtrade предоставляет REST API](https://www.freqtrade.io/en/stable/rest-api/) для внешней интеграции:

```bash
# Получить статус бота
curl http://localhost:8080/api/v1/status

# Получить историю сделок
curl http://localhost:8080/api/v1/trades

# Запустить бэктест через API
curl -X POST http://localhost:8080/api/v1/backtest \
  -H "Content-Type: application/json" \
  -d '{"strategy": "SampleStrategy", "timerange": "20230101-20231231"}'
```

[Есть готовый Web UI](https://github.com/freqtrade/frequi) на Vue.js:

- Дашборд с метриками в реальном времени
- График equity curve
- Список открытых позиций
- Управление стратегиями

### **Observability: метрики и логи**

Профессиональные системы экспортируют метрики в [Prometheus](https://prometheus.io/):

```python
from prometheus_client import Counter, Histogram

trades_total = Counter('trading_bot_trades_total', 'Total trades')
trade_profit = Histogram('trading_bot_trade_profit', 'Profit per trade')

def execute_trade(trade):
    trades_total.inc()
    profit = close_trade(trade)
    trade_profit.observe(profit)
```

Затем визуализируют в [Grafana](https://grafana.com/):

- Winrate за последние 24 часа
- PnL по дням
- Latency до биржи
- Количество rejected ордеров

## Паттерны проектирования для торговых роботов

### **1. Event Sourcing — когда важна история каждого решения**

[Event Sourcing сохраняет каждое изменение состояния как событие](https://www.infoq.com/articles/High-load-transactions-Reveno-CQRS-Event-sourcing-framework/).

**Зачем это нужно в трейдинге:**

- **Аудит:** Можно воспроизвести весь путь от данных до решения
- **Debugging:** "Почему робот купил BTC в 3 часа ночи?" — смотрим события
- **Регуляторные требования:** Некоторые юрисдикции требуют полный audit trail

**Пример событий:**

```
OrderPlaced     { order_id: 1, symbol: "BTCUSDT", side: "BUY", qty: 1.0, price: 50000 }
OrderFilled     { order_id: 1, filled_qty: 1.0, avg_price: 50005 }
PositionOpened  { position_id: 1, symbol: "BTCUSDT", qty: 1.0, entry_price: 50005 }
StopLossHit     { position_id: 1, exit_price: 49500 }
PositionClosed  { position_id: 1, pnl: -505 }
```

**Преимущества:**

- Состояние системы = сумма всех событий
- Можно "перемотать назад" и увидеть состояние в любой момент
- Event store — это источник истины

**Минусы:**

- Сложнее в реализации
- Требует больше места для хранения
- Запросы к "текущему состоянию" требуют обработки всех событий (решается через CQRS)

### **2. CQRS (Command Query Responsibility Segregation)**

[CQRS разделяет чтение и запись](https://github.com/BenjaminBest/StockTradingAnalysisWebsite).

**Проблема без CQRS:**

Запрос "какой у меня PnL за сегодня?" требует обработки тысяч событий.

**Решение:**

- **Write Model:** События записываются в Event Store
- **Read Model:** Отдельная БД с предрассчитанными агрегатами

```
Event Store (Write)          Read Model (Query)
─────────────────            ──────────────────
OrderPlaced                  PositionsTable
OrderFilled         ───▶     TradesTable
PositionOpened               PnLAggregates
PositionClosed               DailyStats
```

**Реализация в [Reveno framework](https://www.infoq.com/articles/High-load-transactions-Reveno-CQRS-Event-sourcing-framework/):**

- Throughput: **миллионы транзакций в секунду**
- Latency: **микросекунды**
- Все операции in-memory

### **3. Микросервисная архитектура**

[MBATS (Microservices Based Algorithmic Trading System)](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System) разбивает систему на независимые сервисы:

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Data Service   │   │ Strategy Service│   │ Execution Svc   │
│                 │   │                 │   │                 │
│ - Market Data   │──▶│ - Signals       │──▶│ - Order Routing │
│ - Normalization │   │ - Risk Checks   │   │ - Position Mgmt │
└─────────────────┘   └─────────────────┘   └─────────────────┘
         │                     │                      │
         └─────────────────────┼──────────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │  Message Bus      │
                     │  (Kafka/RabbitMQ) │
                     └───────────────────┘
```

**Преимущества:**

- Каждый сервис можно масштабировать независимо
- Разные языки для разных задач (Python для стратегий, Rust для исполнения)
- Отказ одного сервиса не роняет всю систему

**Недостатки:**

- Сложность в разработке и деплое
- Network latency между сервисами
- Нужна инфраструктура (Kubernetes, service mesh)

**Когда нужны микросервисы:**

- Объём данных требует горизонтального масштабирования
- Команда > 5 человек
- Разные части системы требуют разных SLA (стратегия может тормозить, исполнение — нет)

### **4. Actor Model для конкурентности**

[NautilusTrader и Hummingbot используют Actor Model](https://dev.to/kpcofgs/nautilustrader-the-open-source-trading-platform-5dji).

**Идея:**

- Каждый "актор" — независимая сущность с собственным state
- Акторы общаются через сообщения (message passing)
- Нет shared state → нет race conditions

**Пример:**

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ Data Actor   │──msg──▶│Strategy Actor│──msg──▶│Execution     │
│              │       │              │       │Actor         │
└──────────────┘       └──────────────┘       └──────────────┘
```

Каждый актор обрабатывает сообщения последовательно → thread-safe по дизайну.

**Реализация в Python:**

```python
import asyncio

class StrategyActor:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def run(self):
        while True:
            msg = await self.queue.get()
            await self.handle_message(msg)

    async def handle_message(self, msg):
        if msg['type'] == 'BAR':
            await self.on_bar(msg['data'])
```

## Полная архитектура: пример реального робота

Соберём всё вместе. Вот как выглядит production-ready торговый робот:

```
┌────────────────────────────────────────────────────────────┐
│                     Communication Layer                     │
│  Telegram Bot │ REST API │ WebSocket │ Prometheus Exporter │
└────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────┐
│                      Strategy Layer                         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Strategy 1  │  │  Strategy 2  │  │  Strategy N  │     │
│  │  (SMA Cross) │  │  (ML Model)  │  │  (Arbitrage) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────┐
│                  Execution & Risk Layer                     │
│                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────┐  │
│  │ Position Manager│  │  Risk Engine   │  │     SOR     │  │
│  └────────────────┘  └────────────────┘  └─────────────┘  │
└────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                     │
│                                                             │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Indicators │  │ Normalizers  │  │ Feature Engineer │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
└────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────┐
│                   Data Ingestion Layer                      │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐   │
│  │ Exchange Adapter│  │ Historical Data │  │  Cache   │   │
│  │   (WebSocket)   │  │   (TimescaleDB) │  │ (Redis)  │   │
│  └─────────────────┘  └─────────────────┘  └──────────┘   │
└────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Event Store     │
                    │  (PostgreSQL)     │
                    └───────────────────┘
```

**Технологический стек (пример):**

- **Language:** Python (стратегии), Rust (исполнение)
- **Data:** TimescaleDB (история), Redis (кэш), PostgreSQL (события)
- **Messaging:** Kafka или RabbitMQ
- **Monitoring:** Prometheus + Grafana
- **Deploy:** Docker + Kubernetes

## Чек-лист: как спроектировать архитектуру своего робота

### **1. Начните с монолита**

Не прыгайте сразу в микросервисы. [Monolithic design проще разворачивать](https://vocal.media/education/architectural-design-patterns-for-high-frequency-algo-trading-bots) и быстрее работает для большинства стратегий.

**Начальная структура:**

```
trading_bot/
├── adapters/          # Подключения к биржам
├── strategies/        # Торговые стратегии
├── indicators/        # Индикаторы
├── execution/         # Исполнение ордеров
├── persistence/       # Работа с БД
├── notifications/     # Telegram, email
└── main.py           # Entry point
```

### **2. Разделяйте слои с первого дня**

Даже в монолите используйте слои. Не пишите:

```python
# ПЛОХО
def trade():
    data = requests.get('https://api.binance.com/...')
    rsi = calculate_rsi(data)
    if rsi > 70:
        requests.post('https://api.binance.com/order', ...)
```

Пишите:

```python
# ХОРОШО
class DataAdapter:
    def fetch_ohlcv(self, symbol): ...

class Indicator:
    def calculate_rsi(self, prices): ...

class Strategy:
    def should_sell(self, rsi): ...

class Executor:
    def place_order(self, symbol, side, qty): ...
```

### **3. Проектируйте для тестирования**

**Dependency Injection:**

```python
class TradingBot:
    def __init__(self, data_source, executor):
        self.data_source = data_source
        self.executor = executor

    def run(self):
        data = self.data_source.get_data()
        if self.strategy.should_buy(data):
            self.executor.buy()

# Production
bot = TradingBot(
    data_source=BinanceAdapter(),
    executor=RealExecutor()
)

# Testing
bot = TradingBot(
    data_source=MockDataSource(),
    executor=MockExecutor()
)
```

### **4. Добавьте observability с первого дня**

Без метрик вы не поймёте, почему робот не работает.

**Минимальный набор:**

```python
import logging
from prometheus_client import Counter, Histogram, Gauge

# Метрики
trades_counter = Counter('trades_total', 'Total trades')
latency_hist = Histogram('order_latency_seconds', 'Order execution latency')
position_gauge = Gauge('open_positions', 'Number of open positions')

# Логирование
logger = logging.getLogger(__name__)

def execute_trade(trade):
    logger.info(f"Executing trade: {trade}")
    start = time.time()

    result = send_order(trade)

    latency = time.time() - start
    latency_hist.observe(latency)
    trades_counter.inc()

    logger.info(f"Trade executed in {latency:.3f}s: {result}")
```

### **5. Планируйте персистентность**

Что нужно сохранять:

- **Все ордера** (для аудита и налогов)
- **Позиции** (для расчёта PnL)
- **Настройки стратегий** (для воспроизводимости)
- **События** (если используете Event Sourcing)

**Минимальная схема БД:**

```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(100),
    symbol VARCHAR(20),
    side VARCHAR(10),
    quantity DECIMAL,
    price DECIMAL,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    quantity DECIMAL,
    entry_price DECIMAL,
    current_price DECIMAL,
    pnl DECIMAL,
    opened_at TIMESTAMP,
    closed_at TIMESTAMP
);
```

### **6. Когда переходить на микросервисы?**

**Сигналы, что пора:**

- Система обрабатывает > 100 инструментов
- Latency критична (HFT)
- Команда > 3 разработчиков
- Разные стратегии требуют разных ресурсов

**Не нужны микросервисы если:**

- Торгуете 1-10 инструментами
- Стратегия на дневных свечах
- Команда 1-2 человека
- Бюджет на инфраструктуру ограничен

## Реальные архитектуры: что используют профессионалы

### **Freqtrade (17k+ stars на GitHub)**

**Архитектура:** Модульный монолит

**Стек:**
- Python
- SQLite/PostgreSQL (персистентность)
- Pandas (обработка данных)
- ccxt (подключение к биржам)

**Подходит для:**
- Розничные трейдеры
- Крипто-валюты
- Стратегии на минутных/часовых свечах

**Не подходит для:**
- HFT (latency > 100ms)
- Фондовый рынок США (нет интеграции)

### **NautilusTrader (2k+ stars)**

**Архитектура:** Event-driven, Rust core + Python API

**Стек:**
- Rust (ядро)
- Python (стратегии)
- Redis (кэш)
- PostgreSQL (события)

**Подходит для:**
- Профессиональные трейдеры
- HFT (latency < 10ms)
- Крипто + традиционные рынки

**Не подходит для:**
- Новичков (крутая кривая обучения)
- Простых стратегий (overkill)

### **MBATS — Microservices Based Algo Trading**

**Архитектура:** Полноценные микросервисы

**Стек:**
- Docker + Kubernetes
- Kafka (message bus)
- Python/Java (сервисы)
- Grafana + Prometheus

**Подходит для:**
- Hedge funds
- Проприетарные фирмы
- Обработка > 1000 инструментов

**Не подходит для:**
- Индивидуальные трейдеры
- Малый бюджет на инфраструктуру

## Типичные ошибки в архитектуре торговых роботов

### **Ошибка 1: Всё в одном файле**

**Проблема:**

```python
# trading_bot.py (2000 строк)
import ccxt
import pandas as pd

def main():
    exchange = ccxt.binance(...)
    data = exchange.fetch_ohlcv(...)
    rsi = calculate_rsi(data)
    if rsi > 70:
        exchange.create_order(...)
    # ... ещё 1950 строк
```

**Почему плохо:**
- Невозможно тестировать
- Невозможно переиспользовать
- Невозможно поддерживать

### **Ошибка 2: Нет абстракций для бирж**

**Проблема:**

```python
# Логика работы с Binance размазана по всему коду
balance = binance.fetch_balance()['USDT']['free']
```

Когда захотите добавить Bybit — придётся менять код в 50 местах.

**Решение:**

```python
class Exchange(ABC):
    @abstractmethod
    def get_balance(self, currency): ...

    @abstractmethod
    def place_order(self, symbol, side, qty): ...

class BinanceExchange(Exchange):
    def get_balance(self, currency):
        return self.client.fetch_balance()[currency]['free']
```

### **Ошибка 3: Нет логирования и метрик**

Без логов вы не поймёте, почему робот сделал сделку в 3 часа ночи.

Без метрик вы не поймёте, что latency до биржи выросла с 50ms до 500ms.

### **Ошибка 4: Synchronous код в асинхронном мире**

Биржевые API асинхронные. WebSocket'ы асинхронные. Если используете `requests` вместо `aiohttp`, теряете производительность.

**Плохо:**

```python
import requests

def get_price(symbol):
    r = requests.get(f'https://api.binance.com/ticker/{symbol}')
    return r.json()['price']

prices = [get_price(s) for s in symbols]  # Последовательно!
```

**Хорошо:**

```python
import asyncio
import aiohttp

async def get_price(session, symbol):
    async with session.get(f'https://api.binance.com/ticker/{symbol}') as r:
        data = await r.json()
        return data['price']

async def get_all_prices(symbols):
    async with aiohttp.ClientSession() as session:
        tasks = [get_price(session, s) for s in symbols]
        return await asyncio.gather(*tasks)  # Параллельно!
```

### **Ошибка 5: Игнорирование backpressure**

Если данные приходят быстрее, чем вы их обрабатываете — память растёт до бесконечности.

**Решение:** Bounded queues

```python
import asyncio

# Очередь с ограничением
queue = asyncio.Queue(maxsize=1000)

async def producer():
    while True:
        data = await fetch_data()
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            logger.warning("Queue full, dropping data")

async def consumer():
    while True:
        data = await queue.get()
        process(data)
```

## Итоги

**Архитектура торгового робота — это не rocket science.** Но это и не `if price > 100: buy()` в одном файле.

**Ключевые принципы:**

1. **Разделяйте слои** — данные, обработка, стратегия, исполнение, коммуникация
2. **Используйте адаптеры** — не пишите логику биржи в стратегии
3. **Проектируйте для тестирования** — dependency injection, моки, абстракции
4. **Добавляйте observability** — логи, метрики, трейсинг
5. **Начинайте с монолита** — не усложняйте раньше времени

**Паттерны, которые работают:**

- **Event Sourcing** — для аудита и debugging
- **CQRS** — для быстрых запросов к состоянию
- **Actor Model** — для конкурентности без боли
- **Микросервисы** — когда масштаб требует

**Открытые реализации для изучения:**

- [Freqtrade](https://github.com/freqtrade/freqtrade) — для понимания модульного монолита
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) — для изучения Rust + Event-Driven
- [Hummingbot](https://github.com/hummingbot/hummingbot) — для market making архитектуры
- [MBATS](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System) — для микросервисов

**Следующий шаг:**

Если вы пишете своего робота — откройте исходники Freqtrade. Потратьте день на изучение архитектуры. Это сэкономит месяцы рефакторинга в будущем.

Если выбираете готовую платформу — [посмотрите сравнение фреймворков]({{site.baseurl}}/2026/02/24/sravnenie-lean-stocksharp-backtrader.html).

Архитектура определяет, насколько далеко вы зайдёте. Спагетти-код работает месяц. Правильная архитектура — годы.

---

**Полезные ссылки:**

Open-source торговые роботы:
- [Freqtrade](https://github.com/freqtrade/freqtrade)
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader)
- [Hummingbot](https://github.com/hummingbot/hummingbot)
- [MBATS Microservices Trading System](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System)

Архитектурные паттерны:
- [Architectural Design Patterns for HFT Bots](https://vocal.media/education/architectural-design-patterns-for-high-frequency-algo-trading-bots)
- [Trading System Design using MicroServices](https://medium.com/@datajedi/trading-system-design-using-microservices-256cda0dc60a)
- [Event Sourcing and CQRS in Trading Systems](https://www.infoq.com/articles/High-load-transactions-Reveno-CQRS-Event-sourcing-framework/)
- [Microservices Pattern: Event sourcing](https://microservices.io/patterns/data/event-sourcing.html)

Документация и гайды:
- [Freqtrade Architecture Guide](https://medium.com/@lufeiy/freqtrade-uncovered-how-machine-learning-powers-open-source-crypto-trading-25b1eab16ad9)
- [NautilusTrader Architecture](https://nautilustrader.io/docs/latest/concepts/architecture/)
- [FreqAI Machine Learning System](https://www.freqtrade.io/en/stable/freqai/)
