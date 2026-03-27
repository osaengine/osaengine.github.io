---
layout: post
title: "Как собрать стек open-source для алготрейдинга под MOEX: от данных до деплоя"
description: "Полный гайд по созданию бесплатной инфраструктуры для алготрейдинга на российском рынке. Python, Backtrader, Docker, TimescaleDB — собираем production-ready систему."
date: 2026-02-10
image: /assets/images/blog/opensource_stack_moex.png
tags: [open-source, MOEX, Python, Backtrader, StockSharp, инфраструктура]
---

"Я не хочу платить 60 тысяч в год за TSLab. Можно ли собрать всё на open-source?"

Можно. И я это сделал.

Последние 4 месяца я собирал стек для алготрейдинга на MOEX полностью из open-source инструментов.

Бесплатно. Независимо. Production-ready.

Вот что получилось.

## Зачем нужен свой стек

Год назад я платил:
- TSLab: 60 тыс/год
- MOEX AlgoPack: 55 тыс/год
- VPS для роботов: 12 тыс/год

**Итого: 127 тысяч в год.**

Потом я задумался: зачем платить за закрытые инструменты, если можно собрать аналог на open-source?

**Что я получил в итоге:**

1. **Полный контроль:** Я понимаю каждый компонент системы
2. **Независимость:** Не привязан к лицензиям и вендорам
3. **Гибкость:** Могу интегрировать ML, внешние API, криптовалюты
4. **Стоимость:** 0 рублей за софт (только VPS за 12 тыс/год)

**Что вошло в стек:**

- **Данные:** MOEX ISS API, Finam Export (бесплатно)
- **Backtesting:** Backtrader (Python, open-source)
- **Live-торговля:** StockSharp, Alor API (open-source)
- **База данных:** TimescaleDB (PostgreSQL для временных рядов)
- **Мониторинг:** Grafana + Prometheus
- **Деплой:** Docker + docker-compose
- **Очереди:** Redis (для кэширования), RabbitMQ (для микросервисов)

Всё бесплатно. Всё работает.

## Архитектура системы

Вот схема полного стека:

```
┌─────────────────────────────────────────────────────────┐
│                    ИСТОЧНИКИ ДАННЫХ                      │
│  MOEX ISS API │ Finam Export │ Alor API │ Tinkoff API   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  СБОР И ХРАНЕНИЕ ДАННЫХ                  │
│  Python скрипт (aiomoex, requests) → TimescaleDB         │
│  Исторические данные + Real-time стакан                  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     БЭКТЕСТИНГ                           │
│  Backtrader / VectorBT PRO / StockSharp Designer         │
│  Оптимизация параметров, walk-forward тесты              │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   ТОРГОВЫЕ РОБОТЫ                        │
│  Python (Alor API / Tinkoff API / StockSharp Connector)  │
│  Логика стратегий, управление позициями, риск-менеджмент │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  МОНИТОРИНГ И ЛОГИ                       │
│  Grafana (дашборды) + Prometheus (метрики)               │
│  Логи в файлы, alerts в Telegram                         │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                        ДЕПЛОЙ                            │
│  Docker (контейнеры) + docker-compose (оркестрация)      │
│  VPS / собственный сервер                                │
└─────────────────────────────────────────────────────────┘
```

Разберём каждый слой.

## Слой 1: Источники данных

[Я уже писал подробно про данные]({{site.baseurl}}/2026/01/27/gde-vzyat-dannye-dlya-algotreydinga-v-rossii.html). Вкратце:

### **MOEX ISS API (бесплатно)**

Официальный API Московской биржи. Даёт:
- Исторические свечи (минутки, часы, дни)
- Стакан (10 уровней)
- Информацию об инструментах

**Установка:**

```bash
pip install aiomoex
```

**Пример (получение свечей):**

```python
import aiomoex
import pandas as pd
import asyncio

async def get_candles(ticker, start_date, end_date):
    async with aiomoex.ISSClientSession():
        data = await aiomoex.get_board_candles(
            ticker,
            start=start_date,
            end=end_date,
            interval=60  # 60 минут
        )
        df = pd.DataFrame(data)
        df['begin'] = pd.to_datetime(df['begin'])
        return df

# Использование
df = asyncio.run(get_candles('SBER', '2024-01-01', '2025-01-01'))
print(df.head())
```

**Ограничения:**
- 10 запросов в секунду (для бесплатного доступа)
- Глубина истории: несколько лет (зависит от инструмента)

### **Finam Export (бесплатно)**

Исторические данные для акций, фьючерсов.

**Python библиотека:**

```bash
pip install finam-export
```

**Пример:**

```python
from finam.export import Exporter, Market, Timeframe

exporter = Exporter()
data = exporter.download(
    market=Market.SHARES,
    ticker='SBER',
    start_date='2024-01-01',
    end_date='2025-01-01',
    timeframe=Timeframe.HOURLY
)

print(data.head())
```

### **MOEX AlgoPack (платно, 55 тыс/год)**

Глубокий стакан (50 уровней), тиковые данные, все сделки.

[Библиотека moexalgo](https://pypi.org/project/moexalgo/):

```bash
pip install moexalgo
```

**Пример:**

```python
import moexalgo

# Нужен токен от https://data.moex.com/
session = moexalgo.session(token='ваш_токен')

# Получаем стакан
orderbook = session.orderbook('SBER')
print(orderbook.head())
```

**Моё решение:** Для простых стратегий — MOEX ISS + Finam (бесплатно). Для HFT — AlgoPack (платно, но незаменимо).

## Слой 2: База данных (TimescaleDB)

Для хранения исторических данных и тиков нужна база.

**Почему TimescaleDB?**

- Это PostgreSQL + расширение для временных рядов
- Быстрее InfluxDB на вставке данных
- SQL-запросы (знакомый синтаксис)
- Сжатие данных (экономия диска)

**Установка (Docker):**

```yaml
# docker-compose.yml
version: '3.8'

services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    container_name: timescaledb
    environment:
      POSTGRES_PASSWORD: your_password
      POSTGRES_DB: algotrading
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data

volumes:
  timescale_data:
```

Запуск:

```bash
docker-compose up -d
```

**Создание таблицы для свечей:**

```sql
CREATE TABLE candles (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT
);

-- Превращаем в hypertable (TimescaleDB)
SELECT create_hypertable('candles', 'time');

-- Индекс для быстрого поиска
CREATE INDEX idx_ticker_time ON candles (ticker, time DESC);
```

**Вставка данных из Python:**

```python
import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host="localhost",
    database="algotrading",
    user="postgres",
    password="your_password"
)

def save_candles(df, ticker):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO candles (time, ticker, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (row['begin'], ticker, row['open'], row['high'],
              row['low'], row['close'], row['volume']))
    conn.commit()

# Сохраняем свечи SBER
df = asyncio.run(get_candles('SBER', '2024-01-01', '2025-01-01'))
save_candles(df, 'SBER')
```

**Сжатие данных (экономия места):**

```sql
-- Автоматическое сжатие данных старше 7 дней
SELECT add_compression_policy('candles', INTERVAL '7 days');
```

**Результат:** Хранение 5 лет дневных свечей по 100 акциям — ~500 МБ (без сжатия), ~150 МБ (со сжатием).

## Слой 3: Бэктестинг (Backtrader)

[Backtrader](https://github.com/WISEPLAT/backtrader_moexalgo) — самый популярный фреймворк для бэктестинга на Python.

**Установка:**

```bash
pip install backtrader
pip install git+https://github.com/WISEPLAT/backtrader_moexalgo  # MOEX интеграция
```

**Простая стратегия (SMA-кросс):**

```python
import backtrader as bt

class SmaCross(bt.Strategy):
    params = (('fast', 20), ('slow', 50),)

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if self.crossover > 0:  # Пересечение снизу вверх
            if not self.position:
                self.buy(size=10)
        elif self.crossover < 0:  # Пересечение сверху вниз
            if self.position:
                self.close()

# Загружаем данные из TimescaleDB
import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host="localhost", database="algotrading",
    user="postgres", password="your_password"
)

df = pd.read_sql("""
    SELECT time, open, high, low, close, volume
    FROM candles
    WHERE ticker = 'SBER'
    ORDER BY time
""", conn, index_col='time', parse_dates=['time'])

# Backtrader Data Feed
data = bt.feeds.PandasData(dataname=df)

cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)
cerebro.adddata(data)
cerebro.broker.setcash(1000000)
cerebro.broker.setcommission(commission=0.0005)  # 0.05% комиссия

print(f'Начальный капитал: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'Конечный капитал: {cerebro.broker.getvalue():.2f}')

cerebro.plot()
```

**Интеграция с MOEX через backtrader_moexalgo:**

```python
from backtrader_moexalgo import MoexAlgoStore
import backtrader as bt

# Подключение к MOEX AlgoPack
store = MoexAlgoStore(token='ваш_токен')

# Создаём datafeed
data = store.getdata(
    dataname='SBER',
    timeframe=bt.TimeFrame.Minutes,
    compression=60,  # 60 минут
    fromdate=datetime(2024, 1, 1),
    todate=datetime(2025, 1, 1)
)

cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(SmaCross)
cerebro.run()
```

**Walk-forward оптимизация:**

```python
# Оптимизация параметров (перебор)
cerebro.optstrategy(SmaCross, fast=range(10, 30, 5), slow=range(40, 60, 5))

results = cerebro.run()

# Лучшая стратегия
best = max(results, key=lambda x: x[0].broker.getvalue())
print(f'Лучшие параметры: fast={best[0].params.fast}, slow={best[0].params.slow}')
```

## Слой 4: Live-торговля (Alor API, Tinkoff API, StockSharp)

Для боевой торговли нужно подключиться к брокеру.

### **Вариант 1: Alor API (WebSocket + REST)**

[Документация Alor API](https://alor.dev/)

**Установка:**

```bash
pip install aiohttp websockets
```

**Пример (получение стакана через WebSocket):**

```python
import asyncio
import websockets
import json

TOKEN = "ваш_токен_alor"

async def subscribe_orderbook():
    uri = "wss://api.alor.ru/ws"

    async with websockets.connect(uri) as ws:
        # Авторизация
        await ws.send(json.dumps({
            "opcode": "authorize",
            "token": TOKEN
        }))

        # Подписка на стакан SBER
        await ws.send(json.dumps({
            "opcode": "subscribe",
            "code": "SBER",
            "market": "MOEX",
            "depth": 20
        }))

        # Получаем обновления
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if 'bids' in data and 'asks' in data:
                print(f"BIDs: {data['bids'][:3]}")
                print(f"ASKs: {data['asks'][:3]}")

asyncio.run(subscribe_orderbook())
```

**Отправка заявки:**

```python
import requests

url = "https://api.alor.ru/commandapi/warptrans/TRADE/v2/client/orders/actions/market"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "X-ALOR-REQID": "unique_request_id"
}

payload = {
    "instrument": {
        "symbol": "SBER",
        "exchange": "MOEX"
    },
    "side": "buy",
    "type": "market",
    "quantity": 10,
    "user": {
        "portfolio": "ваш_портфель"
    }
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### **Вариант 2: Tinkoff Invest API**

```bash
pip install tinkoff-investments
```

**Пример:**

```python
from tinkoff.invest import Client, OrderDirection, OrderType

TOKEN = "ваш_токен_tinkoff"

with Client(TOKEN) as client:
    # Получаем FIGI (идентификатор) SBER
    instruments = client.instruments.shares()
    sber = [i for i in instruments.instruments if i.ticker == "SBER"][0]

    # Отправляем рыночную заявку на покупку
    order = client.orders.post_order(
        figi=sber.figi,
        quantity=10,
        direction=OrderDirection.ORDER_DIRECTION_BUY,
        order_type=OrderType.ORDER_TYPE_MARKET,
        account_id="ваш_account_id"
    )

    print(f"Заявка отправлена: {order.order_id}")
```

### **Вариант 3: StockSharp Connector (C#)**

[StockSharp](https://github.com/StockSharp/StockSharp) — open-source платформа на C# с поддержкой 90+ бирж мира (включая MOEX, Binance, Interactive Brokers, Bybit).

**Установка через NuGet:**

```bash
dotnet add package StockSharp.Alor
dotnet add package StockSharp.MOEX
```

**Пример подключения к Alor:**

```csharp
using StockSharp.Alor;
using StockSharp.Messages;

var connector = new AlorTrader();
connector.Login = "ваш_логин";
connector.Password = "ваш_пароль";

connector.Connected += () => {
    Console.WriteLine("Подключено к Alor");

    var security = new Security { Id = "SBER@TQBR" };
    connector.RegisterSecurity(security);
    connector.SubscribeMarketDepth(security);
};

connector.NewMarketDepth += depth => {
    Console.WriteLine($"BID: {depth.Bids[0].Price}, ASK: {depth.Asks[0].Price}");
};

connector.Connect();
```

**Моё решение:**
- Для Python-стратегий: **Alor API** или **Tinkoff API**
- Для C# и интеграции со StockSharp Designer: **StockSharp Connector**

## Слой 5: Мониторинг (Grafana + Prometheus)

Когда робот торгует на боевом счёте, нужен мониторинг.

**Что отслеживаем:**
- PnL (profit and loss) в реальном времени
- Количество открытых позиций
- Просадка (drawdown)
- Ошибки подключения к API

### **Установка Prometheus + Grafana (Docker):**

```yaml
# docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  grafana_data:
```

**prometheus.yml:**

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'trading_bot'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

**Экспорт метрик из Python-робота:**

```bash
pip install prometheus-client
```

```python
from prometheus_client import start_http_server, Gauge
import time

# Метрики
pnl_gauge = Gauge('trading_pnl', 'Profit and Loss')
position_gauge = Gauge('trading_position', 'Open Position Size')
drawdown_gauge = Gauge('trading_drawdown', 'Current Drawdown')

# Запускаем HTTP-сервер для Prometheus
start_http_server(8000)

# Обновляем метрики в торговом цикле
while True:
    current_pnl = calculate_pnl()  # Ваша функция
    current_position = get_position_size()
    current_drawdown = calculate_drawdown()

    pnl_gauge.set(current_pnl)
    position_gauge.set(current_position)
    drawdown_gauge.set(current_drawdown)

    time.sleep(10)
```

**Настройка дашборда в Grafana:**

1. Открываем http://localhost:3000
2. Логин: `admin`, пароль: `admin`
3. Add Data Source → Prometheus (URL: `http://prometheus:9090`)
4. Создаём дашборд с графиками:
   - `trading_pnl` (PnL во времени)
   - `trading_position` (размер позиции)
   - `trading_drawdown` (просадка)

**Алерты в Telegram:**

```bash
pip install python-telegram-bot
```

```python
import telegram

bot = telegram.Bot(token='ваш_токен_telegram')

def send_alert(message):
    bot.send_message(chat_id='ваш_chat_id', text=message)

# Пример: отправка алерта при просадке > 10%
if current_drawdown > 0.10:
    send_alert(f"⚠️ Просадка превысила 10%: {current_drawdown:.2%}")
```

## Слой 6: Деплой (Docker)

Всё собираем в Docker-контейнеры.

**Структура проекта:**

```
algotrading/
├── docker-compose.yml
├── prometheus.yml
├── trading_bot/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── bot.py
│   └── strategies/
│       └── sma_cross.py
├── data_collector/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── collector.py
└── grafana/
    └── dashboards/
```

**docker-compose.yml (полный стек):**

```yaml
version: '3.8'

services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    container_name: timescaledb
    environment:
      POSTGRES_PASSWORD: your_password
      POSTGRES_DB: algotrading
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data

  redis:
    image: redis:alpine
    container_name: redis
    ports:
      - "6379:6379"

  data_collector:
    build: ./data_collector
    container_name: data_collector
    depends_on:
      - timescaledb
      - redis
    environment:
      DB_HOST: timescaledb
      DB_PASSWORD: your_password
      REDIS_HOST: redis

  trading_bot:
    build: ./trading_bot
    container_name: trading_bot
    depends_on:
      - timescaledb
      - redis
    environment:
      DB_HOST: timescaledb
      DB_PASSWORD: your_password
      REDIS_HOST: redis
      ALOR_TOKEN: ${ALOR_TOKEN}
    ports:
      - "8000:8000"  # Для Prometheus

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  timescale_data:
  grafana_data:
```

**Dockerfile для торгового робота:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]
```

**requirements.txt:**

```
backtrader
aiomoex
aiohttp
websockets
psycopg2-binary
redis
prometheus-client
python-telegram-bot
```

**Запуск всего стека:**

```bash
docker-compose up -d
```

**Проверка:**

```bash
docker-compose ps  # Все сервисы должны быть UP
docker-compose logs trading_bot  # Логи робота
```

## Полный пример: SMA-стратегия от бэктеста до live

### **Шаг 1: Собираем данные**

```python
# data_collector/collector.py
import asyncio
import aiomoex
import psycopg2
import pandas as pd
from datetime import datetime, timedelta

async def collect_data(ticker, days=365):
    conn = psycopg2.connect(
        host="timescaledb",
        database="algotrading",
        user="postgres",
        password="your_password"
    )

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    async with aiomoex.ISSClientSession():
        data = await aiomoex.get_board_candles(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval=60
        )

    df = pd.DataFrame(data)
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO candles (time, ticker, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (row['begin'], ticker, row['open'], row['high'],
              row['low'], row['close'], row['volume']))

    conn.commit()
    print(f"Собрано {len(df)} свечей для {ticker}")

# Собираем данные для нескольких акций
asyncio.run(collect_data('SBER'))
asyncio.run(collect_data('GAZP'))
asyncio.run(collect_data('LKOH'))
```

### **Шаг 2: Бэктест**

```python
# backtest.py
import backtrader as bt
import psycopg2
import pandas as pd

class SmaCross(bt.Strategy):
    params = (('fast', 20), ('slow', 50),)

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if self.crossover > 0:
            if not self.position:
                self.buy(size=10)
        elif self.crossover < 0:
            if self.position:
                self.close()

# Загружаем данные
conn = psycopg2.connect(
    host="localhost", database="algotrading",
    user="postgres", password="your_password"
)

df = pd.read_sql("""
    SELECT time, open, high, low, close, volume
    FROM candles
    WHERE ticker = 'SBER' AND time >= '2024-01-01'
    ORDER BY time
""", conn, index_col='time', parse_dates=['time'])

data = bt.feeds.PandasData(dataname=df)

cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)
cerebro.adddata(data)
cerebro.broker.setcash(1000000)
cerebro.broker.setcommission(commission=0.0005)

print(f'Начальный капитал: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'Конечный капитал: {cerebro.broker.getvalue():.2f}')
```

### **Шаг 3: Live-торговля**

```python
# trading_bot/bot.py
import asyncio
import websockets
import json
import requests
from collections import deque
from prometheus_client import start_http_server, Gauge

ALOR_TOKEN = "ваш_токен"
pnl_gauge = Gauge('trading_pnl', 'PnL')

class SmaCrossLive:
    def __init__(self, ticker, fast=20, slow=50):
        self.ticker = ticker
        self.fast = fast
        self.slow = slow
        self.prices = deque(maxlen=slow)
        self.position = 0

    def on_price(self, price):
        self.prices.append(price)

        if len(self.prices) < self.slow:
            return  # Недостаточно данных

        fast_ma = sum(list(self.prices)[-self.fast:]) / self.fast
        slow_ma = sum(self.prices) / self.slow

        # Сигнал на покупку
        if fast_ma > slow_ma and self.position == 0:
            self.buy()

        # Сигнал на продажу
        elif fast_ma < slow_ma and self.position > 0:
            self.sell()

    def buy(self):
        url = "https://api.alor.ru/commandapi/warptrans/TRADE/v2/client/orders/actions/market"
        payload = {
            "instrument": {"symbol": self.ticker, "exchange": "MOEX"},
            "side": "buy",
            "type": "market",
            "quantity": 10,
            "user": {"portfolio": "ваш_портфель"}
        }
        headers = {"Authorization": f"Bearer {ALOR_TOKEN}"}
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            self.position = 10
            print(f"✅ Куплено {self.ticker}: 10 лотов")

    def sell(self):
        url = "https://api.alor.ru/commandapi/warptrans/TRADE/v2/client/orders/actions/market"
        payload = {
            "instrument": {"symbol": self.ticker, "exchange": "MOEX"},
            "side": "sell",
            "type": "market",
            "quantity": 10,
            "user": {"portfolio": "ваш_портфель"}
        }
        headers = {"Authorization": f"Bearer {ALOR_TOKEN}"}
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            self.position = 0
            print(f"✅ Продано {self.ticker}: 10 лотов")

async def run_bot():
    strategy = SmaCrossLive('SBER')

    uri = "wss://api.alor.ru/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"opcode": "authorize", "token": ALOR_TOKEN}))
        await ws.send(json.dumps({
            "opcode": "subscribe",
            "code": "SBER",
            "market": "MOEX",
            "depth": 1
        }))

        while True:
            message = await ws.recv()
            data = json.loads(message)

            if 'asks' in data and len(data['asks']) > 0:
                price = data['asks'][0]['price']
                strategy.on_price(price)

# Запуск
start_http_server(8000)  # Для Prometheus
asyncio.run(run_bot())
```

## Альтернативы и сравнение

### **VectorBT PRO vs Backtrader**

| Критерий | VectorBT PRO | Backtrader |
|----------|--------------|----------|
| Скорость | Очень быстрый (NumPy) | Медленнее (Python loop) |
| Гибкость | Ограничена | Полная кастомизация |
| Оптимизация | Встроенная (grid search) | Нужно писать вручную |
| Live-торговля | Через [StrateQueue](https://medium.com/@samuel.tinnerholm/from-backtest-to-live-going-live-with-vectorbt-in-2025-step-by-step-guide-681ff5e3376e) | Нативная поддержка |
| Стоимость | Платная (от $50/мес) | Бесплатная (open-source) |

**Моё мнение:** Для быстрого бэктеста — **VectorBT PRO**. Для полного контроля — **Backtrader**.

### **StockSharp vs Backtrader**

| Критерий | StockSharp | Backtrader |
|----------|------------|-----------|
| Язык | C# | Python |
| Экосистема | Меньше библиотек | Огромная (pandas, sklearn, TensorFlow) |
| GUI | Designer (визуальный) | Только код |
| Брокеры | QUIK, Transaq, Alor и т.д. | Нужна интеграция вручную |

**Моё мнение:** Если любите C# и нужен GUI — **StockSharp**. Если Python и ML — **Backtrader**.

## Проблемы и решения

### **Проблема 1: MOEX ISS лимит 10 запросов/сек**

**Решение:** Кэширование через Redis.

```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_candles_cached(ticker, start, end):
    cache_key = f"{ticker}:{start}:{end}"
    cached = r.get(cache_key)

    if cached:
        return json.loads(cached)

    # Запрашиваем с MOEX
    data = asyncio.run(get_candles(ticker, start, end))
    r.setex(cache_key, 3600, json.dumps(data.to_dict()))  # Кэш на 1 час

    return data
```

### **Проблема 2: Docker контейнеры падают при разрыве соединения**

**Решение:** Restart policy в docker-compose.

```yaml
services:
  trading_bot:
    restart: unless-stopped  # Автоматический перезапуск
```

И reconnect логика в коде:

```python
async def run_bot_with_reconnect():
    while True:
        try:
            await run_bot()
        except Exception as e:
            print(f"❌ Ошибка: {e}. Переподключение через 5 сек...")
            await asyncio.sleep(5)
```

### **Проблема 3: TimescaleDB занимает много места**

**Решение:** Политика удаления старых данных.

```sql
-- Удаляем данные старше 2 лет
SELECT add_retention_policy('candles', INTERVAL '2 years');
```

## Чек-лист: собираем стек с нуля

### **День 1: Данные и база**

1. ✅ Установить Docker + docker-compose
2. ✅ Запустить TimescaleDB (docker-compose up)
3. ✅ Создать таблицу candles
4. ✅ Написать скрипт сбора данных (aiomoex → TimescaleDB)
5. ✅ Собрать историю по 5-10 инструментам

### **День 2-3: Бэктестинг**

1. ✅ Установить Backtrader
2. ✅ Написать простую стратегию (SMA-кросс)
3. ✅ Протестировать на исторических данных
4. ✅ Оптимизировать параметры (optstrategy)

### **День 4-5: Live-торговля**

1. ✅ Получить токен от брокера (Alor/Tinkoff)
2. ✅ Подключиться к WebSocket (получение стакана)
3. ✅ Реализовать отправку заявок
4. ✅ Протестировать на демо-счёте

### **День 6: Мониторинг**

1. ✅ Запустить Prometheus + Grafana
2. ✅ Добавить экспорт метрик из робота
3. ✅ Создать дашборд (PnL, позиция, просадка)
4. ✅ Настроить алерты в Telegram

### **День 7: Деплой**

1. ✅ Упаковать всё в Docker
2. ✅ Запустить на VPS
3. ✅ Проверить логи и мониторинг
4. ✅ Запустить на реальном счёте (с минимальным объёмом)

## Итоги

**Собрать свой стек для алготрейдинга на MOEX — реально.**

**Что нужно:**
- Python (основной язык)
- TimescaleDB (данные)
- Backtrader (бэктестинг)
- Alor/Tinkoff API (live-торговля)
- Grafana + Prometheus (мониторинг)
- Docker (деплой)

**Стоимость:** 0 рублей за софт (только VPS ~12 тыс/год).

**Время:** 7 дней от идеи до боевой торговли (если знаете Python).

**Плюсы:**
- Полный контроль
- Независимость
- Гибкость (ML, внешние API, криптовалюты)

**Минусы:**
- Нужно знать программирование
- Нужно поддерживать инфраструктуру
- Нет визуального GUI (всё через код)

**Моё мнение:**

Если вы программист или готовы учить Python — собирайте свой стек. Это инвестиция в независимость.

[Если новичок — начните с конструкторов]({{site.baseurl}}/2026/01/06/mozhno-li-nachat-put-s-konstruktorov.html), затем переходите на open-source.

---

**Полезные ссылки:**

Open-source платформы:
- [Backtrader](https://www.backtrader.com/)
- [Backtrader MOEX Integration](https://github.com/WISEPLAT/backtrader_moexalgo)
- [StockSharp](https://github.com/StockSharp/StockSharp)
- [VectorBT PRO](https://vectorbt.pro/)
- [Jesse Trading Bot](https://github.com/jesse-ai/jesse)

Библиотеки для Python:
- [aiomoex (MOEX ISS API)](https://pypi.org/project/aiomoex/)
- [moexalgo (AlgoPack)](https://pypi.org/project/moexalgo/)
- [finam-export](https://pypi.org/project/finam-export/)

Инфраструктура:
- [TimescaleDB](https://www.timescale.com/)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [Docker](https://www.docker.com/)

Статьи и исследования:
- [Алготрейдинг.рф: Backtrader платформа](https://алготрейдинг.рф/backtrader/platforma-backtrader/)
- [Habr: TimescaleDB](https://habr.com/ru/companies/oleg-bunin/articles/464303/)
- [Medium: VectorBT Going Live 2025](https://medium.com/@samuel.tinnerholm/from-backtest-to-live-going-live-with-vectorbt-in-2025-step-by-step-guide-681ff5e3376e)
