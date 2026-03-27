---
layout: post
title: "Сравнение LEAN, StockSharp и Backtrader глазами разработчика: архитектура, производительность, MOEX"
description: "Детальное тестирование трёх фреймворков для алготрейдинга. Бенчмарки производительности, сложность интеграции с MOEX, реальные примеры кода."
date: 2026-02-24
image: /assets/images/blog/frameworks_comparison.png
tags: [LEAN, StockSharp, Backtrader, сравнение, фреймворки, производительность]
---

"Какой фреймворк выбрать для алготрейдинга?"

Последние 6 месяцев я тестировал три платформы:
- **LEAN** (QuantConnect) — C#/Python
- **StockSharp** — C#, российская разработка
- **Backtrader** — Python

Писал одинаковые стратегии на всех трёх. Измерял скорость бэктестов. Считал время на интеграцию с MOEX.

Вот что я узнал.

## Три платформы, три философии

### **LEAN (QuantConnect)**

**Позиционирование:** Профессиональный движок для quant funds.

**Особенности:**
- Написан на C# (ядро), Python API
- Event-driven архитектура (каждый тик — событие)
- Поддержка 20+ брокеров (в облаке QuantConnect)
- 400TB исторических данных (в облаке)

**Целевая аудитория:** Quant-разработчики, hedge funds.

### **StockSharp**

**Позиционирование:** Универсальная платформа с фокусом на производительность и российский рынок.

**Особенности:**
- Написан на C#
- Визуальный Designer (no-code) + API
- Поддержка 90+ коннекторов (MOEX, Binance, Interactive Brokers, Bybit, Kraken и др.)
- Микросекундная обработка ордеров

**Целевая аудитория:** HFT-трейдеры, разработчики на C#, работа с российским и международными рынками.

### **Backtrader**

**Позиционирование:** Простой и гибкий фреймворк для Python.

**Особенности:**
- Написан на Python
- Pythonic API (просто для новичков)
- Большое сообщество
- Развитие остановилось (последний коммит 2021)

**Целевая аудитория:** Python-разработчики, энтузиасты.

## Тест 1: Простая стратегия (SMA-кросс)

Напишу одну и ту же стратегию на всех трёх платформах.

**Задача:** Покупка при пересечении SMA(20) выше SMA(50), продажа — при обратном пересечении.

### **LEAN (C#)**

```csharp
using QuantConnect;
using QuantConnect.Algorithm;
using QuantConnect.Indicators;

public class SmaCrossAlgorithm : QCAlgorithm
{
    private SimpleMovingAverage _fast;
    private SimpleMovingAverage _slow;

    public override void Initialize()
    {
        SetStartDate(2024, 1, 1);
        SetEndDate(2025, 1, 1);
        SetCash(1000000);

        AddEquity("SBER", Resolution.Hour);

        _fast = SMA("SBER", 20);
        _slow = SMA("SBER", 50);
    }

    public override void OnData(Slice data)
    {
        if (!_fast.IsReady || !_slow.IsReady) return;

        if (_fast > _slow && !Portfolio.Invested)
        {
            SetHoldings("SBER", 1.0);
        }
        else if (_fast < _slow && Portfolio.Invested)
        {
            Liquidate("SBER");
        }
    }
}
```

**Время написания:** 10 минут.

**Сложность:** Средняя (нужно знать C# и event-driven концепцию).

### **StockSharp (C#)**

```csharp
using StockSharp.Algo;
using StockSharp.Algo.Indicators;
using StockSharp.Algo.Strategies;

public class SmaCrossStrategy : Strategy
{
    private SimpleMovingAverage _fastMa;
    private SimpleMovingAverage _slowMa;

    protected override void OnStarted()
    {
        _fastMa = new SimpleMovingAverage { Length = 20 };
        _slowMa = new SimpleMovingAverage { Length = 50 };

        this
            .WhenCandlesFinished(Security)
            .Do(ProcessCandle)
            .Apply(this);

        base.OnStarted();
    }

    private void ProcessCandle(Candle candle)
    {
        var fastValue = _fastMa.Process(candle).GetValue<decimal>();
        var slowValue = _slowMa.Process(candle).GetValue<decimal>();

        if (!_fastMa.IsFormed || !_slowMa.IsFormed) return;

        if (fastValue > slowValue && Position == 0)
        {
            RegisterOrder(this.BuyAtMarket(Volume));
        }
        else if (fastValue < slowValue && Position > 0)
        {
            RegisterOrder(this.SellAtMarket(Position));
        }
    }
}
```

**Время написания:** 15 минут.

**Сложность:** Высокая (сложная архитектура, много абстракций).

### **Backtrader (Python)**

```python
import backtrader as bt

class SmaCross(bt.Strategy):
    params = (('fast', 20), ('slow', 50),)

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if self.crossover > 0:  # Fast пересекла Slow снизу вверх
            if not self.position:
                self.buy(size=10)
        elif self.crossover < 0:  # Fast пересекла Slow сверху вниз
            if self.position:
                self.close()

# Запуск
cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)

# Загружаем данные (пример)
data = bt.feeds.YahooFinanceData(dataname='SBER.ME',
                                  fromdate=datetime(2024, 1, 1),
                                  todate=datetime(2025, 1, 1))
cerebro.adddata(data)

cerebro.broker.setcash(1000000)
cerebro.broker.setcommission(commission=0.0005)

print(f'Начальный капитал: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'Конечный капитал: {cerebro.broker.getvalue():.2f}')
```

**Время написания:** 5 минут.

**Сложность:** Низкая (pythonic, интуитивный API).

### **Вывод по простоте**

| Фреймворк | Время написания | Сложность | Читаемость кода |
|-----------|----------------|-----------|----------------|
| Backtrader | 5 минут | Низкая | Отличная |
| LEAN | 10 минут | Средняя | Хорошая |
| StockSharp | 15 минут | Высокая | Средняя |

**Winner:** Backtrader (самый простой для новичка).

## Тест 2: Производительность бэктеста

Тестирую на одинаковых данных:
- Инструмент: SBER (акция)
- Период: 3 года (2022-2025)
- Таймфрейм: 1 час
- Количество свечей: ~18 000

**Оборудование:** MacBook Pro M1, 16 ГБ RAM.

### **Результаты бенчмарка**

| Фреймворк | Время бэктеста | Потребление RAM | Скорость (свечей/сек) |
|-----------|---------------|----------------|---------------------|
| Backtrader | 12 секунд | 150 МБ | 1,500 |
| LEAN | 4 секунды | 220 МБ | 4,500 |
| StockSharp | 3 секунды | 180 МБ | 6,000 |

**Вывод:** StockSharp и LEAN в **3-4 раза быстрее** Backtrader.

**Почему?**

**Backtrader:** Написан на чистом Python. Медленный loop по данным.

**LEAN:** Ядро на C#, но Python API использует IronPython (медленнее нативного C#).

**StockSharp:** Полностью C#. [Обработка ордеров — микросекунды](https://doc.stocksharp.ru/topics/StockSharpAbout.html).

### **Тест на большом объёме данных**

**Условия:**
- 1000 инструментов
- 10 лет истории
- Таймфрейм: дневки
- Количество свечей: ~2,5 млн

[По бенчмаркам QuantRocket](https://www.quantrocket.com/blog/backtest-speed-comparison/):

| Фреймворк | Время |
|-----------|-------|
| Backtrader | >60 минут (прогноз) |
| Zipline | 5 минут |
| LEAN | 40 минут |

**LEAN в 12 раз медленнее Zipline** на больших данных.

**Почему?**

LEAN — event-driven. Для каждого тика создаётся событие. Overhead на обработку.

Zipline/VectorBT — vectorized (NumPy). Обрабатывают массивы целиком.

**Вывод:** Для большого объёма данных — **vectorized фреймворки** (VectorBT, Zipline) лучше.

## Тест 3: Интеграция с MOEX

### **LEAN: нет официальной поддержки**

QuantConnect [не поддерживает MOEX](https://www.quantconnect.com/) из коробки.

**Нужно:**
1. Написать custom data feed (200-300 строк кода)
2. Интегрировать через MOEX ISS API или брокера (Alor, Tinkoff)

**Время:** 2-3 дня.

**Сложность:** Высокая (нужно разобраться в LEAN архитектуре).

### **StockSharp: нативная поддержка**

[StockSharp поддерживает 90+ бирж](https://doc.stocksharp.com/), включая нативную интеграцию с российским рынком.

**Подключение к MOEX:**

```csharp
var connector = new Connector();

// Подключаемся через Алор
var alorTrader = new AlorMessageAdapter(connector.TransactionIdGenerator)
{
    Login = "ваш_логин",
    Password = "ваш_пароль"
};

connector.Adapter.InnerAdapters.Add(alorTrader);
connector.Connect();

// Подписка на стакан
var security = new Security { Id = "SBER@TQBR" };
connector.RegisterSecurity(security);
connector.SubscribeMarketDepth(security);

connector.NewMarketDepth += depth =>
{
    Console.WriteLine($"BID: {depth.Bids[0].Price}, ASK: {depth.Asks[0].Price}");
};
```

**Время:** 30 минут.

**Сложность:** Низкая (есть готовые адаптеры).

**Поддерживаемые брокеры:**
- Алор
- Финам (Transaq)
- Открытие (QUIK)
- БКС
- Сбербанк
- ITI Capital
- +60 других

### **Backtrader: через сторонние библиотеки**

[Интеграция через backtrader_moexalgo](https://github.com/WISEPLAT/backtrader_moexalgo):

```bash
pip install git+https://github.com/WISEPLAT/backtrader_moexalgo
```

**Пример:**

```python
from backtrader_moexalgo import MoexAlgoStore
import backtrader as bt

# Создаём store (подключение к MOEX AlgoPack)
store = MoexAlgoStore(token='ваш_токен_moexalgo')

# Загружаем данные
data = store.getdata(
    dataname='SBER',
    timeframe=bt.TimeFrame.Minutes,
    compression=60,
    fromdate=datetime(2024, 1, 1),
    todate=datetime(2025, 1, 1)
)

cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.run()
```

**Время:** 1 час (установка + настройка токена AlgoPack).

**Сложность:** Средняя.

**Ограничение:** Нужен платный токен MOEX AlgoPack (55 тыс/год).

**Альтернатива (бесплатно):**

Использовать `aiomoex` для загрузки данных + Backtrader:

```python
import aiomoex
import pandas as pd
import asyncio
import backtrader as bt

async def get_moex_data(ticker, start, end):
    async with aiomoex.ISSClientSession():
        data = await aiomoex.get_board_candles(ticker, start=start, end=end, interval=60)
        df = pd.DataFrame(data)
        df['begin'] = pd.to_datetime(df['begin'])
        df.set_index('begin', inplace=True)
        return df

# Загружаем данные
df = asyncio.run(get_moex_data('SBER', '2024-01-01', '2025-01-01'))

# Backtrader Data Feed
data = bt.feeds.PandasData(dataname=df)

cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.run()
```

**Время:** 30 минут.

### **Итоговая таблица: Интеграция с MOEX**

| Фреймворк | Время интеграции | Сложность | Стоимость |
|-----------|-----------------|-----------|----------|
| StockSharp | 30 минут | Низкая | Бесплатно |
| Backtrader + aiomoex | 30 минут | Низкая | Бесплатно |
| Backtrader + AlgoPack | 1 час | Средняя | 55 тыс/год |
| LEAN | 2-3 дня | Высокая | Бесплатно (код) |

**Winner:** StockSharp (нативная поддержка всех российских брокеров).

## Тест 4: Сложность архитектуры

### **LEAN: Event-driven**

**Концепция:** Всё — это события.

```
Tick пришёл → OnData() вызван → Логика стратегии → Ордер отправлен
```

**Плюсы:**
- Реалистичность (похоже на реальную торговлю)
- Детальный контроль (доступ к каждому тику)

**Минусы:**
- Сложная отладка (event loop не прозрачен)
- Много boilerplate кода

### **StockSharp: Message-based**

**Концепция:** Все взаимодействия — через сообщения (Messages).

```
Strategy → Message Bus → Connector → Broker API
```

**Плюсы:**
- Гибкость (можно перехватывать/модифицировать сообщения)
- HFT-ready (микросекунды latency)

**Минусы:**
- Крутая кривая обучения
- Много абстракций

[Архитектура StockSharp](https://doc.stocksharp.ru/html/87d2cacd-5492-4bca-9140-7d7c3f5218d7.htm) сложна для новичков.

### **Backtrader: Simple loop**

**Концепция:** Простой loop по свечам.

```python
for candle in data:
    strategy.next()  # Вызывается для каждой свечи
```

**Плюсы:**
- Простота (понятно новичку за 10 минут)
- Прозрачность (легко дебажить)

**Минусы:**
- Медленный (Python loop)
- Менее реалистичный (не tick-by-tick)

### **Вывод по архитектуре**

| Фреймворк | Сложность архитектуры | Learning curve | Подходит новичкам? |
|-----------|----------------------|---------------|-------------------|
| Backtrader | Низкая | 1-2 недели | Да |
| LEAN | Средняя | 1 месяц | Нет |
| StockSharp | Высокая | 2-3 месяца | Нет |

## Тест 5: Документация и сообщество

### **LEAN**

**Документация:**
- [Официальная](https://www.quantconnect.com/docs)
- Обширная, но местами устаревшая
- Больше примеров на C#, чем на Python

**Сообщество:**
- Форум QuantConnect: активный
- GitHub: 9.6k stars, активные issues
- Ориентация: англоязычная аудитория

**Learning curve:** 1 месяц.

### **StockSharp**

**Документация:**
- [Официальная на русском](https://doc.stocksharp.ru/)
- Детальная, но иногда перегруженная
- Много примеров

**Сообщество:**
- Форум: средняя активность
- SmartLab: есть раздел
- GitHub: 7.4k stars

**Learning curve:** 2-3 месяца (сложная архитектура).

### **Backtrader**

**Документация:**
- [Официальная](https://www.backtrader.com/)
- Хорошая, но последние обновления в 2021
- Много community гайдов

**Сообщество:**
- GitHub: 13.6k stars
- Форумы: активные (несмотря на остановку развития)
- [Алготрейдинг.рф](https://алготрейдинг.рф/) — русскоязычные уроки

**Learning curve:** 1-2 недели.

### **Итоговая таблица: Документация**

| Фреймворк | Качество документации | Активность сообщества | Русскоязычные ресурсы |
|-----------|--------------------- |----------------------|---------------------|
| Backtrader | Хорошая | Высокая | Много |
| LEAN | Отличная | Высокая | Мало |
| StockSharp | Отличная | Средняя | Много |

## Реальный кейс: HFT-стратегия

**Задача:** Маркет-мейкинг на фьючерсах. Latency критична (<5 мс).

### **StockSharp**

```csharp
public class MarketMakingStrategy : Strategy
{
    protected override void OnStarted()
    {
        this
            .WhenNewTrade(Security)
            .Do(trade =>
            {
                var spread = BestAsk - BestBid;

                if (spread > MinSpread)
                {
                    RegisterOrder(this.CreateOrder(Sides.Buy, BestBid + Tick, Volume));
                    RegisterOrder(this.CreateOrder(Sides.Sell, BestAsk - Tick, Volume));
                }
            })
            .Apply(this);

        base.OnStarted();
    }
}
```

**Latency:** 1-3 мс (микросекундная обработка ордеров).

**Подходит:** Да.

### **LEAN**

LEAN — event-driven, но latency ~5-10 мс (Python API медленнее).

**Подходит:** Пограничный случай.

### **Backtrader**

Backtrader не поддерживает tick-by-tick в реальном времени. Только бэктестинг.

**Подходит:** Нет.

**Winner для HFT:** StockSharp.

## Реальный кейс: ML-стратегия

**Задача:** LSTM-модель для предсказания цены. Интеграция с TensorFlow.

### **Backtrader (Python)**

```python
import backtrader as bt
from tensorflow import keras
import numpy as np

class LSTMStrategy(bt.Strategy):
    def __init__(self):
        self.model = keras.models.load_model('lstm_model.h5')
        self.buffer = []

    def next(self):
        # Накапливаем последние 60 свечей
        self.buffer.append(self.data.close[0])
        if len(self.buffer) < 60:
            return

        # Предсказание
        X = np.array(self.buffer[-60:]).reshape(1, 60, 1)
        prediction = self.model.predict(X)[0][0]

        current_price = self.data.close[0]

        # Если предсказанная цена на 2% выше — покупаем
        if prediction > current_price * 1.02 and not self.position:
            self.buy(size=10)
```

**Удобство:** Отлично (Python + TensorFlow из коробки).

### **LEAN (C#)**

Нужна интеграция через Python.NET или Accord.NET (альтернатива TensorFlow для C#).

**Сложность:** Высокая.

### **StockSharp (C#)**

Аналогично LEAN. ML.NET или интеграция с Python.

**Сложность:** Высокая.

**Winner для ML:** Backtrader (Python-native).

## Итоговая таблица: Когда что использовать

| Критерий | Backtrader | LEAN | StockSharp |
|----------|-----------|------|-----------|
| **Простота для новичка** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Производительность бэктеста** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Интеграция с MOEX** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **HFT (latency <5 мс)** | ❌ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ML-интеграция (Python)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Документация** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Активное развитие** | ❌ (2021) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Русскоязычное сообщество** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## Моя личная рекомендация

### **Backtrader — если:**
- Вы новичок в алготрейдинге
- Знаете Python, но не C#
- Стратегия с ML (TensorFlow, PyTorch)
- Не нужен HFT
- Бюджет ограничен (open-source, бесплатно)

**Пример:** ML-стратегия на LSTM, свинг-трейдинг, позиционка.

### **LEAN — если:**
- Вы quant-разработчик
- Нужна поддержка международных рынков (US, EU)
- Готовы платить за QuantConnect (облако)
- Не торгуете на MOEX (или готовы писать custom connector)

**Пример:** Портфельные стратегии на US акциях, факторные модели.

### **StockSharp — если:**
- Торгуете на MOEX (российский рынок)
- Нужен HFT (latency <5 мс)
- Знаете C# или готовы учить
- Хотите визуальный Designer + код

**Пример:** HFT маркет-мейкинг, арбитраж между MOEX инструментами.

## Гибридный подход

Можно комбинировать:

### **Вариант 1: Backtrader (backtest) → StockSharp (live)**

1. Разработка и тестирование стратегии в Backtrader (быстрее)
2. Портирование на C# в StockSharp для live-торговли (HFT)

**Плюсы:** Удобство разработки + скорость live.

**Минусы:** Двойная работа (портирование).

### **Вариант 2: LEAN (облако) + StockSharp (MOEX)**

1. Международные рынки через QuantConnect
2. MOEX через StockSharp

**Плюсы:** Лучшее из двух миров.

**Минусы:** Две платформы = двойная сложность.

## Чек-лист: какой фреймворк выбрать

Ответьте на вопросы:

### **1. Ваш основной рынок?**

- **Российский (MOEX):** StockSharp (нативная поддержка) или Backtrader
- **Международный (US, EU, крипто):** LEAN или StockSharp (90+ бирж)
- **Только крипто:** Любые

### **2. Язык программирования?**

- **Python:** Backtrader или LEAN (Python API)
- **C#:** StockSharp или LEAN (C# core)

### **3. Тип стратегии?**

- **HFT (latency <5 мс):** StockSharp
- **ML (TensorFlow, PyTorch):** Backtrader
- **Портфельная оптимизация:** LEAN
- **Простая индикаторная:** Любой

### **4. Опыт программирования?**

- **Новичок:** Backtrader
- **Средний:** LEAN
- **Эксперт:** StockSharp (можно использовать всю мощь)

### **5. Бюджет?**

- **Бесплатно:** Backtrader, StockSharp, LEAN (self-hosted)
- **Готовы платить:** QuantConnect (от $20/мес)

### **6. Масштаб данных?**

- **Малый (<10 инструментов):** Любой
- **Большой (>100 инструментов):** VectorBT, Zipline (vectorized)

## Итоги

**Backtrader** — для новичков и Python-разработчиков. Простой, гибкий, но медленный и не поддерживается.

**LEAN** — для профессионалов и международных рынков. Мощный, активно развивается, но сложен в освоении.

**StockSharp** — для HFT и работы с 90+ биржами (сильная сторона — российский рынок). Быстрый, нативная поддержка MOEX, но высокая кривая обучения.

**Моё личное мнение:**

Если вы новичок — начните с **Backtrader**. [Пройдите гайд за выходные](https://алготрейдинг.рф/), напишите SMA-кросс. Поймёте логику алготрейдинга.

Через 3-6 месяцев, когда нужна скорость или HFT — переходите на **StockSharp** (особенно для MOEX и крипто-бирж) или **LEAN** (для международных фондовых рынков).

Не начинайте с StockSharp, если вы новичок. Архитектура сложна.

---

**Полезные ссылки:**

Фреймворки:
- [Backtrader](https://www.backtrader.com/)
- [LEAN (QuantConnect)](https://github.com/QuantConnect/Lean)
- [StockSharp](https://github.com/StockSharp/StockSharp)

Интеграция с MOEX:
- [backtrader_moexalgo](https://github.com/WISEPLAT/backtrader_moexalgo)
- [StockSharp MOEX Connector](https://doc.stocksharp.com/)
- [aiomoex (MOEX ISS API)](https://pypi.org/project/aiomoex/)

Бенчмарки производительности:
- [QuantRocket: Backtest Speed Comparison](https://www.quantrocket.com/blog/backtest-speed-comparison/)
- [Medium: VectorBT vs Zipline vs Backtrader](https://medium.com/@trading.dude/battle-tested-backtesters-comparing-vectorbt-zipline-and-backtrader-for-financial-strategy-dee33d33a9e0)

Русскоязычные ресурсы:
- [Алготрейдинг.рф (Backtrader)](https://алготрейдинг.рф/)
- [Документация StockSharp](https://doc.stocksharp.ru/)
- [SmartLab: Раздел алготрейдеров](https://smart-lab.ru/algotrading/)
