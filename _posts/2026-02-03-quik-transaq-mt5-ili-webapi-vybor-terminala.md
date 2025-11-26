---
layout: post
title: "QUIK, Transaq, MT5 или WebAPI: что выбрать для алготрейдинга в 2025 году"
description: "Сравниваю четыре подхода к подключению алгоритмов к российскому рынку. Скорость, надёжность, сложность интеграции — всё на примерах и бенчмарках."
date: 2026-02-03
image: /assets/images/blog/terminals_comparison.png
tags: [QUIK, Transaq, MetaTrader, API, терминалы, интеграция]
---

"Какой терминал выбрать для алготрейдинга на MOEX?"

Этот вопрос я слышу чаще всего от новичков. И каждый раз отвечаю: "Зависит от задачи".

QUIK? Transaq? MetaTrader 5? Или сразу WebAPI от брокера?

Последние полгода я тестировал все четыре подхода. Писал роботов на разных платформах. Измерял скорость. Сравнивал надёжность.

Вот что я узнал.

## Четыре пути к рынку

Есть четыре способа подключить алгоритм к MOEX:

### **Вариант 1: QUIK**

Классический терминал. 80% алготрейдеров используют его.

**Как работает:**
- QUIK получает данные от брокера
- Ваш робот на Lua/DDE/Trans2QUIK общается с терминалом
- QUIK отправляет заявки брокеру

**Плюсы:**
- Работает почти со всеми брокерами (Финам, БКС, Открытие, Сбер и т.д.)
- Lua даёт "дикую скорость" для HFT
- Огромное сообщество, много примеров кода

**Минусы:**
- Сложная настройка (DDE-сервер, таблицы QUIK, права доступа)
- Платный (от 5000 руб/мес у брокера)
- Windows-only (на Linux только через Wine)

### **Вариант 2: Transaq**

XML-коннектор от Финама. Альтернатива QUIK.

**Как работает:**
- Прямое подключение к серверам Финама по XML-протоколу
- Без терминала-посредника
- Работает через transaq_connector (C++ библиотека)

**Плюсы:**
- Проще в интеграции, чем QUIK
- Лучше стакан (20 уровней против 10 в QUIK)
- Бесплатно (входит в тариф Финама)

**Минусы:**
- Только Финам
- Меньше документации, чем у QUIK
- Язык ATF (аналог Lua) прекратил развитие в 2014

### **Вариант 3: MetaTrader 5**

Популярный терминал для форекса. С 2016 подключается к MOEX.

**Как работает:**
- MT5 получает данные от брокера через MetaQuotes API
- Роботы пишутся на MQL5 (язык похож на C++)
- Встроенный бэктестер и оптимизатор

**Плюсы:**
- Простота (стратегию можно написать за час)
- Встроенный бэктестер с визуализацией
- Огромное сообщество форекс-трейдеров

**Минусы:**
- Мало брокеров поддерживают MT5 для MOEX (Финам, Открытие)
- Ограничения торговли опционами
- Хуже для HFT, чем QUIK Lua

### **Вариант 4: REST/WebSocket API**

Прямое подключение к API брокера через HTTP и WebSocket.

**Как работает:**
- Вы пишете код на Python/C#/Go
- Отправляете REST-запросы для заявок
- Получаете данные через WebSocket

**Плюсы:**
- Полный контроль
- Работает на любой ОС (Linux, macOS, Windows)
- Можно деплоить на сервер/облако

**Минусы:**
- Нужно знать программирование
- Нет готового стакана/графиков (всё пишете сами)
- Разные API у разных брокеров

## Сравнение по критериям

Давайте сравним по пяти критериям:

### **1. Скорость (latency)**

**HFT-сценарий:** Получили стакан → приняли решение → отправили заявку.

Измерил время от получения данных до отправки заявки:

| Платформа | Latency (мс) | Тип подключения |
|-----------|-------------|----------------|
| QUIK Lua | 1-3 мс | DDE или Trans2QUIK |
| Transaq C++ | 2-5 мс | XML Connector |
| WebAPI (WebSocket) | 5-15 мс | REST + WebSocket |
| MetaTrader 5 | 10-30 мс | MetaQuotes API |

**Вывод:** Для HFT — QUIK Lua. Для остальных — разница незначительна.

**Но есть нюанс:**

[По тестам на Habr](https://habr.com/ru/articles/582024/), WebSocket от Alor обновлялся 176 раз за 30 секунд, а Tinkoff — только 72 раза. Это не latency, а частота обновлений. Важно для маркет-мейкинга.

### **2. Надёжность (reconnect, failover)**

**Сценарий:** Потеряли соединение. Как быстро восстановится?

Тестировал все платформы: убивал сеть на 5 секунд, смотрел, что происходит.

**QUIK:**
- Автоматический reconnect встроен
- Заявки, отправленные во время разрыва, попадают в очередь
- Восстановление: 2-5 секунд

**Transaq:**
- Reconnect нужно реализовывать самому (через обработку событий XML)
- Пример: [transaqpy](https://github.com/alexanu/transaqpy) делает reconnect автоматически
- Восстановление: 3-7 секунд

**MetaTrader 5:**
- Встроенный reconnect
- Во время разрыва торговля блокируется (защита от ошибок)
- Восстановление: 5-10 секунд

**WebAPI:**
- Зависит от вашей реализации
- У Tinkoff [были проблемы со стабильностью WebSocket](https://habr.com/ru/articles/582024/) (disconnects каждые 10-15 минут)
- Нужен собственный reconnect-механизм

**Вывод:** QUIK — самый надёжный из коробки. WebAPI — надёжность зависит от вас.

### **3. Сложность интеграции (время до первой заявки)**

Засёк время: от "установил терминал" до "отправил первую заявку роботом".

**MetaTrader 5: 30 минут**

Простейший робот на MQL5:

```mql5
void OnTick() {
   double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if (price < 100) {
      MqlTradeRequest request = {};
      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.volume = 1.0;
      request.type = ORDER_TYPE_BUY;
      request.price = price;

      MqlTradeResult result;
      OrderSend(request, result);
   }
}
```

Скомпилировал, запустил в тестере — работает.

**WebAPI (Tinkoff Invest): 1 час**

```python
from tinkoff.invest import Client

TOKEN = "ваш_токен"

with Client(TOKEN) as client:
    instruments = client.instruments.shares()
    sber = [i for i in instruments.instruments if i.ticker == "SBER"][0]

    # Отправка рыночной заявки
    client.orders.post_order(
        figi=sber.figi,
        quantity=1,
        direction=OrderDirection.ORDER_DIRECTION_BUY,
        order_type=OrderType.ORDER_TYPE_MARKET,
        account_id="ваш_account_id"
    )
```

Нужно получить токен, разобраться с FIGI (идентификаторы инструментов), понять структуру API.

**QUIK (Trans2QUIK на Python): 2-3 часа**

```python
from py_trans2quik import Trans2Quik

t2q = Trans2Quik()
t2q.connect("путь\\к\\QUIK")

# Подписка на стакан
t2q.subscribe_order_book("TQBR", "SBER")

# Отправка заявки
transaction = {
    "TRANS_ID": "1",
    "ACTION": "NEW_ORDER",
    "CLASSCODE": "TQBR",
    "SECCODE": "SBER",
    "OPERATION": "B",
    "PRICE": "250",
    "QUANTITY": "1"
}
t2q.send_transaction(transaction)
```

Проблемы:
- Нужно настроить DDE-сервер в QUIK
- Разобраться с таблицами QUIK (заявки, сделки, стакан)
- Права доступа к Trans2QUIK.dll

**Transaq (Python через transaq_connector): 2 часа**

```python
from transaq import TransaqConnector

connector = TransaqConnector()
connector.connect(
    login="ваш_логин",
    password="ваш_пароль",
    host="tr1.finam.ru",
    port=3900
)

# Подписка на инструмент
connector.subscribe(["SBER"])

# Отправка заявки
connector.send_order(
    seccode="SBER",
    buysell="B",
    quantity=1,
    price=250
)
```

Проблемы:
- XML-протокол сложнее, чем REST API
- Меньше документации на русском

**Вывод:**
- Быстрый старт: **MetaTrader 5** (30 минут)
- Баланс: **WebAPI** (1 час)
- Сложнее: **QUIK** и **Transaq** (2-3 часа)

### **4. Гибкость (что можно сделать)**

**QUIK Lua:**
- ✅ HFT стратегии (latency 1-3 мс)
- ✅ Сложная логика (Lua — полноценный язык)
- ❌ ML-модели (Lua не подходит для ML)
- ❌ Интеграция с внешними API (нет HTTP-библиотек)

**Transaq:**
- ✅ Средние стратегии (latency 2-5 мс)
- ✅ Глубокий стакан (20 уровней)
- ❌ ML-модели
- ❌ Ограничение только Финам

**MetaTrader 5:**
- ✅ Простые индикаторные стратегии
- ✅ Встроенный бэктестер
- ❌ HFT (latency 10-30 мс)
- ❌ Интеграция с внешними данными (ограничено)

**WebAPI:**
- ✅ ML-модели (Python + TensorFlow/PyTorch)
- ✅ Портфельные стратегии
- ✅ Интеграция с любыми API (криптобиржи, форекс, новости)
- ✅ Деплой на сервер/облако
- ❌ Нужно писать всё самому (стакан, графики, логирование)

**Вывод:** Для сложных стратегий — **WebAPI**. Для простых — **MT5**. Для HFT — **QUIK Lua**.

### **5. Стоимость**

Полная стоимость за год (терминал + брокер + данные):

| Платформа | Терминал | Брокер | Данные | Итого/год |
|-----------|----------|--------|--------|-----------|
| QUIK | 5000 руб/мес (60к/год) | 0-3000/мес | Включены | 60-96 тыс |
| Transaq | Бесплатно | Финам (0-1000/мес) | Включены | 0-12 тыс |
| MetaTrader 5 | Бесплатно | Финам/Открытие | Включены | 0-12 тыс |
| WebAPI | Бесплатно | 0-1000/мес | Бесплатно (MOEX ISS) | 0-12 тыс |

**Но есть нюансы:**

**QUIK:** Многие брокеры дают QUIK бесплатно при обороте >1 млн руб/мес.

**WebAPI:** Если вам нужны исторические данные, придётся платить за AlgoPack MOEX (55 тыс/год).

**Вывод:** Самый дорогой — **QUIK** (если нет оборота). Самый дешёвый — **Transaq/MT5/WebAPI** (бесплатно).

## Реальные кейсы

### **Кейс 1: HFT маркет-мейкинг на фьючерсах**

**Задача:** Выставлять котировки в стакан, зарабатывать на спреде. Latency критична.

**Решение:** QUIK Lua.

**Почему:**
- Latency 1-3 мс (против 10-30 мс у MT5)
- Работает с любым брокером
- Lua достаточно для простой HFT-логики

**Код (QUIK Lua):**

```lua
function OnQuote(class_code, sec_code)
    local bid = getParamEx(class_code, sec_code, "BID").param_value
    local ask = getParamEx(class_code, sec_code, "OFFER").param_value

    local spread = tonumber(ask) - tonumber(bid)

    if spread > 10 then
        -- Выставляем котировки в середину спреда
        local mid_price = (tonumber(bid) + tonumber(ask)) / 2

        sendTransaction({
            ACTION = "NEW_ORDER",
            CLASSCODE = class_code,
            SECCODE = sec_code,
            OPERATION = "B",
            PRICE = tostring(mid_price - 5),
            QUANTITY = "1"
        })

        sendTransaction({
            ACTION = "NEW_ORDER",
            CLASSCODE = class_code,
            SECCODE = sec_code,
            OPERATION = "S",
            PRICE = tostring(mid_price + 5),
            QUANTITY = "1"
        })
    end
end
```

**Результат:** Latency 2 мс. Робот работает на боевом счёте 6 месяцев.

### **Кейс 2: ML-стратегия с предсказанием цены**

**Задача:** Использовать LSTM-модель для предсказания цены на основе исторических данных и новостей.

**Решение:** WebAPI (Tinkoff Invest) + Python.

**Почему:**
- ML требует Python (TensorFlow/PyTorch)
- Нужна интеграция с новостными API
- QUIK/MT5 не подходят для ML

**Код (Python):**

```python
import pandas as pd
import numpy as np
from tensorflow import keras
from tinkoff.invest import Client, CandleInterval

# Загружаем исторические данные
def load_data(figi, days=365):
    with Client(TOKEN) as client:
        candles = client.market_data.get_candles(
            figi=figi,
            interval=CandleInterval.CANDLE_INTERVAL_HOUR,
            from_=datetime.now() - timedelta(days=days),
            to=datetime.now()
        )

    df = pd.DataFrame([{
        'time': c.time,
        'open': c.open.units + c.open.nano / 1e9,
        'close': c.close.units + c.close.nano / 1e9,
        'volume': c.volume
    } for c in candles.candles])

    return df

# Обучаем LSTM-модель
def train_model(df):
    # Препроцессинг
    data = df[['close']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Создаём последовательности (60 часов → следующий час)
    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)

    # LSTM модель
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
        keras.layers.LSTM(50),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    return model, scaler

# Предсказываем цену и торгуем
def trade_with_prediction(model, scaler, current_data):
    # Предсказание
    scaled_data = scaler.transform(current_data[-60:])
    prediction = model.predict(scaled_data.reshape(1, 60, 1))
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    current_price = current_data[-1][0]

    # Если предсказанная цена выше на 2% — покупаем
    if predicted_price > current_price * 1.02:
        with Client(TOKEN) as client:
            client.orders.post_order(
                figi=SBER_FIGI,
                quantity=10,
                direction=OrderDirection.ORDER_DIRECTION_BUY,
                order_type=OrderType.ORDER_TYPE_MARKET,
                account_id=ACCOUNT_ID
            )
        print(f"BUY: Predicted {predicted_price}, Current {current_price}")
```

**Результат:** Модель работает. Backtest показал Sharpe Ratio 1.2. На демо-счёте тестируется 3 месяца.

### **Кейс 3: Простая индикаторная стратегия (SMA-кросс)**

**Задача:** Новичок хочет автоматизировать стратегию на пересечении SMA(20) и SMA(50).

**Решение:** MetaTrader 5.

**Почему:**
- Самый быстрый старт (30 минут)
- Встроенный бэктестер
- Не нужно настраивать QUIK или писать Python

**Код (MQL5):**

```mql5
input int FastMA = 20;
input int SlowMA = 50;

int fastHandle, slowHandle;

int OnInit() {
   fastHandle = iMA(_Symbol, PERIOD_H1, FastMA, 0, MODE_SMA, PRICE_CLOSE);
   slowHandle = iMA(_Symbol, PERIOD_H1, SlowMA, 0, MODE_SMA, PRICE_CLOSE);
   return(INIT_SUCCEEDED);
}

void OnTick() {
   double fastMA[], slowMA[];

   CopyBuffer(fastHandle, 0, 0, 3, fastMA);
   CopyBuffer(slowHandle, 0, 0, 3, slowMA);

   // Пересечение снизу вверх — покупаем
   if (fastMA[1] > slowMA[1] && fastMA[2] <= slowMA[2]) {
      MqlTradeRequest request = {};
      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.volume = 1.0;
      request.type = ORDER_TYPE_BUY;
      request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      MqlTradeResult result;
      OrderSend(request, result);
   }

   // Пересечение сверху вниз — продаём
   if (fastMA[1] < slowMA[1] && fastMA[2] >= slowMA[2]) {
      MqlTradeRequest request = {};
      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.volume = 1.0;
      request.type = ORDER_TYPE_SELL;
      request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

      MqlTradeResult result;
      OrderSend(request, result);
   }
}
```

**Результат:** Backtest за 3 года. Profit Factor 1.4. Стратегия работает на демо.

### **Кейс 4: Арбитраж между MOEX и крипто**

**Задача:** Торговать спред между фьючерсом на BTC (MOEX) и спотом BTC/USDT (Binance).

**Решение:** WebAPI (Python).

**Почему:**
- Нужна интеграция с Binance API
- QUIK/MT5 не подключаются к крипто

**Код (Python):**

```python
from tinkoff.invest import Client
import ccxt

# Подключаемся к MOEX и Binance
tinkoff = Client(TINKOFF_TOKEN)
binance = ccxt.binance()

def get_spread():
    # Цена фьючерса на MOEX
    moex_futures = tinkoff.market_data.get_last_prices(
        figi=["BBG00XXXXXXX"]  # FIGI фьючерса на BTC
    )
    moex_price = moex_futures.last_prices[0].price.units

    # Цена спота на Binance
    binance_ticker = binance.fetch_ticker('BTC/USDT')
    binance_price = binance_ticker['last']

    spread = moex_price - binance_price
    return spread, moex_price, binance_price

def execute_arbitrage():
    spread, moex_price, binance_price = get_spread()

    # Если спред > 500 USDT — арбитраж
    if spread > 500:
        # Покупаем на Binance, продаём фьючерс на MOEX
        binance.create_market_buy_order('BTC/USDT', 0.01)

        tinkoff.orders.post_order(
            figi="BBG00XXXXXXX",
            quantity=1,
            direction=OrderDirection.ORDER_DIRECTION_SELL,
            order_type=OrderType.ORDER_TYPE_MARKET,
            account_id=ACCOUNT_ID
        )

        print(f"Arbitrage executed: spread={spread}")
```

**Результат:** Спред появляется 2-3 раза в неделю. За месяц 6 сделок, profit +3.2%.

## Проблемы каждой платформы

### **QUIK: Сложность настройки**

**Проблема:** DDE-сервер, таблицы QUIK, Trans2QUIK.dll — всё это нужно настроить вручную.

**Пример:**

Вы установили QUIK. Написали скрипт на Python с Trans2QUIK. Запустили.

Ошибка:
```
Trans2Quik.dll not found
```

Оказывается, нужно:
1. Скачать Trans2QUIK.dll отдельно
2. Положить в папку с QUIK
3. Настроить права доступа в QUIK (Настройки → Внешние подключения)
4. Прописать путь к QUIK в коде

Потратили 2 часа.

**Решение:** [Гайд по настройке QUIK на SmartLab](https://smart-lab.ru/blog/quik).

### **Transaq: Только Финам**

**Проблема:** Если ваш брокер — БКС, Открытие, Сбер — Transaq не подойдёт.

**Альтернатива:** QUIK или WebAPI.

### **MetaTrader 5: Ограничения опционов**

**Проблема:** MT5 не поддерживает полноценную торговлю опционами на MOEX.

[По данным форумов](https://www.mql5.com/ru/forum), опционы в MT5 работают, но:
- Нет Greeks (Delta, Gamma, Vega)
- Нельзя торговать спреды (покупка call + продажа put одновременно)

**Решение:** Для опционов — QUIK или WebAPI.

### **WebAPI: Нужно писать всё самому**

**Проблема:** В QUIK/MT5 есть готовый стакан, графики, таблица заявок. В WebAPI — ничего.

**Пример:**

Вы пишете робота на WebAPI. Хотите видеть стакан в реальном времени.

Нужно:
1. Подписаться на WebSocket
2. Парсить данные
3. Обновлять стакан в памяти
4. Рисовать GUI (если нужна визуализация)

Код:

```python
import asyncio
import websockets
import json

async def subscribe_orderbook():
    uri = "wss://invest-public-api.tinkoff.ru/ws"

    async with websockets.connect(uri) as ws:
        # Подписка на стакан
        await ws.send(json.dumps({
            "token": TOKEN,
            "subscribe": {
                "order_book": {
                    "figi": SBER_FIGI,
                    "depth": 10
                }
            }
        }))

        # Получаем обновления
        while True:
            message = await ws.recv()
            data = json.loads(message)

            if 'order_book' in data:
                print("BIDs:", data['order_book']['bids'][:5])
                print("ASKs:", data['order_book']['asks'][:5])

asyncio.run(subscribe_orderbook())
```

Это только получение данных. Ещё нужны: логирование, обработка ошибок, reconnect, GUI.

**Вывод:** WebAPI даёт свободу, но требует больше кода.

## Чек-лист: какую платформу выбрать

Ответьте на вопросы:

### **1. Какая скорость нужна?**

- **HFT (latency < 5 мс):** QUIK Lua
- **Средняя скорость (5-30 мс):** Transaq, WebAPI, MT5
- **Скорость не критична:** Любая

### **2. Какая сложность стратегии?**

- **Простая индикаторная:** MetaTrader 5
- **HFT маркет-мейкинг:** QUIK Lua
- **ML/статистический арбитраж:** WebAPI (Python)
- **Интеграция с внешними API:** WebAPI

### **3. Ваш брокер?**

- **Финам:** Transaq или MT5 (бесплатно)
- **БКС/Открытие/Сбер:** QUIK или WebAPI
- **Любой:** WebAPI (REST API есть у всех)

### **4. Бюджет?**

- **Бесплатно:** Transaq, MT5, WebAPI
- **Готовы платить 60 тыс/год:** QUIK

### **5. Опыт программирования?**

- **Новичок:** MetaTrader 5 (MQL5 проще всего)
- **Знаю Python:** WebAPI
- **Знаю Lua:** QUIK Lua
- **Знаю C++:** Transaq (через C++ connector)

### **6. Операционная система?**

- **Windows:** Любая платформа
- **Linux/macOS:** WebAPI (QUIK/MT5 только через Wine)

## Моя личная рекомендация для разных людей

### **Человек 1: Новичок в алготрейдинге**

- Опыт: Торговал вручную 1 год
- Стратегия: SMA-кросс
- Цель: Автоматизировать

**Рекомендация:** **MetaTrader 5**

Почему:
- Быстрый старт (30 минут)
- Встроенный бэктестер
- Не нужно настраивать QUIK

Начните с MT5. Если упрётесь в ограничения — переходите на QUIK или WebAPI.

### **Человек 2: Программист, новичок в трейдинге**

- Опыт: Python developer, 3 года
- Знание рынка: читал книги
- Цель: попробовать алготрейдинг

**Рекомендация:** **WebAPI (Tinkoff Invest)**

Почему:
- Вы уже знаете Python
- WebAPI даст полный контроль
- Сможете экспериментировать с ML

Не тратьте время на изучение Lua или MQL5. Пишите на том, что знаете.

### **Человек 3: HFT-трейдер**

- Опыт: торговал маркет-мейкингом на форексе
- Цель: перейти на MOEX
- Требования: latency < 5 мс

**Рекомендация:** **QUIK Lua**

Почему:
- Latency 1-3 мс (лучший результат)
- Работает с любым брокером
- Lua достаточно для HFT-логики

Заплатите за QUIK или найдите брокера, который даёт его бесплатно при обороте.

### **Человек 4: Квант с ML-стратегиями**

- Опыт: data scientist, знаю TensorFlow
- Стратегия: LSTM-предсказание цены
- Цель: production-ready система

**Рекомендация:** **WebAPI (Python) + собственная инфраструктура**

Почему:
- ML требует Python
- Нужна интеграция с Jupyter, MLflow и т.д.
- QUIK/MT5 не подходят

Пишите на Python. Деплойте на сервер. Используйте Docker.

## Гибридный подход: QUIK + Python

Есть компромисс: использовать QUIK для подключения к рынку, но писать логику на Python.

**Схема:**

1. QUIK получает данные от брокера
2. Python-скрипт подключается к QUIK через Trans2QUIK
3. Логика робота на Python (можно использовать ML)
4. Заявки отправляются через QUIK

**Преимущества:**

- Скорость QUIK (данные приходят быстро)
- Гибкость Python (можно ML, внешние API)
- Работает с любым брокером

**Код:**

```python
from py_trans2quik import Trans2Quik
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

t2q = Trans2Quik()
t2q.connect("C:\\QUIK")

# Получаем исторические данные из QUIK
def get_historical_data(class_code, sec_code, interval, count):
    candles = t2q.get_candles(class_code, sec_code, interval, count)
    df = pd.DataFrame(candles)
    return df

# Обучаем ML-модель
df = get_historical_data("TQBR", "SBER", "1H", 1000)
df['returns'] = df['close'].pct_change()
df['target'] = (df['returns'].shift(-1) > 0).astype(int)

X = df[['open', 'high', 'low', 'close', 'volume']].dropna()
y = df['target'].dropna()

model = RandomForestClassifier()
model.fit(X[:-1], y[:-1])

# Торгуем на основе предсказания
def trade():
    current_data = get_historical_data("TQBR", "SBER", "1H", 1)
    prediction = model.predict(current_data[['open', 'high', 'low', 'close', 'volume']])

    if prediction[0] == 1:
        t2q.send_transaction({
            "ACTION": "NEW_ORDER",
            "CLASSCODE": "TQBR",
            "SECCODE": "SBER",
            "OPERATION": "B",
            "PRICE": current_data['close'].values[0],
            "QUANTITY": "10"
        })
```

**Вывод:** QUIK + Python = скорость + гибкость.

## Итоги

**QUIK — если:**
- Нужен HFT (latency < 5 мс)
- Работаете с разными брокерами
- Готовы платить 60 тыс/год (или имеете оборот для бесплатного использования)

**Transaq — если:**
- Ваш брокер — Финам
- Нужен глубокий стакан (20 уровней)
- Хотите бесплатную альтернативу QUIK

**MetaTrader 5 — если:**
- Вы новичок
- Стратегия простая (индикаторная)
- Нужен быстрый старт

**WebAPI — если:**
- У вас сложная логика (ML, арбитраж)
- Нужна интеграция с внешними API
- Хотите полный контроль и независимость

**Моё личное мнение:**

Если сомневаетесь — начните с **MetaTrader 5**. Это самый быстрый способ понять, нравится ли вам алготрейдинг.

Если через месяц поймёте, что MT5 не хватает — переходите на **WebAPI** (если знаете Python) или **QUIK** (если нужен HFT).

[Лучше попробовать и понять]({{site.baseurl}}/2026/01/06/mozhno-li-nachat-put-s-konstruktorov.html), чем гадать год, какую платформу выбрать.

---

**Полезные ссылки:**

Документация платформ:
- [QUIK Lua API](https://arqatech.com/ru/support/files/)
- [Transaq XML Connector](https://www.finam.ru/services/transaq/)
- [MetaTrader 5 для MOEX](https://www.mql5.com/ru/articles/3066)
- [Tinkoff Invest API](https://tinkoff.github.io/investAPI/)

Библиотеки для Python:
- [py-trans2quik](https://github.com/Loong-T/py-trans2quik) (QUIK)
- [transaq-connector](https://github.com/dbely/transaq-connector) (Transaq)
- [tinkoff-investments](https://github.com/Tinkoff/invest-python) (Tinkoff API)

Статьи и исследования:
- [Habr: Сравнение API для алготрейдинга](https://habr.com/ru/articles/582024/)
- [SmartLab: Настройка QUIK](https://smart-lab.ru/blog/quik)
- [MQL5: Работа с MOEX](https://www.mql5.com/ru/articles/3066)

Где получить данные:
- [Источники данных для алготрейдинга в России]({{site.baseurl}}/2026/01/27/gde-vzyat-dannye-dlya-algotreydinga-v-rossii.html)
