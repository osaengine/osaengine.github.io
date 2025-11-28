---
layout: post
title: "Альтернативные данные в трейдинге: Спутники, сентимент и блокчейн"
description: "Как хедж-фонды используют спутниковые снимки, анализ социальных сетей и on-chain метрики для поиска альфы. Практические примеры с Python."
date: 2026-07-28
image: /assets/images/blog/alternative_data.png
tags: [alternative-data, satellite-imagery, sentiment-analysis, on-chain, machine-learning]
---

Пока большинство трейдеров смотрит на цены и индикаторы, хедж-фонды вроде Two Sigma и Citadel считают машины на парковках через спутники, анализируют настроения в Reddit и отслеживают движение криптовалюты в блокчейне. Рынок альтернативных данных вырос до **$18.74 млрд в 2025 году** и мчится к $135.72 млрд к 2030-му с ошеломляющим CAGR 63.4%. **85% ведущих хедж-фондов** используют два и более источника альтернативных данных, а 78% интегрируют их в торговые стратегии.

В этой статье разберём три мощнейших класса альтернативных данных: спутниковые снимки, анализ сентимента и on-chain метрики. Покажу, как собирать, обрабатывать и использовать эти данные в торговых стратегиях на Python, и почему хедж-фонды, внедрившие машинное обучение с альтернативными данными, получают **20-30% прирост точности прогнозов** по сравнению с традиционными методами.

## Что такое альтернативные данные и почему они работают

**Альтернативные данные (Alternative Data)** — это информация из нестандартных источников, которая не входит в традиционные финансовые отчёты или биржевые данные. Это может быть всё: от количества машин на парковке Walmart до тональности твитов про Tesla, от уровня нефти в танкерах до активности кошельков в Ethereum.

### Почему это даёт преимущество?

Представьте: вы хотите спрогнозировать квартальную выручку ритейлера. Традиционный подход — ждать отчётности через 3 месяца. Альтернативный подход — каждый день анализировать спутниковые снимки парковок их магазинов, считать количество машин и получать **early signal** о продажах за недели до официального релиза. Orbital Insight делает именно это, помогая хедж-фондам предсказывать движения цен на нефть через мониторинг уровней заполнения нефтехранилищ по всему миру **до выхода официальных отчётов**.

Или другой пример: вы торгуете акциями стриминговых сервисов. Вместо того чтобы ждать квартального отчёта о количестве подписчиков, вы **scraping'ите** данные о рейтингах приложений в App Store, отзывах пользователей, тенденциях загрузок — и получаете сигнал о росте или падении популярности в реальном времени.

### Типы альтернативных данных и их эффективность

Исследования 2025 года показывают конкретные цифры эффективности:

- **Social Media Sentiment**: 87% точность прогнозов
- **Transaction Data (данные о покупках)**: +10% улучшение предсказаний
- **Satellite Imagery**: +18% улучшение оценок прибыли компаний
- **Web Traffic & App Data**: Early indicators для оценки популярности продуктов
- **On-Chain Crypto Data**: Корреляция 0.87 с ценой BTC для stablecoin supply

**65% хедж-фондов** используют альтернативные данные для обгона конкурентов, достигая до **3% более высокой годовой доходности**. Сегмент хедж-фондов доминирует на рынке с долей **68% в 2024 году**.

### Основные игроки рынка

**Платформы альтернативных данных в 2025:**

1. **Quandl (теперь Nasdaq Data Link)** — ~350 источников альтернативных данных, от цен на нефть до веб-скрейпинга.
2. **Thinknum** — непрерывный скрейпинг 35+ датасетов: вакансии на LinkedIn, рейтинги приложений, количество сотрудников (4600+ публичных компаний за ~10 лет).
3. **Orbital Insight** — геопространственная аналитика: подсчёт посетителей ритейлеров, количества нефтяных баррелей, активность глобальных цепочек поставок через спутники.
4. **RavenPack** — новостная аналитика в реальном времени с sentiment analysis для финансовых рынков.
5. **YipitData** — данные транзакций, email receipts, поведение потребителей.
6. **Glassnode & CryptoQuant** — on-chain метрики для криптовалют.

Стоимость команды по альтернативным данным начинается от **$1.5-2.5 млн**, и 42% фирм имеют выделенного Data Lead.

## 1. Спутниковые снимки: Считаем прибыль из космоса

Satellite imagery — один из самых мощных источников альтернативных данных. Hedge funds используют спутники для мониторинга:

- **Парковки ритейлеров** → оценка foot traffic и продаж
- **Нефтехранилища** → прогнозирование цен на энергоносители
- **Порты и контейнеровозы** → анализ глобальных цепочек поставок
- **Сельскохозяйственные поля** → оценка урожайности
- **Строительные объекты** → активность застройщиков

### Кейс: Orbital Insight и нефтяные запасы

Orbital Insight помогла хедж-фондам предсказать движения цен на нефть, отслеживая глобальные уровни нефтехранилищ через спутники. Они мониторили уровни заполнения tank farms по всему миру, чтобы позиционировать энергетические сделки **раньше официальных отчётов**.

В 2020 году, когда цены на нефть обвалились до отрицательных значений, спутниковые данные показывали критическое переполнение хранилищ за недели до паники на рынке. Хедж-фонды, использовавшие эти данные, смогли открыть короткие позиции заранее.

### Практика: Анализ спутниковых снимков с Python

Существует несколько способов получить спутниковые снимки для анализа:

1. **Google Earth Engine** — бесплатный доступ к петабайтам спутниковых данных (Landsat, Sentinel).
2. **Planet Labs API** — коммерческие спутниковые снимки высокого разрешения (3-5 метров).
3. **Sentinel Hub** — API для доступа к данным Sentinel-2 (10-метровое разрешение, бесплатно).

Давайте напишем код для подсчёта машин на парковке с использованием компьютерного зрения:

```python
import ee
import numpy as np
import cv2
from datetime import datetime, timedelta
import pandas as pd
from ultralytics import YOLO  # YOLOv8 для детекции объектов
import matplotlib.pyplot as plt

class SatelliteParkingAnalyzer:
    """
    Анализатор спутниковых снимков для подсчёта машин на парковках
    """

    def __init__(self, ee_project: str):
        """
        Инициализация Earth Engine и модели детекции

        Args:
            ee_project: Google Cloud project ID для Earth Engine
        """
        # Аутентификация в Google Earth Engine
        ee.Authenticate()
        ee.Initialize(project=ee_project)

        # Загружаем предобученную модель YOLOv8 для детекции машин
        # В реальности нужна модель, обученная на спутниковых снимках
        self.model = YOLO('yolov8n.pt')

    def get_satellite_image(self,
                           lat: float,
                           lon: float,
                           radius: int = 500,
                           date: str = None) -> np.ndarray:
        """
        Получает спутниковый снимок области через Earth Engine

        Args:
            lat: Широта центра области
            lon: Долгота центра области
            radius: Радиус области в метрах
            date: Дата снимка (YYYY-MM-DD), если None - последний доступный

        Returns:
            RGB изображение как numpy array
        """
        # Создаём точку интереса
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(radius).bounds()

        # Задаём временной диапазон
        if date:
            start_date = date
            end_date = (datetime.strptime(date, '%Y-%m-%d') +
                       timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() -
                         timedelta(days=30)).strftime('%Y-%m-%d')

        # Получаем коллекцию Sentinel-2 (10m разрешение)
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterBounds(point)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                     .select(['B4', 'B3', 'B2']))  # RGB bands

        # Берём медианное изображение для устранения облаков
        image = collection.median()

        # Получаем URL для скачивания
        url = image.getThumbURL({
            'region': region,
            'dimensions': 1024,
            'format': 'png'
        })

        # Загружаем и конвертируем в numpy array
        # В реальности используйте requests для загрузки
        # img_array = download_and_convert(url)

        return url  # Возвращаем URL для примера

    def count_cars(self, image: np.ndarray, confidence: float = 0.5) -> dict:
        """
        Подсчитывает количество машин на изображении

        Args:
            image: RGB изображение парковки
            confidence: Минимальный порог уверенности для детекции

        Returns:
            Словарь с результатами: {count, detections, annotated_image}
        """
        # Запускаем YOLO детекцию
        results = self.model(image, conf=confidence)

        # Фильтруем только машины (class_id = 2 в COCO)
        cars = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 2:  # car class
                    cars.append({
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'confidence': float(box.conf[0])
                    })

        # Рисуем аннотации
        annotated = results[0].plot()

        return {
            'count': len(cars),
            'detections': cars,
            'annotated_image': annotated,
            'timestamp': datetime.now()
        }

    def track_retailer(self,
                      locations: list,
                      ticker: str,
                      days: int = 90) -> pd.DataFrame:
        """
        Отслеживает парковки сети магазинов за период времени

        Args:
            locations: Список координат магазинов [(lat, lon), ...]
            ticker: Тикер компании (например, 'WMT' для Walmart)
            days: Количество дней для анализа

        Returns:
            DataFrame с временным рядом количества машин
        """
        results = []

        # Генерируем даты для анализа (каждые 7 дней)
        dates = pd.date_range(
            end=datetime.now(),
            periods=days // 7,
            freq='7D'
        )

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            daily_count = 0

            for lat, lon in locations:
                try:
                    # Получаем снимок
                    img_url = self.get_satellite_image(lat, lon,
                                                       date=date_str)
                    # В реальности загружаем и анализируем
                    # img = download_image(img_url)
                    # car_data = self.count_cars(img)
                    # daily_count += car_data['count']

                    # Для примера генерируем случайные данные
                    daily_count += np.random.randint(50, 200)

                except Exception as e:
                    print(f"Ошибка для {date_str}, {lat}, {lon}: {e}")
                    continue

            results.append({
                'date': date,
                'ticker': ticker,
                'total_cars': daily_count,
                'avg_per_location': daily_count / len(locations)
            })

        return pd.DataFrame(results)


# Использование
analyzer = SatelliteParkingAnalyzer(ee_project='my-earth-engine-project')

# Координаты нескольких Walmart магазинов (пример)
walmart_locations = [
    (34.0522, -118.2437),  # Los Angeles
    (40.7128, -74.0060),   # New York
    (41.8781, -87.6298),   # Chicago
    (29.7604, -95.3698),   # Houston
]

# Отслеживаем 90 дней
parking_data = analyzer.track_retailer(
    locations=walmart_locations,
    ticker='WMT',
    days=90
)

# Анализ корреляции с ценой акций
import yfinance as yf

# Загружаем цену акций WMT
wmt_stock = yf.download('WMT',
                        start=parking_data['date'].min(),
                        end=parking_data['date'].max())

# Мержим данные
parking_data.set_index('date', inplace=True)
combined = parking_data.join(wmt_stock[['Close']], how='inner')

# Считаем корреляцию
correlation = combined['total_cars'].corr(combined['Close'])
print(f"Корреляция между количеством машин и ценой: {correlation:.3f}")

# Создаём торговую стратегию
combined['cars_ma7'] = combined['total_cars'].rolling(7).mean()
combined['cars_ma28'] = combined['total_cars'].rolling(28).mean()

# Сигнал: когда краткосрочная MA пересекает долгосрочную
combined['signal'] = 0
combined.loc[combined['cars_ma7'] > combined['cars_ma28'], 'signal'] = 1
combined.loc[combined['cars_ma7'] < combined['cars_ma28'], 'signal'] = -1

# Считаем доходность
combined['returns'] = combined['Close'].pct_change()
combined['strategy_returns'] = combined['signal'].shift(1) * combined['returns']

# Результаты
total_return = (1 + combined['strategy_returns']).prod() - 1
sharpe = combined['strategy_returns'].mean() / combined['strategy_returns'].std() * np.sqrt(52)

print(f"\nРезультаты стратегии на спутниковых данных:")
print(f"Общая доходность: {total_return*100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
```

### Реальные результаты

Исследования показывают, что спутниковые данные дают **+18% улучшение оценок прибыли** компаний. Два крупнейших хедж-фонда (Two Sigma и Citadel) активно используют эти инсайты, особенно в секторах retail, agriculture и energy.

**Ключевые преимущества спутниковых данных:**

- **Объективность**: Спутник не врёт, в отличие от пресс-релизов
- **Real-time**: Данные обновляются каждые несколько дней
- **Early signal**: Опережение официальной отчётности на недели
- **Глобальный охват**: Можно отслеживать компании в любой точке мира

**Недостатки:**

- **Высокая стоимость**: Коммерческие снимки высокого разрешения дороги
- **Сложность обработки**: Нужны навыки компьютерного зрения
- **Погодные ограничения**: Облачность может мешать съёмке

## 2. Анализ сентимента: Читаем настроение рынка

Social media sentiment analysis — мощнейший инструмент для gauging общественного мнения о брендах, продуктах, политических событиях, влияющих на рынки. **87% точность прогнозов** делает сентимент одним из самых эффективных источников альтернативных данных.

### Откуда брать данные?

**Основные источники:**

1. **Twitter/X** — мгновенная реакция на новости, trending tickers
2. **Reddit (r/WallStreetBets, r/stocks)** — retail sentiment, мемные акции
3. **StockTwits** — социальная сеть для трейдеров
4. **News APIs** — Bloomberg, Reuters, финансовые новости
5. **Earnings Call Transcripts** — тон и уверенность менеджмента

### NLP модели для финансового сентимента в 2025

**FinBERT** — state-of-the-art модель для финансового sentiment analysis в 2025. Это BERT, дообученный на финансовых текстах, который показывает:

- **97% точность** на Financial PhraseBank (full agreement)
- **86% точность** на полном датасете
- Классификация на 3 класса: positive, negative, neutral

**Альтернативы:**

- **ChatGPT** с prompt engineering для сентимента
- **Claude** для контекстного анализа новостей
- **Llama** с fine-tuning на финансовых данных

### Практика: Sentiment Analysis с FinBERT

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import tweepy  # Twitter API
import praw    # Reddit API

class SentimentAnalyzer:
    """
    Анализатор финансового сентимента с FinBERT
    """

    def __init__(self):
        # Загружаем FinBERT из HuggingFace
        self.tokenizer = BertTokenizer.from_pretrained(
            'ProsusAI/finbert'
        )
        self.model = BertForSequenceClassification.from_pretrained(
            'ProsusAI/finbert'
        )
        self.model.eval()

        # Маппинг классов
        self.labels = ['positive', 'negative', 'neutral']

    def analyze_text(self, text: str) -> dict:
        """
        Анализирует сентимент текста

        Args:
            text: Финансовый текст для анализа

        Returns:
            {sentiment: str, confidence: float, scores: dict}
        """
        # Токенизация
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )

        # Прогон через модель
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Softmax для вероятностей
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

        # Определяем класс с максимальной вероятностью
        predicted_class = torch.argmax(probs).item()
        sentiment = self.labels[predicted_class]
        confidence = probs[predicted_class].item()

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': {
                label: prob.item()
                for label, prob in zip(self.labels, probs)
            }
        }

    def analyze_batch(self, texts: list) -> pd.DataFrame:
        """
        Анализирует батч текстов

        Args:
            texts: Список текстов

        Returns:
            DataFrame с результатами
        """
        results = []

        for text in texts:
            sentiment_data = self.analyze_text(text)
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': sentiment_data['sentiment'],
                'confidence': sentiment_data['confidence'],
                'positive_score': sentiment_data['scores']['positive'],
                'negative_score': sentiment_data['scores']['negative'],
                'neutral_score': sentiment_data['scores']['neutral']
            })

        return pd.DataFrame(results)


class TwitterSentimentCollector:
    """
    Сборщик sentiment из Twitter/X
    """

    def __init__(self, bearer_token: str):
        # Twitter API v2
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.analyzer = SentimentAnalyzer()

    def collect_tweets(self,
                      query: str,
                      max_results: int = 100,
                      days_back: int = 7) -> pd.DataFrame:
        """
        Собирает твиты по запросу

        Args:
            query: Поисковый запрос (например, "$TSLA")
            max_results: Максимальное количество твитов
            days_back: Сколько дней назад искать

        Returns:
            DataFrame с твитами и сентиментом
        """
        # Задаём временной диапазон
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        # Собираем твиты
        tweets = self.client.search_recent_tweets(
            query=query,
            max_results=max_results,
            start_time=start_time,
            end_time=end_time,
            tweet_fields=['created_at', 'public_metrics']
        )

        if not tweets.data:
            return pd.DataFrame()

        # Обрабатываем
        tweet_data = []
        for tweet in tweets.data:
            tweet_data.append({
                'created_at': tweet.created_at,
                'text': tweet.text,
                'likes': tweet.public_metrics['like_count'],
                'retweets': tweet.public_metrics['retweet_count']
            })

        df = pd.DataFrame(tweet_data)

        # Анализируем сентимент
        sentiment_results = self.analyzer.analyze_batch(df['text'].tolist())

        # Объединяем
        result = pd.concat([df.reset_index(drop=True),
                           sentiment_results[['sentiment', 'confidence',
                                             'positive_score', 'negative_score']]],
                          axis=1)

        return result


class RedditSentimentCollector:
    """
    Сборщик sentiment из Reddit
    """

    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        # Reddit API
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.analyzer = SentimentAnalyzer()

    def collect_wsb_sentiment(self,
                             ticker: str,
                             limit: int = 100) -> pd.DataFrame:
        """
        Собирает сентимент из r/wallstreetbets

        Args:
            ticker: Тикер акции (например, 'GME')
            limit: Количество постов

        Returns:
            DataFrame с постами и сентиментом
        """
        subreddit = self.reddit.subreddit('wallstreetbets')

        # Ищем упоминания тикера
        posts = []
        for submission in subreddit.search(ticker, limit=limit,
                                          time_filter='week'):
            posts.append({
                'created_at': datetime.fromtimestamp(submission.created_utc),
                'title': submission.title,
                'text': submission.selftext,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'url': submission.url
            })

        if not posts:
            return pd.DataFrame()

        df = pd.DataFrame(posts)

        # Комбинируем title и text для анализа
        df['combined_text'] = df['title'] + ' ' + df['text']

        # Анализируем сентимент
        sentiment_results = self.analyzer.analyze_batch(
            df['combined_text'].tolist()
        )

        result = pd.concat([df, sentiment_results[['sentiment',
                                                   'confidence',
                                                   'positive_score']]],
                          axis=1)

        return result


# === Торговая стратегия на основе сентимента ===

class SentimentTradingStrategy:
    """
    Торговая стратегия на основе социального сентимента
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.twitter_collector = None  # Инициализируйте с API ключами
        self.reddit_collector = None   # Инициализируйте с API ключами

    def calculate_sentiment_score(self,
                                  twitter_df: pd.DataFrame,
                                  reddit_df: pd.DataFrame) -> float:
        """
        Рассчитывает агрегированный сентимент score

        Args:
            twitter_df: DataFrame с Twitter сентиментом
            reddit_df: DataFrame с Reddit сентиментом

        Returns:
            Sentiment score от -1 (очень негативный) до +1 (очень позитивный)
        """
        scores = []

        # Twitter сентимент (взвешиваем по likes и retweets)
        if not twitter_df.empty:
            twitter_df['engagement'] = (twitter_df['likes'] +
                                       twitter_df['retweets'] * 2)

            # Конвертируем sentiment в числа
            sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
            twitter_df['sentiment_value'] = twitter_df['sentiment'].map(
                sentiment_map
            )

            # Взвешенный score
            twitter_score = (
                (twitter_df['sentiment_value'] *
                 twitter_df['confidence'] *
                 twitter_df['engagement']).sum() /
                twitter_df['engagement'].sum()
            )
            scores.append(('twitter', twitter_score, 0.4))  # вес 40%

        # Reddit сентимент (взвешиваем по score и comments)
        if not reddit_df.empty:
            reddit_df['engagement'] = (reddit_df['score'] +
                                      reddit_df['num_comments'] * 3)

            sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
            reddit_df['sentiment_value'] = reddit_df['sentiment'].map(
                sentiment_map
            )

            reddit_score = (
                (reddit_df['sentiment_value'] *
                 reddit_df['confidence'] *
                 reddit_df['engagement']).sum() /
                reddit_df['engagement'].sum()
            )
            scores.append(('reddit', reddit_score, 0.6))  # вес 60%

        # Взвешенный агрегат
        if scores:
            total_weight = sum(weight for _, _, weight in scores)
            weighted_score = sum(score * weight
                               for _, score, weight in scores) / total_weight
            return weighted_score

        return 0.0

    def backtest(self, start_date: str, end_date: str) -> dict:
        """
        Бэктест стратегии на исторических данных

        Args:
            start_date: Начальная дата (YYYY-MM-DD)
            end_date: Конечная дата (YYYY-MM-DD)

        Returns:
            Словарь с метриками производительности
        """
        # Загружаем цены акций
        stock_data = yf.download(self.ticker,
                                start=start_date,
                                end=end_date)

        # В реальности собираем исторический сентимент
        # Для примера генерируем синтетические данные
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_scores = []

        for date in dates:
            # Симуляция сбора сентимента
            score = np.random.normal(0, 0.3)  # Случайный сентимент
            sentiment_scores.append({
                'date': date,
                'sentiment_score': score
            })

        sentiment_df = pd.DataFrame(sentiment_scores)
        sentiment_df.set_index('date', inplace=True)

        # Объединяем с ценами
        combined = stock_data.join(sentiment_df, how='inner')

        # Генерируем сигналы
        # Long если sentiment > 0.2, Short если < -0.2
        combined['signal'] = 0
        combined.loc[combined['sentiment_score'] > 0.2, 'signal'] = 1
        combined.loc[combined['sentiment_score'] < -0.2, 'signal'] = -1

        # Считаем доходность
        combined['returns'] = combined['Close'].pct_change()
        combined['strategy_returns'] = (combined['signal'].shift(1) *
                                       combined['returns'])

        # Метрики
        total_return = (1 + combined['strategy_returns']).prod() - 1
        sharpe = (combined['strategy_returns'].mean() /
                 combined['strategy_returns'].std() * np.sqrt(252))
        max_drawdown = (combined['strategy_returns'].cumsum().cummax() -
                       combined['strategy_returns'].cumsum()).max()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': (combined['strategy_returns'] > 0).sum() /
                       len(combined),
            'trades': (combined['signal'].diff() != 0).sum()
        }


# Использование
strategy = SentimentTradingStrategy('TSLA')

results = strategy.backtest('2024-01-01', '2025-01-01')

print("Результаты sentiment-based стратегии:")
print(f"Общая доходность: {results['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Максимальная просадка: {results['max_drawdown']*100:.2f}%")
print(f"Win Rate: {results['win_rate']*100:.2f}%")
print(f"Количество сделок: {results['trades']}")
```

### Реальные результаты sentiment analysis

Исследования 2024-2025 показывают впечатляющие результаты:

- **87% точность прогнозов** для social media sentiment
- **FinBERT + LSTM достигли лучшей производительности** для предсказания движения акций в исследовании 2024 года
- Алгоритмы NLP измеряют оптимизм/страх инвесторов и отслеживают trending tickers на форумах вроде r/WallStreetBets

**Кейс GameStop (GME) 2021:**

Анализ Reddit sentiment в r/WallStreetBets заранее показал экспоненциальный рост интереса к GME. Хедж-фонды, отслеживавшие этот сигнал, смогли войти в long позиции до ракетного роста с $20 до $480.

### Продвинутые техники

**1. Aspect-Based Sentiment Analysis (ABSA)**

Вместо общего сентимента анализируем отношение к конкретным аспектам:

```python
# Пример: анализ отношения к разным продуктам Apple
texts = [
    "iPhone 16 amazing but Vision Pro is disappointing",
    "MacBook Pro performance incredible, iPhone battery terrible"
]

# FinBERT можно fine-tune для ABSA
# Результат: {iPhone: positive, Vision Pro: negative, ...}
```

**2. Multi-Modal Sentiment**

Комбинируем текст, изображения, видео для полной картины:

- Анализ тона голоса на earnings calls (bullish vs bearish)
- Компьютерное зрение для анализа body language CEO
- Sentiment из видео обзоров продуктов на YouTube

## 3. On-Chain данные: Блокчейн не врёт

On-chain метрики для криптовалют — это **прозрачность в её чистом виде**. Каждая транзакция, каждое движение средств записано в блокчейне и доступно для анализа. В 2025-2026 on-chain analysis стал критически важным инструментом для крипто-трейдинга.

### Топ-5 on-chain метрик для 2026

По данным аналитиков Nansen и BeInCrypto, вот ключевые метрики:

#### 1. **Stablecoin Supply — Приоритет #1**

Самая важная on-chain метрика 2026 года. **Корреляция с BTC: 0.87**, часто опережает ралли.

- Stablecoin supply вырос с ~$200 млрд до **$305 млрд в 2025**
- **Stablecoin velocity** (отношение объёма транзакций к рыночной капитализации) — чистейший сигнал реальной on-chain активности

**Логика:** Рост stablecoin supply → больше капитала готов войти в крипту → bullish signal.

#### 2. **DEX vs CEX Activity Shift**

Структурный сдвиг в сторону децентрализованных бирж:

- В 2025: CEX spot volumes упали на **27.7%**, DEX активность выросла на **25.3%**
- DEX volumes в среднем **$500 млрд**, пик в июле: **$857 млрд**
- **Solana и Base** — benchmark платформы для onchain spot trading

**Значение:** Рост DEX активности показывает реальное использование блокчейна, а не просто спекуляции.

#### 3. **Exchange Netflows (притоки/оттоки с бирж)**

- **Outflow с бирж** (withdrawal) → bullish (люди держат в cold storage)
- **Inflow на биржи** (deposits) → bearish (подготовка к продаже)

#### 4. **Whale Activity (активность крупных кошельков)**

Отслеживание движений кошельков с балансом > 1000 BTC или > 10,000 ETH.

#### 5. **Network Health Metrics**

- **Active Addresses**: количество уникальных адресов
- **Transaction Count**: объём транзакций
- **Hash Rate** (для PoW): безопасность сети

### Практика: On-Chain анализ с Python

```python
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List

class OnChainAnalyzer:
    """
    Анализатор on-chain метрик для крипто-трейдинга
    """

    def __init__(self, glassnode_api_key: str = None):
        """
        Args:
            glassnode_api_key: API ключ от Glassnode
        """
        self.glassnode_key = glassnode_api_key
        self.base_url = "https://api.glassnode.com/v1/metrics"

    def get_stablecoin_supply(self, days: int = 365) -> pd.DataFrame:
        """
        Получает данные о supply stablecoins

        Args:
            days: Количество дней истории

        Returns:
            DataFrame с данными supply по датам
        """
        # Glassnode API endpoint
        endpoint = f"{self.base_url}/supply/stablecoin_supply_ratio"

        params = {
            'a': 'BTC',  # asset
            'api_key': self.glassnode_key,
            'since': int((datetime.now() -
                         timedelta(days=days)).timestamp()),
            'until': int(datetime.now().timestamp())
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['t'], unit='s')
            df['stablecoin_ratio'] = df['v']
            return df[['date', 'stablecoin_ratio']].set_index('date')
        else:
            # Fallback: генерируем синтетические данные для примера
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            synthetic_data = {
                'date': dates,
                'stablecoin_ratio': np.cumsum(np.random.randn(days) * 0.01) + 10
            }
            return pd.DataFrame(synthetic_data).set_index('date')

    def get_exchange_netflow(self,
                            asset: str = 'BTC',
                            days: int = 90) -> pd.DataFrame:
        """
        Получает данные о притоках/оттоках с бирж

        Args:
            asset: Тикер криптовалюты (BTC, ETH)
            days: Количество дней истории

        Returns:
            DataFrame с netflow (положительный = inflow, отрицательный = outflow)
        """
        endpoint = f"{self.base_url}/transactions/transfers_volume_exchanges_net"

        params = {
            'a': asset,
            'api_key': self.glassnode_key,
            'since': int((datetime.now() -
                         timedelta(days=days)).timestamp())
        }

        # В реальности делаем запрос к API
        # response = requests.get(endpoint, params=params)

        # Для примера генерируем синтетические данные
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        synthetic_data = {
            'date': dates,
            'netflow': np.random.randn(days) * 1000  # в BTC
        }
        df = pd.DataFrame(synthetic_data).set_index('date')

        # Сглаживаем шум
        df['netflow_ma7'] = df['netflow'].rolling(7).mean()

        return df

    def get_whale_transactions(self,
                              asset: str = 'BTC',
                              threshold: float = 100,
                              days: int = 30) -> pd.DataFrame:
        """
        Получает транзакции китов (> threshold)

        Args:
            asset: Криптовалюта
            threshold: Минимальный размер транзакции для кита (в BTC/ETH)
            days: Период анализа

        Returns:
            DataFrame с whale транзакциями
        """
        # Используем CryptoQuant или Glassnode API
        # Для примера генерируем синтетику

        num_transactions = np.random.randint(50, 200)
        dates = pd.date_range(end=datetime.now(), periods=num_transactions,
                             freq='H')

        whale_txs = {
            'timestamp': dates,
            'amount': np.random.exponential(threshold, num_transactions),
            'type': np.random.choice(['deposit', 'withdrawal'],
                                    num_transactions,
                                    p=[0.4, 0.6])  # больше withdrawals
        }

        df = pd.DataFrame(whale_txs)
        df.set_index('timestamp', inplace=True)

        return df

    def calculate_dex_dominance(self, days: int = 180) -> pd.DataFrame:
        """
        Рассчитывает доминирование DEX vs CEX

        Returns:
            DataFrame с % DEX volume от total
        """
        # В реальности используйте Dune Analytics API
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Симуляция роста DEX dominance
        base = 15  # начальная доминация 15%
        trend = np.linspace(0, 10, days)  # рост до 25%
        noise = np.random.randn(days) * 2

        dex_dominance = base + trend + noise
        dex_dominance = np.clip(dex_dominance, 0, 100)

        df = pd.DataFrame({
            'date': dates,
            'dex_dominance_pct': dex_dominance,
            'cex_dominance_pct': 100 - dex_dominance
        }).set_index('date')

        return df


class OnChainTradingStrategy:
    """
    Торговая стратегия на основе on-chain метрик
    """

    def __init__(self, asset: str = 'BTC'):
        self.asset = asset
        self.analyzer = OnChainAnalyzer()

    def generate_signals(self) -> pd.DataFrame:
        """
        Генерирует торговые сигналы на основе on-chain данных

        Returns:
            DataFrame с сигналами
        """
        # Получаем метрики
        stablecoin_data = self.analyzer.get_stablecoin_supply(days=180)
        exchange_flow = self.analyzer.get_exchange_netflow(days=180)
        dex_dominance = self.analyzer.get_dex_dominance(days=180)

        # Объединяем
        combined = stablecoin_data.join([exchange_flow, dex_dominance],
                                       how='inner')

        # Загружаем цену BTC
        import yfinance as yf
        btc_price = yf.download('BTC-USD',
                               start=combined.index.min(),
                               end=combined.index.max())

        combined = combined.join(btc_price[['Close']], how='inner')
        combined.rename(columns={'Close': 'price'}, inplace=True)

        # === Создаём сигналы ===

        # Signal 1: Stablecoin Supply Ratio рост → Bullish
        combined['stablecoin_signal'] = 0
        combined['stablecoin_ma30'] = combined['stablecoin_ratio'].rolling(30).mean()
        combined.loc[combined['stablecoin_ratio'] > combined['stablecoin_ma30'],
                    'stablecoin_signal'] = 1
        combined.loc[combined['stablecoin_ratio'] < combined['stablecoin_ma30'],
                    'stablecoin_signal'] = -1

        # Signal 2: Exchange Netflow отток → Bullish
        combined['netflow_signal'] = 0
        combined.loc[combined['netflow_ma7'] < -500, 'netflow_signal'] = 1  # outflow
        combined.loc[combined['netflow_ma7'] > 500, 'netflow_signal'] = -1  # inflow

        # Signal 3: DEX Dominance рост → Bullish (реальное использование)
        combined['dex_ma20'] = combined['dex_dominance_pct'].rolling(20).mean()
        combined['dex_signal'] = 0
        combined.loc[combined['dex_dominance_pct'] > combined['dex_ma20'],
                    'dex_signal'] = 1

        # Комбинированный сигнал (взвешенный)
        combined['combined_signal'] = (
            combined['stablecoin_signal'] * 0.5 +  # вес 50%
            combined['netflow_signal'] * 0.3 +      # вес 30%
            combined['dex_signal'] * 0.2            # вес 20%
        )

        # Финальный сигнал: long если > 0.3, short если < -0.3
        combined['position'] = 0
        combined.loc[combined['combined_signal'] > 0.3, 'position'] = 1
        combined.loc[combined['combined_signal'] < -0.3, 'position'] = -1

        return combined

    def backtest(self) -> Dict:
        """
        Бэктест on-chain стратегии

        Returns:
            Словарь с метриками
        """
        signals = self.generate_signals()

        # Рассчитываем доходность
        signals['returns'] = signals['price'].pct_change()
        signals['strategy_returns'] = (signals['position'].shift(1) *
                                      signals['returns'])

        # Метрики
        total_return = (1 + signals['strategy_returns']).prod() - 1
        sharpe = (signals['strategy_returns'].mean() /
                 signals['strategy_returns'].std() * np.sqrt(365))

        # Максимальная просадка
        cumulative = (1 + signals['strategy_returns']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_days = (signals['strategy_returns'] > 0).sum()
        total_days = (signals['strategy_returns'] != 0).sum()
        win_rate = winning_days / total_days if total_days > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': (signals['position'].diff() != 0).sum(),
            'signals_df': signals
        }


# === Использование ===

strategy = OnChainTradingStrategy(asset='BTC')
results = strategy.backtest()

print("=" * 50)
print("On-Chain Trading Strategy Results")
print("=" * 50)
print(f"Total Return: {results['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
print(f"Win Rate: {results['win_rate']*100:.2f}%")
print(f"Total Trades: {results['total_trades']}")

# Визуализация
signals_df = results['signals_df']

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# График 1: Цена и позиции
ax1 = axes[0]
ax1.plot(signals_df.index, signals_df['price'], label='BTC Price', color='black')
ax1.fill_between(signals_df.index,
                 signals_df['price'].min(),
                 signals_df['price'].max(),
                 where=(signals_df['position'] == 1),
                 alpha=0.3, color='green', label='Long Position')
ax1.fill_between(signals_df.index,
                 signals_df['price'].min(),
                 signals_df['price'].max(),
                 where=(signals_df['position'] == -1),
                 alpha=0.3, color='red', label='Short Position')
ax1.set_title('BTC Price and Trading Positions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Stablecoin Supply
ax2 = axes[1]
ax2.plot(signals_df.index, signals_df['stablecoin_ratio'],
        label='Stablecoin Supply Ratio')
ax2.plot(signals_df.index, signals_df['stablecoin_ma30'],
        label='MA30', linestyle='--')
ax2.set_title('Stablecoin Supply Ratio (Key On-Chain Metric)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# График 3: Кумулятивная доходность
ax3 = axes[2]
cumulative_returns = (1 + signals_df['strategy_returns']).cumprod()
buy_hold_returns = (1 + signals_df['returns']).cumprod()
ax3.plot(signals_df.index, cumulative_returns,
        label='On-Chain Strategy', linewidth=2)
ax3.plot(signals_df.index, buy_hold_returns,
        label='Buy & Hold', linestyle='--', alpha=0.7)
ax3.set_title('Cumulative Returns: Strategy vs Buy & Hold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('onchain_strategy_results.png', dpi=150)
print("\nГрафик сохранён: onchain_strategy_results.png")
```

### Реальные данные 2025-2026

**Рыночный контекст:**

- Капитализация крипторынка превысила **$3.5 трлн**
- Bitcoin dominance поднялась выше **62%**
- Более **240,000 крипто-миллионеров** по всему миру

**Тренды 2026:**

- **Solana и Base** подтверждены как benchmark платформы для onchain spot trading
- **Onchain платежи** быстро становятся одним из основных use cases блокчейна, конкурируя с трейдингом
- Дебаты о цикличности: традиционный 4-летний цикл (пик в конце 2025) или продление в 2026?

Аналитики рекомендуют держать **15-20% cash positions** к концу 2025 - началу 2026.

## Комбинированная стратегия: Satellite + Sentiment + On-Chain

Максимальная эффективность достигается при **комбинировании всех трёх источников** альтернативных данных. Хедж-фонды вроде Two Sigma и Citadel используют именно такой подход.

### Пример: Мульти-модальная торговая система

```python
class MultiModalTradingSystem:
    """
    Торговая система, комбинирующая спутниковые данные,
    сентимент и on-chain метрики
    """

    def __init__(self, ticker: str, crypto_asset: str = 'BTC'):
        self.ticker = ticker
        self.crypto_asset = crypto_asset

        # Компоненты системы
        self.satellite_analyzer = SatelliteParkingAnalyzer(
            ee_project='my-project'
        )
        self.sentiment_strategy = SentimentTradingStrategy(ticker)
        self.onchain_strategy = OnChainTradingStrategy(crypto_asset)

    def collect_all_signals(self, days: int = 90) -> pd.DataFrame:
        """
        Собирает сигналы из всех источников
        """
        # 1. Спутниковые данные (для retail stocks)
        parking_data = self.satellite_analyzer.track_retailer(
            locations=[(34.05, -118.24)],  # пример
            ticker=self.ticker,
            days=days
        )
        parking_signal = (parking_data['total_cars'] >
                         parking_data['total_cars'].rolling(14).mean()).astype(int)

        # 2. Sentiment данные
        # Собираем Twitter + Reddit sentiment
        # sentiment_score = ...  (из предыдущего раздела)

        # 3. On-chain данные (для крипто-корреляций)
        onchain_results = self.onchain_strategy.backtest()
        onchain_signals = onchain_results['signals_df']['position']

        # Комбинируем сигналы
        # ...

        return combined_signals

    def execute_strategy(self) -> Dict:
        """
        Выполняет мульти-модальную стратегию
        """
        signals = self.collect_all_signals()

        # Взвешиваем сигналы:
        # - Satellite: 30% (для retail stocks)
        # - Sentiment: 40% (высокая точность 87%)
        # - On-chain: 30% (macro crypto trends)

        # final_signal = 0.3*satellite + 0.4*sentiment + 0.3*onchain

        # Бэктест...

        return results


# Использование
system = MultiModalTradingSystem(ticker='WMT', crypto_asset='BTC')
results = system.execute_strategy()
```

### Ключевые преимущества комбинированного подхода

1. **Диверсификация источников** → снижение риска ложных сигналов
2. **Кросс-верификация** → подтверждение сигналов из разных источников
3. **Покрытие разных временных горизонтов**:
   - Спутниковые данные: недели (early earnings signals)
   - Sentiment: дни-часы (реакция на новости)
   - On-chain: часы-минуты (whale movements)

## Вызовы и ограничения альтернативных данных

Несмотря на впечатляющие результаты, альтернативные данные не панацея. Ключевые вызовы:

### 1. Качество данных

**Проблема:** Raw данные вроде спутниковых снимков или соцсетей требуют значительной обработки.

**Решение:**
- Используйте проверенные платформы (Quandl, Thinknum, Orbital Insight)
- Внедряйте data quality checks
- Очищайте outliers и шум

### 2. Регуляторное соответствие

**Проблема:** Использование некоторых альтернативных данных может нарушать privacy laws (GDPR, CCPA) или insider trading rules.

**Решение:**
- Используйте только публичные данные
- Консультируйтесь с legal team
- Избегайте personal identifiable information (PII)

### 3. Сложность интеграции

**Проблема:** Различные источники данных требуют разных навыков (компьютерное зрение, NLP, blockchain analysis).

**Решение:**
- Инвестируйте в multidisciplinary команду
- Используйте готовые платформы и API
- Начинайте с одного источника, постепенно добавляйте другие

### 4. Высокая стоимость

**Проблема:** Коммерческие спутниковые данные, premium API, infrastructure — всё это дорого. Стоимость команды: **$1.5-2.5 млн**.

**Решение:**
- Начните с бесплатных источников (Reddit API, Google Earth Engine)
- Фокусируйтесь на high-ROI источниках (sentiment показывает 87% accuracy)
- Используйте облачные платформы для масштабирования

### 5. Overfitting Risk

**Проблема:** С тысячами альтернативных датасетов легко найти spurious correlations.

**Решение:**
- Используйте robust валидацию (Walk-Forward, Monte Carlo из предыдущей статьи)
- Тестируйте на out-of-sample данных
- Комбинируйте множество источников для диверсификации

## Будущее альтернативных данных: 2026 и далее

Рынок альтернативных данных мчится к **$135.72 млрд к 2030** с CAGR 63.4%. Вот куда движется индустрия:

### 1. AI-генерация инсайтов

**ChatGPT, Claude, Gemini** будут автоматически анализировать петабайты альтернативных данных, генерируя торговые идеи.

```python
# Будущее: AI-агент анализирует все источники
from openai import OpenAI

client = OpenAI()

prompt = """
Проанализируй следующие альтернативные данные для TSLA:
- Спутниковые снимки: рост активности на парковках на 15%
- Sentiment: Reddit +0.7, Twitter +0.5
- On-chain: рост Bitcoin correlation 0.82

Дай торговую рекомендацию.
"""

response = client.chat.completions.create(
    model="gpt-4.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### 2. Real-Time Streaming

Переход от батч-обработки к real-time стримингу альтернативных данных:

- **Kafka** для потоковой обработки
- **Apache Flink** для real-time аналитики
- **WebSocket APIs** для instant сигналов

### 3. Новые источники данных

- **IoT данные**: подключённые автомобили, smart home устройства
- **Biometric data**: retail foot traffic через facial recognition
- **5G geolocation**: ультра-точное отслеживание перемещений
- **Metaverse activity**: поведение в виртуальных мирах

### 4. Democratization

Альтернативные данные перестанут быть эксклюзивом хедж-фондов:

- **Retail platforms** (Robinhood, eToro) интегрируют sentiment signals
- **Open-source датасеты** станут доступнее
- **Low-cost APIs** для малых фирм

## Практические рекомендации: Как начать

### Для индивидуальных трейдеров

1. **Начните с бесплатных источников:**
   - Reddit API (sentiment analysis)
   - Twitter/X API Basic tier
   - Google Earth Engine (спутниковые снимки)
   - CryptoQuant Free tier (on-chain метрики)

2. **Освойте один тип данных:**
   - Sentiment анализ проще всего для старта
   - FinBERT готов out-of-the-box
   - Быстрые результаты (87% accuracy)

3. **Автоматизируйте сбор:**
   - Напишите скрипты для ежедневного сбора
   - Используйте cron jobs / GitHub Actions
   - Храните в SQLite или PostgreSQL

### Для fund managers

1. **Инвестируйте в команду:**
   - Data Scientists с опытом NLP/Computer Vision
   - Blockchain analysts для on-chain
   - Quant researchers для стратегий

2. **Используйте коммерческие платформы:**
   - **Quandl/Nasdaq Data Link**: 350+ источников
   - **Thinknum**: web scraping as a service
   - **Orbital Insight**: спутниковая аналитика
   - **Glassnode/CryptoQuant**: on-chain премиум

3. **Интегрируйте в risk management:**
   - Альтернативные данные как дополнение, не замена fundamentals
   - Диверсифицируйте источники
   - Мониторьте quality drift

### Библиотеки и инструменты Python

```python
# Sentiment Analysis
from transformers import BertTokenizer, BertForSequenceClassification  # FinBERT
import openai  # ChatGPT для анализа

# Web Scraping
import scrapy  # Мощный фреймворк для больших проектов
from bs4 import BeautifulSoup  # Простые задачи
import requests  # HTTP запросы

# Satellite Imagery
import ee  # Google Earth Engine
from ultralytics import YOLO  # YOLOv8 для детекции объектов
import cv2  # OpenCV для обработки изображений

# On-Chain Analytics
import requests  # Glassnode/CryptoQuant APIs
from web3 import Web3  # Прямая работа с блокчейном

# Data Processing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Backtesting
import backtrader as bt
import vectorbt as vbt

# API Clients
import tweepy  # Twitter/X
import praw  # Reddit
import yfinance as yf  # Yahoo Finance
```

## Заключение: Альфа в данных, которых нет у других

Альтернативные данные — это **информационное преимущество** в его чистейшей форме. Пока большинство трейдеров смотрит на одни и те же графики и индикаторы, хедж-фонды уже знают, что:

- Парковки Walmart заполнены на 18% больше, чем в прошлом квартале → Early signal о росте выручки
- Sentiment в Reddit взлетел до +0.8 для конкретного тикера → Retail buying pressure incoming
- Stablecoin supply вырос на 15% за месяц → $305 млрд готовы влиться в крипту

**Ключевые цифры, которые нужно помнить:**

- **$18.74 млрд** — размер рынка альтернативных данных в 2025
- **87%** — точность прогнозов social media sentiment
- **+18%** — улучшение earnings estimates через satellite imagery
- **0.87** — корреляция stablecoin supply с ценой BTC
- **65%** — доля хедж-фондов, использующих alternative data
- **+3%** — годовая outperformance при использовании alt data
- **20-30%** — прирост точности прогнозов с ML + alt data

Рынок мчится к **$135.72 млрд к 2030** с CAGR 63.4%. Те, кто освоит альтернативные данные сейчас, получат огромное преимущество. Two Sigma и Citadel уже доказали: спутники, сентимент и блокчейн — это не будущее трейдинга. Это настоящее.

**Начните с малого:** Reddit sentiment + FinBERT. **Масштабируйтесь постепенно:** добавьте спутниковые снимки, on-chain метрики. **Комбинируйте источники:** диверсификация сигналов снижает риск. **Автоматизируйте:** создайте pipeline для ежедневного сбора и анализа.

Информация — новая нефть. Альтернативные данные — нефтяные скважины, о которых мало кто знает. Копайте.

---

## Источники

- [Alternative Data for Algorithmic Trading: What Works? - LuxAlgo](https://www.luxalgo.com/blog/alternative-data-for-algorithmic-trading-what-works/)
- [How Alternative Data Enhances Hedge Fund Performance in 2025 - PromptCloud](https://www.promptcloud.com/blog/alternative-data-strategies-for-hedge-funds/)
- [Best Alternative Data Sources in 2025 - Papers With Backtest](https://paperswithbacktest.com/datasets/best-alternative-data)
- [Boost Trading Strategies With Satellite And Social Media Data - AI Competence](https://aicompetence.org/boost-trading-strategies-satellite-social-media/)
- [Track These 5 On-Chain Data For Crypto Trading in 2026 - BeInCrypto](https://beincrypto.com/dune-on-chain-signals-crypto-2026/)
- [Alternative Data Market Size, Share | CAGR of 51.5% - Market.us](https://market.us/report/alternative-data-market/)
- [Alternative Investment Industry Statistics 2025 - CoinLaw](https://coinlaw.io/alternative-investment-industry-statistics/)
- [7 Best Python Web Scraping Libraries for 2025 - ScrapingBee](https://www.scrapingbee.com/blog/best-python-web-scraping-libraries/)
- [FinBERT: Financial Sentiment Analysis with BERT - GitHub ProsusAI](https://github.com/ProsusAI/finBERT)
- [FinBERT on Hugging Face](https://huggingface.co/ProsusAI/finbert)
- [Scrapy vs. Beautiful Soup: A Comparison - Oxylabs](https://oxylabs.io/blog/scrapy-vs-beautifulsoup)
- [Alternative Data Market Global Outlook & Forecast 2024-2029 - Business Wire](https://www.businesswire.com/news/home/20241216292507/en/)
