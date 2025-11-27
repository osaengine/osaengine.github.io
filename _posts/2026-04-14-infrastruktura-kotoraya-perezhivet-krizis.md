---
layout: post
title: "Как проектировать алготрейдинг-инфраструктуру так, чтобы она пережила хотя бы один кризис"
description: "Проектируем отказоустойчивую торговую инфраструктуру с High Availability, Disaster Recovery, Kubernetes, circuit breakers и мониторингом. Реальные примеры архитектур с 99.99% uptime и Recovery Time Objective < 1 минуты."
date: 2026-04-14
image: /assets/images/blog/trading_infrastructure.png
tags: [infrastructure, high-availability, disaster-recovery, kubernetes, monitoring, circuit-breakers]
---

В [предыдущей статье]({{ site.baseurl }}/2026/04/07/10-realnyh-problem-torgovogo-robota.html) мы разобрали 10 критических проблем разработки торговых роботов. Теперь поговорим о том, как спроектировать **инфраструктуру**, которая переживёт не только технические сбои, но и **рыночные кризисы**.

Почему это важно? Согласно [Disaster Recovery Journal](https://drj.com/journal_main/building-resilient-systems-high-availability-vs-disaster-recovery/), для **mission-critical систем** (к которым относятся торговые роботы) Recovery Time Objective (RTO) измеряется **секундами**, а не часами. Downtime в 5 минут во время волатильного рынка может стоить **тысячи долларов** упущенной прибыли или, что хуже, **неконтролируемых убытков** из-за незакрытых позиций.

В этой статье мы рассмотрим:
- High Availability (HA) vs Disaster Recovery (DR)
- Архитектуру с использованием Kubernetes и микросервисов
- Circuit breakers и kill switches
- Мониторинг и observability (Prometheus, Grafana)
- Backup и репликацию данных
- Тестирование отказоустойчивости

Все примеры будут с **реальным кодом** и архитектурными диаграммами.

---

## High Availability vs Disaster Recovery: в чём разница и почему нужно и то, и другое

### High Availability (HA)

**High Availability** — это способность системы работать **непрерывно** без сбоев в течение заранее определённого периода. Согласно [Atmosera](https://www.atmosera.com/blog/high-availability-vs-disaster-recovery/), HA-системы достигают впечатляющих показателей uptime:

- **99.9%** ("three nines") = 8.76 часов downtime в год
- **99.99%** ("four nines") = 52.56 минут downtime в год
- **99.999%** ("five nines") = **5.26 минут downtime в год**

Для торговых систем стандарт — **99.99%** минимум.

**Как достигается HA:**
1. Устранение single points of failure (SPOF)
2. Автоматический failover между репликами
3. Load balancing между несколькими узлами
4. Health checks и автоматическое восстановление

### Disaster Recovery (DR)

**Disaster Recovery** — это процесс **восстановления** системы после катастрофического сбоя. По данным [Cloudian](https://cloudian.com/guides/disaster-recovery/disaster-recovery-vs-high-availability/), DR отличается от HA:

| Характеристика | High Availability | Disaster Recovery |
|---|---|---|
| **Цель** | Предотвратить downtime | Восстановить после катастрофы |
| **Подход** | Превентивный | Реактивный |
| **Время реакции** | Секунды/миллисекунды | Минуты/часы |
| **Фокус** | Fault tolerance, автоматический failover | Backup, restore, бизнес-непрерывность |
| **Применение** | Обычные сбои (упал сервер, сеть) | Катастрофы (пожар ДЦ, региональное отключение) |

**Ключевые метрики DR:**

1. **Recovery Time Objective (RTO)** — максимальное время восстановления
   - Для торговых систем: **< 1 минуты** для критичных компонентов

2. **Recovery Point Objective (RPO)** — допустимая потеря данных (в единицах времени)
   - Для торговых систем: **< 1 секунды** (все сделки должны быть сохранены)

Согласно [Trilio](https://trilio.io/resources/high-availability-vs-disaster-recovery/), **оба подхода дополняют друг друга**: HA предотвращает большинство сбоев, DR спасает в катастрофических случаях.

---

## Архитектура: от монолита к микросервисам на Kubernetes

### Проблема монолитной архитектуры

Типичный "первый торговый робот":

```
┌─────────────────────────────────────┐
│      Monolithic Trading Bot         │
│                                     │
│  ┌──────────────────────────────┐  │
│  │ Data Feed Handler            │  │
│  ├──────────────────────────────┤  │
│  │ Strategy Calculation         │  │
│  ├──────────────────────────────┤  │
│  │ Order Management             │  │
│  ├──────────────────────────────┤  │
│  │ Risk Management              │  │
│  ├──────────────────────────────┤  │
│  │ Database Access              │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Проблемы:**

1. **Single Point of Failure** — если любой компонент падает, весь бот останавливается
2. **Невозможность независимого масштабирования** — нельзя увеличить только Strategy Calculation
3. **Сложность deployment** — изменение в одном модуле требует перезапуска всего бота
4. **Blast radius** — баг в одном модуле уничтожает всю систему

### Микросервисная архитектура

Согласно [Medium: Microservices-Based Algorithmic Trading System](https://usquant349.medium.com/microservices-based-algorithmic-trading-system-83a534c3f132), современные торговые платформы используют **микросервисную архитектуру**:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Market Data  │───▶│   Strategy   │───▶│    Order     │
│   Service    │    │   Service    │    │  Management  │
└──────────────┘    └──────────────┘    └──────────────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Message    │    │     Risk     │    │  Position    │
│ Bus (Kafka)  │◀───│  Management  │───▶│   Service    │
└──────────────┘    └──────────────┘    └──────────────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌──────────────────────────────────────────────────────┐
│         TimescaleDB / PostgreSQL (State)             │
└──────────────────────────────────────────────────────┘
```

**Преимущества:**

1. **Изоляция отказов** — проблема в Strategy Service не убивает Order Management
2. **Независимое масштабирование** — можно запустить 5 реплик Strategy Service, но 1 Order Management
3. **Независимый deployment** — обновляем Strategy Service без перезапуска других
4. **Technology flexibility** — Market Data на Rust (производительность), Strategy на Python (гибкость)

### Kubernetes для оркестрации

Согласно [Effectual: Kubernetes Powers Real-Time Trading](https://effectual.ai/trading-at-the-speed-of-now-how-kubernetes-powers-real-time-insights-rapid-agility-in-financial-markets/), **Kubernetes** обеспечивает:

- **Horizontal scaling** — автоматическое масштабирование под нагрузкой
- **Self-healing** — автоматический перезапуск упавших подов
- **Load balancing** — распределение нагрузки между репликами
- **Rolling updates** — обновление без downtime

Пример Kubernetes deployment для Strategy Service:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strategy-service
  namespace: trading
spec:
  replicas: 3  # 3 реплики для HA
  selector:
    matchLabels:
      app: strategy-service
  template:
    metadata:
      labels:
        app: strategy-service
    spec:
      containers:
      - name: strategy
        image: trading/strategy-service:v1.2.3
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        env:
        - name: KAFKA_BROKER
          value: "kafka:9092"
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
      affinity:
        # Anti-affinity: не размещать все реплики на одном узле
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - strategy-service
            topologyKey: kubernetes.io/hostname
---
apiVersion: v1
kind: Service
metadata:
  name: strategy-service
  namespace: trading
spec:
  selector:
    app: strategy-service
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
  type: ClusterIP
---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: strategy-service-hpa
  namespace: trading
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: strategy-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Что делает этот конфиг:**

1. **Replicas: 3** — всегда 3 реплики работают одновременно
2. **livenessProbe** — если под не отвечает на `/health`, Kubernetes его перезапускает
3. **readinessProbe** — под не получает трафик, пока не готов
4. **podAntiAffinity** — реплики размещаются на **разных узлах** (если один узел упадёт, остальные работают)
5. **HorizontalPodAutoscaler** — автоматически масштабирует от 3 до 10 реплик при высокой нагрузке

**Результат:** если одна реплика падает, Kubernetes автоматически запускает новую, а две оставшиеся продолжают работать. **Downtime ≈ 0 секунд**.

---

## Circuit Breakers и Kill Switches: последняя линия защиты

### Проблема: неконтролируемая торговля

Вспомним [Knight Capital (2012)]({{ site.baseurl }}/2026/04/07/10-realnyh-problem-torgovogo-robota.html#problema-4-otsutstvie-risk-management): за 45 минут система отправила **миллионы ордеров**, потому что **не было автоматических выключателей**.

Согласно [FIA: Best Practices for Automated Trading Risk Controls](https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf), критически важны:

1. **Circuit Breakers** — автоматическая остановка при аномалиях
2. **Kill Switches** — немедленное отключение всей торговли
3. **Pre-trade Risk Controls** — проверка ДО отправки ордера

### Реализация Circuit Breakers

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
from collections import deque
import time

@dataclass
class CircuitBreakerConfig:
    # Лимиты убытков
    max_daily_loss_usd: float = 5000
    max_position_loss_usd: float = 1000
    max_drawdown_pct: float = 15.0

    # Лимиты активности
    max_orders_per_second: int = 5
    max_orders_per_minute: int = 100
    max_open_positions: int = 10

    # Лимиты экспозиции
    max_position_size_usd: float = 50000
    max_total_exposure_usd: float = 100000

    # Аномалии
    max_price_deviation_pct: float = 10.0  # от средней цены
    max_consecutive_losses: int = 5

class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.is_halted = False
        self.halt_reason = None

        # Tracking
        self.order_timestamps = deque()
        self.daily_start_capital = None
        self.peak_capital = None
        self.consecutive_losses = 0
        self.open_positions_count = 0
        self.total_exposure_usd = 0

    def check_pre_trade(self, order_params: dict) -> tuple[bool, Optional[str]]:
        """
        Проверка ДО отправки ордера
        Returns: (can_proceed, halt_reason)
        """
        if self.is_halted:
            return False, f"Trading halted: {self.halt_reason}"

        # 1. Проверка частоты ордеров
        now = time.time()
        self.order_timestamps.append(now)

        # Очистка старых timestamps
        while self.order_timestamps and self.order_timestamps[0] < now - 60:
            self.order_timestamps.popleft()

        # Проверка: ордеров в секунду
        recent_orders_1s = sum(1 for t in self.order_timestamps if t > now - 1)
        if recent_orders_1s >= self.config.max_orders_per_second:
            self.halt("Order rate limit exceeded (1s)", critical=True)
            return False, self.halt_reason

        # Проверка: ордеров в минуту
        if len(self.order_timestamps) >= self.config.max_orders_per_minute:
            self.halt("Order rate limit exceeded (1m)", critical=True)
            return False, self.halt_reason

        # 2. Проверка размера позиции
        position_value = order_params['quantity'] * order_params['price']
        if position_value > self.config.max_position_size_usd:
            return False, f"Position size too large: ${position_value:.2f} > ${self.config.max_position_size_usd}"

        # 3. Проверка общей экспозиции
        new_exposure = self.total_exposure_usd + position_value
        if new_exposure > self.config.max_total_exposure_usd:
            return False, f"Total exposure limit: ${new_exposure:.2f} > ${self.config.max_total_exposure_usd}"

        # 4. Проверка количества открытых позиций
        if order_params['side'] == 'BUY' and self.open_positions_count >= self.config.max_open_positions:
            return False, f"Max open positions reached: {self.open_positions_count}"

        # 5. Проверка аномальной цены
        if 'market_price' in order_params:
            price_deviation = abs(order_params['price'] - order_params['market_price']) / order_params['market_price'] * 100
            if price_deviation > self.config.max_price_deviation_pct:
                return False, f"Price anomaly: {price_deviation:.2f}% deviation"

        return True, None

    def check_post_trade(self, current_capital: float, trade_pnl: float):
        """Проверка ПОСЛЕ исполнения сделки"""

        # Инициализация
        if self.daily_start_capital is None:
            self.daily_start_capital = current_capital
            self.peak_capital = current_capital

        # 1. Проверка дневного убытка
        daily_pnl = current_capital - self.daily_start_capital
        if daily_pnl < -self.config.max_daily_loss_usd:
            self.halt(f"Daily loss limit: ${daily_pnl:.2f}", critical=True)
            return

        # 2. Проверка максимальной просадки
        self.peak_capital = max(self.peak_capital, current_capital)
        drawdown_usd = self.peak_capital - current_capital
        drawdown_pct = (drawdown_usd / self.peak_capital) * 100

        if drawdown_pct > self.config.max_drawdown_pct:
            self.halt(f"Max drawdown exceeded: {drawdown_pct:.2f}%", critical=True)
            return

        # 3. Проверка убытка одной позиции
        if trade_pnl < -self.config.max_position_loss_usd:
            self.halt(f"Single position loss: ${trade_pnl:.2f}", critical=False)
            return

        # 4. Проверка серии убыточных сделок
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.halt(f"Consecutive losses: {self.consecutive_losses}", critical=False)
            return

    def halt(self, reason: str, critical: bool = True):
        """Останавливаем торговлю"""
        self.is_halted = True
        self.halt_reason = reason

        # Логируем и отправляем алерт
        severity = "CRITICAL" if critical else "WARNING"
        print(f"[{severity}] CIRCUIT BREAKER TRIPPED: {reason}")

        # Отправка алерта администратору
        self.send_emergency_alert(reason, critical)

        if critical:
            # Для критических ситуаций — закрываем все позиции
            self.emergency_close_all_positions()

    def emergency_close_all_positions(self):
        """Экстренное закрытие всех позиций"""
        print("[EMERGENCY] Closing all positions...")
        # TODO: реализация через API биржи

    def send_emergency_alert(self, reason: str, critical: bool):
        """Отправка экстренного уведомления"""
        # SMS, Telegram, Email, Phone call
        pass

    def reset_daily(self, current_capital: float):
        """Сброс дневных лимитов (вызывается в начале дня)"""
        self.daily_start_capital = current_capital
        self.consecutive_losses = 0
        # Не сбрасываем is_halted — требуется ручное восстановление

# Использование
config = CircuitBreakerConfig(
    max_daily_loss_usd=5000,
    max_orders_per_second=5,
    max_open_positions=10
)

breaker = CircuitBreaker(config)

# Перед отправкой ордера
order = {
    'side': 'BUY',
    'symbol': 'BTC/USDT',
    'quantity': 0.5,
    'price': 50000,
    'market_price': 49950
}

can_trade, reason = breaker.check_pre_trade(order)
if can_trade:
    # Отправляем ордер
    order_id = exchange.send_order(order)

    # После исполнения
    trade_pnl = -150  # убыток $150
    current_capital = 48500
    breaker.check_post_trade(current_capital, trade_pnl)
else:
    print(f"Order blocked: {reason}")
```

### Kill Switch: аварийное отключение

Согласно [Global Banking and Finance](https://www.globalbankingandfinance.com/kill-switches-the-emperor-s-new-clothes-of-high-frequency-trading), **Kill Switch** должен:

1. Мгновенно останавливать **все** торговые операции
2. Отменять **все pending ордера**
3. Опционально закрывать **все открытые позиции**
4. Быть доступным через **физическую кнопку** и **API**

```python
import signal
import sys
from threading import Event

class KillSwitch:
    def __init__(self, order_manager, position_manager):
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.is_activated = Event()

        # Регистрируем обработчик сигналов
        signal.signal(signal.SIGTERM, self.activate)
        signal.signal(signal.SIGINT, self.activate)  # Ctrl+C

    def activate(self, signum=None, frame=None):
        """Активация Kill Switch"""
        if self.is_activated.is_set():
            return  # уже активирован

        print("\n" + "="*60)
        print("🚨 KILL SWITCH ACTIVATED 🚨")
        print("="*60)

        self.is_activated.set()

        # 1. Останавливаем приём новых ордеров
        self.order_manager.stop_accepting_orders()
        print("[1/4] Stopped accepting new orders")

        # 2. Отменяем все pending ордера
        pending_orders = self.order_manager.get_pending_orders()
        for order_id in pending_orders:
            try:
                self.order_manager.cancel_order(order_id)
                print(f"  Cancelled order: {order_id}")
            except Exception as e:
                print(f"  Failed to cancel {order_id}: {e}")
        print(f"[2/4] Cancelled {len(pending_orders)} pending orders")

        # 3. Опционально: закрываем все позиции
        if self.should_close_positions():
            positions = self.position_manager.get_open_positions()
            for symbol, position in positions.items():
                try:
                    self.position_manager.close_position(symbol, reason="KILL_SWITCH")
                    print(f"  Closed position: {symbol}")
                except Exception as e:
                    print(f"  Failed to close {symbol}: {e}")
            print(f"[3/4] Closed {len(positions)} positions")
        else:
            print("[3/4] Positions left open (manual close required)")

        # 4. Останавливаем все стратегии
        self.stop_all_strategies()
        print("[4/4] Stopped all trading strategies")

        print("\n✓ Kill Switch execution completed")
        print("System is now in SAFE MODE - no trading activity")
        print("="*60 + "\n")

        # Отправляем уведомление
        self.notify_admins()

    def should_close_positions(self) -> bool:
        """Спрашиваем, нужно ли закрывать позиции (можно настроить)"""
        # В продакшене это может быть конфигурационный параметр
        return True  # по умолчанию закрываем

    def stop_all_strategies(self):
        """Останавливаем все торговые стратегии"""
        # Сигнал всем стратегиям прекратить работу
        pass

    def notify_admins(self):
        """Уведомление администраторов"""
        # SMS, звонок, Telegram
        pass

# Использование
kill_switch = KillSwitch(order_manager, position_manager)

# Kill Switch можно активировать:
# 1. Программно: kill_switch.activate()
# 2. Через сигнал ОС: kill -TERM <pid>
# 3. Через Ctrl+C в консоли
# 4. Через HTTP API:

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/emergency/kill-switch', methods=['POST'])
def api_kill_switch():
    """HTTP endpoint для активации Kill Switch"""
    # Требуется аутентификация!
    if not verify_emergency_token(request.headers.get('Authorization')):
        return jsonify({'error': 'Unauthorized'}), 401

    kill_switch.activate()
    return jsonify({'status': 'Kill switch activated'}), 200
```

---

## Мониторинг и Observability: видеть проблемы до катастрофы

### Проблема: слепая система

Без мониторинга вы **не знаете**:
- Сколько ордеров отправлено за последнюю минуту
- Какая латентность исполнения
- Сколько ошибок от биржи
- Какой текущий PnL
- Когда система начала вести себя аномально

### Stack: Prometheus + Grafana

Согласно [Rootly: How SREs Use Prometheus and Grafana](https://rootly.com/sre/how-sres-use-prometheus-and-grafana-to-crush-mttr-in-2025), **70% компаний** используют Prometheus и OpenTelemetry для observability. [The Trade Desk перешёл на Prometheus](https://grafana.com/blog/2019/04/24/the-trade-desk-lessons-we-learned-migrating-from-homegrown-monitoring-to-prometheus/), обрабатывая **11 миллионов запросов в секунду**.

Интеграция Prometheus + Grafana + Loki + Tempo снижает **MTTR (Mean Time To Resolution) на 65%** по сравнению с традиционными системами мониторинга.

### Архитектура мониторинга

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Strategy   │───▶│ Prometheus  │───▶│   Grafana   │
│  Service    │    │   (metrics) │    │ (dashboards)│
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Loki     │    │AlertManager │    │  Telegram   │
│   (logs)    │    │   (alerts)  │    │   /Email    │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Инструментация торгового сервиса

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Метрики
orders_sent_total = Counter(
    'trading_orders_sent_total',
    'Total number of orders sent',
    ['side', 'symbol', 'status']
)

order_execution_latency = Histogram(
    'trading_order_execution_latency_seconds',
    'Order execution latency',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

current_positions = Gauge(
    'trading_current_positions',
    'Current number of open positions'
)

current_pnl = Gauge(
    'trading_current_pnl_usd',
    'Current unrealized PnL in USD'
)

strategy_signals = Counter(
    'trading_strategy_signals_total',
    'Trading signals generated',
    ['signal_type', 'symbol']
)

exchange_errors = Counter(
    'trading_exchange_errors_total',
    'Errors from exchange API',
    ['error_type', 'exchange']
)

class InstrumentedTradingBot:
    def __init__(self):
        # Запускаем HTTP сервер для метрик (Prometheus scrapes на :8000/metrics)
        start_http_server(8000)

    def send_order(self, side, symbol, quantity, price):
        """Отправка ордера с метриками"""
        start_time = time.time()

        try:
            # Отправляем ордер
            order_id = self.exchange.send_order(side, symbol, quantity, price)

            # Записываем успешную метрику
            orders_sent_total.labels(side=side, symbol=symbol, status='success').inc()

            # Латентность
            latency = time.time() - start_time
            order_execution_latency.observe(latency)

            return order_id

        except Exception as e:
            # Записываем ошибку
            orders_sent_total.labels(side=side, symbol=symbol, status='error').inc()
            exchange_errors.labels(error_type=type(e).__name__, exchange='binance').inc()
            raise

    def on_strategy_signal(self, signal_type, symbol):
        """Обработка торгового сигнала"""
        strategy_signals.labels(signal_type=signal_type, symbol=symbol).inc()

        # ... логика обработки сигнала

    def update_metrics(self, positions, unrealized_pnl):
        """Обновление текущих метрик"""
        current_positions.set(len(positions))
        current_pnl.set(unrealized_pnl)

# Использование
bot = InstrumentedTradingBot()

# Отправляем ордер
bot.send_order('BUY', 'BTC/USDT', 0.1, 50000)

# Prometheus теперь может скрейпить метрики с :8000/metrics
# Output:
# trading_orders_sent_total{side="BUY",symbol="BTC/USDT",status="success"} 1.0
# trading_order_execution_latency_seconds_bucket{le="0.05"} 1.0
# trading_current_positions 5.0
# trading_current_pnl_usd 1234.56
```

### Prometheus конфигурация

```yaml
# prometheus.yml
global:
  scrape_interval: 5s  # Скрейпим каждые 5 секунд
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'trading-strategy-service'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - trading
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: strategy-service
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace

  - job_name: 'trading-order-service'
    static_configs:
      - targets: ['order-service:8000']

# Правила алертинга
rule_files:
  - 'alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Правила алертинга

```yaml
# alerts.yml
groups:
  - name: trading_alerts
    interval: 10s
    rules:
      # Высокая частота ордеров
      - alert: HighOrderRate
        expr: rate(trading_orders_sent_total[1m]) > 100
        for: 30s
        labels:
          severity: warning
        annotations:
          summary: "High order rate detected"
          description: "Order rate is {{ $value }} orders/sec (threshold: 100)"

      # Высокий процент ошибок
      - alert: HighErrorRate
        expr: |
          rate(trading_exchange_errors_total[5m]) /
          rate(trading_orders_sent_total[5m]) > 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate from exchange"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # Критический drawdown
      - alert: CriticalDrawdown
        expr: trading_current_pnl_usd < -5000
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Critical PnL drawdown"
          description: "Current PnL: ${{ $value }}"

      # Сервис недоступен
      - alert: TradingServiceDown
        expr: up{job="trading-strategy-service"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Trading service is down"
          description: "{{ $labels.pod }} is not responding"

      # Высокая латентность
      - alert: HighOrderLatency
        expr: |
          histogram_quantile(0.95,
            rate(trading_order_execution_latency_seconds_bucket[5m])
          ) > 1.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High order execution latency"
          description: "P95 latency is {{ $value }}s (threshold: 1s)"
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Trading System Overview",
    "panels": [
      {
        "title": "Orders per Second",
        "targets": [
          {
            "expr": "rate(trading_orders_sent_total[1m])",
            "legendFormat": "{{side}} {{symbol}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Current PnL",
        "targets": [
          {
            "expr": "trading_current_pnl_usd"
          }
        ],
        "type": "stat",
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "thresholds": {
              "steps": [
                {"value": -5000, "color": "red"},
                {"value": 0, "color": "yellow"},
                {"value": 1000, "color": "green"}
              ]
            }
          }
        }
      },
      {
        "title": "Order Execution Latency (P95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(trading_order_execution_latency_seconds_bucket[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(trading_exchange_errors_total[5m])",
            "legendFormat": "{{error_type}}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

## Backup и репликация данных: чтобы не потерять всё

### Recovery Point Objective (RPO) < 1 секунды

Для торговой системы **каждая сделка критична**. Потеря даже одной записи о сделке может привести к:
- Неправильному расчёту PnL
- Ошибкам в позициях (думаем позиция открыта, а она закрыта)
- Регуляторным проблемам (нет audit trail)

### PostgreSQL Streaming Replication

```
┌────────────────┐           ┌────────────────┐
│   Primary DB   │──WAL────▶│  Standby DB 1  │
│  (read/write)  │           │  (read-only)   │
└────────────────┘           └────────────────┘
        │                            │
        │ WAL                        │
        ▼                            ▼
┌────────────────┐           ┌────────────────┐
│  Standby DB 2  │           │   S3 Backup    │
│  (read-only)   │           │   (archival)   │
└────────────────┘           └────────────────┘
```

**PostgreSQL конфигурация (primary):**

```ini
# postgresql.conf на primary
wal_level = replica
max_wal_senders = 3
wal_keep_size = 1GB
synchronous_commit = on
synchronous_standby_names = 'standby1'

# Включаем архивацию WAL в S3
archive_mode = on
archive_command = 'aws s3 cp %p s3://trading-db-backups/wal/%f'
```

**Standby конфигурация:**

```ini
# postgresql.conf на standby
hot_standby = on
primary_conninfo = 'host=primary-db port=5432 user=replicator password=xxx'
```

**Результат:**
- Любая запись в primary **мгновенно** реплицируется на standby
- При падении primary, standby автоматически становится primary (с помощью Patroni или repmgr)
- **RPO ≈ 0 секунд** (синхронная репликация)
- **RTO ≈ 30 секунд** (автоматический failover)

### TimescaleDB для time-series данных

Цены, сделки, метрики — это **time-series данные**. TimescaleDB (расширение PostgreSQL) оптимизировано для этого:

```sql
-- Создаём hypertable для цен
CREATE TABLE market_prices (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price NUMERIC NOT NULL,
    volume NUMERIC NOT NULL
);

-- Превращаем в hypertable
SELECT create_hypertable('market_prices', 'time');

-- Автоматическое партиционирование по времени
-- Retention policy: удаляем данные старше 90 дней
SELECT add_retention_policy('market_prices', INTERVAL '90 days');

-- Continuous aggregates для быстрых запросов
CREATE MATERIALIZED VIEW price_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    FIRST(price, time) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, time) AS close,
    SUM(volume) AS volume
FROM market_prices
GROUP BY bucket, symbol;

-- Автоматическое обновление aggregate каждую минуту
SELECT add_continuous_aggregate_policy('price_1min',
    start_offset => INTERVAL '2 minutes',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');
```

### Backup стратегия: 3-2-1 правило

**3** копии данных, на **2** разных типах носителей, **1** копия offsite.

```bash
#!/bin/bash
# Скрипт ежедневного бэкапа

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"
S3_BUCKET="s3://trading-db-backups"

# 1. Full backup с pg_dump
pg_dump -h localhost -U postgres trading_db | gzip > $BACKUP_DIR/trading_db_$DATE.sql.gz

# 2. Загрузка в S3 (offsite)
aws s3 cp $BACKUP_DIR/trading_db_$DATE.sql.gz $S3_BUCKET/daily/

# 3. Загрузка в Glacier для долгосрочного хранения (каждое воскресенье)
if [ $(date +%u) -eq 7 ]; then
    aws s3 cp $BACKUP_DIR/trading_db_$DATE.sql.gz $S3_BUCKET/weekly/ --storage-class GLACIER
fi

# 4. Удаляем локальные бэкапы старше 7 дней
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

# 5. Проверяем, что бэкап валиден
gunzip -t $BACKUP_DIR/trading_db_$DATE.sql.gz
if [ $? -eq 0 ]; then
    echo "Backup successful: trading_db_$DATE.sql.gz"
else
    echo "ERROR: Backup corrupted!" | mail -s "Backup Failed" admin@example.com
fi
```

---

## Тестирование отказоустойчивости: Chaos Engineering

### Проблема: тестируем только happy path

Большинство разработчиков тестируют, что система **работает** при нормальных условиях. Но не тестируют, что будет при:
- Упадёт база данных
- Пропадёт сеть с биржей
- Заполнится диск
- Убьют случайный под в Kubernetes

**Chaos Engineering** — это дисциплина экспериментирования над распределённой системой для уверенности в её способности выдержать турбулентные условия.

### Chaos Toolkit для торговой системы

```yaml
# chaos-experiment.yml
version: 1.0.0
title: "Trading System Resilience Test"
description: "Kill strategy service pod and verify automatic recovery"

steady-state-hypothesis:
  title: "System is healthy and processing orders"
  probes:
    - name: "strategy-service-is-running"
      type: probe
      tolerance: 3  # минимум 3 реплики
      provider:
        type: python
        module: chaosk8s.pod.probes
        func: count_pods
        arguments:
          label_selector: "app=strategy-service"
          ns: "trading"

    - name: "orders-are-being-processed"
      type: probe
      tolerance: true
      provider:
        type: http
        url: "http://prometheus:9090/api/v1/query?query=rate(trading_orders_sent_total[1m])"
        expected_status: 200

method:
  - type: action
    name: "kill-random-strategy-pod"
    provider:
      type: python
      module: chaosk8s.pod.actions
      func: terminate_pods
      arguments:
        label_selector: "app=strategy-service"
        ns: "trading"
        qty: 1  # убиваем 1 под
        rand: true  # случайный

  - type: probe
    name: "wait-for-recovery"
    provider:
      type: python
      func: time.sleep
      arguments:
        seconds: 30

  - type: probe
    name: "verify-pod-count-restored"
    tolerance: 3
    provider:
      type: python
      module: chaosk8s.pod.probes
      func: count_pods
      arguments:
        label_selector: "app=strategy-service"
        ns: "trading"

  - type: probe
    name: "verify-orders-still-processing"
    tolerance: true
    provider:
      type: http
        url: "http://prometheus:9090/api/v1/query?query=rate(trading_orders_sent_total[1m])"
        expected_status: 200

rollbacks: []
```

**Запуск эксперимента:**

```bash
chaos run chaos-experiment.yml
```

**Что проверяет:**
1. Steady state: 3 реплики strategy-service работают, ордера обрабатываются
2. Убиваем 1 случайный под
3. Ждём 30 секунд
4. Проверяем: вернулись ли 3 реплики? (Kubernetes должен автоматически запустить новый под)
5. Проверяем: продолжают ли обрабатываться ордера?

Если эксперимент **проваливается** — значит система **не отказоустойчива**.

### Другие сценарии Chaos Engineering

```python
# Пример: имитация сетевой задержки
from chaoslib.types import Configuration, Secrets

def add_network_latency(configuration: Configuration, secrets: Secrets):
    """Добавляет 500ms латентности к биржевому API"""
    # Используем tc (traffic control) в Linux
    os.system("tc qdisc add dev eth0 root netem delay 500ms")

def remove_network_latency(configuration: Configuration, secrets: Secrets):
    """Убираем латентность"""
    os.system("tc qdisc del dev eth0 root")

# Пример: заполнение диска
def fill_disk(path="/data", size_mb=1000):
    """Заполняет диск на size_mb мегабайт"""
    with open(f"{path}/chaos_filler.tmp", "wb") as f:
        f.write(b"0" * (size_mb * 1024 * 1024))

# Пример: убийство случайного процесса
def kill_random_process(process_name="postgres"):
    """Убивает случайный процесс postgres"""
    os.system(f"pkill -9 {process_name}")
```

---

## Заключение: Checklist отказоустойчивой инфраструктуры

Подведём итоги. Ваша торговая инфраструктура готова к кризисам, если:

### ✓ High Availability
- [ ] Минимум **3 реплики** каждого критичного сервиса
- [ ] **Anti-affinity** rules — реплики на разных узлах
- [ ] **Health checks** (liveness/readiness probes)
- [ ] **Load balancing** между репликами
- [ ] **Auto-scaling** при высокой нагрузке
- [ ] Целевой uptime: **99.99%** (52 минуты downtime в год)

### ✓ Disaster Recovery
- [ ] **RTO < 1 минута** для критичных компонентов
- [ ] **RPO < 1 секунда** (синхронная репликация БД)
- [ ] **Backup**: ежедневный full backup в S3
- [ ] **Backup testing**: регулярная проверка restore
- [ ] **Multi-region** deployment для защиты от региональных сбоев
- [ ] **Runbook** для процедуры восстановления

### ✓ Circuit Breakers & Kill Switches
- [ ] **Pre-trade checks**: размер позиции, лимиты, аномалии цен
- [ ] **Post-trade checks**: дневной PnL, drawdown, серии убытков
- [ ] **Rate limiting**: ордеров в секунду/минуту
- [ ] **Kill switch**: HTTP API + физическая кнопка
- [ ] **Emergency contacts**: SMS/звонок при критических событиях

### ✓ Monitoring & Observability
- [ ] **Prometheus** для метрик (orders, latency, errors, PnL)
- [ ] **Grafana** дашборды в реальном времени
- [ ] **AlertManager** для автоматических алертов
- [ ] **Loki** для централизованных логов
- [ ] **Distributed tracing** (Tempo/Jaeger) для debugging
- [ ] **Heartbeat monitoring** (healthchecks.io)
- [ ] Целевой **MTTR < 5 минут**

### ✓ Data Resilience
- [ ] **PostgreSQL streaming replication** (primary + 2 standby)
- [ ] **TimescaleDB** для time-series (цены, сделки)
- [ ] **Synchronous commit** для критичных данных
- [ ] **3-2-1 backup** (3 копии, 2 типа носителей, 1 offsite)
- [ ] **WAL archiving** в S3/Glacier
- [ ] **Point-in-time recovery** возможность

### ✓ Architecture
- [ ] **Микросервисы** вместо монолита
- [ ] **Event-driven** через Kafka/NATS
- [ ] **Kubernetes** для оркестрации
- [ ] **Stateless services** где возможно
- [ ] **Graceful shutdown** (закрытие соединений перед остановкой)
- [ ] **Idempotency** (повторная отправка ордера не создаёт дубликат)

### ✓ Chaos Engineering
- [ ] Регулярное тестирование failover (минимум раз в месяц)
- [ ] Chaos experiments: убийство подов, сетевые задержки, заполнение диска
- [ ] **Game days**: симуляция кризисных сценариев с командой
- [ ] Документация инцидентов и post-mortems

---

**Ключевой урок**: отказоустойчивость — это **не одна технология**, а **комплексный подход**:

1. **HA** предотвращает 99% проблем
2. **DR** спасает в катастрофических 1%
3. **Circuit Breakers** защищают от финансовых потерь
4. **Monitoring** даёт видимость и быстрое реагирование
5. **Chaos Engineering** проверяет, что всё вышеперечисленное работает

Согласно [Markets and Markets](https://www.marketsandmarkets.com/Market-Reports/recovery-as-a-service-market-962.html), рынок Disaster Recovery as a Service вырастет с $4.7B в 2024 до $72.3B в 2037 — **рост в 15 раз**. Это показывает критичность отказоустойчивости для современного бизнеса.

В следующей статье мы разберём **типичные ошибки начинающих** в алготрейдинге на реальных примерах с кодом и метриками.

---

**Источники:**

- [Building Resilient Systems: High Availability vs. Disaster Recovery](https://drj.com/journal_main/building-resilient-systems-high-availability-vs-disaster-recovery/)
- [High Availability vs Disaster Recovery](https://www.atmosera.com/blog/high-availability-vs-disaster-recovery/)
- [Disaster Recovery vs. High Availability](https://cloudian.com/guides/disaster-recovery/disaster-recovery-vs-high-availability/)
- [Kubernetes Powers Real-Time Trading](https://effectual.ai/trading-at-the-speed-of-now-how-kubernetes-powers-real-time-insights-rapid-agility-in-financial-markets/)
- [Microservices-Based Algorithmic Trading System](https://usquant349.medium.com/microservices-based-algorithmic-trading-system-83a534c3f132)
- [FIA: Best Practices for Automated Trading Risk Controls](https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf)
- [Kill Switches in High Frequency Trading](https://www.globalbankingandfinance.com/kill-switches-the-emperor-s-new-clothes-of-high-frequency-trading/)
- [How SREs Use Prometheus and Grafana to Crush MTTR in 2025](https://rootly.com/sre/how-sres-use-prometheus-and-grafana-to-crush-mttr-in-2025)
- [The Trade Desk: Migrating to Prometheus](https://grafana.com/blog/2019/04/24/the-trade-desk-lessons-we-learned-migrating-from-homegrown-monitoring-to-prometheus/)
- [Disaster Recovery as a Service Market](https://www.marketsandmarkets.com/Market-Reports/recovery-as-a-service-market-962.html)
