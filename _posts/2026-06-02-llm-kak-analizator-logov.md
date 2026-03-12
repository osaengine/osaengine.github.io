---
layout: post
title: "LLM как \"анализатор логов\" алгосистемы: поиск аномалий и повторяющихся ошибок"
description: "Практическое применение LLM для анализа логов торговых систем: автоматическое выявление аномалий, паттернов ошибок, корреляций между событиями. Реальные кейсы и промпты для ChatGPT/Claude."
date: 2026-06-02
image: /assets/images/blog/llm_log_analyzer.png
tags: [LLM, ChatGPT, Claude, logs, debugging, anomaly-detection]
---

В [предыдущей статье]({{ site.baseurl }}/2026/05/26/avtomatizaciya-dokumentacii-s-ai.html) мы автоматизировали документацию с помощью LLM. Теперь применим LLM для решения ещё более практической задачи: **анализа логов торговой системы**.

Торговый робот генерирует гигабайты логов: каждая сделка, каждая ошибка, каждое изменение цены. Человеку невозможно вручную анализировать такой объём. LLM справляется отлично.

---

## Проблема: утопление в логах

### Типичный лог торгового робота

```
2025-03-20 09:15:23.145 [INFO] MarketData - BTC/USDT price update: 50123.45
2025-03-20 09:15:23.167 [INFO] Strategy - RSI: 32.4, Signal: BUY
2025-03-20 09:15:23.234 [INFO] OrderManager - Placing order: BUY 0.1 BTC @ 50123.45
2025-03-20 09:15:23.456 [INFO] Exchange - Order sent: #12345
2025-03-20 09:15:24.123 [INFO] Exchange - Order filled: #12345 @ 50125.30
2025-03-20 09:15:24.145 [INFO] Portfolio - Position opened: BTC/USDT, size: 0.1, entry: 50125.30
2025-03-20 09:15:24.167 [INFO] RiskManager - Stop-loss set: 48621.75 (-3.0%)

... 10,000 строк спустя ...

2025-03-20 14:32:15.234 [ERROR] Exchange - Order placement failed: Insufficient funds
2025-03-20 14:32:15.256 [WARNING] OrderManager - Retrying order #12389...
2025-03-20 14:32:16.123 [ERROR] Exchange - Order placement failed: Insufficient funds
2025-03-20 14:32:16.145 [ERROR] OrderManager - Max retries exceeded for order #12389
2025-03-20 14:32:16.167 [ERROR] Strategy - Failed to open position for ETH/USDT

... ещё 50,000 строк ...

2025-03-20 18:45:32.123 [WARNING] Connection - WebSocket disconnected from binance
2025-03-20 18:45:32.234 [INFO] Connection - Reconnecting to binance...
2025-03-20 18:45:33.456 [INFO] Connection - Connected to binance
2025-03-20 18:45:33.567 [WARNING] MarketData - Price data gap detected: 1.2 seconds

... и так весь день ...
```

**Вопросы, которые нужно ответить:**
1. Почему провалилась сделка ETH/USDT в 14:32?
2. Есть ли паттерн в ошибках "Insufficient funds"?
3. Влияют ли WebSocket disconnects на производительность?
4. Какие аномалии произошли сегодня?

Вручную искать — часы работы. С LLM — 30 секунд.

---

## Решение #1: Поиск ошибок и их кластеризация

### Задача

За день накопилось 500 строк с ERROR/WARNING. Нужно понять:
- Какие ошибки критичные?
- Какие повторяются?
- Есть ли новые типы ошибок?

### Промпт для LLM

```python
from openai import OpenAI
import re
from collections import defaultdict

client = OpenAI()

def extract_errors_from_log(log_file_path):
    """Extract all ERROR and WARNING lines"""
    errors = []

    with open(log_file_path, 'r') as f:
        for line in f:
            if '[ERROR]' in line or '[WARNING]' in line:
                errors.append(line.strip())

    return errors

def analyze_errors_with_llm(errors):
    """Analyze errors with LLM"""

    errors_text = '\n'.join(errors[:100])  # First 100 errors (token limit)

    prompt = f"""
You are a senior DevOps engineer analyzing logs from a cryptocurrency trading bot.

Here are ERROR and WARNING messages from today's logs:

{errors_text}

Analyze these errors and provide:

1. **Critical Issues** (errors that could lose money or crash the system)
   - List each critical issue
   - Explain impact
   - Suggest immediate action

2. **Error Patterns** (recurring errors)
   - Group similar errors
   - Count occurrences
   - Identify root cause

3. **Anomalies** (unusual or new errors)
   - Highlight errors that don't fit patterns
   - Flag potential bugs

4. **Recommendations** (prioritized action items)
   - What to fix first
   - What can wait
   - What to monitor

Format as Markdown report.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a trading system reliability engineer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# Usage
errors = extract_errors_from_log('trading_bot.log')
print(f"Found {len(errors)} errors/warnings")

analysis = analyze_errors_with_llm(errors)
print(analysis)
```

### Пример результата от LLM

```markdown
# Trading Bot Error Analysis - 2025-03-20

## 🚨 Critical Issues (Immediate Action Required)

### 1. Insufficient Funds Errors (47 occurrences)
**Pattern:**
```
[ERROR] Exchange - Order placement failed: Insufficient funds
[ERROR] OrderManager - Max retries exceeded for order #12389
```

**Impact:** HIGH
- Lost trading opportunities (ETH/USDT, SOL/USDT, AVAX/USDT)
- Strategy unable to execute signals
- Potential missed profit

**Root Cause:**
Position sizing calculation appears to be wrong. Orders are placed for amounts larger than available balance.

**Immediate Action:**
1. Check `calculate_position_size()` function in RiskManager
2. Verify that commission fees are included in position size calculation
3. Add balance check BEFORE sending order to exchange

**Code location:** Likely in `src/risk_manager.py:calculate_position_size()`

---

### 2. WebSocket Disconnections (23 occurrences)
**Pattern:**
```
[WARNING] Connection - WebSocket disconnected from binance
[INFO] Connection - Reconnecting to binance...
```

**Impact:** MEDIUM
- Price data gaps (up to 1.2 seconds)
- Potential missed signals during disconnection
- Increased latency after reconnect

**Root Cause:**
Network instability OR exchange-side connection resets.

**Immediate Action:**
1. Check if disconnects correlate with specific times (exchange maintenance?)
2. Implement connection pooling with multiple WebSocket streams
3. Add heartbeat mechanism to detect stale connections earlier

---

## 📊 Error Patterns

### Pattern #1: Position Sizing Errors (47 instances, 31% of all errors)
```
14:32:15 - Insufficient funds for ETH/USDT
14:35:22 - Insufficient funds for SOL/USDT
14:41:08 - Insufficient funds for AVAX/USDT
...
```

**Frequency:** Peaked at 14:30-15:00 (15 errors in 30 minutes)

**Correlation:** All occurred when BTC price was dropping (-3.5%)
- Hypothesis: Stop-losses triggered → freed capital → tried to open new positions → calculation error

---

### Pattern #2: Order Retry Failures (38 instances, 25% of all errors)
```
[WARNING] OrderManager - Retrying order #12389...
[ERROR] OrderManager - Max retries exceeded
```

**Common sequence:**
1. Initial order fails (insufficient funds)
2. Retry 1 fails
3. Retry 2 fails
4. Max retries exceeded

**Problem:** Retrying doesn't make sense if root cause is insufficient funds!

**Fix:** Check error type before retrying. Don't retry "Insufficient funds" errors.

---

### Pattern #3: Price Data Gaps (12 instances, 8% of all errors)
```
[WARNING] MarketData - Price data gap detected: 1.2 seconds
```

**Times:**
- 10:15 (gap: 0.8s)
- 12:45 (gap: 1.2s)
- 18:45 (gap: 1.1s)

**Correlation:** All gaps occurred within 5 minutes of WebSocket disconnect

**Impact:** Strategies received stale data, potentially wrong signals

---

## 🔍 Anomalies (New/Unusual Errors)

### 1. New Error Type (First seen today)
```
[ERROR] Strategy - TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'
```

**Occurred:** 1 time at 16:23:45

**Context:** Immediately after price data gap

**Hypothesis:** Strategy received None instead of price, arithmetic operation failed

**Action:** Add null-check in strategy code:
```python
if price is None:
    logger.warning("Received None price, skipping bar")
    return
```

---

### 2. Unusual Volume Spike Alert
```
[WARNING] RiskManager - Position size capped at max 25% of capital (requested: 45%)
```

**Occurred:** 3 times (unusual, typically 0-1 per day)

**Context:** All during high volatility periods

**Analysis:** Position sizing calculation is correct, but volatility increased → stop-loss wider → smaller position size recommended

**Action:** Consider adding volatility-adjusted position sizing

---

## ✅ Recommendations (Prioritized)

### Priority 1: Fix Insufficient Funds (Today)
1. Add balance check before sending orders
2. Include commission in position size calculation
3. Add unit test: `test_position_size_includes_commission()`

**Estimated fix time:** 2 hours
**Impact:** Eliminates 31% of errors

---

### Priority 2: Improve Error Handling (This Week)
1. Don't retry "Insufficient funds" errors
2. Add specific error handling for different error types
3. Implement exponential backoff for network errors

**Estimated fix time:** 4 hours
**Impact:** Reduces error spam, faster issue detection

---

### Priority 3: WebSocket Reliability (This Week)
1. Implement dual WebSocket connections (primary + backup)
2. Add connection health monitoring
3. Create alert if >5 disconnects per hour

**Estimated fix time:** 1 day
**Impact:** Reduces price data gaps by ~80%

---

### Priority 4: Monitoring (Next Week)
1. Add dashboard for error rates by type
2. Alert if new error type appears
3. Daily summary report of errors

**Estimated fix time:** 1 day
**Impact:** Proactive issue detection

---

## 📈 Statistics

- **Total errors:** 152
- **Total warnings:** 89
- **Unique error types:** 7
- **Most common:** Insufficient funds (31%)
- **Error rate:** 241 per 24h = 10 per hour
- **Trend:** ⬆️ +15% vs yesterday

**Conclusion:** System stability is degrading. Immediate attention required for position sizing bug.
```

### Оценка результата

✅ **Что LLM сделал хорошо:**
- Выделил критические проблемы
- Сгруппировал похожие ошибки
- Нашёл корреляции (errors during BTC drop, after WebSocket disconnect)
- Предложил конкретные действия с приоритетами
- Указал предположительное место в коде

⚠️ **Что нужно проверить:**
- Гипотезы о root cause (нужно подтвердить в коде)
- Оценки времени исправления (могут быть неточными)

**Экономия времени:** 2-3 часа ручного анализа логов → 30 секунд с LLM

---

## Решение #2: Поиск аномалий в паттернах поведения

### Задача

Робот работает нормально 99% времени. Но иногда что-то идёт не так. Как обнаружить аномалии **до** того, как они приведут к убыткам?

### Подход: Анализ временных паттернов

```python
def extract_time_series_metrics(log_file_path):
    """Extract time-series metrics from logs"""

    metrics = {
        'trades_per_hour': defaultdict(int),
        'errors_per_hour': defaultdict(int),
        'avg_latency_per_hour': defaultdict(list),
        'position_sizes': [],
        'pnl_per_trade': []
    }

    with open(log_file_path, 'r') as f:
        for line in f:
            timestamp_str = line[:19]  # "2025-03-20 09:15:23"
            hour = timestamp_str[:13]  # "2025-03-20 09"

            # Count trades
            if 'Order filled' in line:
                metrics['trades_per_hour'][hour] += 1

                # Extract position size
                match = re.search(r'size: ([\d.]+)', line)
                if match:
                    metrics['position_sizes'].append(float(match.group(1)))

            # Count errors
            if '[ERROR]' in line:
                metrics['errors_per_hour'][hour] += 1

            # Extract latency
            if 'latency:' in line:
                match = re.search(r'latency: ([\d.]+)ms', line)
                if match:
                    metrics['avg_latency_per_hour'][hour].append(float(match.group(1)))

            # Extract PnL
            if 'Position closed' in line:
                match = re.search(r'PnL: ([-+]?[\d.]+)', line)
                if match:
                    metrics['pnl_per_trade'].append(float(match.group(1)))

    return metrics

def detect_anomalies_with_llm(metrics):
    """Use LLM to detect anomalies in metrics"""

    # Format metrics for LLM
    summary = f"""
Trading Activity:
{format_time_series(metrics['trades_per_hour'])}

Error Rate:
{format_time_series(metrics['errors_per_hour'])}

Average Latency:
{format_time_series_avg(metrics['avg_latency_per_hour'])}

Position Sizes:
Min: {min(metrics['position_sizes']):.4f}
Max: {max(metrics['position_sizes']):.4f}
Mean: {np.mean(metrics['position_sizes']):.4f}
Std: {np.std(metrics['position_sizes']):.4f}

PnL per Trade:
Min: ${min(metrics['pnl_per_trade']):.2f}
Max: ${max(metrics['pnl_per_trade']):.2f}
Mean: ${np.mean(metrics['pnl_per_trade']):.2f}
Std: ${np.std(metrics['pnl_per_trade']):.2f}
"""

    prompt = f"""
You are an anomaly detection expert for trading systems.

Analyze these metrics from a trading bot's daily logs:

{summary}

Identify:
1. **Statistical Anomalies**
   - Values significantly outside normal range (>2 std deviations)
   - Sudden spikes or drops

2. **Time-based Patterns**
   - Unusual activity at specific hours
   - Degradation over time

3. **Correlations**
   - Does high error rate correlate with high latency?
   - Does trading activity correlate with PnL?

4. **Red Flags**
   - Indicators of potential problems
   - Recommendations for investigation

Be specific with numbers and timestamps.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# Usage
metrics = extract_time_series_metrics('trading_bot.log')
anomalies = detect_anomalies_with_llm(metrics)
print(anomalies)
```

### Пример результата

```markdown
# Anomaly Detection Report - 2025-03-20

## 🔴 Statistical Anomalies

### 1. Trading Activity Spike (14:00-15:00)
**Normal:** 5-8 trades/hour
**Observed:** 23 trades/hour (14:00-15:00)
**Deviation:** +187% (2.9 standard deviations)

**Analysis:**
This is a statistically significant anomaly. Possible causes:
- Strategy entered "aggressive mode" due to market volatility
- Multiple signals triggered simultaneously
- Bug causing duplicate orders

**Investigation needed:**
- Check if all 23 trades were intentional
- Review strategy logic for 14:00-15:00 period
- Verify no duplicate order IDs

---

### 2. Latency Spike (18:45-19:00)
**Normal:** 50-80ms
**Observed:** 450ms average (18:45), peak 1200ms
**Deviation:** +462% (5.6 standard deviations) 🚨

**Critical:** This is extremely unusual!

**Timeline:**
- 18:44:59 - Latency normal (65ms)
- 18:45:15 - WebSocket disconnect
- 18:45:33 - Reconnected
- 18:45:35 - Latency spike to 450ms
- 18:46:12 - Latency gradually returns to normal (120ms)
- 18:47:30 - Latency back to baseline (70ms)

**Root Cause:** WebSocket disconnect caused data backlog → processing delay

**Impact:**
- 3 trades executed during high-latency period
- Potential slippage: estimated $15-30 per trade
- Total impact: ~$45-90 lost to poor execution

**Recommendation:**
Implement circuit breaker: halt trading if latency >200ms

---

### 3. Negative PnL Cluster (16:00-17:00)
**Normal:** 60% win rate, mean PnL +$12 per trade
**Observed (16:00-17:00):** 20% win rate, mean PnL -$28 per trade

**Trades:**
- 16:15 - SOL/USDT: -$45
- 16:23 - ETH/USDT: -$32
- 16:41 - BTC/USDT: -$18
- 16:55 - AVAX/USDT: -$22
- 16:58 - BTC/USDT: +$8 (only winner)

**Analysis:**
5 trades in 1 hour, 4 losses = 80% loss rate (vs normal 40%)

**Common factor:** All occurred during BTC price drop (-2.5% in 45 minutes)

**Hypothesis:** Strategy not adapted for high-volatility downtrend
- Entries based on "oversold" signals
- But market kept dropping (no bounce)
- Stop-losses triggered

**Recommendation:**
- Add market regime filter (don't trade mean reversion in strong trends)
- Increase RSI oversold threshold during volatility (from 30 to 25)

---

## 🟡 Time-based Patterns

### Early Morning Low Activity (00:00-06:00)
**Trades:** 0-1 per hour
**Errors:** 0 per hour

**Analysis:** This is normal and expected
- Low crypto volatility during Asian early morning
- Strategy correctly avoids low-liquidity periods

✅ No action needed

---

### Lunch Hour Quiet (12:00-13:00)
**Trades:** 2 (vs normal 6)
**Errors:** 1 (vs normal 3-4)

**Analysis:**
Lower activity but proportionally fewer errors = system healthy

✅ No action needed

---

### Evening Volatility (18:00-21:00)
**Trades:** 15, 18, 12 (higher than average)
**Errors:** 8, 12, 7 (higher than average)

**Analysis:**
Increased activity correlates with US market open (crypto volatility increases)

**Error/Trade Ratio:**
- Evening: 0.47 errors per trade
- Daytime average: 0.32 errors per trade

**Cause:** High frequency trading during volatility → more failed orders (slippage, price movement)

**Recommendation:**
Accept this as normal, but monitor if ratio exceeds 0.5

---

## 📊 Correlations

### 1. Latency vs Error Rate
**Correlation:** +0.78 (strong positive)

**Finding:** When latency increases, error rate increases

**Specific example:**
- 18:45 - Latency spike to 450ms → 12 errors in next 15 minutes
- Normal periods: latency 60ms → 1-2 errors per 15 minutes

**Conclusion:** High latency directly causes order failures

---

### 2. Trading Activity vs PnL
**Correlation:** -0.42 (moderate negative) ⚠️

**Finding:** More trades = lower average PnL per trade

**Analysis:**
- Hours with 5-7 trades: avg PnL +$15
- Hours with 15+ trades: avg PnL +$6

**Hypothesis:** Over-trading during volatility
- Many signals during chaos
- Lower quality signals
- Worse execution (slippage)

**Recommendation:**
Add "max trades per hour" limit (e.g., 10) to filter low-quality signals

---

### 3. Position Size Variance
**Std deviation:** 0.023 (23% of mean position size)

**Finding:** High variance in position sizes

**Details:**
- Smallest position: 0.05 BTC
- Largest position: 0.18 BTC
- Mean: 0.10 BTC

**Analysis:**
This variance is **correct behavior** (risk-based position sizing)
- Tight stop-loss → larger position
- Wide stop-loss → smaller position

✅ No action needed

---

## 🚩 Red Flags

### 1. Degrading Performance Over Day
**Morning (06:00-12:00):**
- Win rate: 65%
- Avg PnL: +$18

**Afternoon (12:00-18:00):**
- Win rate: 55%
- Avg PnL: +$9

**Evening (18:00-24:00):**
- Win rate: 48%
- Avg PnL: +$3

**Trend:** Performance degrades throughout the day 🚨

**Possible causes:**
1. Market regime change (morning = range, evening = trend)
2. Indicator drift (indicators optimized for certain conditions)
3. Fatigue bug (memory leak? resource exhaustion?)

**Immediate investigation:**
- Check memory/CPU usage over time
- Review if strategy parameters need time-based adjustment
- Verify indicator calculations remain accurate

---

### 2. Inconsistent Stop-Loss Execution
**Expected:** Stop-loss triggers at exactly -3.0%
**Observed:**
- Average: -3.2%
- Range: -2.8% to -4.1%

**Worst case:** -4.1% (expected -3.0%) = 37% worse than planned!

**Analysis:**
Slippage and latency cause stop-loss to execute at worse prices

**Impact:**
- Expected max loss per trade: $90 (3% of $3000 position)
- Actual worst loss: $123 (4.1%)
- Extra loss: $33 per trade (when stop hit)

**Recommendation:**
- Account for slippage in stop-loss calculation
- Set stop at -2.7% to achieve actual -3.0% execution

---

## ✅ Summary & Action Items

**Critical (Fix Today):**
1. Investigate 23 trades spike at 14:00-15:00
2. Implement latency-based circuit breaker (>200ms)
3. Review position sizing for "Insufficient funds" errors

**High Priority (This Week):**
1. Add market regime filter for mean reversion strategy
2. Investigate performance degradation over day
3. Adjust stop-loss for slippage (set at -2.7% for -3.0% actual)

**Medium Priority (Next Week):**
1. Add "max trades per hour" limit (10)
2. Create real-time anomaly detection dashboard
3. Daily metrics comparison (today vs 7-day average)

**Low Priority (Backlog):**
1. Optimize for evening volatility periods
2. A/B test different RSI thresholds

---

**Overall System Health:** 🟡 Yellow (Functional but needs attention)

**Risk Level:** Medium (No immediate crisis, but degrading performance)
```

### Оценка

✅ **LLM отлично:**
- Выявил статистические аномалии (>2 std dev)
- Нашёл временные паттерны
- Обнаружил корреляции между метриками
- Предложил конкретные действия

⚠️ **Проверить:**
- Корреляции (нужно подтвердить статистически)
- Гипотезы о причинах

---

## Решение #3: Real-time мониторинг с алертами от LLM

### Задача

Не ждать конца дня для анализа. Обнаруживать проблемы **в реальном времени**.

### Архитектура

```
Trading Bot → Logs → Log Stream → LLM Analyzer → Alerts (Telegram/Email)
              (файл)  (tail -f)    (каждые 5 мин)   (если аномалия)
```

### Реализация

```python
# scripts/realtime_log_monitor.py
import time
import subprocess
from collections import deque
from openai import OpenAI

client = OpenAI()

class RealtimeLogMonitor:
    def __init__(self, log_file, check_interval=300):  # 5 minutes
        self.log_file = log_file
        self.check_interval = check_interval
        self.recent_lines = deque(maxlen=200)  # Last 200 lines

    def tail_logs(self):
        """Stream new log lines"""
        process = subprocess.Popen(
            ['tail', '-f', '-n', '0', self.log_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.recent_lines.append(line.strip())
                    yield line.strip()
        except KeyboardInterrupt:
            process.kill()

    def analyze_recent_logs(self):
        """Analyze last N lines with LLM"""
        logs_text = '\n'.join(self.recent_lines)

        prompt = f"""
You are a real-time trading system monitor.

Analyze these recent log lines (last 5 minutes):

{logs_text}

Quickly identify if there are any:
1. **Critical errors** (require immediate attention)
2. **Unusual patterns** (spike in errors, unusual trading activity)
3. **System degradation** (increasing latency, memory issues)

If everything is normal, respond: "NORMAL - no issues detected"

If there's an issue, respond in format:
ALERT: [CRITICAL/WARNING]
Issue: [brief description]
Impact: [potential impact]
Action: [what to do]
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300  # Quick response
        )

        return response.choices[0].message.content

    def send_alert(self, alert_message):
        """Send alert via Telegram"""
        import requests

        telegram_token = "YOUR_TELEGRAM_BOT_TOKEN"
        chat_id = "YOUR_CHAT_ID"

        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        requests.post(url, data={
            'chat_id': chat_id,
            'text': f"🚨 Trading Bot Alert 🚨\n\n{alert_message}",
            'parse_mode': 'Markdown'
        })

    def run(self):
        """Main monitoring loop"""
        print(f"Starting real-time log monitor...")
        print(f"Checking every {self.check_interval} seconds")

        last_check = time.time()

        for line in self.tail_logs():
            # Check every N seconds
            if time.time() - last_check >= self.check_interval:
                print("Analyzing recent logs...")

                analysis = self.analyze_recent_logs()

                if not analysis.startswith("NORMAL"):
                    print(f"\n⚠️  ALERT DETECTED:\n{analysis}\n")
                    self.send_alert(analysis)
                else:
                    print("✓ All normal")

                last_check = time.time()

# Usage
monitor = RealtimeLogMonitor('trading_bot.log', check_interval=300)
monitor.run()
```

### Пример алерта

```
🚨 Trading Bot Alert 🚨

ALERT: CRITICAL

Issue: Rapidly increasing error rate detected
- 15 errors in last 5 minutes
- All errors: "Order placement failed: Insufficient funds"
- Affecting multiple symbols: BTC/USDT, ETH/USDT, SOL/USDT

Impact:
- Strategy unable to open new positions
- Missing trading opportunities
- Potential capital allocation bug

Action:
1. Check RiskManager.calculate_position_size() immediately
2. Verify available balance matches expected
3. Consider pausing strategy until fixed

Detected at: 2025-03-20 14:35:12
```

**Результат**: получили алерт через 5 минут после начала проблемы (вместо обнаружения в конце дня).

---

## Решение #4: Контекстный поиск по логам

### Задача

"Почему сделка #12345 провалилась?" — нужно найти всё, что связано с этой сделкой.

### Традиционный подход

```bash
grep "#12345" trading_bot.log
```

**Проблема**: найдёт только строки с "#12345", но не контекст:
- Что было за 10 секунд до?
- Какие другие события произошли одновременно?
- Есть ли связь с другими ошибками?

### LLM-подход: Контекстный поиск

```python
def contextual_search_with_llm(log_file, query):
    """Search logs with context understanding"""

    # Extract relevant section (grep + context)
    result = subprocess.check_output(
        ['grep', '-A', '20', '-B', '20', query, log_file],
        text=True
    )

    prompt = f"""
User is investigating: "{query}"

Here are log lines containing this query with 20 lines before/after context:

{result}

Provide:
1. **What happened** (chronological summary)
2. **Root cause** (why did it fail/succeed)
3. **Related events** (other things happening at same time)
4. **Next steps** (where to look next)

Focus on causality, not just listing lines.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# Usage
analysis = contextual_search_with_llm('trading_bot.log', '#12345')
print(analysis)
```

### Пример результата

```markdown
# Investigation: Order #12345

## What Happened (Timeline)

**14:32:10.123** - Strategy generated BUY signal for ETH/USDT
- RSI: 28.4 (oversold)
- Price: $2,450.30
- Signal strength: HIGH

**14:32:10.234** - RiskManager calculated position size
- Available capital: $8,500
- Risk per trade: 1% = $85
- Stop-loss distance: 3% = $73.50
- Calculated size: 1.156 ETH
- Position value: $2,832.35

**14:32:10.345** - OrderManager attempted to place order
- Order ID: #12345
- Type: MARKET BUY
- Size: 1.156 ETH
- Estimated cost: $2,832.35

**14:32:10.456** ❌ - Exchange rejected order
- Error: "Insufficient funds"
- Required: $2,832.35 + commission ($2.83) = $2,835.18
- Available: $2,800.00

**Wait, discrepancy! RiskManager thought available capital was $8,500, but exchange says only $2,800!**

**14:32:10.567** - OrderManager retried (attempt 1)
- Same error

**14:32:10.678** - OrderManager retried (attempt 2)
- Same error

**14:32:10.789** - Max retries exceeded, order failed

---

## Root Cause

**Capital calculation mismatch between RiskManager and Exchange**

**Investigation of earlier logs reveals:**

**14:30:05.123** - Position opened: BTC/USDT
- Size: 0.12 BTC @ $50,000 = $6,000
- This locked $6,000 of capital

**14:32:10** (2 minutes later) - RiskManager still thinks capital is $8,500
- **BUG:** RiskManager didn't update available capital after BTC position opened!

**Actual available capital:**
- Total: $8,500
- Locked in BTC: $6,000
- Free: $2,500 (but exchange reported $2,800 due to small price increase in BTC)

**Root cause identified:** `RiskManager.get_available_capital()` doesn't subtract open positions!

---

## Related Events

**Same timestamp (14:32:10):**
- 3 other signals generated (SOL/USDT, AVAX/USDT, LINK/USDT)
- All failed with same "Insufficient funds" error
- Total missed opportunities: 4 trades

**Common factor:** All attempted during period when BTC position was open

---

## Next Steps

1. **Immediate fix:** Update `RiskManager.get_available_capital()`
   ```python
   def get_available_capital(self):
       total = self.broker.get_value()
       locked = sum(pos.value for pos in self.open_positions)
       return total - locked
   ```

2. **Verify fix:** Add unit test
   ```python
   def test_available_capital_excludes_open_positions():
       manager = RiskManager(capital=10000)
       manager.open_position('BTC', value=6000)
       assert manager.get_available_capital() == 4000
   ```

3. **Prevent recurrence:** Add assertion before placing order
   ```python
   assert order_value <= self.get_available_capital(), \
       f"Order value {order_value} exceeds available capital"
   ```

4. **Investigate:** Check if there are other places where capital calculation is wrong
   - Search codebase for `broker.get_value()` calls
   - Ensure all use updated `get_available_capital()` method

---

**Conclusion:** Bug found! RiskManager not accounting for locked capital in open positions.
```

### Оценка

✅ **LLM превосходно:**
- Построил хронологию событий
- Выявил противоречие в цифрах ($8,500 vs $2,800)
- Нашёл root cause через связь с более ранним событием (BTC position)
- Предложил конкретное исправление с кодом

**Экономия времени:** 30-60 минут debugging → 1 минута с LLM

---

## Best Practices

### 1. Структурированные логи

**Плохо:**
```
Error placing order
```

**Хорошо:**
```
[ERROR] OrderManager - Order placement failed | order_id=#12345 | symbol=ETH/USDT | error=InsufficientFunds | available=2800 | required=2835
```

LLM лучше парсит структурированные логи.

### 2. Контекстная информация

Включайте в логи:
- Timestamp с миллисекундами
- Модуль/компонент
- Уровень (ERROR/WARNING/INFO)
- Ключевые параметры (order_id, symbol, price, size)

### 3. Периодический анализ

Запускайте LLM-анализ:
- **Real-time:** каждые 5 минут (критические алерты)
- **Hourly:** каждый час (детальный анализ)
- **Daily:** конец дня (полный отчёт)

### 4. Сохраняйте инсайты

```python
# Save LLM analysis to database
def save_analysis(timestamp, analysis, alert_level):
    db.insert({
        'timestamp': timestamp,
        'analysis': analysis,
        'alert_level': alert_level,
        'status': 'new'
    })
```

Создайте базу знаний проблем и решений.

---

## Заключение

**LLM как анализатор логов даёт:**
✅ Автоматическая кластеризация ошибок
✅ Выявление аномалий и паттернов
✅ Real-time мониторинг с алертами
✅ Контекстный поиск (понимание причинно-следственных связей)
✅ Экономия 10-20 часов/неделю на анализе логов

**Ограничения:**
❌ Token limits (нельзя отправить 1GB логов за раз)
❌ Стоимость (OpenAI API ~ $0.03 за 1K tokens)
❌ Нужно проверять гипотезы (LLM может ошибаться в сложных случаях)

**ROI:** При стоимости ~$50-100/месяц на API экономия времени окупает затраты в 10-20 раз.

В следующей статье: **Автоматический разбор торговых дней с помощью ИИ** — ежедневный анализ того, где нарушается риск-менеджмент.
