---
layout: post
title: "Автоматизация документации и ченджлогов торговых роботов с помощью ИИ"
description: "Практическое руководство по автоматизации документации торговых систем с помощью LLM: генерация README, API docs, changelogs, release notes. Реальные промпты, скрипты и GitHub Actions workflows."
date: 2026-05-26
image: /assets/images/blog/ai_documentation.png
tags: [LLM, ChatGPT, Claude, documentation, changelog, automation]
---

В [предыдущей статье]({{ site.baseurl }}/2026/05/19/generaciya-strategiy-s-llm.html) мы обсудили генерацию стратегий с LLM. Теперь перейдём к менее glamorous, но критически важной задаче: **документации**.

Документация — это то, что все ненавидят писать, но все хотят читать. LLM может автоматизировать 80% процесса. Разберём, как именно.

---

## Проблема: документация всегда устаревает

### Типичный сценарий

```python
# strategy.py - последнее изменение: 2025-12-15
class MyStrategy:
    def __init__(self, rsi_period=14, stop_loss=0.03):
        # Добавили новый параметр trailing_stop
        self.trailing_stop = 0.02
        ...
```

```markdown
# README.md - последнее изменение: 2025-10-01
## Parameters
- `rsi_period` (int): RSI period. Default: 14
- `stop_loss` (float): Stop-loss percentage. Default: 0.03
```

**Проблема**: README не содержит `trailing_stop` — **документация устарела**!

### Последствия

- Новые разработчики не знают о параметре
- Пользователи не используют функционал
- Тратится время на вопросы "как это работает?"
- Code review медленнее (надо читать сам код)

---

## Решение #1: Автоматическая генерация README из docstrings

### Шаг 1: Пишем хорошие docstrings

```python
class MeanReversionStrategy:
    """
    Mean Reversion Trading Strategy using Bollinger Bands

    This strategy enters long positions when price touches the lower Bollinger Band
    and exits when price returns to the middle band (mean).

    Parameters
    ----------
    bb_period : int, default=20
        Period for Bollinger Bands SMA calculation.
        Typical range: 10-50. Shorter = more signals, longer = smoother.

    bb_std : float, default=2.0
        Number of standard deviations for Bollinger Bands.
        Common values: 1.5 (tight), 2.0 (standard), 2.5 (wide)

    stop_loss_pct : float, default=0.03
        Stop-loss percentage below entry price.
        Example: 0.03 = 3% stop-loss

    risk_per_trade : float, default=0.01
        Fraction of capital to risk per trade.
        Example: 0.01 = risk 1% of portfolio

    trailing_stop : bool, default=False
        Enable trailing stop to protect profits.
        If True, stop-loss moves up as price increases.

    trailing_stop_pct : float, default=0.02
        Trailing stop distance as percentage.
        Only used if trailing_stop=True.

    Attributes
    ----------
    position : dict or None
        Current open position. None if no position.

    entry_price : float or None
        Price at which current position was entered.

    stop_price : float or None
        Current stop-loss price level.

    Examples
    --------
    Basic usage:

    >>> strategy = MeanReversionStrategy(bb_period=20, stop_loss_pct=0.03)
    >>> cerebro.addstrategy(strategy)
    >>> cerebro.run()

    With trailing stop:

    >>> strategy = MeanReversionStrategy(
    ...     bb_period=20,
    ...     trailing_stop=True,
    ...     trailing_stop_pct=0.02
    ... )

    Notes
    -----
    - Strategy works best in ranging markets
    - Requires minimum 200 bars of historical data
    - Commission should be set to 0.1% or actual exchange fees

    See Also
    --------
    TrendFollowingStrategy : Alternative for trending markets
    MLEnhancedStrategy : Machine learning version

    References
    ----------
    .. [1] Bollinger, J. (2001). Bollinger on Bollinger Bands.
           McGraw-Hill Education.
    """

    def __init__(self, bb_period=20, bb_std=2.0, stop_loss_pct=0.03,
                 risk_per_trade=0.01, trailing_stop=False, trailing_stop_pct=0.02):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stop_loss_pct = stop_loss_pct
        self.risk_per_trade = risk_per_trade
        self.trailing_stop = trailing_stop
        self.trailing_stop_pct = trailing_stop_pct

        self.position = None
        self.entry_price = None
        self.stop_price = None

    def calculate_position_size(self, capital, entry_price, stop_price):
        """
        Calculate position size based on risk management rules.

        Uses fixed fractional position sizing to risk a specific
        percentage of capital per trade.

        Parameters
        ----------
        capital : float
            Current total portfolio value in USD

        entry_price : float
            Intended entry price for the trade

        stop_price : float
            Stop-loss price level

        Returns
        -------
        float
            Position size (number of shares/contracts to buy)

        Examples
        --------
        >>> strategy = MeanReversionStrategy(risk_per_trade=0.02)
        >>> size = strategy.calculate_position_size(
        ...     capital=10000,
        ...     entry_price=100,
        ...     stop_price=97
        ... )
        >>> print(size)
        66.67  # Risk $200 (2% of $10k) with $3 stop = 66.67 shares

        Notes
        -----
        Position size is capped at 25% of capital to prevent
        over-concentration in single position.
        """
        risk_amount = capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_price)

        if stop_distance == 0:
            return 0

        size = risk_amount / stop_distance

        # Cap at 25% of capital
        max_size = (capital * 0.25) / entry_price
        size = min(size, max_size)

        return size
```

### Шаг 2: Автоматическая генерация README с LLM

```python
# scripts/generate_readme.py
import ast
import inspect
from pathlib import Path
import openai

def extract_docstrings(file_path):
    """Extract all docstrings from Python file"""
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())

    docstrings = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                docstrings[node.name] = docstring

    return docstrings

def generate_readme_with_llm(docstrings, project_name):
    """Generate README.md using LLM"""

    # Prepare context
    context = f"""
Project: {project_name}

Extracted docstrings from code:

{chr(10).join(f"### {name}:{chr(10)}{doc}{chr(10)}" for name, doc in docstrings.items())}
"""

    prompt = f"""
You are a technical writer specializing in trading systems documentation.

Given the docstrings from a trading strategy codebase, generate a comprehensive README.md file.

{context}

The README should include:

1. **Title and Brief Description** (1-2 sentences)
2. **Features** (bullet list of main capabilities)
3. **Installation** (how to install dependencies)
4. **Quick Start** (minimal example to get running)
5. **Parameters** (table with all parameters, their types, defaults, descriptions)
6. **Usage Examples** (2-3 realistic examples)
7. **API Reference** (brief description of main classes/methods)
8. **Performance Notes** (requirements, best practices)
9. **Contributing** (how to contribute)
10. **License**

Format in Markdown. Make it professional and comprehensive.
"""

    # Call LLM
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a technical documentation expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3  # Lower temperature for more consistent output
    )

    readme_content = response.choices[0].message.content
    return readme_content

def main():
    # Extract docstrings from all Python files
    strategy_files = Path('strategies').glob('*.py')
    all_docstrings = {}

    for file in strategy_files:
        docstrings = extract_docstrings(file)
        all_docstrings.update({f"{file.stem}.{k}": v for k, v in docstrings.items()})

    # Generate README
    readme = generate_readme_with_llm(all_docstrings, "Mean Reversion Trading Bot")

    # Save
    with open('README.md', 'w') as f:
        f.write(readme)

    print("✓ README.md generated successfully")

if __name__ == '__main__':
    main()
```

### Результат

```markdown
# Mean Reversion Trading Bot

A production-ready mean reversion trading strategy using Bollinger Bands for cryptocurrency and stock markets.

## Features

- 📊 Bollinger Bands-based mean reversion entries
- 🛡️ Comprehensive risk management (position sizing, stop-loss, trailing stop)
- 📈 Backtesting support with Backtrader framework
- 🔧 Highly configurable via parameters
- 📝 Extensive logging and trade tracking
- ✅ Tested on 3+ years of historical data

## Installation

```bash
pip install backtrader pandas numpy
```

For live trading (optional):
```bash
pip install ccxt  # For exchange connectivity
```

## Quick Start

```python
import backtrader as bt
from strategies.mean_reversion import MeanReversionStrategy

# Create Cerebro engine
cerebro = bt.Cerebro()

# Add strategy
cerebro.addstrategy(MeanReversionStrategy,
                    bb_period=20,
                    stop_loss_pct=0.03)

# Add data
data = bt.feeds.YahooFinanceData(dataname='BTC-USD',
                                  fromdate=datetime(2023, 1, 1),
                                  todate=datetime(2024, 12, 31))
cerebro.adddata(data)

# Set initial capital
cerebro.broker.setcash(10000)

# Run
cerebro.run()
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bb_period` | int | 20 | Period for Bollinger Bands SMA. Range: 10-50 |
| `bb_std` | float | 2.0 | Standard deviations for bands. Common: 1.5-2.5 |
| `stop_loss_pct` | float | 0.03 | Stop-loss percentage (e.g., 0.03 = 3%) |
| `risk_per_trade` | float | 0.01 | Capital risked per trade (e.g., 0.01 = 1%) |
| `trailing_stop` | bool | False | Enable trailing stop |
| `trailing_stop_pct` | float | 0.02 | Trailing stop distance percentage |

## Usage Examples

### Example 1: Conservative Settings

For low-risk, stable returns:

```python
strategy = MeanReversionStrategy(
    bb_period=30,        # Longer period = fewer signals
    bb_std=2.5,          # Wider bands = stricter entry
    stop_loss_pct=0.02,  # Tight stop-loss
    risk_per_trade=0.005 # Risk only 0.5% per trade
)
```

Expected: Low volatility, Sharpe ~1.0, fewer trades.

### Example 2: Aggressive Settings

For higher risk/reward:

```python
strategy = MeanReversionStrategy(
    bb_period=10,        # Shorter period = more signals
    bb_std=1.5,          # Tighter bands = more entries
    stop_loss_pct=0.05,  # Wider stop
    risk_per_trade=0.02, # Risk 2% per trade
    trailing_stop=True,  # Protect profits
    trailing_stop_pct=0.03
)
```

Expected: Higher volatility, potential for larger gains/losses.

### Example 3: Multiple Timeframes

Run strategy on different timeframes:

```python
# Daily timeframe
data_daily = bt.feeds.YahooFinanceData(dataname='BTC-USD', timeframe=bt.TimeFrame.Days)
cerebro.adddata(data_daily)
cerebro.addstrategy(MeanReversionStrategy, bb_period=20)

# 4-hour timeframe
data_4h = bt.feeds.YahooFinanceData(dataname='BTC-USD', timeframe=bt.TimeFrame.Minutes, compression=240)
cerebro.resampledata(data_4h)
cerebro.addstrategy(MeanReversionStrategy, bb_period=50)
```

## API Reference

### `MeanReversionStrategy`

Main strategy class implementing Bollinger Bands mean reversion.

**Key Methods:**

- `calculate_position_size(capital, entry_price, stop_price)`: Calculates position size using fixed fractional method
- `next()`: Main strategy logic executed on each bar
- `notify_order(order)`: Handles order execution notifications
- `stop()`: Called when strategy finishes (prints final statistics)

## Performance Notes

**Requirements:**
- Minimum 200 bars of historical data for indicator warmup
- Works best on liquid markets (BTC, ETH, major stocks)
- Recommended timeframes: 1h, 4h, 1d

**Best Practices:**
- Always set `commission` parameter to actual exchange fees (typically 0.001-0.002)
- Test with realistic slippage (0.05-0.1%)
- Use out-of-sample testing (30% of data held out)
- Monitor performance over multiple market regimes (bull, bear, range)

**Known Limitations:**
- Underperforms in strong trending markets
- Requires manual parameter tuning for different assets
- Not suitable for illiquid altcoins

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a Pull Request

## License

MIT License - see LICENSE file for details.

---

**Disclaimer:** This software is for educational purposes. Trading involves risk. Past performance does not guarantee future results.
```

### Оценка результата

✅ **Преимущества автогенерации:**
- README всегда актуален (генерируется из docstrings)
- Консистентный формат
- Экономия времени: 2 часа ручного написания → 5 минут автоматически
- Не забываются параметры (все извлекаются из кода)

⚠️ **Что нужно проверить:**
- Примеры кода (LLM может придумать несуществующие API)
- Установочные команды (pip packages)
- Ссылки на внешние ресурсы

---

## Решение #2: Автоматическая генерация Changelog

### Проблема

Вы делаете 10-20 коммитов в день. Через месяц нужно выпустить релиз и написать changelog. Но вы уже забыли, что именно менялось!

### Решение: GitHub Actions + LLM

```yaml
# .github/workflows/generate_changelog.yml
name: Generate Changelog

on:
  push:
    tags:
      - 'v*'  # Trigger on version tags (v1.0.0, v1.1.0, etc.)

jobs:
  changelog:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Fetch all history for git log

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install openai gitpython

      - name: Generate Changelog
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python scripts/generate_changelog.py

      - name: Commit Changelog
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add CHANGELOG.md
          git commit -m "docs: Update CHANGELOG for ${{ github.ref_name }}" || echo "No changes"
          git push
```

### Скрипт генерации Changelog

```python
# scripts/generate_changelog.py
import os
import subprocess
from datetime import datetime
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_commits_since_last_tag():
    """Get all commits since last version tag"""
    try:
        # Get previous tag
        prev_tag = subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0', 'HEAD^'],
            text=True
        ).strip()
    except:
        # No previous tag, get all commits
        prev_tag = None

    # Get commit log
    if prev_tag:
        cmd = ['git', 'log', f'{prev_tag}..HEAD', '--pretty=format:%H|||%s|||%b|||%an|||%ad', '--date=short']
    else:
        cmd = ['git', 'log', '--pretty=format:%H|||%s|||%b|||%an|||%ad', '--date=short']

    output = subprocess.check_output(cmd, text=True)

    commits = []
    for line in output.split('\n'):
        if not line:
            continue

        parts = line.split('|||')
        commits.append({
            'hash': parts[0][:7],
            'subject': parts[1],
            'body': parts[2],
            'author': parts[3],
            'date': parts[4]
        })

    return commits

def categorize_commits_with_llm(commits):
    """Use LLM to categorize commits into changelog sections"""

    commits_text = '\n'.join([
        f"- {c['hash']}: {c['subject']}" + (f"\n  {c['body']}" if c['body'] else "")
        for c in commits
    ])

    prompt = f"""
You are a technical writer creating a changelog for a trading bot software release.

Given these git commits, generate a well-organized CHANGELOG entry.

Commits:
{commits_text}

Categorize commits into these sections:
- **New Features** (feat: prefix or new functionality)
- **Bug Fixes** (fix: prefix or bug corrections)
- **Performance Improvements** (perf: prefix or optimizations)
- **Documentation** (docs: prefix)
- **Breaking Changes** (BREAKING: in commit message or major API changes)
- **Deprecated** (deprecated features)
- **Other Changes** (refactoring, chore, etc.)

For each section:
1. Group related commits
2. Use clear, user-friendly language (not technical git messages)
3. Mention impact on users
4. Include commit hashes in parentheses

Format in Markdown. If a section has no commits, omit it.

Example format:

## [1.2.0] - 2025-03-15

### New Features
- Added trailing stop functionality for better profit protection (#abc123, #def456)
- Support for multiple exchanges via CCXT integration (#ghi789)

### Bug Fixes
- Fixed position sizing calculation error that could cause over-leverage (#jkl012)
- Resolved race condition in order execution (#mno345)

### Performance Improvements
- Optimized indicator calculations (30% faster backtesting) (#pqr678)

Generate the changelog now:
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a technical writer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    changelog = response.choices[0].message.content
    return changelog

def update_changelog_file(new_entry):
    """Prepend new entry to CHANGELOG.md"""
    changelog_path = 'CHANGELOG.md'

    # Read existing changelog
    if os.path.exists(changelog_path):
        with open(changelog_path, 'r') as f:
            existing = f.read()
    else:
        existing = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"

    # Prepend new entry
    updated = existing.replace(
        "# Changelog\n\n",
        f"# Changelog\n\n{new_entry}\n\n"
    )

    # Write back
    with open(changelog_path, 'w') as f:
        f.write(updated)

def main():
    print("Fetching commits...")
    commits = get_commits_since_last_tag()

    if not commits:
        print("No new commits since last tag")
        return

    print(f"Found {len(commits)} commits")
    print("Generating changelog with LLM...")

    changelog_entry = categorize_commits_with_llm(commits)

    print("Updating CHANGELOG.md...")
    update_changelog_file(changelog_entry)

    print("✓ Changelog generated successfully")
    print("\n" + "="*60)
    print(changelog_entry)
    print("="*60)

if __name__ == '__main__':
    main()
```

### Пример сгенерированного Changelog

```markdown
# Changelog

## [1.3.0] - 2025-03-20

### New Features

- **Multi-timeframe Support**: Strategy can now run on multiple timeframes simultaneously (1h, 4h, 1d), allowing for better signal confirmation (#a3f2c1, #b7e4d2)
- **Telegram Notifications**: Real-time alerts for trade entries, exits, and errors sent directly to Telegram (#c9f1a3)
- **Dynamic Parameter Adjustment**: Strategy automatically adjusts Bollinger Band periods based on market volatility (#d2e8b4)

### Bug Fixes

- **Critical: Position Sizing Error**: Fixed calculation bug that could result in positions larger than intended, potentially causing over-leverage. All users should upgrade immediately (#e4f7c2)
- **Order Execution Race Condition**: Resolved issue where rapid price movements could cause duplicate orders (#f1a9d3)
- **Stop-Loss Not Triggering**: Fixed edge case where stop-loss wouldn't trigger during gap downs (#a7b3e1)

### Performance Improvements

- **40% Faster Backtesting**: Optimized indicator calculations using vectorized numpy operations instead of loops (#b2c4f6)
- **Reduced Memory Usage**: Historical data now uses rolling window (1000 bars) instead of keeping entire history, reducing RAM usage by 60% (#c8d2a5)

### Documentation

- Added comprehensive API documentation for all public methods (#d4e1b7)
- Updated README with performance benchmarks on different assets (#e9f2c3)
- Created troubleshooting guide for common setup issues (#f3a7d1)

### Breaking Changes

- **Parameter Rename**: `stop_loss` parameter renamed to `stop_loss_pct` for clarity. Update your configuration files. Migration: `stop_loss=0.03` → `stop_loss_pct=0.03` (#a1b5c9)
- **Minimum Python Version**: Now requires Python 3.9+ (previously 3.7+) due to new type hints (#b6c2d8)

### Deprecated

- `legacy_position_sizing()` method is deprecated and will be removed in v2.0. Use `calculate_position_size()` instead (#c3d9e4)

### Other Changes

- Refactored risk management module for better testability (#d7e2f1)
- Updated dependencies: `backtrader==1.9.78`, `pandas==2.1.0` (#e5f3a2)
- Improved error messages with actionable suggestions (#f2a6c7)

---

**Migration Guide**: See [MIGRATING.md](./MIGRATING.md) for detailed upgrade instructions from v1.2.x to v1.3.0.

**Full Diff**: [v1.2.0...v1.3.0](https://github.com/user/repo/compare/v1.2.0...v1.3.0)
```

### Оценка

✅ **Преимущества:**
- Автоматическая категоризация (LLM умнее простого regex по "feat:", "fix:")
- Понятный язык для пользователей (не сырые git messages)
- Выделение breaking changes
- Экономия времени: 1-2 часа → 5 минут

⚠️ **Проверить вручную:**
- Breaking changes (LLM может пропустить)
- Критичные баги (убедиться что выделены)
- Ссылки на issues/PRs (могут быть неверными)

---

## Решение #3: Генерация API Documentation

### Шаг 1: Docstrings в формате Sphinx/NumPy

```python
class TradingBot:
    """
    Main trading bot orchestrator.

    Manages strategy execution, order routing, and risk management
    across multiple markets and timeframes.

    Parameters
    ----------
    strategy : Strategy
        Trading strategy instance to execute
    exchanges : list of str
        Exchange IDs to connect to (e.g., ['binance', 'bybit'])
    config : dict, optional
        Configuration dictionary with keys:
        - 'max_positions': int, maximum concurrent positions
        - 'capital': float, initial capital in USD
        - 'risk_per_trade': float, fraction of capital to risk

    Attributes
    ----------
    is_running : bool
        Whether bot is currently running
    positions : dict
        Currently open positions, keyed by symbol
    orders : dict
        Pending orders, keyed by order ID

    Methods
    -------
    start()
        Start the trading bot
    stop()
        Stop the trading bot gracefully
    get_status()
        Get current bot status and statistics

    Examples
    --------
    >>> from strategies import MeanReversionStrategy
    >>> strategy = MeanReversionStrategy()
    >>> bot = TradingBot(strategy, exchanges=['binance'])
    >>> bot.start()

    See Also
    --------
    Strategy : Base class for trading strategies
    RiskManager : Risk management utilities

    Notes
    -----
    Bot runs in separate thread. Use `stop()` to gracefully shutdown.
    """

    def __init__(self, strategy, exchanges, config=None):
        self.strategy = strategy
        self.exchanges = exchanges
        self.config = config or {}

        self.is_running = False
        self.positions = {}
        self.orders = {}

    def start(self):
        """
        Start the trading bot.

        Connects to exchanges, initializes strategy, and begins
        processing market data.

        Raises
        ------
        ConnectionError
            If unable to connect to any exchange
        ValueError
            If strategy is not properly configured

        Examples
        --------
        >>> bot = TradingBot(strategy, ['binance'])
        >>> bot.start()
        Bot started successfully
        """
        if self.is_running:
            raise RuntimeError("Bot is already running")

        # Connect to exchanges
        for exchange_id in self.exchanges:
            self._connect_exchange(exchange_id)

        # Initialize strategy
        self.strategy.initialize()

        # Start main loop
        self.is_running = True
        self._main_loop()

    def stop(self):
        """
        Stop the trading bot gracefully.

        Closes all positions, cancels pending orders, and
        disconnects from exchanges.

        Returns
        -------
        dict
            Final statistics including:
            - 'total_pnl': float, total profit/loss in USD
            - 'num_trades': int, total number of trades
            - 'win_rate': float, percentage of winning trades

        Examples
        --------
        >>> stats = bot.stop()
        >>> print(f"Total PnL: ${stats['total_pnl']:.2f}")
        Total PnL: $1234.56
        """
        self.is_running = False

        # Close positions
        for symbol in list(self.positions.keys()):
            self._close_position(symbol)

        # Cancel orders
        for order_id in list(self.orders.keys()):
            self._cancel_order(order_id)

        # Calculate stats
        stats = self._calculate_statistics()

        return stats

    def get_status(self):
        """
        Get current bot status and statistics.

        Returns
        -------
        dict
            Status dictionary with keys:
            - 'is_running': bool
            - 'uptime': int, seconds since start
            - 'num_positions': int
            - 'num_pending_orders': int
            - 'current_pnl': float, unrealized PnL in USD
            - 'capital': float, current total capital

        Examples
        --------
        >>> status = bot.get_status()
        >>> print(f"Open positions: {status['num_positions']}")
        Open positions: 3
        >>> print(f"Current PnL: ${status['current_pnl']:.2f}")
        Current PnL: $234.56
        """
        return {
            'is_running': self.is_running,
            'uptime': self._get_uptime(),
            'num_positions': len(self.positions),
            'num_pending_orders': len(self.orders),
            'current_pnl': self._calculate_unrealized_pnl(),
            'capital': self._get_total_capital()
        }
```

### Шаг 2: Генерация HTML документации с Sphinx

```bash
# Установка
pip install sphinx sphinx-rtd-theme

# Инициализация
sphinx-quickstart docs

# Конфигурация docs/conf.py
extensions = [
    'sphinx.ext.autodoc',      # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',     # Support NumPy/Google style docstrings
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.autosummary',  # Generate summary tables
]

html_theme = 'sphinx_rtd_theme'
```

```rst
.. docs/index.rst

TradingBot Documentation
========================

.. automodule:: trading_bot
   :members:
   :undoc-members:
   :show-inheritance:

API Reference
=============

TradingBot
----------

.. autoclass:: TradingBot
   :members:
   :special-members: __init__

Strategy
--------

.. autoclass:: Strategy
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

```bash
# Build HTML docs
cd docs
make html

# Docs available at docs/_build/html/index.html
```

### Шаг 3: Улучшение документации с LLM

Даже хорошие docstrings можно улучшить. Используем LLM для генерации **user guide**:

```python
# scripts/generate_user_guide.py
import openai

def generate_user_guide_with_llm(api_docs):
    """Generate user-friendly guide from API documentation"""

    prompt = f"""
You are a technical writer creating a user guide for a trading bot.

Given this API documentation:

{api_docs}

Create a comprehensive USER GUIDE with:

1. **Getting Started** (installation, first run)
2. **Core Concepts** (what is a strategy, position, order)
3. **Common Use Cases** (with full code examples):
   - Running a simple backtest
   - Live trading with risk management
   - Multi-exchange arbitrage
   - Handling errors and recovery
4. **Best Practices** (dos and don'ts)
5. **Troubleshooting** (common errors and solutions)
6. **FAQ**

Make it beginner-friendly but technically accurate.
Use real code examples that actually work.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content
```

**Результат**: 20-30 страничный User Guide с примерами, который LLM сгенерировал за 2 минуты (вместо дня ручной работы).

---

## Решение #4: Release Notes для нетехнических пользователей

### Проблема

CHANGELOG технический. Нужны **Release Notes** для трейдеров, которые не понимают "refactored position sizing module".

### Решение: LLM переписывает Changelog в понятный язык

```python
# scripts/generate_release_notes.py
import openai

def generate_release_notes(changelog_entry, version):
    """Convert technical changelog to user-friendly release notes"""

    prompt = f"""
You are writing release notes for a trading bot used by cryptocurrency traders.

Your audience: traders with varying technical knowledge, from beginners to experts.

Given this technical CHANGELOG:

{changelog_entry}

Rewrite it as RELEASE NOTES that:

1. Start with a brief summary (2-3 sentences: what's new, why it matters)
2. Highlight **impact on users** (not technical details)
3. Use simple language (avoid jargon like "refactored", "optimized numpy operations")
4. Include **screenshots/examples** where helpful (describe what screenshot would show)
5. Add **migration instructions** if there are breaking changes
6. End with "What's next" (upcoming features)

Format for blog post / newsletter.

Version: {version}
Release Date: {datetime.now().strftime('%B %d, %Y')}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5  # Slightly higher for more engaging writing
    )

    return response.choices[0].message.content
```

### Пример результата

**Technical Changelog:**
```
## [1.3.0]
### New Features
- Added multi-timeframe support (#a3f2c1)
- Telegram notifications (#c9f1a3)
### Performance Improvements
- Optimized indicator calculations (30% faster) (#b2c4f6)
```

**LLM-Generated Release Notes:**
```markdown
# Trading Bot v1.3.0: Multi-Timeframe Analysis & Instant Alerts

**Released:** March 20, 2025

We're excited to announce version 1.3.0, our biggest update yet! This release brings two highly-requested features that will transform how you trade: **Multi-Timeframe Analysis** and **Telegram Alerts**.

## What's New

### 📊 Multi-Timeframe Confirmation (Game Changer!)

Ever wished you could confirm your 1-hour signals with the 4-hour or daily trend? Now you can!

**What this means for you:**
- **Better entries**: Don't get caught in false breakouts. Your bot now checks multiple timeframes before entering a trade.
- **Higher win rate**: Early testing shows 15-20% improvement in win rate when using 3-timeframe confirmation.
- **Same capital, smarter trades**: No need to split your capital across multiple bots.

**Example:**
```python
# Old way: Single timeframe
bot = TradingBot(timeframe='1h')

# New way: Multi-timeframe confirmation
bot = TradingBot(timeframes=['1h', '4h', '1d'])
# Bot only enters when ALL timeframes agree!
```

[Screenshot would show: Dashboard with 3 timeframe charts aligned, green checkmarks showing confirmation]

### 🔔 Telegram Notifications: Stay in the Loop

Get instant alerts on your phone when your bot:
- Opens a new position
- Closes a position (with profit/loss)
- Hits a stop-loss
- Encounters an error

**Setup in 30 seconds:**
1. Message @BotFather on Telegram
2. Create a bot, get your token
3. Add token to config: `telegram_token = "YOUR_TOKEN"`
4. Done! 🎉

[Screenshot would show: Sample Telegram message "🟢 Position opened: BTC/USDT Long @ $50,000 | Target: $51,500 | Stop: $49,000"]

### ⚡ 40% Faster Backtesting

Behind the scenes, we've completely rewritten how indicators are calculated. You won't notice any difference except:
- Backtests that took 5 minutes now finish in 3 minutes
- Your computer's fan won't sound like a jet engine anymore
- Same accuracy, just faster

## Breaking Changes (Important!)

### Parameter Rename

We've renamed `stop_loss` to `stop_loss_pct` to make it clearer that it's a percentage.

**What you need to do:**
```python
# Old (will show warning):
strategy = Strategy(stop_loss=0.03)

# New:
strategy = Strategy(stop_loss_pct=0.03)
```

Update your config files before upgrading. The old parameter will be removed in v2.0 (6 months from now).

### Обновление Python Required

Если вы используете старую версию Python, вам нужно обновиться.

**Почему?** Новые версии Python дают лучшие type hints, что означает меньше багов и более понятный код.

**How to upgrade:**
```bash
# Check current version
python --version

# If < 3.9, install latest
# Windows: download from python.org
# Mac: brew install python@3.11
# Linux: apt install python3.11
```

## What's Next

We're already working on v1.4.0, planned for May 2025:
- **Portfolio mode**: Run multiple strategies on one account
- **ML-powered entry timing**: Machine learning to optimize when to enter (not just signals)
- **Mobile app**: Monitor your bots from your phone (not just notifications)

## Upgrade Now

```bash
pip install --upgrade trading-bot

# Or from source:
git pull
pip install -e .
```

Questions? Join our Discord: discord.gg/trading-bot

Happy trading! 📈

— The TradingBot Team
```

### Оценка

✅ **Преимущества:**
- Понятно для нетехнических пользователей
- Фокус на пользе (не на технических деталях)
- Engaging writing style
- Включает screenshots descriptions
- Migration instructions

⚠️ **Проверить:**
- Цифры производительности ("40% faster") — LLM может преувеличить
- Обещания в "What's next" (не обещать того, что не планируется)

---

## Решение #5: Автоматическая генерация Docstrings

### Проблема

У вас 500 строк кода без docstrings. Писать вручную — неделя работы.

### Решение: LLM генерирует docstrings из кода

```python
# scripts/add_docstrings.py
import ast
import astor
import openai

def generate_docstring_for_function(func_code, func_name):
    """Generate NumPy-style docstring for function"""

    prompt = f"""
Generate a comprehensive NumPy-style docstring for this Python function:

```python
{func_code}
```

The docstring should include:
- Brief description (1 line)
- Detailed description (1-2 paragraphs)
- Parameters section with types and descriptions
- Returns section with type and description
- Examples section with 1-2 usage examples
- Notes section with important information
- Raises section if applicable

Follow NumPy docstring convention strictly.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content

def add_docstrings_to_file(file_path):
    """Add docstrings to all functions/classes in file"""

    with open(file_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    # Find functions/classes without docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if ast.get_docstring(node) is None:
                # Generate docstring
                func_code = astor.to_source(node)
                docstring = generate_docstring_for_function(func_code, node.name)

                # Insert docstring
                docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                node.body.insert(0, docstring_node)

    # Write back
    with open(file_path, 'w') as f:
        f.write(astor.to_source(tree))

    print(f"✓ Added docstrings to {file_path}")

# Usage
add_docstrings_to_file('strategy.py')
```

**Результат**: все функции получают подробные docstrings автоматически.

⚠️ **Важно**: обязательно проверьте сгенерированные docstrings вручную! LLM может неправильно понять логику.

---

## Best Practices: CI/CD Pipeline для документации

### Полный workflow

```yaml
# .github/workflows/docs.yml
name: Documentation Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  generate-docs:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install sphinx sphinx-rtd-theme openai gitpython

      - name: Check docstrings coverage
        run: |
          python scripts/check_docstring_coverage.py
          # Fails if coverage < 80%

      - name: Generate README
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python scripts/generate_readme.py

      - name: Build Sphinx docs
        run: |
          cd docs
          make html

      - name: Check for broken links
        run: |
          pip install linkchecker
          linkchecker docs/_build/html/

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build/html

  generate-changelog:
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Generate Changelog
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python scripts/generate_changelog.py

      - name: Generate Release Notes
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python scripts/generate_release_notes.py

      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          body_path: RELEASE_NOTES.md
          draft: false
          prerelease: false
```

---

## Заключение

**Автоматизация документации с LLM экономит:**
- README: 2 часа → 5 минут (**96% экономия**)
- Changelog: 1 час → 5 минут (**92% экономия**)
- API docs: 1 день → 1 час (**87% экономия**)
- Release notes: 2 часа → 10 минут (**92% экономия**)

**Общая экономия**: ~20-30 часов в месяц на крупном проекте.

**Важно:**
✅ Всегда проверяйте сгенерированный контент вручную
✅ Используйте низкую temperature (0.2-0.4) для технической документации
✅ Сохраняйте промпты в репозитории для воспроизводимости

В следующей статье: **LLM как анализатор логов** — автоматический поиск аномалий и повторяющихся ошибок в логах торговой системы.
