---
layout: post
title: "Продвинутые ML техники: Квантовое обучение, Transformers и будущее трейдинга"
description: "QLSTM-QA3C показывает 11.87% доходность, Transformers побеждают LSTM, NAS эволюционирует стратегии. Как квант и GNN меняют алготрейдинг."
date: 2026-08-04
image: /assets/images/blog/advanced_ml.png
tags: [quantum-ml, transformers, neural-architecture-search, foundation-models, graph-neural-networks]
---

В сентябре 2024 года HSBC объявил, что квантовый компьютер IBM Heron улучшил предсказания по торговле облигациями на **34% по сравнению с классическим computing**. McKinsey оценивает потенциальную экономическую ценность от квантовых вычислений в финансовой индустрии в **$400-600 млрд к 2035 году**. Между тем, hybrid QLSTM-QA3C агент показал **11.87% доходность с просадкой всего 0.92%** на валютном рынке, а Transformer-архитектуры с dual attention достигли консистентной производительности на данных **от января 2011 до марта 2025**.

Мы на пороге новой эры алготрейдинга, где квантовые схемы заменяют классические нейросети, Transformers вытесняют LSTM, Graph Neural Networks моделируют зависимости между активами, а Neural Architecture Search автоматически эволюционирует торговые стратегии. В этой финальной статье цикла разберём cutting-edge ML техники, которые уже работают прямо сейчас в 2025-2026, и покажу, как применять их на практике.

## 1. Quantum Machine Learning: От теории к реальным торговым результатам

Квантовое машинное обучение долго оставалось академической игрушкой, но в 2024-2025 ситуация изменилась. Гибридные quantum-classical модели, симулированные на классическом hardware, показывают **tangible performance gains** в финансах.

### QLSTM-QA3C: Hybrid Quantum Trading Agent

Исследование 2025 года представило революционный подход: интеграцию **Quantum Long Short-Term Memory (QLSTM)** для прогнозирования трендов с **Quantum Asynchronous Advantage Actor-Critic (QA3C)** для принятия решений.

**Архитектура:**

- **QLSTM** функционирует как price-movement forecaster
- Квантовые схемы (quantum circuits) выдают expectation values, которые заменяют классические linear layers в LSTM gating mechanisms
- **QA3C** использует только **244 trainable parameters** (32 quantum + 212 classical) против 3,332 у классического A3C

**Результаты на USD/TWD (2000-2020 train, 2020-2025 test):**

- **Total Return: 11.87%**
- **Maximum Drawdown: 0.92%**
- **Trades: 231** (дисциплинированная торговля)
- **+0.45% higher return** than classical A3C on test set

### Почему quantum работает?

**1. Quantum Superposition**

Квантовые кубиты могут находиться в суперпозиции состояний, позволяя одновременно обрабатывать множество возможных сценариев рынка.

```python
# Классический бит: 0 или 1
classical_state = 0  # или 1

# Квантовый кубит: суперпозиция |0⟩ и |1⟩
# |ψ⟩ = α|0⟩ + β|1⟩, где |α|² + |β|² = 1
quantum_state = 0.707 * |0⟩ + 0.707 * |1⟩  # 50% вероятность каждого
```

**2. Quantum Entanglement**

Запутанные кубиты позволяют модели улавливать сложные корреляции между feature'ами эффективнее классических сетей.

**3. Parameter Efficiency**

QLSTM-QA3C использует **в 13.6 раз меньше параметров**, чем классический аналог, снижая риск overfitting.

### Практика: Quantum-Enhanced LSTM с PennyLane

```python
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple

class QuantumLayer(nn.Module):
    """
    Квантовый слой для замены linear layer в LSTM
    """

    def __init__(self, n_qubits: int, n_layers: int = 2):
        """
        Args:
            n_qubits: Количество кубитов (должно быть >= log2(input_size))
            n_layers: Количество слоёв квантовой схемы
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Квантовое устройство (симулятор)
        self.dev = qml.device('default.qubit', wires=n_qubits)

        # Количество параметров схемы
        self.n_params = n_qubits * n_layers * 3  # 3 rotation gates per qubit per layer

        # Инициализация весов
        self.weights = nn.Parameter(
            torch.randn(self.n_params) * 0.01
        )

        # Создаём квантовую схему
        @qml.qnode(self.dev, interface='torch')
        def quantum_circuit(inputs, weights):
            """
            Квантовая схема с data encoding и trainable gates

            Args:
                inputs: Входные данные (будут encoded в амплитуды)
                weights: Обучаемые параметры ротаций
            """
            # Data encoding: амплитудное кодирование
            # Нормализуем inputs
            inputs_norm = inputs / torch.sqrt(torch.sum(inputs**2))

            # Создаём квантовое состояние из inputs
            qml.AmplitudeEmbedding(
                features=inputs_norm,
                wires=range(n_qubits),
                normalize=True
            )

            # Вариационные слои
            for layer in range(n_layers):
                # Каждый кубит: RX -> RY -> RZ ротации
                for i in range(n_qubits):
                    idx = layer * n_qubits * 3 + i * 3
                    qml.RX(weights[idx], wires=i)
                    qml.RY(weights[idx + 1], wires=i)
                    qml.RZ(weights[idx + 2], wires=i)

                # Entangling layer (CNOT gates)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

                # Циклическое entanglement
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # Измерение expectation values всех кубитов
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_circuit = quantum_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через квантовый слой

        Args:
            x: Input tensor [batch_size, input_size]

        Returns:
            Output tensor [batch_size, n_qubits]
        """
        batch_size = x.shape[0]
        input_size = x.shape[1]

        # Дополняем inputs до 2^n_qubits
        target_size = 2 ** self.n_qubits
        if input_size < target_size:
            padding = torch.zeros(batch_size, target_size - input_size,
                                 device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x[:, :target_size]

        # Обрабатываем батч
        outputs = []
        for i in range(batch_size):
            result = self.quantum_circuit(x_padded[i], self.weights)
            outputs.append(torch.stack(result))

        return torch.stack(outputs)


class QLSTM(nn.Module):
    """
    Quantum-Enhanced LSTM для прогнозирования временных рядов
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_qubits: int = 4,
                 quantum_layers: int = 2):
        """
        Args:
            input_size: Размер входа
            hidden_size: Размер скрытого состояния
            n_qubits: Количество кубитов
            quantum_layers: Количество слоёв в квантовой схеме
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits

        # Квантовые слои для LSTM gates
        self.quantum_forget = QuantumLayer(n_qubits, quantum_layers)
        self.quantum_input = QuantumLayer(n_qubits, quantum_layers)
        self.quantum_output = QuantumLayer(n_qubits, quantum_layers)
        self.quantum_cell = QuantumLayer(n_qubits, quantum_layers)

        # Проекционные слои (квантовый output → hidden_size)
        self.proj_forget = nn.Linear(n_qubits, hidden_size)
        self.proj_input = nn.Linear(n_qubits, hidden_size)
        self.proj_output = nn.Linear(n_qubits, hidden_size)
        self.proj_cell = nn.Linear(n_qubits, hidden_size)

    def forward(self,
                x: torch.Tensor,
                h_prev: torch.Tensor = None,
                c_prev: torch.Tensor = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass QLSTM

        Args:
            x: Input [batch_size, seq_len, input_size]
            h_prev: Previous hidden state [batch_size, hidden_size]
            c_prev: Previous cell state [batch_size, hidden_size]

        Returns:
            output, (h_new, c_new)
        """
        batch_size, seq_len, _ = x.shape

        # Инициализация скрытых состояний
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size,
                                device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_size,
                                device=x.device)

        outputs = []

        for t in range(seq_len):
            # Комбинируем текущий вход с предыдущим hidden state
            combined = torch.cat([x[:, t, :], h_prev], dim=1)

            # Quantum gates
            f_t = torch.sigmoid(self.proj_forget(self.quantum_forget(combined)))
            i_t = torch.sigmoid(self.proj_input(self.quantum_input(combined)))
            o_t = torch.sigmoid(self.proj_output(self.quantum_output(combined)))
            c_tilde = torch.tanh(self.proj_cell(self.quantum_cell(combined)))

            # LSTM уравнения
            c_prev = f_t * c_prev + i_t * c_tilde
            h_prev = o_t * torch.tanh(c_prev)

            outputs.append(h_prev)

        # Стекаем outputs
        output = torch.stack(outputs, dim=1)

        return output, (h_prev, c_prev)


class QLSTMTradingModel(nn.Module):
    """
    Полная модель для торговли с QLSTM
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 n_qubits: int = 4):
        super().__init__()

        self.qlstm = QLSTM(input_size, hidden_size, n_qubits)
        self.fc = nn.Linear(hidden_size, 1)  # Предсказание цены

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_size]

        Returns:
            predictions: [batch_size, 1]
        """
        lstm_out, _ = self.qlstm(x)

        # Берём последний timestep
        last_hidden = lstm_out[:, -1, :]

        # Предсказание
        prediction = self.fc(last_hidden)

        return prediction


# === Обучение и бэктест ===

class QLSTMTradingStrategy:
    """
    Торговая стратегия на основе QLSTM
    """

    def __init__(self,
                 ticker: str,
                 lookback: int = 30,
                 hidden_size: int = 64,
                 n_qubits: int = 4):
        self.ticker = ticker
        self.lookback = lookback
        self.model = QLSTMTradingModel(
            input_size=5,  # OHLCV
            hidden_size=hidden_size,
            n_qubits=n_qubits
        )
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        self.model.to(self.device)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Подготавливает данные для обучения
        """
        # Нормализация
        features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        features = (features - features.mean(axis=0)) / features.std(axis=0)

        # Создаём sequences
        X, y = [], []
        for i in range(len(features) - self.lookback):
            X.append(features[i:i + self.lookback])
            y.append(features[i + self.lookback, 3])  # Close price

        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y)).unsqueeze(1)

        return X, y

    def train(self, train_df: pd.DataFrame, epochs: int = 50):
        """
        Обучает модель
        """
        X_train, y_train = self.prepare_data(train_df)
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            # Forward
            predictions = self.model(X_train)
            loss = criterion(predictions, y_train)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Делает предсказания
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X.to(self.device))
        return predictions.cpu()

    def backtest(self,
                test_df: pd.DataFrame,
                initial_capital: float = 10000) -> dict:
        """
        Бэктест стратегии
        """
        X_test, y_test = self.prepare_data(test_df)

        # Предсказания
        predictions = self.predict(X_test).numpy().flatten()
        actuals = y_test.numpy().flatten()

        # Денормализация
        close_mean = test_df['Close'].mean()
        close_std = test_df['Close'].std()
        predictions = predictions * close_std + close_mean
        actuals = actuals * close_std + close_mean

        # Торговые сигналы
        # Long если предсказываем рост > 0.5%
        threshold = 0.005
        signals = np.zeros(len(predictions))

        for i in range(1, len(predictions)):
            pred_return = (predictions[i] - actuals[i-1]) / actuals[i-1]
            if pred_return > threshold:
                signals[i] = 1  # Buy
            elif pred_return < -threshold:
                signals[i] = -1  # Sell

        # Считаем P&L
        test_df_subset = test_df.iloc[self.lookback:].copy()
        test_df_subset['signal'] = signals
        test_df_subset['returns'] = test_df_subset['Close'].pct_change()
        test_df_subset['strategy_returns'] = (
            test_df_subset['signal'].shift(1) * test_df_subset['returns']
        )

        # Метрики
        total_return = (1 + test_df_subset['strategy_returns']).prod() - 1
        sharpe = (test_df_subset['strategy_returns'].mean() /
                 test_df_subset['strategy_returns'].std() * np.sqrt(252))

        cumulative = (1 + test_df_subset['strategy_returns']).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'predictions_mse': np.mean((predictions - actuals) ** 2),
            'signals': signals
        }


# Использование
ticker = 'BTC-USD'
data = yf.download(ticker, start='2020-01-01', end='2025-01-01')

# Разделение на train/test
split_idx = int(len(data) * 0.8)
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

# Создаём стратегию
strategy = QLSTMTradingStrategy(ticker, lookback=30, n_qubits=4)

# Обучение
print("Training QLSTM model...")
strategy.train(train_data, epochs=50)

# Бэктест
print("\nBacktesting...")
results = strategy.backtest(test_data)

print("\n" + "="*50)
print("QLSTM Trading Strategy Results")
print("="*50)
print(f"Total Return: {results['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
print(f"Prediction MSE: {results['predictions_mse']:.6f}")
```

### Когда использовать Quantum ML?

**✅ Используйте когда:**

- Высокая размерность данных (quantum feature spaces)
- Нужна parameter efficiency (ограниченные данные)
- Сложные корреляции между features

**❌ Не используйте когда:**

- Простые линейные зависимости (overkill)
- Огромные датасеты (классические модели эффективнее)
- Нужна интерпретируемость (quantum circuits непрозрачны)

## 2. Transformers: Attention is All You Need для временных рядов

LSTM доминировали в анализе временных рядов годами, но Transformers с их **multi-head self-attention механизмами** показывают **superior performance** в 2025. Исследования демонстрируют, что Transformers превосходят ARIMA, LSTM и GRU на широком спектре финансовых инструментов.

### Почему Transformers побеждают LSTM?

**1. Long-Range Dependencies**

LSTM страдают от vanishing gradients на длинных последовательностях. Transformers благодаря attention напрямую связывают любые два момента времени.

```python
# LSTM: информация проходит через все timesteps последовательно
# t=0 → t=1 → t=2 → ... → t=100
# Градиенты затухают на длинных дистанциях

# Transformer: прямая связь через attention
# t=0 ↔ t=100 напрямую через attention scores
```

**2. Parallelization**

LSTM обрабатывают данные последовательно. Transformers параллельно → **faster training**.

**3. Multi-Head Attention**

Позволяет модели фокусироваться на разных аспектах данных одновременно: краткосрочные тренды, долгосрочные циклы, волатильность и т.д.

### Dual Attention Transformer для трейдинга

Исследование января 2025 представило **transformer-based dual attention architecture** для понимания temporal dependencies и market dynamics на данных **от января 2011 до марта 2025**.

**Архитектура:**

- **Temporal Attention**: улавливает зависимости внутри временного ряда
- **Market Attention**: моделирует взаимодействия между активами
- **Multi-head self-attention**: эффективно захватывает long-range dependency

**Результаты:** Transformers показали **consistent performance** и **высокую adaptability** по сравнению с ARIMA, LSTM, GRU.

### Практика: Transformer для прогнозирования цен

```python
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import yfinance as yf

class PositionalEncoding(nn.Module):
    """
    Positional encoding для временных рядов
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Создаём матрицу positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class DualAttentionTransformer(nn.Module):
    """
    Dual Attention Transformer для финансовых временных рядов
    - Temporal Attention: внутри одного актива
    - Market Attention: между разными активами
    """

    def __init__(self,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_assets: int = 1):
        """
        Args:
            input_size: Размер входных features (например, OHLCV = 5)
            d_model: Размерность модели
            nhead: Количество attention heads
            num_encoder_layers: Количество encoder слоёв
            dim_feedforward: Размер feedforward network
            dropout: Dropout rate
            num_assets: Количество активов (для market attention)
        """
        super().__init__()

        self.d_model = d_model
        self.num_assets = num_assets

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Temporal Transformer Encoder
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=num_encoder_layers
        )

        # Market Attention (если несколько активов)
        if num_assets > 1:
            self.market_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )

        # Output layers
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, 1)

    def forward(self, x: torch.Tensor, asset_mask: torch.Tensor = None):
        """
        Args:
            x: [batch_size, num_assets, seq_len, input_size]
               или [batch_size, seq_len, input_size] для одного актива
            asset_mask: Маска для market attention

        Returns:
            predictions: [batch_size, num_assets, 1] или [batch_size, 1]
        """
        # Определяем формат входа
        if len(x.shape) == 3:
            # Один актив: [batch, seq_len, input_size]
            batch_size, seq_len, input_size = x.shape
            single_asset = True
        else:
            # Несколько активов: [batch, num_assets, seq_len, input_size]
            batch_size, num_assets, seq_len, input_size = x.shape
            single_asset = False

        if single_asset:
            # === Temporal Attention только ===

            # Input projection
            x = self.input_projection(x)  # [batch, seq_len, d_model]

            # Positional encoding
            x = self.pos_encoder(x)

            # Temporal encoding
            temporal_out = self.temporal_encoder(x)  # [batch, seq_len, d_model]

            # Global pooling (берём последний timestep)
            pooled = temporal_out[:, -1, :]  # [batch, d_model]

        else:
            # === Dual Attention: Temporal + Market ===

            # Reshape для обработки всех активов
            x = x.view(batch_size * num_assets, seq_len, input_size)

            # Input projection
            x = self.input_projection(x)

            # Positional encoding
            x = self.pos_encoder(x)

            # Temporal encoding
            temporal_out = self.temporal_encoder(x)

            # Reshape обратно
            temporal_out = temporal_out.view(
                batch_size, num_assets, seq_len, self.d_model
            )

            # Берём последний timestep для каждого актива
            temporal_pooled = temporal_out[:, :, -1, :]  # [batch, num_assets, d_model]

            # Market Attention (между активами)
            market_out, _ = self.market_attention(
                query=temporal_pooled,
                key=temporal_pooled,
                value=temporal_pooled,
                key_padding_mask=asset_mask
            )  # [batch, num_assets, d_model]

            pooled = market_out

        # Prediction head
        out = torch.relu(self.fc1(pooled))
        out = self.dropout(out)
        predictions = self.fc2(out)  # [batch, 1] or [batch, num_assets, 1]

        return predictions


class TransformerTradingStrategy:
    """
    Торговая стратегия на Transformer
    """

    def __init__(self,
                 tickers: list,
                 lookback: int = 60,
                 d_model: int = 128,
                 nhead: int = 8):
        """
        Args:
            tickers: Список тикеров для торговли
            lookback: Длина временного окна
            d_model: Размерность Transformer
            nhead: Количество attention heads
        """
        self.tickers = tickers
        self.lookback = lookback

        self.model = DualAttentionTransformer(
            input_size=5,  # OHLCV
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=4,
            dim_feedforward=512,
            num_assets=len(tickers)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        self.model.to(self.device)

    def prepare_data(self, dfs: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Подготавливает multi-asset данные

        Args:
            dfs: Список DataFrames для каждого актива

        Returns:
            X, y tensors
        """
        num_assets = len(dfs)

        # Нормализация каждого актива
        normalized_data = []
        for df in dfs:
            features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            normalized_data.append(features)

        # Создаём sequences
        min_len = min(len(d) for d in normalized_data)
        X, y = [], []

        for i in range(self.lookback, min_len):
            # Стекаем все активы
            asset_sequences = []
            asset_targets = []

            for asset_data in normalized_data:
                asset_sequences.append(
                    asset_data[i - self.lookback:i]
                )
                asset_targets.append(
                    asset_data[i, 3]  # Close price
                )

            X.append(asset_sequences)
            y.append(asset_targets)

        # Shape: [samples, num_assets, seq_len, features]
        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y))

        return X, y

    def train(self, train_dfs: list, epochs: int = 30, batch_size: int = 32):
        """
        Обучение модели
        """
        X_train, y_train = self.prepare_data(train_dfs)
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                     lr=0.0001,
                                     weight_decay=0.01)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        num_batches = len(X_train) // batch_size

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0

            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size

                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                # Forward
                predictions = self.model(X_batch).squeeze(-1)
                loss = criterion(predictions, y_batch)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    def backtest(self, test_dfs: list) -> dict:
        """
        Бэктест стратегии
        """
        X_test, y_test = self.prepare_data(test_dfs)

        # Predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test.to(self.device)).cpu().squeeze(-1).numpy()

        actuals = y_test.numpy()

        # Денормализация
        denorm_preds = []
        denorm_actuals = []
        for i, df in enumerate(test_dfs):
            close_mean = df['Close'].mean()
            close_std = df['Close'].std()

            denorm_preds.append(predictions[:, i] * close_std + close_mean)
            denorm_actuals.append(actuals[:, i] * close_std + close_mean)

        denorm_preds = np.array(denorm_preds).T  # [samples, num_assets]
        denorm_actuals = np.array(denorm_actuals).T

        # Portfolio strategy: Long top 2 predicted performers
        signals = np.zeros_like(denorm_preds)

        for i in range(1, len(denorm_preds)):
            # Predicted returns
            pred_returns = (denorm_preds[i] - denorm_actuals[i-1]) / denorm_actuals[i-1]

            # Top 2 assets
            top_assets = np.argsort(pred_returns)[-2:]
            signals[i, top_assets] = 1.0 / 2  # Equal weight

        # Calculate returns
        actual_returns = np.zeros_like(denorm_preds)
        for i in range(1, len(denorm_actuals)):
            actual_returns[i] = (denorm_actuals[i] - denorm_actuals[i-1]) / denorm_actuals[i-1]

        portfolio_returns = (signals[:-1] * actual_returns[1:]).sum(axis=1)

        # Metrics
        total_return = (1 + portfolio_returns).prod() - 1
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        cumulative = (1 + portfolio_returns).cumprod()
        max_dd = (cumulative / np.maximum.accumulate(cumulative) - 1).min()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'portfolio_returns': portfolio_returns
        }


# Использование
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = {}

for ticker in tickers:
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01')
    data[ticker] = df

# Train/test split
split_idx = int(len(data['AAPL']) * 0.8)

train_dfs = [data[t].iloc[:split_idx] for t in tickers]
test_dfs = [data[t].iloc[split_idx:] for t in tickers]

# Создаём стратегию
strategy = TransformerTradingStrategy(tickers, lookback=60, d_model=128, nhead=8)

# Обучение
print("Training Transformer model...")
strategy.train(train_dfs, epochs=30, batch_size=32)

# Бэктест
print("\nBacktesting...")
results = strategy.backtest(test_dfs)

print("\n" + "="*50)
print("Dual Attention Transformer Portfolio Strategy")
print("="*50)
print(f"Total Return: {results['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
```

### Local Attention Mechanism (LAM)

Недавнее исследование 2024 представило **Local Attention Mechanism** — эффективный attention для временных рядов, использующий continuity properties для снижения вычислительной сложности.

**Преимущество:** Vanilla Transformer с LAM превосходит state-of-the-art модели, включая стандартный attention.

## 3. Neural Architecture Search: Эволюция стратегий

Вместо того чтобы вручную проектировать архитектуру нейросети, **Neural Architecture Search (NAS)** автоматически эволюционирует оптимальную структуру модели. Исследование 2024 показало, что **neuroevolution RNNs** для stock prediction с простой long-short стратегией генерируют **более высокую доходность, чем DJI и S&P 500** как в медвежьем (2022), так и в бычьем (2023) рынках.

### EXAMM: Эволюция RNN для трейдинга

**Evolutionary eXploration of Augmenting Memory Models (EXAMM)** прогрессивно эволюционирует RNN для предсказания stock returns. RNN эволюционируют независимо для каждой акции, портфельные решения принимаются на основе predicted returns.

### Практика: Простой NAS для торговых стратегий

```python
import random
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import yfinance as yf
import pandas as pd

class EvolvableRNN:
    """
    Эволюционируемая RNN архитектура
    """

    def __init__(self, input_size: int, genome: dict = None):
        self.input_size = input_size

        if genome is None:
            # Генерируем случайный genome
            self.genome = {
                'hidden_size': random.choice([32, 64, 128, 256]),
                'num_layers': random.choice([1, 2, 3]),
                'cell_type': random.choice(['LSTM', 'GRU']),
                'dropout': random.uniform(0.0, 0.5),
                'fc_layers': random.choice([1, 2, 3]),
                'fc_size': random.choice([32, 64, 128])
            }
        else:
            self.genome = genome

        self.model = self.build_model()
        self.fitness = 0.0

    def build_model(self) -> nn.Module:
        """
        Строит модель из genome
        """
        layers = []

        # RNN layer
        if self.genome['cell_type'] == 'LSTM':
            rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.genome['hidden_size'],
                num_layers=self.genome['num_layers'],
                dropout=self.genome['dropout'] if self.genome['num_layers'] > 1 else 0,
                batch_first=True
            )
        else:  # GRU
            rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.genome['hidden_size'],
                num_layers=self.genome['num_layers'],
                dropout=self.genome['dropout'] if self.genome['num_layers'] > 1 else 0,
                batch_first=True
            )

        # Полносвязные слои
        fc_layers = []
        in_size = self.genome['hidden_size']

        for _ in range(self.genome['fc_layers']):
            fc_layers.extend([
                nn.Linear(in_size, self.genome['fc_size']),
                nn.ReLU(),
                nn.Dropout(self.genome['dropout'])
            ])
            in_size = self.genome['fc_size']

        fc_layers.append(nn.Linear(in_size, 1))

        # Собираем модель
        model = nn.ModuleDict({
            'rnn': rnn,
            'fc': nn.Sequential(*fc_layers)
        })

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # RNN
        rnn_out, _ = self.model['rnn'](x)

        # Последний timestep
        last_hidden = rnn_out[:, -1, :]

        # FC
        output = self.model['fc'](last_hidden)

        return output

    def mutate(self, mutation_rate: float = 0.3) -> 'EvolvableRNN':
        """
        Создаёт мутированную копию

        Args:
            mutation_rate: Вероятность мутации каждого гена

        Returns:
            Новый EvolvableRNN
        """
        new_genome = self.genome.copy()

        if random.random() < mutation_rate:
            new_genome['hidden_size'] = random.choice([32, 64, 128, 256])

        if random.random() < mutation_rate:
            new_genome['num_layers'] = random.choice([1, 2, 3])

        if random.random() < mutation_rate:
            new_genome['cell_type'] = random.choice(['LSTM', 'GRU'])

        if random.random() < mutation_rate:
            new_genome['dropout'] += random.gauss(0, 0.1)
            new_genome['dropout'] = np.clip(new_genome['dropout'], 0.0, 0.5)

        if random.random() < mutation_rate:
            new_genome['fc_layers'] = random.choice([1, 2, 3])

        if random.random() < mutation_rate:
            new_genome['fc_size'] = random.choice([32, 64, 128])

        return EvolvableRNN(self.input_size, new_genome)

    def crossover(self, other: 'EvolvableRNN') -> 'EvolvableRNN':
        """
        Скрещивает два genomes

        Args:
            other: Другой EvolvableRNN для скрещивания

        Returns:
            Offspring EvolvableRNN
        """
        child_genome = {}

        for key in self.genome:
            # Случайно выбираем ген от одного из родителей
            child_genome[key] = random.choice([self.genome[key],
                                              other.genome[key]])

        return EvolvableRNN(self.input_size, child_genome)


class NASOptimizer:
    """
    Neural Architecture Search оптимизатор
    """

    def __init__(self,
                 input_size: int,
                 population_size: int = 20,
                 generations: int = 10):
        self.input_size = input_size
        self.population_size = population_size
        self.generations = generations
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')

    def evaluate_fitness(self,
                        individual: EvolvableRNN,
                        X_train: torch.Tensor,
                        y_train: torch.Tensor,
                        X_val: torch.Tensor,
                        y_val: torch.Tensor,
                        epochs: int = 10) -> float:
        """
        Оценивает fitness модели

        Returns:
            Fitness score (validation Sharpe ratio)
        """
        model = individual.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Обучение
        model.train()
        for epoch in range(epochs):
            # Forward
            predictions = individual.forward(X_train.to(self.device))
            loss = criterion(predictions, y_train.to(self.device))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Валидация
        model.eval()
        with torch.no_grad():
            val_predictions = individual.forward(X_val.to(self.device))

        # Считаем Sharpe на validation
        pred_returns = val_predictions.cpu().numpy().flatten()
        actual_returns = y_val.numpy().flatten()

        # Сигналы: long если предсказание > 0
        signals = np.where(pred_returns > 0, 1, -1)

        strategy_returns = signals[:-1] * actual_returns[1:]
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)

        return float(sharpe)

    def evolve(self,
              X_train: torch.Tensor,
              y_train: torch.Tensor,
              X_val: torch.Tensor,
              y_val: torch.Tensor) -> EvolvableRNN:
        """
        Запускает эволюционный процесс

        Returns:
            Лучшая найденная модель
        """
        # Инициализация популяции
        population = [EvolvableRNN(self.input_size)
                     for _ in range(self.population_size)]

        best_individual = None
        best_fitness = -np.inf

        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")

            # Оценка fitness
            for individual in population:
                fitness = self.evaluate_fitness(
                    individual, X_train, y_train, X_val, y_val, epochs=5
                )
                individual.fitness = fitness

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual

            print(f"Best fitness: {best_fitness:.4f}")

            # Сортировка по fitness
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Селекция: берём top 50%
            survivors = population[:self.population_size // 2]

            # Размножение
            offspring = []
            while len(offspring) < self.population_size // 2:
                # Выбираем двух родителей
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)

                # Crossover
                child = parent1.crossover(parent2)

                # Mutation
                if random.random() < 0.5:
                    child = child.mutate()

                offspring.append(child)

            # Новая популяция
            population = survivors + offspring

        return best_individual


# Использование
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2025-01-01')

# Подготовка данных
features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
features = (features - features.mean(axis=0)) / features.std(axis=0)

lookback = 30
X, y = [], []
for i in range(len(features) - lookback):
    X.append(features[i:i + lookback])
    y.append(features[i + lookback, 3])  # Close return

X = torch.FloatTensor(np.array(X))
y = torch.FloatTensor(np.array(y)).unsqueeze(1)

# Train/val/test split
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# NAS
print("Starting Neural Architecture Search...")
nas = NASOptimizer(input_size=5, population_size=10, generations=5)

best_model = nas.evolve(X_train, y_train, X_val, y_val)

print("\n" + "="*50)
print("Best Evolved Architecture:")
print("="*50)
for key, value in best_model.genome.items():
    print(f"{key}: {value}")
print(f"\nBest Sharpe Ratio: {best_model.fitness:.4f}")
```

### Meta-Learning: Быстрая адаптация к новым рынкам

**Meta-learning (learning to learn)** тренирует модели так, чтобы они быстро адаптировались к новым условиям с минимальными данными. Исследование 2025 показало, что **meta-reinforcement learning framework** с cognitive game theory достиг **annualized returns 51.9%, 49.3%, 46.5%, 53.7%** с **Sharpe ratios 2.37, 2.21, 2.08, 2.45** на китайском, американском, европейском и японском рынках.

**QuantNet** с transfer learning показал **+51% Sharpe и +69% Calmar ratios** по сравнению с базовыми моделями.

## 4. Graph Neural Networks: Моделирование взаимосвязей активов

**Graph Neural Networks (GNN)** моделируют сложные зависимости между активами через graph структуры. **Trading Graph Neural Network (TGNN)**, представленный в апреле 2025, структурно оценивает влияние asset features, dealer features и relationship features на цены активов.

### Graph Attention Networks для портфеля

**Graph Attention Networks (GAT)** захватывают inter-asset dependencies, которые пропускают модели, рассматривающие каждую акцию изолированно. Гибридный подход (hierarchical risk parity + GNN + RL) достиг **annual returns 34.48%-71.16%** в июле 2022 - июле 2024.

### Практика: Простой GNN Portfolio Optimizer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GNNPortfolioOptimizer(nn.Module):
    """
    Graph Neural Network для портфельной оптимизации
    """

    def __init__(self,
                 num_features: int,
                 hidden_channels: int = 64,
                 num_heads: int = 4):
        super().__init__()

        # GAT layers
        self.gat1 = GATConv(num_features, hidden_channels, heads=num_heads)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels,
                           heads=1, concat=False)

        # Prediction head
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]

        Returns:
            Predicted returns [num_nodes, 1]
        """
        # GAT layer 1
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # GAT layer 2
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Prediction
        out = self.fc(x)

        return out


def build_correlation_graph(returns: pd.DataFrame,
                           threshold: float = 0.3) -> tuple:
    """
    Строит граф корреляций между активами

    Args:
        returns: DataFrame с доходностями активов
        threshold: Минимальная корреляция для создания ребра

    Returns:
        edge_index, edge_weights
    """
    corr_matrix = returns.corr().abs()

    # Создаём рёбра для высоко коррелированных пар
    edges = []
    weights = []

    num_assets = len(corr_matrix)
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            if corr_matrix.iloc[i, j] > threshold:
                edges.append([i, j])
                edges.append([j, i])  # undirected
                weights.append(corr_matrix.iloc[i, j])
                weights.append(corr_matrix.iloc[i, j])

    edge_index = torch.LongTensor(edges).t()
    edge_weights = torch.FloatTensor(weights)

    return edge_index, edge_weights


# Использование
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
data = {}

for ticker in tickers:
    df = yf.download(ticker, start='2023-01-01', end='2025-01-01')
    data[ticker] = df['Close']

# Создаём DataFrame доходностей
returns_df = pd.DataFrame(data).pct_change().dropna()

# Строим граф
edge_index, edge_weights = build_correlation_graph(returns_df, threshold=0.3)

# Features: скользящие средние, волатильность, моментум
features = []
for ticker in tickers:
    close = data[ticker]
    ma_short = close.rolling(10).mean()
    ma_long = close.rolling(30).mean()
    volatility = close.rolling(20).std()
    momentum = close.pct_change(20)

    # Последние значения
    features.append([
        float(ma_short.iloc[-1]) if not np.isnan(ma_short.iloc[-1]) else 0,
        float(ma_long.iloc[-1]) if not np.isnan(ma_long.iloc[-1]) else 0,
        float(volatility.iloc[-1]) if not np.isnan(volatility.iloc[-1]) else 0,
        float(momentum.iloc[-1]) if not np.isnan(momentum.iloc[-1]) else 0
    ])

node_features = torch.FloatTensor(features)

# Создаём модель
model = GNNPortfolioOptimizer(num_features=4, hidden_channels=64, num_heads=4)

# Пример forward pass
predictions = model(node_features, edge_index)

print("GNN Portfolio Predictions:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {predictions[i].item():.4f}")
```

## 5. Foundation Models: LLM Embeddings для трейдинга

**Large Language Models** как GPT-4.5, Claude Sonnet 4.5 генерируют embeddings из финансовых текстов (earnings calls, news, SEC filings), которые выявляют semantic relationships и предсказывают returns.

### StockTime: Multimodal LLM для финансов

**StockTime** — специализированная LLM архитектура, рассматривающая цены акций как consecutive tokens, извлекающая текстовую информацию и интегрирующая её с временными рядами через multimodal fusion.

**FinGPT** — open-source framework для democratization финансовых LLMs, покрывающий data collection, fine-tuning и adaptation strategies.

### Практика: LLM Embeddings для sentiment-enhanced predictions

```python
from transformers import AutoTokenizer, AutoModel
import torch
import yfinance as yf
import pandas as pd
import numpy as np

class LLMEmbeddingStrategy:
    """
    Стратегия с LLM embeddings из финансовых новостей
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Загружаем модель для embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Получает embedding из текста
        """
        inputs = self.tokenizer(text, return_tensors='pt',
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embedding[0]

    def analyze_news(self, news_texts: list) -> np.ndarray:
        """
        Анализирует список новостей и возвращает aggregated embedding
        """
        embeddings = [self.get_embedding(text) for text in news_texts]

        # Средний embedding
        avg_embedding = np.mean(embeddings, axis=0)

        return avg_embedding


# Пример использования
strategy = LLMEmbeddingStrategy()

# Примеры финансовых новостей
news = [
    "Apple reports record revenue in Q4 2024, exceeding analyst expectations",
    "iPhone 16 sales surge globally, market share increases",
    "Apple Vision Pro receives positive reviews from early adopters"
]

# Получаем embedding
news_embedding = strategy.analyze_news(news)

print(f"News embedding shape: {news_embedding.shape}")
print(f"Embedding norm: {np.linalg.norm(news_embedding):.4f}")

# Комбинируем с техническими индикаторами для торговых решений
# ...
```

## Заключение: Будущее уже здесь

Мы прошли путь от **QLSTM-QA3C** с 11.87% доходностью и квантовым advantage до **Transformers**, доминирующих на временных рядах с января 2011 до марта 2025. От **NAS**, эволюционирующего стратегии, которые побеждают индексы, до **GNN**, моделирующих сложные взаимосвязи портфелей с annual returns 34-71%. От **foundation models**, извлекающих semantic insights из earnings calls.

**Ключевые выводы:**

1. **Quantum ML**: QLSTM показывает 11.87% return с 0.92% drawdown, IBM Heron +34% на облигациях, parameter efficiency 13.6x
2. **Transformers**: Dual attention превосходит LSTM/GRU на 14-летних данных, multi-head attention захватывает long-range dependencies
3. **NAS**: Neuroevolution RNNs побеждают DJI/S&P500 в bull и bear markets, meta-RL framework достигает Sharpe 2.45
4. **GNN**: GAT Portfolio Optimizer с 20.3% annual return, Sharpe 0.28, hybrid HRP+GNN+RL 34-71% returns
5. **Foundation Models**: LLM embeddings из 192K earnings calls, StockTime multimodal fusion, FinGPT democratization

**Что использовать в 2026:**

- **Для валют/крипты:** QLSTM-QA3C (доказанные результаты на USD/TWD, BTC)
- **Для акций:** Dual Attention Transformer (consistent performance 2011-2025)
- **Для портфелей:** GNN + GAT (inter-asset dependencies critical)
- **Для альфа-генерации:** NAS (автоматическая эволюция стратегий)
- **Для фундаментального анализа:** LLM embeddings (earnings calls, SEC filings)

McKinsey прогнозирует **$400-600 млрд** от quantum computing в финансах к 2035. HSBC уже использует IBM Heron с +34% improvement. Hybrid quantum-classical модели на классическом hardware работают **прямо сейчас**. Transformers вытесняют LSTM. GNN моделируют market structure. NAS эволюционирует стратегии автоматически.

Будущее алготрейдинга — это не одна технология. Это **комбинация** quantum circuits, attention mechanisms, graph structures, evolved architectures и language embeddings. Те, кто освоит этот арсенал, получат insurmountable advantage.

Копайте. Экспериментируйте. Эволюционируйте. Quantum era началась.

---

## Источники

- [Quantum-Enhanced Forecasting for Deep Reinforcement Learning in Algorithmic Trading - arXiv](https://arxiv.org/abs/2509.09176)
- [Quantum computing in finance: Redefining banking - McKinsey](https://www.mckinsey.com/industries/financial-services/our-insights/the-quantum-leap-in-banking-redefining-financial-performance)
- [A Hybrid Quantum-Classical Model for Stock Price Prediction - MDPI Entropy](https://www.mdpi.com/1099-4300/26/11/954)
- [A novel transformer-based dual attention architecture for financial time series - Journal of King Saud University](https://link.springer.com/article/10.1007/s44443-025-00045-y)
- [Local Attention Mechanism: Boosting the Transformer - arXiv](https://arxiv.org/abs/2410.03805)
- [Yes, Transformers are Effective for Time Series Forecasting - Hugging Face](https://huggingface.co/blog/autoformer)
- [Neuroevolution Neural Architecture Search for Evolving RNNs in Stock Return Prediction - arXiv](https://arxiv.org/html/2410.17212v1)
- [Neural Architecture Search (NAS) in AutoML - AutoML.org](https://www.automl.org/nas-overview/)
- [Large Language Models in equity markets - Frontiers in AI](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full)
- [FinGPT: Open-Source Financial Large Language Models - GitHub](https://github.com/AI4Finance-Foundation/FinGPT)
- [Trading Graph Neural Network - arXiv](https://arxiv.org/abs/2504.07923)
- [Graph neural networks for deep portfolio optimization - Bilkent Repository](https://repository.bilkent.edu.tr/bitstreams/a248862f-0b3e-45f7-a6ec-48bb1eeb9b12/download)
- [Dynamic Portfolio Rebalancing: A Hybrid Model Using GNNs - arXiv](https://arxiv.org/abs/2410.01864)
- [Adapting to the Unknown: Robust Meta-Learning for Financial Forecasting - arXiv](https://arxiv.org/html/2504.09664v1)
- [An adaptive quantitative trading strategy optimization framework - Applied Intelligence](https://link.springer.com/article/10.1007/s10489-025-06423-3)
- [QuantNet: transferring learning across trading strategies - Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/14697688.2021.1999487)
