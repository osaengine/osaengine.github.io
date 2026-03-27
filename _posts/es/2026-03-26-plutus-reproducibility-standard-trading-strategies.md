---
layout: post
title: "PLUTUS: un nuevo estándar de reproducibilidad para estrategias de trading"
description: "El framework de código abierto PLUTUS estandariza la descripción y prueba de estrategias de trading algorítmico. Analizamos por qué esto importa."
date: 2026-03-26
image: /assets/images/blog/plutus-framework.png
tags: [open-source, algo trading, PLUTUS, standards]
lang: es
---

## El problema de la reproducibilidad

En el trading algorítmico existe un problema fundamental: cuando alguien publica una "estrategia rentable," reproducir los resultados es prácticamente imposible. Las razones:

- **Parámetros no especificados** -- el autor olvidó mencionar configuraciones clave
- **Datos diferentes** -- las fuentes de datos proporcionan precios ligeramente distintos
- **Supuestos ocultos** -- comisiones, slippage, tiempo de ejecución
- **Diferencias entre plataformas** -- el mismo algoritmo produce resultados diferentes en distintos motores de backtest

El framework **PLUTUS** fue creado para resolver este problema.

## Qué es PLUTUS

**PLUTUS** es un framework de código abierto para la descripción, prueba y publicación estandarizada de estrategias de trading.

Desarrollado por un grupo internacional de investigadores y publicado en [GitHub](https://github.com/algotrade-plutus) bajo licencia MIT.

## Arquitectura

PLUTUS define cuatro componentes estandarizados:

### 1. Strategy Specification (Especificación de la estrategia)

Descripción formal de la estrategia en formato YAML/JSON:

```yaml
strategy:
  name: "Mean Reversion RSI"
  version: "1.0"
  author: "researcher@university.edu"

  signals:
    entry_long:
      condition: "RSI(14) < 30 AND SMA(50) > SMA(200)"
    exit_long:
      condition: "RSI(14) > 70 OR stop_loss(-2%)"

  parameters:
    rsi_period: 14
    sma_fast: 50
    sma_slow: 200
    stop_loss_pct: -2.0

  universe:
    type: "equity"
    market: "US"
    filter: "S&P 500 constituents"

  execution:
    order_type: "market"
    slippage_model: "fixed_bps(5)"
    commission_model: "per_share(0.005)"
```

### 2. Data Specification (Especificación de datos)

Descripción estandarizada de los datos:

- Fuente (Yahoo Finance, Polygon, MOEX)
- Período (inicio, fin)
- Frecuencia (1 minuto, 1 hora, 1 día)
- Procesamiento (ajustado/sin ajustar, método de relleno)
- Hash de datos para verificación

### 3. Backtest Engine (Motor de backtest)

Motor de backtest estandarizado con:

- Lógica definida de procesamiento de órdenes
- Orden fijo de cálculos dentro de la barra
- Modelo de slippage transparente
- Informe con más de 50 métricas

### 4. Report Format (Formato de informe)

Formato de informe unificado que incluye:

- Curva de equidad
- Todas las métricas (Sharpe, Sortino, Max DD, Calmar, etc.)
- Distribución de operaciones
- Análisis por períodos temporales
- Resultados walk-forward

## Por qué esto importa

### Para investigadores

Publicar una estrategia en formato PLUTUS permite a otros investigadores **reproducir exactamente** los resultados. Es lo que el mundo científico tiene desde hace tiempo para los experimentos, pero que faltaba en el trading algorítmico.

### Para profesionales

El formato estandarizado simplifica:

- **Comparación de estrategias** -- todas las métricas se calculan igual
- **Auditoría** -- cada parámetro puede verificarse
- **Portabilidad** -- transferir una estrategia entre plataformas

### Para agentes IA

PLUTUS es especialmente útil para agentes LLM que generan estrategias de trading. El formato estandarizado permite:

- Validar automáticamente la especificación
- Ejecutar backtests sin configuración manual
- Comparar resultados con un benchmark

## Estado actual

- **Versión**: 0.8 (beta)
- **Lenguajes**: Python (principal), adaptadores para C# y Java
- **Mercados soportados**: EE.UU., UE, China, Cripto
- **Integraciones**: Backtrader, Zipline, VectorBT, QuantConnect

PLUTUS es un paso hacia un trading algorítmico más transparente y científico. Si desarrollas estrategias de trading, merece la pena prestarle atención.
