---
layout: post
title: "Comparacion de LEAN, StockSharp y Backtrader desde la perspectiva de un desarrollador: arquitectura, rendimiento, MOEX"
description: "Pruebas detalladas de tres frameworks de algotrading. Benchmarks de rendimiento, complejidad de integracion con MOEX y ejemplos reales de codigo."
date: 2026-02-24
image: /assets/images/blog/frameworks_comparison.png
tags: [LEAN, StockSharp, Backtrader, comparison, frameworks, performance]
lang: es
---

"Que framework elegir para trading algoritmico?"

Los ultimos 6 meses probe tres plataformas: **LEAN** (QuantConnect), **StockSharp** y **Backtrader**. Escribi estrategias identicas en las tres. Medi velocidad de backtests. Conte el tiempo de integracion con MOEX.

## Tres plataformas, tres filosofias

**LEAN:** Motor profesional para fondos cuantitativos. Nucleo en C#, API Python, arquitectura event-driven.

**StockSharp:** Plataforma universal enfocada en rendimiento y mercado ruso. C#, 90+ conectores, procesamiento de ordenes en microsegundos.

**Backtrader:** Framework Python simple y flexible. Pero el desarrollo se detuvo en 2021.

## Benchmarks de rendimiento

3 anos de datos por hora (~18.000 velas):

| Framework | Tiempo de backtest | Velocidad (velas/seg) |
|-----------|-------------------|---------------------|
| Backtrader | 12 segundos | 1.500 |
| LEAN | 4 segundos | 4.500 |
| StockSharp | 3 segundos | 6.000 |

StockSharp y LEAN son **3-4x mas rapidos** que Backtrader.

## Integracion con MOEX

**StockSharp:** Soporte nativo de 90+ bolsas. Configuracion: 30 minutos, gratis.

**Backtrader:** Via bibliotecas de terceros. 30 min-1 hora.

**LEAN:** Sin soporte oficial. 2-3 dias de desarrollo personalizado.

## Tabla resumen

| Criterio | Backtrader | LEAN | StockSharp |
|----------|-----------|------|-----------|
| Para principiantes | 5/5 | 3/5 | 2/5 |
| Rendimiento backtest | 2/5 | 4/5 | 5/5 |
| Integracion MOEX | 4/5 | 2/5 | 5/5 |
| HFT | No | 3/5 | 5/5 |
| Integracion ML | 5/5 | 3/5 | 2/5 |

Principiantes: empiecen con **Backtrader**. Cuando necesiten velocidad o HFT, pasen a **StockSharp** o **LEAN**.

---

**Enlaces utiles:**

- [Backtrader](https://www.backtrader.com/)
- [LEAN](https://github.com/QuantConnect/Lean)
- [StockSharp](https://github.com/StockSharp/StockSharp)
