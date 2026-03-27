---
layout: post
title: "Robots de IA en el mercado real: que nos ensena Alpha Arena y otros benchmarks"
description: "El primer benchmark de traders IA con dinero real. Los modelos chinos vencieron a ChatGPT y Gemini. Analizamos resultados, estrategias ganadoras y que significa para el algotrading."
date: 2026-03-10
image: /assets/images/blog/ai_arena_benchmark.png
tags: [AI, LLM, Alpha Arena, benchmark, trading robots, DeepSeek, Qwen]
lang: es
---

Hace dos semanas [analice la arquitectura de robots open-source](/es/blog/2026/03/03/how-opensource-robot-works-inside.html). Logica clasica: indicadores, senales, if-then.

Hoy — sobre IA que toma decisiones de trading por si misma. Sin indicadores. Sin reglas. Solo: "aqui tienes $10.000, opera."

En octubre-noviembre de 2025 se celebro [Alpha Arena](https://nof1.ai/) — **el primer benchmark publico de traders IA con dinero real**.

Seis LLMs (ChatGPT, Claude, Gemini, Qwen 3 MAX, DeepSeek, Grok) recibieron $10.000 cada uno y operaron criptomonedas en [Hyperliquid DEX](https://hyperliquid.xyz/) durante dos semanas.

Los resultados fueron impactantes: **los modelos chinos aplastaron a los occidentales**. Qwen 3 MAX gano. ChatGPT y Gemini perdieron mas del 60% de su capital.

## Resultados

| Modelo | Capital final | Cambio | Max Drawdown | Operaciones | Sharpe |
|--------|--------------|--------|--------------|-------------|--------|
| **Qwen 3 MAX** | **$13.247** | **+32,5%** | -12% | 43 | 1,8 |
| DeepSeek | $12.891 | +28,9% | -15% | 67 | 1,5 |
| Claude | $11.204 | +12,0% | -18% | 89 | 0,9 |
| Grok | $9.687 | -3,1% | -22% | 124 | 0,2 |
| ChatGPT | $3.845 | **-61,6%** | -68% | 203 | -1,2 |
| Gemini | $3.412 | **-65,9%** | -71% | 187 | -1,4 |

## Por que ganaron los modelos chinos

**1. Disciplina:** Qwen hizo 43 operaciones, apalancamiento max 2x. ChatGPT hizo 203 operaciones, apalancamiento hasta 10x.

**2. Adaptacion a volatilidad:** DeepSeek redujo posiciones en periodos volatiles. Gemini ignoro la volatilidad.

**3. Datos de entrenamiento:** Entrenados en mercados chinos donde la alta volatilidad es la norma.

## Lecciones para algotraders

1. **La frecuencia de trading mata** — mas operaciones = peores resultados
2. **El apalancamiento amplifica errores** — si no esta probado, mantener <3x
3. **La adaptacion es mas importante que la optimizacion**
4. **El win rate esta sobrevalorado, el R/R esta infravalorado**
5. **Las comisiones son un gasto real**

## Que significa para el futuro

Los LLM deben usarse como herramientas (sentimiento, ideas, debugging), no como traders autonomos. Enfoque hibrido: combinar indicadores clasicos con contexto LLM. Los LLM chinos entran en escena: DeepSeek es open-source y 10x mas barato que ChatGPT.

---

**Enlaces utiles:**

- [Alpha Arena](https://nof1.ai/)
- [Season 1 Results](https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/)
- [Numerai](https://numer.ai/)
- [Quantiacs](https://quantiacs.com/)
