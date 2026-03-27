---
layout: post
title: "ChatGPT genera la idea, el constructor arma el robot: futuro del algotrading o hype temporal?"
description: "Pase un mes probando el combo IA + constructores visuales. Genere estrategias con ChatGPT/Claude, las arme en TSLab. Esto es lo que paso."
date: 2026-01-13
image: /assets/images/blog/ai_visual_builders.png
tags: [AI, ChatGPT, Claude, builders, automation]
lang: es
---

"Describeme una estrategia basada en cruce de EMA con filtro RSI."

ChatGPT entrega la logica en 10 segundos. Abro TSLab, armo los bloques. En 15 minutos — un robot listo.

Suena como un sueno. Pero funciona en la practica?

El ultimo mes estuve probando el combo: IA para generar ideas, constructores visuales para armar. Esta es la realidad.

## Experimento: 10 estrategias de ChatGPT → TSLab

De 10 estrategias: 3 mostraron ganancia en backtest (>20% anual), 5 cerca de cero, 2 con perdidas.

## Problema #1: La IA no entiende el contexto del mercado

ChatGPT genera estrategias logicamente correctas. Pero no conoce: la especificidad del instrumento, el regimen actual del mercado, ni tu estilo de trading. La IA necesita direccion muy precisa.

## Problema #2: Los constructores limitan la complejidad

[Claude puede generar estrategias complejas](https://medium.com/@austin-starks/i-let-claude-opus-4-create-a-trading-strategy-it-destroyed-the-market-c200bf1a19a4). Pero el constructor visual no lo soporta. La IA puede generar estrategias mas complejas de lo que el constructor puede armar.

## Problema #3: La IA alucina indicadores

ChatGPT a veces sugiere indicadores que no existen en el constructor. Hay que saber que indicadores tiene tu constructor.

## Que funciona: Los prompts correctos

**Mal prompt:** "Inventa una estrategia de trading"

**Buen prompt:** "Sugiere una estrategia para velas horarias de EUR/USD (forex). Usa solo estos indicadores: SMA, EMA, RSI, MACD. Volatilidad media del par: 50 pips/dia. Objetivo: 3-5 operaciones por semana. Stop-loss hasta 30 pips."

## Futuro o hype?

**No es el futuro. Es una herramienta.**

IA + constructores no reemplazaran al programador cuantitativo. Pero aceleraran el trabajo. Util para principiantes, reduce la barrera de entrada. Pero no es una panacea. Si quieres comprension profunda — aprende programacion.

---

**Enlaces utiles:**
- [Medium: Claude Trading Strategy](https://medium.com/@austin-starks/i-let-claude-opus-4-create-a-trading-strategy-it-destroyed-the-market-c200bf1a19a4)
- [PickMyTrade: Claude for Trading Guide](https://blog.pickmytrade.trade/claude-4-1-for-trading-guide/)
