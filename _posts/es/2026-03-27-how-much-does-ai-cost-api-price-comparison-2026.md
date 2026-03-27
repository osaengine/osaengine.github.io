---
layout: post
title: "Cuánto cuesta la IA: comparación de precios de API en 2026"
description: "Comparación completa de precios de API de modelos de lenguaje en 2026: GPT-5, Claude, Gemini, DeepSeek y Qwen. Calculamos costes para casos de uso típicos."
date: 2026-03-27
image: /assets/images/blog/llm-api-prices-2026.png
tags: [AI, API, pricing, comparison]
lang: es
---

## La guerra de precios

El mercado de API de LLM en 2026 está viviendo una auténtica guerra de precios. En el último año, el coste de inferencia ha caído entre **2 y 5 veces** según el proveedor. Veamos cuánto cuestan ahora los principales modelos y cómo elegir la mejor opción.

## Tabla de precios (marzo 2026)

### Modelos insignia

| Modelo | Input ($/1M) | Output ($/1M) | Input cacheado | Contexto |
|--------|-------------|---------------|---------------|----------|
| **GPT-5.3** | $8,00 | $24,00 | $2,00 | 128K |
| **Claude Opus 4.6** | $15,00 | $75,00 | $3,75 | 200K |
| **Claude Sonnet 4.6** | $3,00 | $15,00 | $0,75 | 256K |
| **Gemini 3.1 Pro** | $3,50 | $10,50 | $0,88 | 1M |
| **DeepSeek V3** | $0,27 | $1,10 | $0,07 | 128K |
| **Qwen 3 72B** | $0,40 | $1,20 | -- | 128K |

### Modelos ligeros

| Modelo | Input ($/1M) | Output ($/1M) | Contexto |
|--------|-------------|---------------|----------|
| **GPT-5.3 Mini** | $0,40 | $1,60 | 128K |
| **Claude Haiku 3.5** | $0,80 | $4,00 | 200K |
| **Gemini 3.1 Flash** | $0,15 | $0,60 | 1M |
| **DeepSeek V3 Lite** | $0,07 | $0,28 | 64K |
| **Qwen 3 7B** | $0,05 | $0,15 | 32K |

### Modelos de razonamiento

| Modelo | Input ($/1M) | Output ($/1M) |
|--------|-------------|---------------|
| **o3** | $10,00 | $40,00 |
| **o4-mini** | $1,10 | $4,40 |
| **DeepSeek R1** | $0,55 | $2,19 |
| **Claude Sonnet 4.6 (extended)** | $3,00 | $15,00 |

## Cuánto cuesta en la práctica

### Escenario 1: Análisis de un informe financiero

- Tamaño del documento: ~30.000 tokens (input)
- Respuesta del modelo: ~2.000 tokens (output)

| Modelo | Coste por solicitud |
|--------|---------------------|
| GPT-5.3 | $0,29 |
| Claude Sonnet 4.6 | $0,12 |
| Gemini 3.1 Pro | $0,13 |
| DeepSeek V3 | **$0,01** |

### Escenario 2: Análisis diario de noticias (100 artículos)

- Input: ~500.000 tokens/día
- Output: ~50.000 tokens/día

| Modelo | Coste/día | Coste/mes |
|--------|-----------|-----------|
| GPT-5.3 | $5,20 | $156 |
| Claude Sonnet 4.6 | $2,25 | $67,50 |
| Gemini 3.1 Pro | $2,28 | $68,25 |
| DeepSeek V3 | **$0,19** | **$5,64** |

### Escenario 3: Sistema de trading con agentes (24/7)

- Solicitudes por día: ~1.000
- Input medio: 10.000 tokens
- Output medio: 1.000 tokens
- Mensual: 300M input + 30M output

| Modelo | Coste/mes |
|--------|-----------|
| GPT-5.3 | $3.120 |
| Claude Opus 4.6 | $6.750 |
| Claude Sonnet 4.6 | $1.350 |
| Gemini 3.1 Pro | $1.365 |
| DeepSeek V3 | **$114** |

## Costes ocultos

El precio por token no es el único factor:

### Límites de velocidad (Rate limits)

- OpenAI: 500-10.000 RPM (según el plan)
- Anthropic: 1.000-4.000 RPM
- Google: hasta 60.000 RPM
- DeepSeek: limitaciones bajo carga alta

### Latencia

- GPT-5.3: ~800ms TTFT
- Claude Sonnet 4.6: ~600ms TTFT
- Gemini 3.1 Pro: ~500ms TTFT
- DeepSeek V3: ~1200ms TTFT (por la geografía de servidores)

### Fiabilidad (Uptime)

- OpenAI: 99,8% (incidentes ocasionales)
- Anthropic: 99,9%
- Google: 99,95%
- DeepSeek: 99,5% (infraestructura joven)

## Recomendaciones

| Caso de uso | Mejor opción | Razón |
|-------------|-------------|-------|
| Análisis masivo de datos | DeepSeek V3 | Precio |
| Decisiones críticas | Claude Opus 4.6 | Calidad |
| Programación | Claude Sonnet 4.6 | SWE-Bench |
| Contexto largo | Gemini 3.1 Pro | 1M tokens |
| Opción económica | Qwen 3 7B (autoalojado) | Gratis |

Los precios siguen bajando. Lo que hoy cuesta 100 $/mes podría costar 20 $ dentro de un año. Planifica tu infraestructura teniendo en cuenta esta tendencia.
