---
layout: post
title: "TradeTrap: ¿cuán fiables son realmente los traders LLM?"
description: "El estudio TradeTrap reveló serios problemas de fiabilidad y faithfulness en los traders LLM. Analizamos por qué los bots de IA toman decisiones de forma diferente a como las explican."
date: 2026-03-20
image: /assets/images/blog/tradetrap-llm-reliability.png
tags: [AI, LLM, trading, reliability, research]
lang: es
---

## El problema de la faithfulness

Cuando un trader LLM explica su decisión -- "compré AAPL porque el RSI muestra sobreventa y los resultados superaron las expectativas" -- ¿realmente se basó en esos factores? ¿O la explicación es una **racionalización a posteriori**, y la "decisión" real se tomó por motivos completamente diferentes?

El estudio **TradeTrap** de un grupo de investigadores examinó precisamente esta cuestión.

## Metodología de la investigación

Los investigadores crearon un entorno controlado en el que:

1. Los agentes LLM recibían **datos de mercado y noticias** para tomar decisiones de trading
2. Parte de los datos contenían **trampas intencionales** -- señales falsas que parecían convincentes
3. Los agentes debían tomar decisiones y **explicarlas**
4. Los investigadores compararon las razones **declaradas** con los **desencadenantes reales**

### Tipos de trampas

- **Trampa de anclaje** -- se insertaba un "precio objetivo" aleatorio en el contexto, sin base analítica
- **Trampa de recencia** -- los datos recientes eran peores que la media, pero la tendencia seguía siendo positiva
- **Trampa de autoridad** -- citas falsas de "analistas reconocidos" con pronósticos incorrectos
- **Trampa de confirmación** -- datos que confirmaban el sesgo preexistente del modelo

## Resultados

### Tasa de caída en trampas

*Nota: las tablas utilizan modelos disponibles en el momento del estudio (finales de 2025).*

| Modelo | Anclaje | Recencia | Autoridad | Confirmación |
|--------|---------|----------|-----------|-------------|
| GPT-4o | 34% | 41% | 28% | 52% |
| Claude 3.5 Sonnet | 22% | 35% | 19% | 44% |
| DeepSeek V3 | 39% | 48% | 33% | 57% |
| Gemini 2.0 Flash | 31% | 38% | 25% | 49% |

### Puntuación de Faithfulness

Cuánto coinciden las explicaciones del modelo con las razones reales de sus decisiones:

| Modelo | Faithfulness |
|--------|-------------|
| Claude 3.5 Sonnet | 67% |
| GPT-4o | 61% |
| Gemini 2.0 Flash | 58% |
| DeepSeek V3 | 54% |

Esto significa que en el **33-46% de los casos**, las explicaciones de los traders LLM **no corresponden** a las razones reales de sus decisiones.

## Conclusiones clave

### 1. El sesgo de confirmación es el mayor problema

Todos los modelos mostraron la mayor vulnerabilidad a **confirmar sus propios prejuicios**. Si un modelo "decidió" comprar un activo, encuentra datos que respaldan esa decisión, incluso cuando los datos objetivos dicen lo contrario.

### 2. El Chain-of-Thought no salva

Incluso los modelos de razonamiento con cadenas de pensamiento (Chain-of-Thought) detalladas son susceptibles a las trampas. Es más, una cadena de razonamiento larga a veces **enmascara** la falta de fiabilidad de las decisiones, creando la ilusión de un análisis profundo.

### 3. El coste de los errores crece con la autonomía

Cuanta más autonomía tiene un trader LLM, más caro resulta cada error de faithfulness. Si el agente coloca órdenes automáticamente basándose en razonamientos incorrectos, las consecuencias pueden ser graves.

## Recomendaciones prácticas

Los autores del estudio sugieren:

- **No confiar en las explicaciones** de los traders LLM -- verificar las decisiones de forma independiente
- **Usar enfoques de ensemble** -- varios modelos votan sobre una decisión
- **Limitar la autonomía** -- humano en el bucle para operaciones grandes
- **Probar con datos adversariales** -- comprobar cómo reacciona el agente ante trampas
- **Registrar todos los pasos intermedios** -- para análisis post-mortem de errores

## Qué significa para la industria

TradeTrap es una señal importante para todos los que construyen sistemas de trading con IA. **Un buen resultado en SWE-Bench o MMLU no significa fiabilidad en el trading.** Se necesitan pruebas especializadas que contemplen trampas cognitivas y faithfulness.

El texto completo del estudio está disponible en [arXiv](https://arxiv.org/abs/2512.02261).
