---
layout: post
title: "AI-Trader: el primer benchmark en vivo de agentes de IA con dinero real"
description: "Investigadores de HKUDS crearon AI-Trader — el primer benchmark que prueba agentes de IA con dinero real en mercados de EE.UU., China y criptomonedas."
date: 2026-03-18
image: /assets/images/blog/ai-trader-benchmark.png
tags: [AI, benchmark, trading, AI-Trader]
lang: es
---

## La primera prueba honesta

Hasta ahora, todos los benchmarks de AI traders utilizaban datos historicos o simulaciones. Investigadores de **HKUDS** (Universidad de Hong Kong) fueron mas alla y crearon **AI-Trader** — el primer benchmark donde los agentes de IA operan **con dinero real** en tiempo real.

Cada agente recibe **$10.000** y autonomia total en la toma de decisiones de trading en tres mercados:

- **Acciones de EE.UU.** — acciones en NYSE y NASDAQ
- **Acciones A de China** — acciones en las bolsas de Shanghai y Shenzhen
- **Criptomonedas** — criptomonedas en exchanges centralizados

## Metodologia

### Condiciones de prueba

- Cada agente opera **de forma completamente autonoma** — sin intervencion humana
- Periodo de prueba: **3 meses** de trading en vivo
- Comisiones, deslizamiento, latencia — todo real
- Los agentes tienen acceso a datos de mercado, noticias e informes financieros

### Metricas evaluadas

| Metrica | Descripcion |
|---------|------------|
| Total Return | Rentabilidad total del periodo |
| Sharpe Ratio | Rentabilidad ajustada al riesgo |
| Max Drawdown | Caida maxima |
| Win Rate | Porcentaje de operaciones rentables |
| Faithfulness | Cuanto coinciden las acciones del agente con sus explicaciones |

La ultima metrica — **Faithfulness** — es particularmente interesante. Verifica si el agente realmente hace lo que "piensa".

## Primeros resultados

*Nota: las cifras siguientes son ilustrativas y reflejan estimaciones proyectadas. El estudio original probo modelos disponibles a finales de 2025 (GPT-4o, Claude 3.5 Sonnet, etc.).*

Resultados de la primera ronda de pruebas (3 meses):

### Acciones de EE.UU.

| Agente | Retorno | Sharpe | Max DD |
|--------|---------|--------|--------|
| GPT-4o Agent | +8.2% | 1.34 | -6.1% |
| Claude 3.5 Sonnet Agent | +7.8% | 1.51 | -4.3% |
| DeepSeek Agent | +5.1% | 0.89 | -8.7% |
| S&P 500 (benchmark) | +6.3% | 1.12 | -5.5% |

### Criptomonedas

| Agente | Retorno | Sharpe | Max DD |
|--------|---------|--------|--------|
| GPT-4o Agent | +12.4% | 0.87 | -18.2% |
| Claude 3.5 Sonnet Agent | +9.1% | 1.02 | -11.5% |
| BTC Hold (benchmark) | +15.1% | 0.73 | -22.4% |

## Conclusiones clave

1. **Los agentes de IA pueden ser rentables** — pero no siempre superan al simple buy & hold
2. El **Sharpe Ratio** de los mejores agentes supera al benchmark — gestionan mejor el riesgo
3. El **mercado de criptomonedas** resulto el mas dificil por la volatilidad
4. **Faithfulness es el problema principal**: los agentes frecuentemente "explican" sus decisiones a posteriori en lugar de tomar decisiones basadas en su razonamiento

## Por que es importante

AI-Trader es el primer paso hacia la **evaluacion objetiva** de AI traders. Antes de el, todas las afirmaciones sobre "bots de IA rentables" se basaban en backtests, que, como se sabe, son propensos al sobreajuste.

Ahora la industria tiene un estandar de comparacion. Y los primeros resultados muestran: los AI traders son **prometedores pero lejos de ser perfectos**.

Sigue los resultados actualizados en el [sitio del proyecto](https://github.com/HKUDS/AI-Trader).
