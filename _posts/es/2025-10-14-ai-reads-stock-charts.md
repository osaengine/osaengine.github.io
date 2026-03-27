---
layout: post
title: "Puede la IA leer graficos bursatiles? Un experimento con DistilBERT"
description: "Un desarrollador enseno a un modelo de lenguaje a predecir movimientos de precios mediante descripciones textuales de graficos. Las pruebas con mas de 200 acciones de la Bolsa de Moscu mostraron un AUC de 0.53."
date: 2025-10-14
image: /assets/images/blog/llm_stock_charts.png
tags: [machine learning, Moscow Exchange, experiment]
lang: es
---

Mikhail Shardin realizo un experimento: puede un modelo de lenguaje predecir precios si se describen los graficos en texto?

## La idea

En lugar de cotizaciones crudas, el modelo recibia descripciones en lenguaje natural: precio subiendo fuertemente, volumen aumentando, cerca de resistencia.

El modelo DistilBERT fue entrenado para predecir subidas de precio al dia siguiente.

## Resultados

Probado en mas de 200 acciones de la Bolsa de Moscu:

- AUC promedio: 0.53 (ligeramente mejor que aleatorio)
- Mejores resultados: AFLT (0.72), RTSB (0.70), PIKK (0.70)
- Peores resultados: PLZL (0.33), VJGZP (0.33)

Para fines de trading el resultado es debil, pero el modelo capto patrones sin acceso directo a los numeros -- eso ya es interesante.

## Tecnologia

Python + PyTorch + Hugging Face + Docker. Validacion walk-forward, procesamiento vectorizado con pandas. Todo el proceso es reproducible.

**Codigo en GitHub:** [github.com/empenoso/llm-stock-market-predictor](https://github.com/empenoso/llm-stock-market-predictor)

---

**Fuente:** [Habr](https://habr.com/ru/articles/955612/) | **Autor:** Mikhail Shardin
