---
layout: post
title: "Gemini 3.1 Pro: un millon de tokens de contexto y 77% en ARC-AGI-2"
description: "Google ha presentado Gemini 3.1 Pro con una ventana de contexto de 1 millon de tokens y un resultado del 77% en ARC-AGI-2. La cuota de mercado de Google AI crece."
date: 2026-03-14
image: /assets/images/blog/gemini-3-1-pro.png
tags: [AI, Gemini, Google, benchmarks]
lang: es
---

## Un millon de tokens: para que sirve

Google sigue apostando por el contexto largo. **Gemini 3.1 Pro** mantiene su ventana record de **1 millon de tokens** — aproximadamente 700.000 palabras, o varios libros completos.

En terminos practicos, esto significa:

- Cargar una **base de codigo completa** de un proyecto mediano en una sola solicitud
- Analizar **informes financieros anuales** sin perder contexto
- Trabajar con **largos historiales de conversacion** y documentacion
- Procesar **transcripciones de varias horas** de reuniones y llamadas

## ARC-AGI-2: prueba de razonamiento abstracto

El benchmark [ARC-AGI-2](https://arcprize.org/) evalua la capacidad del modelo para el razonamiento abstracto — tareas que un nino resuelve facilmente pero que desconciertan a la mayoria de los sistemas de IA.

Gemini 3.1 Pro obtuvo un **77% en ARC-AGI-2**, uno de los mejores resultados entre los modelos comerciales:

| Modelo | ARC-AGI-2 |
|--------|-----------|
| Claude Sonnet 4.6 | 79.2% |
| **Gemini 3.1 Pro** | **77.0%** |
| GPT-5.3 | 74.5% |
| DeepSeek V3 | 71.3% |

## La cuota de mercado crece

Segun los analistas, la cuota de Google en el mercado de APIs de LLM ha crecido del 12% al 18% en los ultimos seis meses. Razones principales:

### Politica de precios

Google ofrece algunos de los precios mas competitivos:

- **Input**: $3.50 / 1M tokens
- **Output**: $10.50 / 1M tokens
- Para contexto >128K: $7 / $21 por millon

### Ecosistema Google Cloud

La integracion con **Vertex AI**, **BigQuery** y otros servicios de Google Cloud hace que Gemini sea atractivo para clientes corporativos que ya utilizan la infraestructura en la nube de Google.

### Multimodalidad

Gemini 3.1 Pro soporta de forma nativa:

- Texto, imagenes, audio y video
- Generacion y analisis de codigo
- Trabajo con tablas y datos estructurados

## Que significa esto para los traders

El contexto largo de 1M de tokens abre posibilidades interesantes:

1. **Cargar historiales de operaciones completos** de periodos extensos para analizar patrones
2. **Analizar simultaneamente** multiples informes financieros de empresas
3. **Procesar el flujo de noticias de un dia completo** sin perder detalles importantes

Dicho esto, hay que tener en cuenta que la calidad del trabajo con informacion al principio y al final de un contexto largo puede variar — el llamado problema de "perdido en el medio".

Google se consolida firmemente entre los tres lideres del mercado de IA, y Gemini 3.1 Pro es un argumento solido en esta carrera.
