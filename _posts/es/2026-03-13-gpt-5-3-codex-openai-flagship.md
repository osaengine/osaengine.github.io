---
layout: post
title: "GPT-5.3 Codex: OpenAI actualiza su modelo insignia"
description: "OpenAI ha lanzado GPT-5.3 Codex, una versión actualizada de su modelo insignia con capacidades de programación mejoradas. Lo comparamos con Claude 4.6 Opus."
date: 2026-03-13
image: /assets/images/blog/gpt-5-3-codex.png
tags: [AI, GPT-5, OpenAI, coding]
lang: es
---

## GPT-5.3 Codex: novedades

OpenAI continua desarrollando la linea GPT-5. La nueva version **GPT-5.3 Codex** se posiciona como el mejor modelo de la compania para tareas de programacion. Segun OpenAI, el modelo muestra mejoras significativas en:

- **Generacion de codigo** en todos los lenguajes populares
- **Depuracion y refactorizacion** de bases de codigo existentes
- **Explicacion de codigo** — el modelo "ve" mejor la arquitectura de los proyectos
- **Generacion de tests** — la generacion de pruebas unitarias es notablemente mas precisa

## Benchmarks

Resultados de pruebas independientes:

| Prueba | GPT-5.3 Codex | Claude Opus 4.6 | Claude Sonnet 4.6 |
|--------|---------------|-----------------|-----------------|
| SWE-Bench Verified | 78.4% | 76.1% | 82.1% |
| HumanEval+ | 95.8% | 94.3% | 96.2% |
| MBPP+ | 88.2% | 87.1% | 89.7% |
| Codeforces Rating | 1847 | 1792 | 1801 |

GPT-5.3 Codex supera con confianza a Claude Opus 4.6 en tareas de codificacion, pero sigue por detras de Claude Sonnet 4.6 en SWE-Bench.

## Mejoras clave

### Contexto ampliado para codigo

GPT-5.3 Codex cuenta con **128K tokens de contexto** optimizados para archivos de codigo. OpenAI afirma que el modelo puede mantener en "memoria" la estructura de un proyecto de varios cientos de archivos.

### Function calling mejorado

Para los desarrolladores que usan la API, el **function calling** se ha vuelto mas fiable. El modelo genera esquemas JSON de llamadas con mayor precision y "inventa" parametros inexistentes con menor frecuencia.

### Modo Codex Agent

OpenAI presento el modo **Codex Agent**, en el que el modelo puede:

- Ejecutar comandos secuencialmente en la terminal
- Leer y modificar archivos
- Ejecutar tests e iterar sobre los resultados

Esta es una respuesta directa a **Claude Code** de Anthropic y productos de agentes similares.

## Precios

GPT-5.3 Codex esta disponible a traves de la API con los siguientes precios:

- **Input**: $8 / 1M tokens
- **Output**: $24 / 1M tokens
- **Input en cache**: $2 / 1M tokens

Esto situa al modelo en el segmento de precio medio — mas caro que DeepSeek, pero mas barato que Claude Opus.

## Que elegir para bots de trading?

Para los desarrolladores de sistemas de trading algoritmico, la eleccion entre GPT-5.3 y Claude depende de la tarea:

- **Para escribir estrategias desde cero** — Claude Sonnet 4.6 muestra los mejores resultados
- **Para integracion con APIs existentes** — GPT-5.3 Codex gana gracias a su preciso function calling
- **Para analisis de datos de mercado** — ambas opciones funcionan bien, pero GPT-5.3 es mas rapido en generacion en streaming

La competencia entre modelos solo se intensifica, y eso es una excelente noticia para los usuarios finales.
