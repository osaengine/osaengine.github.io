---
layout: post
title: "Por qué el 95% de los algo traders minoristas pierden dinero: análisis con datos"
description: "Analizamos las estadísticas reales de pérdidas en MOEX, los costes ocultos del trading algorítmico y por qué los backtests mienten. Basado en datos de Habr y estadísticas bursátiles."
date: 2026-03-18
image: /assets/images/blog/ritejl-algotrejding-poteri.png
tags: [algo trading, MOEX, statistics, risks]
lang: es
---

## La dura verdad

La cifra de "el 95% de los traders pierden dinero" se ha convertido en un meme, pero detrás hay datos reales. Veamos por qué el trading algorítmico -- especialmente para los participantes minoristas -- sigue siendo una actividad extremadamente difícil.

## Estadísticas de MOEX

Según los datos de la Bolsa de Moscú y el análisis de [Habr](https://habr.com/):

- **El 76% de los traders activos** en MOEX son no rentables en un año
- Entre quienes utilizan trading algorítmico, aproximadamente el **~70%** son no rentables -- algo mejor, pero no radicalmente distinto
- Pérdida media del algo trader minorista: **-12% anual** (después de comisiones)
- Solo el **3-5%** obtiene beneficios de forma consistente en un horizonte de más de 3 años

## Costes ocultos que matan las estrategias

### 1. Comisiones del exchange y del bróker

Comisiones típicas para minoristas en MOEX:

```
Comisión del exchange (mercado de acciones):
- Maker: 0,01% del volumen de la operación
- Taker: 0,015% del volumen de la operación

Comisión del bróker:
- Del 0,03% al 0,06% (depende del bróker y la tarifa)

Total ida y vuelta (apertura + cierre):
- Mínimo: 0,08% del volumen
- Típico: 0,12-0,15% del volumen
```

Con 10 operaciones al día y un tamaño medio de posición de 100.000 rublos:

```
10 operaciones × 0,12% × 100.000 = 1.200 RUB/día
× 250 días de negociación = 300.000 RUB/año
```

Eso supone **300.000 rublos al año** solo en comisiones. Con un depósito de 1.000.000 de rublos, eso es un 30% anual que necesitas ganar solo para llegar al punto de equilibrio.

### 2. Deslizamiento (Slippage)

El slippage es la diferencia entre el precio al que la estrategia "quería" entrar y el precio al que realmente se ejecutó la orden:

- En instrumentos líquidos (Sberbank, Gazprom): **0,01-0,05%**
- En instrumentos menos líquidos: **0,1-0,5%**
- En momentos de noticias: **1-5%+**

### 3. Impacto de mercado (Market Impact)

Si tu orden es significativa respecto al libro de órdenes, tú mismo mueves el precio en tu contra. Para un trader minorista esto es raro en instrumentos líquidos, pero en valores poco negociados es un problema serio.

## Por qué los backtests mienten

### Sesgo de anticipación (Look-ahead Bias)

El error más común: utilizar datos que aún no estaban disponibles en el momento de la decisión. Ejemplos:

- Usar el precio de cierre del día para tomar una decisión **ese mismo día**
- Usar datos ajustados que se modificaron retroactivamente

### Sesgo de supervivencia (Survivorship Bias)

Un backtest sobre acciones del S&P 500 solo tiene en cuenta las empresas **que sobrevivieron**. Las que quebraron o fueron adquiridas no aparecen en la muestra, creando la ilusión de una mayor rentabilidad.

### Sobreajuste (Overfitting)

El enemigo más traicionero:

```
Cuantos más parámetros tiene una estrategia,
mejor funciona con datos históricos
y peor lo hace en el mercado real.
```

Si tu estrategia tiene más de 10 parámetros y muestra un 200% de rentabilidad anual en backtest, lo más probable es que esté sobreajustada.

### Cambio de régimen (Regime Change)

Los mercados cambian. Una estrategia que funcionó en 2020-2023 puede dejar de funcionar por completo en 2024-2026. Ejemplos:

- Las estrategias de volatilidad diseñadas antes del COVID se rompieron en la pandemia
- Las estrategias momentum ajustadas para un mercado alcista pierden en mercados laterales
- Las estrategias de arbitraje "se cierran" a medida que más personas las copian

## El coste real del algo trading

Más allá de los costes de operación:

| Concepto | Coste anual |
|----------|-------------|
| Servidor (VPS/colocation) | 30.000 - 300.000 RUB |
| Datos (históricos + tiempo real) | 10.000 - 100.000 RUB |
| Software (plataforma, herramientas) | 0 - 50.000 RUB |
| Tu propio tiempo | incalculable |

## Qué hacer si aún así quieres intentarlo

1. **Empieza con poco** -- con un depósito que puedas permitirte perder
2. **Incluye TODOS los costes** en el backtest -- comisiones, slippage, latencia
3. **Prueba con datos fuera de muestra** -- divide el historial en conjuntos de entrenamiento y de prueba
4. **Limita el número de parámetros** -- cuanto más simple sea la estrategia, mejor
5. **Usa análisis walk-forward** -- revisa los parámetros regularmente
6. **Empieza con trading en papel** -- prueba la estrategia en tiempo real sin dinero
7. **Diversifica** -- no apuestes todo a una sola estrategia

El trading algorítmico no es un "botón de hacer dinero". Es un trabajo serio de ingeniería y análisis que requiere disciplina, capital y honestidad contigo mismo.
