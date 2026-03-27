---
layout: post
title: "Por que tu robot creado en un constructor pierde en real: 5 errores de depuracion que nadie menciona"
description: "El backtest mostraba +300% anual. En real, menos 15% en un mes. Analizamos las trampas tipicas de depuracion de estrategias visuales y como evitarlas."
date: 2025-12-23
image: /assets/images/blog/debug_visual_strategies.png
tags: [debugging, backtesting, mistakes, visual builders, testing]
lang: es
---

Hace dos semanas recibi un mensaje de un lector. Habia montado una estrategia en TSLab. Un backtest sobre tres anos de historia mostro resultados fantasticos: +280% anual, drawdown maximo del 8%.

Desplego la estrategia en una cuenta demo. Despues de un mes, el resultado: menos 12%.

Que salio mal? El problema no fue el constructor. El problema fue **como testifico**.

Esta es una historia clasica. Los constructores visuales hacen que ensamblar una estrategia sea facil. Pero no la hacen **correcta**. Y la mayoria de los errores ocurren no durante el ensamblaje, sino durante las pruebas.

En este articulo: cinco trampas en las que caen el 90% de los principiantes con constructores. Y como evitarlas.

## Error #1: Sobreoptimizacion (Curve Fitting)

**Que es:**

Tomas una estrategia, ejecutas optimizacion de parametros. Pruebas SMA de 10 a 100 con paso 1. Pruebas RSI de 20 a 80 con paso 5. Encuentras la combinacion que da el mejor resultado en datos historicos.

Felicidades: acabas de crear una estrategia que funciona **solo** en ese periodo historico especifico.

**Por que es peligroso:**

[El curve fitting es cuando una estrategia se adapta tan fuertemente a los datos historicos](https://www.quantifiedstrategies.com/curve-fitting-trading/) que deja de funcionar en datos nuevos. No encontraste un patron del mercado, sino ruido aleatorio.

**Ejemplo real:**

Optimizacion de cruce de SMA en datos 2020-2023. Mejor resultado: SMA(37) y SMA(83). Retorno +180% anual.

Ejecucion en 2024: menos 5%.

Por que? Porque la combinacion 37/83 no tiene base logica. Es ajuste al ruido.

**Como reconocerlo:**

- Demasiados parametros (mas de 3-4)
- Resultados historicos perfectos (200%+ anual sin drawdowns)
- [Los parametros parecen aleatorios](https://www.buildalpha.com/3-simple-ways-to-reduce-the-risk-of-curve-fitting/) (37, 83 en lugar de numeros redondos como 20, 50)
- Los resultados caen bruscamente al cambiar un parametro en 1-2 unidades

**Como evitarlo:**

### 1. Limita el numero de parametros

[Para pruebas clasicas, usa no mas de 2 parametros optimizables](https://empirix.ru/pereoptimizacziya-strategij/). Cuantos menos, mejor.

Una estrategia simple vive mas tiempo. Una compleja muere rapido.

### 2. Pruebas fuera de muestra (Out-of-Sample)

Divide la historia en dos partes:
- **In-Sample** (70%): Optimizacion de parametros
- **Out-of-Sample** (30%): Verificacion de resultados

Si los resultados Out-of-Sample son significativamente peores, es sobreoptimizacion.

En TSLab: Optimiza en 2020-2022, prueba en 2023.

En Designer: Misma logica, cambia manualmente el periodo.

### 3. Walk-Forward Analysis

Aun mas fiable: [ejecuta una ventana deslizante](https://eatradingacademy.com/help/strategy-builders/expert-advisor-studio/strategy-tools-optimization/walk-forward-testing/).

Ejemplo:
- Optimiza en 2020-2021, prueba en 2022
- Optimiza en 2021-2022, prueba en 2023
- Optimiza en 2022-2023, prueba en 2024

Si la estrategia se mantiene en todos los periodos, es robusta.

### 4. Verifica la estabilidad de parametros

Construye un mapa de calor de los resultados de optimizacion.

Si el mejor resultado es un unico "punto caliente" en un mar de rojo, es sobreoptimizacion.

Si hay una amplia "meseta" de buenos resultados, la estrategia es estable ante cambios de parametros. Eso es bueno.

TSLab y NinjaTrader muestran graficos 3D de optimizacion. Usalos.

## Error #2: Look-Ahead Bias (Sesgo de anticipacion)

**Que es:**

Tu estrategia usa accidentalmente informacion que **aun no estaba disponible** en el momento de la decision.

**Ejemplo clasico:**

Usas un indicador en el **cierre** de la vela, pero la senal se genera en la **apertura** de la siguiente.

Problema: cuando una vela cierra, ya conoces su High/Low/Close. En el trading real, no.

**Donde ocurre:**

### En TSLab:

[TSLab cuenta el tiempo de la vela como el tiempo de inicio](https://doc.tslab.pro/tslab/rabota-s-programmoi/torgovlya-agentami-robotami/rabota-agenta-i-osobye-situacii). Si no lo tienes en cuenta, es facil crear look-ahead.

Ejemplo: El bloque "Precio de cierre" en la vela N devuelve un valor que solo sera conocido **despues** de que esa vela cierre.

Si generas una senal basada en Close[0], eso es look-ahead. Debes usar Close[1].

### En Designer:

Lo mismo. Designer trabaja con velas cerradas. Si tu logica se basa en la vela actual, verifica si esos datos estan disponibles en tiempo real.

### En NinjaTrader:

Strategy Builder tiene una opcion "Calculate on bar close". Si esta desactivada, las senales se generan en cada tick, incluyendo velas no cerradas. Si esta activada, solo al cierre.

Para la mayoria de estrategias necesitas "Calculate on bar close = true".

**Como evitarlo:**

1. **Usa solo velas cerradas**
   - Si la estrategia es en H1, la senal aparece solo despues del cierre de la vela horaria
   - No uses datos de la vela actual para generar senales

2. **Verifica retrasos en los datos**
   - Los datos macroeconomicos se publican con retraso
   - Las noticias no aparecen instantaneamente
   - [Los reportes financieros se revisan](https://analystprep.com/study-notes/cfa-level-2/problems-in-backtesting/)

3. **Ejecuta en demo antes de probar en real**
   - Si el backtest muestra 100 operaciones al mes y la demo 10, el problema es look-ahead

## Error #3: Survivorship Bias (Sesgo de supervivencia)

**Que es:**

Pruebas una estrategia en acciones que **existen hoy**. Pero en tres anos, algunas empresas quebraron, fueron deslistadas o adquiridas.

No estan en tu backtest. Pero existieron en el trading real.

**Ejemplo real:**

Estrategia en acciones rusas. Backtest 2020-2023. La lista de acciones probadas incluye:
- Sberbank ✅
- Gazprom ✅
- Yandex ✅
- TCS Holding ✅

Pero faltan:
- Rusal (deslistada en 2022) ❌
- Moscow Exchange (deslisteo temporal 2022) ❌
- Acciones que cayeron 90% y desaparecieron del radar ❌

Tu estrategia "olvido" las perdidas en esos instrumentos. [El survivorship bias infla los retornos en 1-4% anual](https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained/).

**Donde ocurre:**

### En TSLab y Designer:

Si cargas listas de acciones a traves de la conexion del broker, solo obtienes acciones **actuales**. Las deslistadas no estan.

### En NinjaTrader:

Mismo problema con futuros. Los contratos vencidos a menudo no entran en el backtest.

**Como evitarlo:**

1. **Usa bases de datos con valores deslistados**
   - [QuantConnect, Norgate Data](https://medium.com/auquan/backtesting-biases-and-how-to-avoid-them-776180378335) proporcionan datos libres de survivorship bias
   - Para el mercado ruso, es mas dificil, hay pocas bases asi

2. **Prueba en un indice, no en acciones seleccionadas**
   - Si la estrategia es en acciones de MOEX, toma todo el indice MOEX, no solo el top 10

3. **Verifica cuantos valores desaparecieron durante el periodo de prueba**
   - Si pruebas 3 anos y la lista de acciones no cambio, hay un problema

4. **Agrega filtros de liquidez**
   - La estrategia no deberia operar acciones con volumen diario inferior a 10 millones de rublos
   - Esto reduce el riesgo de entrar en acciones antes del deslisteo

## Error #4: Ignorar comisiones, deslizamiento y realidades de ejecucion

**Que es:**

Los backtests asumen: siempre compras al precio que quieres. Las ordenes se ejecutan instantaneamente. Comision = 0.

Realidad: comisiones, deslizamiento, retrasos, ejecucion parcial.

**Ejemplo real:**

Estrategia en velas de un minuto. 200 operaciones al mes. Beneficio promedio por operacion: 0.15%.

Comision del broker: 0.05% entrada, 0.05% salida. Total 0.1% ida y vuelta.

**Beneficio neto:** 0.15% - 0.1% = 0.05% por operacion.

200 operaciones * 0.05% = 10% al mes. Parece bien.

Pero agrega deslizamiento de 0.03% por operacion. Ahora: 0.15% - 0.1% - 0.03% = **0.02%**.

200 operaciones * 0.02% = **4% al mes**. Ya no tan impresionante.

Y si el spread es amplio (accion iliquida), deslizamiento de 0.1%? La estrategia es **deficitaria**.

**Como evitarlo:**

### 1. Configura comisiones en el constructor

**TSLab:**
Configuracion → Trading → Comisiones. Ingresa las comisiones reales del broker (normalmente 0.03-0.05%).

**Designer:**
La ventana de backtest tiene un campo "Comision". Configuralo en valores absolutos o porcentajes.

**NinjaTrader:**
Strategy → Properties → Commission. Ingresa la comision por contrato.

**fxDreema:**
En el codigo MQL generado, necesitas agregar verificaciones de spread manualmente.

### 2. Agrega deslizamiento (slippage)

TSLab y NinjaTrader permiten configurar el slippage por separado. Para un trader retail en acciones liquidas: 1-3 ticks.

Para iliquidas: 5-10 ticks o mas.

### 3. Prueba con spread real

Si la estrategia opera dentro del spread (scalping), verifica si el beneficio cubre el tamano del spread.

Formula simple:
```
Beneficio por operacion > Comision * 2 + Spread promedio + Deslizamiento
```

Si no, la estrategia no sobrevivira en real.

### 4. Verifica el numero de operaciones

[Cuantas mas operaciones, mayor el impacto de las comisiones](https://www.quantifiedstrategies.com/survivorship-bias-in-backtesting/).

100 operaciones al ano: las comisiones no son criticas.

1000 operaciones al ano: las comisiones pueden comerse todo el beneficio.

**Regla:** Si la estrategia rinde menos de 0.5% por operacion despues de comisiones, esta al limite. El menor deterioro del mercado la matara.

## Error #5: Ausencia de Forward Testing

**Que es:**

Un backtest es una prueba en el pasado. Un forward test es una prueba en el futuro (pero sin dinero real).

[El forward testing muestra como funciona la estrategia con datos que nunca ha visto](https://academy.ftmo.com/lesson/forward-testing-of-trading-strategies/).

**Por que importa:**

Supongamos que optimizaste una estrategia en 2020-2023. Los resultados son excelentes. La lanzas en real en 2024.

Problema: el mercado en 2024 puede comportarse diferente. La volatilidad cambio. Las correlaciones se rompieron.

El forward testing en una cuenta demo te permite verificar esto **antes** de perder dinero.

**Como hacer Forward Testing:**

### 1. Ejecuta en cuenta demo

**Duracion minima:** [3-6 meses](https://academy.ftmo.com/lesson/forward-testing-of-trading-strategies/).

Por que tanto tiempo? Porque:
- Necesitas capturar diferentes regimenes de mercado (tendencia, rango, volatilidad)
- Necesitas un minimo de 50-100 operaciones
- Necesitas verificar la resistencia psicologica (si, incluso en demo)

### 2. Lleva un diario de operaciones

Registra:
- Entrada/salida
- Razon de la operacion (que bloque genero la senal)
- Desviacion del backtest (si la hay)

Si los resultados en demo son **significativamente** peores que el backtest, algo esta roto. Vuelve a la depuracion.

### 3. Compara metricas

| Metrica | Backtest | Forward Test |
|---------|----------|--------------|
| Win Rate | 65% | ? |
| Beneficio promedio | 1.2% | ? |
| Perdida promedio | -0.8% | ? |
| Drawdown maximo | 12% | ? |
| Operaciones/mes | 20 | ? |

Si la desviacion supera el 20-30%, hay un problema.

### 4. Usa paper trading en las plataformas

**TradingView:** [Paper trading gratuito](https://wundertrading.com/journal/en/learn/article/paper-trading-tradingview) via cuenta virtual.

**AlgoTest:** [Paper trading con analiticas detalladas](https://docs.algotest.in/strategy-builder/paper-trading-analysing/).

**TSLab/Designer:** Ejecucion en simulacion con conexion real al broker (pero sin enviar ordenes).

### 5. No te apresures

El error mas comun: probar una semana en demo, ver ganancias, desplegar en real.

Una semana no es nada. Necesitas al menos 2-3 meses para entender como se comporta la estrategia en diferentes condiciones.

## Checklist antes de lanzar una estrategia en real

Antes de pulsar "Start" en una cuenta real, repasa esta lista:

### Pruebas

- [ ] Estrategia probada en al menos 2 anos de historia
- [ ] Prueba out-of-sample realizada (30% de la historia)
- [ ] Numero de parametros ≤ 3
- [ ] Parametros logicamente justificados (no ajuste al ruido)
- [ ] Resultados estables al cambiar parametros ±10%

### Sesgos

- [ ] Verificada ausencia de look-ahead bias (solo velas cerradas)
- [ ] Survivorship bias considerado (o minimizado con filtros)
- [ ] Comisiones realistas agregadas (0.03-0.05%)
- [ ] Deslizamiento agregado (1-3 ticks para instrumentos liquidos)
- [ ] Estrategia rentable despues de comisiones y deslizamiento

### Forward Testing

- [ ] Estrategia probada en cuenta demo al menos 3 meses
- [ ] Al menos 50 operaciones acumuladas
- [ ] Resultados en demo cercanos al backtest (desviacion <30%)
- [ ] Diario de operaciones mantenido
- [ ] Probada en diferentes regimenes de mercado (tendencia, rango, volatilidad)

### Gestion de riesgo

- [ ] Riesgo maximo por operacion ≤ 2% de la cuenta
- [ ] Drawdown maximo en backtest ≤ 20%
- [ ] Plan de accion existente para drawdown >15%
- [ ] Tamano de posicion calculado segun volatilidad del instrumento

Si cualquier punto no se cumple, no vayas a real.

## Herramientas de depuracion en constructores

### TSLab

**Pros:**
- Depurador integrado con ejecucion paso a paso
- Visualizacion de operaciones en graficos
- Reporte detallado por cada operacion
- [Visualizacion 3D de optimizacion](https://vc.ru/u/715109-tslab/204062-optimizaciya-mehanicheskih-torgovyh-sistem)

**Contras:**
- [Sin prueba out-of-sample automatica](http://forum.tslab.ru/ubb/ubbthreads.php?ubb=showflat&Number=86791)
- Problemas con datos de ticks

### StockSharp Designer

**Pros:**
- Configuracion flexible de comisiones y deslizamiento
- Soporte para datos de ticks y libro de ordenes
- Exportacion a C# para depuracion profunda

**Contras:**
- Menos documentacion de depuracion
- Visualizacion inferior a TSLab

### NinjaTrader Strategy Builder

**Pros:**
- Integracion con Visual Studio para depuracion de codigo
- Logs de ejecucion detallados
- Market Replay para pruebas paso a paso

**Contras:**
- Mas dificil de configurar para principiantes
- Caro ($1,500 por licencia vitalicia)

### fxDreema

**Pros:**
- Genera codigo MQL depurable en MetaEditor
- Tester visual de MetaTrader

**Contras:**
- Limitaciones de la version gratuita (10 conexiones entre bloques)
- Necesitas conocer MQL para depuracion profunda

## Conclusiones

Los constructores visuales hacen facil la creacion de estrategias. Pero la depuracion sigue siendo dificil.

**Cinco errores principales:**

1. **Sobreoptimizacion** — ajuste al ruido historico
2. **Look-ahead bias** — uso de datos futuros
3. **Survivorship bias** — ignorar valores deslistados
4. **Ignorar comisiones** — supuestos irrealistas de ejecucion
5. **Sin forward testing** — ir a real sin verificacion en demo

**Que hacer:**

- Limita parametros (≤3)
- Haz pruebas out-of-sample
- Verifica look-ahead bias
- Agrega comisiones y deslizamiento realistas
- Prueba en demo al menos 3 meses

[Un backtest correcto](https://www.morpher.com/ru/blog/backtesting-trading-strategies) no se trata de curvas de rentabilidad bonitas. Se trata de una respuesta honesta a la pregunta: "Funcionara esto en real?"

Si el backtest muestra 300% anual, lo mas probable es que haya un error en alguna parte. Retornos realistas para el algotrading retail: 20-50% anual con drawdown del 10-20%.

Si tus resultados son mucho mejores, vuelve a los puntos anteriores. Algo te falto.

---

**Enlaces utiles:**

Investigacion y recursos:
- [TradingView: How to Debug Pine Script](https://trading-strategies.academy/archives/401)
- [FTMO Academy: Forward Testing of Trading Strategies](https://academy.ftmo.com/lesson/forward-testing-of-trading-strategies/)
- [AlgoTest: Paper Trading Guide](https://docs.algotest.in/strategy-builder/paper-trading-analysing/)
- [QuantifiedStrategies: Curve Fitting in Trading](https://www.quantifiedstrategies.com/curve-fitting-trading/)
- [Build Alpha: 3 Ways to Reduce Curve-Fitting Risk](https://www.buildalpha.com/3-simple-ways-to-reduce-the-risk-of-curve-fitting/)
- [AlgoTrading101: What is Overfitting in Trading?](https://algotrading101.com/learn/what-is-overfitting-in-trading/)
- [Auquan: Backtesting Biases and How To Avoid Them](https://medium.com/auquan/backtesting-biases-and-how-to-avoid-them-776180378335)
- [LuxAlgo: Survivorship Bias Explained](https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained/)
- [Empirix: Strategy Over-Optimization](https://empirix.ru/pereoptimizacziya-strategij/)
- [LONG/SHORT: Backtesting Strategies on Historical Data](https://long-short.pro/uspeshnaya-proverka-algoritmicheskih-torgovyh-strategih-na-istoricheskih-dannyh-chast-1-oshibki-okazyvayuschie-vliyanie-309/)
- [TSLab Documentation: Agent Operation and Special Situations](https://doc.tslab.pro/tslab/rabota-s-programmoi/torgovlya-agentami-robotami/rabota-agenta-i-osobye-situacii)
- [EA Trading Academy: Walk Forward Testing](https://eatradingacademy.com/help/strategy-builders/expert-advisor-studio/strategy-tools-optimization/walk-forward-testing/)
