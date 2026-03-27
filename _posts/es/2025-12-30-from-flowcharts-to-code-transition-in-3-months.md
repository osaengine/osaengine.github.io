---
layout: post
title: "De diagramas de flujo a codigo: como pase de constructores a programacion en 3 meses"
description: "La historia real de la transicion de constructores visuales a programacion completa. El plan, errores, herramientas y por que es mas facil de lo que parece."
date: 2025-12-30
image: /assets/images/blog/visual_to_code.png
tags: [learning, programming, Python, transition, builders]
lang: es
---

Hace un ano ensamblaba estrategias en TSLab. Diagramas de flujo, drag-and-drop, sin codigo. Funcionaba. Hasta que choque con los limites.

Necesitaba un indicador personalizado. Necesitaba estadisticas de operaciones en tiempo real. Necesitaba integracion con una API externa.

El constructor no podia manejarlo.

Decidi aprender programacion. Hace tres meses escribi mi primera linea en Python. Hoy mi robot opera, y todo el codigo es mio.

Esta no es una historia de "soy un genio de la programacion." Es una historia de "cualquiera puede hacerlo si sabe por donde empezar."

## Por que decidi aprender a programar

**Detonante #1: Choque con las limitaciones del constructor**

Queria agregar un stop-loss adaptativo basado en ATR. TSLab tiene un bloque ATR. Tiene un bloque de stop-loss. Pero no hay bloque para "ajustar dinamicamente el stop-loss cada vela basandose en ATR."

**Detonante #2: Vendor Lock-In**

Todo lo que construi en TSLab solo vive en TSLab. Si la plataforma cierra, actualiza o se rompe, mis estrategias mueren. El codigo en Python es un archivo. Es mio para siempre.

**Detonante #3: Curiosidad**

Entendia la logica de las estrategias. Veia las conexiones entre bloques. Pero que pasa *dentro*? El constructor ocultaba la complejidad. Pero cuando algo se rompia, no entendia *por que*. El codigo da control. Control total.

## Hoja de ruta: 3 meses de cero a robot funcional

### **Semanas 1-4: Fundamentos de Python**

Variables, tipos de datos, condiciones, bucles, funciones, manejo de archivos. 1-2 horas al dia, 5 dias a la semana.

**Primer resultado:** Un script que lee un CSV con cotizaciones, calcula una media movil e imprime cuando SMA(20) cruza SMA(50).

### **Semanas 5-8: Bibliotecas de analisis de datos**

**Pandas**, **NumPy**, **Matplotlib**. Funciones para calcular cualquier indicador:

```python
import pandas as pd

def sma(data, period):
    return data['Close'].rolling(window=period).mean()

def rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### **Semanas 9-12: Backtrader — primer sistema de trading**

```python
import backtrader as bt

class SmaCross(bt.Strategy):
    params = (('fast', 20), ('slow', 50),)

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(period=self.params.fast)
        self.sma_slow = bt.indicators.SMA(period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if self.crossover > 0:
            if not self.position:
                self.buy()
        elif self.crossover < 0:
            if self.position:
                self.sell()
```

La misma logica que tenia en TSLab. Pero **controlo cada linea**.

## Errores que cometi

1. **Intente aprender todo a la vez** — Sobrecarga de informacion. Solucion: una fuente a la vez.
2. **Leia pero no escribia codigo** — Regla: por cada hora de teoria, una hora de practica.
3. **No hacia proyectos** — [Fije una meta: estrategia funcional en Backtrader al final de 3 meses](https://algotrading101.com/learn/quantitative-trader-guide/).
4. **Miedo a preguntar** — Stack Overflow, Reddit (r/algotrading). La gente ayuda si la pregunta esta bien formulada.

## Cuando tiene sentido aprender programacion y cuando no

### **Aprende programacion si:**
1. Chocaste con las limitaciones del constructor
2. Necesitas logica personalizada (ML, arbitraje, portafolios)
3. Planeas dedicarte al algotrading en serio durante anos
4. Te interesa el proceso

### **No aprendas programacion si:**
1. Tu estrategia cabe en los bloques del constructor y funciona
2. No tienes tiempo (minimo 1-2 horas al dia por 3 meses)
3. Operas manualmente y solo quieres automatizar una idea
4. La programacion te causa rechazo

## Que cambio tras el paso al codigo

**Pros:** Control total, independencia de plataformas, gratis, comprension profunda, comunidad enorme.

**Contras:** Sin visualizacion, mas tiempo al inicio, depuracion mas dificil, requiere aprendizaje.

## Conclusiones

Hace un ano pensaba: "La programacion es para informaticos. Yo solo soy un trader."

Hoy entiendo: la programacion es una herramienta. Como Excel. Como TradingView. No me converti en desarrollador. Escribi 500 lineas de codigo que hacen lo que necesito. Y eso es **suficiente**.

Programar para algotrading no es "convertirse en programador." Es "automatizar tu idea sin limitaciones." Y es mas facil de lo que parece.

---

**Enlaces utiles:**

- [Should I Use C# Or Python To Build Trading Bots?](https://spreadbet.ai/python-or-c-trading-bots/)
- [Top Languages for Building Custom Trading Bots](https://blog.traderize.com/posts/top-languages-trading-bots/)
- [AlgoTrading101: Quantitative Trader's Roadmap](https://algotrading101.com/learn/quantitative-trader-guide/)
- [Start Algorithmic Trading: Beginner's Roadmap](https://startalgorithmictrading.com/beginners-algo-trading-roadmap)
