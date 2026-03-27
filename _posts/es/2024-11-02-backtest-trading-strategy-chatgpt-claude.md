---
layout: post
title: "Buscando una idea de trading con ChatGPT y Claude: de los datos al backtest"
description: "Exploramos cómo se puede usar la IA para el análisis de datos, la búsqueda de ineficiencias y la creación de una estrategia de trading usando datos de criptomonedas a nivel de minutos como ejemplo."
date: 2024-11-02
image: /assets/images/blog/ai-trading-strategy-preview.png
tags: [ChatGPT, Claude]
lang: es
---

En este artículo, decidí comparar dos servicios populares — [ChatGPT](https://chatgpt.com/) y [Claude.ai](https://claude.ai/) — y ver cómo manejan la tarea de encontrar ineficiencias de trading a noviembre de 2024. Evalué su funcionalidad y facilidad de uso para determinar cuál es más adecuado para el análisis de datos y el desarrollo de una estrategia de trading rentable.

Para simplificar la recolección de datos, utilicé **[Hydra](https://stocksharp.ru/store/%D1%81%D0%BA%D0%B0%D1%87%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BC%D0%B0%D1%80%D0%BA%D0%B5%D1%82-%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85/)** — posiblemente la mejor herramienta gratuita para descargar datos de mercado.

Descargué datos de BTCUSDT a nivel de minutos para 2024, que sumaban aproximadamente 25 MB, y los exporté a un archivo CSV.

![](/assets/images/blog/hydra_2.png)

![](/assets/images/blog/hydra_3.png)

Hydra tiene su propia analítica integrada, pero más adelante verás lo atrasada que está en comparación con las capacidades de la IA, donde ni siquiera necesitas escribir código tú mismo:

![](/assets/images/blog/hydra_4.png)

Sin embargo, la parte principal de mi trabajo no fue la recolección de datos, sino su análisis y la búsqueda de ideas para la estrategia. En lugar de buscar enfoques manualmente, decidí confiar en la IA y descubrir qué estrategias sugeriría, qué patrones e ineficiencias podría identificar en los datos, y cómo optimizar los parámetros para las pruebas. Con la ayuda de **ChatGPT**, pude no solo realizar un análisis detallado, sino también ejecutar un backtest de la estrategia con los datos.

---

### Preparación de datos

Tras recibir los datos a nivel de minutos, los cargué en Python (el código lo escribió la IA — yo solo escribía en texto lo que necesitaba) y comencé con el preprocesamiento. Esto incluyó asignar nombres a cada columna y combinar la fecha y la hora en una sola columna para simplificar el análisis de series temporales.

---

### Búsqueda de ineficiencias con IA

Después del preprocesamiento, pregunté a la IA sobre posibles ineficiencias y patrones que pudieran ser útiles para el desarrollo de la estrategia. ChatGPT sugirió varios enfoques:

1. **Clústeres de volatilidad** — Las horas con alta volatilidad podrían ser adecuadas para una estrategia de momentum.
2. **Tendencia a la reversión a la media** — Cuando el precio se desvía del nivel promedio, se podría aplicar una estrategia de reversión a la media.
3. **Patrones de momentum** — En ciertas horas se observaron movimientos de precio sostenidos, lo que podría servir como señales para una estrategia de seguimiento de tendencia.

![](/assets/images/blog/volatility-clusters.png)

---

### Desarrollo de la estrategia

Basándome en las sugerencias de la IA, elegí dos estrategias para probar:

1. **Reversión a la media (Mean Reversion)**: Abrir una posición corta cuando el precio se desvía significativamente por encima del promedio y una posición larga cuando se desvía por debajo. La posición se cierra cuando el precio regresa a la media.

2. **Estrategia de momentum**: Abrir una posición en la dirección de la tendencia durante períodos de alta volatilidad. Si el rendimiento es positivo y está por encima del umbral, se abre una posición de compra; si es negativo y está por debajo del umbral, una posición de venta.

Para cada estrategia se definieron reglas básicas de entrada y salida, junto con stop-losses para la gestión de riesgos.

![](/assets/images/blog/hourly-returns.png)

---

### Backtesting de las estrategias

Con la ayuda de ChatGPT, también pude hacer backtest de ambas estrategias para ver cómo habrían funcionado con datos históricos. Los resultados de las pruebas mostraron la curva de capital para la estrategia de reversión a la media (ver el gráfico abajo).

El gráfico muestra cómo podría haber cambiado la capitalización del portafolio al seguir la estrategia. Se puede observar que la estrategia demostró un crecimiento estable en ciertos períodos, pero también hubo momentos de retroceso. Esto confirma la importancia del ajuste de parámetros y la gestión de riesgos.

![](/assets/images/blog/mean-reversion-equity-curve.png)

---

### Claude.ai

Durante mi trabajo, también intenté usar **Claude Sonnet** de Anthropic, que había anunciado recientemente su función de análisis de datos (más detalles [aquí](https://www.anthropic.com/news/analysis-tool)). La idea parecía prometedora: subir un archivo de 25 MB para que Claude ayudara con el análisis.

![](/assets/images/blog/claude_analytics.png)

Sin embargo, me encontré con varias dificultades. Desafortunadamente, la función resultó estar en fase inicial y sin pulir — mi archivo ni siquiera se cargaba. Al final lo dividí en partes más pequeñas, pero debido a errores previos, rápidamente alcancé el límite de solicitudes. Todo lo que logré obtener fue un error al intentar generar un gráfico.

![](/assets/images/blog/claude_error_1.png)

Aunque me encanta trabajar con Claude, espero que los ingenieros del proyecto perfeccionen esta función y amplíen significativamente la ventana de carga de datos. Esto permitiría analizar archivos grandes de manera más eficiente y abriría nuevas posibilidades para trabajar con grandes volúmenes de información.

![](/assets/images/blog/claude_error_2.png)

---

### Conclusión

Usar ChatGPT me permitió no solo analizar datos, sino también hacerle preguntas a la IA sobre métodos adecuados para la creación de estrategias. Este enfoque no solo generó nuevas ideas, sino que también ayudó a probar hipótesis rápidamente y obtener recomendaciones que podrían haber pasado desapercibidas con un enfoque tradicional. El método en el que la IA ayuda a descubrir ideas y parámetros de estrategias abre nuevas posibilidades para el desarrollo flexible y adaptativo de estrategias de trading.
