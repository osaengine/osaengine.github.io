---
layout: post
title: "Alpaca lanza un servidor MCP con soporte de IA: guia en video detallada y codigo abierto"
description: "El servidor oficial Model Context Protocol para la API de Trading de Alpaca permite operar acciones y opciones a traves de Claude AI, VS Code y otros IDEs. La empresa preparo un tutorial en video de cinco pasos y publico el codigo fuente en GitHub."
date: 2025-08-04
image: /assets/images/blog/alpaca_mcp_server_release.png
tags: [Alpaca, MCP, algorithmic trading, AI, GitHub]
lang: es
---

A finales de julio, Alpaca presento un **servidor MCP oficial** para su Trading API y publico de inmediato un **tutorial en video de cinco minutos** sobre como desplegarlo localmente y conectarlo a Claude AI. Segun los desarrolladores, el nuevo servidor deberia simplificar la creacion de estrategias de trading en lenguaje natural y reducir el tiempo entre la idea y la ejecucion de la operacion.

## Que muestra el video

El autor del video ["How to Set Up Alpaca MCP Server to Trade with Claude AI"](https://www.youtube.com/watch?v=W9KkdTZEvGM) demuestra la configuracion en cinco pasos:

1. Clonar el repositorio y crear un entorno virtual
2. Configurar las variables de entorno con las claves API de Alpaca
3. Iniciar el servidor (transporte stdio o HTTP)
4. Conectar a Claude Desktop mediante `mcp.json`
5. Primeras solicitudes de trading en lenguaje natural

De esta manera, incluso sin conocimientos profundos de Python, puede probar rapidamente el trading a traves de un asistente de IA.

## Funciones principales del servidor MCP

* **Datos de mercado**: cotizaciones en tiempo real, barras historicas, opciones y Greeks
* **Gestion de cuenta**: saldo, poder de compra, estado de la cuenta
* **Posiciones y ordenes**: apertura, liquidacion, historial de operaciones
* **Opciones**: busqueda de contratos, estrategias multi-leg
* **Acciones corporativas**: calendario de resultados, splits, dividendos
* **Watchlist** y busqueda de activos

La lista completa de funciones esta disponible en el README del repositorio.

## Repositorio en GitHub

El proyecto esta publicado bajo licencia **MIT**, ya ha acumulado **170+ estrellas y ~50 forks**, y recibe activamente pull requests de la comunidad. La ultima actualizacion esta fechada el 31 de julio de 2025.

```bash
git clone https://github.com/alpacahq/alpaca-mcp-server.git
cd alpaca-mcp-server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python alpaca_mcp_server.py
```

## Por que es importante

El servidor MCP convierte la API de Alpaca en un "sandbox" para modelos de IA: ahora puede **cubrir posiciones, crear watchlists y colocar ordenes limitadas simplemente formulando comandos en lenguaje natural**. Para los traders, esto significa:

- **Prototipado de estrategias** mas rapido sin codigo adicional
- Integracion con Claude AI, VS Code, Cursor y otras herramientas de desarrollo
- Posibilidad de conectar multiples cuentas (paper y live) mediante variables de entorno

Alpaca continua su camino hacia la democratizacion del trading algoritmico, y la comunidad ya esta agregando soporte para nuevos lenguajes y transportes. Si queria probar el trading con IA, ahora es un excelente momento para empezar.

> **Enlaces:**
> -- Guia en video en YouTube: <https://www.youtube.com/watch?v=W9KkdTZEvGM>
> -- Repositorio en GitHub: <https://github.com/alpacahq/alpaca-mcp-server>
