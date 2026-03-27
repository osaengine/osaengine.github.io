---
layout: post
title: "MCP: el protocolo que conecta la IA con cuentas de corretaje"
description: "El Model Context Protocol de Anthropic permite a los modelos de IA interactuar directamente con sistemas externos, incluidas plataformas de corretaje. Explicamos como funciona."
date: 2026-03-15
image: /assets/images/blog/mcp-protocol-trading.png
tags: [AI, MCP, trading, brokers, Claude]
lang: es
---

## Que es MCP

**Model Context Protocol (MCP)** es un protocolo abierto desarrollado por [Anthropic](https://www.anthropic.com/) que estandariza la forma en que los modelos de IA se conectan a fuentes de datos externas y herramientas. En terminos simples: MCP es un "puerto USB" para la IA.

Antes de MCP, cada integracion de IA con un servicio externo requeria escribir codigo personalizado. MCP ofrece un estandar unificado que funciona con un modelo **cliente-servidor**:

```
Modelo de IA (cliente) <-> Protocolo MCP <-> Servidor MCP <-> Servicio externo
```

## Como funciona MCP con cuentas de corretaje

La arquitectura MCP proporciona tres tipos de interaccion:

### 1. Resources (Recursos)
Obtencion de datos: cotizaciones, cartera, historial de operaciones, saldo de la cuenta.

### 2. Tools (Herramientas)
Ejecucion de acciones: colocacion de ordenes, modificacion de ordenes, cancelacion de posiciones.

### 3. Prompts
Plantillas de interaccion predefinidas: "Muestra mi cartera", "Analiza mi historial de operaciones de la semana".

## Ejemplo de interaccion

Imagina una conversacion con un asistente de IA:

> **Tu**: Que posiciones tengo actualmente en mi cartera?
>
> **IA** *(consulta al broker via MCP)*: Tienes 3 posiciones en tu cartera:
> - AAPL: 50 acciones, +4.2%
> - GOOGL: 20 acciones, -1.1%
> - SBER: 100 acciones, +2.8%
>
> Rentabilidad total de la cartera: +3.1%

O incluso:

> **Tu**: Pon un stop-loss en AAPL a $210.
>
> **IA**: He creado una orden stop de venta de 50 acciones de AAPL a $210. Confirmas?

## Seguridad

La seguridad es la cuestion clave para las integraciones financieras. MCP incluye:

- **Autorizacion OAuth 2.0** — protocolo estandar para acceso seguro
- **Permisos granulares** — se puede permitir solo lectura de datos sin operaciones
- **Logs de auditoria** — todas las acciones se registran
- **Confirmacion de operaciones** — las acciones criticas requieren consentimiento explicito del usuario

## Quien ya soporta MCP

A marzo de 2026, hay servidores MCP disponibles para:

- **TradeStation** — el primer broker con integracion MCP completa
- **FactSet**, **S&P Global**, **MSCI** — a traves de los plugins financieros de Anthropic
- Otros brokers y plataformas estan explorando opciones de integracion

## Que significa esto para el algo trading

MCP abre el camino hacia el **trading gestionado por IA**, donde el modelo no solo genera senales, sino que puede:

1. Analizar de forma autonoma las condiciones del mercado
2. Formular decisiones de trading
3. Ejecutarlas a traves del broker
4. Monitorizar resultados y ajustar la estrategia

Esto no significa que debas dar a la IA control total sobre tu cuenta. Pero el enfoque de **humano en el bucle** — donde la IA propone y el humano confirma — ya esta aqui.

## Como empezar

Si quieres experimentar con MCP:

1. Instala [Claude Desktop](https://claude.ai/download) u otro cliente compatible con MCP
2. Conecta el servidor MCP de tu broker
3. Empieza en modo solo lectura — solo visualizacion de datos
4. Anade capacidades gradualmente a medida que crece tu confianza

MCP no es una revolucion de un dia — es la base de una nueva era de interaccion humano-IA en las finanzas.
