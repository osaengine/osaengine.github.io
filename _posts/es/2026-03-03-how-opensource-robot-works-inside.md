---
layout: post
title: "Como funciona un robot open-source tipico por dentro: arquitectura de una estrategia simple"
description: "Analizamos la arquitectura real de robots de trading: desde capas y componentes hasta patrones de diseno. Freqtrade, NautilusTrader, microservicios, Event Sourcing — que funciona en la practica."
date: 2026-03-03
image: /assets/images/blog/opensource_robot_architecture.png
tags: [architecture, open-source, trading robots, Event Sourcing, microservices]
lang: es
---

Hace tres meses [compare frameworks open-source para MOEX](/es/blog/2026/02/24/comparing-lean-stocksharp-backtrader.html). Descubri que elegir. Pero la pregunta seguia: **como funciona todo esto por dentro?**

Hoy no es sobre elegir una plataforma. Es sobre **lo que pasa bajo el capo**.

Analice la arquitectura de cuatro robots open-source populares: [Freqtrade](https://github.com/freqtrade/freqtrade), [NautilusTrader](https://github.com/nautechsystems/nautilus_trader), [Hummingbot](https://github.com/hummingbot/hummingbot) y el sistema de microservicios [MBATS](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System).

Conclusion: a pesar de diferentes lenguajes y objetivos, **los patrones se repiten**.

## Arquitectura por capas: la base

La mayoria de los robots siguen una arquitectura de 5 capas:

```
Capa 5: Comunicacion    <- Telegram, Web UI, API
Capa 4: Estrategia      <- Logica de trading
Capa 3: Ejecucion y riesgo <- Ordenes, gestion de riesgo
Capa 2: Procesamiento   <- Indicadores, normalizacion
Capa 1: Ingesta de datos <- Conexiones a bolsas
```

## Patrones de diseno

### 1. Event Sourcing
Guarda cada cambio de estado como evento. Crucial para auditoria y debugging.

### 2. CQRS
Separa lecturas de escrituras.

### 3. Microservicios
MBATS divide el sistema en servicios independientes comunicandose via Kafka/RabbitMQ.

### 4. Actor Model
Cada actor tiene estado independiente, se comunica por mensajes. Sin estado compartido = sin condiciones de carrera.

## Checklist de diseno

1. **Empiece con un monolito**
2. **Separe capas desde el dia uno**
3. **Disene para pruebas** — inyeccion de dependencias
4. **Agregue observabilidad desde el dia uno**
5. **Planifique la persistencia**
6. **Microservicios solo cuando:** >100 instrumentos, HFT, equipo >3 personas

La arquitectura determina hasta donde llegaras. El codigo espagueti funciona un mes. La arquitectura correcta — anos.

---

**Enlaces utiles:**

- [Freqtrade](https://github.com/freqtrade/freqtrade)
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader)
- [Hummingbot](https://github.com/hummingbot/hummingbot)
- [MBATS](https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System)
