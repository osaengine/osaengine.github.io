---
layout: post
title: "SaaS vs open-source: cuando tiene sentido construir tu propio motor de algotrading"
description: "Analisis detallado del costo total de propiedad, gastos ocultos y punto de equilibrio. Cuando TSLab es mas barato que tu propio stack, y cuando es al reves."
date: 2026-02-17
image: /assets/images/blog/saas_vs_opensource.png
tags: [SaaS, open-source, TCO, infrastructure, platform choice]
lang: es
---

"Construir tu propio stack o pagar por un SaaS listo?"

Hace un ano cambie de TSLab (60k/ano) a mi propio stack (Python + Backtrader + Docker). Pense que ahorraria. Resulta que no es tan simple.

El ultimo ano calcule el **verdadero** costo total de propiedad (TCO) de ambos enfoques.

## La ilusion de lo gratuito en open-source

**Mito:** Open-source es gratis. **Realidad:** El software es gratis. Tiempo, infraestructura, soporte — no lo son.

**2024:** TSLab + MOEX AlgoPack + VPS = **127k/ano.**

**2025 (realidad):** Stack open-source: infraestructura 54k/ano + tiempo = **primer ano 534k.**

Open-source es **4.4x mas caro** el primer ano.

## Cuando SaaS es mas barato

1. No eres programador
2. Estrategias simples
3. Etapa de prueba de ideas
4. Capital <5M rublos

## Cuando open-source es mas barato

1. Eres programador
2. Estrategias complejas (ML, arbitraje)
3. Capital >10M rublos
4. Escala (5+ usuarios)
5. HFT

## Checklist: SaaS o open-source?

1. **Eres programador?** Si -> Open-source. No -> SaaS.
2. **Estrategia simple?** Si -> SaaS. No -> Open-source.
3. **Capital?** <5M -> SaaS. >10M -> Open-source.
4. **El tiempo es dinero?** Si -> SaaS. No -> Open-source.

## Mi recomendacion

Si eres principiante, **empieza con SaaS**. Despues de 6-12 meses, cuando choques con los limites de la plataforma, pasa a open-source.

**Calcula el TCO honestamente.** Incluye el tiempo.

---

**Enlaces utiles:**

- [QuantConnect Pricing](https://www.quantconnect.com/pricing/)
- [Backtrader](https://www.backtrader.com/)
- [LEAN](https://github.com/QuantConnect/Lean)
- [StockSharp](https://github.com/StockSharp/StockSharp)
