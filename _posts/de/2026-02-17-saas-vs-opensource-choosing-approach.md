---
layout: post
title: "SaaS vs. Open-Source: Wann es sich lohnt, eine eigene Algotrading-Engine zu bauen"
description: "Detaillierte Analyse der Gesamtbetriebskosten, versteckter Ausgaben und des Break-Even-Punkts. Wann TSLab guenstiger ist als der eigene Stack, und wann umgekehrt."
date: 2026-02-17
image: /assets/images/blog/saas_vs_opensource.png
tags: [SaaS, open-source, TCO, infrastructure, platform choice]
lang: de
---

"Eigenen Stack bauen oder fuer fertiges SaaS zahlen?"

Vor einem Jahr wechselte ich von TSLab (60k/Jahr) zu meinem eigenen Stack (Python + Backtrader + Docker). Dachte, ich wuerde sparen. Stellt sich heraus — so einfach ist das nicht.

Im letzten Jahr berechnete ich die **echten** Gesamtbetriebskosten (TCO) beider Ansaetze.

## Die Illusion des kostenlosen Open-Source

**Mythos:** Open-Source ist kostenlos. **Realitaet:** Die Software ist kostenlos. Zeit, Infrastruktur, Support — nicht.

**2024:** TSLab + MOEX AlgoPack + VPS = **127k/Jahr.**

**2025 (Realitaet):** Open-Source-Stack: Infrastruktur 54k/Jahr + Zeit = **erstes Jahr 534k.**

Open-Source ist im ersten Jahr **4,4x teurer**.

## Wann SaaS guenstiger ist

1. Sie sind kein Programmierer
2. Einfache Strategien
3. Ideentestphase
4. Kapital <5M Rubel

## Wann Open-Source guenstiger ist

1. Sie sind Programmierer
2. Komplexe Strategien (ML, Arbitrage)
3. Kapital >10M Rubel
4. Skalierung (5+ Nutzer)
5. HFT

## Checkliste: SaaS oder Open-Source?

1. **Programmierer?** Ja -> Open-Source. Nein -> SaaS.
2. **Einfache Strategie?** Ja -> SaaS. Nein -> Open-Source.
3. **Kapital?** <5M -> SaaS. >10M -> Open-Source.
4. **Zeit ist Geld?** Ja -> SaaS. Nein -> Open-Source.

## Meine Empfehlung

Wenn Sie Anfaenger sind, **beginnen Sie mit SaaS**. Nach 6-12 Monaten, wenn Sie an Plattformgrenzen stossen, wechseln Sie zu Open-Source.

**Berechnen Sie den TCO ehrlich.** Beruecksichtigen Sie die Zeit.

---

**Nuetzliche Links:**

- [QuantConnect Pricing](https://www.quantconnect.com/pricing/)
- [Backtrader](https://www.backtrader.com/)
- [LEAN](https://github.com/QuantConnect/Lean)
- [StockSharp](https://github.com/StockSharp/StockSharp)
