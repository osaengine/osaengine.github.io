---
layout: post
title: "Warum Ihr im Baukasten erstellter Roboter im Live-Handel verliert: 5 Debugging-Fehler, die niemand anspricht"
description: "Der Backtest zeigte +300% Jahresrendite. Im Live-Handel minus 15% in einem Monat. Wir analysieren typische Fallstricke beim Debugging visueller Strategien und wie man sie vermeidet."
date: 2025-12-23
image: /assets/images/blog/debug_visual_strategies.png
tags: [debugging, backtesting, mistakes, visual builders, testing]
lang: de
---

Vor zwei Wochen erhielt ich eine Nachricht von einem Leser. Er hatte eine Strategie in TSLab zusammengebaut. Ein Backtest ueber drei Jahre Historiedaten zeigte fantastische Ergebnisse: +280% Jahresrendite, maximaler Drawdown 8%.

Er setzte die Strategie auf ein Demokonto. Nach einem Monat das Ergebnis: minus 12%.

Was lief schief? Das Problem war nicht der Baukasten. Das Problem war, **wie er getestet hat**.

Das ist eine klassische Geschichte. Visuelle Baukastensysteme machen den Zusammenbau einer Strategie einfach. Aber sie machen sie nicht **korrekt**. Und die meisten Fehler passieren nicht beim Zusammenbau, sondern beim Testen.

In diesem Artikel: fuenf Fallstricke, in die 90% der Baukasten-Anfaenger tappen. Und wie man sie vermeidet.

## Fehler #1: Ueberoptimierung (Curve Fitting)

**Was ist das:**

Sie nehmen eine Strategie, fuehren Parameteroptimierung durch. Testen SMA von 10 bis 100 mit Schritt 1. Testen RSI von 20 bis 80 mit Schritt 5. Finden die Kombination mit dem besten historischen Ergebnis.

Gratulation: Sie haben gerade eine Strategie erstellt, die **nur** in diesem spezifischen historischen Zeitraum funktioniert.

**Warum es gefaehrlich ist:**

[Curve Fitting bedeutet, dass sich eine Strategie so stark an historische Daten anpasst](https://www.quantifiedstrategies.com/curve-fitting-trading/), dass sie bei neuen Daten nicht mehr funktioniert. Sie haben kein Marktmuster gefunden, sondern zufaelliges Rauschen.

**Reales Beispiel:**

SMA-Cross-Optimierung auf Daten 2020-2023. Bestes Ergebnis: SMA(37) und SMA(83). Rendite +180% pro Jahr.

Ausfuehrung in 2024: minus 5%.

Warum? Weil die Kombination 37/83 keine logische Grundlage hat. Es ist Anpassung an Rauschen.

**Wie man es erkennt:**

- Zu viele Parameter (mehr als 3-4)
- Perfekte historische Ergebnisse (200%+ jaehrlich ohne Drawdowns)
- [Parameter wirken zufaellig](https://www.buildalpha.com/3-simple-ways-to-reduce-the-risk-of-curve-fitting/) (37, 83 statt runder Zahlen wie 20, 50)
- Ergebnisse fallen stark ab, wenn ein Parameter um 1-2 Einheiten geaendert wird

**Wie man es vermeidet:**

### 1. Begrenzen Sie die Parameteranzahl

[Verwenden Sie fuer klassisches Testen maximal 2 optimierbare Parameter](https://empirix.ru/pereoptimizacziya-strategij/). Je weniger, desto besser.

Eine einfache Strategie lebt laenger. Eine komplexe stirbt schnell.

### 2. Out-of-Sample-Tests

Teilen Sie die Historie in zwei Teile:
- **In-Sample** (70%): Parameteroptimierung
- **Out-of-Sample** (30%): Ergebnisverifikation

Wenn Out-of-Sample-Ergebnisse deutlich schlechter sind: Ueberoptimierung.

In TSLab: Optimieren Sie auf 2020-2022, testen Sie auf 2023.

In Designer: Gleiche Logik, aendern Sie manuell den Zeitraum.

### 3. Walk-Forward-Analyse

Noch zuverlaessiger: [Fuehren Sie ein gleitendes Fenster durch](https://eatradingacademy.com/help/strategy-builders/expert-advisor-studio/strategy-tools-optimization/walk-forward-testing/).

Beispiel:
- Optimierung auf 2020-2021, Test auf 2022
- Optimierung auf 2021-2022, Test auf 2023
- Optimierung auf 2022-2023, Test auf 2024

Wenn die Strategie in allen Zeitraeumen haelt, ist sie robust.

### 4. Pruefen Sie die Parameterstabilitaet

Erstellen Sie eine Heatmap der Optimierungsergebnisse.

Wenn das beste Ergebnis ein einzelner "Hot Spot" in einem Meer aus Rot ist: Ueberoptimierung.

Wenn es ein breites "Plateau" guter Ergebnisse gibt: Die Strategie ist stabil gegenueber Parameteraenderungen. Das ist gut.

TSLab und NinjaTrader zeigen 3D-Optimierungsdiagramme. Nutzen Sie sie.

## Fehler #2: Look-Ahead Bias (Vorausschau-Verzerrung)

**Was ist das:**

Ihre Strategie verwendet versehentlich Informationen, die zum Zeitpunkt der Entscheidungsfindung **noch nicht verfuegbar** waren.

**Klassisches Beispiel:**

Sie verwenden einen Indikator auf dem **Schlusskurs** einer Kerze, aber das Signal wird bei der **Eroeffnung** der naechsten generiert.

Problem: Wenn eine Kerze schliesst, kennen Sie bereits High/Low/Close. Im realen Handel nicht.

**Wo es vorkommt:**

### In TSLab:

[TSLab zaehlt die Kerzenzeit als Startzeit](https://doc.tslab.pro/tslab/rabota-s-programmoi/torgovlya-agentami-robotami/rabota-agenta-i-osobye-situacii). Wenn Sie das nicht beruecksichtigen, ist ein Look-Ahead leicht gemacht.

Beispiel: Der Block "Schlusskurs" bei Kerze N gibt einen Wert zurueck, der erst **nach** dem Schluss dieser Kerze bekannt sein wird.

Wenn Sie ein Signal basierend auf Close[0] generieren: Look-Ahead. Verwenden Sie Close[1].

### In Designer:

Dasselbe. Designer arbeitet mit geschlossenen Kerzen. Wenn Ihre Logik auf der aktuellen Kerze basiert, pruefen Sie, ob diese Daten in Echtzeit verfuegbar sind.

### In NinjaTrader:

Strategy Builder hat die Option "Calculate on bar close". Wenn deaktiviert: Signale werden bei jedem Tick generiert, einschliesslich ungeschlossener Kerzen. Wenn aktiviert: nur beim Schluss.

Fuer die meisten Strategien brauchen Sie "Calculate on bar close = true".

**Wie man es vermeidet:**

1. **Verwenden Sie nur geschlossene Kerzen**
   - Bei H1-Strategie erscheint das Signal nur nach dem Schluss der Stundenkerze
   - Verwenden Sie keine Daten der aktuellen Kerze fuer die Signalgenerierung

2. **Pruefen Sie Datenverzoegerungen**
   - Makrooekonomische Daten werden mit Verzoegerung veroeffentlicht
   - Nachrichten erscheinen nicht sofort
   - [Finanzberichte werden ueberarbeitet](https://analystprep.com/study-notes/cfa-level-2/problems-in-backtesting/)

3. **Laufen Sie auf Demo vor dem Live-Test**
   - Wenn der Backtest 100 Trades pro Monat zeigt und das Demo 10: Look-Ahead-Problem

## Fehler #3: Survivorship Bias (Ueberlebensverzerrung)

**Was ist das:**

Sie testen eine Strategie auf Aktien, die **heute existieren**. Aber ueber drei Jahre sind einige Unternehmen bankrottgegangen, dekotiert oder uebernommen worden.

Sie sind nicht in Ihrem Backtest. Aber im realen Handel waren sie da.

**Reales Beispiel:**

Strategie auf russische Aktien. Backtest 2020-2023. Die Liste der getesteten Aktien umfasst:
- Sberbank ✅
- Gazprom ✅
- Yandex ✅
- TCS Holding ✅

Aber es fehlen:
- Rusal (dekotiert 2022) ❌
- Moscow Exchange (voruebergehende Dekotierung 2022) ❌
- Aktien, die 90% gefallen und vom Radar verschwunden sind ❌

Ihre Strategie hat die Verluste bei diesen Instrumenten "vergessen". [Survivorship Bias ueberhoecht die Rendite um 1-4% pro Jahr](https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained/).

**Wo es vorkommt:**

### In TSLab und Designer:

Wenn Sie Aktienlisten ueber eine Brokerverbindung laden, erhalten Sie nur **aktuelle** Aktien. Dekotierte sind nicht dabei.

### In NinjaTrader:

Gleiches Problem mit Futures. Abgelaufene Kontrakte fehlen oft im Backtest.

**Wie man es vermeidet:**

1. **Verwenden Sie Datenbanken mit dekotierten Wertpapieren**
   - [QuantConnect, Norgate Data](https://medium.com/auquan/backtesting-biases-and-how-to-avoid-them-776180378335) bieten survivorship-bias-freie Daten
   - Fuer den russischen Markt schwieriger, solche Datenbanken sind selten

2. **Testen Sie am Index, nicht an ausgewaehlten Aktien**
   - Bei MOEX-Aktienstrategie: Nehmen Sie den gesamten MOEX-Index, nicht nur die Top 10

3. **Pruefen Sie, wie viele Wertpapiere im Testzeitraum verschwanden**
   - Wenn Sie 3 Jahre testen und die Aktienliste unveraendert blieb: Problem

4. **Fuegen Sie Liquiditaetsfilter hinzu**
   - Strategie sollte keine Aktien mit Tagesumsatz unter 10 Mio. Rubel handeln
   - Das reduziert das Risiko, in Aktien vor der Dekotierung einzusteigen

## Fehler #4: Ignorieren von Gebuehren, Slippage und Ausfuehrungsrealitaeten

**Was ist das:**

Backtests nehmen an: Sie kaufen immer zum gewuenschten Preis. Auftraege werden sofort ausgefuehrt. Kommission = 0.

Realitaet: Kommissionen, Slippage, Verzoegerungen, Teilausfuehrung.

**Reales Beispiel:**

Strategie auf Minutenkerzen. 200 Trades pro Monat. Durchschnittsgewinn pro Trade: 0,15%.

Brokerkommission: 0,05% Eintritt, 0,05% Austritt. Gesamt 0,1% Roundtrip.

**Nettogewinn:** 0,15% - 0,1% = 0,05% pro Trade.

200 Trades * 0,05% = 10% pro Monat. Klingt gut.

Aber fuegen Sie Slippage von 0,03% pro Trade hinzu. Jetzt: 0,15% - 0,1% - 0,03% = **0,02%**.

200 Trades * 0,02% = **4% pro Monat**. Nicht mehr so beeindruckend.

Und wenn der Spread breit ist (illiquide Aktie), Slippage 0,1%? Die Strategie ist **unprofitabel**.

**Wie man es vermeidet:**

### 1. Konfigurieren Sie Kommissionen im Baukasten

**TSLab:**
Einstellungen → Handel → Kommissionen. Geben Sie die tatsaechlichen Brokerkommissionen ein (typischerweise 0,03-0,05%).

**Designer:**
Das Backtest-Fenster hat ein Feld "Kommission". Stellen Sie es in absoluten Werten oder Prozenten ein.

**NinjaTrader:**
Strategy → Properties → Commission. Geben Sie Kommission pro Kontrakt ein.

**fxDreema:**
Im generierten MQL-Code muessen Sie Spread-Pruefungen manuell hinzufuegen.

### 2. Fuegen Sie Slippage hinzu

TSLab und NinjaTrader erlauben separate Slippage-Konfiguration. Fuer Retail-Trader bei liquiden Aktien: 1-3 Ticks.

Fuer illiquide: 5-10 Ticks oder mehr.

### 3. Testen Sie mit realem Spread

Wenn die Strategie innerhalb des Spreads handelt (Scalping): Pruefen Sie, ob der Gewinn den Spread abdeckt.

Einfache Formel:
```
Gewinn pro Trade > Kommission * 2 + Durchschnittlicher Spread + Slippage
```

Wenn nicht: Die Strategie ueberlebt den Live-Handel nicht.

### 4. Pruefen Sie die Anzahl der Trades

[Je mehr Trades, desto staerker der Einfluss der Kommissionen](https://www.quantifiedstrategies.com/survivorship-bias-in-backtesting/).

100 Trades pro Jahr: Kommissionen nicht kritisch.

1000 Trades pro Jahr: Kommissionen koennen den gesamten Gewinn auffressen.

**Regel:** Wenn die Strategie nach Kommissionen weniger als 0,5% pro Trade bringt, ist sie am Limit. Die geringste Marktverschlechterung wird sie vernichten.

## Fehler #5: Kein Forward Testing

**Was ist das:**

Ein Backtest testet die Vergangenheit. Ein Forward Test testet die Zukunft (aber ohne echtes Geld).

[Forward Testing zeigt, wie eine Strategie mit Daten funktioniert, die sie nie gesehen hat](https://academy.ftmo.com/lesson/forward-testing-of-trading-strategies/).

**Warum es wichtig ist:**

Angenommen, Sie haben eine Strategie auf 2020-2023 optimiert. Ergebnisse sind hervorragend. Sie starten live in 2024.

Problem: Der Markt in 2024 verhaelt sich moeglicherweise anders. Volatilitaet hat sich geaendert. Korrelationen sind gebrochen.

Forward Testing auf einem Demokonto ermoeglicht die Ueberpruefung **bevor** Sie Geld verlieren.

**Wie man Forward Testing durchfuehrt:**

### 1. Auf Demokonto laufen lassen

**Mindestdauer:** [3-6 Monate](https://academy.ftmo.com/lesson/forward-testing-of-trading-strategies/).

Warum so lange? Weil:
- Verschiedene Marktregime erfasst werden muessen (Trend, Range, Volatilitaet)
- Mindestens 50-100 Trades benoetigt werden
- Psychologische Belastbarkeit geprueft werden muss (ja, auch auf Demo)

### 2. Fuehren Sie ein Handelsjournal

Erfassen Sie:
- Ein-/Ausstieg
- Handelsgrund (welcher Block das Signal generiert hat)
- Abweichung vom Backtest (falls vorhanden)

Wenn Demo-Ergebnisse **deutlich** schlechter als der Backtest sind: Etwas ist kaputt. Zurueck zum Debugging.

### 3. Vergleichen Sie Metriken

| Metrik | Backtest | Forward Test |
|--------|----------|--------------|
| Win Rate | 65% | ? |
| Durchschn. Gewinn | 1,2% | ? |
| Durchschn. Verlust | -0,8% | ? |
| Max. Drawdown | 12% | ? |
| Trades/Monat | 20 | ? |

Wenn die Abweichung 20-30% uebersteigt: Problem.

### 4. Nutzen Sie Paper Trading auf Plattformen

**TradingView:** [Kostenloses Paper Trading](https://wundertrading.com/journal/en/learn/article/paper-trading-tradingview) ueber virtuelles Konto.

**AlgoTest:** [Paper Trading mit detaillierten Analysen](https://docs.algotest.in/strategy-builder/paper-trading-analysing/).

**TSLab/Designer:** Simulation mit realer Brokerverbindung (aber ohne Orderversand).

### 5. Nicht hetzen

Der haeufigste Fehler: Eine Woche auf Demo testen, Gewinn sehen, live deployen.

Eine Woche ist nichts. Sie brauchen mindestens 2-3 Monate, um zu verstehen, wie sich die Strategie unter verschiedenen Bedingungen verhaelt.

## Checkliste vor dem Live-Start einer Strategie

Bevor Sie "Start" auf dem Live-Konto druecken, gehen Sie diese Liste durch:

### Tests

- [ ] Strategie auf mindestens 2 Jahren Historie getestet
- [ ] Out-of-Sample-Test durchgefuehrt (30% der Historie)
- [ ] Parameteranzahl ≤ 3
- [ ] Parameter logisch begruendet (keine Rauschanpassung)
- [ ] Ergebnisse stabil bei Parameteraenderung um ±10%

### Verzerrungen

- [ ] Keine Look-Ahead-Bias verifiziert (nur geschlossene Kerzen)
- [ ] Survivorship Bias beruecksichtigt (oder durch Filter minimiert)
- [ ] Realistische Kommissionen hinzugefuegt (0,03-0,05%)
- [ ] Slippage hinzugefuegt (1-3 Ticks fuer liquide Instrumente)
- [ ] Strategie nach Kommissionen und Slippage profitabel

### Forward Testing

- [ ] Strategie auf Demokonto mindestens 3 Monate getestet
- [ ] Mindestens 50 Trades gesammelt
- [ ] Demo-Ergebnisse nahe am Backtest (Abweichung <30%)
- [ ] Handelsjournal gefuehrt
- [ ] In verschiedenen Marktregimen getestet (Trend, Range, Volatilitaet)

### Risikomanagement

- [ ] Maximales Risiko pro Trade ≤ 2% des Kontos
- [ ] Maximaler Backtest-Drawdown ≤ 20%
- [ ] Aktionsplan fuer Drawdown >15% vorhanden
- [ ] Positionsgroesse basierend auf Instrumentenvolatilitaet berechnet

Wenn auch nur ein Punkt nicht erfuellt ist: Nicht live gehen.

## Debugging-Tools in Baukastensystemen

### TSLab

**Vorteile:**
- Integrierter Debugger mit schrittweiser Ausfuehrung
- Trade-Visualisierung auf Charts
- Detaillierter Bericht pro Trade
- [3D-Optimierungsvisualisierung](https://vc.ru/u/715109-tslab/204062-optimizaciya-mehanicheskih-torgovyh-sistem)

**Nachteile:**
- [Kein automatischer Out-of-Sample-Test](http://forum.tslab.ru/ubb/ubbthreads.php?ubb=showflat&Number=86791)
- Probleme mit Tickdaten

### StockSharp Designer

**Vorteile:**
- Flexible Kommissions- und Slippage-Einstellungen
- Unterstuetzung von Tick- und Orderbuchdaten
- Export nach C# fuer tiefes Debugging

**Nachteile:**
- Weniger Debugging-Dokumentation
- Visualisierung schwaecher als TSLab

### NinjaTrader Strategy Builder

**Vorteile:**
- Visual Studio-Integration fuer Code-Debugging
- Detaillierte Ausfuehrungsprotokolle
- Market Replay fuer schrittweises Testen

**Nachteile:**
- Fuer Anfaenger schwerer einzurichten
- Teuer ($1.500 fuer Lifetime)

### fxDreema

**Vorteile:**
- Generiert MQL-Code, der im MetaEditor debuggt werden kann
- Visueller MetaTrader-Tester

**Nachteile:**
- Einschraenkungen der kostenlosen Version (10 Verbindungen zwischen Bloecken)
- MQL-Kenntnisse fuer tiefes Debugging erforderlich

## Fazit

Visuelle Baukastensysteme machen die Strategieerstellung einfach. Aber das Debugging bleibt schwierig.

**Fuenf Hauptfehler:**

1. **Ueberoptimierung** — Anpassung an historisches Rauschen
2. **Look-Ahead Bias** — Verwendung zukuenftiger Daten
3. **Survivorship Bias** — Ignorieren dekotierter Wertpapiere
4. **Ignorieren von Kommissionen** — unrealistische Ausfuehrungsannahmen
5. **Kein Forward Testing** — Live-Start ohne Demo-Verifikation

**Was zu tun ist:**

- Parameter begrenzen (≤3)
- Out-of-Sample-Tests durchfuehren
- Auf Look-Ahead-Bias pruefen
- Realistische Kommissionen und Slippage hinzufuegen
- Mindestens 3 Monate auf Demo testen

[Ein korrekter Backtest](https://www.morpher.com/ru/blog/backtesting-trading-strategies) dreht sich nicht um huebsche Renditegrafiken. Es geht um eine ehrliche Antwort auf die Frage: "Wird das live funktionieren?"

Wenn ein Backtest 300% jaehrliche Rendite zeigt, ist hoechstwahrscheinlich irgendwo ein Fehler. Realistische Renditen fuer Retail-Algotrading: 20-50% jaehrlich bei 10-20% Drawdown.

Wenn Ihre Ergebnisse deutlich besser sind, gehen Sie die obigen Punkte nochmals durch. Sie haben etwas uebersehen.

---

**Nuetzliche Links:**

Forschung und Ressourcen:
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
