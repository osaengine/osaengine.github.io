---
layout: post
title: "Donde termina la ayuda de la IA y comienza la autodestrucción del depósito: riesgos de la caja negra"
description: "Los traders con IA pierden millones, los reguladores dan la alarma y el 85% de los traders no confía en sistemas de caja negra. Analizamos casos reales de fracasos, flash crashes y por qué la explicabilidad importa más que la rentabilidad."
date: 2026-03-24
image: /assets/images/blog/ai_black_box_risks.png
tags: [AI, risks, black box, explainability, regulation, flash crash]
lang: es
---

Hace una semana [mostré cómo un LLM puede ayudar a un quant]({{site.baseurl}}/2026/03/17/mozhet-li-llm-zamenit-kvant-analitika.html). Creamos una estrategia con +9,84%, Sharpe 0,52. Todo funciona.

Pero hay un lado oscuro. **Los traders con IA están perdiendo millones.** No porque los modelos sean malos. Sino porque **nadie entiende por qué hacen lo que hacen**.

En 2023, un importante hedge fund perdió **50 millones de dólares en un solo día** cuando su IA de caja negra comenzó a realizar "unexplained trades" durante la volatilidad. [La causa aún no se ha encontrado](https://ts2.tech/en/black-box-ai-exposed-hidden-algorithms-risks-and-breakthroughs-in-2025/).

Entre 2019 y 2025, [la CFTC documentó decenas de casos](https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/AITradingBots.html) en los que "bots de IA" prometían "above-average returns", pero los clientes perdieron **1.700 millones de dólares** (30.000 BTC).

Hoy analizamos: **dónde exactamente la ayuda de la IA se convierte en catástrofe**, qué riesgos conlleva el trading de caja negra y por qué [el 85% de los traders no confía en la IA](https://www.ampfi.app/blog/transparent-ai-vs-black-box-trading-systems).

## Qué es una "caja negra" en el trading con IA

**Black box AI** es un sistema que toma decisiones pero **no explica por qué**.

### **Ejemplo de algoritmo clásico (caja blanca):**

```python
def should_buy(price, sma_50, sma_200):
    if sma_50 > sma_200 and price < sma_50 * 0.98:
        return True  # Golden cross + pullback
    return False
```

**Claro:**
- Si la MA a corto plazo > a largo plazo (tendencia alcista)
- Y el precio retrocedió un 2% por debajo de la MA a corto plazo (punto de entrada)
- Comprar

Se puede explicar al cliente, al regulador y a uno mismo.

### **Ejemplo de IA caja negra:**

```python
model = NeuralNetwork(layers=[128, 64, 32, 1])
model.train(historical_data)

def should_buy(market_data):
    prediction = model.predict(market_data)
    return prediction > 0.5  # Buy if model says "yes"
```

**No está claro:**
- ¿Por qué el modelo dijo "sí"?
- ¿Qué features utilizó?
- ¿Qué ocurrirá si el mercado cambia?

**El problema:** Una red neuronal con millones de parámetros es una [caja negra](https://www.voiceflow.com/blog/blackbox-ai). Ves la entrada (datos) y la salida (decisión), pero **no ves la lógica**.

### **Por qué esto es crítico en el trading:**

1. **Hay dinero en juego** — los errores cuestan dinero real
2. **Regulación** — los reguladores exigen explicaciones (SEC, FCA, ESMA)
3. **Gestión de riesgos** — no se puede gestionar lo que no se entiende
4. **Confianza** — los clientes no entregarán dinero basándose en "porque la IA lo dijo"

## Casos reales: cuando los traders con IA perdieron millones

### **Caso 1: Hedge fund, $50M en un día (2023)**

[Historia](https://ts2.tech/en/black-box-ai-exposed-hidden-algorithms-risks-and-breakthroughs-in-2025/):

**Qué ocurrió:**

- Un importante hedge fund usaba IA propietaria para operar con acciones
- La IA operaba de forma autónoma, sin confirmación humana
- El 15 de marzo de 2023, durante un pico de volatilidad (colapso de SVB), la IA comenzó a realizar "unexplained trades"
- En 4 horas realizó 1.247 operaciones (normalmente ~50 al día)
- Resultado: **-$50M** (-8% AUM)

**Por qué ocurrió:**

La IA vio un patrón que interpretó como una "oportunidad de arbitraje". Pero en realidad era **ruido de microestructura de mercado** (rebote bid-ask + liquidez escasa).

**Por qué no se detuvo:**

El algoritmo operaba tan rápido que cuando los gestores de riesgos lo notaron, era demasiado tarde. El kill switch existía, pero se activó solo después de 3,5 horas (cadena de aprobación manual).

**Lección:**

Una caja negra sin **explicabilidad en tiempo real** = una bomba de relojería.

### **Caso 2: CFTC vs bots de trading con IA — $1.700M en pérdidas (2019-2025)**

[La CFTC emitió una advertencia](https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/AITradingBots.html):

**El esquema:**

- Empresas venden "bots de trading con IA" prometiendo "máquinas automatizadas de hacer dinero"
- Prometen rendimientos del 10-30% mensual
- Toman el dinero de los clientes bajo gestión o venden software

**Resultados:**

- Los clientes perdieron **1.700 millones de dólares** (incluyendo 30.000 BTC)
- La mayoría de las "IA" resultaron ser scripts simples o esquemas Ponzi
- Ningún sistema reveló su lógica de trading ("IA propietaria")

**Caso típico:**

La empresa X prometía "IA de deep learning entrenada con 10 años de datos". Un cliente depositó $100.000. Después de 6 meses, saldo: $23.000. Pidió una explicación. Respuesta: "Market conditions changed, AI adapting". 3 meses más: saldo $5.000. La empresa X desapareció.

**Lección:**

Si la IA no explica sus decisiones, es una **señal de alarma**. O es un fraude, o los desarrolladores no entienden qué hace su sistema.

### **Caso 3: Flash Crash de 2010 — $1 billón en 36 minutos**

[6 de mayo de 2010](https://en.wikipedia.org/wiki/2010_flash_crash):

**Qué ocurrió:**

- 14:32 EDT: El Dow Jones comenzó a caer
- En 5 minutos cayó **998,5 puntos** (9%)
- Acciones individuales cotizaron a $0,01 (caída de casi el 100%)
- En 36 minutos el mercado se recuperó
- Capital total "evaporado": **$1 billón**

**La causa:**

[La investigación de la SEC mostró](https://pmc.ncbi.nlm.nih.gov/articles/PMC8978471/):

1. Un gran operador institucional colocó una orden de venta por $4.100M a través de un algoritmo
2. Los algoritmos HFT comenzaron a operar entre ellos (hot potato)
3. La liquidez se evaporó instantáneamente
4. Los algoritmos comenzaron a "vender agresivamente" para salir de posiciones
5. Efecto cascada

**Cita de la SEC:**

> "In the absence of appropriate controls, the speed with which automated trading systems enter orders can turn a manageable error into an extreme event with widespread impact."

**Lección:**

Los algoritmos interactúan de forma impredecible. **Un algoritmo + miles de otros = riesgo sistémico**.

### **Caso 4: Knight Capital — $440M en 45 minutos (2012)**

[1 de agosto de 2012](https://www.luxalgo.com/blog/lessons-from-algo-trading-failures/):

**Qué ocurrió:**

- Knight Capital desplegó un nuevo software de trading
- Debido a un bug, el algoritmo comenzó a enviar **millones de órdenes**
- En 45 minutos ejecutó operaciones por $7.000 millones
- Resultado: **-$440M** (más que los ingresos anuales)
- La empresa quebró

**La causa:**

El código antiguo no fue eliminado. El nuevo algoritmo activó accidentalmente la lógica antigua. La lógica antigua estaba diseñada para testing, no para producción.

**Lección:**

**El código no es IA**, pero el principio es el mismo: automatización sin control = catástrofe.

## Por qué el 85% de los traders no confía en la IA de caja negra

[Un estudio de 2025](https://www.ampfi.app/blog/transparent-ai-vs-black-box-trading-systems) mostró:

**Desconfianza en la IA de caja negra:**
- El 85% de los traders no confía en sistemas sin explicaciones
- El 62% prefiere modelos más simples con transparencia
- El 78% exige un "humano en el bucle" para las decisiones finales

**Razones de la desconfianza:**

### **1. Imposibilidad de explicar las pérdidas**

**Escenario:**

Tu robot de IA opera durante 3 meses. Resultado: +15%. ¡Excelente!

Mes 4: -25%. ¿Qué pasó?

Le preguntas a la IA (si es posible). Respuesta (si la hay): "Market regime changed."

Tú: "¿Qué régimen exactamente? ¿Qué cambió?"

IA: "..."

**El problema:** No puedes saber si es un **drawdown temporal** (aguantar) o un **fallo fundamental** (la estrategia ya no funciona).

### **2. Requisitos regulatorios**

[EU AI Act (2025)](https://www.finance-watch.org/wp-content/uploads/2025/03/Artificial_intelligence_in_finance_report_final.pdf) y la SEC exigen:

- Transparencia en "sistemas de IA de alto riesgo" (incluido el trading)
- Capacidad de explicar decisiones
- Supervisión humana

**Cita del EU AI Act:**

> "High-risk AI systems shall be designed in such a way to ensure transparency and enable users to interpret the system's output and use it appropriately."

**El problema:**

Si tu IA es una caja negra, estás **infringiendo la regulación**. Multas de hasta **35 millones de euros o el 7% de los ingresos globales**.

### **3. Imposibilidad de depurar**

**Algoritmo clásico:**

```python
# La estrategia pierde dinero. Depuración:
print(f"SMA crossover signals: {signals}")
print(f"Entry prices: {entries}")
print(f"Stop losses hit: {stops_hit}")
# Veo el problema: los stops son demasiado ajustados
```

**IA caja negra:**

```python
# La estrategia pierde dinero. Depuración:
print(model.weights)  # [0.234, -0.891, 0.445, ... 10,000 números]
# ???
# ¿Qué significa esto? ¿Qué peso es responsable de qué?
```

**No puedes mejorar lo que no entiendes.**

### **4. Psicología: miedo a perder el control**

[La investigación muestra](https://www.pymnts.com/artificial-intelligence-2/2025/black-box-ai-what-it-is-and-why-it-matters-to-businesses/):

Las personas prefieren el **control** sobre la **optimalidad**.

**Experimento:**

- Grupo A: Usa IA de caja negra con Sharpe 1,5
- Grupo B: Usa una estrategia simple con Sharpe 1,0, pero entiende la lógica

**Resultado:**

- El 72% prefirió el Grupo B
- Razón: "I trust what I understand"

**Cita de un participante:**

> "I'd rather make 10% and sleep well, than make 15% and wake up wondering if AI will blow up my account tomorrow."

## Tipos de riesgos en el trading de caja negra

### **Riesgo 1: Sobreajuste (el asesino de estrategias número 1)**

**Qué es:**

El modelo se ajustó perfectamente a los datos históricos, pero **no funciona con datos nuevos**.

**Ejemplo:**

Una red neuronal entrenada en 2020-2023 (mercado alcista). Ve un patrón: "cuando Bitcoin sube 5 días seguidos, el día 6 la subida continúa en el 80% de los casos."

2024: mercado bajista. El patrón no funciona. El modelo sigue comprando el día 6 de subida. Resultado: pérdidas.

**Por qué es un problema de caja negra:**

Con un algoritmo clásico, puedes ver la regla y cambiarla. Con una red neuronal, no.

**Estadísticas:**

[La investigación muestra](https://digitaldefynd.com/IQ/ai-in-finance-case-studies/): el 60-70% de los modelos de ML en finanzas sufren de sobreajuste al desplegarse.

### **Riesgo 2: Concept Drift (el mercado cambia, el modelo no)**

**Qué es:**

Las propiedades estadísticas del mercado cambian; el modelo sigue operando con patrones antiguos.

**Ejemplos de concept drift:**

- **Crash del COVID en 2020:** Las correlaciones entre activos cambiaron
- **Subidas de tipos de la Fed en 2022:** Las estrategias de momentum dejaron de funcionar
- **Hype de IA en 2023:** Las acciones tech comenzaron a comportarse de manera diferente

**El problema:**

La caja negra no dice: "¡Atención! ¡Concept drift detectado!" Simplemente sigue perdiendo dinero.

### **Riesgo 3: Entradas adversarias**

**Qué es:**

Datos especialmente diseñados para engañar a la IA.

**Ejemplo en trading:**

Las firmas de HFT usan **spoofing** (colocan y cancelan órdenes grandes). Esto crea liquidez ficticia.

La IA de caja negra ve "gran demanda" y compra. El spoofer cancela las órdenes. La IA compró a un precio alto.

**Caso real:**

[La investigación mostró](https://www.theregreview.org/2025/11/25/smith-ai-and-the-future-of-market-manipulation/): los sistemas de IA son especialmente vulnerables a la manipulación del mercado porque **no entienden la intención** (demanda genuina vs. ficticia).

### **Riesgo 4: Fallos computacionales**

**Qué es:**

La IA requiere recursos computacionales. Si los recursos son insuficientes, las decisiones se retrasan.

**Ejemplos:**

- **Corte de internet:** Desconexión de API → la IA no ve datos → pierde señales de salida
- **Sobrecarga del servidor:** Durante la volatilidad, la carga aumenta → la latencia crece
- **Problemas del proveedor cloud:** AWS caído → tu IA caída

[Estadísticas](https://www.ccn.com/education/crypto/ai-crypto-trading-bots-how-they-make-and-lose-millions/): el 40% de los fallos de bots de IA están relacionados con **problemas de infraestructura**, no con los modelos.

### **Riesgo 5: Flash Crashes (riesgo sistémico)**

**Qué es:**

Múltiples sistemas de IA operando simultáneamente crean bucles de retroalimentación.

**Mecanismo:**

```
1. IA #1 ve una caída → vende
2. IA #2 ve la venta de IA #1 → vende
3. IA #3 ve la caída de #1 y #2 → vende
...
N. El precio se desplomó un 20% en un minuto
```

[La investigación muestra](https://journals.sagepub.com/doi/10.1177/03063127211048515): **14 micro-flash crashes ocurren diariamente** en exchanges de criptomonedas.

**Cita de la investigación:**

> "HFT provides liquidity in good times when least needed and takes it away when most needed, thereby contributing rather than mitigating instability."

## IA explicable (XAI): ¿solución o marketing?

### **Qué es XAI:**

[Explainable AI](https://kaliham.com/2025-guide-to-explainable-ai-in-forex-trading-clarity-compliance-confidence/) — métodos que hacen comprensibles las decisiones de la IA para los humanos.

**Métodos populares:**

### **1. SHAP (SHapley Additive exPlanations)**

**Idea:** Mostrar qué features contribuyen más a la decisión.

**Ejemplo:**

```python
import shap

# Modelo entrenado
model = RandomForest()
model.fit(X_train, y_train)

# Explicar predicción
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[0])

# Salida:
# RSI:         +0.15  (empuja hacia compra)
# Volume:      +0.08
# MA_cross:    +0.12
# Volatility:  -0.05  (empuja hacia venta)
# ...
# TOTAL:       +0.30  → BUY signal
```

**Ahora está claro:** El modelo compra principalmente por el RSI y el cruce de MA.

### **2. LIME (Local Interpretable Model-agnostic Explanations)**

**Idea:** Aproximar el modelo complejo con uno simple (lineal) **localmente**.

**Ejemplo:**

```python
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(X_train)
exp = explainer.explain_instance(X_test[0], model.predict_proba)

# Salida:
# IF RSI > 65 AND Volume > avg → -0.4 (sell signal)
# IF MA_short > MA_long → +0.6 (buy signal)
```

Se ve: localmente el modelo se parece a la regla "cruce de MA > RSI sobrecomprado".

### **3. Mecanismos de atención (para redes neuronales)**

**Idea:** La propia red neuronal muestra en qué "se fija" al tomar una decisión.

**Ejemplo (Transformer para series temporales):**

```
Model decision: BUY
Attention weights:
- Last 5 candles:    0.02 (ignorar)
- Candles 10-15:     0.35 (¡importante!)
- Candles 20-30:     0.15
- Volume spike:      0.40 (¡muy importante!)
```

**Interpretación:** El modelo compró por un pico de volumen hace 10 velas + un patrón de hace 10-15 velas.

### **¿Funciona XAI en la realidad?**

**Pros:**

- El [informe de McKinsey 2025](https://ts2.tech/en/black-box-ai-exposed-hidden-algorithms-risks-and-breakthroughs-in-2025/) llama a XAI un "strategic enabler" para la adopción de IA

- Los bancos que usan XAI mostraron **mayor confianza de los clientes**

- **Los costes de gestión de riesgo de modelos disminuyeron** (depuración más fácil)

**Contras:**

- Las explicaciones de XAI a veces son **engañosas** (muestran correlación, no causalidad)

- Los modelos complejos (redes neuronales profundas) siguen siendo **no totalmente explicables**

- XAI ralentiza la inferencia (sobrecarga computacional)

**Conclusión:**

XAI ayuda, pero **no resuelve el problema completamente**. Un modelo complejo seguirá siendo complejo.

## Regulación: qué exigen las autoridades

### **EU AI Act (2025)**

[Entró en vigor el 1 de agosto de 2024, con introducción gradual de requisitos](https://www.finance-watch.org/wp-content/uploads/2025/03/Artificial_intelligence_in_finance_report_final.pdf):

**Requisitos para "IA de alto riesgo" (incluido el trading):**

1. **Transparencia:** Los sistemas deben ser transparentes
2. **Supervisión humana:** Un humano debe poder intervenir
3. **Precisión:** Los sistemas deben ser fiables
4. **Robustez:** Protección contra ataques adversarios
5. **Documentación:** Documentación detallada de la lógica

**Multas:** Hasta 35 millones de euros o el 7% de los ingresos globales (lo que sea mayor).

**Qué significa esto:**

Si tu robot de IA es una caja negra, estás **infringiendo la ley** en la UE.

### **SEC (EE.UU.)**

[La SEC ha iniciado acciones de cumplimiento](https://www.congress.gov/crs_external_products/IF/HTML/IF13103.html) contra empresas por **"AI washing"** — afirmaciones falsas sobre el uso de IA.

**Ejemplos de infracciones:**

- Afirmaban "AI-powered" pero usaban reglas simples de if-then
- Prometían "deep learning" pero no revelaban cómo funciona el modelo
- Exageraban la precisión de los modelos

**Posición de la SEC:**

> "AI washing could lead to failures to comply with disclosure requirements and lead to investor harm."

### **FCA (Reino Unido) y ESMA (UE)**

Exigen:

- **Toma de decisiones transparente** para el trading automatizado
- **Kill switch** (capacidad de detener el sistema instantáneamente)
- **Informe post-operación** (explicación de por qué se realizó la operación)

## Cómo protegerse de los riesgos de la IA caja negra

### **1. Use sistemas híbridos**

**Idea:** La IA genera señales, un humano toma la decisión final.

**Ejemplo:**

```python
class HybridTradingSystem:
    def __init__(self):
        self.ai_model = DeepLearningModel()
        self.risk_manager = HumanRiskManager()

    def trade(self, market_data):
        # La IA genera señal
        ai_signal = self.ai_model.predict(market_data)
        confidence = self.ai_model.get_confidence()

        # Explicación
        explanation = self.get_explanation(market_data, ai_signal)

        # Aprobación humana para baja confianza
        if confidence < 0.7:
            approved = self.risk_manager.approve(ai_signal, explanation)
            if not approved:
                return None

        return ai_signal
```

**Resultado:** La IA acelera, el humano controla.

### **2. Implemente XAI desde el primer día**

**No haga:**

```python
model.predict(X)  # Obtiene respuesta, no sabe por qué
```

**Haga:**

```python
prediction, explanation = model.predict_with_explanation(X)
log(f"Decision: {prediction}, Reason: {explanation}")
```

**Siempre registre las explicaciones.** Cuando haya pérdidas, sabrá por qué.

### **3. Monitoree regularmente el concept drift**

**Código:**

```python
from scipy import stats

def detect_drift(recent_predictions, historical_predictions):
    # KS-test para comparar distribuciones
    statistic, pvalue = stats.ks_2samp(recent_predictions, historical_predictions)

    if pvalue < 0.05:
        alert("Concept drift detected! Model may be outdated.")
        return True
    return False

# Cada día
if detect_drift(last_30_days_predictions, training_period_predictions):
    retrain_model()
```

### **4. Circuit breakers y kill switches**

**Reglas:**

- Pérdida máxima diaria: -5%
- Operaciones máximas por hora: 100
- Tamaño máximo de posición: 10% del portafolio

**Código:**

```python
class CircuitBreaker:
    def __init__(self):
        self.daily_loss = 0
        self.trades_this_hour = 0

    def check_before_trade(self, trade):
        # Comprobar pérdida diaria
        if self.daily_loss < -0.05:
            raise CircuitBreakerTripped("Daily loss limit exceeded")

        # Comprobar frecuencia de operaciones
        if self.trades_this_hour > 100:
            raise CircuitBreakerTripped("Hourly trade limit exceeded")

        # Comprobar tamaño de posición
        if trade.size > self.portfolio_value * 0.10:
            raise CircuitBreakerTripped("Position size too large")
```

### **5. Haga backtesting en escenarios extremos**

No pruebe solo en condiciones de mercado "normales".

**Pruebe en:**

- Crash del COVID (marzo 2020)
- Flash crash (mayo 2010)
- Colapso de SVB (marzo 2023)
- Colapso de FTX (noviembre 2022)

**Pregunta:** ¿Sobreviviría su IA a un día con -20%?

### **6. Comience con poco capital**

**No haga:**

"El backtest mostró Sharpe 2,0, ¡meto todo el portafolio!"

**Haga:**

"El backtest mostró Sharpe 2,0, empezaré con el 5% del portafolio. En 3 meses, aumentaré."

**Estadísticas:**

[La investigación muestra](https://www.lse.ac.uk/research/research-for-the-world/ai-and-tech/ai-and-stock-market): el 80% de las estrategias con buenos backtests **fracasan en los primeros 3 meses** en operativa real.

## Conclusiones

**¿Puede la IA ayudar en el trading?** Sí.

**¿Puede la IA perjudicar?** Sí. Y mucho.

**Conclusiones clave:**

1. **La IA de caja negra es un riesgo** — el 85% de los traders no confía en sistemas sin explicaciones
2. **Las pérdidas reales son enormes** — desde $50M (hedge fund) hasta $1.700M (casos CFTC)
3. **Los reguladores exigen transparencia** — EU AI Act, SEC, FCA
4. **XAI ayuda pero no es una panacea** — los modelos complejos seguirán siendo complejos
5. **El enfoque híbrido es más seguro** — la IA genera, el humano decide

**Recomendaciones prácticas:**

- Use XAI (SHAP, LIME) para explicar decisiones
- Implemente circuit breakers y kill switches
- Monitoree el concept drift regularmente
- Comience con poco capital
- Pruebe en escenarios extremos
- NO confíe en "bots de IA" sin lógica transparente
- NO despliegue una caja negra en todo su portafolio
- NO ignore los requisitos regulatorios

**Próximo artículo:**

[Experimento: LLM + algoritmo clásico]({{site.baseurl}}/2026/03/31/eksperiment-llm-plus-klassika.html) — ¿podemos mejorar una estrategia con filtros de IA manteniendo la explicabilidad?

La IA es una herramienta poderosa. Pero como cualquier herramienta poderosa, requiere **precaución, control y comprensión**.

Rentabilidad sin comprensión no es una ventaja. Es una ruleta.

---

**Enlaces útiles:**

Riesgos de la IA caja negra:
- [Black Box AI: Hidden Algorithms and Risks in 2025](https://ts2.tech/en/black-box-ai-exposed-hidden-algorithms-risks-and-breakthroughs-in-2025/)
- [AI in Finance: How to Trust a Black Box?](https://www.finance-watch.org/wp-content/uploads/2025/03/Artificial_intelligence_in_finance_report_final.pdf)
- [Transparent AI vs Black Box Trading Systems](https://www.ampfi.app/blog/transparent-ai-vs-black-box-trading-systems)
- [Why Blackbox AI Matters to Businesses](https://www.voiceflow.com/blog/blackbox-ai)

Casos reales de fracasos:
- [CFTC: AI Won't Turn Trading Bots into Money Machines](https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/AITradingBots.html)
- [How AI Crypto Trading Bots Lose Millions](https://www.ccn.com/education/crypto/ai-crypto-trading-bots-how-they-make-and-lose-millions/)
- [Lessons from Algo Trading Failures](https://www.luxalgo.com/blog/lessons-from-algo-trading-failures/)
- [Systemic Failures in Algorithmic Trading](https://pmc.ncbi.nlm.nih.gov/articles/PMC8978471/)

Flash crashes y riesgo sistémico:
- [2010 Flash Crash](https://en.wikipedia.org/wiki/2010_flash_crash)
- [How Trading Algorithms Trigger Flash Crashes](https://hackernoon.com/how-trading-algorithms-can-trigger-flash-crashes)
- [AI and Market Manipulation](https://www.theregreview.org/2025/11/25/smith-ai-and-the-future-of-market-manipulation/)

IA explicable:
- [2025 Guide to Explainable AI in Forex Trading](https://kaliham.com/2025-guide-to-explainable-ai-in-forex-trading-clarity-compliance-confidence/)
- [Understanding Black Box AI: Challenges and Solutions](https://www.ewsolutions.com/understanding-black-box-ai/)
- [Risks and Remedies for Black Box AI](https://c3.ai/blog/risks-and-remedies-for-black-box-artificial-intelligence/)

Regulación:
- [AI in Capital Markets: Policy Issues](https://www.congress.gov/crs-product/IF13103)
- [IOSCO Report on Artificial Intelligence](https://www.iosco.org/library/pubdocs/pdf/IOSCOPD788.pdf)
