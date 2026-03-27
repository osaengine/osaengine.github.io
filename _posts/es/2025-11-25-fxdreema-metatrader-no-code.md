---
layout: post
title: "fxDreema: cuando tienes MetaTrader pero no quieres aprender MQL"
description: "MetaTrader instalado, broker conectado, pero escribir MQL es un dolor. Encontre fxDreema -- un constructor visual de robots para MT4/MT5. Cuento que resulto y si vale $95 al ano."
date: 2025-11-25
image: /assets/images/blog/fxdreema_builder.png
tags: [fxDreema, MetaTrader, MT4, MT5, no-code, forex, visual builder]
lang: es
---

Tenia un problema. MetaTrader instalado, broker conectado, estrategia en la cabeza. Pero para implementarla, necesito aprender MQL4 o MQL5. Y es un lenguaje con sus propias rarezas, documentacion semi-inglesa, y foros donde la mitad de las respuestas son "lee el manual."

Busque en Google "MetaTrader sin programacion" y di con fxDreema. Un constructor visual de robots. Arrastras bloques -- obtienes un expert advisor. Suena simple. Decidi probarlo.

## Como funciona (o deberia funcionar)

fxDreema es una aplicacion web. **Punto importante:** no es un producto de MetaQuotes (los desarrolladores de MetaTrader). Es una herramienta de terceros hecha por entusiastas. Un equipo pequeno que construye un constructor para la plataforma de otro.

Entras al sitio, te registras, empiezas a construir una estrategia. Sin instalaciones, sin IDEs. Todo en el navegador.

**La idea:** tomas bloques (indicadores, condiciones, acciones), los conectas con flechas, como un diagrama de flujo. El programa genera codigo MQL. Descargas el archivo, lo metes en MetaTrader -- robot listo.

En teoria es hermoso. En teoria.

Me registre (gratis), abri el editor. Y efectivamente -- bloques visuales, como en Scratch o Node-RED. Arrastrar, conectar. Hay una biblioteca de bloques listos: indicadores, comprobaciones de precio, ordenes.

Arme una estrategia simple: si RSI esta por debajo de 30 -- comprar, si esta por encima de 70 -- vender. Un clasico. Pulse "Generar codigo," descargue el archivo .mq4. Lo meti en MetaTrader.

**Se ejecuto.** Sin errores. El robot opera.

Primera reaccion: "Vaya, esto realmente funciona."

## Despues empezaron los matices

Las estrategias simples se arman facilmente. Cruces de medias, RSI, MACD -- todo esta en bloques listos. 15-20 minutos y el robot esta listo.

Pero quise anadir un trailing stop. Y ahi descubri que la version gratuita tiene un limite: **maximo 10 "conexiones"** entre bloques.

10 conexiones son aproximadamente 5-6 bloques con condiciones. Para estrategias simples alcanza. Para algo mas complejo -- llegas al limite.

Ok, pense, comprare la version completa. Fui a ver los precios.

**$95 al ano.** O $33 por 3 meses.

Lo pense. $95 no es una locura. Pero la pregunta es: que obtengo por ese dinero?

- Eliminacion del limite de 10 conexiones
- Conversion de MQL4 a MQL5 (y viceversa)
- Parece que eso es todo

Sin soporte. Sin actualizaciones de la biblioteca de indicadores. Solo la eliminacion de una limitacion artificial.

## Intento de construir algo mas complejo

Decidi no comprar de inmediato, sino ver que podia exprimir de la version gratuita. Simplifique la estrategia, elimine verificaciones innecesarias, me ajuste a 10 conexiones.

Genere el codigo. Lo ejecute en MetaTrader en una cuenta demo.

**Problema numero uno:** Visualmente todo se ve claro -- bloques, flechas. Pero cuando la estrategia empieza a perder dinero, depurarla en fxDreema es doloroso. Hay que abrir el navegador, mirar el diagrama, cambiar bloques, regenerar codigo, meterlo en MetaTrader, reiniciar.

En codigo normal (en MQL o Python) abres el archivo, cambias un par de lineas, guardas. Aqui -- todo un ciclo.

**Problema numero dos:** El codigo MQL generado se ve... extrano. Variables con nombres automaticos, logica dispersa entre funciones, comentarios en ingles (si es que los hay). Dificil de leer. Aun mas dificil de modificar manualmente.

Es decir, si fxDreema no puede construir lo que necesitas -- estas atrapado. El codigo se genera, pero trabajar con el como codigo normal no funciona.

## Comparacion con lo que ya habia probado

En las ultimas semanas probe varios constructores visuales. Esto es lo que surge:

**TSLab/StockSharp Designer** -- diagramas de flujo, se ve la logica, exportable a C#. Funciona con brokers rusos.

**NinjaTrader** -- interfaz de tabla (no bloques), hecho para futuros americanos. $1,500 por la licencia.

**fxDreema** -- diagramas de flujo como Designer, pero solo para MetaTrader. $95 al ano. Y la version gratuita tiene un limite estricto de complejidad.

fxDreema tiene una ventaja: funciona en el navegador. No hay que instalar nada. Entras, construyes, descargas, ejecutas.

Pero eso tambien es una desventaja. Todo esta online. Si el sitio se cae -- te quedas sin herramienta.

**Y aqui lo interesante:** fxDreema no es un producto oficial de MetaQuotes. Es un servicio de terceros que genera codigo para la plataforma de otro. Un equipo pequeno, el proyecto vive de las suscripciones de usuarios.

Que pasa si manana MetaQuotes cambia algo en MQL y el codigo deja de compilar? O si los desarrolladores de fxDreema cierran el proyecto? Tus diagramas quedaran en sus servidores. El codigo generado tambien esta atado a su arquitectura.

Con plataformas oficiales (TSLab, NinjaTrader) al menos esta claro que no cerraran el ano que viene. Aqui -- hay riesgo.

## Para quien es realmente

Lo pense durante varios dias. Aqui esta mi conclusion.

fxDreema es adecuado si:

- Ya tienes MetaTrader (MT4 o MT5) y un broker
- Operas forex o CFDs a traves de MetaTrader
- Necesitas una estrategia de indicadores simple (cruces, niveles, RSI/MACD)
- No quieres aprender MQL
- Estas dispuesto a pagar ~$95/ano por la conveniencia

fxDreema NO es adecuado si:

- Operas el mercado ruso (MOEX, futuros rusos)
- Necesitas logica compleja con muchas condiciones
- Quieres una solucion gratuita (el limite de 10 conexiones se acaba muy rapido)
- Planeas modificar codigo manualmente (el MQL generado es ilegible)
- Quieres estabilidad y garantias (es un servicio de terceros, no un producto oficial)

## Que hice al final

No compre la suscripcion. Arme una estrategia simple en la version gratuita, descargue el codigo, lo meti en MetaTrader. Funciona.

Pero para la siguiente estrategia simplemente abri un tutorial de MQL5 y escribi el codigo a mano. Una hora para aprender la sintaxis basica, otra hora para escribir -- y tengo un expert advisor funcional. Sin limites. Sin suscripciones. Con control total.

**La paradoja:** fxDreema fue creado para eliminar la necesidad de aprender MQL. Pero cuando llegas a los limites del constructor visual, terminas decidiendo que hubiera sido mas facil aprender el lenguaje.

Pagar $95 al ano a un servicio de terceros que podria cerrar en cualquier momento, por una herramienta que ahorra un par de horas de aprendizaje? Cada uno decide por si mismo. Para mi, no cuadro.

## Conclusion honesta

fxDreema no es una mala herramienta. Realmente funciona. Los diagramas de flujo se arman facil, el codigo se genera, los robots se ejecutan.

Pero es una herramienta con un rango de uso muy estrecho:

- Solo MetaTrader (MT4/MT5)
- Solo estrategias simples (en la version gratuita)
- Solo si estas dispuesto a pagar para quitar los limites

Si ya operas a traves de MetaTrader, quieres automatizar una estrategia de indicadores simple y no quieres lidiar con programacion -- prueba la version gratuita. Quiza 10 conexiones sean suficientes para ti.

Pero si planeas dedicarte seriamente al trading algoritmico -- aprende MQL o pasa a algo mas flexible. Los constructores visuales tarde o temprano chocan con sus limitaciones. Y entonces tendras que programar de todas formas.

Pase dos dias en fxDreema. Arme tres estrategias, las ejecute en el tester, vi los resultados. Al final volvi al codigo.

Quiza simplemente no es para mi. O quiza los constructores visuales siempre son un compromiso entre simplicidad y control.

---

**Enlaces utiles:**

- [Sitio oficial de fxDreema](https://fxdreema.com/)
- [Documentacion y ejemplos](https://fxdreema.com/forum/)
