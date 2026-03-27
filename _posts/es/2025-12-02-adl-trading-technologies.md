---
layout: post
title: "ADL de Trading Technologies: cuando un constructor de estrategias cuesta como un apartamento"
description: "Quise probar ADL (Algo Design Lab) de Trading Technologies, un constructor visual para profesionales. Pero me encontre con un problema: es una solucion empresarial con precios 'bajo consulta'. Esto es lo que descubri."
date: 2025-12-02
image: /assets/images/blog/adl_trading_tech.png
tags: [ADL, Trading Technologies, no-code, institutional trading, futures]
lang: es
---

Durante las ultimas semanas, he estado probando constructores visuales de estrategias. TSLab por 60.000 rublos al ano, StockSharp Designer gratis, NinjaTrader por 150.000, fxDreema por 10.000. Era logico anadir a la lista ADL (Algo Design Lab) de Trading Technologies, un constructor para "profesionales".

Fui al sitio web de Trading Technologies. Encontre la pagina de ADL. Capturas de pantalla bonitas, marketing sobre drag-and-drop, programacion visual, integracion con una plataforma de grado institucional.

**Pregunta:** Donde esta el boton de "Descargar" o al menos "Probar"?

Respuesta: no existe.

## Como obtener acceso a ADL (spoiler: no se puede)

ADL no es un producto independiente. Es un modulo dentro de TT Platform Pro de Trading Technologies. Para obtener ADL, primero necesitas acceso a TT Platform.

Empece a buscar como hacerlo.

**Opcion 1:** Registrarse en el sitio de TT y descargar la plataforma.

Lo intente. El sitio tiene un formulario de "Contactenos". Lo llene. Indique que queria probar ADL para una resena. Un dia despues recibi una respuesta: "Gracias por su interes. Nos pondremos en contacto con usted para discutir sus necesidades de trading."

Pasaron dos semanas. Nadie me contacto.

**Opcion 2:** Encontrar un broker que ofrezca acceso a TT Platform.

Busque en Google. AMP Futures, Optimus Futures, Discount Trading: varios brokers estadounidenses ofrecen TT Platform. Pero en todas partes es lo mismo: "Precios bajo consulta", "Depende del volumen de trading", "Contactenos para una oferta personalizada".

Me puse en contacto con uno de los brokers. Pregunte sobre el acceso a TT Platform + ADL.

Respuesta: "La suscripcion minima a TT Platform Pro es de $1,500 al mes. Mas comisiones por operaciones. Mas pago por datos de mercado. ADL esta incluido gratis si tiene TT Platform Pro."

$1,500 al mes. **$18,000 al ano.** En rublos, eso es aproximadamente **1,8 millones**.

Por un constructor visual de estrategias.

## Lo que pude averiguar sin acceso

Como no pude probar ADL realmente, tuve que recopilar informacion por partes: documentacion de TT, videos de YouTube, foros, opiniones de traders.

**Que es ADL:**

Es un constructor visual de algoritmos integrado en TT Platform. Interfaz drag-and-drop, bloques para condiciones y acciones, backtesting con datos historicos. Conceptualmente similar a TSLab o StockSharp Designer.

**Diferencia clave:** ADL vive dentro de una plataforma de trading profesional. TT Platform es utilizada por fondos de cobertura, traders propietarios, actores institucionales. No es un producto retail.

**Que puede hacer (segun la documentacion):**

- Construccion visual de algoritmos mediante bloques
- Backtesting con datos historicos
- Simulacion en mercado en tiempo real
- Integracion con Order Management System (OMS)
- Ejecucion de algoritmos directamente en el libro de ordenes
- Monitoreo de rendimiento en tiempo real

**Que NO puede hacer:**

- Funcionar fuera de TT Platform (no hay exportacion de codigo)
- Funcionar gratis o al menos a bajo costo
- Ser accesible para un trader retail comun

## Para quien es esto?

Pense en esto durante varios dias. Y esto es lo que concluyo.

ADL no es para traders retail. Ni siquiera para traders individuales activos. Es para actores institucionales:

- Firmas de trading propietario
- Fondos de cobertura
- Creadores de mercado
- Grandes gestoras de activos

Personas que operan millones de dolares al dia. Para ellos, $1,500 al mes por una plataforma es calderilla comparado con sus volumenes.

**La paradoja:** ADL se posiciona como "un constructor con el que cualquiera puede crear algoritmos". Pero para obtener acceso, necesitas pagar como un actor institucional.

## Comparacion con lo que probe

Durante las ultimas semanas trabaje realmente con cuatro constructores visuales:

**TSLab** — 60.000 rublos al ano. Diagramas de flujo, mercado ruso, idioma ruso. Funciona, pero es caro para lo que ofrece.

**StockSharp Designer** — gratis. Open-source, diagramas de flujo, exportacion de codigo. Mercados rusos e internacionales. Menos maduro, pero funcionalmente cercano a TSLab.

**NinjaTrader Strategy Builder** — 150.000 rublos de por vida o 120.000 al ano. Interfaz tabular (no bloques), solo mercados internacionales. Producto maduro, pero para un nicho reducido.

**fxDreema** — 10.000 rublos al ano. Diagramas de flujo en el navegador, solo MetaTrader. Un proyecto secundario de entusiastas. Funciona, pero hay riesgo de que cierre.

**ADL** — 1,8 millones de rublos al ano (minimo). Constructor visual dentro de una plataforma profesional. No pude probarlo, pero segun las opiniones, es una herramienta solida para quienes realmente la necesitan.

La diferencia de precio: 30 veces mas que TSLab y 180 veces mas que fxDreema.

## Conclusion honesta: no pude probarlo

Normalmente en mis articulos escribo sobre experiencia real. Instale, probe, me encontre con problemas, saque conclusiones.

Con ADL eso no sucedio.

**La razon es simple:** es una solucion empresarial. No tienen version demo. No hay periodo de prueba. Ni siquiera precios publicos. Todo es "contactenos", "oferta personalizada", "depende de los volumenes".

Podria haber escrito un articulo basado en los materiales de marketing de TT. Pero no habria sido mi articulo, sino una repeticion de publicidad ajena.

En su lugar, decidi escribir honestamente: **ADL parece una herramienta poderosa, pero no es para traders comunes.**

Si operas con millones de dolares en futuros estadounidenses, trabajas en una firma propietaria o fondo de cobertura, y necesitas un constructor visual con infraestructura de grado institucional, ADL puede ser una buena opcion.

Pero si eres un trader individual que quiere construir un robot para la Bolsa de Moscu o simplemente probar el trading algoritmico, olvidate de ADL. Demasiado caro. Demasiado dificil obtener acceso. Demasiado enfocado en el nivel institucional.

## Lo que hice en su lugar

Sin poder acceder a ADL, volvi a lo que ya habia probado:

- **StockSharp Designer** — gratis, funciona con brokers rusos, open-source
- **fxDreema** — 10.000 al ano, si operas a traves de MetaTrader
- **TSLab** — 60.000 al ano, si quieres una solucion lista con soporte

Los tres ofrecen programacion visual. Los tres son realmente accesibles. Los tres se pueden probar en 20 minutos.

**Mi conclusion:** Para el 99% de los traders, ADL es una imagen bonita en el sitio web de Trading Technologies. Inaccesible, caro, institucional.

Quiza algun dia tendre acceso a TT Platform. Entonces escribire una resena completa de ADL con pruebas reales y capturas de pantalla.

Por ahora, esta es la historia de una plataforma que no pude probar. Pero que ilustra perfectamente la diferencia entre el trading algoritmico retail e institucional.

Los actores institucionales pagan millones por infraestructura. Los traders retail construyen robots con bibliotecas open-source gratuitas.

Dos universos diferentes. ADL pertenece a aquel donde la tarifa minima de la plataforma cuesta como un buen auto.

---

**Enlaces utiles:**

- [Pagina oficial de ADL](https://tradingtechnologies.com/trading/algo-trading/adl/)
- [Documentacion de TT Platform](https://library.tradingtechnologies.com/)
