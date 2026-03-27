---
layout: post
title: "IA vs Open Source: que cambio realmente y donde esta el limite"
description: "Un analisis detallado de como los modelos de codigo modernos han cambiado el equilibrio entre la generacion y las bibliotecas existentes en el trading algoritmico."
date: 2025-09-03
image: /assets/images/blog/ai_vs_oss.png
tags: [AI, Open Source, algorithmic trading, development]
lang: es
---

Publique un nuevo articulo en Habr: **["IA vs Open Source: que cambio realmente y donde esta el limite"](https://habr.com/ru/articles/943670/)**

Con la aparicion de modelos de codigo funcionales, surgio un camino de desarrollo mas pragmatico: formular un requisito, escribir pruebas y obtener un modulo pequeno y comprensible sin dependencias innecesarias. No es una guerra contra el OSS, sino un desplazamiento del punto de equilibrio.

## Puntos principales del articulo:

### Que cambio
- **Antes**: "primero la biblioteca." Buscar una biblioteca, aceptar dependencias transitivas, leer documentacion.
- **Ahora**: "descripcion -> pruebas -> implementacion." Modulos pequenos y verificables en lugar de "combinados" monoliticos.

### Donde la IA ya reemplaza a las bibliotecas
1. **Mini-implementaciones**: indicadores (EMA/SMA/RSI), estadisticas, reglas de riesgo
2. **Integraciones especificas**: clientes REST/WebSocket con solo 2-3 metodos necesarios
3. **Generacion de esqueletos**: estructuras de backtesting, esquemas de datos
4. **Adaptadores**: mapeo entre exchanges, migraciones de codigo

### Donde la IA NO deberia reemplazar al OSS
- Criptografia y protocolos seguros
- Protocolos binarios (FIX/ITCH/OUCH/FAST)
- Motores de bases de datos, compiladores, runtimes
- Solvers numericos y optimizadores

### Consejos practicos
- Mantener los modulos pequenos
- Describir el comportamiento con palabras simples
- Hacer verificaciones minimas para merges seguros
- Generar sin dependencias externas

En el trading algoritmico esto es especialmente relevante: menos dependencias significa menos riesgos, artefactos mas compactos, auditorias mas simples e iteraciones mas rapidas.

**Conclusion clave**: Elige la herramienta segun el contexto. Una tarea especifica que sea facil de describir y verificar es candidata para generacion. Todo lo demas -- mejor usar OSS probado.
