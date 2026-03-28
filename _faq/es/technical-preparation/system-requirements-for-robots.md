---
lang: es
layout: faq_article
title: "¿Cuáles son los requisitos mínimos del sistema para ejecutar un robot de trading?"
section: technical
order: 4
---

Un robot de trading requiere un entorno computacional estable. Los requisitos mínimos del sistema dependen de la complejidad de la estrategia, la frecuencia de las operaciones y el volumen de datos procesados.

## Requisitos mínimos:

1. **Sistema operativo:**
   - Windows 10/11, Linux o macOS.
   - Soporte para versiones de servidor del SO para trabajo ininterrumpido.

2. **Procesador:**
   - Para estrategias sencillas: 2 núcleos (Intel i3/AMD Ryzen 3 o superior).
   - Para estrategias de alta frecuencia: 4+ núcleos (Intel i7/AMD Ryzen 5 o superior).

3. **Memoria RAM:**
   - 4 GB para tareas básicas.
   - 8+ GB para el procesamiento de grandes volúmenes de datos.

4. **Conexión a internet:**
   - Acceso estable con baja latencia (<50 ms hasta la bolsa).
   - Se recomienda un servidor dedicado o VPS para minimizar interrupciones.

5. **Espacio en disco:**
   - 10 GB para la instalación de la plataforma y almacenamiento de logs.
   - Disco SSD para acelerar las operaciones.

## Equipamiento recomendado:

- Para algoritmos complejos, utilice servidores en la nube como **[AWS](https://aws.amazon.com/)**, **[Google Cloud](https://cloud.google.com/)** o **[Azure](https://azure.microsoft.com/)**.
- Para el trading de alta frecuencia, configure un servidor cerca de la bolsa.

## Consejos:

- Asegúrese de que su ordenador o servidor esté actualizado y soporte un funcionamiento estable.
- Utilice fuentes de alimentación de respaldo (SAI/UPS) para protegerse de cortes de energía.
- Limpie regularmente los logs y actualice el software.
