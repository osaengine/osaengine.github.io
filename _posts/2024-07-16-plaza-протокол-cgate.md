---
layout: post
title: "Плаза-протокол: что это и зачем он нужен"
description: "Детальный анализ работы Плаза-протокола и его компонентов."
date: 2024-07-16
image: /assets/images/blog/plaza-protocol-cgate.png
tags: [Plaza, HFT, MOEX]
---

Привет, друзья!

Сегодня мы рассмотрим, что такое Плаза-протокол, как он работает и зачем он нужен. Также обсудим, почему Плаза-протокол постепенно уходит в прошлое и какие решения приходят ему на смену.

## Что такое Плаза-протокол?

Плаза-протокол (Plaza II) — это коммуникационный протокол, разработанный Московской биржей для обеспечения высокоскоростной и надежной передачи торговой и рыночной информации между участниками торгов. Он состоит из нескольких компонентов, каждый из которых выполняет свою важную роль.

### Основные компоненты Плаза-протокола

1. **Системные библиотеки Plaza-2**
2. **Маршрутизатор сообщений P2MQRouter**
3. **Шлюзовая библиотека cgate**
4. **Заголовочный файл cgate.h**

### Как установить и настроить Плаза-протокол

Для начала работы с Плаза-протоколом необходимо установить соответствующие компоненты. Вот пример установки и настройки на Linux:

#### Установка из zip-архива

```sh
chmod 755 ./install.sh
./install.sh ./cgate_linux_amd64-7.12.0.103.zip
```

Затем укажите логин и пароль для подключения к системе:

```ini
[AS:NS]
USERNAME=<your login>
PASSWORD=<your password>
```

### Создание и управление соединениями

Пример создания соединения:

```c
cg_conn_t* conn;
cg_err_t result = cg_conn_new("p2tcp://127.0.0.1:4001;app_name=test", &conn);
if (result != CG_ERR_OK) {
    printf("Error creating connection: %d
", result);
    return -1;
}

result = cg_conn_open(conn, 0);
if (result != CG_ERR_OK) {
    printf("Error opening connection: %d
", result);
    cg_conn_destroy(conn);
    return -1;
}

// Обработка соединения
cg_conn_process(conn, CG_TIMEOUT_INFINITE);

// Закрытие и уничтожение соединения
cg_conn_close(conn);
cg_conn_destroy(conn);
```

### Работа с подписчиками и публикаторами

Подписчики и публикаторы играют ключевую роль в Плаза-протоколе. Они привязаны к конкретному соединению и обрабатываются из того же потока, что и соединение.

#### Пример создания подписчика:

```c
cg_listener_t* listener;
result = cg_lsn_new(conn, "p2repl://FORTS_FUTAGGR50_REPL", &listener);
if (result != CG_ERR_OK) {
    printf("Error creating listener: %d
", result);
    cg_conn_close(conn);
    cg_conn_destroy(conn);
    return -1;
}

result = cg_lsn_open(listener, 0);
if (result != CG_ERR_OK) {
    printf("Error opening listener: %d
", result);
    cg_lsn_destroy(listener);
    cg_conn_close(conn);
    cg_conn_destroy(conn);
    return -1;
}

// Обработка сообщений подписчика
while (1) {
    result = cg_conn_process(conn, CG_TIMEOUT_INFINITE);
    if (result != CG_ERR_OK) {
        printf("Error processing connection: %d
", result);
        break;
    }
}

// Закрытие и уничтожение подписчика
cg_lsn_close(listener);
cg_lsn_destroy(listener);
cg_conn_close(conn);
cg_conn_destroy(conn);
```

### Работа с данными и схемами

Обработка данных в Плаза-протоколе ведётся через схемы данных. Вот основные моменты:

#### Сборка и настройка схем

Для управления схемами данных используется утилита `schemetool`. Вот пример использования:

```sh
schemetool makesrc --input scheme.xml --output output_dir
```

#### Получение потоков репликации

Пример получения реплики данных и вывода их в лог:

```c
cg_listener_t* repl_listener;
result = cg_lsn_new(conn, "p2repl://FORTS_FUTCOMMON_REPL", &repl_listener);
if (result != CG_ERR_OK) {
    printf("Error creating replication listener: %d
", result);
    cg_conn_close(conn);
    cg_conn_destroy(conn);
    return -1;
}

result = cg_lsn_open(repl_listener, 0);
if (result != CG_ERR_OK) {
    printf("Error opening replication listener: %d
", result);
    cg_lsn_destroy(repl_listener);
    cg_conn_close(conn);
    cg_conn_destroy(conn);
    return -1;
}

// Обработка сообщений репликации
while (1) {
    result = cg_conn_process(conn, CG_TIMEOUT_INFINITE);
    if (result != CG_ERR_OK) {
        printf("Error processing connection: %d
", result);
        break;
    }
}

// Закрытие и уничтожение репликационного подписчика
cg_lsn_close(repl_listener);
cg_lsn_destroy(repl_listener);
cg_conn_close(conn);
cg_conn_destroy(conn);
```

### Логины и их виды

Для работы с Плаза-протоколом вам потребуются разные виды логинов:

- **Тестовые логины**: используются для разработки и тестирования.
- **Production-логины**: получаются после сертификации и используются в реальной среде.

### Почему Плаза-протокол устарел?

Как и всё хорошее, Плаза-протокол потихоньку уходит на покой. На смену ему приходят современные решения, такие как TWIME и SIMBA, которые обеспечивают более высокую производительность, гибкость и надёжность. Так что, если вы ещё пользуетесь Плаза-протоколом, самое время задуматься о переходе на что-то новое и более эффективное.

### Заключение

Вот так, без лишней воды и занудства, мы разобрали основы работы с Плаза-протоколом. Теперь у вас есть базовое понимание, как это работает и почему стоит задуматься о переходе на более современные решения. Надеюсь, было весело и полезно!

Если у вас остались вопросы или вы хотите больше примеров и деталей, не стесняйтесь обращаться. Удачи в разработке!
