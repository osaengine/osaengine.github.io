---
layout: post
title: "Импорт данных с Московской биржи в Google Sheets"
description: "Как быстро и эффективно импортировать данные о ценах на акции с Московской биржи в Google Sheets с использованием Google Apps Script."
date: 2024-08-02
image: /assets/images/blog/google-sheet-moex-stocks.png
tags: [ChatGPT, MOEX, SmartLab]
---

Привет, друзья!

Недавно на платформе [Smart-Lab](https://smart-lab.ru/blog/1045048.php) его владелец Тимофей задал вопрос о том, как импортировать данные о ценах на акции с Московской биржи в Google Sheets. Ответ оказался неожиданно простым, несмотря на то, что никто не упомянул об использовании ИИ и ChatGPT, которые помогли найти решение всего за 5 минут.

## Шаг 1: Создание нового документа в Google Sheets

Первым шагом необходимо создать новый документ Google Sheets. Перейдите на сайт [Google Sheets](https://sheets.google.com) и создайте новый документ или откройте существующий.

## Шаг 2: Открытие редактора сценариев

В вашем документе Google Sheets перейдите в меню "Расширения" и выберите "Apps Script". Это откроет редактор сценариев, где вы сможете написать и выполнить скрипт для импорта данных.

## Шаг 3: Написание скрипта для импорта данных

В редакторе сценариев удалите все существующие скрипты и вставьте следующий код:

```javascript
function importMoexData() {
  var url = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.xml';
  try {
    var response = UrlFetchApp.fetch(url);
    var xml = response.getContentText();
    var document = XmlService.parse(xml);
    var root = document.getRootElement();
    var dataElements = root.getChildren('data');
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    sheet.clear();

    dataElements.forEach(function(dataElement) {
      var rows = dataElement.getChild('rows').getChildren('row');

      // Добавляем заголовки
      if (rows.length > 0) {
        var headers = rows[0].getAttributes().map(function(attr) {
          return attr.getName();
        });
        sheet.appendRow(headers);
      }

      // Добавляем данные
      rows.forEach(function(row) {
        var data = row.getAttributes().map(function(attr) {
          var value = attr.getValue();
          // Заменяем точки на запятые в значениях
          if (!isNaN(value.replace('.', '').replace(',', ''))) {
            value = value.replace('.', ',');
          }
          return value;
        });
        sheet.appendRow(data);
      });
    });
  } catch (e) {
    Logger.log('Error: ' + e.message);
  }
}
```

Этот скрипт извлекает данные с сайта Московской биржи в формате XML и импортирует их в ваш Google Sheet, заменяя точки на запятые в числовых значениях.

## Шаг 4: Выполнение скрипта

Сохраните скрипт, нажав на значок диска в верхней части редактора. Затем выполните скрипт, нажав на значок треугольника (выполнить). Скрипт загрузит данные с указанного URL и импортирует их в ваш Google Sheet.

## Заключение

Таким образом, всего за несколько минут вы можете настроить автоматический импорт данных с Московской биржи в Google Sheets, используя Google Apps Script. Этот метод позволяет экономить время и автоматизировать рутинные задачи, что особенно полезно для трейдеров и аналитиков. 

Хотя вопрос был задан Тимофеем на Smart-Lab, этот метод может быть полезен многим, кто ищет эффективные способы работы с финансовыми данными. Надеюсь, эта статья поможет вам в ваших начинаниях!