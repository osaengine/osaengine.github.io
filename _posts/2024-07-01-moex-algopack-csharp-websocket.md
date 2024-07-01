---
layout: post
title: "Пример подключения по WebSocket к Алгопаку от Мосбиржи"
description: "Пример кода для подключения по WebSocket к Алгопаку от Московской биржи, демонстрирующий упрощение доступа и функциональности для трейдеров."
date: 2024-07-01
image: /assets/images/moex_algopack.png
tags: [MOEX, AlgoPack, WebSocket, C#]
---

Мосбиржа недавно [выпустила пример работы с WebSocket](moex-algopack-websockets.html), который доступен только на Python. Поскольку протокол не является стандартным и запросы выполняются не через JSON, мы подготовили пример на C#, чтобы облегчить интеграцию для разработчиков, использующих эту платформу.

Работа в действии выглядит так:

![MOEX AlgoPack websockets](/assets/images/blog/algopack_websocket_csharp.png)

## Пример кода

### Класс для работы с WebSocket

```csharp
namespace OsaEngine.MoexAlgoPack;

using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Net.WebSockets;

public class MoexAlgoPackSocketClient(string url) : IAsyncDisposable
{
    private readonly Uri _uri = new(url);
    private readonly ClientWebSocket _clientWebSocket = new();

    public async ValueTask ConnectAsync(string domain = "DEMO", string login = "guest", string passcode = "guest", CancellationToken cancellationToken = default)
    {
        await _clientWebSocket.ConnectAsync(_uri, cancellationToken);
        await SendAsync($"CONNECT\ndomain:{domain}\nlogin:{login}\npasscode:{passcode}\n\n\0", cancellationToken);
    }

    public ValueTask SubscribeAsync(object id, string destination, string selector, CancellationToken cancellationToken = default)
    {
        return SendAsync($"SUBSCRIBE\nid:{id}\ndestination:{destination}\nselector:{selector}\n\n\0", cancellationToken);
    }

    public async ValueTask SendAsync(string message, CancellationToken cancellationToken = default)
    {
        var messageBytes = Encoding.UTF8.GetBytes(message);
        var segment = new ArraySegment<byte>(messageBytes);
        await _clientWebSocket.SendAsync(segment, WebSocketMessageType.Text, true, cancellationToken);
    }

    public async ValueTask ReceiveAsync(Action<string> received, CancellationToken cancellationToken = default)
    {
        var buffer = new byte[1024 * 4];

        while (_clientWebSocket.State == WebSocketState.Open)
        {
            var result = await _clientWebSocket.ReceiveAsync(new(buffer), cancellationToken);

            if (result.MessageType == WebSocketMessageType.Close)
            {
                await _clientWebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, string.Empty, default);
            }
            else
            {
                var message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                received(message);
            }
        }
    }

    public async ValueTask CloseAsync()
    {
        await _clientWebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", default);
        _clientWebSocket.Dispose();
    }

    ValueTask IAsyncDisposable.DisposeAsync()
    {
        GC.SuppressFinalize(this);
        return CloseAsync();
    }
}
```

### Пример использования

```csharp
var client = new MoexAlgoPackSocketClient("wss://iss.moex.com/infocx/v3/websocket");
await client.ConnectAsync("passport", "<email>", "<password>");

_ = Task.Run(async () =>
{
    await Task.Yield();
    await client.ReceiveAsync(Console.WriteLine);
});

await Task.Delay(2000);
await client.SubscribeAsync(Guid.NewGuid(), "MXSE.orderbooks", "TICKER=\"MXSE.TQBR.SBER\"");

Console.ReadLine();

await client.CloseAsync();
```

## Заключение

Этот пример демонстрирует, как можно легко подключиться к Алгопаку от Мосбиржи и начать получать данные в реальном времени через WebSocket. Это улучшение значительно повышает доступность и функциональность для трейдеров.

Для получения более подробной информации и доступа к документации, пожалуйста, посетите [Мосбиржа Алгопак](https://moexalgo.github.io/api/websocket/).