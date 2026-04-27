# proj4

Учебная веб-служба на `C#` и `ASP.NET Core`, которая моделирует процесс бронирования переговорки как машину состояний, выдерживает повторную доставку событий, выполняет компенсацию при частичном сбое и показывает наблюдаемость через журналы, health checks и метрики.

## Что реализовано

- машина состояний с минимум четырьмя состояниями;
- хранение процесса в памяти по `processKey`;
- идемпотентность по `idempotencyKey`;
- компенсация при сбое шага `SendNotification`;
- сквозной `correlationId` в ответах и журнале;
- `live` и `ready` health checks;
- метрики успешных и ошибочных переходов, дублей и компенсаций;
- грубая оценка задержки шага через `Histogram`.

## Установка

Если `dotnet` еще не установлен, для `Arch Linux`:

```bash
sudo pacman -S dotnet-sdk-8.0 aspnet-runtime-8.0
```

## Состояния и события

Состояния процесса:

- `NotStarted`
- `Requested`
- `RoomReserved`
- `NotificationSent`
- `Completed`
- `Compensated`

Основные события:

- `StartBooking`
- `ReserveRoom`
- `SendNotification`
- `CompleteBooking`

Если `SendNotification` приходит с `simulateFailure=true`, выполняется компенсация: состояние переводится в `Compensated`, а событие считается ошибочным переходом с откатом.

## Запуск

```bash
cd proj4
dotnet restore
dotnet run --urls http://localhost:5090
```

Swagger будет доступен по адресу:

```text
http://localhost:5090/swagger
```

## Ручной прогон

Старт процесса:

```bash
curl -i \
  -H "Content-Type: application/json" \
  -H "X-Correlation-Id: corr-1" \
  -d '{"processKey":"room-101","idempotencyKey":"evt-1","eventType":"StartBooking"}' \
  http://localhost:5090/api/bookings/events
```

Резервирование переговорки:

```bash
curl -i \
  -H "Content-Type: application/json" \
  -H "X-Correlation-Id: corr-2" \
  -d '{"processKey":"room-101","idempotencyKey":"evt-2","eventType":"ReserveRoom"}' \
  http://localhost:5090/api/bookings/events
```

Повторная доставка того же события:

```bash
curl -i \
  -H "Content-Type: application/json" \
  -H "X-Correlation-Id: corr-3" \
  -d '{"processKey":"room-101","idempotencyKey":"evt-2","eventType":"ReserveRoom"}' \
  http://localhost:5090/api/bookings/events
```

Сбой и компенсация:

```bash
curl -i \
  -H "Content-Type: application/json" \
  -H "X-Correlation-Id: corr-4" \
  -d '{"processKey":"room-101","idempotencyKey":"evt-3","eventType":"SendNotification","simulateFailure":true}' \
  http://localhost:5090/api/bookings/events
```

Проверка состояния процесса:

```bash
curl http://localhost:5090/api/bookings/room-101
```

## Наблюдаемость

Liveness:

```bash
curl -i http://localhost:5090/health/live
```

Readiness:

```bash
curl -i http://localhost:5090/health/ready
```

Readiness становится неуспешной после достижения порога критических сбоев. В текущей конфигурации порог задается через `Operations:CriticalFailureThreshold` и по умолчанию равен `2`.

Сводка метрик:

```bash
curl http://localhost:5090/metrics-summary
```

## Тесты

```bash
dotnet test proj4/tests/Proj4.Tests/Proj4.Tests.csproj
```

Проверки покрывают:

- корректные переходы машины состояний;
- повторную доставку по `idempotencyKey`;
- компенсацию при сбое;
- деградацию readiness;
- публикацию метрик и работу HTTP API.
