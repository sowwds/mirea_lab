# proj3

Веб-служба на `C#` и `ASP.NET Core` с приоритетом источников конфигурации, ранней валидацией настроек и базовой защитой через проверку `Origin`, rate limiting и security headers.

## Что реализовано

- чтение настроек из `appsettings.json`, переменных окружения и аргументов командной строки;
- приоритет источников: `json -> env -> cli`;
- ранняя валидация критичных настроек и отказ в запуске при ошибках;
- серверная блокировка недоверенного `Origin`;
- ограничение частоты запросов;
- защитные заголовки ответа;
- два режима работы: `Study` и `Production`.

## Установка

```bash
sudo pacman -S dotnet-sdk-8.0 aspnet-runtime-8.0
```

## Запуск

Учебный режим:

```bash
cd proj3
dotnet restore
dotnet run --urls http://localhost:5081
```

Боевой режим:

```bash
cd proj3
dotnet run --urls http://localhost:5081 -- \
  --App:Mode=Production \
  --App:AllowedOrigins:0=https://example.edu \
  --App:RateLimiting:PermitLimit=2 \
  --App:RateLimiting:WindowSeconds=30
```

## Ручные проверки

Доверенный источник:

```bash
curl -i \
  -H "Origin: https://example.edu" \
  http://localhost:5081/api/items
```

Ожидаемо: `200 OK`.

Недоверенный источник:

```bash
curl -i \
  -H "Origin: https://evil.example" \
  http://localhost:5081/api/items
```

Превышение лимита в учебном режиме:

```bash
curl -i http://localhost:5081/api/items
curl -i http://localhost:5081/api/items
curl -i http://localhost:5081/api/items
curl -i http://localhost:5081/api/items
```

Ожидаемо: последний запрос возвращает `429` с подробным сообщением.

Превышение лимита в боевом режиме:

```bash
curl -i http://localhost:5081/api/items
curl -i http://localhost:5081/api/items
curl -i http://localhost:5081/api/items
```

Ожидаемо: последний запрос возвращает `429` с коротким сообщением `Too many requests.`.

Невалидная конфигурация:

```bash
cd proj3
dotnet run --no-build -- --config appsettings.invalid.json
```

## Тесты

```bash
dotnet test proj3/tests/Proj3.Tests/Proj3.Tests.csproj
```

## Критичные настройки

- `App:AllowedOrigins`
- `App:RateLimiting:*`
- `App:Mode`

Эти параметры проверяются на старте через `ValidateOnStart`.
