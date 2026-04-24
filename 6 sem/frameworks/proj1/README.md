# proj1 Ефремов Алексей Игоревич

Мини веб-служба на `C#` и `ASP.NET Core` для практического задания про конвейер обработки запросов.

## Что реализовано

- `GET /api/items` возвращает список учебных задач.
- `GET /api/items/{id}` возвращает задачу по идентификатору.
- `POST /api/items` создает новую задачу в памяти процесса.
- Данные хранятся без базы данных во встроенном in-memory сервисе.
- Ошибки возвращаются в едином формате: `code`, `message`, `requestId`.
- Конвейер состоит из трех middleware: логирование запроса/ответа, обработка исключений, измерение времени выполнения.

## Предметная область

В проекте используется каталог учебных задач.

Поля сущности:

- `id`
- `title`
- `subject`
- `difficulty`
- `estimatedHours`

Примеры правил валидации:

- название задачи не должно быть пустым;
- предмет не должен быть пустым;
- сложность должна быть от `1` до `10`;
- ожидаемое количество часов не должно быть отрицательным.

## Запуск

Если `dotnet` еще не установлен, для `Arch Linux` поставьте SDK:

```bash
sudo pacman -S dotnet-sdk-8.0 aspnet-runtime-8.0
```

```bash
cd proj1
dotnet restore
dotnet run --urls http://localhost:5078
```

После запуска можно открыть Swagger:

```text
http://localhost:5078/swagger
```

## Тесты

Проверка предметной логики и HTTP-сценариев:

```bash
dotnet test proj1/tests/Proj1.Tests/Proj1.Tests.csproj
```

## Ручной прогон

Получить список:

```bash
curl http://localhost:5078/api/items
```

Получить несуществующий элемент:

```bash
curl -i http://localhost:5078/api/items/999
```

Создать элемент:

```bash
curl -i \
  -H "Content-Type: application/json" \
  -d '{"title":"Prepare defense","subject":"Frameworks","difficulty":6,"estimatedHours":5}' \
  http://localhost:5078/api/items
```

Получить ошибку валидации:

```bash
curl -i \
  -H "Content-Type: application/json" \
  -d '{"title":"","subject":"Frameworks","difficulty":6,"estimatedHours":5}' \
  http://localhost:5078/api/items
```
