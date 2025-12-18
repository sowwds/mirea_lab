# Rate Limiting и Redis Configuration

## Обзор

Проект использует многоуровневую систему rate limiting для защиты API и распределенный Redis для хранения данных о лимитах.

## Архитектура Rate Limiting

### 1. Nginx Level (Первый уровень защиты)
- Лимит: 10 запросов в секунду для `/api/*`
- Burst: 20 запросов
- Конфигурация: `services/php-web/nginx.conf`

### 2. Laravel Level (Второй уровень защиты)
- Использует Redis для распределенного хранения счетчиков
- Разные лимиты для разных типов эндпоинтов:

#### Лимиты по типам эндпоинтов:
- **Общие страницы**: 60 запросов/минуту
- **API прокси (ISS)**: 30 запросов/минуту
- **Внешние API (JWST, Astronomy)**: 20 запросов/минуту
- **Телеметрия API**: 40 запросов/минуту
- **Загрузка файлов**: 10 запросов/минуту

## Redis Configuration

Redis используется для:
1. **Rate Limiting** - хранение счетчиков запросов
2. **Cache** - кеширование данных (CACHE_DRIVER=redis)
3. **Sessions** - хранение сессий (SESSION_DRIVER=redis)
4. **Queues** - очереди задач (QUEUE_CONNECTION=redis)

### Настройка в docker-compose.yml:
```yaml
redis:
  image: redis:7-alpine
  container_name: cache_redis
  command: ["redis-server","--save","60","1000","--loglevel","warning"]
```

### Переменные окружения:
- `REDIS_HOST=redis`
- `REDIS_PORT=6379`
- `CACHE_DRIVER=redis`
- `SESSION_DRIVER=redis`
- `QUEUE_CONNECTION=redis`

## Использование в коде

### В маршрутах (routes/web.php):
```php
Route::middleware(['throttle:60,1'])->group(function () {
    // 60 запросов в минуту
});
```

### В AppServiceProvider:
```php
RateLimiter::for('api', function (Request $request) {
    return Limit::perMinute(60)->by($request->user()?->id ?: $request->ip());
});
```

## Ответ при превышении лимита

При превышении лимита возвращается HTTP 429 с JSON ответом:
```json
{
    "error": "Too Many Requests",
    "message": "Превышен лимит запросов. Попробуйте снова через X секунд.",
    "retry_after": 30
}
```

Заголовки ответа:
- `Retry-After: 30` - секунды до следующего запроса
- `X-RateLimit-Limit: 60` - максимальное количество запросов
- `X-RateLimit-Remaining: 0` - оставшиеся запросы

## Мониторинг

Для мониторинга rate limiting в Redis можно использовать:
```bash
docker exec -it cache_redis redis-cli
KEYS rate_limit:*
```

