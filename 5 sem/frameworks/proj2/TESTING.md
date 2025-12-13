# Инструкция по тестированию API

Этот документ содержит `curl` команды для проверки функциональности

## Переменные

```bash
export API_URL="http://localhost:8000/api/v1"
export TEST_EMAIL="test-user-$(date +%s)@example.com"
export TEST_PASSWORD="password123"
```

## 1. Регистрация пользователя

Создаем нового пользователя. Ожидаем ответ `201 Created` и данные пользователя.

```bash
curl -X POST "${API_URL}/users/register" \
-H "Content-Type: application/json" \
-d '{
  "name": "Test User",
  "email": "'"$TEST_EMAIL"'",
  "password": "'"$TEST_PASSWORD"'"
}'
```

## 2. Повторная регистрация (ожидаем ошибку)

Пытаемся создать пользователя с тем же email. Ожидаем ответ `409 Conflict`.

```bash
curl -X POST "${API_URL}/users/register" \
-H "Content-Type: application/json" \
-d '{
  "name": "Another User",
  "email": "'"$TEST_EMAIL"'",
  "password": "anotherpassword"
}'
```

## 3. Вход пользователя (логин)

Используем данные для входа. Ожидаем ответ `200 OK` и JWT токен.

```bash
# Выполняем запрос и сохраняем токен в переменную AUTH_TOKEN
export AUTH_TOKEN=$(curl -s -X POST "${API_URL}/users/login" \
-H "Content-Type: application/json" \
-d '{
  "email": "'"$TEST_EMAIL"'",
  "password": "'"$TEST_PASSWORD"'"
}' | jq -r .token)

# Проверяем, что токен получен
echo "Auth Token: $AUTH_TOKEN"
```

## 4. Доступ к защищенному маршруту без токена

Пытаемся получить профиль без токена. Ожидаем ответ `401 Unauthorized`.

```bash
curl -I -X GET "${API_URL}/users/profile"
```

## 5. Доступ к профилю с валидным токеном

Получаем профиль с токеном авторизации. Ожидаем ответ `200 OK` и данные профиля.

```bash
curl -X GET "${API_URL}/users/profile" \
-H "Authorization: Bearer ${AUTH_TOKEN}"
```

## 6. Создание заказа

Создаем новый заказ. Ожидаем ответ `201 Created` и данные заказа.

```bash
curl -X POST "${API_URL}/orders" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${AUTH_TOKEN}" \
-d '{
  "items": [
    { "productId": 1, "quantity": 2 },
    { "productId": 2, "quantity": 1 }
  ]
}'
```

## 7. Получение списка заказов

Получаем все заказы текущего пользователя. Ожидаем `200 OK` и массив с одним заказом.

```bash
curl -X GET "${API_URL}/orders" \
-H "Authorization: Bearer ${AUTH_TOKEN}"
```

## 8. Получение конкретного заказа

Получаем созданный заказ по его ID (в примере ID=1). Ожидаем `200 OK` и данные заказа.

```bash
# Замените '1' на реальный ID заказа из предыдущего шага, если он другой
curl -X GET "${API_URL}/orders/1" \
-H "Authorization: Bearer ${AUTH_TOKEN}"
```
