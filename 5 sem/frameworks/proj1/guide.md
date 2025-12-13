# Руководство по запуску и настройке

Это краткое руководство по настройке и запуску приложения.

---

## 1. Настройка Базы Данных (PostgreSQL)

1.  **Запустите PostgreSQL:** Убедитесь, что ваш локальный сервер PostgreSQL запущен.
2.  **Создайте БД:** В `psql` или другом клиенте выполните:
    ```sql
    CREATE DATABASE defects_db;
    ```
3.  **Создайте таблицы:** Подключитесь к `defects_db` (`\c defects_db`) и выполните следующий скрипт:
    ```sql
    -- Включение генерации UUID
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";

    -- Создание ENUM типов
    CREATE TYPE "Role" AS ENUM ('ENGINEER', 'MANAGER', 'OBSERVER');
    CREATE TYPE "Priority" AS ENUM ('LOW', 'MEDIUM', 'HIGH');
    CREATE TYPE "DefectStatus" AS ENUM ('NEW', 'IN_PROGRESS', 'UNDER_REVIEW', 'CLOSED', 'CANCELLED');

    -- Создание таблицы пользователей
    CREATE TABLE "User" (
        "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        "email" TEXT NOT NULL UNIQUE,
        "password" TEXT NOT NULL,
        "role" "Role" NOT NULL DEFAULT 'ENGINEER',
        "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- Создание таблицы дефектов
    CREATE TABLE "Defect" (
        "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        "title" TEXT NOT NULL,
        "description" TEXT,
        "priority" "Priority" NOT NULL DEFAULT 'LOW',
        "status" "DefectStatus" NOT NULL DEFAULT 'NEW',
        "assigneeId" UUID NOT NULL REFERENCES "User"(id) ON DELETE CASCADE,
        "attachments" TEXT[] NOT NULL DEFAULT '{}',
        "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- Триггер для автоматического обновления 'updatedAt'
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
       NEW."updatedAt" = now();
       RETURN NEW;
    END;
    $$ language 'plpgsql';

    CREATE TRIGGER update_user_updated_at BEFORE UPDATE ON "User" FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
    CREATE TRIGGER update_defect_updated_at BEFORE UPDATE ON "Defect" FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
    ```

---

## 2. Настройка Backend

1.  **Откройте `backend/.env`**.
2.  **Укажите `DATABASE_URL`** для подключения к вашей локальной базе данных.
    ```dotenv
    # Пример:
    DATABASE_URL="postgresql://YOUR_USER:YOUR_PASSWORD@localhost:5432/defects_db"
    ```
    Убедитесь, что `YOUR_USER` и `YOUR_PASSWORD` соответствуют существующему пользователю в PostgreSQL. Распространенной ошибкой является `"role "user" does not exist"`, которая означает, что указанный пользователь не найден в СУБД.

---

## 3. Запуск Приложения

1.  **Установите зависимости:** В корневой директории проекта выполните `npm install`.
2.  **Запустите проект:** Выполните команду:
    ```bash
    npm run dev
    ```
    Эта команда одновременно запустит:
    *   **Backend сервер** на `http://localhost:3000`
    *   **Frontend сервер** на `http://localhost:5173`

3.  **Откройте приложение:** Перейдите в браузере по адресу `http://localhost:5173`.
