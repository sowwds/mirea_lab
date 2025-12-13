# Project Setup and Running Guide

This guide will walk you through setting up and running the Defect Management application, which consists of a Node.js/Express backend and a React/Vite frontend.

---

## 1. Database Setup (PostgreSQL)

The application uses a local PostgreSQL database. Follow these steps to prepare your database:

1.  **Ensure PostgreSQL Server is Running:** Make sure your local PostgreSQL server is active and accessible.
2.  **Connect to PostgreSQL:** Open your terminal or a PostgreSQL client (like `psql` or DBeaver) and connect to your PostgreSQL instance.
3.  **Create a Database:** If you don't have one already, create a database named `defects_db`:
    ```sql
    CREATE DATABASE defects_db;
    ```
4.  **Connect to the New Database:** Connect to the `defects_db` database:
    ```sql
    \c defects_db;
    ```
5.  **Create Tables and Types:** Execute the following SQL commands to create the necessary enums, tables, and update triggers. This ensures your database schema matches the application's expectations.

    **Important:** The `pgcrypto` extension is required for `gen_random_uuid()`.
    ```sql
    -- Enable UUID generation
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";

    -- Create Enum types
    CREATE TYPE "Role" AS ENUM ('ENGINEER', 'MANAGER', 'OBSERVER');
    CREATE TYPE "Priority" AS ENUM ('LOW', 'MEDIUM', 'HIGH');
    CREATE TYPE "DefectStatus" AS ENUM ('NEW', 'IN_PROGRESS', 'UNDER_REVIEW', 'CLOSED', 'CANCELLED');

    -- Create User Table
    CREATE TABLE "User" (
        "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        "email" TEXT NOT NULL UNIQUE,
        "password" TEXT NOT NULL,
        "role" "Role" NOT NULL DEFAULT 'ENGINEER',
        "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- Create Defect Table
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

    -- Trigger to automatically update 'updatedAt' timestamp
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

## 2. Backend Environment Configuration

The backend requires a `.env` file to correctly connect to your PostgreSQL database.

1.  **Open `.env`:** Navigate to the `backend/` directory and open the `.env` file.
2.  **Configure `DATABASE_URL`:** Update the `DATABASE_URL` variable with your PostgreSQL connection string. This is crucial for the backend to connect to your database.

    ```dotenv
    # Example:
    # DATABASE_URL="postgresql://YOUR_USER:YOUR_PASSWORD@localhost:5432/defects_db"

    # JWT Secret Key - Change this to a long, random string in production!
    JWT_SECRET="your-super-secret-key-that-should-be-long-and-random"
    ```

    **Important Considerations for `DATABASE_URL`:**
    *   **User/Role Mismatch:** The most common issue is that the `YOUR_USER` part of the `DATABASE_URL` does not match an existing PostgreSQL username (role) on your system. PostgreSQL will report an error like `"role \"your_user\" does not exist"`.
    *   **Valid Username:** Ensure the `YOUR_USER` matches an *existing* PostgreSQL user.
        *   A common default user is `postgres`. If you haven't created others, try using `postgresql://postgres:YOUR_PASSWORD@localhost:5432/defects_db`.
        *   If you have a different existing user, use that.
    *   **Password:** Make sure `YOUR_PASSWORD` is correct for the specified PostgreSQL user.

3.  **Optional: Creating a Dedicated PostgreSQL User**
    If you prefer not to use the default `postgres` user or if your existing user doesn't work, you can create a dedicated user:
    *   Connect to PostgreSQL as an administrator (e.g., `psql -U postgres`).
    *   Run the following commands, choosing a strong password for `myappuser`:
        ```sql
        CREATE USER myappuser WITH PASSWORD 'mypassword';
        GRANT ALL PRIVILEGES ON DATABASE defects_db TO myappuser;
        ```
    *   Then, update your `DATABASE_URL` to use these new credentials:
        `DATABASE_URL="postgresql://myappuser:mypassword@localhost:5432/defects_db"`

    **Note:** Ensure your `JWT_SECRET` is set to a strong, random value for security.

---

## 3. Running the Application

Once your database is set up and the backend's `.env` file is configured, you can start both the frontend and backend with a single command.

1.  **Open Terminal:** Navigate to the project's root directory in your terminal.
2.  **Install Root Dependencies:** Run `npm install` in the root directory if you haven't already. This installs `npm-run-all` which is used to concurrently run backend and frontend.
3.  **Run Backend and Frontend:** Execute the following command:
    ```bash
    npm run dev
    ```
    This command will:
    *   Start the **backend server** (Node.js/Express) which will listen on `http://localhost:3000`.
    *   Start the **frontend server** (React/Vite) which will be accessible at `http://localhost:5173`.

4.  **Access the Application:** Open your web browser and navigate to `http://localhost:5173`.

---

## 4. Testing the Authentication Flow

After the application is running, you can test the basic authentication:

1.  **Register:** Go to the "Register" page (or directly to `http://localhost:5173/register`) and create a new user with an email and password.
2.  **Login:** After successful registration, you should be redirected to the "Login" page. Log in using the credentials you just created.
3.  **Dashboard:** Upon successful login, you should be automatically taken to the "Defects Dashboard" page (`http://localhost:5173/`).
4.  **Persistence:** Refresh the dashboard page to confirm that your login session persists.
5.  **Logout:** Click the "Logout" button in the navigation bar. You should be returned to the login page.

If you encounter any issues, please check the console logs in your browser (F12) and the terminal where you ran `npm run dev` for error messages.
