The `permission denied for table User` error, even after `GRANT ALL PRIVILEGES ON DATABASE`, indicates a common PostgreSQL privilege nuance. `GRANT ALL PRIVILEGES ON DATABASE` grants access to the database itself, but doesn't automatically extend to all *objects within that database* (like tables, sequences, or custom types, especially if they were created by a different user or before the grant).

**To ensure your user has full permissions on all existing and future objects within the `defects_db` database, please follow these steps:**

1.  **Connect to your `defects_db` database as an administrative user** (e.g., the `postgres` user, or the user who originally created the tables).
    ```bash
    psql -U postgres -d defects_db
    ```
    (Replace `postgres` with your admin user if different, and enter your password when prompted.)

2.  **Execute the following SQL commands:**
    These commands will grant comprehensive privileges to `defects_db_user` (replace `defects_db_user` with the actual username you're using in your `DATABASE_URL` from `backend/.env`):
    ```sql
    -- Grant privileges on the 'public' schema (where your tables reside)
    GRANT ALL ON SCHEMA public TO defects_db_user;

    -- Grant privileges on ALL existing tables, sequences, functions, and types in the 'public' schema
    GRANT ALL ON ALL TABLES IN SCHEMA public TO defects_db_user;
    GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO defects_db_user;
    GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO defects_db_user;
    GRANT ALL ON ALL TYPES IN SCHEMA public TO defects_db_user;

    -- Grant default privileges for any FUTURE tables, sequences, or functions created in the 'public' schema
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO defects_db_user;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO defects_db_user;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO defects_db_user;
    ```
    (Remember to replace `defects_db_user` with your actual username.)

3.  **Exit `psql`:**
    ```bash
    \q
    ```

4.  **Verify `DATABASE_URL`:** Double-check your `backend/.env` file to ensure the `DATABASE_URL` is using the correct username (e.g., `defects_db_user`) and password.

After performing these steps, the backend server should restart. Please check its logs to confirm it starts successfully and then re-test the application in your browser.
