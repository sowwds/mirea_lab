The error `ERROR: relation "defect" does not exist` when trying to select from the `Defect` table is due to **case-sensitivity in PostgreSQL table names**.

When tables are created with double quotes (e.g., `CREATE TABLE "Defect" (...)`), PostgreSQL stores the name exactly as provided, including its capitalization. If you then try to query it without quotes (e.g., `SELECT * FROM Defect;`), PostgreSQL converts the name to lowercase (`defect`) and looks for a table named `defect`, which doesn't exist.

**To correctly query your tables, you need to use double quotes around the table name:**

```sql
SELECT * FROM "Defect";
SELECT * FROM "User";
```

**To see all tables and their exact names in your `defects_db` database:**

1.  Connect to your `defects_db` database in `psql`:
    ```bash
    psql -U YOUR_USER -d defects_db
    ```
    (Replace `YOUR_USER` with the actual user you are connecting with.)
2.  Once connected, type:
    ```sql
    \dt
    ```
    This command lists all tables in the current database and will show you their exact, case-sensitive names.
