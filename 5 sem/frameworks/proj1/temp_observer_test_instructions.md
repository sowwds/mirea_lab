The implementation for the `OBSERVER` role is now complete on both the backend and frontend.
5
**Backend Changes:**
*   `OBSERVER` users are now denied from creating (POST) and updating (PUT) defects.
*   `ENGINEER` and `OBSERVER` users are now explicitly denied from deleting (DELETE) defects (only `MANAGER` can delete).

**Frontend Changes:**
*   The 'Create Defect' button is hidden for `OBSERVER` users.
*   'Edit' buttons are hidden for `OBSERVER` users.
*   'Delete' buttons are only visible for `MANAGER` users.

To fully test this, you will need to manually change a user's role in your PostgreSQL database. For example, if you want to make a user an `OBSERVER`:

```sql
UPDATE "User" SET "role" = 'OBSERVER' WHERE email = 'your_registered_email@example.com';
```

Or to make a user a `MANAGER` (to test deletion):

```sql
UPDATE "User" SET "role" = 'MANAGER' WHERE email = 'your_registered_email@example.com';
```

After updating the role in the database, log in with that user in the application and observe the UI changes and backend permissions.

Please let me know if you encounter any issues or if the role-based restrictions work as expected.
