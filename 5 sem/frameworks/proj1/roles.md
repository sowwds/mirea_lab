# Role-Based Access Control (RBAC) in the Defect Management System

This document outlines the available user roles and how role-based access control is implemented in the application.

---

## 1. Available User Roles

The application defines three distinct roles, each with different levels of access and permissions:

*   **`ENGINEER`**:
    *   **Default Role:** This is the default role assigned to new users upon registration.
    *   **Permissions:** Can create, view, and edit defects.

*   **`MANAGER`**:
    *   **Permissions:** Possesses all `ENGINEER` permissions (create, view, edit defects), and additionally has the authority to **delete defects**.

*   **`OBSERVER`**:
    *   **Permissions:** (Currently not explicitly implemented for differentiation but can be easily extended) This role is envisioned for users who should only have read-only access to defects. They would be able to view defects but not create, edit, or delete them.

---

## 2. How Role-Based Access Control is Implemented

The RBAC system relies on JSON Web Tokens (JWTs) for authentication and a middleware for authorization:

1.  **Authentication (Login/Registration):**
    *   When a user successfully registers or logs in, the backend generates a **JWT token**.
    *   This token contains essential user information, including the `userId` and the user's `role`.

2.  **Authentication Middleware (`authenticateToken`):**
    *   Any route that requires authentication (like all defect-related routes) uses the `authenticateToken` middleware.
    *   This middleware intercepts incoming requests, verifies the authenticity and validity of the JWT token present in the request headers.
    *   If the token is valid, it extracts the `userId` and `role` from the token and attaches them to the `req.user` object (e.g., `req.user.userId`, `req.user.role`). If the token is invalid or missing, the request is rejected with a `401 Unauthorized` or `403 Forbidden` status.

3.  **Authorization Checks (In Route Handlers):**
    *   Within specific route handlers, an explicit check is performed on `req.user.role` to determine if the authenticated user has the necessary permissions for the requested action.
    *   **Example (Deletion):** In the `backend/src/routes/defects.ts` file, the `DELETE /api/defects/:id` route handler includes a check:
        ```typescript
        if (req.user.role !== 'MANAGER') {
            return res.status(403).json({ error: 'Permission denied. Only managers can delete defects.' });
        }
        ```
        This ensures that only users with the `MANAGER` role can successfully delete defects.

---

## 3. How to Test Different Roles

*   Currently, all users registered through the frontend default to the `ENGINEER` role.
*   To test the `MANAGER` role's delete functionality, you would need to **manually update a user's role in your PostgreSQL database**. For example:
    ```sql
    UPDATE "User" SET "role" = 'MANAGER' WHERE email = 'your_registered_email@example.com';
    ```
    After updating the role, log in as that user, and you should be able to delete defects from the dashboard.

---

## 4. Future Enhancements (Extending RBAC)

*   **`OBSERVER` Role Implementation:** To fully implement the `OBSERVER` role, you would add similar `req.user.role` checks to `POST` (create), `PUT` (edit), and `DELETE` routes for defects, ensuring that `OBSERVER` users are blocked from these actions.
*   **Frontend UI Restrictions:** The frontend can also be enhanced to dynamically show/hide buttons or UI elements based on the logged-in user's role, providing a more intuitive user experience.
*   **More Granular Permissions:** For more complex scenarios, you could implement more granular permissions (e.g., can edit only defects assigned to them) by adding more sophisticated checks in the middleware or route handlers.
