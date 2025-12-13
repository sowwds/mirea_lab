I have added a test suite to the backend with 3 integration tests, as you requested.

**What I've done:**
1.  **Refactored the Backend:** I separated the Express app setup (`app.ts`) from the server startup logic (`index.ts`) to make the application testable.
2.  **Installed Testing Libraries:** Added `Jest` and `Supertest` for running tests and making HTTP requests to the app.
3.  **Created Tests:** A new test file `backend/__tests__/app.test.ts` has been created with the following tests:
    - A simple test to ensure the server is running.
    - A test to ensure protected routes return a `401 Unauthorized` error without a token.
    - A multi-step test that verifies user registration, login, and accessing a protected route with the resulting token.

**How to Run the Tests:**
1.  Navigate to the `backend` directory in your terminal.
2.  Run the command:
    ```bash
    npm test
    ```
3.  This will execute Jest, which will find and run the tests. Please ensure your backend is **NOT** running when you run the tests, as the test runner will start its own instance of the app. If you have `npm run dev` running in a terminal, please stop it first (`Ctrl+C`).
