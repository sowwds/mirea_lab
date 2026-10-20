using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace RefactoringExample
{
    public class Program
    {
        // глобальная конфигурация
        private static Dictionary<string, object> cfg = new Dictionary<string, object>
        {
            { "dbConnection", "Server=localhost;Database=test;User=sa;Password=12345;" },
            { "maxRetries", 3 }
        };

        public static void Main(string[] args)
        {
            Console.WriteLine("=== Starting Application ===\n");
            
            InitApp();
            
            // автоматическое тестирование (упрощённое)
            RunTests();
            
            Console.WriteLine("\n=== Application Completed ===");
        }

        public static void InitApp()
        {
            // настройка приложения
            Console.WriteLine("Initializing application...");
            
            // длинный метод настройки middleware
            SetupMiddleware();
            
            // запуск приложения
            Console.WriteLine("Application started!");
        }

        public static void SetupMiddleware()
        {
            Console.WriteLine("Setting up middleware...");
            
            Task.Run(async () =>
            {
                try
                {
                    Console.WriteLine("Checking authentication...");
                    var authResult = await CheckAuth("user123");
                    if (authResult.IsAuthenticated)
                    {
                        Console.WriteLine($"User {authResult.UserId} authenticated successfully");
                        
                        Console.WriteLine("Fetching user data...");
                        var userData = await GetUserData(authResult.UserId);
                        if (userData.IsValid)
                        {
                            Console.WriteLine($"User role: {userData.Role}");
                            
                            Console.WriteLine("Checking permissions...");
                            var permissions = await GetPermissions(userData.Role);
                            if (permissions.CanAccess)
                            {
                                Console.WriteLine("Access granted to all resources");
                            }
                            else
                            {
                                Console.WriteLine("Access denied - insufficient permissions");
                            }
                        }
                        else
                        {
                            Console.WriteLine("Invalid user data - cannot proceed");
                        }
                    }
                    else
                    {
                        Console.WriteLine("Authentication failed - access denied");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error in middleware setup: {ex.Message}");
                }
            }).Wait();
            
            Console.WriteLine("Middleware setup completed");
        }

        public static bool ValidateUser(Dictionary<string, object> userData)
        {
            Console.WriteLine($"Validating user: {userData["username"]}");
            
            if (!userData.ContainsKey("username") || string.IsNullOrEmpty(userData["username"].ToString()))
            {
                Console.WriteLine("Validation failed: username is required");
                return false;
            }

            userData["username"] = userData["username"].ToString().Trim().ToLower();
            Console.WriteLine($"Sanitized username: {userData["username"]}");
            
            if (!userData.ContainsKey("email") || !userData["email"].ToString().Contains("@"))
            {
                Console.WriteLine("Validation failed: invalid email format");
                return false;
            }

            if (!IsValidEmail(userData["email"].ToString()))
            {
                Console.WriteLine("Validation failed: email is invalid");
                return false;
            }

            Console.WriteLine("User validation successful");
            return true;
        }

        public static bool IsValidEmail(string email)
        {
            Console.WriteLine($"Validating email: {email}");
            if (!email.Contains("@"))
            {
                return false;
            }
            return true;
        }

        public static bool ChkUsrInpt(string val)
        {
            Console.WriteLine($"Checking user input: {val}");
            return !string.IsNullOrEmpty(val) && val.Length > 2;
        }

        // имитация асинхронных методов
        public static async Task<AuthResult> CheckAuth(string username)
        {
            Console.WriteLine($"Authenticating user: {username}");
            await Task.Delay(100);
            return new AuthResult { IsAuthenticated = true, UserId = 123 };
        }

        public static async Task<UserData> GetUserData(int userId)
        {
            Console.WriteLine($"Fetching data for user ID: {userId}");
            await Task.Delay(100);
            return new UserData { IsValid = true, Role = "admin" };
        }

        public static async Task<PermissionResult> GetPermissions(string role)
        {
            Console.WriteLine($"Checking permissions for role: {role}");
            await Task.Delay(100);
            return new PermissionResult { CanAccess = true };
        }

        /*
         * Этот метод обрабатывает пользовательский ввод, валидирует его,
         * создает пользователя и сохраняет его в базу данных. Пока работает...
         */
        public static void ProcessUserRegistration(Dictionary<string, object> userData)
        {
            Console.WriteLine("\n--- Processing User Registration ---");
            
            if (!ValidateUser(userData))
            {
                Console.WriteLine("User registration failed: validation error");
                return;
            }

            var user = new Dictionary<string, object>
            {
                { "username", userData["username"] },
                { "email", userData["email"] },
                { "createdAt", DateTime.Now }
            };

            // сохранение в базу
            Console.WriteLine($"User {user["username"]} created successfully");

            // отправка email
            SendWelcomeEmail(user["email"].ToString());
            
            Console.WriteLine("--- User Registration Completed ---\n");
        }

        public static void SendWelcomeEmail(string email)
        {
            // имитация отправки email
            Console.WriteLine($"Sending welcome email to {email}");
        }

        public static void RunTests()
        {
            Console.WriteLine("\n=== Running Tests ===\n");
            
            // тестирование проверки пользовательского ввода
            Console.WriteLine("Testing user input validation:");
            Console.WriteLine($"Result for 'abc': {ChkUsrInpt("abc")}");
            Console.WriteLine($"Result for 'a': {ChkUsrInpt("a")}");
            
            // тестирование валидации email
            Console.WriteLine("\nTesting email validation:");
            Console.WriteLine($"Result for 'test@example.com': {IsValidEmail("test@example.com")}");
            Console.WriteLine($"Result for 'invalid-email': {IsValidEmail("invalid-email")}");
            
            // тестирование регистрации пользователей
            Console.WriteLine("\nTesting user registration:");
            
            var testUser1 = new Dictionary<string, object>
            {
                { "username", "  JohnDoe  " },
                { "email", "john@example.com" }
            };
            ProcessUserRegistration(testUser1);
            
            var testUser2 = new Dictionary<string, object>
            {
                { "username", "Alice" },
                { "email", "invalid-email" } // невалидный email
            };
            ProcessUserRegistration(testUser2);
            
            var testUser3 = new Dictionary<string, object>
            {
                { "username", "Bob" },
                { "email", "bob@example.com" }
            };
            ProcessUserRegistration(testUser3);
            
            Console.WriteLine("=== Tests Completed ===\n");
        }
    }
    
    public class AuthResult
    {
        public bool IsAuthenticated { get; set; }
        public int UserId { get; set; }
    }

    public class UserData
    {
        public bool IsValid { get; set; }
        public string Role { get; set; }
    }

    public class PermissionResult
    {
        public bool CanAccess { get; set; }
    }
}