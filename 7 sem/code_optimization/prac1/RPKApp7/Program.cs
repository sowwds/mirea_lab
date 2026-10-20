using System;
using System.Collections.Generic;
using System.Text.Json;

namespace RefactoringExample
{
    public class Program
    {
        private static List<object> requests = new List<object>();

        public static void ConnectToMySQL()
        {
            // логика подключения к MySQL
            Console.WriteLine("connecting to MySQL database...");
            var connectionString = "Server=localhost;Database=test;User=root;Password=12345;";
            Console.WriteLine($"using connection string: {connectionString}");
        }

        public static void ConnectToMongoDB()
        {
            // логика подключения к MongoDB
            Console.WriteLine("connecting to MongoDB database...");
            var connectionString = "mongodb://localhost:27017/test";
            Console.WriteLine($"using connection string: {connectionString}");
        }

        public static void HandleRequest(object req, object res)
        {
            requests.Add(req);
            
            object value = GetValueFromRequest(req);
            object result = ProcessValue(value);
            
            if (IsApiRequest(req))
            {
                HandleApiRequest(value, result, res);
            }
            else
            {
                HandlePageRequest(value, result, res);
            }
        }

        public static Dictionary<string, object> DeepCopy(Dictionary<string, object> obj)
        {
            var newObj = new Dictionary<string, object>();
            foreach (var key in obj.Keys)
            {
                if (obj[key] is Dictionary<string, object> nestedObj)
                {
                    newObj[key] = DeepCopy(nestedObj); // рекурсивное копирование
                }
                else
                {
                    newObj[key] = obj[key]; // поверхностное копирование
                }
            }
            return newObj;
        }

        public static object EvaluateExpression(string expr)
        {
            try
            {
                var dataTable = new System.Data.DataTable();
                var result = dataTable.Compute(expr, "");
                return result;
            }
            catch
            {
                return null;
            }
        }

        private static object GetValueFromRequest(object req)
        {
            // имитация получения значения из запроса
            return new { Value = "test" };
        }

        private static object ProcessValue(object value)
        {
            // имитация обработки значения
            return new { Result = "processed" };
        }

        private static bool IsApiRequest(object req)
        {
            // имитация проверки типа запроса
            return true;
        }

        private static void HandleApiRequest(object value, object result, object res)
        {
            // имитация обработки API запроса
            Console.WriteLine("handling API request");
            Console.WriteLine($"value: {JsonSerializer.Serialize(value)}");
            Console.WriteLine($"result: {JsonSerializer.Serialize(result)}");
        }

        private static void HandlePageRequest(object value, object result, object res)
        {
            // имитация обработки page запроса
            Console.WriteLine("handling page request");
            Console.WriteLine($"value: {JsonSerializer.Serialize(value)}");
            Console.WriteLine($"result: {JsonSerializer.Serialize(result)}");
        }

        public static void ConfigureServiceA()
        {
            var config = new Dictionary<string, object>
            {
                { "timeout", 30 },
                { "retryCount", 3 },
                { "logLevel", "debug" }
            };
            Console.WriteLine("configuring service A");
        }

        public static void ConfigureServiceB()
        {
            var config = new Dictionary<string, object>
            {
                { "timeout", 30 },
                { "retryCount", 3 },
                { "logLevel", "info" }
            };
            Console.WriteLine("configuring service B");
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("starting application...");
            
            ConnectToMySQL();
            ConnectToMongoDB();
            
            var request = new { Path = "/api/data", Method = "GET" };
            var response = new { Status = 200, Content = "" };
            HandleRequest(request, response);
            
            var original = new Dictionary<string, object>
            {
                { "name", "test" },
                { "nested", new Dictionary<string, object> { { "value", 123 } } }
            };
            var copy = DeepCopy(original);
            Console.WriteLine($"original: {JsonSerializer.Serialize(original)}");
            Console.WriteLine($"copy: {JsonSerializer.Serialize(copy)}");
            
            var result = EvaluateExpression("2 + 3 * 4");
            Console.WriteLine($"eval result: {result}");
            
            ConfigureServiceA();
            ConfigureServiceB();
            
            for (int i = 0; i < 100; i++)
            {
                requests.Add(new { Id = i, Data = new string('x', 1000) });
            }
            Console.WriteLine($"requests count: {requests.Count}");
            
            Console.WriteLine("application completed.");
        }
    }
}