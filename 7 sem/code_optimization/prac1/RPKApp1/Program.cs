using System;
using System.Collections.Generic;
using System.Text.Json;

namespace RefactoringExample
{
    public class Program
    {
        private static List<Dictionary<string, object>> users = new List<Dictionary<string, object>>();
        
        // конфигурация
        private static Dictionary<string, object> config = new Dictionary<string, object>
        {
            { "reportType", "detailed" }
        };

        public static void ProcessUser(Dictionary<string, object> userData)
        {
            // валидация возраста
            if ((int)userData["age"] < 18)
            {
                Console.WriteLine("User is too young");
                return;
            }

            // валидация имени
            if (((string)userData["name"]).Length > 100)
            {
                Console.WriteLine("Name is too long");
                return;
            }

            // расчет скидки
            double discount = 0;
            if ((bool)userData["isVIP"])
            {
                discount = 0.15;
            }
            else
            {
                if ((int)userData["orders"] > 5)
                {
                    discount = 0.1;
                }
                else
                {
                    discount = 0;
                }
            }

            // создание пользователя
            var a = new Dictionary<string, object>
            {
                { "name", userData["name"] },
                { "age", userData["age"] },
                { "discount", discount },
                { "isActive", true }
            };

            // сохранение
            users.Add(a);
            Console.WriteLine("User saved: " + JsonSerializer.Serialize(a));

            // генерация отчета
            GenerateReport(
                (string)a["name"], 
                (int)a["age"], 
                (double)a["discount"], 
                (bool)a["isActive"], 
                DateTime.Now, 
                (string)config["reportType"]
            );
        }

        public static void GenerateReport(string n, int a, double d, bool i, DateTime dt, string t)
        {
            Console.WriteLine("Starting report generation...");
            
            // проверка типа отчета
            if (t == "detailed")
            {
                Console.WriteLine("=== DETAILED REPORT ===");
            }
            else if (t == "summary")
            {
                Console.WriteLine("=== SUMMARY REPORT ===");
            }
            else
            {
                Console.WriteLine("=== STANDARD REPORT ===");
            }
            
            // форматирование заголовка
            string header = $"Report for: {n}";
            Console.WriteLine(header);
            Console.WriteLine(new string('-', header.Length));
            
            // основная информация (дублирование формата вывода)
            Console.WriteLine($"Age: {a}");
            Console.WriteLine($"Discount: {d:P0}");
            Console.WriteLine($"Active status: {(i ? "Active" : "Inactive")}");
            Console.WriteLine($"Report date: {dt:yyyy-MM-dd HH:mm:ss}");
            
            // генерация раздела с деталями
            Console.WriteLine("\nDETAILS:");
            Console.WriteLine("Account information:");
            
            if (a < 25)
            {
                if (d > 0.1)
                {
                    Console.WriteLine("Young customer with high discount");
                }
                else
                {
                    Console.WriteLine("Young customer with standard discount");
                }
            }
            else if (a >= 25 && a < 40)
            {
                if (d > 0.15)
                {
                    Console.WriteLine("Adult customer with premium discount");
                }
                else
                {
                    Console.WriteLine("Adult customer with standard discount");
                }
            }
            else
            {
                if (d > 0.2)
                {
                    Console.WriteLine("Senior customer with special discount");
                }
                else
                {
                    Console.WriteLine("Senior customer with standard discount");
                }
            }
            
            // дополнительная информация на основе статуса
            if (i)
            {
                Console.WriteLine("Customer is currently active");
                Console.WriteLine("Last activity: " + dt.AddDays(-7).ToString("yyyy-MM-dd"));
            }
            else
            {
                Console.WriteLine("Customer is not active");
                Console.WriteLine("Account will be archived soon");
            }
            
            // генерация рекомендаций (искусственно сложная логика)
            Console.WriteLine("\nRECOMMENDATIONS:");
            if (!i && a < 30)
            {
                Console.WriteLine("Consider reactivation campaign for young inactive customer");
            }
            else if (i && d < 0.05)
            {
                Console.WriteLine("Consider loyalty program for active customer with low discount");
            }
            else if (a > 60 && d < 0.1)
            {
                Console.WriteLine("Consider senior discount program");
            }
            
            // футер отчета
            Console.WriteLine("\n" + new string('=', 50));
            Console.WriteLine("Report generated by: Automated Reporting System");
            Console.WriteLine($"Generation timestamp: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            Console.WriteLine(new string('=', 50));
            
            // имитация сохранения отчета в разные форматы
            if (t == "detailed")
            {
                Console.WriteLine("Saving detailed report to database...");
                // Имитация длительной операции
                System.Threading.Thread.Sleep(100);
                Console.WriteLine("Report saved successfully");
            }
            
            Console.WriteLine($"Report for {n} completed at {DateTime.Now:HH:mm:ss}");
        }

        public static double CalculateDiscount(bool isVIP, int orders)
        {
            double discount = 0;
            if (isVIP)
            {
                discount = 0.15;
            }
            else
            {
                if (orders > 5)
                {
                    discount = 0.1;
                }
                else
                {
                    discount = 0;
                }
            }
            return discount;
        }

        static void Main(string[] args)
        {
            var userData = new Dictionary<string, object>
            {
                { "name", "John Doe" },
                { "age", 25 },
                { "isVIP", true },
                { "orders", 10 }
            };
            
            ProcessUser(userData);
            
            var discount = CalculateDiscount(true, 10);
            Console.WriteLine("Calculated discount: " + discount);
        }
    }
}