using System;
using System.Collections.Generic;
using System.Linq;

namespace RefactoringExample
{
    public class Order
    {
        public List<Dictionary<string, object>> Items { get; set; }
        public Dictionary<string, object> User { get; set; }
        public string Status { get; set; } = "pending";
        
        private static Mailer mailer = new Mailer();
        private static SMSService smsService = new SMSService();
        
        public static List<Dictionary<string, object>> Inventory = new List<Dictionary<string, object>>();

        public Order(List<Dictionary<string, object>> items, Dictionary<string, object> user)
        {
            Items = items;
            User = user;
        }

        public Dictionary<string, object> PlaceOrder()
        {
            // проверка наличия товаров
            foreach (var item in Items)
            {
                var product = Inventory.Find(p => (string)p["id"] == (string)item["id"]);
                if (product == null || (int)product["stock"] < (int)item["quantity"])
                {
                    throw new Exception($"Product {item["id"]} is out of stock");
                }
            }

            // обновление инвентаря
            foreach (var item in Items)
            {
                var product = Inventory.Find(p => (string)p["id"] == (string)item["id"]);
                product["stock"] = (int)product["stock"] - (int)item["quantity"];
            }

            // создание записи в БД
            var orderRecord = DB.Create("orders", new Dictionary<string, object>
            {
                { "userId", User["id"] },
                { "items", Items },
                { "status", Status }
            });

            // отправка email
            mailer.Send(new Dictionary<string, object>
            {
                { "to", User["email"] },
                { "subject", "Your order is placed!" },
                { "text", $"Your order {orderRecord["id"]} has been placed." }
            });

            // отправка SMS
            if (User.ContainsKey("phone"))
            {
                smsService.DoStuff((string)User["phone"], $"Your order {orderRecord["id"]} is placed.");
            }

            // Логирование
            Console.WriteLine($"Order placed: {orderRecord["id"]}");
            return orderRecord;
        }
    }

    public class Mailer
    {
        public void Send(Dictionary<string, object> options)
        {
            Console.WriteLine($"Sending email to: {options["to"]}");
        }
    }

    public class SMSService
    {
        public void DoStuff(string phone, string message)
        {
            Console.WriteLine($"Sending SMS to {phone}: {message}");
        }
    }

    public static class DB
    {
        public static Dictionary<string, object> Create(string table, Dictionary<string, object> data)
        {
            // Заглушка для создания записи в БД
            data["id"] = Guid.NewGuid().ToString();
            return data;
        }
    }

    public class Program
    {
        public static Dictionary<string, object> CreateProduct(string name, string desc, decimal price, string cat, 
                                                             int stock, string supplier, List<string> tags, bool isActive)
        {
            return new Dictionary<string, object>
            {
                { "name", name },
                { "description", desc },
                { "price", price },
                { "category", cat },
                { "stock", stock },
                { "supplier", supplier },
                { "tags", tags },
                { "isActive", isActive }
            };
        }

        static void Main(string[] args)
        {
            // инициализация инвентаря
            Order.Inventory.Add(new Dictionary<string, object>
            {
                { "id", "prod1" },
                { "stock", 10 }
            });

            // создание заказа
            var order = new Order(
                new List<Dictionary<string, object>>
                {
                    new Dictionary<string, object> { { "id", "prod1" }, { "quantity", 2 } }
                },
                new Dictionary<string, object>
                {
                    { "id", "user1" },
                    { "email", "user@example.com" },
                    { "phone", "123-456-7890" }
                }
            );

            try
            {
                var result = order.PlaceOrder();
                Console.WriteLine($"Order created: {result["id"]}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }

            var product = CreateProduct(
                "Laptop", 
                "Gaming laptop", 
                999.99m, 
                "Electronics", 
                5, 
                "TechSupplier", 
                new List<string> { "gaming", "laptop", "electronics" }, 
                true
            );
            
            Console.WriteLine($"Product created: {product["name"]}");
        }
    }
}