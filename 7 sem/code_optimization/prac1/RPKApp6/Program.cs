using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace RefactoringExample
{
    public class DataImporter
    {
        private List<Dictionary<string, string>> data = new List<Dictionary<string, string>>();

        public async Task ImportData(string filePath)
        {
            try
            {
                // чтение файла
                string[] lines = File.ReadAllLines(filePath);
                
                // парсинг CSV
                string[] headers = lines[0].Split(',');
                
                var processedData = lines.Skip(1)
                    .Where(line => !string.IsNullOrEmpty(line))
                    .Select(line => 
                    {
                        string[] values = line.Split(',');
                        var d = new Dictionary<string, string>(); // непонятное имя 'd'
                        for (int j = 0; j < headers.Length; j++)
                        {
                            d[headers[j].Trim()] = values[j].Trim();
                        }
                        return d;
                    });
                
                // валидация
                foreach (var record in processedData)
                {
                    if (record["status"] == "processed")
                    {
                        data.Add(record);
                    }
                    else
                    {
                        Console.WriteLine($"Skipping record with status: {record["status"]}");
                    }
                }
                
                // сохранение в базу
                await SaveToDatabase();
            }
            catch (Exception ex)
            {
                
            }
        }

        public bool IsProcessed(Dictionary<string, string> record)
        {
            return record["status"] == "processed";
        }

        private async Task SaveToDatabase()
        {
            // имитация сохранения в базу данных
            foreach (var record in data)
            {
                Console.WriteLine($"Saving record: {record["id"]}");
                await Task.Delay(10); // имитация асинхронной операции
            }
        }
    }

    public class OrderManager
    {
        public void Proc(List<Dictionary<string, string>> data)
        {
            // обработка данных
            Console.WriteLine($"Processing {data.Count} records");
        }

        public void Mgr(List<Dictionary<string, string>> data)
        {
            // управление данными
            Console.WriteLine($"Managing {data.Count} records");
        }
    }

    class Program
    {
        static async Task Main(string[] args)
        {
            // создание тестового CSV файла
            string testFilePath = "test_data.csv";
            CreateTestCsvFile(testFilePath);

            var importer = new DataImporter();
            await importer.ImportData(testFilePath);

            var orderManager = new OrderManager();
            
            // создание тестовых данных
            var testData = new List<Dictionary<string, string>>
            {
                new Dictionary<string, string> { {"id", "1"}, {"status", "processed"} },
                new Dictionary<string, string> { {"id", "2"}, {"status", "pending"} }
            };

            orderManager.Proc(testData);
            orderManager.Mgr(testData);

            // демонстрация проверки статуса
            foreach (var record in testData)
            {
                if (importer.IsProcessed(record))
                {
                    Console.WriteLine($"Record {record["id"]} is processed");
                }
                else
                {
                    Console.WriteLine($"Record {record["id"]} is not processed");
                }
            }

            // TODO: реализовать экспорт данных
            // public void ExportData(string filePath) 
            // {
            //     // код экспорта
            // }

            Console.WriteLine("Application completed. Press any key to exit.");
            Console.ReadKey();
        }

        static void CreateTestCsvFile(string filePath)
        {
            // создание тестового CSV файла
            string[] lines = 
            {
                "id,name,status",
                "1,Product A,processed",
                "2,Product B,pending",
                "3,Product C,processed",
                "4,Product D,rejected"
            };

            File.WriteAllLines(filePath, lines);
            Console.WriteLine($"Created test CSV file: {filePath}");
        }
    }
}