using System;
using System.Collections.Generic;

namespace RefactoringExample
{
    public class Calculator
    {
        public List<string> history = new List<string>();
        public double LastResult = 0;

        public double Add(double a, double b)
        {
            return RecordResult(a + b, $"Added {b} to number");
        }

        // Прежнее имя оставлено для совместимости с исходным кодом.
        public double addNumber(double a, double b) => Add(a, b);

        public double PerformOperation(string operation, double a, double b)
        {
            double result;
            string? historyEntry = null;
            switch (operation)
            {
                case "add":
                    result = a + b;
                    historyEntry = $"Added {a} and {b}";
                    break;
                case "subtract":
                    result = a - b;
                    historyEntry = $"Subtracted {b} from {a}";
                    break;
                case "multiply":
                    result = a * b;
                    historyEntry = $"Multiplied {a} by {b}";
                    break;
                case "divide":
                    result = a / b;
                    historyEntry = $"Divided {a} by {b}";
                    break;
                default:
                    result = double.NaN;
                    break;
            }
            return RecordResult(result, historyEntry);
        }

        public double Sum(params double[] numbers)
        {
            double total = 0;
            foreach (double number in numbers)
            {
                total += number;
            }
            return RecordResult(total, $"Summed {numbers.Length} numbers");
        }

        public double sum(params double[] args) => Sum(args);

        private double RecordResult(double result, string? historyEntry)
        {
            if (historyEntry != null)
            {
                history.Add(historyEntry);
            }
            LastResult = result;
            return result;
        }
    }

    public class History
    {
        public List<string> entries = new List<string>();

        public History() { }

        public History(IEnumerable<string> initialEntries)
        {
            entries.AddRange(initialEntries);
        }

        public void Display()
        {
            Console.WriteLine("History:");
            entries.ForEach(entry => Console.WriteLine($"- {entry}"));
        }

        public void display() => Display();

        public void AddEntry(string entry)
        {
            entries.Add(entry);
        }

        public void addEntry(string entry) => AddEntry(entry);
    }

    public static class Globals
    {
        public static Calculator calc = new Calculator();
    }

    class Program
    {
        static void Main(string[] args)
        {
            RunExample(Globals.calc);
            if (args.Length > 0)
            {
                args[0] = "mutated";
            }
        }

        private static void RunExample(Calculator calculator)
        {
            double additionResult = calculator.Add(5, 3);
            double operationResult = calculator.PerformOperation("add", 10, 20);
            double sumResult = calculator.Sum(1, 2, 3, 4);

            PrintResults(additionResult, operationResult, sumResult, calculator.LastResult);
            new History(calculator.history).Display();
        }

        private static void PrintResults(double additionResult, double operationResult,
            double sumResult, double lastResult)
        {
            Console.WriteLine("Results: " + additionResult + " " + operationResult + " " + sumResult);
            Console.WriteLine("Last result: " + lastResult);
        }
    }

    public static class Configurator
    {
        public static void configureCalculator(bool enableLogging, bool enableHistory, bool enableAdvanced)
        {
            // Исходная заглушка: настройки пока не реализованы.
        }
    }
}
