using System;
using System.Collections.Generic;

namespace RefactoringExample
{
    public class Calculator
    {
        public List<string> history = new List<string>();
        public double LastResult = 0;

        public double addNumber(double a, double b)
        {
            a = a + b;
            history.Add($"Added {b} to number");
            this.LastResult = a;
            return a;
        }

        public double PerformOperation(string operation, double a, double b)
        {
            double result;
            switch (operation)
            {
                case "add":
                    result = a + b;
                    history.Add($"Added {a} and {b}");
                    break;
                case "subtract":
                    result = a - b;
                    history.Add($"Subtracted {b} from {a}");
                    break;
                case "multiply":
                    result = a * b;
                    history.Add($"Multiplied {a} by {b}");
                    break;
                case "divide":
                    result = a / b;
                    history.Add($"Divided {a} by {b}");
                    break;
                default:
                    result = double.NaN;
                    break;
            }
            this.LastResult = result;
            return result;
        }

        public double sum(params double[] args)
        {
            double total = 0;
            for (int i = 0; i < args.Length; i++)
            {
                total += args[i];
            }
            history.Add($"Summed {args.Length} numbers");
            this.LastResult = total;
            return total;
        }
    }

    public class History
    {
        public List<string> entries = new List<string>();

        public void display()
        {
            Console.WriteLine("History:");
            entries.ForEach(entry =>
            {
                Console.WriteLine($"- {entry}");
            });
        }

        public void addEntry(string entry)
        {
            entries.Add(entry);
        }
    }

    public static class Globals
    {
        public static Calculator calc = new Calculator();
    }

    class Program
    {
        static void Main(string[] args)
        {
            object[] arguments = args;

            double result1 = Globals.calc.addNumber(5, 3);
            double result2 = Globals.calc.PerformOperation("add", 10, 20);
            double result3 = Globals.calc.sum(1, 2, 3, 4);

            Console.WriteLine("Results: " + result1 + " " + result2 + " " + result3);
            Console.WriteLine("Last result: " + Globals.calc.LastResult);

            History hist = new History();

            foreach (var h in Globals.calc.history)
            {
                hist.addEntry(h);
            }
            hist.display();

            if (arguments.Length > 0)
            {
                arguments[0] = "mutated";
            }
        }
    }

    public static class Configurator
    {
        public static void configureCalculator(bool enableLogging, bool enableHistory, bool enableAdvanced)
        {
            // ...
        }
    }
}
