using System;
using System.Collections.Generic;
using System.IO;

namespace RefactoringExample
{
    public class Program
    {
        private static List<Dictionary<string, object>> employees = new List<Dictionary<string, object>>();

        public static double CalculateSalary(Dictionary<string, object> emp)
        {
            Console.WriteLine("Starting salary calculation for employee " + emp["id"]);
            
            // расчет базового оклада
            double baseSalary = (int)emp["hoursWorked"] * (double)emp["rate"];
            
            // Расчет надбавок
            double allowance = 0;
            if ((string)emp["grade"] == "A")
            {
                allowance = baseSalary * 0.2;
            }
            else if ((string)emp["grade"] == "B")
            {
                allowance = baseSalary * 0.1;
            }
            else if ((string)emp["grade"] == "C")
            {
                allowance = baseSalary * 0.05;
            }
            else
            {
                allowance = 0;
            }
            
            // удержания
            double deductions = 0;
            if ((bool)emp["insurance"])
            {
                deductions += baseSalary * 0.05;
            }
            if ((bool)emp["taxExempt"])
            {
                deductions += 0;
            }
            else
            {
                deductions += baseSalary * 0.1;
            }
            
            // итоговая зарплата
            double netSalary = baseSalary + allowance - deductions;
            
            DateTime today = DateTime.Today;
            string formattedDate = today.Year + "-" + today.Month + "-" + today.Day;
            
            // логирование
            Console.WriteLine("Salary calculated for employee " + emp["id"] + " on " + formattedDate);
            
            File.AppendAllText("salary_report.txt", $"Employee {emp["id"]}: {netSalary} on {formattedDate}\n");
            
            return netSalary;
        }

        public static string GetEmployeeGrade(Dictionary<string, object> emp)
        {
            if ((string)emp["department"] == "IT")
            {
                if ((int)emp["yearsOfExperience"] > 10)
                {
                    return "A";
                }
                else if ((int)emp["yearsOfExperience"] > 5)
                {
                    return "B";
                }
                else
                {
                    return "C";
                }
            }
            else if ((string)emp["department"] == "HR")
            {
                if ((int)emp["yearsOfExperience"] > 8)
                {
                    return "A";
                }
                else if ((int)emp["yearsOfExperience"] > 4)
                {
                    return "B";
                }
                else
                {
                    return "C";
                }
            }
            else
            {
                return "C";
            }
        }

        public static void GenerateReport()
        {
            DateTime today = DateTime.Today;
            string formattedDate = today.Year + "-" + today.Month + "-" + today.Day;
            Console.WriteLine("Report generated on " + formattedDate);
        }

        static void Main(string[] args)
        {
            var employee = new Dictionary<string, object>
            {
                { "id", "emp123" },
                { "hoursWorked", 160 },
                { "rate", 25.0 },
                { "grade", "B" },
                { "insurance", true },
                { "taxExempt", false },
                { "department", "IT" },
                { "yearsOfExperience", 7 }
            };
            
            double salary = CalculateSalary(employee);
            Console.WriteLine($"Calculated salary: {salary}");
            
            string grade = GetEmployeeGrade(employee);
            Console.WriteLine($"Employee grade: {grade}");
            
            GenerateReport();
        }
    }
}