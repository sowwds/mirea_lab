using System;
using System.Collections.Generic;

namespace RefactoringExample
{
    public class Animal
    {
        public string Type { get; set; }
        public string Name { get; set; }

        public Animal(string type, string name)
        {
            Type = type;
            Name = name;
        }

        public void MakeNoise()
        {
            if (Type == "dog")
            {
                Console.WriteLine("Woof!");
                // Воспроизведение звука собаки (имитация)
                Console.WriteLine("Playing sound: ./sounds/dog.mp3");
            }
            else if (Type == "cat")
            {
                Console.WriteLine("Meow!");
                Console.WriteLine("Playing sound: ./sounds/cat.mp3");
            }
            else if (Type == "bird")
            {
                Console.WriteLine("Tweet!");
                Console.WriteLine("Playing sound: ./sounds/bird.mp3");
            }
            else
            {
                Console.WriteLine("Unknown animal sound");
            }
        }

        public void Display()
        {
            Console.WriteLine($"This is a {Type} named {Name}");
        }

        public void Eat(string food)
        {
            Console.WriteLine($"{Name} is eating {food}");
        }
    }

    public class Zoo
    {
        public void RunZoo()
        {
            var animals = new List<Animal>
            {
                new Animal("dog", "Rex"),
                new Animal("cat", "Whiskers"),
                new Animal("bird", "Tweety")
            };

            Console.WriteLine("=== Welcome to the Zoo! ===");

            // показ животных
            Console.WriteLine("\n--- Showing Animals ---");
            foreach (var animal in animals)
            {
                animal.Display();
            }

            // издание звуков
            Console.WriteLine("\n--- Animal Sounds ---");
            foreach (var animal in animals)
            {
                animal.MakeNoise();
            }

            // кормление животных
            Console.WriteLine("\n--- Feeding Animals ---");
            foreach (var animal in animals)
            {
                if (animal.Type == "dog")
                {
                    Console.WriteLine($"Feeding dog food to {animal.Name}");
                    animal.Eat("dog food");
                }
                else if (animal.Type == "cat")
                {
                    Console.WriteLine($"Feeding cat food to {animal.Name}");
                    animal.Eat("cat food");
                }
                else if (animal.Type == "bird")
                {
                    Console.WriteLine($"Feeding bird food to {animal.Name}");
                    animal.Eat("bird food");
                }
            }

            Console.WriteLine("\n--- Cleaning Enclosures ---");
            foreach (var animal in animals)
            {
                Console.WriteLine($"Cleaning {animal.Type} enclosure for {animal.Name}");
            }

            Console.WriteLine("\n=== Zoo Tour Completed ===");
        }

        public void DoSomethingWithAnimals(string thing)
        {
            Console.WriteLine($"Doing {thing} with animals");
        }

        public void ProcessAnimalData(int x)
        {
            Console.WriteLine($"Processing animal data with value: {x}");
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Starting Zoo Application...\n");

            var zoo = new Zoo();
            
            // демонстрация работы зоопарка
            zoo.RunZoo();

            Console.WriteLine("\n--- Testing Other Methods ---");
            zoo.DoSomethingWithAnimals("feeding");
            zoo.ProcessAnimalData(42);

            Console.WriteLine("\n--- Adding New Animal (Problematic) ---");
            var newAnimal = new Animal("lion", "Simba");
            newAnimal.Display();
            newAnimal.MakeNoise();

            Console.WriteLine("\nZoo Application Completed.");
        }
    }

    public static class AnimalHelper
    {
        public static void PerformOperation(Animal animal, string operation)
        {
            switch (operation)
            {
                case "feed":
                    if (animal.Type == "dog")
                    {
                        Console.WriteLine($"Feeding dog food to {animal.Name}");
                    }
                    else if (animal.Type == "cat")
                    {
                        Console.WriteLine($"Feeding cat food to {animal.Name}");
                    }
                    else if (animal.Type == "bird")
                    {
                        Console.WriteLine($"Feeding bird food to {animal.Name}");
                    }
                    break;
                case "clean":
                    Console.WriteLine($"Cleaning {animal.Type} enclosure for {animal.Name}");
                    break;
                default:
                    Console.WriteLine($"Unknown operation for {animal.Name}");
                    break;
            }
        }
    }
}