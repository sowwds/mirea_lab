// вариант 9: умножение матриц

using System;
using System.Diagnostics;

class Program
{
    static int[,] MultiplyMatrices(int[,] a, int[,] b)
    {
        int n = a.GetLength(0);
        int[,] result = new int[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = 0;
                for (int k = 0; k < n; k++)
                {
                    result[i, j] += a[i, k] * b[k, j];
                }
            }
        }
        return result;
    }

    static void Main()
    {
        int[,] a = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
        int[,] b = { { 9, 8, 7 }, { 6, 5, 4 }, { 3, 2, 1 } };
        Stopwatch sw = Stopwatch.StartNew();
        int[,] result = MultiplyMatrices(a, b);
        sw.Stop();
        Console.WriteLine("Результат умножения матриц:");
        for (int i = 0; i < result.GetLength(0); i++)
        {
            for (int j = 0; j < result.GetLength(1); j++)
            {
                Console.Write(result[i, j] + " ");
            }
            Console.WriteLine();
        }
        Console.WriteLine("Время: " + sw.ElapsedMilliseconds + " мс");
    }
}