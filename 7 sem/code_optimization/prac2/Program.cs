using System;
using System.Diagnostics;
using System.Linq;

class Program
{
    static int[,] MultiplyMatricesOriginal(int[,] a, int[,] b)
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

    // Вход: квадратные матрицы одного размера.
    static int[,] MultiplyMatrices(int[,] a, int[,] b)
    {
        int n = a.GetLength(0);
        int[,] result = new int[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < n; k++)
            {
                int factor = a[i, k];
                if (factor == 0)
                    continue;
                for (int j = 0; j < n; j++)
                    result[i, j] += factor * b[k, j];
            }
        }
        return result;
    }

    static void Main(string[] args)
    {
        if (args.Contains("--check")) { Check(); return; }
        if (args.Contains("--bench")) { Benchmark(); return; }
        int[,] a = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
        int[,] b = { { 9, 8, 7 }, { 6, 5, 4 }, { 3, 2, 1 } };
        Stopwatch sw = Stopwatch.StartNew();
        int[,] result = MultiplyMatrices(a, b);
        sw.Stop();
        Console.WriteLine("Результат умножения матриц:");
        for (int i = 0; i < result.GetLength(0); i++)
        {
            for (int j = 0; j < result.GetLength(1); j++)
                Console.Write(result[i, j] + " ");
            Console.WriteLine();
        }
        Console.WriteLine("Время: " + sw.ElapsedMilliseconds + " мс");
    }

    static void Equal(int[,] expected, int[,] actual)
    {
        if (expected.GetLength(0) != actual.GetLength(0) || expected.GetLength(1) != actual.GetLength(1))
            throw new Exception("Размеры результатов различаются");
        for (int i = 0; i < expected.GetLength(0); i++)
            for (int j = 0; j < expected.GetLength(1); j++)
                if (expected[i, j] != actual[i, j])
                    throw new Exception($"Не совпал элемент [{i}, {j}]");
    }

    static void Check()
    {
        int count = 0;
        void Test(int[,] a, int[,] b)
        {
            Equal(MultiplyMatricesOriginal(a, b), MultiplyMatrices(a, b));
            count++;
        }
        int[,] a = {{1,2,3},{4,5,6},{7,8,9}};
        int[,] b = {{9,8,7},{6,5,4},{3,2,1}};
        Equal(new int[,] {{30,24,18},{84,69,54},{138,114,90}}, MultiplyMatrices(a,b));
        Test(a,b);
        Test(new int[0,0],new int[0,0]);
        Test(new int[,] {{-3}}, new int[,] {{7}});
        Test(new int[3,3],b);
        Test(a,new int[3,3]);
        Test(a,new int[,] {{1,0,0},{0,1,0},{0,0,1}});
        Test(new int[,] {{int.MaxValue,int.MinValue},{-1,0}},new int[,] {{2,-1},{3,int.MaxValue}});
        var rng = new Random(42);
        for (int sample = 0; sample < 100; sample++)
        {
            int n = rng.Next(1,16);
            int[,] left = new int[n,n], right = new int[n,n];
            for (int i=0;i<n;i++)
                for (int j=0;j<n;j++)
                {
                    left[i,j]=rng.Next(-5,6);
                    right[i,j]=rng.Next(-5,6);
                }
            int[,] leftCopy=(int[,])left.Clone(), rightCopy=(int[,])right.Clone();
            Test(left,right);
            Equal(leftCopy,left);
            Equal(rightCopy,right);
        }
        Console.WriteLine($"Проверки пройдены: {count}/{count}");
    }

    static double Measure(Func<int[,],int[,],int[,]> multiply, int[,] a, int[,] b)
    {
        multiply(a,b);
        multiply(a,b);
        double[] samples=new double[5];
        for (int repeat=0;repeat<samples.Length;repeat++)
        {
            var sw=Stopwatch.StartNew();
            int[,] result=multiply(a,b);
            sw.Stop();
            GC.KeepAlive(result);
            samples[repeat]=sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        return samples[samples.Length/2];
    }

    static void Benchmark()
    {
        foreach (bool sparse in new[] {false,true})
        {
            Console.WriteLine(sparse ? "\nРазреженная A: два ненулевых элемента в каждой строке" : "\nПлотные матрицы");
            Console.WriteLine("| N | Исходный, мс | Новый, мс | Ускорение |");
            Console.WriteLine("| --- | --- | --- | --- |");
            foreach (int n in new[] {64,128,256,384,512})
            {
                var rng=new Random(42);
                int[,] a=new int[n,n],b=new int[n,n];
                for (int i=0;i<n;i++)
                    for (int j=0;j<n;j++)
                    {
                        if (!sparse) a[i,j]=rng.Next(1,10);
                        b[i,j]=rng.Next(1,10);
                    }
                if (sparse)
                    for (int i=0;i<n;i++)
                    {
                        a[i,i]=rng.Next(1,10);
                        a[i,(i+1)%n]=rng.Next(1,10);
                    }
                Equal(MultiplyMatricesOriginal(a,b),MultiplyMatrices(a,b));
                double before=Measure(MultiplyMatricesOriginal,a,b);
                double after=Measure(MultiplyMatrices,a,b);
                Console.WriteLine(FormattableString.Invariant($"| {n} | {before:F4} | {after:F4} | {before/after:F2}× |"));
            }
        }
    }
}
