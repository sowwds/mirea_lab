using System;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;

namespace RefactoringExample
{
    public class Logger
    {
        public void logToConsole(string message)
        {
            Console.WriteLine($"[LOG] {DateTime.UtcNow.ToString("o")}: {message}");
        }

        public void logToFile(string message)
        {
            try
            {
                File.AppendAllText("app.log", $"[LOG] {DateTime.UtcNow.ToString("o")}: {message}\n");
            }
            catch (Exception)
            {
                Console.Error.WriteLine("Failed to write to log file");
            }
        }
    }

    public class Server
    {
        public int ServerPort;
        public string ServerHost;
        private Logger logger;

        public Server(int port, string host)
        {
            this.ServerPort = port;
            this.ServerHost = host;
            this.logger = new Logger();
        }

        public void startServer()
        {
            HttpListener listener = new HttpListener();
            string prefix = $"http://{this.ServerHost}:{this.ServerPort}/";
            listener.Prefixes.Add(prefix);

            try
            {
                listener.Start();
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
                return;
            }

            this.logger.logToConsole($"Server started at {this.ServerHost}:{this.ServerPort}");
            this.logger.logToFile($"Server started at {this.ServerHost}:{this.ServerPort}");

            if (false)
            {
                Console.WriteLine("This will never be printed");
            }

            Timer timer = new Timer(state =>
            {
                this.logger.logToConsole("Server heartbeat");
            }, null, 60000, 60000);

            // серверный цикл (упрощённый)
            while (true)
            {
                var context = listener.GetContext();
                var req = context.Request;
                var res = context.Response;

                // обработка запроса
                this.logger.logToConsole($"Request received: {req.Url}");

                string responseString = "Hello World";
                byte[] buffer = Encoding.UTF8.GetBytes(responseString);
                res.ContentLength64 = buffer.Length;
                using (Stream output = res.OutputStream)
                {
                    output.Write(buffer, 0, buffer.Length);
                }
            }
        }
    }

    public static class ConfigLoader
    {
        public static void loadConfig()
        {
            try
            {
                // ... загрузка конфигурации
                throw new Exception("Config file not found");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
            }
        }
    }
    
    public static class MiddlewareApplier
    {
        public static void applyMiddleware(Server server)
        {
            // ... применение middleware
        }
    }
    
    public static class Program
    {
        public static void Main(string[] args)
        {
            // загружаем конфиг
            ConfigLoader.loadConfig();

            // создаём сервер на localhost:8080
            var server = new Server(8080, "localhost");

            MiddlewareApplier.applyMiddleware(server);

            Thread serverThread = new Thread(() =>
            {
                server.startServer();
            });
            serverThread.IsBackground = true;
            serverThread.Start();

            // даём серверу немного времени на запуск
            Thread.Sleep(500);

            // минимальное тестирование: делаем простой HTTP GET запрос к серверу
            try
            {
                var request = (HttpWebRequest)WebRequest.Create("http://localhost:8080/");
                request.Method = "GET";
                using (var response = (HttpWebResponse)request.GetResponse())
                using (var stream = response.GetResponseStream())
                using (var reader = new StreamReader(stream))
                {
                    string body = reader.ReadToEnd();
                    Console.WriteLine("Test request response body:");
                    Console.WriteLine(body);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Test request failed: {ex.Message}");
            }

            Thread.Sleep(2000);

            Console.WriteLine("Main finished. Server may still be running in background thread.");
        }
    }
}

