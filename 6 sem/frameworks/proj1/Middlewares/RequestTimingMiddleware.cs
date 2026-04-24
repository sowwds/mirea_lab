using System.Diagnostics;

namespace proj1.Middlewares;

public sealed class RequestTimingMiddleware(
    RequestDelegate next,
    ILogger<RequestTimingMiddleware> logger)
{
    public const string ElapsedMillisecondsKey = "ElapsedMilliseconds";

    public async Task InvokeAsync(HttpContext context)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            await next(context);
        }
        finally
        {
            stopwatch.Stop();
            context.Items[ElapsedMillisecondsKey] = stopwatch.ElapsedMilliseconds;

            logger.LogInformation(
                "Request {RequestId} completed in {ElapsedMs} ms",
                context.TraceIdentifier,
                stopwatch.ElapsedMilliseconds);
        }
    }
}
