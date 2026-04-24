namespace proj1.Middlewares;

public sealed class RequestResponseLoggingMiddleware(
    RequestDelegate next,
    ILogger<RequestResponseLoggingMiddleware> logger)
{
    public async Task InvokeAsync(HttpContext context)
    {
        logger.LogInformation(
            "Request started {Method} {Path}. RequestId: {RequestId}",
            context.Request.Method,
            context.Request.Path,
            context.TraceIdentifier);

        try
        {
            await next(context);
        }
        finally
        {
            var elapsedMs = context.Items.TryGetValue(RequestTimingMiddleware.ElapsedMillisecondsKey, out var value)
                ? value
                : "n/a";

            logger.LogInformation(
                "Request finished {Method} {Path} with {StatusCode}. RequestId: {RequestId}. ElapsedMs: {ElapsedMs}",
                context.Request.Method,
                context.Request.Path,
                context.Response.StatusCode,
                context.TraceIdentifier,
                elapsedMs);
        }
    }
}
