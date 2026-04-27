namespace proj4.Middlewares;

public sealed class CorrelationIdMiddleware(
    RequestDelegate next,
    ILogger<CorrelationIdMiddleware> logger)
{
    public const string CorrelationIdKey = "CorrelationId";
    private const string HeaderName = "X-Correlation-Id";

    public async Task InvokeAsync(HttpContext context)
    {
        var correlationId = context.Request.Headers.TryGetValue(HeaderName, out var providedValue) &&
                            !string.IsNullOrWhiteSpace(providedValue)
            ? providedValue.ToString()
            : Guid.NewGuid().ToString("N");

        context.Items[CorrelationIdKey] = correlationId;
        context.Response.Headers[HeaderName] = correlationId;

        using (logger.BeginScope(new Dictionary<string, object>
        {
            ["CorrelationId"] = correlationId
        }))
        {
            await next(context);
        }
    }
}
