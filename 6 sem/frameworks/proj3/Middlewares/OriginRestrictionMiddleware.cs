using Microsoft.Extensions.Options;
using proj3.Configuration;

namespace proj3.Middlewares;

public sealed class OriginRestrictionMiddleware(
    RequestDelegate next,
    IOptions<AppOptions> appOptions)
{
    public async Task InvokeAsync(HttpContext context)
    {
        if (!context.Request.Headers.TryGetValue("Origin", out var originValues) ||
            string.IsNullOrWhiteSpace(originValues))
        {
            await next(context);
            return;
        }

        var origin = originValues.ToString();
        var allowed = appOptions.Value.AllowedOrigins.Contains(origin, StringComparer.OrdinalIgnoreCase);
        if (allowed)
        {
            await next(context);
            return;
        }

        context.Response.StatusCode = StatusCodes.Status403Forbidden;

        var message = appOptions.Value.Mode == RuntimeMode.Study
            ? $"Origin '{origin}' is not trusted."
            : "Forbidden.";

        await context.Response.WriteAsJsonAsync(new
        {
            code = "origin_forbidden",
            message
        });
    }
}
