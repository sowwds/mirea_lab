using Microsoft.Extensions.Options;
using proj3.Configuration;

namespace proj3.Middlewares;

public sealed class SecurityHeadersMiddleware(
    RequestDelegate next,
    IOptions<AppOptions> appOptions)
{
    public async Task InvokeAsync(HttpContext context)
    {
        context.Response.OnStarting(() =>
        {
            context.Response.Headers["X-Frame-Options"] = "DENY";
            context.Response.Headers["X-Content-Type-Options"] = "nosniff";
            context.Response.Headers["Cache-Control"] = appOptions.Value.Mode == RuntimeMode.Study
                ? "no-store"
                : "no-store, no-cache";

            return Task.CompletedTask;
        });

        await next(context);
    }
}
