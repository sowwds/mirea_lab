using System.Threading.RateLimiting;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.Extensions.Options;
using proj3.Configuration;
using proj3.Core;
using proj3.Middlewares;
using proj3.Services;

try
{
    var configFile = CommandLineConfigResolver.ResolveConfigFile(args);

    var builder = WebApplication.CreateBuilder(args);

    builder.Configuration.Sources.Clear();
    builder.Configuration
        .AddJsonFile(configFile, optional: false, reloadOnChange: false)
        .AddEnvironmentVariables(prefix: "PROJ3_")
        .AddCommandLine(args);

    builder.Services.AddControllers();
    builder.Services.AddEndpointsApiExplorer();
    builder.Services.AddSwaggerGen();

    builder.Services.AddSingleton<IStudyItemService, InMemoryStudyItemService>();
    builder.Services.AddProblemDetails();

    builder.Services
        .AddOptions<AppOptions>()
        .Bind(builder.Configuration.GetSection(AppOptions.SectionName))
        .ValidateDataAnnotations()
        .Validate(options => options.AllowedOrigins.Length > 0, "At least one allowed origin must be configured.")
        .Validate(options => options.AllowedOrigins.All(origin => Uri.IsWellFormedUriString(origin, UriKind.Absolute)), "Allowed origins must be valid absolute URIs.")
        .Validate(options => options.RateLimiting.PermitLimit > 0, "RateLimiting:PermitLimit must be greater than zero.")
        .Validate(options => options.RateLimiting.WindowSeconds > 0, "RateLimiting:WindowSeconds must be greater than zero.")
        .Validate(options => options.Mode is RuntimeMode.Study or RuntimeMode.Production, "Mode must be Study or Production.")
        .ValidateOnStart();

    var appOptions = builder.Configuration.GetSection(AppOptions.SectionName).Get<AppOptions>() ?? new AppOptions();

    builder.Services.AddCors(options =>
    {
        options.AddPolicy("TrustedOrigins", policy =>
        {
            policy.WithOrigins(appOptions.AllowedOrigins)
                .AllowAnyHeader()
                .AllowAnyMethod();
        });
    });

    builder.Services.AddRateLimiter(options =>
    {
        options.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
        options.OnRejected = async (context, _) =>
        {
            var mode = context.HttpContext.RequestServices.GetRequiredService<IOptions<AppOptions>>().Value.Mode;
            var message = mode == RuntimeMode.Study
                ? "Rate limit exceeded. Reduce the request frequency and try again."
                : "Too many requests.";

            await context.HttpContext.Response.WriteAsJsonAsync(new
            {
                code = "rate_limit_exceeded",
                message
            });
        };

        options.GlobalLimiter = PartitionedRateLimiter.Create<HttpContext, string>(httpContext =>
        {
            var appSettings = httpContext.RequestServices.GetRequiredService<IOptions<AppOptions>>().Value;
            var key = httpContext.Connection.RemoteIpAddress?.ToString() ?? "unknown";

            return RateLimitPartition.GetFixedWindowLimiter(
                key,
                _ => new FixedWindowRateLimiterOptions
                {
                    PermitLimit = appSettings.RateLimiting.PermitLimit,
                    Window = TimeSpan.FromSeconds(appSettings.RateLimiting.WindowSeconds),
                    QueueProcessingOrder = QueueProcessingOrder.OldestFirst,
                    QueueLimit = 0,
                    AutoReplenishment = true
                });
        });
    });

    var app = builder.Build();
    var resolvedOptions = app.Services.GetRequiredService<IOptions<AppOptions>>().Value;

    app.UseExceptionHandler();

    if (resolvedOptions.Mode == RuntimeMode.Study)
    {
        app.UseSwagger();
        app.UseSwaggerUI();
    }

    app.UseMiddleware<SecurityHeadersMiddleware>();
    app.UseMiddleware<OriginRestrictionMiddleware>();
    app.UseCors("TrustedOrigins");
    app.UseRateLimiter();

    app.UseAuthorization();

    app.MapControllers();

    app.Run();
}
catch (OptionsValidationException exception)
{
    Console.Error.WriteLine($"Startup failed: {string.Join(" ", exception.Failures)}");
    Environment.ExitCode = 1;
}
catch (Exception exception)
{
    Console.Error.WriteLine($"Startup failed: {exception.Message}");
    Environment.ExitCode = 1;
}

public partial class Program;
