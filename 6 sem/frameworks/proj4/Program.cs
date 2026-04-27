using proj4.Configuration;
using proj4.Health;
using proj4.Metrics;
using proj4.Middlewares;
using proj4.Services;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.Converters.Add(new JsonStringEnumConverter());
    });
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.Configure<OperationsOptions>(
    builder.Configuration.GetSection(OperationsOptions.SectionName));

builder.Services.AddSingleton<BookingMetrics>();
builder.Services.AddSingleton<BookingHealthState>();
builder.Services.AddSingleton<IBookingProcessService, BookingProcessService>();

builder.Services.AddHealthChecks()
    .AddCheck<LivenessHealthCheck>("live", tags: ["live"])
    .AddCheck<ReadinessHealthCheck>("ready", tags: ["ready"]);

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseMiddleware<CorrelationIdMiddleware>();

app.UseAuthorization();

app.MapControllers();
app.MapHealthChecks("/health/live", new Microsoft.AspNetCore.Diagnostics.HealthChecks.HealthCheckOptions
{
    Predicate = registration => registration.Tags.Contains("live")
});
app.MapHealthChecks("/health/ready", new Microsoft.AspNetCore.Diagnostics.HealthChecks.HealthCheckOptions
{
    Predicate = registration => registration.Tags.Contains("ready")
});

app.Run();

public partial class Program;
