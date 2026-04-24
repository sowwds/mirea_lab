using System.Text.Json;
using proj1.Contracts;
using proj1.Exceptions;

namespace proj1.Middlewares;

public sealed class ExceptionHandlingMiddleware(
    RequestDelegate next,
    ILogger<ExceptionHandlingMiddleware> logger)
{
    public async Task InvokeAsync(HttpContext context)
    {
        try
        {
            await next(context);
        }
        catch (Exception exception)
        {
            await HandleExceptionAsync(context, exception);
        }
    }

    private async Task HandleExceptionAsync(HttpContext context, Exception exception)
    {
        var requestId = context.TraceIdentifier;

        var (statusCode, code, message, level) = exception switch
        {
            DomainValidationException validationException => (
                StatusCodes.Status400BadRequest,
                "validation_error",
                validationException.Message,
                LogLevel.Warning),
            NotFoundException notFoundException => (
                StatusCodes.Status404NotFound,
                "not_found",
                notFoundException.Message,
                LogLevel.Information),
            _ => (
                StatusCodes.Status500InternalServerError,
                "internal_error",
                "Unexpected server error.",
                LogLevel.Error)
        };

        logger.Log(level, exception, "Request {RequestId} failed with code {ErrorCode}", requestId, code);

        var response = new ErrorResponse(code, message, requestId);
        context.Response.StatusCode = statusCode;
        await context.Response.WriteAsJsonAsync(response, new JsonSerializerOptions(JsonSerializerDefaults.Web));
    }
}
