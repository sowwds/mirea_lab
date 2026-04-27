using System.Net;
using Microsoft.AspNetCore.Mvc.Testing;

namespace Proj3.Tests;

public sealed class ApiSecurityTests
{
    [Fact]
    public async Task TrustedOrigin_ShouldReceiveCorsAndSecurityHeaders()
    {
        using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var request = new HttpRequestMessage(HttpMethod.Get, "/api/items");
        request.Headers.Add("Origin", "http://localhost:3000");

        var response = await client.SendAsync(request);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("http://localhost:3000", response.Headers.GetValues("Access-Control-Allow-Origin").Single());
        Assert.Equal("DENY", response.Headers.GetValues("X-Frame-Options").Single());
        Assert.Equal("nosniff", response.Headers.GetValues("X-Content-Type-Options").Single());
        Assert.Equal("no-store", response.Headers.GetValues("Cache-Control").Single());
    }

    [Fact]
    public async Task UntrustedOrigin_ShouldNotReceiveCorsHeader()
    {
        using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var request = new HttpRequestMessage(HttpMethod.Get, "/api/items");
        request.Headers.Add("Origin", "https://evil.example");

        var response = await client.SendAsync(request);

        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
        Assert.False(response.Headers.Contains("Access-Control-Allow-Origin"));
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("origin_forbidden", body);
        Assert.Contains("not trusted", body);
    }

    [Fact]
    public async Task RateLimiter_ShouldRejectRequests_WhenLimitIsExceeded()
    {
        using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();

        for (var i = 0; i < 3; i++)
        {
            var response = await client.GetAsync("/api/items");
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        }

        var rejected = await client.GetAsync("/api/items");

        Assert.Equal((HttpStatusCode)429, rejected.StatusCode);
        var body = await rejected.Content.ReadAsStringAsync();
        Assert.Contains("rate_limit_exceeded", body);
        Assert.Contains("Reduce the request frequency", body);
    }

    [Fact]
    public async Task ProductionMode_ShouldUseStricterRateLimitMessage()
    {
        using var rootFactory = new WebApplicationFactory<Program>();
        using var factory = rootFactory.WithWebHostBuilder(builder =>
        {
            builder.UseSetting("App:Mode", "Production");
            builder.UseSetting("App:RateLimiting:PermitLimit", "1");
            builder.UseSetting("App:RateLimiting:WindowSeconds", "60");
            builder.UseSetting("App:AllowedOrigins:0", "https://example.edu");
        });

        using var client = factory.CreateClient();

        var first = await client.GetAsync("/api/items");
        var second = await client.GetAsync("/api/items");

        Assert.Equal(HttpStatusCode.OK, first.StatusCode);
        Assert.Equal((HttpStatusCode)429, second.StatusCode);

        var body = await second.Content.ReadAsStringAsync();
        Assert.Contains("Too many requests.", body);
        Assert.DoesNotContain("Reduce the request frequency", body);
    }

    [Fact]
    public async Task ProductionMode_ShouldUseTerseForbiddenOriginMessage()
    {
        using var rootFactory = new WebApplicationFactory<Program>();
        using var factory = rootFactory.WithWebHostBuilder(builder =>
        {
            builder.UseSetting("App:Mode", "Production");
            builder.UseSetting("App:AllowedOrigins:0", "https://example.edu");
        });

        using var client = factory.CreateClient();
        using var request = new HttpRequestMessage(HttpMethod.Get, "/api/items");
        request.Headers.Add("Origin", "https://evil.example");

        var response = await client.SendAsync(request);

        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("Forbidden.", body);
        Assert.DoesNotContain("not trusted", body);
    }
}
