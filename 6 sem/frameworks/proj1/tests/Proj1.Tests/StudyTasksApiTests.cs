using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using proj1.Contracts;

namespace Proj1.Tests;

public sealed class StudyTasksApiTests(WebApplicationFactory<Program> factory)
    : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client = factory.CreateClient();

    [Fact]
    public async Task GetById_ShouldReturnUniformError_WhenItemDoesNotExist()
    {
        var response = await _client.GetAsync("/api/items/999");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);

        var error = await response.Content.ReadFromJsonAsync<ErrorResponse>();
        Assert.NotNull(error);
        Assert.Equal("not_found", error!.Code);
        Assert.False(string.IsNullOrWhiteSpace(error.RequestId));
    }

    [Fact]
    public async Task PostThenGet_ShouldReturnCreatedItem()
    {
        var createRequest = new CreateStudyTaskRequest("Prepare defense", "Frameworks", 6, 5);

        var createResponse = await _client.PostAsJsonAsync("/api/items", createRequest);

        Assert.Equal(HttpStatusCode.Created, createResponse.StatusCode);

        var created = await createResponse.Content.ReadFromJsonAsync<StudyTaskDto>();
        Assert.NotNull(created);

        var getResponse = await _client.GetAsync($"/api/items/{created!.Id}");
        Assert.Equal(HttpStatusCode.OK, getResponse.StatusCode);

        var fetched = await getResponse.Content.ReadFromJsonAsync<StudyTaskDto>();
        Assert.NotNull(fetched);
        Assert.Equal(created.Id, fetched!.Id);
        Assert.Equal("Prepare defense", fetched.Title);
    }

    [Fact]
    public async Task Post_ShouldReturnUniformError_WhenPayloadIsInvalid()
    {
        var createRequest = new CreateStudyTaskRequest("", "Frameworks", 6, 5);

        var response = await _client.PostAsJsonAsync("/api/items", createRequest);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var error = await response.Content.ReadFromJsonAsync<ErrorResponse>();
        Assert.NotNull(error);
        Assert.Equal("validation_error", error!.Code);
        Assert.False(string.IsNullOrWhiteSpace(error.RequestId));
    }
}
