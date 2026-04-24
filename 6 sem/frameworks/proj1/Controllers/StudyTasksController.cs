using Microsoft.AspNetCore.Mvc;
using proj1.Contracts;
using proj1.Exceptions;
using proj1.Services;

namespace proj1.Controllers;

[ApiController]
[Route("api/items")]
public sealed class StudyTasksController(IStudyTaskService studyTaskService) : ControllerBase
{
    [HttpGet]
    public ActionResult<IReadOnlyCollection<StudyTaskDto>> GetAll()
    {
        return Ok(studyTaskService.GetAll());
    }

    [HttpGet("{id:int}")]
    public ActionResult<StudyTaskDto> GetById(int id)
    {
        var item = studyTaskService.GetById(id);
        if (item is null)
        {
            throw new NotFoundException($"Task with id {id} was not found.");
        }

        return Ok(item);
    }

    [HttpPost]
    public ActionResult<StudyTaskDto> Create([FromBody] CreateStudyTaskRequest request)
    {
        var created = studyTaskService.Create(request);
        return CreatedAtAction(nameof(GetById), new { id = created.Id }, created);
    }
}
