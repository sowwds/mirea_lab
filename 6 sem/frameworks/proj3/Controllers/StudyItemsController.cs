using Microsoft.AspNetCore.Mvc;
using proj3.Contracts;
using proj3.Services;

namespace proj3.Controllers;

[ApiController]
[Route("api/items")]
public sealed class StudyItemsController(IStudyItemService studyItemService) : ControllerBase
{
    [HttpGet]
    public ActionResult<IReadOnlyCollection<StudyItemDto>> GetAll()
    {
        return Ok(studyItemService.GetAll());
    }
}
