"""Контракты запросов и ответов API билетов."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Name = Annotated[str, Field(min_length=1, max_length=100)]
Price = Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)]
Seats = Annotated[int, Field(ge=0, le=300, strict=True)]


class TicketReplace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Name
    price: Price
    seats: Seats

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value):
        return value.strip() if isinstance(value, str) else value


class TicketCreate(TicketReplace):
    seats: Seats = 0


class TicketPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Name | None = None
    price: Price | None = None
    seats: Seats | None = None

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value):
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def nonempty_fields(self):
        if not self.model_fields_set or any(
            getattr(self, field) is None for field in self.model_fields_set
        ):
            raise ValueError("PATCH requires non-null fields")
        return self


class TicketOut(TicketReplace):
    model_config = ConfigDict(from_attributes=True)
    id: int


class SeatsChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delta: int = Field(strict=True)

    @field_validator("delta")
    @classmethod
    def nonzero(cls, value):
        if value == 0:
            raise ValueError("delta must not be zero")
        return value
