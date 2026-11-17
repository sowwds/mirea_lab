"""Фабрика API билетов с SQLite, ролями и конвертацией цен."""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import get_role, login, require_admin
from app.database import Base, Ticket, get_db, make_engine
from app.rates import RateUnavailable, get_rates
from app.schemas import SeatsChange, TicketCreate, TicketOut, TicketPatch, TicketReplace


def find_ticket(session, ticket_id):
    """Найти билет или вернуть 404."""
    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    return ticket


def save_ticket(session, ticket):
    """Сохранить билет с откатом при конфликте уникального имени."""
    session.add(ticket)
    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(409, "Ticket name already exists") from error
    session.refresh(ticket)
    return ticket


def create_app(database_url="sqlite:///./tickets.db"):
    """Создать независимое приложение и ресурсы его жизненного цикла."""
    engine = make_engine(database_url)

    @asynccontextmanager
    async def lifespan(application):
        Base.metadata.create_all(engine)
        yield
        engine.dispose()

    application = FastAPI(title="Tickets API — variant 2", lifespan=lifespan)
    application.state.engine = engine
    application.state.tokens = {}

    @application.get("/health")
    def health():
        return {"status": "ok"}

    application.post("/auth/login")(login)

    @application.post("/tickets", response_model=TicketOut, status_code=201)
    def create_ticket(
        body: TicketCreate, session: Session = Depends(get_db), role=Depends(require_admin)
    ):
        return save_ticket(session, Ticket(**body.model_dump()))

    @application.get("/tickets", response_model=list[TicketOut])
    def list_tickets(
        q: str | None = None,
        min_price: float | None = Query(None, ge=0, allow_inf_nan=False),
        max_price: float | None = Query(None, ge=0, allow_inf_nan=False),
        limit: int = Query(10, ge=1, le=100),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_db), role=Depends(get_role),
    ):
        if min_price is not None and max_price is not None and min_price > max_price:
            raise HTTPException(422, "min_price exceeds max_price")
        statement = select(Ticket)
        if q is not None:
            statement = statement.where(Ticket.name.icontains(q, autoescape=True))
        if min_price is not None:
            statement = statement.where(Ticket.price >= min_price)
        if max_price is not None:
            statement = statement.where(Ticket.price <= max_price)
        return session.scalars(statement.order_by(Ticket.id).offset(offset).limit(limit)).all()

    @application.get("/tickets/{ticket_id}", response_model=TicketOut)
    def get_ticket(ticket_id: int, session: Session = Depends(get_db), role=Depends(get_role)):
        return find_ticket(session, ticket_id)

    @application.put("/tickets/{ticket_id}", response_model=TicketOut)
    def replace_ticket(
        ticket_id: int, body: TicketReplace,
        session: Session = Depends(get_db), role=Depends(require_admin),
    ):
        ticket = find_ticket(session, ticket_id)
        for field, value in body.model_dump().items():
            setattr(ticket, field, value)
        return save_ticket(session, ticket)

    @application.patch("/tickets/{ticket_id}", response_model=TicketOut)
    def patch_ticket(
        ticket_id: int, body: TicketPatch,
        session: Session = Depends(get_db), role=Depends(require_admin),
    ):
        ticket = find_ticket(session, ticket_id)
        for field, value in body.model_dump(exclude_unset=True).items():
            setattr(ticket, field, value)
        return save_ticket(session, ticket)

    @application.delete("/tickets/{ticket_id}", status_code=204)
    def delete_ticket(
        ticket_id: int, session: Session = Depends(get_db), role=Depends(require_admin)
    ):
        ticket = find_ticket(session, ticket_id)
        session.delete(ticket)
        session.commit()
        return Response(status_code=204)

    @application.post("/tickets/{ticket_id}/seats", response_model=TicketOut)
    def change_seats(
        ticket_id: int, body: SeatsChange,
        session: Session = Depends(get_db), role=Depends(require_admin),
    ):
        ticket = find_ticket(session, ticket_id)
        new_seats = ticket.seats + body.delta
        if not 0 <= new_seats <= 300:
            raise HTTPException(409, "Seats must remain between 0 and 300")
        ticket.seats = new_seats
        return save_ticket(session, ticket)

    @application.get("/tickets/{ticket_id}/price")
    def converted_price(
        ticket_id: int, currency: Literal["USD", "EUR"],
        session: Session = Depends(get_db), role=Depends(get_role), rates=Depends(get_rates),
    ):
        ticket = find_ticket(session, ticket_id)
        try:
            rate = rates.rate(currency)
        except RateUnavailable as error:
            raise HTTPException(502, "Exchange service unavailable") from error
        return {"currency": currency, "price": ticket.price * rate}

    return application
