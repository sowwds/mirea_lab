"""SQLite-модель и зависимость сессии БД."""

from fastapi import Request
from sqlalchemy import CheckConstraint, Float, Integer, String, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        CheckConstraint("price > 0"),
        CheckConstraint("seats >= 0 AND seats <= 300"),
        {"sqlite_autoincrement": True},
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    price: Mapped[float] = mapped_column(Float)
    seats: Mapped[int] = mapped_column(Integer, default=0)


def make_engine(url):
    """Создать SQLite engine с общим соединением для потоков TestClient."""
    engine = create_engine(
        url, connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def unicode_lower(connection, record):
        connection.create_function("lower", 1, lambda value: value.casefold(), deterministic=True)

    return engine


def get_db(request: Request):
    """Предоставить сессию на время обработки запроса."""
    with Session(request.app.state.engine) as session:
        yield session
