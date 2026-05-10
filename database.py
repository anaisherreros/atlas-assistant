from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import BigInteger, DateTime, String, Text, func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    telegram_chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class ChatSession(Base):
    """Estado por chat de Telegram: agente conversacional activo."""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    telegram_chat_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    active_agent: Mapped[str] = mapped_column(String(32), default="personal")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


def normalize_database_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    raise ValueError(
        "DATABASE_URL debe ser una URL de PostgreSQL "
        "(postgresql://... o postgres://...)"
    )


def create_engine(database_url: str):
    return create_async_engine(
        normalize_database_url(database_url),
        echo=False,
        pool_pre_ping=True,
    )


async def init_db(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def save_message(
    session: AsyncSession,
    *,
    telegram_chat_id: int,
    telegram_user_id: int,
    role: str,
    content: str,
) -> ChatMessage:
    row = ChatMessage(
        telegram_chat_id=telegram_chat_id,
        telegram_user_id=telegram_user_id,
        role=role,
        content=content,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def fetch_conversation_messages(
    session: AsyncSession,
    *,
    telegram_chat_id: int,
    limit: int = 60,
) -> Sequence[ChatMessage]:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.telegram_chat_id == telegram_chat_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    rows.reverse()
    return rows


def messages_to_anthropic(
    rows: Sequence[ChatMessage],
) -> list[dict[str, str]]:
    return [{"role": r.role, "content": r.content} for r in rows]


async def get_active_agent(
    session: AsyncSession,
    *,
    telegram_chat_id: int,
) -> str:
    stmt = select(ChatSession).where(
        ChatSession.telegram_chat_id == telegram_chat_id,
    )
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        return "personal"
    return row.active_agent or "personal"


async def set_active_agent(
    session: AsyncSession,
    *,
    telegram_chat_id: int,
    active_agent: str,
) -> None:
    stmt = select(ChatSession).where(
        ChatSession.telegram_chat_id == telegram_chat_id,
    )
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        session.add(
            ChatSession(
                telegram_chat_id=telegram_chat_id,
                active_agent=active_agent,
            )
        )
    else:
        row.active_agent = active_agent
    await session.commit()
