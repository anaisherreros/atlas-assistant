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


class ConversationMemory(Base):
    __tablename__ = "conversation_memories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    telegram_chat_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    key_facts: Mapped[str] = mapped_column(Text, default="{}")
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    messages_count: Mapped[int] = mapped_column(default=0)


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


async def fetch_known_chat_ids(
    session: AsyncSession,
) -> list[int]:
    stmt = select(ChatSession.telegram_chat_id).order_by(ChatSession.telegram_chat_id.asc())
    result = await session.execute(stmt)
    chat_ids = [chat_id for chat_id in result.scalars().all() if isinstance(chat_id, int) and chat_id > 0]
    return chat_ids


def should_update_memory(message_count: int) -> bool:
    if message_count <= 0:
        return False
    return message_count % 20 == 0


async def get_memory(
    session: AsyncSession,
    *,
    telegram_chat_id: int,
) -> ConversationMemory | None:
    stmt = select(ConversationMemory).where(
        ConversationMemory.telegram_chat_id == telegram_chat_id,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def save_memory(
    session: AsyncSession,
    *,
    telegram_chat_id: int,
    summary: str,
    key_facts: str = "{}",
) -> ConversationMemory:
    memory = await get_memory(session, telegram_chat_id=telegram_chat_id)
    if memory is None:
        memory = ConversationMemory(
            telegram_chat_id=telegram_chat_id,
            summary=summary,
            key_facts=key_facts,
            messages_count=0,
        )
        session.add(memory)
    else:
        memory.summary = summary
        memory.key_facts = key_facts
        memory.messages_count = 0
    await session.commit()
    await session.refresh(memory)
    return memory


async def update_memory(
    session: AsyncSession,
    *,
    telegram_chat_id: int,
    new_info: str,
    key_facts: str = "{}",
) -> ConversationMemory:
    memory = await get_memory(session, telegram_chat_id=telegram_chat_id)
    if memory is None:
        memory = ConversationMemory(
            telegram_chat_id=telegram_chat_id,
            summary=new_info,
            key_facts=key_facts,
            messages_count=0,
        )
        session.add(memory)
    else:
        memory.summary = new_info
        memory.key_facts = key_facts
        memory.messages_count = 0
    await session.commit()
    await session.refresh(memory)
    return memory


async def increment_memory_counter(
    session: AsyncSession,
    *,
    telegram_chat_id: int,
) -> int:
    memory = await get_memory(session, telegram_chat_id=telegram_chat_id)
    if memory is None:
        memory = ConversationMemory(
            telegram_chat_id=telegram_chat_id,
            summary="",
            key_facts="{}",
            messages_count=1,
        )
        session.add(memory)
    else:
        memory.messages_count += 1
    await session.commit()
    await session.refresh(memory)
    return memory.messages_count


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


async def ensure_chat_session(
    session: AsyncSession,
    *,
    telegram_chat_id: int,
    active_agent: str = "personal",
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
        await session.commit()


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
