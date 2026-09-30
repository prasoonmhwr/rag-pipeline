from typing import AsyncGenerator
from fastapi import Header, HTTPException, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import AsyncSessionLocal

async def get_current_user_id(x_user_id: str | None = Header(default=None)) -> str:
    # Placeholder for real auth — swap this for JWT/session validation in production.
    # The point right now is the shape: something upstream of the database resolves
    # who's asking, and hands us a stable user_id.
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-Id header")
    return x_user_id

async def get_db_session_for_user(
    user_id: str = Depends(get_current_user_id),
) -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT set_config('app.current_user_id', :uid, false)"), {"uid": user_id})
        yield session