from sqlalchemy import text
from app.db import AsyncSessionLocal

async def get_db_session_for_user(user_id: str):
    async with AsyncSessionLocal() as session:
        await session.execute(text("SET app.current_user_id = :uid"), {"uid": user_id})
        yield session