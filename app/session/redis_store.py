"""Per-session conversation history, cached in Redis (separate from the document index)."""
import json
from functools import lru_cache

from redis.asyncio import Redis

from app.config import settings


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=True)


def _key(session_id: str) -> str:
    return f"session:{session_id}:history"


async def get_history(session_id: str) -> list[tuple[str, str]]:
    """Return recent (role, content) turns, oldest first."""
    raw_turns = await get_redis().lrange(_key(session_id), 0, -1)
    return [tuple(json.loads(turn)) for turn in raw_turns]


async def append_turn(session_id: str, role: str, content: str) -> None:
    redis = get_redis()
    key = _key(session_id)
    await redis.rpush(key, json.dumps([role, content]))
    # Keep only the most recent N turns so the prompt and Redis memory stay bounded.
    await redis.ltrim(key, -settings.session_history_max_turns, -1)
    await redis.expire(key, settings.session_ttl_seconds)
