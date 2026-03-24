"""In-memory per-IP rate limiter — toggleable via ENABLE_RATE_LIMIT env var."""

import logging
import time
from collections import defaultdict, deque

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP.

    Tracks request timestamps per IP in a deque. Requests older than 60 s
    are evicted before each check.  When ENABLE_RATE_LIMIT=false the
    middleware is a no-op.
    """

    WINDOW_SECONDS = 60

    def __init__(self, app) -> None:
        super().__init__(app)
        # ip -> deque of request timestamps (float)
        self._windows: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.enable_rate_limit:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._windows[client_ip]

        # Evict timestamps outside the current window
        while window and window[0] < now - self.WINDOW_SECONDS:
            window.popleft()

        if len(window) >= settings.rate_limit_rpm:
            retry_after = int(self.WINDOW_SECONDS - (now - window[0])) + 1
            logger.warning("Rate limit exceeded for IP %s", client_ip)
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error": "Too Many Requests",
                    "detail": f"Rate limit of {settings.rate_limit_rpm} rpm exceeded.",
                    "retry_after_seconds": retry_after,
                },
            )

        window.append(now)
        return await call_next(request)
