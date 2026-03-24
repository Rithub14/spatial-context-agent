"""API key authentication middleware — toggleable via ENABLE_AUTH env var."""

import logging

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header on every request when ENABLE_AUTH is true.

    When ENABLE_AUTH=false all requests pass through without inspection.
    Health endpoint is always exempted so load balancers can probe freely.
    """

    EXEMPT_PATHS = {"/health"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.enable_auth or request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if api_key != settings.api_key:
            logger.warning("Rejected request — invalid or missing API key")
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Invalid or missing X-API-Key header."},
            )

        return await call_next(request)
