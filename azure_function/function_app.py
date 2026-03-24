"""Azure Function — HTTP gateway that proxies /api/* requests to the ACI FastAPI container."""

import logging
import os

import azure.functions as func
import httpx

logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# ACI_ENDPOINT is injected via app settings in Terraform, e.g. http://<fqdn>:8000
_ACI_ENDPOINT = os.environ.get("ACI_ENDPOINT", "http://localhost:8000")
_TIMEOUT_SECONDS = 30


@app.route(route="{*route}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def proxy(req: func.HttpRequest) -> func.HttpResponse:
    """Forward all /api/* and /health requests to the FastAPI container on ACI.

    Headers (including X-API-Key) are forwarded transparently so the FastAPI
    auth middleware sees the original credentials.
    """
    path = req.route_params.get("route", "")
    target_url = f"{_ACI_ENDPOINT}/{path}"

    # Preserve original query string
    if req.params:
        query_string = "&".join(f"{k}={v}" for k, v in req.params.items())
        target_url = f"{target_url}?{query_string}"

    # Forward relevant headers; strip hop-by-hop headers that httpx handles itself
    forward_headers = {
        k: v
        for k, v in req.headers.items()
        if k.lower() not in {"host", "content-length", "transfer-encoding"}
    }

    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.request(
                method=req.method,
                url=target_url,
                headers=forward_headers,
                content=req.get_body(),
            )

        logger.info(
            "Proxied %s /%s → %d (%d bytes)",
            req.method, path, response.status_code, len(response.content),
        )

        return func.HttpResponse(
            body=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            mimetype=response.headers.get("content-type", "application/json"),
        )

    except httpx.TimeoutException:
        logger.error("Upstream timeout proxying %s /%s", req.method, path)
        return func.HttpResponse(
            body='{"error":"Gateway Timeout","detail":"Upstream service did not respond in time."}',
            status_code=504,
            mimetype="application/json",
        )
    except Exception as exc:
        logger.error("Gateway error proxying %s /%s: %s", req.method, path, exc)
        return func.HttpResponse(
            body='{"error":"Bad Gateway","detail":"Could not reach upstream service."}',
            status_code=502,
            mimetype="application/json",
        )
