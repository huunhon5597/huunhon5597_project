import os
import time
import requests
import json
from typing import Optional

_CACHE: dict = {}


class TokenError(RuntimeError):
    pass


def _load_default_config():
    """Load configuration from environment with sensible defaults."""
    # Try to get from Streamlit secrets first (if available)
    # Streamlit secrets are automatically available in os.environ when deployed
    return {
        "url": os.environ.get(
            "VCI_TOKEN_URL",
            "https://trading.vietcap.com.vn/api/iam-external-service/v1/authentication/login",
        ),
        "username": os.environ.get("VCI_USERNAME", ""),
        "password": os.environ.get("VCI_PASSWORD", ""),
    }


def _default_headers():
    # include several headers to mimic a browser request; server expects some of these
    return {
        "Accept": "application/json",
        "Accept-Language": os.environ.get("VCI_ACCEPT_LANGUAGE", "en-US,en;q=0.9,vi;q=0.8"),
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Origin": os.environ.get("VCI_ORIGIN", "https://trading.vietcap.com.vn"),
        "Referer": os.environ.get(
            "VCI_REFERER",
            "https://trading.vietcap.com.vn/iq/coverage?login-from=individual",
        ),
        "User-Agent": os.environ.get(
            "VCI_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        ),
        "client-id": os.environ.get("VCI_CLIENT_ID", "a670914c-8964-4b2c-a289-6de4d5b9d2c4"),
        "client-secret": os.environ.get("VCI_CLIENT_SECRET", "42IGbQ9oXZ1p2JK"),
        "device-id": os.environ.get("VCI_DEVICE_ID", "194d5c0250f11306"),
        "grant-type": "password",
    }


def get_token(force_refresh: bool = False, cache_ttl: int = 50) -> str:
    """Return an authentication token for VCI trading API.

    Args:
        force_refresh: if True, ignore any cached token and request a new one.
        cache_ttl: seconds to keep token cached. Default 50s; adjust as needed.

    Returns:
        token string

    Raises:
        TokenError on network or unexpected response formats.
    """
    now = time.time()
    if not force_refresh:
        cached = _CACHE.get("token")
        if cached and now - cached["ts"] < cache_ttl:
            return cached["value"]

    cfg = _load_default_config()
    url = cfg["url"]
    
    # Validate credentials are set
    if not cfg["username"] or not cfg["password"]:
        raise TokenError("VCI_USERNAME and VCI_PASSWORD environment variables are not set. "
                        "Please configure them in Streamlit Cloud secrets or .env file.")
    
    payload_dict = {"username": cfg["username"], "password": cfg["password"]}
    headers = _default_headers()

    try:
        # send as JSON so requests sets the correct body and Content-Type
        resp = requests.post(url, headers=headers, json=payload_dict, timeout=10)
    except requests.RequestException as exc:
        raise TokenError(f"request failed: {exc}") from exc

    # Provide clearer errors for non-200 responses and invalid JSON
    if not resp.ok:
        snippet = resp.text[:400].replace('\n', ' ')
        raise TokenError(f"bad response: status={resp.status_code} body={snippet}")

    try:
        data = resp.json()
    except ValueError:
        raise TokenError(f"invalid json response (status={resp.status_code}): {resp.text[:400]}")

    # Expected structure: {"data": {"token": "..."}}
    token = None
    try:
        token = data.get("data", {}).get("token")
    except Exception:
        token = None

    if not token:
        raise TokenError(f"token not found in response: {data}")

    _CACHE["token"] = {"value": token, "ts": now}
    return token
