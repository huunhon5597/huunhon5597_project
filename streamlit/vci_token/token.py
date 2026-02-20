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
    return {
        "url": os.environ.get(
            "VCI_TOKEN_URL",
            "https://trading.vietcap.com.vn/api/iam-external-service/v1/authentication/login",
        ),
        "username": os.environ.get("VCI_USERNAME", "068C405016"),
        "password": os.environ.get(
            "VCI_PASSWORD",
            "a2298c602ad9f0236358e853123c43ebbb90fe2419f3e91dd38e10967dbef8f5ab0b68220166d0ca4ef01dbf85f265305ec9c6c7a54437392de5be69cef01107215e623c68531d7b2e3df97a85a771b36d04cd7fa040547a730005f1f9225a53866ce13c5c8960a12c1fbed6d89cd1d8c2324d0e34ccd2c5d641dad0de422296b6ff674e67a2fadb6362e3aad37d59c8aefaaf04789d15aaccd0ce79d79b2b064603e6de681f1ba3de0cf18febd9cb24ce51f403e6866c8116debfcf503e6fa9cd487f3408db2c6a05b15cd1511ee3261653d54a5921c05dc1e2fecded9b44b7523b8040c502f051a72b6b6ed51676f99cada5d7d7b451e874b633d32a7022ae",
        ),
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
