import requests

BASE_URL = "http://localhost:8000"

_TIMEOUT_ANALYZE = 30
_TIMEOUT_HEALTH = 5


def analyze_text(
    text: str,
    document_type: str = "essay",
    language: str | None = None,
) -> dict:
    payload: dict = {"text": text, "document_type": document_type}
    if language is not None:
        payload["language"] = language
    response = requests.post(
        f"{BASE_URL}/api/v1/analyze",
        json=payload,
        timeout=_TIMEOUT_ANALYZE,
    )
    response.raise_for_status()
    return response.json()


def health_check() -> dict:
    response = requests.get(f"{BASE_URL}/health", timeout=_TIMEOUT_HEALTH)
    response.raise_for_status()
    return response.json()
