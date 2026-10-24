from urllib.parse import urlparse

import httpx


class CloudError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Ollama Cloud request failed with HTTP {status_code}: {detail[:300]}")
        self.status_code = status_code


class CostCeilingExceeded(RuntimeError):
    pass


def assert_cloud_host(base_url: str) -> None:
    host = (urlparse(base_url).hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        raise RuntimeError("Refusing local Ollama host; this run is Ollama Cloud only")
    if urlparse(base_url).scheme != "https":
        raise RuntimeError("Ollama Cloud base URL must use HTTPS")


class OllamaCloudClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 300) -> None:
        assert_cloud_host(base_url)
        if not api_key:
            raise RuntimeError("OLLAMA_API_KEY is empty")
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_seconds, connect=30),
        )

    def close(self) -> None:
        self._client.close()

    def list_models(self) -> dict[str, str]:
        response = self._request("GET", "/api/tags")
        return {
            item.get("name", ""): item.get("digest", "")
            for item in response.get("models", [])
        }

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        }
        try:
            return self._request("POST", "/api/chat", payload)
        except CloudError as error:
            if error.status_code == 400 and "think" in str(error).casefold():
                payload.pop("think")
                return self._request("POST", "/api/chat", payload)
            raise

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        try:
            response = self._client.request(method, path, json=payload)
        except httpx.HTTPError as error:
            raise CloudError(0, error.__class__.__name__) from None
        if response.status_code >= 400:
            raise CloudError(response.status_code, response.text)
        try:
            body = response.json()
        except ValueError:
            raise CloudError(response.status_code, "Response was not JSON") from None
        if not isinstance(body, dict):
            raise CloudError(response.status_code, "Response JSON was not an object")
        return body
