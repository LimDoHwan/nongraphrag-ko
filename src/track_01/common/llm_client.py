"""Gemma (외부 vLLM 서버) 호출 wrapper.

KT Cloud vLLM 프록시는 쿠키 인증을 사용하므로 표준 OpenAI SDK 대신
httpx 로 직접 요청한다. (.env: GEMMA_URL / GEMMA_COOKIE / GEMMA_MODEL)
"""
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self, max_retries: int = 4, timeout: float = 240.0):
        self.url = os.getenv("GEMMA_URL")
        self.model = os.getenv("GEMMA_MODEL")
        cookie = os.getenv("GEMMA_COOKIE")
        if not (self.url and self.model and cookie):
            raise RuntimeError(
                "GEMMA_URL / GEMMA_MODEL / GEMMA_COOKIE 가 .env 에 설정되어야 합니다."
            )
        self.headers = {
            "Content-Type": "application/json",
            "Cookie": cookie,
        }
        self.max_retries = max_retries
        self.client = httpx.Client(timeout=timeout, verify=True)

    def chat_full(self, messages: list, temperature: float = 0.0,
                  max_tokens: int = 1024) -> tuple[str, str]:
        """(content, finish_reason) 반환. finish_reason != 'stop' 이면 잘림."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.post(self.url, headers=self.headers,
                                        json=payload)
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                return (choice["message"]["content"],
                        choice.get("finish_reason", "stop"))
            except Exception as e:  # noqa: BLE001
                last_err = e
                # 지수 백오프 (쿠키 만료/일시적 지연 대비)
                time.sleep(2 ** attempt)
        raise RuntimeError(
            f"LLM 호출 {self.max_retries}회 실패. 마지막 오류: {last_err}"
        )

    def chat(self, messages: list, temperature: float = 0.0,
             max_tokens: int = 1024) -> str:
        return self.chat_full(messages, temperature, max_tokens)[0]

    def complete(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


class OpenAIClient:
    """OpenAI Chat Completions wrapper. LLMClient 와 동일 인터페이스.

    .env: OPENAI_KEY (또는 OPENAI_API_KEY). 모델 기본값 gpt-4o.
    """

    def __init__(self, model: str = "gpt-4o", max_retries: int = 5,
                 timeout: float = 120.0):
        from openai import OpenAI
        key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_KEY 가 .env 에 설정되어야 합니다.")
        self.model = model
        # SDK 자체 재시도 + 타임아웃 사용 (429/일시 오류 자동 백오프)
        self.client = OpenAI(api_key=key, max_retries=max_retries,
                             timeout=timeout)

    def chat_full(self, messages: list, temperature: float = 0.0,
                  max_tokens: int = 1024) -> tuple[str, str]:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        return (choice.message.content or "",
                choice.finish_reason or "stop")

    def chat(self, messages: list, temperature: float = 0.0,
             max_tokens: int = 1024) -> str:
        return self.chat_full(messages, temperature, max_tokens)[0]

    def complete(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


def make_client(backend: str = "gemma", model: str = None):
    """backend 에 따라 적절한 클라이언트 생성."""
    if backend == "openai":
        return OpenAIClient(model=model or "gpt-4o")
    return LLMClient()


if __name__ == "__main__":
    print(LLMClient().complete("안녕하세요"))
