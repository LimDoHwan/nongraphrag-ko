"""평가 메트릭: Contain-Acc (보조), GPT-Acc (주).

- Contain-Acc: 한국어 조사 제거 후 substring match.
- GPT-Acc: Gemma(외부 vLLM) 기반 의미 일치 판정.
"""
import re

from .llm_client import LLMClient

# 한국어 조사 패턴 (어절 끝 보정용)
_JOSA_PATTERN = re.compile(
    r"(은|는|을|를|이|가|와|과|에서|에게|으로|로|에|의|도|만|까지|부터)\b"
)


def _normalize(text: str) -> str:
    text = _JOSA_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def contain_acc(prediction: str, answer: str) -> bool:
    """조사 제거 후 정답이 예측 안에 포함되는지."""
    return _normalize(answer) in _normalize(prediction)


_EVAL_PROMPT = """다음 질문에 대한 예측 답변이 정답과 의미적으로 일치하는지 판정하세요.

질문: {question}
정답: {ground_truth}
예측 답변: {prediction}

일치하면 "YES", 불일치하면 "NO"로만 답하세요."""


def gpt_acc(question: str, ground_truth: str, prediction: str,
            client: LLMClient = None) -> bool:
    client = client or LLMClient()
    out = client.complete(
        _EVAL_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        ),
        max_tokens=8,
    )
    return out.strip().upper().startswith("YES")
