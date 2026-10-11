"""Track 2 - 한국어 복합명사 추출 (공용).

계획서 §11.3 [1] / §11.2: bridge entity 는 반드시 **복합명사**(단일 명사 제외).
복합명사 = 연속된 명사 형태소(NNG/NNP) 2개 이상의 묶음.
  예) "정보화기획과" -> 정보화/기획/과 (3 morph) => 복합명사 ✓
      "예산"       -> 예산 (1 morph)            => 단일명사 ✗ (제외)
"""
from kiwipiepy import Kiwi

_NOUN = {"NNG", "NNP"}


def extract_compounds(text: str, kiwi: Kiwi, min_morphs: int = 2):
    """연속 명사 형태소(NNG/NNP) 묶음 중 min_morphs 개 이상을 복합명사로 반환.

    동일 passage 내 중복 포함 리스트. (DF 계산 시 호출부에서 set 처리)
    """
    out = []
    run = []
    for tok in kiwi.tokenize(text):
        if tok.tag in _NOUN:
            run.append(tok.form)
        else:
            if len(run) >= min_morphs:
                out.append("".join(run))
            run = []
    if len(run) >= min_morphs:
        out.append("".join(run))
    return out
