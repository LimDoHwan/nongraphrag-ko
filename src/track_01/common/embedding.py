"""bge-m3 임베딩 wrapper (3개 시스템 공통).

DEVICE 환경변수(cuda/cpu)를 따르되, cuda 불가/OOM 시 cpu 로 자동 폴백.
"""
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


class Embedder:
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or os.getenv(
            "EMBEDDING_MODEL_PATH", "./models/bge-m3")
        device = device or os.getenv("DEVICE", "cuda")
        try:
            self.model = SentenceTransformer(self.model_path, device=device)
            self.device = device
        except Exception:  # noqa: BLE001 - cuda 불가/OOM 등
            self.model = SentenceTransformer(self.model_path, device="cpu")
            self.device = "cpu"

    def encode(self, texts, batch_size: int = 32,
               normalize: bool = True) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,  # cosine == inner product
            show_progress_bar=False,
        )
        return vecs[0] if single else vecs


if __name__ == "__main__":
    emb = Embedder()
    v = emb.encode(["테스트 문장"])
    print(f"device={emb.device}, shape={np.asarray(v).shape}")
