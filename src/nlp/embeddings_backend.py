from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SentenceTransformerEmbedder:
    """
    Minimal backend wrapper to satisfy BERTopic's expected interface.

    BERTopic (some versions) calls:
      - embedding_model.embed_documents(documents, verbose=?)
    Optionally it may call:
      - embedding_model.embed_words(words, verbose=?)
    """
    model: SentenceTransformer
    batch_size: int = 64
    normalize_embeddings: bool = False

    def embed_documents(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        # SentenceTransformer uses encode(...)
        emb = self.model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=verbose,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return emb

    def embed_words(self, words: List[str], verbose: bool = False) -> np.ndarray:
        # used in some topic representations
        emb = self.model.encode(
            words,
            batch_size=self.batch_size,
            show_progress_bar=verbose,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return emb
