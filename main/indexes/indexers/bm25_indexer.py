import re
import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text):
    return re.findall(r'\w+', text.lower())


class BM25Indexer:
    def __init__(self, name="indexer_BM25", serialized_state=None):
        self.name = name
        if serialized_state is not None:
            self._corpus_tokens = serialized_state["corpus_tokens"]
            self._ids = serialized_state["ids"]
            self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        else:
            self._corpus_tokens = []
            self._ids = []
            self._bm25 = None

    def get_name(self):
        return self.name

    def index_texts(self, ids, texts):
        tokenized = [_tokenize(t) for t in texts]
        self._corpus_tokens.extend(tokenized)
        self._ids.extend(ids)
        # Rebuild BM25 index with full corpus
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def remove_ids(self, ids):
        ids_to_remove = set(int(i) for i in ids)
        keep = [(tok, doc_id) for tok, doc_id in zip(self._corpus_tokens, self._ids)
                if int(doc_id) not in ids_to_remove]
        if keep:
            self._corpus_tokens, self._ids = map(list, zip(*keep))
        else:
            self._corpus_tokens, self._ids = [], []
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def serialize(self):
        return {
            "corpus_tokens": self._corpus_tokens,
            "ids": self._ids,
        }

    def search(self, text, number_of_results=10):
        if not self._bm25 or not self._ids:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)

        query_tokens = _tokenize(text)
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score descending, excluding zero-score entries
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = [i for i in sorted_indices if scores[i] > 0][:number_of_results]

        if not top_indices:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)

        result_ids = np.array([[self._ids[i] for i in top_indices]], dtype=np.int64)
        # Negate scores so lower = better (consistent with L2 distance convention)
        result_scores = np.array([[-scores[i] for i in top_indices]], dtype=np.float32)

        return result_scores, result_ids

    def get_size(self):
        return len(self._ids)
