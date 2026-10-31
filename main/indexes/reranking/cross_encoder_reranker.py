import logging
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Token lengths round up to a multiple of this, so a batch has one of 4 lengths
# (128..512) instead of any length up to 512. PyTorch on the Mac GPU (MPS) keeps
# memory for every input shape it sees and never frees it (empty_cache does not
# help): unpadded, the server grew ~3 GB per new query until macOS killed it at
# ~150 GB; padded, a probe of varied batches stopped growing at ~5.7 GB.
PAD_TO_MULTIPLE_OF = 128


class _BucketedTokenizer:
    """Wraps a tokenizer so every call pads to a multiple of PAD_TO_MULTIPLE_OF.

    Not an instance of PreTrainedTokenizerBase, so CrossEncoderTrainer would not
    infer it; this reranker is never trained.
    """

    _OWN = ("_tokenizer", "_multiple")

    def __init__(self, tokenizer, multiple):
        self._tokenizer = tokenizer
        self._multiple = multiple

    def __call__(self, *args, **kwargs):
        kwargs["pad_to_multiple_of"] = self._multiple
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        # Own fields and dunders never delegate: during deepcopy/unpickle
        # _tokenizer is not set yet, and a delegated __deepcopy__ would copy
        # the inner tokenizer instead of the wrapper.
        if name in self._OWN or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self._tokenizer, name)

    def __setattr__(self, name, value):
        # CrossEncoder.max_length's setter writes tokenizer.model_max_length.
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self._tokenizer, name, value)


class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model.

    Cross-encoders read query and document together through the full transformer,
    enabling much deeper relevance understanding than comparing pre-computed embeddings.
    """

    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self._model_name = model_name
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        self.model.tokenizer = _BucketedTokenizer(self.model.tokenizer, PAD_TO_MULTIPLE_OF)
        logger.info(f"Cross-encoder reranker loaded: {model_name}")

    @property
    def model_name(self):
        return self._model_name

    def rerank(self, query, scores, indexes, chunk_texts, top_k, return_ce_scores=False):
        """Rerank candidates using cross-encoder scores.

        Args:
            query: The search query string.
            scores: numpy array of shape (1, n) with original scores.
            indexes: numpy array of shape (1, n) with chunk IDs.
            chunk_texts: list of chunk text strings, same length as indexes[0].
            top_k: number of results to return.
            return_ce_scores: if True, also return a list of (chunk_id, ce_score)
                in CE-descending order across ALL input candidates (not just top_k).
                Use case: tracing — caller wants to see how every candidate scored.

        Returns:
            (new_scores, new_indexes) in numpy format consistent with indexer output
            (scores negated so lower = better). If return_ce_scores is True,
            returns (new_scores, new_indexes, ce_breakdown) instead.
        """
        if len(indexes[0]) == 0 or not chunk_texts:
            if return_ce_scores:
                return scores, indexes, []
            return scores, indexes

        pairs = [(query, text) for text in chunk_texts]
        ce_scores = self.model.predict(pairs, batch_size=8)

        # Sort by cross-encoder score descending (higher = more relevant)
        full_order = np.argsort(ce_scores)[::-1]
        ranked_indices = full_order[:top_k]

        new_indexes = np.array([[int(indexes[0][i]) for i in ranked_indices]], dtype=np.int64)
        # Negate scores so lower = better (consistent with L2 / RRF convention)
        new_scores = np.array([[-float(ce_scores[i]) for i in ranked_indices]], dtype=np.float32)

        if return_ce_scores:
            ce_breakdown = [(int(indexes[0][i]), float(ce_scores[i])) for i in full_order]
            return new_scores, new_indexes, ce_breakdown

        return new_scores, new_indexes
