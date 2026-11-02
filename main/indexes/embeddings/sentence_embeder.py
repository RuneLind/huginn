import os

# Use cached models only — skip HF Hub network requests for faster startup.
# Must be set BEFORE importing sentence_transformers / huggingface_hub,
# because huggingface_hub reads HF_HUB_OFFLINE into a constant at import time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer

from ..bucketed_tokenizer import PAD_TO_MULTIPLE_OF, BucketedTokenizer

PASSAGE_BATCH_SIZE = 32


def _needs_bounded_shapes(model):
    # Only the Mac GPU keeps memory per input shape. On the CPU, padding and fill
    # only cost time: measured up to 10x slower on short passages.
    return model.device.type == "mps"


class SentenceEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", query_prefix="", passage_prefix=""):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model = SentenceTransformer(model_name)
        self._bound_shapes = _needs_bounded_shapes(self.model)
        if self._bound_shapes:
            self.model.tokenizer = BucketedTokenizer(self.model.tokenizer, PAD_TO_MULTIPLE_OF)

    def embed(self, text):
        return self.model.encode(text)

    def embed_query(self, text):
        return self.model.encode(self.query_prefix + text)

    def embed_passages(self, texts):
        prefixed = [self.passage_prefix + t for t in texts]
        # Empty rows fill the last batch, so its size is not one more input shape
        # the Mac GPU keeps memory for. encode sorts by length: they stay in that batch.
        fill = -len(prefixed) % PASSAGE_BATCH_SIZE if self._bound_shapes else 0
        vectors = self.model.encode(prefixed + [""] * fill, batch_size=PASSAGE_BATCH_SIZE)
        return vectors[: len(prefixed)]

    def get_number_of_dimensions(self):
        return self.model.get_sentence_embedding_dimension()
