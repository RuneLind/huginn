import os
from sentence_transformers import SentenceTransformer

# Use cached models only — skip HF Hub network requests for faster startup
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

class SentenceEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", query_prefix="", passage_prefix=""):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        return self.model.encode(text)

    def embed_query(self, text):
        return self.model.encode(self.query_prefix + text)

    def embed_passages(self, texts):
        prefixed = [self.passage_prefix + t for t in texts]
        return self.model.encode(prefixed)

    def get_number_of_dimensions(self):
        return self.model.get_sentence_embedding_dimension()
