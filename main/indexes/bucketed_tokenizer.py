# Token lengths round up to a multiple of this, so a batch has one of 4 lengths
# (128..512) instead of any length up to 512. PyTorch on the Mac GPU (MPS) keeps
# memory for every input shape it sees and never frees it (empty_cache does not
# help): unpadded, the server grew ~3 GB per new query until macOS killed it at
# ~150 GB; padded, a probe of varied reranker batches stopped growing at ~5.7 GB.
PAD_TO_MULTIPLE_OF = 128


class BucketedTokenizer:
    """Wraps a tokenizer so every call pads to a multiple of PAD_TO_MULTIPLE_OF.

    Not an instance of PreTrainedTokenizerBase, so CrossEncoderTrainer would not
    infer it; the models using it are never trained.
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
