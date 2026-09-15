# Serve-only huginn image. Build it with scripts/container/image_build.py, which
# stages src/ (git archive of a pinned commit) and collections/ (the
# package_collection.py tarballs), verifies both, and checks the image. A build
# from the repo root has neither folder in its context and fails.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

# Hash-pinned; --torch-backend cpu takes PyTorch's own packages from its CPU index.
RUN --mount=from=ghcr.io/astral-sh/uv:0.8.14@sha256:f3660c56d5b08d6c516360981bedc439f499b9bf37f46a216018da3777a74011,source=/uv,target=/tmp/uv \
    --mount=type=bind,source=src/requirements/serve.txt,target=/tmp/serve.txt \
    /tmp/uv pip install --system --torch-backend cpu --require-hashes --no-cache --verbose -r /tmp/serve.txt

RUN --mount=type=bind,source=src/scripts/container,target=/tmp/container \
    HF_HUB_DISABLE_XET=1 python /tmp/container/fetch_models.py fetch /tmp/container/models.lock.json

COPY src/ /app/
COPY collections/ /app/

WORKDIR /app
# USER: UID 1069 has no passwd entry, and torch calls getpass.getuser() at import.
ENV HF_HUB_OFFLINE=1 \
    HUGINN_QUERY_LOG=off \
    USER=huginn
USER 1069:1069
EXPOSE 8321
ENTRYPOINT ["python", "knowledge_api_server.py", "--host", "0.0.0.0", "--port", "8321"]
