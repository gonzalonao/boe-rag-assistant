# BOE RAG Assistant — container image for a Hugging Face Space (Docker SDK).
#
# The build bakes everything the running container needs into the image so the
# cold start does no network I/O and no passage encoding:
#   * CPU-only PyTorch (avoids the multi-GB CUDA wheel on a CPU Space),
#   * the corpus Parquet (pulled from the published HF dataset),
#   * precomputed E5 passage embeddings,
#   * the embedding + cross-encoder model weights (HF cache).
# At runtime the service just loads them and serves on port 7860.
FROM python:3.12-slim

# Hugging Face Spaces run the container as uid 1000; match it so the baked-in
# caches and data are owned by (and writable for) the runtime user.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    BOE_CORPUS_PATH=/home/user/app/data/corpus/boe-2024.parquet \
    BOE_EMBEDDINGS_PATH=/home/user/app/data/corpus/boe-2024-embeddings.npz \
    BOE_REPORTS_DIR=/home/user/app/reports

WORKDIR /home/user/app

# Install CPU-only torch first so sentence-transformers reuses it instead of
# pulling the default CUDA build.
RUN pip install --no-cache-dir --user \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install the package (dependencies cached on their own layer).
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user ".[api,ml,ui,hub]"

# App data and the build scripts.
COPY --chown=user reports ./reports
COPY --chown=user scripts ./scripts

# Bake the corpus, precomputed embeddings, and model weights into the image.
# The embeddings matrix is fetched (not recomputed) so the build does not encode
# the whole corpus on CPU — that scales with the corpus and risks build timeouts.
# It is republished in lock-step with the corpus; app.build_engine re-encodes at
# boot if the ids ever fail to match, so a stale matrix can't serve wrong results.
ARG CORPUS_REPO_ID=gonzalonao/boe-corpus
RUN python scripts/fetch_corpus.py --repo-id "${CORPUS_REPO_ID}" --out "${BOE_CORPUS_PATH}" \
 && python scripts/fetch_embeddings.py --repo-id "${CORPUS_REPO_ID}" --out "${BOE_EMBEDDINGS_PATH}" \
 && python scripts/warm_models.py

EXPOSE 7860
CMD ["uvicorn", "boe_rag.service.app:app", "--host", "0.0.0.0", "--port", "7860"]
