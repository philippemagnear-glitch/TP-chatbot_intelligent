FROM python:3.13-slim-bookworm

# uv installe les versions du projet sans recopier l'environnement Windows.
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Bibliothèque OpenMP utilisée par les calculs natifs de FAISS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

# Garder les chemins attendus, sans embarquer les données ni les secrets.
COPY App/API/rag_api.py ./App/API/rag_api.py
COPY App/API/openagenda.py ./App/API/openagenda.py
COPY App/Preprocessing/preprocess_openagenda.py ./App/Preprocessing/preprocess_openagenda.py
COPY App/RAG/chatbot.py App/RAG/build_index.py App/RAG/prompt_system_chatbot.md ./App/RAG/
COPY App/RAG/build_embeddings.py ./App/RAG/build_embeddings.py
COPY docker_entrypoint.py ./docker_entrypoint.py

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/App/Data \
    && chown appuser:appuser /app/App/Data
USER appuser

EXPOSE 8000
CMD ["python", "docker_entrypoint.py"]
