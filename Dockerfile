# Runtime only: test tools, notebooks and model encoders stay outside the image.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////var/lib/sacvoc/sac_voc.db

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 sacvoc \
    && useradd --uid 10001 --gid 10001 --create-home sacvoc \
    && mkdir -p /var/lib/sacvoc \
    && chown 10001:10001 /var/lib/sacvoc

COPY --chown=10001:10001 app/ ./app/
COPY --chown=10001:10001 data/ ./data/

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
