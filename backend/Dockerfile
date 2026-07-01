FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch/torchvision from PyTorch's CPU wheel index to avoid
# pulling the multi-GB CUDA build (the model runs on CPU).
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

WORKDIR /app/app

EXPOSE 8000

# Health check hits the root route, which reports the API + loaded classes
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/').read()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
