FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Cache installed requirements between source-code changes.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY . .

EXPOSE 8502

CMD ["python", "-m", "streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8502", "--server.headless=true"]
