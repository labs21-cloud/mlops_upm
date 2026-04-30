FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY tests ./tests
COPY README.md .

EXPOSE 8000

CMD sh -c "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"