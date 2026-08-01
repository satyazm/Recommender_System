FROM python:3.11-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/serving/ src/serving/
COPY models/ models/

ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
