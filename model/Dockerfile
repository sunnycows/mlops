# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app_lstm:app", "--host", "0.0.0.0", "--port", "80"]
