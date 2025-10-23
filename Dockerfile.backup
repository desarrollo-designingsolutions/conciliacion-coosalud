# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=America/Bogota

RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
  && rm -rf /var/lib/apt/lists/*
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY csv_conciliation_loader.py ./

# Directorios de IO (se montan como volúmenes con docker compose)
RUN mkdir -p /data/in /data/out

ENTRYPOINT ["python", "csv_conciliation_loader.py"]
CMD ["--help"]