# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=America/Bogota

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    libreoffice \
    libreoffice-calc \
  && rm -rf /var/lib/apt/lists/*
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar TODOS los archivos de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p /data/in /data/out

# Exponer puerto de Streamlit
EXPOSE 8501

# Ejecutar Streamlit por defecto (producción)
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]