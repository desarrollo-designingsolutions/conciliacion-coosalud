"""
Módulo de almacenamiento S3 para archivos persistentes.

Si S3_BUCKET no está configurado, todas las funciones retornan error
y el sistema sigue usando disco local (fallback).

Variables de entorno:
  S3_BUCKET  — nombre del bucket (obligatorio para activar S3)
  S3_REGION  — región AWS (default: us-east-1)
  S3_PREFIX  — prefijo para las keys (default: "conciliacion/")
"""

import os
import logging
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception


def s3_enabled() -> bool:
    return bool(os.getenv("S3_BUCKET")) and boto3 is not None


def get_s3_client():
    """
    Retorna (client, bucket, prefix) o (None, None, error_msg).
    """
    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        return None, None, "S3_BUCKET no configurado"
    if boto3 is None:
        return None, None, "boto3 no instalado"

    region = os.getenv("S3_REGION", "us-east-1").strip()
    prefix = os.getenv("S3_PREFIX", "conciliacion/").strip()

    try:
        client = boto3.client("s3", region_name=region)
        return client, bucket, prefix
    except Exception as exc:
        return None, None, f"Error creando cliente S3: {exc}"


def build_s3_key(subpath: str) -> str:
    """Construye la key completa: {S3_PREFIX}{subpath}"""
    prefix = os.getenv("S3_PREFIX", "conciliacion/").strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"{prefix}{subpath}"


def upload_file(local_path: str, s3_key: str) -> tuple[bool, str]:
    """
    Sube archivo local a S3.
    Retorna (True, s3_key) o (False, error_msg).
    """
    client, bucket, _ = get_s3_client()
    if client is None:
        return False, bucket or "S3 no disponible"

    try:
        client.upload_file(local_path, bucket, s3_key)
        logging.info(f"S3: subido {local_path} → s3://{bucket}/{s3_key}")
        return True, s3_key
    except Exception as exc:
        logging.error(f"S3: error subiendo {local_path}: {exc}")
        return False, f"Error subiendo a S3: {exc}"


def download_file_bytes(s3_key: str) -> tuple[bytes | None, str]:
    """
    Descarga archivo de S3 a memoria.
    Retorna (bytes, "") o (None, error_msg).
    """
    client, bucket, _ = get_s3_client()
    if client is None:
        return None, bucket or "S3 no disponible"

    try:
        response = client.get_object(Bucket=bucket, Key=s3_key)
        data = response["Body"].read()
        return data, ""
    except Exception as exc:
        logging.error(f"S3: error descargando {s3_key}: {exc}")
        return None, f"Error descargando de S3: {exc}"


def generate_presigned_url(s3_key: str, filename: str, expiration: int = 3600) -> str | None:
    """
    Genera una URL pre-firmada para descargar un archivo de S3 con el nombre indicado.
    Retorna la URL o None si falla.
    """
    client, bucket, _ = get_s3_client()
    if client is None:
        return None
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": bucket,
                "Key": s3_key,
                "ResponseContentDisposition": f'attachment; filename="{filename}"',
            },
            ExpiresIn=expiration,
        )
        return url
    except Exception as exc:
        logging.error(f"S3: error generando URL pre-firmada para {s3_key}: {exc}")
        return None


def read_file_bytes(file_path: str) -> bytes | None:
    """
    Lee archivo desde disco local o S3 según la ruta.
    - Si empieza con '/' → disco local
    - Si no → S3 key
    Retorna bytes o None si no se pudo leer.
    """
    if not file_path:
        return None

    if file_path.startswith("/"):
        p = Path(file_path)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return f.read()
            except Exception:
                return None
        return None

    # S3
    data, err = download_file_bytes(file_path)
    if err:
        logging.warning(f"No se pudo leer desde S3 ({file_path}): {err}")
    return data
