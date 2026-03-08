"""Conexion a Snowflake para el dashboard BI."""

import os
import snowflake.connector
import streamlit as st

SF_DEFAULTS = {
    "account": "gra06523.us-east-1",
    "user": "CT_UTAM",
    "role": "CT_088_UTAM",
    "warehouse": "COO_CUENTAS_MEDICAS_DWH",
    "database": "DWH_DEV",
    "schema": "UTAM",
}


def get_sf_conn():
    """Retorna una conexion a Snowflake usando variables de entorno o defaults."""
    password = os.environ.get("SNOWFLAKE_PASSWORD", "")
    if not password:
        st.error("Variable de entorno SNOWFLAKE_PASSWORD no configurada.")
        st.stop()

    return snowflake.connector.connect(
        user=os.environ.get("SNOWFLAKE_USER", SF_DEFAULTS["user"]),
        password=password,
        account=os.environ.get("SNOWFLAKE_ACCOUNT", SF_DEFAULTS["account"]),
        role=os.environ.get("SNOWFLAKE_ROLE", SF_DEFAULTS["role"]),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", SF_DEFAULTS["warehouse"]),
        database=os.environ.get("SNOWFLAKE_DATABASE", SF_DEFAULTS["database"]),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", SF_DEFAULTS["schema"]),
        session_parameters={"CLIENT_SESSION_KEEP_ALIVE": True},
        autocommit=True,
    )
