#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalcula completamente la tabla thirds_summary_conciliation a partir de
conciliation_results + auditory_final_reports + invoice_audits.

Uso:
  python rebuild_thirds_summary_conciliation.py [--no-truncate] [--verbose]

Requiere variables de entorno:
  MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD
"""
import os
import sys
import argparse
import pymysql


def get_conn():
    host = os.getenv("MYSQL_HOST")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    db = os.getenv("MYSQL_DB")
    user = os.getenv("MYSQL_USER")
    pw = os.getenv("MYSQL_PASSWORD")
    if not all([host, db, user, pw]):
        raise RuntimeError(
            "Variables de entorno MySQL incompletas (MYSQL_HOST, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD)"
        )
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=pw,
        database=db,
        autocommit=False,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
    )
    with conn.cursor() as cur:
        cur.execute("SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci")
        cur.execute("SET collation_connection = 'utf8mb4_unicode_ci'")
    conn.commit()
    return conn


AGG_SQL = """
SELECT
  ia.third_id AS nit,
  COALESCE(SUM(cr.accepted_value_eps), 0) AS valor_aceptado_eps,
  COALESCE(SUM(cr.accepted_value_ips), 0) AS valor_aceptado_ips,
  COALESCE(SUM(cr.eps_ratified_value), 0) AS valor_ratificado
FROM conciliation_results cr
INNER JOIN auditory_final_reports afr
  ON afr.id = cr.auditory_final_report_id
INNER JOIN invoice_audits ia
  ON ia.id = afr.factura_id
GROUP BY ia.third_id
"""


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS thirds_summary_conciliation (
  nit BIGINT NOT NULL,
  valor_aceptado_eps DECIMAL(18,2) NOT NULL DEFAULT 0,
  valor_aceptado_ips DECIMAL(18,2) NOT NULL DEFAULT 0,
  valor_ratificado   DECIMAL(18,2) NOT NULL DEFAULT 0,
  PRIMARY KEY (nit)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-truncate", action="store_true", help="No vaciar la tabla; hacer upsert")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            if not args.no_truncate:
                cur.execute("TRUNCATE TABLE thirds_summary_conciliation")
                insert_sql = (
                    "INSERT INTO thirds_summary_conciliation (nit, valor_aceptado_eps, valor_aceptado_ips, valor_ratificado) "
                    + AGG_SQL
                )
                cur.execute(insert_sql)
                affected = cur.rowcount
                conn.commit()
                print(f"Tabla reconstruida vía TRUNCATE+INSERT. Filas: {affected}")
            else:
                # Upsert completo
                upsert_sql = (
                    "INSERT INTO thirds_summary_conciliation (nit, valor_aceptado_eps, valor_aceptado_ips, valor_ratificado) "
                    + AGG_SQL +
                    " ON DUPLICATE KEY UPDATE "
                    " valor_aceptado_eps=VALUES(valor_aceptado_eps),"
                    " valor_aceptado_ips=VALUES(valor_aceptado_ips),"
                    " valor_ratificado=VALUES(valor_ratificado)"
                )
                cur.execute(upsert_sql)
                affected = cur.rowcount
                conn.commit()
                print(f"Tabla actualizada vía UPSERT. Filas afectadas: {affected}")
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
