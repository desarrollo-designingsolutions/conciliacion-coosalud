# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Healthcare invoice reconciliation system for Coosalud (Colombian EPS). Manages conciliation between EPS and IPS providers: CSV validation, bulk database insertion, acta (reconciliation act) generation, and reporting. All UI text and comments are in Spanish.

## Tech Stack

- **Python 3.11**, no ORM — raw SQL via pymysql
- **Streamlit** for web UI (port 8501)
- **pandas + openpyxl** for CSV/Excel processing
- **MySQL** (utf8mb4 collation) as the database
- **Docker / Docker Compose** for containerization
- **AWS ECS + ECR** for deployment via GitHub Actions (OIDC auth)

## Running the Application

```bash
# Build Docker image
docker compose build

# Run Streamlit web UI (port 8501)
docker compose up web

# Run CLI CSV processing
docker compose run app --csv /data/in/file.csv --out-dir /data/out --verbose

# Run locally without Docker
streamlit run app_streamlit.py
python csv_conciliation_loader.py --csv <path> --out-dir <path> [--verbose] [--dry-run]

# Rebuild thirds_summary_conciliation table
python rebuild_thirds_summary_conciliation.py          # TRUNCATE + INSERT
python rebuild_thirds_summary_conciliation.py --no-truncate  # upsert only
```

## Environment Variables

Copy `.env.example` to `.env`. Required: `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DB`, `MYSQL_USER`, `MYSQL_PASSWORD`. Optional: `APP_USERS` (format: `email:password,email:password`), `ACTA_GENERATION_USER`.

## Architecture

Three entry points, one shared database:

1. **`app_streamlit.py`** (~2100 lines) — Streamlit web app with 4 pages:
   - *Cargar CSV*: upload → dry-run validation → confirm → insert → generate actas
   - *Resumen*: summary by NIT with state tracking and acta downloads
   - *Detalle NIT*: per-NIT detail with state management and PDF upload
   - *Reportes*: batch CSV report generation (aggregated or detailed)

2. **`csv_conciliation_loader.py`** (~1200 lines) — CLI engine for CSV processing:
   - Reads CSV with auto-encoding detection (utf-8-sig, latin-1)
   - 5-level validation: headers → UUIDs → numeric fields → sum equality (`IPS + EPS + RATIFICADO == VALOR_GLOSA`) → required fields
   - Bulk insert via temporary table (`tmp_conciliation_in`) with 5000-row chunks
   - Generates Excel actas from `XXXXXXX_GUIA.xlsx` template per NIT
   - Exit codes: 0=success, 1=validation errors, 2=read error, 3=MySQL error

3. **`rebuild_thirds_summary_conciliation.py`** (~115 lines) — Aggregation maintenance utility

## Key Database Tables

- `conciliation_results` — main reconciliation entries (write target)
- `auditory_final_reports` / `invoice_audits` — source audit data (read)
- `thirds_summary` / `thirds_summary_conciliation` — pre-aggregated summaries
- `nit_states` — NIT status tracking (latest state via MAX(created_at))
- `conciliation_acta_files` / `conciliation_acta_files_pdf` — generated acta metadata
- `third_departments` — geographic location lookup
- `reconciliation_groups` — grouping mechanism

## Data Flow

CSV files go into `data/in/`. All outputs go to `data/out/`: `actas_conciliacion/` (Excel), `actas_pdf/` (PDF), `reportes_masivos/` (CSV), `log_errores.csv`, `carga_exitosa_join.csv`, `runtime.log`.

## Code Conventions

- No test suite exists; validation is done through CSV dry-run mode
- No ORM — all database interaction is raw parameterized SQL via pymysql
- Colombian locale: timezone `America/Bogota`, currency formatting with `$` prefix and `,` thousands separator
- Streamlit caching: `@st.cache_data(ttl=300)` on summary queries
- The `app` docker-compose service mounts source files as read-only for hot-reload; `web` mounts the full directory

## CI/CD

GitHub Actions (`.github/workflows/deploy-ecr.yml`): builds Docker image, pushes to ECR (`conciliacion-coosalud`), forces ECS redeployment. Triggered on push/PR to main.

## Idioma

- Siempre responder en español
