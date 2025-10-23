# CSV Conciliation Loader (Docker)
Ver README completo en el mensaje del chat.

## Recalcular thirds_summary_conciliation

- Ejecuta `python rebuild_thirds_summary_conciliation.py` para reconstruir completamente la tabla `thirds_summary_conciliation` con los totales actuales de `conciliation_results`.
- Por defecto hace `TRUNCATE + INSERT` completo. Si prefieres solo upsert sin truncar: `python rebuild_thirds_summary_conciliation.py --no-truncate`.
- Requiere variables `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DB`, `MYSQL_USER`, `MYSQL_PASSWORD`.
# Test deploy
