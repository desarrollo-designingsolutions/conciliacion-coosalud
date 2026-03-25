---
description: "Guía para agregar features nuevas a la app de conciliación: nueva página Streamlit, nuevo tipo de reporte, nuevo módulo, modificar flujo CSV. Incluye checklist de patrón de código, testing y deploy."
---

# Skill: Feature Conciliación

## Propósito
Guiar paso a paso la implementación de nuevas funcionalidades en la app de conciliación, siguiendo los patrones existentes del código.

## Antes de Empezar

1. Leer el `CLAUDE.md` del proyecto para entender la arquitectura
2. Identificar qué tipo de feature es (ver secciones abajo)
3. Confirmar con el usuario el alcance antes de escribir código

## Tipo 1: Nueva Página en Streamlit

### Patrón
Cada página es una función `render_xxx_page()` que se llama desde el routing principal en `app_streamlit.py`.

### Paso a paso

1. **Decidir ubicación**: Si la página es compleja (>200 líneas), crear archivo nuevo. Si es simple, agregar en `app_streamlit.py`.

2. **Crear la función de renderizado**:
```python
def render_nueva_page():
    """Página de [descripción]."""
    st.header("Título de la Página")

    # Conexión BD (patrón existente)
    cfg, missing = get_db_config()  # o usar get_mysql_conn() según archivo
    if missing:
        st.error(f"Faltan variables: {', '.join(missing)}")
        return

    conn = pymysql.connect(**cfg, cursorclass=pymysql.cursors.DictCursor)
    try:
        # ... lógica de la página
        pass
    finally:
        conn.close()
```

3. **Agregar al menú lateral** en `app_streamlit.py`:
```python
# Buscar la sección del st.sidebar.radio o selectbox
# Agregar la nueva opción
pagina = st.sidebar.radio("Menú", [
    "Cargar CSV",
    "Resumen",
    "Detalle NIT",
    "Reportes",
    "Dashboard BI",
    "Mesas Conciliación",
    "Nueva Página",  # ← agregar aquí
])
```

4. **Agregar el routing**:
```python
elif pagina == "Nueva Página":
    render_nueva_page()  # Si está en archivo aparte: from nuevo_modulo import render_nueva_page
```

5. **Si el archivo es nuevo**, agregar el import al inicio de `app_streamlit.py`.

### Patrón de queries
```python
# Queries como constantes al inicio del archivo
QUERY_NUEVA = """
SELECT ...
FROM ...
WHERE ...
"""

# Ejecución con parámetros (NUNCA string formatting)
with conn.cursor() as cur:
    cur.execute(QUERY_NUEVA, (param1, param2))
    rows = cur.fetchall()
df = pd.DataFrame(rows)
```

### Patrón de descarga
```python
# Excel
buffer = io.BytesIO()
df.to_excel(buffer, index=False, engine='openpyxl')
st.download_button("Descargar Excel", buffer.getvalue(),
                   file_name="reporte.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# CSV
csv_data = df.to_csv(index=False)
st.download_button("Descargar CSV", csv_data, file_name="reporte.csv", mime="text/csv")
```

## Tipo 2: Nuevo Tipo de Reporte

### Paso a paso

1. Escribir la query SQL y probarla directamente en BD
2. Agregar la query como constante en el archivo correspondiente
3. Crear función de renderizado con filtros Streamlit
4. Agregar botón de descarga Excel/CSV
5. Si es un reporte masivo (batch), integrarlo en la sección de Reportes existente

## Tipo 3: Modificar Flujo CSV

### ⚠️ CUIDADO — Esto afecta producción directamente

1. Identificar qué nivel de validación afecta (hay 5 niveles en `csv_conciliation_loader.py`)
2. Hacer el cambio
3. **SIEMPRE probar con `--dry-run`** antes de cualquier ejecución real:
```bash
python csv_conciliation_loader.py --csv test.csv --out-dir /tmp/test --verbose --dry-run
```
4. Verificar que CSVs existentes siguen pasando validación

## Tipo 4: Nueva Migración de BD

1. Crear archivo en `migrations/` con formato: `NNN_descripcion.sql`
2. La migración debe ser idempotente (usar `IF NOT EXISTS`, `IF EXISTS`)
3. Documentar en el commit qué hace y por qué

## Deploy

⚠️ **Todo push a `main` dispara deploy automático a producción.**

### Pre-deploy checklist
- [ ] Probé localmente con `streamlit run app_streamlit.py`
- [ ] Si modifiqué CSV loader, probé con `--dry-run`
- [ ] Si agregué dependencia, la agregué a `requirements.txt`
- [ ] Si agregué variable de entorno, la documenté en `.env.example`
- [ ] No hay credenciales hardcodeadas
- [ ] Los cambios son backwards-compatible (no rompen datos existentes)

### Proceso
```bash
git add <archivos>
git commit -m "feat: descripción de la feature"
git push origin main
# → GitHub Actions: build → ECR → ECS redeploy (automático)
```

### Rollback (si algo sale mal)
```bash
# Ver deployments recientes
aws ecs describe-services --cluster conciliacion-cluster --services conciliacion-svc

# Revertir al commit anterior
git revert HEAD
git push origin main
# → Nuevo deploy con el código anterior
```

## REFLECTION (obligatorio antes de hacer commit)

### Checklist
- [ ] La feature sigue los patrones existentes del código
- [ ] Queries usan parámetros (%s), NO string formatting
- [ ] Conexiones BD se cierran correctamente (try/finally o context manager)
- [ ] Textos de UI en español
- [ ] Sin credenciales hardcodeadas
- [ ] Si es cambio de BD: migración creada y es idempotente
- [ ] Probado localmente antes de push
- [ ] El CLAUDE.md se actualizó si la feature cambia la arquitectura
