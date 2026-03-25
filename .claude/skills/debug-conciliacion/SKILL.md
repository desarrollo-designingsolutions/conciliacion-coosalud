---
description: "Diagnostica problemas en el sistema de conciliación: CSVs que no cargan, actas faltantes, discrepancias de valores, estados inconsistentes. Incluye queries de diagnóstico y flujo de investigación."
---

# Skill: Debug Conciliación

## Propósito
Investigar y resolver problemas operativos del sistema de conciliación de cuentas médicas de Coosalud.

## Casos Comunes

### 1. CSV no carga / tiene errores de validación
**Síntomas**: El usuario reporta que un CSV falla al cargar.

**Investigación**:
```bash
# Probar en modo dry-run para ver errores sin escribir en BD
python csv_conciliation_loader.py --csv <ruta_csv> --out-dir /tmp/debug --verbose --dry-run
```

**Causas frecuentes**:
- Headers incorrectos → verificar contra `EXPECTED_HEADERS` en `csv_conciliation_loader.py`
- UUIDs de `auditory_final_report_id` no existen en BD
- Suma `IPS + EPS + RATIFICADO != VALOR_GLOSA` (tolerancia: $1 COP)
- Encoding incorrecto → el sistema prueba utf-8-sig y latin-1 automáticamente
- Separador incorrecto → debe ser `,` (coma)

### 2. Acta no se generó para un NIT
**Investigación**:
```sql
-- Verificar si hay resultados de conciliación para el NIT
SELECT COUNT(*) AS total_registros,
       COUNT(DISTINCT cr.invoice_audit_id) AS facturas
FROM conciliation_results cr
INNER JOIN invoice_audits ia ON ia.id = cr.invoice_audit_id
INNER JOIN thirds t ON t.id = ia.third_id
WHERE t.nit = '<NIT>';

-- Verificar si ya existe el acta
SELECT * FROM conciliation_acta_files
WHERE nit = '<NIT>'
ORDER BY created_at DESC LIMIT 5;

-- Verificar estado del NIT
SELECT * FROM nit_states
WHERE nit = '<NIT>'
ORDER BY created_at DESC LIMIT 1;
```

### 3. Valores no cuadran entre acta y BD
**Investigación**:
```sql
-- Comparar totales de conciliación para un NIT
SELECT
    t.nit,
    t.name AS razon_social,
    COUNT(cr.id) AS total_registros,
    SUM(cr.accepted_value_ips) AS total_ips,
    SUM(cr.accepted_value_eps) AS total_eps,
    SUM(cr.eps_ratified_value) AS total_ratificado
FROM conciliation_results cr
INNER JOIN auditory_final_reports afr ON afr.id = cr.auditory_final_report_id
INNER JOIN invoice_audits ia ON ia.id = cr.invoice_audit_id
INNER JOIN thirds t ON t.id = ia.third_id
WHERE t.nit = '<NIT>'
GROUP BY t.nit, t.name;

-- Verificar integridad: valor_glosa debe = ips + eps + ratificado
SELECT
    afr.numero_factura,
    afr.valor_glosa,
    cr.accepted_value_ips,
    cr.accepted_value_eps,
    cr.eps_ratified_value,
    (cr.accepted_value_ips + cr.accepted_value_eps + cr.eps_ratified_value) AS suma,
    afr.valor_glosa - (cr.accepted_value_ips + cr.accepted_value_eps + cr.eps_ratified_value) AS diferencia
FROM conciliation_results cr
INNER JOIN auditory_final_reports afr ON afr.id = cr.auditory_final_report_id
INNER JOIN invoice_audits ia ON ia.id = cr.invoice_audit_id
INNER JOIN thirds t ON t.id = ia.third_id
WHERE t.nit = '<NIT>'
HAVING ABS(diferencia) > 1
LIMIT 20;
```

### 4. Estado de un NIT no se actualizó
**Investigación**:
```sql
-- Historial de estados del NIT
SELECT * FROM nit_states
WHERE nit = '<NIT>'
ORDER BY created_at DESC;

-- Verificar thirds_summary_conciliation
SELECT * FROM thirds_summary_conciliation
WHERE nit = '<NIT>';
```

**Solución**: Si la tabla resumen está desactualizada:
```bash
python rebuild_thirds_summary_conciliation.py
```

### 5. Mesas de conciliación con datos incorrectos
**Investigación**:
```sql
-- Ver datos de la mesa para un NIT
SELECT * FROM mesas_conciliacion
WHERE nit = '<NIT>'
ORDER BY fecha DESC;

-- Verificar campos split (antiguo vs nuevo formato)
-- Tablas antiguas usan: valor_aceptado_eps, valor_aceptado_ips
-- Tablas nuevas usan: eps_acepta, ips_acepta
-- AMBOS pueden tener valores → hay que sumarlos
```

### 6. Dashboard BI no muestra datos actualizados
**Causa**: Los datos del dashboard vienen de Snowflake, que se sincroniza manualmente.

**Verificación**:
```sql
-- En Snowflake: verificar última sincronización
SELECT MAX(UPDATED_AT) AS ultima_actualizacion
FROM DWH_DEV.UTAM.SYNC_CONCILIATION_RESULTS;
```

**Solución**: Ejecutar sync desde `cliente-coosalud`:
```bash
cd ../cliente-coosalud/scripts/aws
export $(grep -v '^#' .env | xargs)
python3 delta_sync_mysql_snowflake.py
```

## REFLECTION (obligatorio antes de entregar diagnóstico)

### Checklist
- [ ] Se identificó la causa raíz, no solo el síntoma
- [ ] Las queries de diagnóstico se ejecutaron y sus resultados se presentaron
- [ ] Si hubo corrección, se verificó que el problema se resolvió
- [ ] Si el problema es recurrente, se propuso cómo prevenirlo (validación adicional, check automático, etc.)
- [ ] No se modificaron datos de producción sin confirmación explícita del usuario
