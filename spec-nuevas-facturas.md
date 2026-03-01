# Especificación: Cargar Nuevas Facturas

Antes de escribir código, analiza esta especificación y proponme un plan de implementación paso a paso. No escribas código aún.

## Descripción General

Crear un nuevo proceso llamado **"Cargar nuevas facturas"**, agregarlo en un nuevo menú. La función recibe un archivo CSV y realiza validaciones, inserciones en base de datos y genera un archivo Excel como salida.

---

## 1. Formato del Archivo CSV

El archivo CSV tiene la siguiente estructura de columnas:

```
nit;razon_social;factura;modalidad;fecha_prestacion;fecha_factura;fecha_radicacion;valor_factura;glosa;radicacion
```

- **Formato:** CSV separado por `;` (punto y coma)
- **Todos los campos son obligatorios**

---

## 2. Validaciones del Archivo

### 2.1 Campo `nit`
- Solo debe haber un **único número de NIT** en la columna `nit` del archivo.
- Debe ser **numérico**.
- El NIT se consulta en la tabla `thirds`:

```sql
SELECT *
FROM thirds
WHERE id = '{nit}';
```

### 2.2 Campo `factura`
- **No puede haber facturas duplicadas** en el archivo.
- Las facturas **no deben existir en la base de datos** (son nuevas). Validación:

```sql
SELECT invoice_number
FROM invoice_audits
WHERE third_id = '{nit}'
AND invoice_number IN ('factura1', 'factura2', 'factura...');
```

### 2.3 Campos de fecha: `fecha_prestacion`, `fecha_factura`, `fecha_radicacion`
- Deben estar en formato: **yyyy-mm-dd**

### 2.4 Campo `valor_factura`
- Campo **numérico**.
- Puede llevar decimal, y de ser así el separador decimal en el CSV es `,` (coma).
- En SQL se debe formatear para ser cargado con `.` como separador decimal.

### 2.5 Campo `glosa`
- Campo **numérico**.
- Debe ser **mayor que cero**.
- Debe ser **menor o igual que `valor_factura`**.
- Puede llevar decimal, y de ser así el separador decimal en el CSV es `,` (coma).
- En SQL se debe formatear para ser cargado con `.` como separador decimal.

### 2.6 Campo `radicacion`
- Solo acepta los siguientes valores: `radicación_nueva`, `antigua`

---

## 3. Log de Errores

Se deben validar **todos los campos** y por cada error se debe llenar el siguiente log de errores para ser descargado por el usuario:

```
consecutivo, fila, columna, dato, error, fecha
```

---

## 4. Inserciones en Base de Datos

Si el archivo pasa todas las validaciones, se realizan los siguientes inserts en orden:

### 4.1 Insert en `invoice_audits`

Ejemplo de referencia para crear el script:

```sql
INSERT INTO invoice_audits
SELECT
    UUID() AS id,
    '9e5aec58-a962-4670-8188-b41c6d0149a3' AS company_id,
    nit AS third_id,
    NULL AS filling_invoice_id,
    factura AS invoice_number,
    valor_factura AS total_value,
    'inclusiones_2026' AS origin,
    fecha_factura AS expedition_date,
    fecha_factura AS date_entry,
    fecha_factura AS date_departure,
    modalidad AS modality,
    'SUBSIDIADO' AS regimen,
    'POS' AS coverage,
    NULL AS contract_number,
    NOW() AS created_at,
    NOW() AS updated_at,
    NULL AS deleted_at,
    'devolución aplistaf' AS status_radication,
    radicacion AS radication
```

### 4.2 Insert en `auditory_final_reports`

Ejemplo de referencia (usa tabla temporal con la importación del archivo, pero si hay una mejor forma, implementarla):

```sql
INSERT INTO auditory_final_reports
SELECT
    UUID() AS id,
    ia.id AS factura_id,
    '31b734de-e29d-4e34-ad7c-83809ac1d32a' AS servicio_id,
    ia.origin AS origin,
    t.id AS nit,
    t.`name` AS razon_social,
    ia.invoice_number AS numero_factura,
    ia.expedition_date AS fecha_inicio,
    ia.expedition_date AS fecha_fin,
    ia.modality AS modalidad,
    ia.regimen AS regimen,
    ia.coverage AS cobertura,
    ia.contract_number AS contrato,
    'CC' AS tipo_documento,
    '000000' AS numero_documento,
    'NA' AS primer_nombre,
    'NA' AS segundo_nombre,
    'NA' AS primer_apellido,
    'NA' AS segundo_apellido,
    'M' AS genero,
    '000000' AS codigo_servicio,
    'SERVICIOS FACTURADOS' AS descripcion_servicio,
    1 AS cantidad_servicio,
    ia.total_value AS valor_unitario_servicio,
    ia.total_value AS valor_total_servicio,
    '223' AS codigos_glosa,
    'la factura presenta diferencias con los valores que fueron pactados' AS observaciones_glosas,
    nf.glosa AS valor_glosa,
    (ia.total_value - nf.glosa) AS valor_aprobado,
    NOW() AS created_at,
    NOW() AS updated_at,
    NULL AS deleted_at
FROM invoice_audits ia
INNER JOIN thirds t
    ON t.id = ia.third_id
INNER JOIN {nit}_nuevas_{fecha} nf
    ON nf.factura COLLATE utf8mb4_unicode_ci = ia.invoice_number COLLATE utf8mb4_unicode_ci
    AND nf.nit COLLATE utf8mb4_unicode_ci = t.id COLLATE utf8mb4_unicode_ci
WHERE t.id = '{nit}'
AND ia.created_at > '{fecha_proceso}'
AND ia.origin = '{nit}_nuevas_{fecha}';
```

### 4.3 Insert en `estados_auditory_final_reports`

```sql
INSERT INTO estados_auditory_final_reports
SELECT
    afr.id AS ID,
    afr.factura_id AS FACTURA_ID,
    'contabilizada' AS ESTADO
FROM auditory_final_reports afr
WHERE afr.nit = '{nit}'
AND afr.created_at > '{fecha_proceso}'
AND afr.origin = '{nit}_nuevas_{fecha}';
```

---

## 5. Generación de Archivo Excel (Sabana)

Después de las inserciones, generar un archivo Excel para el usuario:

- **Nombre del archivo:** `{nit}_sabana_{fecha_hora}` (fecha y hora para hacerlo único)
- **Encabezados en MAYÚSCULA**

Query para generar el archivo:

```sql
SELECT
    aud.id,
    aud.factura_id,
    aud.servicio_id,
    aud.origin,
    aud.nit,
    aud.razon_social,
    aud.numero_factura,
    aud.fecha_inicio,
    aud.fecha_fin,
    aud.modalidad,
    aud.regimen,
    aud.cobertura,
    aud.contrato,
    aud.tipo_documento,
    aud.numero_documento,
    aud.primer_nombre,
    aud.segundo_nombre,
    aud.primer_apellido,
    aud.segundo_apellido,
    aud.genero,
    aud.codigo_servicio,
    aud.descripcion_servicio,
    aud.cantidad_servicio,
    aud.valor_unitario_servicio,
    aud.valor_total_servicio,
    aud.codigos_glosa,
    REGEXP_REPLACE(aud.observaciones_glosas, '[\\r\\n\\t]+', ' ') AS observaciones_glosas,
    aud.valor_glosa,
    aud.valor_aprobado,
    NULL AS estado_respuesta,
    NULL AS numero_de_autorizacion,
    NULL AS respuesta_de_rips,
    NULL AS valor_aceptado_ips,
    NULL AS valor_aceptado_eps,
    NULL AS valor_ratificado_eps,
    NULL AS observaciones
FROM auditory_final_reports aud
INNER JOIN estados_auditory_final_reports eafr
    ON eafr.ID = aud.id
INNER JOIN invoice_audits ia
    ON ia.id = aud.factura_id
INNER JOIN thirds t
    ON t.id = ia.third_id
LEFT JOIN conciliation_results ci
    ON ci.auditory_final_report_id = aud.id
WHERE t.id = '{nit}'
AND eafr.ESTADO = 'contabilizada'
AND aud.valor_glosa > 0
AND ci.id IS NULL;
```

---

## 6. Historial de Importaciones

- Por cada proceso de importación se debe generar un **registro en una tabla** para que el usuario pueda descargar la sabana en el futuro.
- En la **interfaz gráfica**, mostrar una tabla con filtros que muestre:
  - NIT
  - Razón social
  - Valor factura
  - Glosa
  - Fecha de creación
  - Usuario que crea

---

## 7. Consideraciones

- Revisa cómo están hechos los otros procesos del sistema y sigue la misma arquitectura y patrones.
- Propón primero el plan de implementación dividido en pasos.
- Espera mi aprobación antes de implementar cada paso.
