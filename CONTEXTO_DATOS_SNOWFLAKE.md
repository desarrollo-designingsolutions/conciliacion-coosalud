# Contexto de Datos - Snowflake para Dashboard BI

Este documento describe la fuente de datos centralizada en Snowflake que alimenta el dashboard BI de conciliacion de cuentas medicas de Coosalud.

---

## 1. Conexion a Snowflake

```
Account:    gra06523.us-east-1   (IMPORTANTE: incluir ".us-east-1")
User:       CT_UTAM
Role:       CT_088_UTAM
Warehouse:  COO_CUENTAS_MEDICAS_DWH
Database:   DWH_DEV
Schema:     UTAM
Password:   JWT token (variable de entorno SNOWFLAKE_PASSWORD)
```

### Ejemplo de conexion Python

```python
import snowflake.connector

conn = snowflake.connector.connect(
    user="CT_UTAM",
    password=os.environ["SNOWFLAKE_PASSWORD"],
    account="gra06523.us-east-1",
    role="CT_088_UTAM",
    warehouse="COO_CUENTAS_MEDICAS_DWH",
    database="DWH_DEV",
    schema="UTAM",
    session_parameters={"CLIENT_SESSION_KEEP_ALIVE": True},
)
```

---

## 2. Origen de los datos

Los datos se originan en **MySQL (RDS)** y se sincronizan a Snowflake mediante un script de full-sync:

- **Script**: `/Users/andres/code/coosalud/aws/sync_mysql_snowflake.py`
- **Metodo**: Keyset pagination (50K chunks) + staging table + DROP/RENAME
- **Frecuencia**: Manual (se ejecuta cuando hay cambios significativos)
- **Maneja**: Eliminaciones fisicas (staging reemplaza tabla completa)

La aplicacion actual (`csv_conciliation_docker`) escribe en MySQL. Los datos fluyen asi:

```
App Streamlit (este proyecto)
    |
    v
MySQL (RDS) ----sync----> Snowflake (SYNC_* tables)
    |                          |
    v                          v
Operacion diaria          Dashboard BI (nuevo)
```

---

## 3. Tablas disponibles en Snowflake

Todas las tablas tienen prefijo `SYNC_` y son propiedad del role `CT_088_UTAM`.

### SYNC_THIRDS (Prestadores/Proveedores IPS)
**Filas**: ~1,794 | **MySQL source**: `thirds`

| Columna    | Tipo             | Descripcion                           |
|------------|------------------|---------------------------------------|
| ID         | VARCHAR(36)      | PK - UUID o NIT (es la FK de otras tablas) |
| COMPANY_ID | VARCHAR(36)      | Empresa (Coosalud)                    |
| NAME       | VARCHAR(255)     | Razon social del prestador            |
| NIT        | VARCHAR(255)     | Numero de Identificacion Tributaria   |
| CREATED_AT | TIMESTAMP_NTZ(9) | Fecha creacion                        |
| UPDATED_AT | TIMESTAMP_NTZ(9) | Fecha actualizacion                   |
| DELETED_AT | TIMESTAMP_NTZ(9) | Soft delete                           |

**IMPORTANTE sobre ID vs NIT**:
- `thirds.ID` es la PK y es la FK que usan las demas tablas
- `thirds.NIT` es el identificador fiscal visible
- En 264 registros, `ID != NIT` (la mayoria son iguales)
- Siempre hacer JOIN por `ID`, nunca por `NIT`

---

### SYNC_INVOICE_AUDITS (Facturas)
**Filas**: ~2,814,048 | **MySQL source**: `invoice_audits`

| Columna           | Tipo             | Descripcion                        |
|-------------------|------------------|------------------------------------|
| ID                | VARCHAR(36)      | PK - UUID o numerico               |
| COMPANY_ID        | VARCHAR(36)      | Empresa                            |
| THIRD_ID          | VARCHAR(36)      | FK -> SYNC_THIRDS.ID               |
| FILING_INVOICE_ID | VARCHAR(36)      | ID de radicacion                   |
| INVOICE_NUMBER    | VARCHAR(255)     | Numero de factura (ej: FESM20123)  |
| TOTAL_VALUE       | NUMBER(15,2)     | Valor total facturado              |
| ORIGIN            | VARCHAR(255)     | Origen/lote de carga               |
| EXPEDITION_DATE   | TIMESTAMP_NTZ(9) | Fecha de expedicion                |
| DATE_ENTRY        | DATE             | Fecha de ingreso                   |
| DATE_DEPARTURE    | DATE             | Fecha de salida                    |
| MODALITY          | VARCHAR(255)     | Modalidad (Evento, Capita, PGP...) |
| REGIMEN           | VARCHAR(255)     | Regimen (SUBSIDIADO, CONTRIBUTIVO) |
| COVERAGE          | VARCHAR(255)     | Cobertura (POS, NO POS)            |
| CONTRACT_NUMBER   | VARCHAR(255)     | Numero de contrato                 |
| STATUS_RADICATION | VARCHAR(50)      | Estado de radicacion               |
| RADICATION        | VARCHAR(255)     | Tipo de radicacion                 |
| CREATED_AT        | TIMESTAMP_NTZ(9) | Fecha creacion                     |
| UPDATED_AT        | TIMESTAMP_NTZ(9) | Fecha actualizacion                |
| DELETED_AT        | TIMESTAMP_NTZ(9) | Soft delete                        |

---

### SYNC_AUDITORY_FINAL_REPORTS (Servicios auditados - detalle)
**Filas**: ~15,130,740 | **MySQL source**: `auditory_final_reports`

Esta es la tabla mas grande. Contiene un registro por cada servicio de cada factura.

| Columna                 | Tipo             | Descripcion                         |
|-------------------------|------------------|-------------------------------------|
| ID                      | VARCHAR(36)      | PK - UUID                           |
| FACTURA_ID              | VARCHAR(36)      | FK -> SYNC_INVOICE_AUDITS.ID        |
| SERVICIO_ID             | VARCHAR(36)      | ID del servicio                     |
| ORIGIN                  | VARCHAR(255)     | Origen/lote de carga                |
| NIT                     | VARCHAR(36)      | FK -> SYNC_THIRDS.ID (NO es el NIT fiscal!) |
| RAZON_SOCIAL            | VARCHAR(255)     | Nombre del prestador                |
| NUMERO_FACTURA          | VARCHAR(255)     | Numero de factura (desnormalizado)  |
| FECHA_INICIO            | DATE             | Fecha inicio de prestacion          |
| FECHA_FIN               | DATE             | Fecha fin de prestacion             |
| MODALIDAD               | VARCHAR(255)     | Modalidad de contratacion           |
| REGIMEN                 | VARCHAR(255)     | Regimen del afiliado                |
| COBERTURA               | VARCHAR(255)     | Cobertura                           |
| CONTRATO                | VARCHAR(255)     | Numero de contrato                  |
| TIPO_DOCUMENTO          | VARCHAR(255)     | Tipo doc del paciente               |
| NUMERO_DOCUMENTO        | VARCHAR(255)     | Numero doc del paciente             |
| PRIMER_NOMBRE           | VARCHAR(255)     | Nombre del paciente                 |
| SEGUNDO_NOMBRE          | VARCHAR(255)     | Segundo nombre                      |
| PRIMER_APELLIDO         | VARCHAR(255)     | Primer apellido                     |
| SEGUNDO_APELLIDO        | VARCHAR(255)     | Segundo apellido                    |
| GENERO                  | VARCHAR(50)      | Genero del paciente                 |
| CODIGO_SERVICIO         | VARCHAR(255)     | Codigo CUPS/medicamento             |
| DESCRIPCION_SERVICIO    | VARCHAR(255)     | Descripcion del servicio            |
| CANTIDAD_SERVICIO       | NUMBER(10,0)     | Cantidad                            |
| VALOR_UNITARIO_SERVICIO | NUMBER(15,2)     | Valor unitario                      |
| VALOR_TOTAL_SERVICIO    | NUMBER(15,2)     | Valor total del servicio            |
| CODIGOS_GLOSA           | VARCHAR(16MB)    | Codigos de glosa aplicados          |
| OBSERVACIONES_GLOSAS    | VARCHAR(16MB)    | Observaciones de la glosa           |
| VALOR_GLOSA             | NUMBER(15,2)     | Valor glosado                       |
| VALOR_APROBADO          | NUMBER(15,2)     | Valor aprobado en auditoria         |
| CREATED_AT              | TIMESTAMP_NTZ(9) | Fecha creacion                      |
| UPDATED_AT              | TIMESTAMP_NTZ(9) | Fecha actualizacion                 |
| DELETED_AT              | TIMESTAMP_NTZ(9) | Soft delete                         |

**IMPORTANTE**: La columna `NIT` apunta a `SYNC_THIRDS.ID`, NO al NIT fiscal. Es un nombre confuso heredado del diseho original.

---

### SYNC_ESTADOS_AUDITORY_FINAL_REPORTS (Estado de cada servicio)
**Filas**: ~15,270,140 | **MySQL source**: `estados_auditory_final_reports`

| Columna    | Tipo         | Descripcion                                |
|------------|--------------|--------------------------------------------|
| ID         | VARCHAR(36)  | PK y FK -> SYNC_AUDITORY_FINAL_REPORTS.ID  |
| FACTURA_ID | VARCHAR(255) | FK -> SYNC_INVOICE_AUDITS.ID               |
| ESTADO     | VARCHAR(255) | Estado del servicio                        |

**Estados posibles** (ya unificados, solo existen estos 3):

| Estado         | Registros   | Significado                           |
|----------------|-------------|---------------------------------------|
| contabilizada  | 12,807,863  | Factura activa en proceso             |
| eliminar       | 2,293,116   | Marcada para eliminacion (duplicada)  |
| devolucion     | 169,152     | Devuelta al prestador                 |

Nota: Antes existian 5 estados (`DEVOLUCION 2025` y `eliminada`), fueron unificados el 2026-03-06.

---

### SYNC_CONCILIATION_RESULTS (Resultados de conciliacion)
**Filas**: ~6,917,707 | **MySQL source**: `conciliation_results`

| Columna                  | Tipo             | Descripcion                              |
|--------------------------|------------------|------------------------------------------|
| ID                       | VARCHAR(36)      | PK - UUID                                |
| AUDITORY_FINAL_REPORT_ID | VARCHAR(36)      | FK -> SYNC_AUDITORY_FINAL_REPORTS.ID     |
| INVOICE_AUDIT_ID         | VARCHAR(36)      | FK -> SYNC_INVOICE_AUDITS.ID             |
| RECONCILIATION_GROUP_ID  | VARCHAR(36)      | Grupo de conciliacion (acta)             |
| RESPONSE_STATUS          | VARCHAR(255)     | Estado de respuesta                      |
| AUTORIZATION_NUMBER      | VARCHAR(255)     | Numero de autorizacion                   |
| ACCEPTED_VALUE_IPS       | NUMBER(15,2)     | Valor aceptado por el prestador (IPS)    |
| ACCEPTED_VALUE_EPS       | NUMBER(15,2)     | Valor aceptado por la EPS (Coosalud)     |
| EPS_RATIFIED_VALUE       | NUMBER(15,2)     | Valor ratificado por la EPS              |
| CREATED_AT               | TIMESTAMP_NTZ(9) | Fecha de creacion del acta               |
| UPDATED_AT               | TIMESTAMP_NTZ(9) | Fecha de actualizacion                   |
| OBSERVATION              | VARCHAR(16MB)    | Observaciones de conciliacion            |

**Regla de integridad**: `ACCEPTED_VALUE_IPS + ACCEPTED_VALUE_EPS + EPS_RATIFIED_VALUE = VALOR_GLOSA` del servicio asociado.

---

## 4. Modelo de relaciones (JOINs)

```
SYNC_THIRDS (t)
  |
  +-- t.ID = ia.THIRD_ID
  |
SYNC_INVOICE_AUDITS (ia)
  |
  +-- ia.ID = aud.FACTURA_ID
  |
SYNC_AUDITORY_FINAL_REPORTS (aud)
  |
  +-- aud.ID = eafr.ID          --> SYNC_ESTADOS_AUDITORY_FINAL_REPORTS (eafr)
  |
  +-- aud.ID = ci.AUDITORY_FINAL_REPORT_ID  --> SYNC_CONCILIATION_RESULTS (ci)
  |                                               (tambien ci.INVOICE_AUDIT_ID = ia.ID)
```

### JOIN tipico completo

```sql
SELECT ...
FROM SYNC_AUDITORY_FINAL_REPORTS aud
INNER JOIN SYNC_ESTADOS_AUDITORY_FINAL_REPORTS eafr ON eafr.ID = aud.ID
INNER JOIN SYNC_INVOICE_AUDITS ia ON ia.ID = aud.FACTURA_ID
INNER JOIN SYNC_THIRDS t ON t.ID = ia.THIRD_ID
LEFT JOIN SYNC_CONCILIATION_RESULTS ci ON ci.AUDITORY_FINAL_REPORT_ID = aud.ID
```

---

## 5. Flujo de negocio

```
1. Prestador (IPS) factura servicios a Coosalud (EPS)
2. Coosalud audita las facturas -> auditory_final_reports (detalle por servicio)
3. Se asigna un estado:
   - contabilizada: factura valida, entra al proceso
   - eliminar: duplicada o invalida
   - devolucion: devuelta al prestador
4. Si hay glosa (valor_glosa > 0), entra a conciliacion:
   - EPS y IPS negocian
   - Resultado en conciliation_results:
     * accepted_value_ips: lo que acepta el prestador
     * accepted_value_eps: lo que acepta Coosalud
     * eps_ratified_value: lo que Coosalud ratifica (no negocia)
5. Se genera un acta de conciliacion (PDF/Excel)
```

### Clasificacion de servicios para reportes

Un servicio **contabilizado con glosa** puede estar en uno de estos estados:

| Condicion | Estado |
|---|---|
| Sin registro en conciliation_results | **Sin conciliar** |
| ratified > 0 AND ips=0 AND eps=0 | **Ratificada** |
| ratified > 0 AND (ips>0 OR eps>0) | **Conciliada y ratificada** |
| ratified = 0 AND (ips>0 OR eps>0) | **Conciliada** |
| ratified = 0 AND ips=0 AND eps=0 (pero tiene registro) | **En proceso** |
| valor_glosa = 0 | **Aprobada en auditoria** (sin glosa) |

---

## 6. Queries utiles para el Dashboard

### Reporte de conciliacion por prestador

```sql
WITH servicios AS (
    SELECT
        aud.NIT, aud.RAZON_SOCIAL, aud.FACTURA_ID,
        aud.ID AS SERVICIO_ID, aud.VALOR_GLOSA,
        MAX(CASE WHEN ci.ID IS NOT NULL THEN 1 ELSE 0 END) AS ES_CONCILIADO,
        SUM(COALESCE(ci.ACCEPTED_VALUE_EPS, 0)) AS ACCEPTED_VALUE_EPS,
        SUM(COALESCE(ci.ACCEPTED_VALUE_IPS, 0)) AS ACCEPTED_VALUE_IPS,
        SUM(COALESCE(ci.EPS_RATIFIED_VALUE, 0)) AS EPS_RATIFIED_VALUE
    FROM SYNC_AUDITORY_FINAL_REPORTS aud
    INNER JOIN SYNC_ESTADOS_AUDITORY_FINAL_REPORTS eafr ON eafr.ID = aud.ID
    INNER JOIN SYNC_INVOICE_AUDITS ia ON ia.ID = aud.FACTURA_ID
    INNER JOIN SYNC_THIRDS t ON t.ID = ia.THIRD_ID
    LEFT JOIN SYNC_CONCILIATION_RESULTS ci ON ci.AUDITORY_FINAL_REPORT_ID = aud.ID
    WHERE eafr.ESTADO = 'contabilizada' AND aud.VALOR_GLOSA > 0
    GROUP BY aud.NIT, aud.RAZON_SOCIAL, aud.FACTURA_ID, aud.ID, aud.VALOR_GLOSA
)
SELECT
    NIT, RAZON_SOCIAL,
    COUNT(DISTINCT FACTURA_ID) AS CANTIDAD_FACTURAS,
    COUNT(*) AS CANTIDAD_SERVICIOS,
    SUM(VALOR_GLOSA) AS VALOR_GLOSA,
    COUNT(DISTINCT CASE WHEN ES_CONCILIADO = 1 THEN FACTURA_ID END) AS CANT_FACTURAS_CONCILIADAS,
    SUM(ES_CONCILIADO) AS CANT_SERVICIOS_CONCILIADOS,
    SUM(CASE WHEN ES_CONCILIADO = 1 THEN VALOR_GLOSA ELSE 0 END) AS VALOR_GLOSA_CONCILIADA,
    SUM(ACCEPTED_VALUE_EPS) AS VALOR_ACEPTADO_EPS,
    SUM(ACCEPTED_VALUE_IPS) AS VALOR_ACEPTADO_IPS,
    SUM(EPS_RATIFIED_VALUE) AS VALOR_RATIFICADO,
    COUNT(DISTINCT CASE WHEN ES_CONCILIADO = 0 THEN FACTURA_ID END) AS CANT_FACTURAS_NO_CONCILIADAS,
    SUM(CASE WHEN ES_CONCILIADO = 0 THEN 1 ELSE 0 END) AS CANT_SERVICIOS_NO_CONCILIADOS,
    SUM(CASE WHEN ES_CONCILIADO = 0 THEN VALOR_GLOSA ELSE 0 END) AS VALOR_GLOSA_NO_CONCILIADA
FROM servicios
GROUP BY NIT, RAZON_SOCIAL
ORDER BY NIT
```

### Universo de facturas por estado

```sql
SELECT
    eafr.ESTADO,
    COUNT(DISTINCT aud.FACTURA_ID) AS cantidad_facturas,
    COUNT(aud.ID) AS cantidad_servicios,
    SUM(aud.VALOR_TOTAL_SERVICIO) AS valor_total,
    SUM(aud.VALOR_GLOSA) AS valor_glosa,
    SUM(aud.VALOR_APROBADO) AS valor_aprobado
FROM SYNC_AUDITORY_FINAL_REPORTS aud
INNER JOIN SYNC_ESTADOS_AUDITORY_FINAL_REPORTS eafr ON eafr.ID = aud.ID
GROUP BY eafr.ESTADO
ORDER BY eafr.ESTADO
```

### Facturas con actas generadas en un periodo

```sql
SELECT
    t.NIT, t.NAME AS RAZON_SOCIAL,
    COUNT(DISTINCT cr.ID) AS total_actas,
    COUNT(DISTINCT cr.INVOICE_AUDIT_ID) AS facturas_con_acta,
    MIN(cr.CREATED_AT) AS primera_acta,
    MAX(cr.CREATED_AT) AS ultima_acta
FROM SYNC_CONCILIATION_RESULTS cr
INNER JOIN SYNC_INVOICE_AUDITS ia ON ia.ID = cr.INVOICE_AUDIT_ID
INNER JOIN SYNC_THIRDS t ON t.ID = ia.THIRD_ID
WHERE cr.CREATED_AT >= '2026-01-01'
GROUP BY t.NIT, t.NAME
ORDER BY t.NIT
```

### Detalle por modalidad (con homologacion)

```sql
SELECT
    CASE
        WHEN UPPER(aud.MODALIDAD) LIKE '%PPE%CAPITA%' THEN 'PPE Capita'
        WHEN UPPER(aud.MODALIDAD) LIKE '%PPE%EVENTO%' THEN 'PPE Evento'
        WHEN UPPER(aud.MODALIDAD) LIKE '%PGE%' THEN 'PGE'
        WHEN UPPER(aud.MODALIDAD) LIKE '%PGP%' OR UPPER(aud.MODALIDAD) LIKE '%PAGO GLOBAL PROSPECTIVO%' THEN 'PGP'
        WHEN UPPER(aud.MODALIDAD) LIKE '%PRESUPUESTO GLOBAL%' THEN 'Presupuesto Global'
        WHEN UPPER(aud.MODALIDAD) LIKE '%MIA%' OR UPPER(aud.MODALIDAD) LIKE '%MAIS%' THEN 'MIA'
        WHEN UPPER(aud.MODALIDAD) LIKE '%PAQUETE%' THEN 'Paquete Integral'
        WHEN UPPER(aud.MODALIDAD) LIKE '%CAPITA%' THEN 'Capita'
        WHEN UPPER(aud.MODALIDAD) LIKE '%EVENT%' OR UPPER(aud.MODALIDAD) LIKE '%COVID%' OR UPPER(aud.MODALIDAD) LIKE '%NO POS%' THEN 'Evento'
        WHEN UPPER(aud.MODALIDAD) LIKE '%PAGO POR SERVICIO%' THEN 'Pago por Servicio'
        WHEN UPPER(aud.MODALIDAD) LIKE '%EPISODIO%' THEN 'Pago por episodio'
        ELSE 'Sin identificar'
    END AS MODALIDAD_HOMOLOGADA,
    COUNT(DISTINCT aud.FACTURA_ID) AS cantidad_facturas,
    SUM(aud.VALOR_TOTAL_SERVICIO) AS valor_total,
    SUM(aud.VALOR_GLOSA) AS valor_glosa,
    SUM(aud.VALOR_APROBADO) AS valor_aprobado
FROM SYNC_AUDITORY_FINAL_REPORTS aud
INNER JOIN SYNC_ESTADOS_AUDITORY_FINAL_REPORTS eafr ON eafr.ID = aud.ID
WHERE eafr.ESTADO = 'contabilizada' AND aud.VALOR_GLOSA > 0
GROUP BY MODALIDAD_HOMOLOGADA
ORDER BY valor_glosa DESC
```

---

## 7. Consideraciones importantes para el dashboard

### Performance
- Las tablas SYNC_AUDITORY y SYNC_ESTADOS tienen 15M+ registros cada una
- Siempre filtrar por `eafr.ESTADO = 'contabilizada'` cuando sea posible
- Usar `COUNT(DISTINCT factura_id)` para contar facturas (no COUNT(*))
- Para queries pesados, considerar materializar vistas o tablas de resumen

### Evitar duplicacion por JOIN
- El LEFT JOIN a `conciliation_results` puede generar multiples filas por servicio si un servicio tiene mas de un registro de conciliacion
- Solucion: agrupar a nivel de servicio (GROUP BY aud.ID) antes de agregar a nivel de prestador

### Fechas
- CREATED_AT en SYNC_ESTADOS tiene timestamps corruptos (todos 2026-03-04, fecha de backfill)
- Para cortes temporales, usar CREATED_AT de SYNC_AUDITORY_FINAL_REPORTS o SYNC_CONCILIATION_RESULTS
- MySQL tiene fechas invalidas ("0000-00-00") que se convirtieron a NULL en Snowflake

### Relacion con este proyecto (csv_conciliation_docker)
- Este proyecto escribe en MySQL: `conciliation_results`, `invoice_audits`, `auditory_final_reports`, `estados_auditory_final_reports`
- Los datos se sincronizan a Snowflake manualmente (script de sync)
- El dashboard BI lee de Snowflake (read-only)
- Cualquier dato nuevo insertado por este proyecto no aparecera en Snowflake hasta la proxima sincronizacion

### Tablas MySQL adicionales (no sincronizadas a Snowflake)
Estas tablas existen solo en MySQL y son usadas por este proyecto:
- `thirds_summary` / `thirds_summary_conciliation` — resumenes pre-agregados
- `nit_states` — tracking de estados por NIT
- `conciliation_acta_files` / `conciliation_acta_files_pdf` — metadata de actas generadas
- `third_departments` — departamentos por prestador
- `reconciliation_groups` — grupos de conciliacion
- `new_invoices_imports` — historial de importaciones de nuevas facturas

---

## 8. Volumetria actual (marzo 2026)

| Metrica | Valor |
|---|---|
| Prestadores (IPS) | 1,794 |
| Facturas totales | 2,814,048 |
| Servicios auditados | 15,130,740 |
| Registros de estado | 15,270,140 |
| Resultados de conciliacion | 6,917,707 |
| Servicios contabilizados | 12,807,863 |
| Servicios contabilizados con glosa | ~6,929,542 |
| Servicios conciliados | ~6,911,966 |
| Servicios sin conciliar | ~17,576 |
