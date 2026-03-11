-- Migración: Crear tabla conciliation_proceedings para numeración de actas
-- Fecha: 2026-03-11

CREATE TABLE IF NOT EXISTS conciliation_proceedings (
    id CHAR(36) NOT NULL PRIMARY KEY,
    acta_number INT NOT NULL,
    provider_nit VARCHAR(50) NOT NULL,
    provider_name VARCHAR(255) DEFAULT NULL,
    file_name VARCHAR(500) DEFAULT NULL,
    file_path VARCHAR(1000) DEFAULT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'generated',
    created_by VARCHAR(255) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_acta_number (acta_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agregar columna proceeding_id a conciliation_results
ALTER TABLE conciliation_results
    ADD COLUMN proceeding_id CHAR(36) DEFAULT NULL,
    ADD INDEX idx_proceeding_id (proceeding_id),
    ADD CONSTRAINT fk_cr_proceeding
        FOREIGN KEY (proceeding_id) REFERENCES conciliation_proceedings(id)
        ON DELETE SET NULL ON UPDATE CASCADE;
