-- Migración: Crear tabla conciliation_results_history para trazabilidad de re-conciliaciones
-- Fecha: 2026-03-12

CREATE TABLE IF NOT EXISTS conciliation_results_history (
    id CHAR(36) NOT NULL PRIMARY KEY,
    conciliation_result_id CHAR(36) NOT NULL,
    proceeding_id_before CHAR(36) DEFAULT NULL,
    proceeding_id_after CHAR(36) DEFAULT NULL,
    accepted_value_ips_before DECIMAL(18,2) NOT NULL,
    accepted_value_eps_before DECIMAL(18,2) NOT NULL,
    eps_ratified_value_before DECIMAL(18,2) NOT NULL,
    accepted_value_ips_after DECIMAL(18,2) NOT NULL,
    accepted_value_eps_after DECIMAL(18,2) NOT NULL,
    eps_ratified_value_after DECIMAL(18,2) NOT NULL,
    observation TEXT DEFAULT NULL,
    created_by VARCHAR(255) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_crh_result (conciliation_result_id),
    INDEX idx_crh_proceeding_after (proceeding_id_after),
    CONSTRAINT fk_crh_result
        FOREIGN KEY (conciliation_result_id) REFERENCES conciliation_results(id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
