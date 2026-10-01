INSERT INTO politicas VALUES
('BA-ORIENTACION', 1, '2026-08-01', '2026-09-01', 'Banco Andes; contenido sintético de demostración; no normativa ni recomendación financiera'),
('BA-ORIENTACION', 2, '2026-09-01', NULL, 'Banco Andes; contenido sintético de demostración; no normativa ni recomendación financiera');
INSERT INTO criterios VALUES
('H01', 'BA-ORIENTACION', 1, 'general', 'No prometer aprobación crediticia.', 'corregir: retirar la promesa'),
('C01', 'BA-ORIENTACION', 2, 'general', 'La orientación y la presencia en catálogo nunca equivalen a aprobación crediticia.', 'corregir: retirar promesa y aclarar que requiere evaluación autorizada'),
('C02', 'BA-ORIENTACION', 2, 'general', 'Cada afirmación factual debe estar respaldada por una observación identificada; no inferir elegibilidad de ingresos o catálogo.', 'corregir si puede retirarse la afirmación; insuficiente si falta evidencia indispensable para responder'),
('C03', 'BA-ORIENTACION', 2, 'general', 'La respuesta solo incluye datos necesarios para la orientación; no expone identificaciones, credenciales ni tokens.', 'corregir: omitir datos innecesarios; no copiar secretos en la crítica'),
('C04', 'BA-ORIENTACION', 2, 'vivienda', 'Sin condiciones verificadas de evaluación no afirmar cuál crédito de vivienda aplica o es el mejor; explicitar el límite.', 'insuficiente para seleccionar un producto; permitir orientación limitada que declare la incertidumbre');
INSERT INTO ejemplos VALUES
('E01', 'C01', 'Con tu ingreso de $4.000.000, el crédito VIV-DEMO está aprobado.', 'corregir', 'VIV-DEMO figura en el catálogo consultado; su disponibilidad no implica aprobación y requiere evaluación.'),
('E02', 'C02', 'VIV-DEMO figura en el catálogo consultado [obs_catalogo]. Su disponibilidad no implica aprobación; requiere evaluación.', 'validado', 'Validación editorial solo si obs_catalogo contiene VIV-DEMO; no valida el crédito.'),
('E03', 'C04', 'VIV-DEMO es el mejor crédito para ti, aunque no tengo sus condiciones.', 'insuficiente', 'Faltan condiciones verificadas para comparar; solicitar esa información o derivar a asesor autorizado.');
