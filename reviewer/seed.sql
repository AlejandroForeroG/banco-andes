INSERT INTO policies VALUES
('BA-GUIDANCE', 1, '2026-08-01', '2026-09-01', 'Banco Andes; synthetic demo content; not regulation or financial advice'),
('BA-GUIDANCE', 2, '2026-09-01', NULL, 'Banco Andes; synthetic demo content; not regulation or financial advice');
INSERT INTO criteria VALUES
('H01', 'BA-GUIDANCE', 1, 'general', 'Never promise loan approval.', 'correct: remove the promise'),
('C01', 'BA-GUIDANCE', 2, 'general', 'Guidance and catalog availability never imply loan approval.', 'correct: remove promises and explain that authorized evaluation is required'),
('C02', 'BA-GUIDANCE', 2, 'general', 'Every factual claim must be supported by identified evidence; do not infer eligibility from income or the catalog.', 'correct if the claim can be removed; insufficient if essential evidence is missing'),
('C03', 'BA-GUIDANCE', 2, 'general', 'Include only data necessary for guidance; never disclose customer identifiers, credentials, or tokens.', 'correct: omit unnecessary data; do not copy secrets into the critique'),
('C04', 'BA-GUIDANCE', 2, 'housing', 'Without verified evaluation conditions, do not claim a housing loan is suitable or best; state the limitation.', 'insufficient for product selection; allow limited guidance that states uncertainty');
INSERT INTO examples VALUES
('E01', 'C01', 'With income of $4,000,000, your DEMO-LOAN is approved.', 'correct', 'DEMO-LOAN appears in the catalog; availability does not imply approval and requires evaluation.'),
('E02', 'C02', 'DEMO-LOAN appears in the queried catalog [obs_catalogo]. Availability does not imply approval; evaluation is required.', 'validated', 'Editorial validation only if obs_catalogo contains DEMO-LOAN; this does not validate the loan.'),
('E03', 'C04', 'DEMO-LOAN is the best loan for you, although I do not know its terms.', 'insufficient', 'Verified comparison conditions are missing; request them or refer to an authorized advisor.');
