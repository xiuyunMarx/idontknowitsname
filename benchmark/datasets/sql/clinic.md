Data dictionary: clinic database

== Table doctors ==
Practitioners on staff.
- doctors.id: integer surrogate key, referenced by appointments.doctor_id
- doctors.name: Dr. plus surname
- doctors.specialty: one of cardiology, dermatology, orthopedics, pediatrics

== Table patients ==
Registered patients.
- patients.id: integer surrogate key, referenced by appointments.patient_id
- patients.name: given name only
- patients.city: home city; one of Berlin, Osaka, Porto, Austin, Nairobi, Oslo
- patients.born_year: birth year as an integer; age is relative to 2025

== Table appointments ==
One row per scheduled visit, whatever its outcome.
- appointments.id: integer surrogate key
- appointments.patient_id: foreign key to patients.id
- appointments.doctor_id: foreign key to doctors.id
- appointments.scheduled_at: ISO date YYYY-MM-DD, all in 2025
- appointments.status: one of done, cancelled, no_show; only done counts as completed

== Value ranges (from the current data) ==
doctors: 4 rows
- doctors.id: 4 distinct, min 1, max 4
- doctors.name: 4 distinct, min Dr. Chen, max Dr. Voss
- doctors.specialty: 4 distinct, min cardiology, max pediatrics
patients: 16 rows
- patients.id: 16 distinct, min 1, max 16
- patients.name: 16 distinct, min Eli, max Tom
- patients.city: 6 distinct, min Austin, max Porto
- patients.born_year: 14 distinct, min 1949, max 1999
appointments: 140 rows
- appointments.id: 140 distinct, min 1, max 140
- appointments.patient_id: 16 distinct, min 1, max 16
- appointments.doctor_id: 4 distinct, min 1, max 4
- appointments.scheduled_at: 117 distinct, min 2025-01-01, max 2025-12-26
- appointments.status: 3 distinct, min cancelled, max no_show

== Glossary (question phrase -> SQL idiom) ==
- completed appointment: appointments.status = done
- no-show rate: COUNT of status = no_show divided by COUNT(*), times 100
- cancelled: appointments.status = cancelled
- distinct patients: COUNT(DISTINCT patient_id)
- in June 2025: scheduled_at LIKE 2025-06-% as a text pattern
- oldest: ORDER BY patients.born_year ASC
- busiest doctor: COUNT(appointments) GROUP BY doctor_id ORDER BY count DESC LIMIT 1
- never completed: patients whose id is NOT IN the patient_ids of done appointments

== Worked query patterns (X stands for the literal from the question) ==
- busiest doctor: SELECT d.name, COUNT(*) AS n FROM appointments a JOIN doctors d ON d.id = a.doctor_id GROUP BY d.id ORDER BY n DESC LIMIT 1
- cancelled count: SELECT COUNT(*) FROM appointments WHERE status = X
- distinct patients per specialty: SELECT d.specialty, COUNT(DISTINCT a.patient_id) AS n FROM appointments a JOIN doctors d ON d.id = a.doctor_id GROUP BY d.specialty ORDER BY n DESC LIMIT 1
- oldest patients: SELECT name, born_year FROM patients ORDER BY born_year ASC LIMIT 5
- appointments per doctor in a month: SELECT d.name, COUNT(*) AS n FROM appointments a JOIN doctors d ON d.id = a.doctor_id WHERE a.scheduled_at LIKE X GROUP BY d.id ORDER BY n DESC
- no-show share: SELECT 100.0 * SUM(status = X) / COUNT(*) FROM appointments

== Conventions ==
- Dates are ISO text YYYY-MM-DD; compare and slice them as strings (substr, LIKE, BETWEEN).
- Surrogate keys are INTEGER PRIMARY KEY; every foreign key is declared with REFERENCES.
- Money is REAL in dollars with no currency column; round only in the final SELECT.
- Names carry no apostrophes; string literals may be single quoted safely.
- Prefer explicit JOIN ... ON over comma joins; alias every table.
- Aggregations must GROUP BY every non-aggregated selected column.
- Use LIMIT for top-N questions and ORDER BY the aggregate, not the key.
- NULL means unknown or still open; filter with IS NULL / IS NOT NULL, never = NULL.
- Percent answers are 100.0 * part / whole; cast one side to REAL to avoid integer division.
- Return one result set; no temporary tables, no PRAGMA, no multiple statements.
- SQLite has no RIGHT JOIN and no FULL JOIN; rewrite with LEFT JOIN from the other side.
- Boolean expressions evaluate to 0 or 1, so SUM(status = X) counts matching rows.
- String comparison is case sensitive; the stored values are all lower case except names.
- julianday(text) turns an ISO date into a day number; subtract two for a duration in days.
- There is no schema qualifier; refer to tables by their bare names.
- Ties in top-N questions are broken by the surrogate key ascending.