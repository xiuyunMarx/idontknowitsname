Data dictionary: library database

== Table books ==
Catalog of titles; one physical copy each.
- books.id: integer surrogate key, referenced by loans.book_id
- books.title: full title, no apostrophes
- books.author: author as initial plus surname; several authors have multiple books
- books.genre: one of fiction, scifi, travel, cooking, poetry
- books.year: publication year as an integer

== Table members ==
Registered borrowers.
- members.id: integer surrogate key, referenced by loans.member_id
- members.name: given name only
- members.joined_at: ISO date of registration, all in 2024

== Table loans ==
One row per borrowing event.
- loans.id: integer surrogate key
- loans.book_id: foreign key to books.id
- loans.member_id: foreign key to members.id
- loans.borrowed_at: ISO date the book left the library, all in 2025
- loans.returned_at: ISO date of return, or NULL while the loan is still open

== Value ranges (from the current data) ==
books: 12 rows
- books.id: 12 distinct, min 1, max 12
- books.title: 12 distinct, min A Field Guide to Falling, max The Quiet Harbor
- books.author: 7 distinct, min H. Sato, max T. Lindqvist
- books.genre: 5 distinct, min cooking, max travel
- books.year: 12 distinct, min 2011, max 2023
members: 12 rows
- members.id: 12 distinct, min 1, max 12
- members.name: 12 distinct, min Ana, max Leo
- members.joined_at: 12 distinct, min 2024-01-10, max 2024-12-26
loans: 120 rows
- loans.id: 120 distinct, min 1, max 120
- loans.book_id: 12 distinct, min 1, max 12
- loans.member_id: 12 distinct, min 1, max 12
- loans.borrowed_at: 97 distinct, min 2025-01-01, max 2025-12-21
- loans.returned_at: 69 distinct, min 2025-01-02, max 2025-12-22

== Glossary (question phrase -> SQL idiom) ==
- borrowed most: COUNT(*) of loans GROUP BY book_id, ORDER BY the count DESC LIMIT 1
- still out / not returned: loans WHERE returned_at IS NULL
- loan duration in days: julianday(returned_at) - julianday(borrowed_at), returned loans only
- never borrowed: books whose id is NOT IN (SELECT book_id FROM loans)
- newest: ORDER BY books.year DESC
- in 2025: borrowed_at BETWEEN 2025-01-01 and 2025-12-31 as text comparison
- most active member: COUNT(loans) GROUP BY member_id ORDER BY count DESC LIMIT 1
- books per genre: COUNT(*) GROUP BY genre

== Worked query patterns (X stands for the literal from the question) ==
- most borrowed book: SELECT b.title, COUNT(*) AS n FROM loans l JOIN books b ON b.id = l.book_id GROUP BY b.id ORDER BY n DESC LIMIT 1
- open loans: SELECT COUNT(*) FROM loans WHERE returned_at IS NULL
- most active member in a year: SELECT m.name, COUNT(*) AS n FROM loans l JOIN members m ON m.id = l.member_id WHERE l.borrowed_at BETWEEN date(2025, 1, 1) AND date(2025, 12, 31) GROUP BY m.id ORDER BY n DESC LIMIT 1
- books per genre: SELECT genre, COUNT(*) AS n FROM books GROUP BY genre ORDER BY n DESC
- average loan duration: SELECT AVG(julianday(returned_at) - julianday(borrowed_at)) FROM loans WHERE returned_at IS NOT NULL
- never borrowed: SELECT title FROM books WHERE id NOT IN (SELECT book_id FROM loans)

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