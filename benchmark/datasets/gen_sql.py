"""Generate the text2sql dataset: three small but real SQLite databases in
distinct domains (seeded, deterministic) plus a routed question pool. Every
question is labeled with its home database so visit-by routing accuracy is
checkable; the harder questions exist to make the model earn real SQL errors,
which is what drives the repair-loop branch.

    python benchmark/datasets/gen_sql.py [out_dir]
"""
import os
import random
import sqlite3
import sys

FIRST = ["Ana", "Bo", "Chen", "Dara", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun",
         "Kira", "Leo", "Mia", "Noor", "Omar", "Pia", "Quinn", "Rui", "Sana", "Tom"]
CITIES = ["Berlin", "Osaka", "Porto", "Austin", "Nairobi", "Oslo"]
PRODUCTS = [("laptop stand", 39.0), ("usb hub", 24.5), ("webcam", 59.0),
            ("desk mat", 19.0), ("keyboard", 89.0), ("monitor arm", 129.0),
            ("headset", 74.5), ("dock", 199.0)]
# titles stay apostrophe-free on purpose: schema and row reprs land inside
# prompts, and the server field parser still trips on stray single quotes
BOOKS = [("The Quiet Harbor", "N. Ashe", "fiction", 2011),
         ("Salt and Circuit", "R. Okafor", "scifi", 2019),
         ("Maps for Nobody", "T. Lindqvist", "travel", 2015),
         ("The Fermentation Handbook", "M. Duarte", "cooking", 2020),
         ("Glass Winters", "N. Ashe", "fiction", 2016),
         ("A Field Guide to Falling", "J. Petrov", "poetry", 2013),
         ("Debugging the Sky", "R. Okafor", "scifi", 2022),
         ("Bread Alone", "M. Duarte", "cooking", 2017),
         ("The Last Ferry Out", "T. Lindqvist", "travel", 2021),
         ("Small Gods of the Kitchen", "H. Sato", "cooking", 2018),
         ("Orbit Decay", "L. Marsh", "scifi", 2014),
         ("The Cartographers Daughter", "T. Lindqvist", "fiction", 2023)]
DOCTORS = [("Voss", "cardiology"), ("Iqbal", "dermatology"),
           ("Chen", "orthopedics"), ("Mora", "pediatrics")]
STATUSES = ["done", "done", "done", "cancelled", "no_show"]

# (database, question) — the db label is what the visit-by router must recover
# from nothing but the consistency of the question with each node's schema.
QUESTIONS = [
    ("shop", "How many customers are there in each city?"),
    ("shop", "Which three customers spent the most in total, and how much did each spend?"),
    ("shop", "What is the average order value per city?"),
    ("shop", "Which product brought in the highest revenue?"),
    ("shop", "How many orders contained more than two distinct products?"),
    ("shop", "List the five most recent orders with the customer name and order total."),
    ("shop", "Which customers never ordered a keyboard?"),
    ("shop", "What share of total revenue came from Berlin customers?"),
    ("shop", "For each month of 2025, how many orders were placed?"),
    ("shop", "Which product is most often bought together with the usb hub?"),
    ("library", "Which book has been borrowed the most times?"),
    ("library", "How many loans are still not returned?"),
    ("library", "Which member borrowed the most books in 2025?"),
    ("library", "How many books are there in each genre?"),
    ("library", "Which author has the most books in the collection?"),
    ("library", "What is the average loan duration in days for returned loans?"),
    ("library", "Which books have never been borrowed?"),
    ("library", "List the three newest books with their authors."),
    ("clinic", "Which doctor has the most appointments?"),
    ("clinic", "How many appointments were cancelled?"),
    ("clinic", "Which specialty saw the most distinct patients?"),
    ("clinic", "List the five oldest patients with their birth years."),
    ("clinic", "How many appointments did each doctor have in June 2025?"),
    ("clinic", "What share of all appointments were no-shows?"),
    ("clinic", "Which patients never had a completed appointment?"),
    ("clinic", "How many patients come from each city?"),
]


def _fresh(path):
    if os.path.exists(path):
        os.remove(path)
    return sqlite3.connect(path)


def build_shop(path):
    conn = _fresh(path)
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL,
                            joined_at TEXT NOT NULL);
    CREATE TABLE products  (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price REAL NOT NULL);
    CREATE TABLE orders    (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
                            ordered_at TEXT NOT NULL);
    CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders(id),
                              product_id INTEGER NOT NULL REFERENCES products(id),
                              quantity INTEGER NOT NULL, PRIMARY KEY (order_id, product_id));
    """)
    rng = random.Random(42)
    for i, name in enumerate(FIRST, start=1):
        c.execute("INSERT INTO customers VALUES (?,?,?,?)",
                  (i, name, rng.choice(CITIES), f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"))
    for i, (name, price) in enumerate(PRODUCTS, start=1):
        c.execute("INSERT INTO products VALUES (?,?,?)", (i, name, price))
    oid = 0
    for _ in range(160):
        oid += 1
        cust = rng.randint(1, len(FIRST))
        c.execute("INSERT INTO orders VALUES (?,?,?)",
                  (oid, cust, f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"))
        for pid in rng.sample(range(1, len(PRODUCTS) + 1), rng.randint(1, 4)):
            c.execute("INSERT INTO order_items VALUES (?,?,?)", (oid, pid, rng.randint(1, 3)))
    conn.commit()
    conn.close()


def build_library(path):
    conn = _fresh(path)
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE books   (id INTEGER PRIMARY KEY, title TEXT NOT NULL, author TEXT NOT NULL,
                          genre TEXT NOT NULL, year INTEGER NOT NULL);
    CREATE TABLE members (id INTEGER PRIMARY KEY, name TEXT NOT NULL, joined_at TEXT NOT NULL);
    CREATE TABLE loans   (id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL REFERENCES books(id),
                          member_id INTEGER NOT NULL REFERENCES members(id),
                          borrowed_at TEXT NOT NULL, returned_at TEXT);
    """)
    rng = random.Random(43)
    for i, (title, author, genre, year) in enumerate(BOOKS, start=1):
        c.execute("INSERT INTO books VALUES (?,?,?,?,?)", (i, title, author, genre, year))
    for i, name in enumerate(FIRST[:12], start=1):
        c.execute("INSERT INTO members VALUES (?,?,?)",
                  (i, name, f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"))
    for lid in range(1, 121):
        month, day = rng.randint(1, 12), rng.randint(1, 24)
        returned = None if rng.random() < 0.3 else f"2025-{month:02d}-{day + rng.randint(1, 4):02d}"
        c.execute("INSERT INTO loans VALUES (?,?,?,?,?)",
                  (lid, rng.randint(1, len(BOOKS)), rng.randint(1, 12),
                   f"2025-{month:02d}-{day:02d}", returned))
    conn.commit()
    conn.close()


def build_clinic(path):
    conn = _fresh(path)
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE doctors  (id INTEGER PRIMARY KEY, name TEXT NOT NULL, specialty TEXT NOT NULL);
    CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL,
                           born_year INTEGER NOT NULL);
    CREATE TABLE appointments (id INTEGER PRIMARY KEY,
                               patient_id INTEGER NOT NULL REFERENCES patients(id),
                               doctor_id INTEGER NOT NULL REFERENCES doctors(id),
                               scheduled_at TEXT NOT NULL, status TEXT NOT NULL);
    """)
    rng = random.Random(44)
    for i, (name, spec) in enumerate(DOCTORS, start=1):
        c.execute("INSERT INTO doctors VALUES (?,?,?)", (i, f"Dr. {name}", spec))
    for i, name in enumerate(FIRST[4:], start=1):
        c.execute("INSERT INTO patients VALUES (?,?,?,?)",
                  (i, name, rng.choice(CITIES), rng.randint(1948, 2012)))
    for aid in range(1, 141):
        c.execute("INSERT INTO appointments VALUES (?,?,?,?,?)",
                  (aid, rng.randint(1, 16), rng.randint(1, len(DOCTORS)),
                   f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                   rng.choice(STATUSES)))
    conn.commit()
    conn.close()


# ---- data dictionaries -----------------------------------------------------------
# The analyst context = schema + sample rows (built by the program from the .db) +
# this dictionary: table purpose, column meaning, domain glossary and SQL idioms.
# Written to <db>.md; sized so the whole context is ~8 KB (~2.2k tokens).

COLUMNS = {
    "shop": {
        "customers": ("One row per registered buyer.", [
            ("id", "integer surrogate key, referenced by orders.customer_id"),
            ("name", "given name only, unique in this dataset"),
            ("city", "shipping city; one of " + ", ".join(CITIES)),
            ("joined_at", "ISO date YYYY-MM-DD of account creation, all in 2024")]),
        "products": ("Catalog of sellable items with a fixed list price.", [
            ("id", "integer surrogate key, referenced by order_items.product_id"),
            ("name", "lower-case product name, e.g. usb hub, keyboard"),
            ("price", "unit list price in dollars, REAL; revenue = quantity * price")]),
        "orders": ("One row per checkout; line items live in order_items.", [
            ("id", "integer surrogate key"),
            ("customer_id", "foreign key to customers.id"),
            ("ordered_at", "ISO date YYYY-MM-DD, all in 2025; month = substr(ordered_at, 1, 7)")]),
        "order_items": ("One row per (order, product); the only place quantities live.", [
            ("order_id", "foreign key to orders.id"),
            ("product_id", "foreign key to products.id"),
            ("quantity", "units of the product in that order, 1 to 3")]),
    },
    "library": {
        "books": ("Catalog of titles; one physical copy each.", [
            ("id", "integer surrogate key, referenced by loans.book_id"),
            ("title", "full title, no apostrophes"),
            ("author", "author as initial plus surname; several authors have multiple books"),
            ("genre", "one of fiction, scifi, travel, cooking, poetry"),
            ("year", "publication year as an integer")]),
        "members": ("Registered borrowers.", [
            ("id", "integer surrogate key, referenced by loans.member_id"),
            ("name", "given name only"),
            ("joined_at", "ISO date of registration, all in 2024")]),
        "loans": ("One row per borrowing event.", [
            ("id", "integer surrogate key"),
            ("book_id", "foreign key to books.id"),
            ("member_id", "foreign key to members.id"),
            ("borrowed_at", "ISO date the book left the library, all in 2025"),
            ("returned_at", "ISO date of return, or NULL while the loan is still open")]),
    },
    "clinic": {
        "doctors": ("Practitioners on staff.", [
            ("id", "integer surrogate key, referenced by appointments.doctor_id"),
            ("name", "Dr. plus surname"),
            ("specialty", "one of " + ", ".join(s for _, s in DOCTORS))]),
        "patients": ("Registered patients.", [
            ("id", "integer surrogate key, referenced by appointments.patient_id"),
            ("name", "given name only"),
            ("city", "home city; one of " + ", ".join(CITIES)),
            ("born_year", "birth year as an integer; age is relative to 2025")]),
        "appointments": ("One row per scheduled visit, whatever its outcome.", [
            ("id", "integer surrogate key"),
            ("patient_id", "foreign key to patients.id"),
            ("doctor_id", "foreign key to doctors.id"),
            ("scheduled_at", "ISO date YYYY-MM-DD, all in 2025"),
            ("status", "one of done, cancelled, no_show; only done counts as completed")]),
    },
}

GLOSSARY = {
    "shop": [
        ("order total", "SUM(order_items.quantity * products.price) over the lines of one order"),
        ("revenue", "the same sum over any set of orders; never use products.price alone"),
        ("average order value", "SUM of order totals divided by COUNT(DISTINCT orders.id)"),
        ("bought together", "two products sharing an order_id in order_items"),
        ("recent", "ORDER BY orders.ordered_at DESC"),
        ("share of revenue", "revenue of the subset divided by total revenue, times 100"),
        ("distinct products in an order", "COUNT(DISTINCT product_id) GROUP BY order_id"),
        ("never ordered X", "customers whose id is NOT IN the customer_ids of orders containing X"),
    ],
    "library": [
        ("borrowed most", "COUNT(*) of loans GROUP BY book_id, ORDER BY the count DESC LIMIT 1"),
        ("still out / not returned", "loans WHERE returned_at IS NULL"),
        ("loan duration in days", "julianday(returned_at) - julianday(borrowed_at), returned loans only"),
        ("never borrowed", "books whose id is NOT IN (SELECT book_id FROM loans)"),
        ("newest", "ORDER BY books.year DESC"),
        ("in 2025", "borrowed_at BETWEEN 2025-01-01 and 2025-12-31 as text comparison"),
        ("most active member", "COUNT(loans) GROUP BY member_id ORDER BY count DESC LIMIT 1"),
        ("books per genre", "COUNT(*) GROUP BY genre"),
    ],
    "clinic": [
        ("completed appointment", "appointments.status = done"),
        ("no-show rate", "COUNT of status = no_show divided by COUNT(*), times 100"),
        ("cancelled", "appointments.status = cancelled"),
        ("distinct patients", "COUNT(DISTINCT patient_id)"),
        ("in June 2025", "scheduled_at LIKE 2025-06-% as a text pattern"),
        ("oldest", "ORDER BY patients.born_year ASC"),
        ("busiest doctor", "COUNT(appointments) GROUP BY doctor_id ORDER BY count DESC LIMIT 1"),
        ("never completed", "patients whose id is NOT IN the patient_ids of done appointments"),
    ],
}

PATTERNS = {
    "shop": [
        ("customers per city",
         "SELECT c.city, COUNT(*) AS n FROM customers c GROUP BY c.city ORDER BY n DESC"),
        ("top spenders",
         "SELECT c.name, SUM(oi.quantity * p.price) AS spent FROM customers c JOIN orders o ON o.customer_id = c.id "
         "JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id "
         "GROUP BY c.id ORDER BY spent DESC LIMIT 3"),
        ("orders per month",
         "SELECT substr(o.ordered_at, 1, 7) AS month, COUNT(*) AS n FROM orders o GROUP BY month ORDER BY month"),
        ("product revenue",
         "SELECT p.name, SUM(oi.quantity * p.price) AS revenue FROM products p JOIN order_items oi "
         "ON oi.product_id = p.id GROUP BY p.id ORDER BY revenue DESC LIMIT 1"),
        ("orders with many distinct products",
         "SELECT COUNT(*) FROM (SELECT order_id FROM order_items GROUP BY order_id HAVING COUNT(DISTINCT product_id) > 2)"),
        ("bought together with X",
         "SELECT p2.name, COUNT(*) AS n FROM order_items a JOIN order_items b ON a.order_id = b.order_id "
         "AND a.product_id <> b.product_id JOIN products p1 ON p1.id = a.product_id JOIN products p2 ON p2.id = b.product_id "
         "WHERE p1.name = X GROUP BY p2.id ORDER BY n DESC LIMIT 1"),
    ],
    "library": [
        ("most borrowed book",
         "SELECT b.title, COUNT(*) AS n FROM loans l JOIN books b ON b.id = l.book_id GROUP BY b.id ORDER BY n DESC LIMIT 1"),
        ("open loans", "SELECT COUNT(*) FROM loans WHERE returned_at IS NULL"),
        ("most active member in a year",
         "SELECT m.name, COUNT(*) AS n FROM loans l JOIN members m ON m.id = l.member_id "
         "WHERE l.borrowed_at BETWEEN date(2025, 1, 1) AND date(2025, 12, 31) GROUP BY m.id ORDER BY n DESC LIMIT 1"),
        ("books per genre", "SELECT genre, COUNT(*) AS n FROM books GROUP BY genre ORDER BY n DESC"),
        ("average loan duration",
         "SELECT AVG(julianday(returned_at) - julianday(borrowed_at)) FROM loans WHERE returned_at IS NOT NULL"),
        ("never borrowed", "SELECT title FROM books WHERE id NOT IN (SELECT book_id FROM loans)"),
    ],
    "clinic": [
        ("busiest doctor",
         "SELECT d.name, COUNT(*) AS n FROM appointments a JOIN doctors d ON d.id = a.doctor_id GROUP BY d.id ORDER BY n DESC LIMIT 1"),
        ("cancelled count", "SELECT COUNT(*) FROM appointments WHERE status = X"),
        ("distinct patients per specialty",
         "SELECT d.specialty, COUNT(DISTINCT a.patient_id) AS n FROM appointments a JOIN doctors d ON d.id = a.doctor_id "
         "GROUP BY d.specialty ORDER BY n DESC LIMIT 1"),
        ("oldest patients", "SELECT name, born_year FROM patients ORDER BY born_year ASC LIMIT 5"),
        ("appointments per doctor in a month",
         "SELECT d.name, COUNT(*) AS n FROM appointments a JOIN doctors d ON d.id = a.doctor_id "
         "WHERE a.scheduled_at LIKE X GROUP BY d.id ORDER BY n DESC"),
        ("no-show share", "SELECT 100.0 * SUM(status = X) / COUNT(*) FROM appointments"),
    ],
}

CONVENTIONS = [
    "Dates are ISO text YYYY-MM-DD; compare and slice them as strings (substr, LIKE, BETWEEN).",
    "Surrogate keys are INTEGER PRIMARY KEY; every foreign key is declared with REFERENCES.",
    "Money is REAL in dollars with no currency column; round only in the final SELECT.",
    "Names carry no apostrophes; string literals may be single quoted safely.",
    "Prefer explicit JOIN ... ON over comma joins; alias every table.",
    "Aggregations must GROUP BY every non-aggregated selected column.",
    "Use LIMIT for top-N questions and ORDER BY the aggregate, not the key.",
    "NULL means unknown or still open; filter with IS NULL / IS NOT NULL, never = NULL.",
    "Percent answers are 100.0 * part / whole; cast one side to REAL to avoid integer division.",
    "Return one result set; no temporary tables, no PRAGMA, no multiple statements.",
    "SQLite has no RIGHT JOIN and no FULL JOIN; rewrite with LEFT JOIN from the other side.",
    "Boolean expressions evaluate to 0 or 1, so SUM(status = X) counts matching rows.",
    "String comparison is case sensitive; the stored values are all lower case except names.",
    "julianday(text) turns an ISO date into a day number; subtract two for a duration in days.",
    "There is no schema qualifier; refer to tables by their bare names.",
    "Ties in top-N questions are broken by the surrogate key ascending.",
]


def _value_ranges(name, db_path):
    """Per-table row counts and per-column distinct counts / min / max, from the
    built database itself: the part of a real data dictionary an analyst reads
    before writing a filter."""
    conn = sqlite3.connect(db_path)
    out = ["== Value ranges (from the current data) =="]
    for table in COLUMNS[name]:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        out.append(f"{table}: {n} rows")
        for col, _ in COLUMNS[name][table][1]:
            d, lo, hi = conn.execute(
                f"SELECT COUNT(DISTINCT {col}), MIN({col}), MAX({col}) FROM {table}").fetchone()
            out.append(f"- {table}.{col}: {d} distinct, min {lo}, max {hi}")
    conn.close()
    return out


def data_dictionary(name, db_path):
    out = [f"Data dictionary: {name} database", ""]
    for table, (purpose, cols) in COLUMNS[name].items():
        out.append(f"== Table {table} ==")
        out.append(purpose)
        for col, meaning in cols:
            out.append(f"- {table}.{col}: {meaning}")
        out.append("")
    out.extend(_value_ranges(name, db_path))
    out.append("")
    out.append("== Glossary (question phrase -> SQL idiom) ==")
    for term, idiom in GLOSSARY[name]:
        out.append(f"- {term}: {idiom}")
    out.append("")
    out.append("== Worked query patterns (X stands for the literal from the question) ==")
    for what, sql in PATTERNS[name]:
        out.append(f"- {what}: {sql}")
    out.append("")
    out.append("== Conventions ==")
    out.extend(f"- {c}" for c in CONVENTIONS)
    text = "\n".join(out)
    assert "'" not in text, name
    return text


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "sql")
    os.makedirs(out_dir, exist_ok=True)
    for name, build in [("shop", build_shop), ("library", build_library), ("clinic", build_clinic)]:
        db = os.path.join(out_dir, f"{name}.db")
        build(db)
        tables = sqlite3.connect(db).execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        md = os.path.join(out_dir, f"{name}.md")
        with open(md, "w") as f:
            f.write(data_dictionary(name, db))
        print(f"{db}: {tables} tables; {md}: {os.path.getsize(md)} bytes")
    qp = os.path.join(out_dir, "questions.tsv")
    with open(qp, "w") as f:
        f.write("\n".join(f"{db}\t{q}" for db, q in QUESTIONS) + "\n")
    print(f"{qp}: {len(QUESTIONS)} questions across 3 databases")
