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


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "sql")
    os.makedirs(out_dir, exist_ok=True)
    for name, build in [("shop", build_shop), ("library", build_library), ("clinic", build_clinic)]:
        db = os.path.join(out_dir, f"{name}.db")
        build(db)
        tables = sqlite3.connect(db).execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        print(f"{db}: {tables} tables")
    qp = os.path.join(out_dir, "questions.tsv")
    with open(qp, "w") as f:
        f.write("\n".join(f"{db}\t{q}" for db, q in QUESTIONS) + "\n")
    print(f"{qp}: {len(QUESTIONS)} questions across 3 databases")
