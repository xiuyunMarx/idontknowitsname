"""Build the InterCode-SQL task pack for benchmark/evaluation/programs/intercode_sql.jac.

Sources (downloaded at build time, nothing vendored by hand):
  princeton-nlp/intercode  data/sql/spider/ic_spider_dev.json   1034 Spider dev tasks
  princeton-nlp/intercode  data/sql/spider/ic_spider_dbs.sql    MySQL dump of the 19 dev databases

InterCode serves those databases from MySQL. The Jac tenant runs them in an
in-memory SQLite database instead, so the dump is converted here once and shipped
as per-database statement lists. Every gold query is executed against the converted
database: a task enters the pool only if its gold query runs and returns at least one
row, which is also what proves the conversion faithful.

    python benchmark/evaluation/datasets/build/build_intercode_sql.py
"""

import argparse
import json
import os
import random
import re
import sqlite3
import urllib.request
from collections import Counter, defaultdict

RAW = "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/sql/spider"
DEV_URL = f"{RAW}/ic_spider_dev.json"
DBS_URL = f"{RAW}/ic_spider_dbs.sql"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "..", "intercode_sql_30.json")
SEED = 20260824
N_CASES = 30
MAX_PER_DB = 3


# --------------------------------------------------------------------- download

def fetch(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        print(f"[fetch] {url}")
        urllib.request.urlretrieve(url, path)
    return path


# ------------------------------------------------------------------- conversion

def split_statements(sql: str):
    """Split a mysqldump on `;`, ignoring separators inside quoted strings."""
    out, buf, quote, escaped = [], [], None, False
    for ch in sql:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\" and quote is not None:
            buf.append(ch)
            escaped = True
            continue
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            continue
        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


SKIP_PREFIXES = ("/*!", "SET ", "LOCK TABLES", "UNLOCK TABLES", "CREATE USER", "GRANT ",
                 "FLUSH ", "DROP TABLE", "DROP DATABASE", "--")

# `KEY x (y)` / `UNIQUE KEY ...` are MySQL index declarations inside CREATE TABLE;
# SQLite has no such column-list syntax, and the tasks never depend on an index.
INDEX_LINE = re.compile(r"^\s*(UNIQUE\s+|FULLTEXT\s+|SPATIAL\s+)?KEY\s", re.I)
TABLE_OPTIONS = re.compile(r"\)\s*ENGINE=.*$", re.I | re.S)
COLUMN_NOISE = re.compile(
    r"\s+(CHARACTER SET \w+|COLLATE \w+|AUTO_INCREMENT|unsigned|zerofill)", re.I)
# MySQL index prefix length on a text key column, e.g. PRIMARY KEY ("Year_awarded"(255)).
# The `(` sits directly on the closing quote, which is what separates it from a type
# declaration like "Channel" varchar(255).
KEY_PREFIX_LEN = re.compile(r'("(?:[^"]|"")+")\(\d+\)')


def clean_statement(stmt: str) -> str:
    stmt = re.sub(r"/\*!.*?\*/", " ", stmt, flags=re.S)
    # Backticks become double quotes rather than disappearing: Spider column names
    # include `18_49_Rating_Share` and `Official_ratings_(millions)`, which SQLite
    # only accepts quoted.
    stmt = stmt.replace("`", '"')
    stmt = KEY_PREFIX_LEN.sub(r"\1", stmt)
    stmt = stmt.replace("\\'", "''")
    if stmt.upper().startswith("CREATE TABLE"):
        head, _, body = stmt.partition("(")
        body = TABLE_OPTIONS.sub(")", "(" + body)
        lines = []
        for line in body.splitlines():
            if INDEX_LINE.match(line):
                continue
            lines.append(COLUMN_NOISE.sub("", line))
        body = "\n".join(lines)
        # dropping index lines can leave a dangling comma before the closing paren
        body = re.sub(r",\s*\)\s*$", "\n)", body.strip())
        stmt = head.strip() + " " + body
    return stmt.strip()


def convert_dump(sql_text: str) -> dict:
    """MySQL dump -> {database name: [sqlite statements]} in file order."""
    databases, current = {}, None
    for stmt in split_statements(sql_text):
        stripped = re.sub(r"/\*!.*?\*/", " ", stmt, flags=re.S).strip()
        use = re.match(r"USE\s+`?([^`\s]+)`?", stripped, re.I)
        if use:
            current = use.group(1)
            databases.setdefault(current, [])
            continue
        if re.match(r"CREATE DATABASE", stripped, re.I):
            continue
        if current is None or stripped.upper().startswith(SKIP_PREFIXES):
            continue
        if not re.match(r"(CREATE TABLE|INSERT INTO)", stripped, re.I):
            continue
        databases[current].append(clean_statement(stmt))
    return databases


def build_db(statements) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    for stmt in statements:
        conn.execute(stmt)
    conn.commit()
    return conn


# ---------------------------------------------------------------------- selection

def pick_cases(pool, seed=SEED, n=N_CASES, max_per_db=MAX_PER_DB):
    """A spread over hardness and databases, seeded so the pack is reproducible."""
    rng = random.Random(seed)
    by_hardness = defaultdict(list)
    for task in pool:
        by_hardness[task["hardness"]].append(task)
    for bucket in by_hardness.values():
        rng.shuffle(bucket)

    order = ["easy", "medium", "hard", "extra"]
    quota = {h: n // len(order) for h in order}
    for h in order[: n % len(order)]:
        quota[h] += 1

    picked, per_db = [], Counter()
    for h in order:
        taken = 0
        for task in by_hardness.get(h, []):
            if taken >= quota[h]:
                break
            if per_db[task["db"]] >= max_per_db:
                continue
            picked.append(task)
            per_db[task["db"]] += 1
            taken += 1
    # top up from anywhere if a hardness bucket ran dry under the per-database cap
    if len(picked) < n:
        chosen = {id(t) for t in picked}
        for task in rng.sample(pool, len(pool)):
            if len(picked) >= n:
                break
            if id(task) in chosen or per_db[task["db"]] >= max_per_db:
                continue
            picked.append(task)
            per_db[task["db"]] += 1
    picked.sort(key=lambda t: (order.index(t["hardness"]), t["db"], t["id"]))
    return picked


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("BUILD_CACHE", "/tmp/intercode_build"))
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    dev = json.load(open(fetch(DEV_URL, args.cache)))
    dump = open(fetch(DBS_URL, args.cache), encoding="utf-8", errors="replace").read()

    databases = convert_dump(dump)
    print(f"[convert] {len(databases)} databases, "
          f"{sum(len(v) for v in databases.values())} statements")

    conns, broken = {}, []
    for name, statements in databases.items():
        try:
            conns[name] = build_db(statements)
        except Exception as err:  # a database that will not build is reported, not hidden
            broken.append((name, str(err)))
    for name, err in broken:
        print(f"[convert] FAILED to build {name}: {err}")

    pool, rejected = [], Counter()
    for index, record in enumerate(dev):
        db = record["db"]
        if db not in conns:
            rejected["no database"] += 1
            continue
        try:
            rows = conns[db].execute(record["gold"]).fetchall()
        except Exception:
            rejected["gold errors"] += 1
            continue
        if not rows:
            rejected["gold returns no rows"] += 1
            continue
        # A gold answer of a single 0 or 1 is guessable, so execution match stops
        # measuring anything on that task.
        if len(rows) == 1 and len(rows[0]) == 1 and rows[0][0] in (0, 1, None, ""):
            rejected["gold result is a trivial scalar"] += 1
            continue
        pool.append({
            "id": f"ic-sql-{index:04d}",
            "db": db,
            "question": record["query"],
            "gold": record["gold"],
            "hardness": record["hardness"],
            "gold_rows": len(rows),
        })
    print(f"[verify] {len(pool)}/{len(dev)} dev tasks usable; rejected: {dict(rejected)}")

    cases = pick_cases(pool)
    used_dbs = sorted({task["db"] for task in cases})
    pack = {
        "databases": {name: {"setup": databases[name]} for name in used_dbs},
        "tasks": cases,
    }
    with open(args.out, "w") as handle:
        json.dump(pack, handle, indent=1, ensure_ascii=False)

    # re-verify from the shipped pack alone, the way the tenant will load it
    reload = json.load(open(args.out))
    checked = 0
    for task in reload["tasks"]:
        conn = build_db(reload["databases"][task["db"]]["setup"])
        assert conn.execute(task["gold"]).fetchall(), task["id"]
        checked += 1
    size = os.path.getsize(args.out) / 1024
    print(f"[pack] {args.out} | {checked} tasks | {len(used_dbs)} databases | {size:.0f} KB")
    print(f"[pack] hardness: {dict(Counter(t['hardness'] for t in cases))}")
    print(f"[pack] databases: {dict(Counter(t['db'] for t in cases))}")


if __name__ == "__main__":
    main()
