"""InterCode-SQL style execution-feedback repair loop (evaluation tenant).

    generate_sql -> execute -> validate
                                 |-- clean result -> finalize
                                 `-- error / empty -> diagnose -> revise_sql -> execute ...

The loop is a real loop in the program, not a ReAct loop inside one call: the
repair callsites run once per turn, their arguments change every turn, and the
back edge `revise_sql -> diagnose` is what the may-run-next topology records.
Execution and validation are deterministic — SQLite over an in-memory Spider
style database, no model in the judging path — so the systems measurement does
not move with model quality.

What drives the loop is execution feedback alone (a SQLite error, or a query
that returns no rows), never the gold query. The gold query is used once, after
the loop, to print an execution-match verdict, so text-to-SQL accuracy is
reported next to but separately from TTFT and cache metrics.

Turn control:
  $SQL_MAX_TURNS   (default 3) hard cap on repair turns.
  $SQL_FORCE_TURNS (default 0) minimum repair turns, taken even when the first
                   query already runs clean. Set it to fix the callsite sequence
                   across tasks when a systems comparison needs equal work.

Run:

    python start_server.py --model Qwen/Qwen2.5-7B-Instruct \
        --program intercode_sql:benchmark/evaluation/programs/intercode_sql.jac:8973

    SQL_TASK_INDEX=0 jac run benchmark/evaluation/programs/intercode_sql.jac

Inputs: $SQL_TASK_INDEX into the task pack, or $SQL_QUESTION plus $SQL_DB to ask
something of your own (the verdict is then UNGRADED).
Tasks: $INTERCODE_TASKS, a JSON object with `databases` (name -> {"setup": [sql,
...]}) and `tasks` ({id, db, question, gold, hardness}). The bundled
`datasets/intercode_sql_30.json` holds 30 questions from the InterCode-SQL dev set
(Spider 1.0) over 16 of its databases, converted to SQLite (see datasets/README.md);
`datasets/intercode_sql_tasks.json` is the small hand-written smoke fixture.
"""

import os;
import json;
import time;
import sqlite3;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");
glob MAX_TURNS = int(os.environ.get("SQL_MAX_TURNS") or "3");
glob FORCE_TURNS = int(os.environ.get("SQL_FORCE_TURNS") or "0");

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-7B-Instruct"),
    comm_ip="localhost",
    comm_port=8973,
    program_name="intercode_sql"
);


# ------------------------------------------------------------------ task pack

def pack_path() -> str {
    here = os.path.dirname(os.path.abspath(__file__));
    return str(os.environ.get("INTERCODE_TASKS")
               or os.path.join(here, "..", "datasets", "intercode_sql_30.json"));
}

def read_pack() -> any {
    path = pack_path();
    if not os.path.exists(path) {
        print(f"[intercode_sql] task pack missing: {path}");
        return {};
    }
    handle = open(path, "r");
    data = json.loads(handle.read());
    handle.close();
    return data;
}

glob PACK: any = read_pack();

def task_count() -> int {
    count = 0;
    if "tasks" in PACK {
        for item in PACK["tasks"] {
            count += 1;
        }
    }
    return count;
}

def task_field(name: str) -> str {
    if task_count() == 0 {
        return "";
    }
    idx = int(os.environ.get("SQL_TASK_INDEX") or "0") % task_count();
    record = PACK["tasks"][idx];
    if name in record {
        return str(record[name]);
    }
    return "";
}

glob TASK_ID: str = task_field("id");
glob HARDNESS: str = task_field("hardness");
glob DB_NAME: str = str(os.environ.get("SQL_DB") or task_field("db"));
glob QUESTION: str = str(os.environ.get("SQL_QUESTION") or task_field("question"));
# The gold query never enters a prompt; it only grades the final SQL.
glob GOLD: str = "";


def load_gold() -> str {
    if os.environ.get("SQL_QUESTION") {
        return "";
    }
    return task_field("gold");
}

def open_db() -> any {
    conn = sqlite3.connect(":memory:");
    if "databases" in PACK and DB_NAME in PACK["databases"] {
        for stmt in PACK["databases"][DB_NAME]["setup"] {
            conn.execute(str(stmt));
        }
        conn.commit();
    } else {
        print(f"[intercode_sql] database missing from pack: {DB_NAME}");
    }
    return conn;
}

glob DB: any = open_db();
glob GOLD_SQL: str = load_gold();


def schema_text() -> str {
    parts: list[str] = [];
    for row in DB.execute("SELECT sql FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall() {
        parts.append(str(row[0]) + ";");
    }
    return "\n".join(parts);
}

glob SCHEMA: str = schema_text();


def table_names() -> list[str] {
    names: list[str] = [];
    for row in DB.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall() {
        names.append(str(row[0]));
    }
    return names;
}


# ------------------------------------------------------- deterministic harness

def clean_sql(text: str) -> str {
    # Small models wrap the query in a fence or a label; the harness strips that
    # before execution, and the model still sees its own raw answer next turn.
    body = text.strip();
    if "```" in body {
        chunk = body.split("```")[1];
        if chunk.lower().startswith("sql") {
            chunk = chunk[3:];
        }
        body = chunk.strip();
    }
    for label in ["sql:", "query:", "answer:"] {
        if body.lower().startswith(label) {
            body = body[len(label):].strip();
        }
    }
    return body.rstrip(";").strip();
}

def run_query(sql: str) -> list {
    out: list = [];
    for row in DB.execute(sql).fetchall() {
        out.append(row);
    }
    return out;
}

def execute_sql(sql: str) -> str {
    # Run one candidate query and report exactly what the database said.
    time.sleep(0.20 * TOOL_SCALE);
    query = clean_sql(sql);
    head = query.lower();
    if not (head.startswith("select") or head.startswith("with")) {
        return "ERROR: this environment runs SELECT statements only.";
    }
    try {
        rows = run_query(query);
    } except Exception as err {
        return "ERROR: " + str(err);
    }
    if not rows {
        return "OK: 0 rows returned.";
    }
    lines: list[str] = [];
    for row in rows[:5] {
        lines.append(str(row));
    }
    body: str = f"OK: {len(rows)} rows\n" + "\n".join(lines);
    if len(rows) > 5 {
        body += f"\n... {len(rows) - 5} more rows";
    }
    return body;
}

def needs_repair(feedback: str) -> bool {
    return feedback.startswith("ERROR") or feedback.startswith("OK: 0 rows");
}

def sorted_view(rows: list) -> list[str] {
    view: list[str] = [];
    for row in rows {
        view.append(str(row));
    }
    view.sort();
    return view;
}

# Execution match against the gold query: MATCH, MISMATCH, ERROR, UNGRADED.
# Order matters only when the gold query orders its rows, which is the usual
# execution-match rule for Spider-derived tasks.
def grade(sql: str) -> str {
    if not GOLD_SQL {
        return "UNGRADED";
    }
    try {
        rows = run_query(clean_sql(sql));
    } except Exception {
        return "ERROR";
    }
    gold_rows = run_query(GOLD_SQL);
    if "order by" in GOLD_SQL.lower() {
        if sorted_view(rows) != sorted_view(gold_rows) {
            return "MISMATCH";
        }
        if str(rows) != str(gold_rows) {
            return "MISMATCH";
        }
        return "MATCH";
    }
    if sorted_view(rows) != sorted_view(gold_rows) {
        return "MISMATCH";
    }
    return "MATCH";
}


# ------------------------------------------------------------ inspection tools

def show_schema(table: str) -> str {
    time.sleep(0.20 * TOOL_SCALE);
    want = table.strip().lower();
    if not want or want == "all" {
        return SCHEMA;
    }
    for row in DB.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND lower(name) = ?", (want, )
    ).fetchall() {
        return str(row[0]) + ";";
    }
    return f"No table named '{table}'. Tables: " + ", ".join(table_names());
}

def sample_rows(table: str) -> str {
    time.sleep(0.25 * TOOL_SCALE);
    want = table.strip().lower();
    name = "";
    for candidate in table_names() {
        if candidate.lower() == want {
            name = candidate;
        }
    }
    if not name {
        return f"No table named '{table}'. Tables: " + ", ".join(table_names());
    }
    cursor = DB.execute("SELECT * FROM " + name + " LIMIT 3");
    rows = cursor.fetchall();
    columns: list[str] = [];
    for description in cursor.description {
        columns.append(str(description[0]));
    }
    lines: list[str] = [name + "(" + ", ".join(columns) + ")"];
    for row in rows {
        lines.append(str(row));
    }
    return "\n".join(lines);
}

sem show_schema = "Show the CREATE TABLE statement for one table, or for every table when the argument is empty.";
sem show_schema.table = "the table name, or an empty string for the whole schema";
sem sample_rows = "Show the column names and the first three rows of one table, to check the values a filter has to match.";
sem sample_rows.table = "the table name to sample";


# ------------------------------------------------------------- agent reasoning

def generate_sql(question: str, schema: str) -> str by llm(
    tools=[show_schema, sample_rows],
    temperature=0.0,
    max_tokens=256,
    max_react_iterations=4
);

def diagnose(question: str, sql: str, feedback: str) -> str by llm(
    tools=[show_schema, sample_rows],
    temperature=0.0,
    max_tokens=224,
    max_react_iterations=3
);

def revise_sql(question: str, schema: str, sql: str, diagnosis: str) -> str by llm(
    temperature=0.0,
    max_tokens=256
);

def finalize(question: str, sql: str, result: str) -> str by llm(
    temperature=0.0,
    max_tokens=160
);

sem generate_sql = "Write one SQLite query that answers the question over this schema. Inspect a table when the column names or the stored values are not obvious from the schema. Return only the query.";
sem generate_sql.question = "the question to answer in SQL";
sem generate_sql.schema = "the CREATE TABLE statements of the database";

sem diagnose = "Explain in one or two sentences why the query did not work: name the wrong table, column, join or filter, and check the schema or the stored values when the message alone does not say. Do not write the corrected query here.";
sem diagnose.question = "the question the query is supposed to answer";
sem diagnose.sql = "the query that was just executed";
sem diagnose.feedback = "what the database returned: an error message, or the rows the query produced";

sem revise_sql = "Rewrite the query so it fixes exactly what the diagnosis names, keeping everything that already worked. Return only the corrected query.";
sem revise_sql.question = "the question to answer in SQL";
sem revise_sql.schema = "the CREATE TABLE statements of the database";
sem revise_sql.sql = "the query that failed";
sem revise_sql.diagnosis = "what the diagnosis says is wrong with it";

sem finalize = "Answer the question in one sentence from the rows the query returned, quoting the values.";
sem finalize.question = "the original question";
sem finalize.sql = "the query that produced the result";
sem finalize.result = "the rows the database returned";


walker SqlAgent {
    has question: str = "";
    has schema: str = "";
    has sql: str = "";
    has feedback: str = "";
    has diagnosis: str = "";
    has answer: str = "";
    has turns: int = 0;

    # The repair loop is a loop in the walker, so its callsites repeat with new
    # arguments and `revise_sql -> diagnose` is a back edge in the topology. The
    # query lives in walker state on purpose: two byLLM calls write it, which is
    # what lets the fresh query rebind into the downstream prompts each turn.
    can solve with Root entry {
        self.sql = generate_sql(self.question, self.schema);
        self.feedback = execute_sql(self.sql);

        while (needs_repair(self.feedback) or self.turns < FORCE_TURNS) and self.turns < MAX_TURNS {
            self.turns += 1;
            time.sleep(0.15 * TOOL_SCALE);
            self.diagnosis = diagnose(self.question, self.sql, self.feedback);
            time.sleep(0.15 * TOOL_SCALE);
            self.sql = revise_sql(self.question, self.schema, self.sql, self.diagnosis);
            self.feedback = execute_sql(self.sql);
        }

        time.sleep(0.15 * TOOL_SCALE);
        self.answer = finalize(self.question, self.sql, self.feedback);
    }

    can report with exit {
        print("SQL:", clean_sql(self.sql));
        print("RESULT:", self.feedback);
        print("ANSWER:", self.answer);
        print(f"VERDICT: task={TASK_ID} hardness={HARDNESS} turns={self.turns} match={grade(self.sql)}");
    }
}


with entry:__main__ {
    print("TASK:", TASK_ID, "| db:", DB_NAME, "| hardness:", HARDNESS);
    print("QUESTION:", QUESTION);
    SqlAgent(question=QUESTION, schema=SCHEMA) spawn root;
}
