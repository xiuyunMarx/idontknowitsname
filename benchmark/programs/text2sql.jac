# Routed Text2SQL: three Database nodes in distinct domains hang off root; a
# normalize call cleans the question first (so the routing visit is not the
# session's opening request and its prompt is rebuildable from the normalize
# reply), then a visit-by hop selects the database whose schema is consistent
# with the question, and a data-dependent repair loop runs on that node — only
# an actual execution error takes the revise branch, so the branch probability
# the server learns comes from real model behavior, not from the driver.
# Call chain seen by the server:
#   Text2Sql.normalize_question -> [visit root -> Database, select=1]
#     -> Database.gen_sql -> (Database.revise_sql × 0..3, on real errors)
#     -> Database.summarize_result
# Dynamic bytes: the question (resp rule off normalize_question), the failing
# SQL (resp of the previous call), and each schema — which rides in its node's
# `self` zone as a per-database constant, doubling as the routing signal.
import os;
import sys;
import time;
import sqlite3;
import from jaclang.byllm.llm { InterceptorLLM }
glob llm = InterceptorLLM(model_name="Qwen/Qwen3-8B", call_params={"temperature": 0.0});

glob MAX_REPAIRS = 3,
     INSPECT_S = 0.15,   # result validation pause after each execution: the JIT window
     PREP_S = 0.3;       # ticket write after normalization, before routing: the probe window

def load_schema(db_path: str) -> str {
    conn = sqlite3.connect(db_path);
    rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall();
    conn.close();
    return ";\n".join([r[0] for r in rows]) + ";";
}

node Database{
    has name: str;
    has db_path: str;
    has schema: str = "";

    def run_sql(sql: str) -> str{
        conn = sqlite3.connect(self.db_path);
        try {
            return repr(conn.execute(sql).fetchmany(20));
        } except Exception as e {
            return f"SQL error: {e}";
        } finally {
            conn.close();
        }
    }

    def gen_sql(question: str) -> str by llm();
    def revise_sql(question: str, sql: str, error: str) -> str by llm();
    def summarize_result(question: str, rows: str) -> str by llm();

    can serve with Text2Sql entry {
        sql = self.gen_sql(visitor.question);
        result = "";
        repairs = 0;
        while repairs <= MAX_REPAIRS {
            result = self.run_sql(sql);
            time.sleep(INSPECT_S);
            if not result.startswith("SQL error") {
                break;
            }
            repairs += 1;
            if repairs > MAX_REPAIRS {
                break;
            }
            sql = self.revise_sql(visitor.question, sql, result);
        }
        visitor.answer = self.summarize_result(visitor.question, result);
    }
}
sem Database = "One SQLite database and the analyst that answers questions about it; a question belongs here only when the tables it needs appear in the schema.";
sem Database.schema = "CREATE TABLE statements of this database.";
sem Database.gen_sql = "Write one SQLite query answering the question against the schema in self. Return only the SQL.";
sem Database.revise_sql = "The query failed with the given error. Return only the corrected SQLite query for the schema in self.";
sem Database.summarize_result = "Answer the question in one sentence from the returned rows.";
sem Database.serve = "answers one analytics question against this database";

walker Text2Sql{
    has question: str;
    has answer: str = "";

    def normalize_question(question: str) -> str by llm();

    can start with Root entry {
        self.question = self.normalize_question(self.question);
        time.sleep(PREP_S);
        visit [-->] by llm(select=1, intent="Route the question to the database whose schema contains the tables it asks about");
    }

    can wrap_up with exit {
        print(self.answer if self.answer else "(no answer produced)");
    }
}
sem Text2Sql.question = "The analytics question, verbatim.";
sem Text2Sql.answer = "The final one-sentence answer.";
sem Text2Sql.normalize_question = "Rewrite the question as one clear, self-contained analytics request; keep the meaning, return only the rewrite.";

with entry {
    default_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "sql");
    db_dir = os.environ.get("T2S_DB_DIR") or default_dir;
    question = os.environ.get("T2S_QUESTION")
        or (sys.argv[1] if len(sys.argv) > 1 else
            "Which three customers spent the most in total, and how much did each spend?");

    for fname in sorted(os.listdir(db_dir)) {
        if fname.endswith(".db") {
            path = os.path.join(db_dir, fname);
            root ++> Database(name=fname[:-3], db_path=path, schema=load_schema(path));
        }
    }
    Text2Sql(question=question) spawn root;
}
