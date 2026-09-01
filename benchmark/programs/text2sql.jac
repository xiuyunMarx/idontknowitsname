# Routed Text2SQL over three domain analysts. Each analyst carries its database
# context (schema + sample rows + data dictionary, ~2.2k tokens) as the FIRST
# argument of its SQL calls: a KV prefix shared by every session routed to that
# analyst, and byte-known to the server before the routed call arrives because it
# is a per-callsite constant. A normalize call cleans the question first (so the
# routing visit is not the opening request and its prompt is rebuildable from the
# normalize reply), a visit-by hop picks the analyst whose tables fit the question,
# and a data-dependent repair loop runs on that node — only a real SQLite error
# takes the revise branch, so the branch probability the server learns comes from
# model behavior, not from the driver.
# Call chain seen by the server:
#   Text2Sql.normalize_question -> [visit root -> <X>Analyst, select=1]
#     -> <X>Analyst.gen_sql -> (<X>Analyst.revise_sql × 0..3, on real errors)
#     -> <X>Analyst.summarize_result
# Gaps model remote tool latency, seeded per session so paired runs replay them:
#   PREP     0.5-1.0 s  ticket write after normalization, before routing
#   INSPECT  1.5-3.0 s  warehouse execution + result validation after each run
import os;
import sys;
import time;
import random;
import sqlite3;
import from jaclang.byllm.llm { InterceptorLLM }
glob llm = InterceptorLLM(model_name="Qwen/Qwen3-8B", call_params={"temperature": 0.0});

glob MAX_REPAIRS = 3;
glob CONTEXT: dict[str, str] = {};   # analyst name -> database context, loaded once per run


def load_context(db_path: str, dict_path: str) -> str {
    conn = sqlite3.connect(db_path);
    parts = ["== Schema and sample rows =="];
    for r in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall() {
        name: str = str(r[0]);
        sql: str = str(r[1]);
        rows: list = conn.execute("SELECT * FROM " + name + " LIMIT 5").fetchall();
        parts.append(sql + ";");
        parts.append("-- sample rows of " + name + ": " + repr(rows));
    }
    conn.close();
    f = open(dict_path, "r");
    dictionary = f.read();
    f.close();
    return "\n".join(parts) + "\n\n" + dictionary;
}


node Analyst{
    has name: str;
    has tables: str;
    has db_path: str;

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

    def gen_sql(context: str, question: str) -> str by llm();
    def revise_sql(context: str, question: str, sql: str, error: str) -> str by llm();
    def summarize_result(question: str, rows: str) -> str by llm();

    can serve with Text2Sql entry {
        rng = random.Random(visitor.raw_question);
        context = CONTEXT[self.name];
        sql = self.gen_sql(context, visitor.question);
        result = "";
        repairs = 0;
        while repairs <= MAX_REPAIRS {
            result = self.run_sql(sql);
            time.sleep(rng.uniform(1.5, 3.0));
            if not result.startswith("SQL error") {
                break;
            }
            repairs += 1;
            if repairs > MAX_REPAIRS {
                break;
            }
            sql = self.revise_sql(context, visitor.question, sql, result);
        }
        visitor.answer = self.summarize_result(visitor.question, result);
    }
}
sem Analyst.tables = "The tables this analyst can query; a question belongs here only when the tables it needs are listed.";
sem Analyst.gen_sql = "Write one SQLite query answering the question against the schema in the context. Return only the SQL.";
sem Analyst.revise_sql = "The query failed with the given error. Return only the corrected SQLite query for the schema in the context.";
sem Analyst.summarize_result = "Answer the question in one sentence from the returned rows.";
sem Analyst.serve = "answers one analytics question against this database";

# One node type per database: the callsite key carries the concrete class, so each
# analyst has its own gen_sql callsite whose first argument is a constant.
node ShopAnalyst(Analyst){}
node LibraryAnalyst(Analyst){}
node ClinicAnalyst(Analyst){}
sem ShopAnalyst = "Analyst for the online shop database: customers, products, orders and order items";
sem LibraryAnalyst = "Analyst for the lending library database: books, members and loans";
sem ClinicAnalyst = "Analyst for the outpatient clinic database: doctors, patients and appointments";


walker Text2Sql{
    has question: str;
    has raw_question: str = "";
    has answer: str = "";

    def normalize_question(question: str) -> str by llm();

    can start with Root entry {
        self.raw_question = self.question;
        rng = random.Random(self.raw_question);
        self.question = self.normalize_question(self.question);
        time.sleep(rng.uniform(0.5, 1.0));
        visit [-->] by llm(select=1, intent="Route the question to the analyst whose tables contain what it asks about");
    }

    can wrap_up with exit {
        print(self.answer if self.answer else "(no answer produced)");
    }
}
sem Text2Sql.question = "The analytics question, normalized.";
sem Text2Sql.raw_question = "The question as the user typed it.";
sem Text2Sql.answer = "The final one-sentence answer.";
sem Text2Sql.normalize_question = "Rewrite the question as one clear, self-contained analytics request; keep the meaning, return only the rewrite.";

with entry {
    default_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "sql");
    db_dir = os.environ.get("T2S_DB_DIR") or default_dir;
    question = os.environ.get("T2S_QUESTION")
        or (sys.argv[1] if len(sys.argv) > 1 else
            "Which three customers spent the most in total, and how much did each spend?");

    CONTEXT["shop"] = load_context(os.path.join(db_dir, "shop.db"), os.path.join(db_dir, "shop.md"));
    CONTEXT["library"] = load_context(os.path.join(db_dir, "library.db"), os.path.join(db_dir, "library.md"));
    CONTEXT["clinic"] = load_context(os.path.join(db_dir, "clinic.db"), os.path.join(db_dir, "clinic.md"));
    root ++> ShopAnalyst(name="shop", tables="customers, products, orders, order_items",
                         db_path=os.path.join(db_dir, "shop.db"));
    root ++> LibraryAnalyst(name="library", tables="books, members, loans",
                            db_path=os.path.join(db_dir, "library.db"));
    root ++> ClinicAnalyst(name="clinic", tables="doctors, patients, appointments",
                           db_path=os.path.join(db_dir, "clinic.db"));
    Text2Sql(question=question) spawn root;
}
