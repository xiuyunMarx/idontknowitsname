"""Text-to-SQL with an execution-feedback repair loop, three models alternating.

    generate_sql[2.5-3B, tools] -> execute -> (clean -> finalize[0.5B])
                                          `-> diagnose[Qwen3-1.7B, tools] -> revise_sql[2.5-3B] -> execute ...

Per-request model sequence: C (G C)* E, so the coder and the critic ping-pong
while a request is being repaired; the loop is decided by SQLite feedback only.
    $SQL_MAX_TURNS   (default 3) cap on repair turns
    $SQL_FORCE_TURNS (default 0) minimum repair turns (fixes the sequence across tasks)
Databases (Spider dev, SQLite) and tasks are inline. Inputs: $SQL_TASK_INDEX, or
$SQL_QUESTION plus $SQL_DB."""

import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");
glob PORT = int(os.environ.get("SERVE_PORT") or "8964");

glob coder = InterceptorLLM(model_name="Qwen/Qwen2.5-3B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="text2sql");
glob critic = InterceptorLLM(model_name="Qwen/Qwen3-1.7B", comm_ip="localhost", comm_port=PORT, program_name="text2sql");
glob writer = InterceptorLLM(model_name="Qwen/Qwen2.5-0.5B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="text2sql");

import json;
import sqlite3;

glob MAX_TURNS = int(os.environ.get("SQL_MAX_TURNS") or "3");
glob FORCE_TURNS = int(os.environ.get("SQL_FORCE_TURNS") or "0");

glob SETUP: dict[str, str] = {
    "network_1": "CREATE TABLE \"Friend\" (\n  \"student_id\" int NOT NULL,\n  \"friend_id\" int NOT NULL,\n  PRIMARY KEY (\"student_id\",\"friend_id\"),\n  CONSTRAINT \"Friend_FK_0_0\" FOREIGN KEY (\"friend_id\") REFERENCES \"Highschooler\" (\"ID\"),\n  CONSTRAINT \"Friend_FK_1_0\" FOREIGN KEY (\"student_id\") REFERENCES \"Highschooler\" (\"ID\")\n)\nINSERT INTO \"Friend\" VALUES (1661,1025),(1468,1101),(1381,1247),(1709,1247),(1782,1304),(1934,1304),(1782,1316),(1510,1381),(1782,1468),(1247,1501),(1911,1501),(1101,1641),(1468,1641),(1304,1661),(1510,1689),(1689,1709),(1689,1782),(1247,1911),(1316,1934),(1501,1934)\nCREATE TABLE \"Highschooler\" (\n  \"ID\" int NOT NULL,\n  \"name\" text,\n  \"grade\" int DEFAULT NULL,\n  PRIMARY KEY (\"ID\")\n)\nINSERT INTO \"Highschooler\" VALUES (1025,'John',12),(1101,'Haley',10),(1247,'Alexis',11),(1304,'Jordan',12),(1316,'Austin',11),(1381,'Tiffany',9),(1468,'Kris',10),(1501,'Jessica',11),(1510,'Jordan',9),(1641,'Brittany',10),(1661,'Logan',12),(1689,'Gabriel',9),(1709,'Cassandra',9),(1782,'Andrew',10),(1911,'Gabriel',11),(1934,'Kyle',12)\nCREATE TABLE \"Likes\" (\n  \"student_id\" int NOT NULL,\n  \"liked_id\" int NOT NULL,\n  PRIMARY KEY (\"student_id\",\"liked_id\"),\n  CONSTRAINT \"Likes_FK_0_0\" FOREIGN KEY (\"student_id\") REFERENCES \"Highschooler\" (\"ID\"),\n  CONSTRAINT \"Likes_FK_1_0\" FOREIGN KEY (\"liked_id\") REFERENCES \"Highschooler\" (\"ID\")\n)\nINSERT INTO \"Likes\" VALUES (1025,1101),(1911,1247),(1316,1304),(1247,1468),(1641,1468),(1934,1501),(1709,1689),(1689,1709),(1782,1709),(1501,1934)",
    "poker_player": "CREATE TABLE \"people\" (\n  \"People_ID\" int NOT NULL,\n  \"Nationality\" text,\n  \"Name\" text,\n  \"Birth_Date\" text,\n  \"Height\" double DEFAULT NULL,\n  PRIMARY KEY (\"People_ID\")\n)\nINSERT INTO \"people\" VALUES (1,'Russia','Aleksey Ostapenko','May 26, 1986',207),(2,'Bulgaria','Teodor Salparov','August 16, 1982',182),(3,'Russia','Roman Bragin','April 17, 1987',187),(4,'Russia','Sergey Grankin','January 22, 1987',193),(5,'Russia','Yevgeni Sivozhelez','August 8, 1986',196),(6,'Russia','Maksim Botin','July 14, 1983',194),(7,'Russia','Semen Poltavskiy','February 8, 1981',205)\nCREATE TABLE \"poker_player\" (\n  \"Poker_Player_ID\" int NOT NULL,\n  \"People_ID\" int DEFAULT NULL,\n  \"Final_Table_Made\" double DEFAULT NULL,\n  \"Best_Finish\" double DEFAULT NULL,\n  \"Money_Rank\" double DEFAULT NULL,\n  \"Earnings\" double DEFAULT NULL,\n  PRIMARY KEY (\"Poker_Player_ID\"),\n  CONSTRAINT \"poker_player_FK_0_0\" FOREIGN KEY (\"People_ID\") REFERENCES \"people\" (\"People_ID\")\n)\nINSERT INTO \"poker_player\" VALUES (1,1,42,1,68,476090),(2,2,10,2,141,189233),(3,5,21,1,166,104871),(4,6,19,2,58,596462),(5,7,26,3,154,142800)",
    "museum_visit": "CREATE TABLE \"museum\" (\n  \"Museum_ID\" int NOT NULL,\n  \"Name\" text,\n  \"Num_of_Staff\" int DEFAULT NULL,\n  \"Open_Year\" text,\n  PRIMARY KEY (\"Museum_ID\")\n)\nINSERT INTO \"museum\" VALUES (1,'Plaza Museum',62,'2000'),(2,'Capital Plaza Museum',25,'2012'),(3,'Jefferson Development Museum',18,'2010'),(4,'Willow Grande Museum',17,'2011'),(5,'RiverPark Museum',16,'2008'),(6,'Place Tower Museum',16,'2008'),(7,'Central City District Residential Museum',15,'2010'),(8,'ZirMed Gateway Museum',12,'2009')\nCREATE TABLE \"visit\" (\n  \"Museum_ID\" int NOT NULL,\n  \"visitor_ID\" int NOT NULL,\n  \"Num_of_Ticket\" int DEFAULT NULL,\n  \"Total_spent\" double DEFAULT NULL,\n  PRIMARY KEY (\"Museum_ID\",\"visitor_ID\"),\n  CONSTRAINT \"visit_FK_0_0\" FOREIGN KEY (\"visitor_ID\") REFERENCES \"visitor\" (\"ID\"),\n  CONSTRAINT \"visit_FK_1_0\" FOREIGN KEY (\"Museum_ID\") REFERENCES \"museum\" (\"Museum_ID\")\n)\nINSERT INTO \"visit\" VALUES (1,5,20,320.14),(2,3,24,209.98),(2,5,4,89.98),(4,3,10,320.44),(4,6,3,20.44),(8,6,2,19.98)\nCREATE TABLE \"visitor\" (\n  \"ID\" int NOT NULL,\n  \"Name\" text,\n  \"Level_of_membership\" int DEFAULT NULL,\n  \"Age\" int DEFAULT NULL,\n  PRIMARY KEY (\"ID\")\n)\nINSERT INTO \"visitor\" VALUES (1,'Gonzalo Higuaín ',8,35),(2,'Guti Midfielder',5,28),(3,'Arjen Robben',1,27),(4,'Raúl Brown',2,56),(5,'Fernando Gago',6,36),(6,'Rafael van der Vaart',1,25)",
    "concert_singer": "CREATE TABLE \"concert\" (\n  \"concert_ID\" int NOT NULL,\n  \"concert_Name\" text,\n  \"Theme\" text,\n  \"Stadium_ID\" int DEFAULT NULL,\n  \"Year\" text,\n  PRIMARY KEY (\"concert_ID\"),\n  CONSTRAINT \"concert_FK_0_0\" FOREIGN KEY (\"Stadium_ID\") REFERENCES \"stadium\" (\"Stadium_ID\")\n)\nINSERT INTO \"concert\" VALUES (1,'Auditions','Free choice',1,'2014'),(2,'Super bootcamp','Free choice 2',2,'2014'),(3,'Home Visits','Bleeding Love',2,'2015'),(4,'Week 1','Wide Awake',10,'2014'),(5,'Week 1','Happy Tonight',9,'2015'),(6,'Week 2','Party All Night',7,'2015')\nCREATE TABLE \"singer\" (\n  \"Singer_ID\" int NOT NULL,\n  \"Name\" text,\n  \"Country\" text,\n  \"Song_Name\" text,\n  \"Song_release_year\" text,\n  \"Age\" int DEFAULT NULL,\n  \"Is_male\" varchar(255) DEFAULT NULL,\n  PRIMARY KEY (\"Singer_ID\")\n)\nINSERT INTO \"singer\" VALUES (1,'Joe Sharp','Netherlands','You','1992',52,'F'),(2,'Timbaland','United States','Dangerous','2008',32,'T'),(3,'Justin Brown','France','Hey Oh','2013',29,'T'),(4,'Rose White','France','Sun','2003',41,'F'),(5,'John Nizinik','France','Gentleman','2014',43,'T'),(6,'Tribal King','France','Love','2016',25,'T')\nCREATE TABLE \"singer_in_concert\" (\n  \"concert_ID\" int NOT NULL,\n  \"Singer_ID\" int NOT NULL,\n  PRIMARY KEY (\"concert_ID\",\"Singer_ID\"),\n  CONSTRAINT \"singer_in_concert_FK_0_0\" FOREIGN KEY (\"Singer_ID\") REFERENCES \"singer\" (\"Singer_ID\"),\n  CONSTRAINT \"singer_in_concert_FK_1_0\" FOREIGN KEY (\"concert_ID\") REFERENCES \"concert\" (\"concert_ID\")\n)\nINSERT INTO \"singer_in_concert\" VALUES (1,2),(6,2),(1,3),(2,3),(5,3),(4,4),(1,5),(3,5),(2,6),(5,6)\nCREATE TABLE \"stadium\" (\n  \"Stadium_ID\" int NOT NULL,\n  \"Location\" text,\n  \"Name\" text,\n  \"Capacity\" int DEFAULT NULL,\n  \"Highest\" int DEFAULT NULL,\n  \"Lowest\" int DEFAULT NULL,\n  \"Average\" int DEFAULT NULL,\n  PRIMARY KEY (\"Stadium_ID\")\n)\nINSERT INTO \"stadium\" VALUES (1,'Raith Rovers','Stark''s Park',10104,4812,1294,2106),(2,'Ayr United','Somerset Park',11998,2363,1057,1477),(3,'East Fife','Bayview Stadium',2000,1980,533,864),(4,'Queen''s Park','Hampden Park',52500,1763,466,730),(5,'Stirling Albion','Forthbank Stadium',3808,1125,404,642),(6,'Arbroath','Gayfield Park',4125,921,411,638),(7,'Alloa Athletic','Recreation Park',3100,1057,331,637),(9,'Peterhead','Balmoor',4000,837,400,615),(10,'Brechin City','Glebe Park',3960,780,315,552)",
    "cre_Doc_Template_Mgt": "CREATE TABLE \"Ref_Template_Types\" (\n  \"Template_Type_Code\" char(15) NOT NULL,\n  \"Template_Type_Description\" varchar(255) NOT NULL,\n  PRIMARY KEY (\"Template_Type_Code\")\n)\nINSERT INTO \"Ref_Template_Types\" VALUES ('AD','Advertisement'),('BK','Book'),('CV','CV'),('PP','Paper'),('PPT','Presentation')\nCREATE TABLE \"Templates\" (\n  \"Template_ID\" int NOT NULL,\n  \"Version_Number\" int NOT NULL,\n  \"Template_Type_Code\" char(15) NOT NULL,\n  \"Date_Effective_From\" datetime DEFAULT NULL,\n  \"Date_Effective_To\" datetime DEFAULT NULL,\n  \"Template_Details\" varchar(255) NOT NULL,\n  PRIMARY KEY (\"Template_ID\"),\n  CONSTRAINT \"Templates_FK_0_0\" FOREIGN KEY (\"Template_Type_Code\") REFERENCES \"Ref_Template_Types\" (\"Template_Type_Code\")\n)\nINSERT INTO \"Templates\" VALUES (1,5,'PP','2005-11-12 07:09:48','2008-01-05 14:19:28',''),(4,4,'BK','2002-03-02 14:39:49','2001-04-18 09:29:52',''),(6,2,'PPT','1975-05-20 22:51:19','1992-05-02 20:06:11',''),(7,8,'PPT','1993-10-07 02:33:04','1975-07-16 04:52:10',''),(8,3,'BK','1983-07-16 21:16:16','1976-10-28 10:08:50',''),(9,2,'BK','1997-04-17 08:29:44','1994-12-07 13:26:23',''),(10,1,'PPT','2003-06-05 04:03:45','2007-06-06 06:18:53',''),(11,6,'BK','1996-02-04 11:27:24','1995-09-19 22:27:48',''),(14,7,'AD','1975-10-20 02:28:58','1979-11-04 08:58:39',''),(15,9,'CV','1986-12-09 14:51:36','1993-03-24 14:30:23',''),(16,5,'CV','2012-04-05 07:11:42','1980-05-07 12:15:47',''),(18,5,'PP','1984-08-07 13:36:26','1998-05-12 12:51:29',''),(19,7,'AD','1999-06-21 11:10:30','1974-09-14 06:34:39',''),(20,6,'BK','1986-11-14 12:20:18','2008-08-08 18:36:43',''),(21,9,'AD','2002-08-25 13:26:23','2015-09-06 01:08:44',''),(22,0,'PP','2005-02-20 00:31:34','1989-11-24 19:06:06',''),(23,2,'BK','1979-12-24 10:28:16','2000-10-22 11:57:12',''),(24,8,'PP','2008-08-01 13:57:26','1973-01-12 14:13:34',''),(25,5,'PP','1979-10-20 21:23:20','2006-02-06 23:52:04','')\nCREATE TABLE \"Documents\" (\n  \"Document_ID\" int NOT NULL,\n  \"Template_ID\" int DEFAULT NULL,\n  \"Document_Name\" varchar(255) DEFAULT NULL,\n  \"Document_Description\" varchar(255) DEFAULT NULL,\n  \"Other_Details\" varchar(255) DEFAULT NULL,\n  PRIMARY KEY (\"Document_ID\"),\n  CONSTRAINT \"Documents_FK_0_0\" FOREIGN KEY (\"Template_ID\") REFERENCES \"Templates\" (\"Template_ID\")\n)\nINSERT INTO \"Documents\" VALUES (1,7,'Introduction of OS','n',NULL),(3,6,'Summer Show','u',NULL),(76,20,'Robbin CV','y',NULL),(80,14,'Welcome to NY','h',NULL),(82,11,'Data base','w',NULL),(2394,10,'Customer reviews','y',NULL),(3830,14,'Do not panic','k',NULL),(33930,1,'How Google people work','z',NULL),(50123,22,'Learning French','r',NULL),(651512,21,'How to write a CV','f',NULL),(801801,4,'How to read a book','w',NULL),(3540024,8,'Palm reading','y',NULL),(16514113,25,'A history of Arts','h',NULL),(385906526,11,'About Korea','b',NULL)\nCREATE TABLE \"Paragraphs\" (\n  \"Paragraph_ID\" int NOT NULL,\n  \"Document_ID\" int NOT NULL,\n  \"Paragraph_Text\" varchar(255) DEFAULT NULL,\n  \"Other_Details\" varchar(255) DEFAULT NULL,\n  PRIMARY KEY (\"Paragraph_ID\"),\n  CONSTRAINT \"Paragraphs_FK_0_0\" FOREIGN KEY (\"Document_ID\") REFERENCES \"Documents\" (\"Document_ID\")\n)\nINSERT INTO \"Paragraphs\" VALUES (7,2394,'Korea',NULL),(9,3,'Somalia',NULL),(65,50123,'Palestinian Territory',NULL),(241,651512,'Jersey',NULL),(3708,33930,'UK',NULL),(9946,385906526,'Fiji',NULL),(16615,80,'Japan',NULL),(1104059,3830,'Senegal',NULL),(243399026,651512,'Indonesia',NULL),(280120913,2394,'Ukraine',NULL),(510442723,2394,'Korea',NULL),(571212310,16514113,'Brazil',NULL),(608931827,80,'Micronesia',NULL),(765713812,16514113,'Ireland',NULL),(946335436,3540024,'Papua New Guinea',NULL)"
};
glob TASK_DB: list[str] = [
    "cre_Doc_Template_Mgt",
    "cre_Doc_Template_Mgt",
    "poker_player",
    "concert_singer",
    "poker_player",
    "cre_Doc_Template_Mgt",
    "museum_visit",
    "network_1",
    "network_1",
    "network_1",
    "concert_singer"
];
glob TASK_QUESTION: list[str] = [
    "Return the type code of the template type with the description \"Book\".",
    "How many templates do we have?",
    "What are the names of people who are not from Russia?",
    "What is the average, minimum, and maximum age for all French singers?",
    "Give the names of poker players who have earnings above 300000.",
    "What are the ids for templates that are not used in any documents?",
    "find the names of museums which have more staff than the minimum staff number of all museums opened after 2010.",
    "Show ids of all students who do not have any friends.",
    "Find the average grade of all students who have some friends.",
    "Show the names of students who have a grade higher than 5 and have at least 2 friends.",
    "What is the name and capacity of the stadium with the most concerts after 2013 ?"
];
glob TASK_GOLD: list[str] = [
    "SELECT template_type_code FROM Ref_template_types WHERE template_type_description = \"Book\"",
    "SELECT count(*) FROM Templates",
    "SELECT Name FROM people WHERE Nationality != \"Russia\"",
    "SELECT avg(age) , min(age) , max(age) FROM singer WHERE country = 'France'",
    "SELECT T1.Name FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID WHERE T2.Earnings > 300000",
    "SELECT template_id FROM Templates EXCEPT SELECT template_id FROM Documents",
    "SELECT name FROM museum WHERE num_of_staff > (SELECT min(num_of_staff) FROM museum WHERE open_year > 2010)",
    "SELECT id FROM Highschooler EXCEPT SELECT student_id FROM Friend",
    "SELECT avg(grade) FROM Highschooler WHERE id IN (SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id)",
    "SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id = T2.id WHERE T2.grade > 5 GROUP BY T1.student_id HAVING count(*) >= 2",
    "select t2.name , t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id where t1.year > 2013 group by t2.stadium_id order by count(*) desc limit 1"
];
glob TASK_HARDNESS: list[str] = [
    "easy",
    "easy",
    "easy",
    "medium",
    "medium",
    "hard",
    "hard",
    "hard",
    "hard",
    "hard",
    "extra"
];

def task_index() -> int {
    return int(os.environ.get("SQL_TASK_INDEX") or "0") % len(TASK_QUESTION);
}

glob DB_NAME: str = str(os.environ.get("SQL_DB") or TASK_DB[task_index()]);
glob QUESTION: str = str(os.environ.get("SQL_QUESTION") or TASK_QUESTION[task_index()]);
glob HARDNESS: str = TASK_HARDNESS[task_index()];

def load_gold() -> str {
    if os.environ.get("SQL_QUESTION") {
        return "";
    }
    return TASK_GOLD[task_index()];
}

def open_db() -> any {
    conn = sqlite3.connect(":memory:");
    if DB_NAME in SETUP {
        conn.executescript(SETUP[DB_NAME]);
        conn.commit();
    } else {
        print(f"[text2sql] unknown database: {DB_NAME}");
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

def clean_sql(text: str) -> str {
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
    if sorted_view(rows) != sorted_view(gold_rows) {
        return "MISMATCH";
    }
    if "order by" in GOLD_SQL.lower() and str(rows) != str(gold_rows) {
        return "MISMATCH";
    }
    return "MATCH";
}

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


def generate_sql(question: str, schema: str) -> str by coder(
    tools=[show_schema, sample_rows],
    temperature=0.0,
    max_tokens=256,
    max_react_iterations=4
);

def diagnose(question: str, sql: str, feedback: str) -> str by critic(
    tools=[show_schema, sample_rows],
    temperature=0.0,
    max_tokens=224,
    max_react_iterations=3
);

def revise_sql(question: str, schema: str, sql: str, diagnosis: str) -> str by coder(
    temperature=0.0,
    max_tokens=256
);

def finalize(question: str, sql: str, result: str) -> str by writer(
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
        print(f"VERDICT: db={DB_NAME} hardness={HARDNESS} turns={self.turns} match={grade(self.sql)}");
    }
}


with entry:__main__ {
    print("QUESTION:", QUESTION, "| db:", DB_NAME);
    SqlAgent(question=QUESTION, schema=SCHEMA) spawn root;
}
