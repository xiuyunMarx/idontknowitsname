"""Generate the math_pipeline question pool: deterministic GSM8K-style word
problems with computable answers (kept alongside for later accuracy checks).

    python benchmark/datasets/gen_math.py [n] [out_dir]
"""
import os
import random
import sys

TEMPLATES = [
    ("A warehouse ships {a} boxes of {b} items each; {c} items are returned. "
     "How many items stay shipped?", lambda a, b, c: a * b - c),
    ("A bakery makes {a} trays of {b} rolls every morning and sells {c} rolls by noon. "
     "How many rolls are left at noon?", lambda a, b, c: a * b - c),
    ("Tickets cost {a} dollars each. A group buys {b} tickets and uses a {c} dollar voucher. "
     "What do they pay in total?", lambda a, b, c: a * b - c),
    ("A tank holds {a} liters and drains {b} liters per hour. "
     "How many liters remain after {c} hours?", lambda a, b, c: a - b * c),
    ("A courier drives {a} km per trip and makes {b} trips a day. "
     "How far does the courier drive in {c} days?", lambda a, b, c: a * b * c),
    ("A class of {a} students splits into teams of {b}; the rest join the {c} referees. "
     "How many people are referees or unteamed?", lambda a, b, c: a % b + c),
]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "math")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(7)
    q_path = os.path.join(out_dir, "questions.txt")
    a_path = os.path.join(out_dir, "answers.txt")
    with open(q_path, "w") as fq, open(a_path, "w") as fa:
        for i in range(n):
            text, fn = TEMPLATES[i % len(TEMPLATES)]
            a, b, c = rng.randint(7, 60), rng.randint(3, 24), rng.randint(2, 30)
            fq.write(text.format(a=a, b=b, c=c) + "\n")
            fa.write(str(fn(a, b, c)) + "\n")
    print(f"{q_path}: {n} questions (answers in {a_path})")
