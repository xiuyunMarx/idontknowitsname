"""Driver for one benchmark case (rag_qa-018) against the guard server."""

import os;
import time;
import from main { interact }

with entry {
    sid = os.environ.get("CASE_SID", f"exp-rag-018-{int(time.time())}");
    result = root spawn interact(message="How are task dependencies represented between Task nodes in this example, and how does the DependsOn edge affect task readiness?", session_id=sid);
    payload = result.reports[0] if result.reports else {};
    print("AGENT:", payload.get("agent"));
    print("RESPONSE:", payload.get("response"));
}
