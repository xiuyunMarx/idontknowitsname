"""Optional, line-buffered raw request events for benchmark analysis."""
import json
import os
import time


class RequestTrace:
    def __init__(self):
        path = os.environ.get("BENCH_REQUEST_LOG")
        self.file = open(path, "a", buffering=1) if path else None

    def emit(self, event, request_id, **fields):
        if self.file is not None:
            self.file.write(json.dumps({"event": event, "request_id": request_id,
                                        "time_ns": time.time_ns(),
                                        "monotonic_ns": time.monotonic_ns(),
                                        **fields}) + "\n")

    def close(self):
        if self.file is not None:
            self.file.close()
