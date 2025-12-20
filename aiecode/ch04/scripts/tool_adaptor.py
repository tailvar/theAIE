"""minimal tool adapter with timeout and retries.

No external deps beyond the standard library.

TODO(Ch6-Ex1): Add a circuit breaker to call_tool() that trips after
               N failures within a time window.
TODO(Ch6-Ex2): Validate payload fields against ToolSpec.schema before calling
               the tool and return a clear error if keys are missing.
TODO(Ch6-Ex3): Keep side effects sandboxed (file system, network). The demo
               writes only inside a TemporaryDirectory.
"""

from __future__ import annotations  # Future‑proof typing of annotations.

import threading  # Run work in a separate thread to support timeouts.
import time  # Measure latency and implement backoff.
from dataclasses import dataclass  # Lightweight data containers.
from typing import Any, Callable, Dict, Tuple, Final # Type hints for clarity.
from collections import deque

# --- circuit breaker (module-levl, per tool name) ---
CB_WINDOW_S = 60        # "in a minute"
CB_TRIP_FAILS = 3       # trip after 3 failures
CB_COOLDOWN_S = 60      # How long the circuit stays open

@dataclass
class _CircuitState:
    failure_times: Deque[float]
    open_until: float

_CB_LOCK = threading.Lock()
_CB: Dict[str, _CircuitState] = {}

# tag::ch06_adapter[]
@dataclass  # Describe how a tool should be called.
class ToolSpec:
    name: str  # Logical tool name for logs.
    schema: Dict[str, str]  # Expected payload keys → human type hints.
    timeout_ms: int = 200  # Max time per attempt in milliseconds.
    max_retries: int = 2  # Attempts after the first try.
    backoff_ms: int = 50  # Initial backoff between retries in ms.


def call_with_timeout(
    fn: Callable[..., Any], *, timeout_ms: int, **kwargs
) -> Tuple[bool, Any]:
    """Run the function in a thread and wait up to timeout_ms.

    Returns (ok, result_or_error).
    """

    result: Dict[str, Any] = {}  # Shared box for success values.
    error: Dict[str, str] = {}  # Shared box for error messages.

    def runner() -> None:
        try:
            result["value"] = fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 (demo safety)
            error["msg"] = str(exc)

    t = threading.Thread(target=runner, daemon=True)  # Run in a worker thread.
    start = time.perf_counter()  # High‑resolution timer start.
    t.start()
    t.join(timeout_ms / 1000.0)  # Wait up to the timeout.
    elapsed_ms = int((time.perf_counter() - start) * 1000)  # Total wait time.
    if t.is_alive():  # Thread still running → consider this a timeout.
        return False, (f"timeout after {elapsed_ms} ms")
    if error:
        return False, error["msg"]
    return True, result["value"]


def call_tool(
    spec: ToolSpec, *, tool: Callable[..., Dict[str, Any]], payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Call a tool with retries and a timeout; return {result,status,latency_ms}."""

    attempts = 0  # Count attempts including the first try.
    backoff = spec.backoff_ms / 1000.0  # Convert ms → seconds.
    start_total = time.perf_counter()  # Time the whole call (retries included).

    # --- Validate the payload against thespec.schema (required keys) ---
    expected_keys = list(spec.schema.keys())
    missing = [k for k in expected_keys if k not in payload]
    if missing:
        latency_ms = int((time.perf_counter() - start_total)* 1000)
        expected_pretty = ", ".join(f"{k}:{spec.schema[k]}" for k in expected_keys) or "(none)"
        missing_pretty = ", ".join(missing)
        return {
            "result": (
                f"invalid payload for tool '{spec.name}': missing required key(s): {missing_pretty}"
                f"Expected: {expected_pretty}"
            ),
            "status":"error",
            "latency_ms":latency_ms,
            "missing":missing,
            "expected": spec.schema
        }

    # --- Circuit breaker: block fast if open ---
    now = time.time()
    with _CB_LOCK:
        st = _CB.get(spec.name)
        if st is None:
            st = _CircuitState(failure_times=deque(), open_until=0.0)
            _CB[spec.name] = st

        if st.open_until > now:
            latency_ms = int((time.perf_counter() - start_total) * 1000)
            return {
                "result":f"circuit open (cooldown until {int(st.open_until)} epoch",
                "status":"error",
                "latency_ms":latency_ms,
            }

    while True:
        attempts += 1
        # TODO(Ch6-Ex2): Check payload keys against spec.schema before calling.
        ok, value = call_with_timeout(
            tool, timeout_ms=spec.timeout_ms, payload=payload
        )
        if ok:  # Success: package result with status + latency.
            latency_ms = int((time.perf_counter() - start_total) * 1000)
            return {
                "result": value.get("result", value),
                "status": "ok",
                "latency_ms": latency_ms,
            }

        # --- record a failure, maybe trip the circuit ---
        now = time.time()
        with _CB_LOCK:
            st = _CB.setdefault(spec.name, _CircuitState(failure_times=deque(), open_until=0.0))

            # keep only failures in the last minute
            cutoff = now - CB_WINDOW_S
            while st.failure_times and st.failure_times[0] < cutoff:
                st.failure_times.popleft()

            st.failure_times.append(now)

            # trip if >= 3 failures in the last minute
            if len(st.failure_times) >= CB_TRIP_FAILS:
                st.open_until = now + CB_COOLDOWN_S

        if attempts > spec.max_retries:  # Retries exhausted → return error.
            latency_ms = int((time.perf_counter() - start_total) * 1000)
            return {
                "result": str(value),
                "status": "error",
                "latency_ms": latency_ms,
            }
        time.sleep(backoff)  # Short wait before the next attempt.
        backoff *= 2  # Exponential backoff to reduce load.


# Demo tools

def flaky_tool(*, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fail once, then succeed; simulate latency with sleep."""

    n = int(payload.get("n", 0))  # Read a small state flag.
    time.sleep(0.12)  # Simulate some work.
    if n == 0:  # First call fails, second succeeds.
        payload["n"] = 1  # Flip the flag for the next attempt.
        raise RuntimeError("transient failure")  # Trigger a retry.
    return {"result": "ok"}  # Success payload.


def echo_file(*, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Write to a sandboxed file path provided by the caller."""

    sandbox_root = payload["sandbox_root"]  # Sandboxed temp file path.
    rel_path = payload["path"]  # Text to write.
    msg = payload["message"]

    safe_path = _safe_path_in_sandbox(sandbox_root, rel_path)

    with open(safe_path, "w", encoding="utf-8") as f:  # Create/truncate file.
        f.write(msg)  # Write text bytes.
    return {"result": f"wrote {len(msg)} bytes", "path": safe_path}  # Report result size.

def _safe_path_in_sandbox(sandbox_root:str, user_path:str) -> str:
    """
    Resolve user_path against sandbox_root and ensure the final path stays
    inside sandbox_root. Reject absolute paths and '../' traversal escapes.
    """
    root_real: Final[str] = os.path.realpath(sandbox_root)

    # Treat user_path as *relative* to sandbox_root, even if attacker supplies an absolute path
    rel = os.path.normpath(user_path).lstrip("/\\")
    candidate = os.path.realpath(os.path.join(root_real, rel))

    # Muat be root itself or inside root/
    if candidate != root_real and not candidate.startswith(root_real + os.sep):
        raise ValueError(f"path escapes sandbox: {user_path!r}")
    return candidate

def demo() -> None:
    """Run two demo calls: flaky tool and sandboxed file writer."""

    spec = ToolSpec(  # Configure retries and timeout for the flaky tool.
        name="flaky", schema={"n": "int"}, timeout_ms=150, max_retries=2
    )
    out = call_tool(spec, tool=flaky_tool, payload={"n": 0})  # First fails, then ok.
    print(out)  # Show status and latency.

    import tempfile, os  # Local imports for the small demo.

    with tempfile.TemporaryDirectory() as tmp:  # Sandbox FS writes.
        spec2 = ToolSpec(name="ehco_file", schema={"sandbox_root":"str","path":"str","message":"str"})
        out2 = call_tool(
            spec2, tool=echo_file, payload={"sandbox_root":tmp, "path":"note.txt","message":"hello"}
        )
        print(out2)  # Show success write.


if __name__ == "__main__":
    demo()
# end::ch06_adapter[]

