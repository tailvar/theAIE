"""Logging demo: illustrates policy, tools, memory, environment.

Retention policy
--------------------------
This agent stores every turn in an append-only episodic memory. As memory grows,
it will eventually become too large to read or use firectly The planned policy:

1. keep all recent turns verbatim (short-term episodic memory)
2. Periodically summarise older turns into a compact fom:
 - collapse old raw entries into a higher level summary string.
 - preserve only saliet facts, tasks and states
3. Prune very old or redundant entries once they are represented
   in the summary
4. Summaries should be stable (idempotent) so repeated summarisation
   does not drift.

This file currently includes a placeholder summarize() function that will
implement this behavious in Chapter 5. For now it is a no-op, called once
at the end of the run_loop() to establish the control flow.
 """

from __future__ import annotations

from asyncio import timeout
from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from databricks.sdk.retries import retried

# TODO(Ch3-Ex1): Extend _execute_tool() with timeout_ms + max_retries and
#                exponential backoff.
# TODO(Ch3-Ex2): Add a simple tool-call budget in run_loop().
# Typing aliases for clarity.
MAX_TURNS=3
MAX_TOOL_CALLS=5

@dataclass
class Observation:
    """Container for a single turn’s input."""

    text: str  # Natural-language description of the task or signal.
    turn: int  # Loop iteration number for traceability.


class Tool(Protocol):
    """Protocol for callables that accept a payload and return structured data."""

    def __call__(self, *, payload: Dict[str, str]) -> Dict[str, str]:
        ...  # No implementation here; concrete tools provide it.


class Memory:
    """Extremely small episodic memory store."""

    def __init__(self) -> None:
        self.episodic: List[Dict[str, str]] = []  # Append-only log of past turns.

    def write(self, *, record: Dict[str, str]) -> None:
        """Persist a dictionary describing the latest turn."""

        # Keep entries in arrival order for easy replay.
        self.episodic.append(record)

    def read(self) -> Dict[str, str]:
        """Return a lightweight summary for the policy."""

        history = " | ".join(event["observation"] for event in self.episodic)
        return {"history": history}  # Policy receives a single string snapshot.


def policy(
    observation: Observation, memory_snapshot: Dict[str, str]
) -> Dict[str, str]:
    """Decide which tool to call based on text cues and memory."""

    text = observation.text.lower()
    if "calculate" in text:
        expression = text.split("calculate", 1)[1].strip()
        return {"tool": "calculator", "payload": {"expression": expression}}
    if "extract numbers" in text or "find numbers" in text:
        return {"tool":"extract_numbers", "payload":{"text": observation.text}}
    if "remember" in text:
        return {"tool": "memory_write", "payload": {"note": observation.text}}
    if "flaky" in text:
        return{"tool":"flaky","payload":{"x":"1"}}
    # Default branch: echo the request back to the user with history context.
    reply = f"(history: {memory_snapshot.get('history', '∅')}) {observation.text}"
    return {"tool": "echo", "payload": {"message": reply}}


def _execute_tool(
    *, tools: Dict[str, Tool],
        name: str,
        payload: Dict[str, str],
        timeout_ms: int | None=None,
        max_retries: int=0,
        backoff_base_ms: int=50,
) -> Dict[str, str]:
    """Call a tool with a place to add timeouts/retries.

    - timeout_ms: per-attempt wall clock timeout (None = no timeout)
    - max_retries: number of retries after the first attempt (0=no retries)
      Total attempts = 1 + max_retries
    - backoff_base_ms: base delay for exponential backoff between retries
      Sleep = backoff_base_ms * (2**(attempt_index-1)) for attempt_index>=1
    """
    tool = tools[name]

    def call_once_with_timeout() -> Dict[str, str]:
        if timeout_ms is None:
            return tool(payload=payload)

        # Run tool in a thread si we can enforce a wall-clock timeout.
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(tool, payload=payload)
            try:
                return fut.result(timeout=timeout_ms / 1000.0)
            except FuturesTimeoutError as e:
                fut.cancel()
                raise TimeoutError(f"tool exceeded {timeout_ms}ms") from e

    total_attempts = 1 + max_retries
    last_err: str | None = None

    for attempt in range(1, total_attempts + 1):
        start = time.monotonic()
        status = "ok"

        try:
            result = call_once_with_timeout()
            latency_ms = int((time.monotonic() - start) * 1000)
            print(
                f"[tool_call] name={name} attempt={attempt}/{total_attempts}"
                f"latency_ms={latency_ms} status={status}"
            )
            return result

        except TimeoutError as e:
            status = "timeout"
            last_err = str(e)
        except Exception:
            status = "error"
            last_err = f"{type(e).__name__}:{e}"

        latency_ms = int((time.monotonic() - start) * 10000)
        print(
            f"[tool_call] name={name} attempt={attempt}/{total_attempts}"
            f"latency_ms={latency_ms} status={status}"
            )

        # if we still have retries left, back off and try again
        if attempt <= max_retries:
            delay_ms = backoff_base_ms * (2**(attempt - 1))
            print(
                f"[tool_retry] name={name} retry={attempt}/{max_retries}"
                f"backoff_ms={delay_ms} reason={status}"
            )
            time.sleep(delay_ms / 1000.0)
        else:
            # Retries exhausted: return a clear failure without throwing.
            return {"result": f"ERROR: TOOL_FAILED name={name} after={total_attempts} last={last_err}"}

def run_loop(observations: List[Observation], tools: Dict[str, Tool]) -> None:
    """Wire policy, tools, and memory together with logging."""

    memory = Memory()  # Initialize the tiny episodic memory.
    tool_calls = 0
    for obs in observations:  # Iterate over input turns.
        # ------ TURN BUDGET -------
        if obs.turn > MAX_TURNS:
            print("budget exhausted: max_turns reached")
            break

        snapshot = memory.read()  # Read memory before making a decision.
        decision = policy(obs, snapshot)  # Policy selects the next tool.
        tool_name = decision["tool"]  # Extract chosen tool name.

        # ------ TOOL-CALL BUDGET -----
        if tool_calls >= MAX_TOOL_CALLS:
            print("budget exhausted: max_tool_calls reached")
            break

        # Execute via seam that can add timeouts/retries.
        response = _execute_tool(
            tools=tools,
            name=tool_name,
            payload=decision["payload"],
            timeout_ms=100, # per-attempt timeout
            max_retries=2, # two retries => thhree attempts total
        )
        tool_calls += 1

        # Persist episode for later reads.
        memory.write(
            record={"observation": obs.text, "response": response["result"]}
        )
        print(
            f"turn={obs.turn} tool={tool_name} response={response['result']}"
        )  # Emit telemetry for this turn.

    # NEW: call summarize() once at end, but memory remains unchanged.
    summarize(memory)

def summarize(memory: Memory) -> None:
    """
    Placeholder summarisation hook.

    For now this does nothing. In chapter 5, this function will:
        - inspect memory.episodic
        - condense older entries into higher level summaries
        - rewrite memory.episodic to keep memory bounded
    """
    pass


def _calculator(*, payload: Dict[str, str]) -> Dict[str, str]:
    """Toy calculator; eval is safe here because payload is under our control."""

    expression = payload["expression"]
    result = eval(expression, {"__builtins__": {}})  # Evaluate simple arithmetic.
    return {"result": str(result)}

def _slow_calculator(*, payload: Dict[str, str]) -> Dict[str, str]:
    time.sleep(0.2) # 200ms (will exceeed a 100ms timeout)
    return _calculator(payload=payload)

def _flaky_fail(*, payload: Dict[str, str]) -> Dict[str, str]:
    """Always fails quickly to demonstrate retries and failure path."""
    raise RuntimeError("simulated failure")

def _memory_write(*, payload: Dict[str, str]) -> Dict[str, str]:
    """Pretend to persist a note and acknowledge it."""

    note = payload["note"]
    return {"result": f"noted: {note}"}


def _echo(*, payload: Dict[str, str]) -> Dict[str, str]:
    """Return the message payload as-is."""

    return {"result": payload["message"]}

def _extract_numbers(*, payload: Dict[str,str]) -> Dict[str,str]:
    """Extract all digit sequences from the given text

    Returns:
        {"result": "<comma-separated list>"} on success or
        {"result": "error:..."} if there are no numbers or input is bad
"""
    import re

    text = payload.get("text","")
    if not text:
        return {"result": "error: no text provided for number extraction"}

    numbers = re.findall(r"\d+", text)
    if not numbers:
        return {"result": "error: no numbers found in input"}

    joined = ", ".join(numbers)
    return {"result": f"numbers: {joined}"}

def with_timeout(tool: Tool, *, timeout_ms:int) -> Tool:
    """Return a tool wrapped with a wall clock timeout.

    Implementation uses a thread + duture timeout. If the ool exceeds the
    timeout, we raise TimeoutError so the caller can decide what to do"""
    def _wrapped(*, payload: Dict[str, str]) -> Dict[str, str]:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(tool, payload=payload)
            try:
                return fut.result(timeout=timeout_ms / 1000.0)
            except FuturesTimeoutError as e:
                # Bets effort cancel, the underlying thread may still complete.
                fut.cancel()
                raise TimeoutError(f"tool exceeded {timeout_ms}ms") from e
    return _wrapped


if __name__ == "__main__":
    tools: Dict[str, Tool] = {
        "calculator": with_timeout(_slow_calculator, timeout_ms=100),
        "memory_write": _memory_write,
        "echo": _echo,
        "extract_numbers": _extract_numbers,
        "flaky": _flaky_fail
    }
    sample_observations = [
        Observation(text="Calculate 2 + 3", turn=1),
        Observation(text="Remember to send the report", turn=2),
        Observation(text="Please extract numbers from abc123 and 45def", turn=3),
        Observation(text="Any updates?", turn=4),
        Observation(text="Extract numbers from this text with no digits", turn=5),
        Observation(text="Use flaky tool please", turn=6),
        Observation(text="Remember to send the report", turn=7),
    ]
    run_loop(sample_observations, tools)


    
