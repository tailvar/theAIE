"""chapter 1 calculator agent example"""
from __future__ import annotations
import operator
import re
from dataclasses import dataclass
from typing import Optional, Callable, Dict

Ops: Dict[str, Callable[[float, float], float]] = {
    "add": operator.add, # Map the word "add" to th addition operator
    "plus": operator.add,
    "sum": operator.add, # allow "sum" to trigger addition
    "subtract": operator.sub, # words that imply subtraction map to operator.sub
    "minus": operator.sub,
    "minus": operator.sub,
    "multiply": operator.mul,
    "times": operator.mul,
    "divide": operator.truediv, # Division words use true division
    "over": operator.truediv,
    "power": operator.pow,
}

def calculator(a: float, b: float, op: str) -> float:
    """Apply the requested arithmetic operation to the inputs."""
    if op not in Ops: # Reject operations that are not in the supported map
        raise ValueError(f"Unknown operation: {op}")
    return Ops[op](a,b) # invoke the operator associated with the keyword

@dataclass
class Observation:
    text: str # Raw natural language query describing the task.

def parse(text: str) -> Optional[tuple[float, float, str]]:
    """Extract two numbers and an operation keyword from the text."""
    # Lowercase and trim whitespace for easier matching
    normalized = text.lower().strip()
    # collapse phrases like "divide by" into a simpler form.
    normalized = normalized.replace(" by ", " ")
    # look for onne of the operational keywords
    match = re.search(
        r"(add|plus|sum|subtract|minus|multiply|times|divide|over|power)",
        normalized,
    )
    # Capture integer or decimal numbers
    numbers = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    # Bail out unless we find an operator and a least two numbers
    if not match or len(numbers) < 2:
        return None
    # Convert the first two numbers into floats
    first, second = map(float, numbers[:2])
    # return operands plus the operation keyword
    return first, second, match.group(1)

def policy(obs: Observation) -> str:
    """Decide whether to answer directly or use the calculator tool"""
    parsed = parse(obs.text) # attempt to interpret the observation text.
    if not parsed: # If parsing fails, respond with a saf fallback message.
        return "I can only handle simple arithmetic like 'add 2 and 3'."
    a, b, op = parsed # Unpack the parsed operands and operator kyword
    try:
        result = calculator(a, b, op) # Offload arithmetic to the calculator tool.
    except Exception as exc: # catch tool errors (e.g. divisible by zero)
        return f"Tool error: {exc}"
    return f"{result:g}" # format the numeric word compactly

def run_demo() -> None:
    """Demonstrate the policy on a samll batch of requests"""
    demos = [
        "add 2 and 3",
        "2 plus 3",
        "multiply 6 and 7",
        "divide 8 and 2",
        "subtract 10 and 3",
        "what is the capital of France",
        "3 power 3"
    ] # Diverse prompts covering supported and unsupported tasks.
    for prompt in demos:
        print(f"{prompt} -> {policy(Observation(prompt))}")

if __name__ == "__main__":
    run_demo()


