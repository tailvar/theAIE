"""Minimal schema + validator for structured JSON outputs

Defines a tiiny schema that captures required keys and types, and a 4
validate function that checks an instance generically. This
avoids ad-hoc validation"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, Mapping

# simple schema: key -> expected python type
SCHEMA: Dict[str, type] = {
    "company":str,"revenue":int,"currency":str
}

def validate(
        obj: Mapping[str, Any], schema: Mapping[str, type]
)-> Tuple[bool, Optional[str]]:

    """Validate `obj` against `schema`; return (ok, error)"""
    for key, type in schema.items():
        if key not in obj:
            return False, f",missing key {key}"
        if not isinstance(obj[key], type):
            return False, f"bad type for {key}: expected {typ.__name__}"
    return True, None