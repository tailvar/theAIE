"""Micro-evaluation harness for the structured extractor"""
from __future__ import annotations

from typing import List, Dict

from structured_extraction import mock_generate, validate_payload

def main() -> None:
    oracle: List[Dict[str,object]] = [
        {
            "text": "ACME reported revenue of $1.2m",
            "expect":{
                "company":"ACME","revenue":1_200_000,"currency":"USD"
            },
        },
        {
            "text": "MEGA GmbH posted a EUR 950k in revenue",
            "expect":{
                "company": "MEGA", "revenue": 950_000, "currency": "EUR"
            },
        },
        {
            "text": "ALPHA quarterly revenue was $250k",
            "expect": {
                "company": "ALPHA", "revenue": 250_000, "currency": "USD"
            },
        },
        ]
    passed = 0
    for item in oracle:
        ext = mock_generate(item["text"])
        obj = {
            "company": ext.company,
            "revenue": ext.revenue,
            "currency": ext.currency,
        }
        ok, err = validate_payload(obj)
        if ok and obj == item["expect"]:
            passed += 1
        failed = len(oracle) - passed
        print(f"passed={passed}, failed={failed}")

if __name__ == "__main__":
    main()