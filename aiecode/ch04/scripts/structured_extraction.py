"""Structured information extraction with JSON validation (no API required).

CLI:
  python structured_extractiom.py --text "ACME reported revenue of $1.2m."
"""

from __future__ import annotations  # postpone annotations

import argparse  # parse flags
import json  # JSON parsing/printing
import os  # backend selection via env
import re  # simple extraction
from dataclasses import dataclass  # output structure
from typing import Dict, Optional, Tuple  # type hints

@dataclass
class Extract:  # structured output
    company: str  # company name (uppercased token)
    revenue: int  # revenue in whole currency units
    currency: str  # ISO code (USD/EUR)


def validate_payload(  # signature
    obj: Dict[str, object]
) -> Tuple[bool, Optional[str]]:
    """Validate that obj has required keys and types; return (ok, error)."""
    required = {"company": str, "revenue": int, "currency": str}  # schema
    for k, typ in required.items():  # iterate fields
        if k not in obj:  # missing key
            return False, f"missing key: {k}"  # error
        if not isinstance(obj[k], typ):  # wrong type
            return False, f"bad type for {k}: expected {typ.__name__}"  # error
    return True, None  # ok


def _norm_number(text: str) -> int:  # parse money expression
    """Normalize money like "$1.2m" or "EUR 950k" to integer units."""
    m = re.search(r"([\$€]?)(\d+(?:\.\d+)?)\s*([mk]?)", text, re.I)  # regex
    if not m:  # no match
        raise ValueError("no monetary value found")  # signal
    amount = float(m.group(2))  # numeric part
    suffix = (m.group(3) or "").lower()  # scale suffix
    mult = 1_000_000 if suffix == "m" else 1_000 if suffix == "k" else 1  # scale
    return int(round(amount * mult))  # integer units


def mock_generate(text: str) -> Extract:  # deterministic mock
    """Deterministic extractor: parse company, revenue, and currency."""
    tokens = re.findall(r"[A-Za-z]+", text)  # token list
    company = next(  # pick
        (t for t in tokens if t.isupper() and len(t) >= 2), "UNKNOWN"
    )
    currency = "USD" if "$" in text or "usd" in text.lower() else (  # currency
        "EUR" if "€" in text or "eur" in text.lower() else "USD"  # fallback
    )
    revenue = _norm_number(text)  # normalize value
    return Extract(company=company, revenue=revenue, currency=currency)  # build


def build_parser() -> argparse.ArgumentParser:  # CLI builder
    p = argparse.ArgumentParser(  # parser
        description="Structured extraction to JSON."
    )
    p.add_argument("--text", required=True, help="Input text")  # text
    p.add_argument(  # choose backend
        "--backend",
        choices=["mock", "openai"],  # allowed backends
        default=os.environ.get("LLM_BACKEND", "mock"),  # env default
        help="Extraction backend (default: mock)",  # help text
    )
    return p  # return parser


def main() -> None:  # entry point=30/60
    args = build_parser().parse_args()  # parse args
    if args.backend == "openai":  # optional live call
        try:  # import guarded
            import re
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise SystemExit("OPENAI_API_KEY not set")
            print("using openai backend")
            client = OpenAI(api_key=api_key)

            prompt = (
                "Extract company, revenue, and currency.\n"
                "Return ONLY valid JSON (no markdown, no commentary).\n"
                'Schema: {"company": string, "revenue": integer, "currency": string}\n'
                f"Text: {args.text}"
            )

            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You output ONLY JSON. No extra text."},
                    {"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},  # <-- forces JSON
            )

            content = resp.choices[0].message.content or "{}"
            obj = json.loads(content) # parse JSON
        except Exception as e:  # failure path
            raise SystemExit(f"openai backend failed: {e}")  # exit
    else:  # mock path
        ext = mock_generate(args.text)  # extract locally
        obj = {  # obj
            "company": ext.company,
            "revenue": ext.revenue,
            "currency": ext.currency,
        }
    ok, err = validate_payload(obj)  # validate JSON
    if not ok:  # fail fast
        raise SystemExit(f"validation failed: {err}")  # error
    print(json.dumps(obj, ensure_ascii=False))  # print JSON


if __name__ == "__main__":  # script guard
    main()  # run





