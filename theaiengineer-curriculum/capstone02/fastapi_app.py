"""Optional FastAPI preview for Chapter 2.

Run locally (requires FastAPI + Uvicorn):
  pip install fastapi uvicorn
  PYTHONPATH=code/capstone02/src uvicorn code.capstone02.fastapi_app:app --reload

Then open:
  http://127.0.0.1:8000/stats?seed=42&n=5

CLI run PYTHONPATH=src uvicorn theaiengineer-curriculum.capstone02.fastapi_app:app --reload
then open http://127.0.0.1:8000/stats?seed=7&n=5
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from mymltool.core import compute_values  # assumes mymltool is under /app/src

app = FastAPI(title="Chapter 2 — Stats API")


class StatsOut(BaseModel):
    mean: float
    stdev: float


@app.get("/stats", response_model=StatsOut)
def stats(seed: int = 123, n: int = 5) -> StatsOut:
    _, mu, sigma = compute_values(seed, n)
    return StatsOut(mean=mu, stdev=sigma)


