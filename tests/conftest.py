"""Synthetic Who&When fixtures, built in a tmp dir.

They mirror the *shape* of the real annotations — the compound HC role, the
string-typed ``mistake_step``, an agent/step disagreement — which is what the
loaders are actually being tested against. No real data is committed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest


def _alg_file(i: int) -> dict:
    return {
        "question": "Who won the 1998 final?",
        "ground_truth": "France",
        # File 3 names an agent that does not act at the annotated step.
        "mistake_agent": "WebSurfer" if i != 3 else "Coder",
        "mistake_step": "1",
        "mistake_reason": "hallucinated a source",
        "history": [
            {"role": "assistant", "name": "Manager", "content": "Plan: search, then verify."},
            {
                "role": "assistant",
                "name": "WebSurfer",
                "content": "Searching the web for the 1998 final result and reading the top hit.",
            },
            {"role": "assistant", "name": "Verifier", "content": "The answer is France."},
        ],
    }


def _hc_file(i: int) -> dict:
    history = [
        {"role": "Orchestrator (thought)", "content": "I should ask WebSurfer to look this up."},
        {"role": "Orchestrator (-> WebSurfer)", "content": "WebSurfer, find the capital of Peru."},
        {"role": "WebSurfer", "content": "Address: http://example.com\nViewport position: top"},
    ]
    # One longer trajectory so length-sensitive code (changepoint, turn blocks,
    # the length stratum in E9) sees more than a toy.
    if i == 2:
        for k in range(8):
            history += [
                {"role": "Orchestrator (thought)", "content": f"Round {k}: still not resolved."},
                {"role": "WebSurfer", "content": f"Opened result {k}; nothing conclusive."},
            ]
    history.append({"role": "Orchestrator (final answer)", "content": "The answer is Lima."})
    return {
        "question": "What is the capital of Peru?",
        "ground_truth": "Lima",
        "mistake_agent": "WebSurfer",
        "mistake_step": "2",
        "mistake_reason": "read the wrong page",
        "history": history,
    }


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("masattr_data")
    alg = root / "whowhen" / "Algorithm-Generated"
    hc = root / "whowhen" / "Hand-Crafted"
    alg.mkdir(parents=True)
    hc.mkdir(parents=True)
    for i in range(4):
        (alg / f"alg_{i}.json").write_text(json.dumps(_alg_file(i)))
    for i in range(3):
        (hc / f"hc_{i}.json").write_text(json.dumps(_hc_file(i)))
    return root


@pytest.fixture(scope="session")
def paper1_scores(tmp_path_factory) -> Path:
    """A stand-in for paper 1's step-labeled corpus, already scored by a judge.

    The judge is deliberately overconfident here — true correctness rate is
    ``p**2`` — so a working calibration has something to fix.
    """
    path = tmp_path_factory.mktemp("paper1") / "scores.jsonl"
    rng = random.Random(0)
    types = ["thought", "action", "observation", "answer"]
    with path.open("w") as fh:
        for i in range(3000):
            p = rng.random()
            fh.write(
                json.dumps(
                    {"p_raw": p, "type": types[i % len(types)], "correct": rng.random() < p**2}
                )
                + "\n"
            )
    return path


@pytest.fixture()
def records(data_root):
    from masattr.loaders.whowhen_ag import load as load_alg
    from masattr.loaders.whowhen_hc import load as load_hc

    alg, _ = load_alg(data_root / "whowhen" / "Algorithm-Generated")
    hc, _ = load_hc(data_root / "whowhen" / "Hand-Crafted")
    return {"alg": alg, "hc": hc}


@pytest.fixture()
def scores(records):
    """Mock-judge scores for every fixture file, grouped by file key."""
    from masattr.judge.client import MockClient
    from masattr.judge.score import by_file, score_corpus

    client = MockClient(seed=1)
    rows = []
    for recs in records.values():
        for ts in score_corpus(recs, client):
            rows.extend(ts.scores)
    return by_file(rows)
