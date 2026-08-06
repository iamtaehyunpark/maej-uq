"""Synthetic fixtures in each of the four source formats.

Built in a tmp dir rather than committed, so the repo carries no data and the
fixtures stay obviously synthetic. They mirror the *shape* of the real dumps
(nesting, key names, the compound HC role, the string-typed ``mistake_step``),
which is what the loaders are actually being tested against.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

N_TASKS = 3
N_RUNS = 4


def _camel_steps(correct: bool) -> list[dict]:
    return [
        {"role": "user", "output": "Compute 17 * 3."},
        {"role": "assistant", "output": "Here is the plan: first multiply, then verify."},
        {"role": "assistant", "output": "17 * 3 = 51" if correct else "17 * 3 = 49"},
        {"role": "assistant", "output": "The answer is 51." if correct else "The answer is 49."},
    ]


def _autogen_steps(correct: bool) -> list[dict]:
    return [
        {
            "role": "assistant",
            "agent": "StarAgent",
            "turn": 0,
            "type": "plan",
            "output": json.dumps({"analyst_task": "solve it", "verifier_task": "check it"}),
        },
        {
            "role": "assistant",
            "agent": "Analyst",
            "turn": 1,
            "type": "solve",
            "output": "Working through the options carefully; option B fits the stem.",
        },
        {"role": "assistant", "agent": "Verifier", "turn": 1, "type": "solve", "output": "B"},
        {
            "role": "assistant",
            "agent": "StarAgent",
            "turn": 2,
            "type": "final",
            "output": "The answer is B." if correct else "The answer is D.",
        },
    ]


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("masuq_data")

    matu = root / "matu" / "quick_start" / "results"
    matu.mkdir(parents=True)

    camel: dict = {}
    autogen: dict = {}
    camel_acc: dict = {}
    autogen_acc: dict = {}
    for t in range(N_TASKS):
        tid = f"task_{t}"
        camel[tid] = []
        autogen[tid] = []
        camel_acc[tid] = []
        autogen_acc[tid] = []
        for r in range(N_RUNS):
            ok = (t + r) % 3 != 0
            camel[tid].append(
                {"question": "Compute 17 * 3.", "answer": "51", "steps": _camel_steps(ok)}
            )
            autogen[tid].append(_autogen_steps(ok))
            camel_acc[tid].append(ok)
            autogen_acc[tid].append(ok)

    (matu / "conversation_logs_Math_qwen2.5.json").write_text(json.dumps(camel))
    (matu / "conversation_logs_MMLU_Autogen_qwen2.5.json").write_text(json.dumps(autogen))
    (matu / "accuracy_dict_Math_qwen2.5.pkl").write_bytes(pickle.dumps(camel_acc))
    (matu / "accuracy_dict_MMLU_Autogen_qwen2.5.pkl").write_bytes(pickle.dumps(autogen_acc))

    alg = root / "whowhen" / "Algorithm-Generated"
    alg.mkdir(parents=True)
    for i in range(4):
        (alg / f"alg_{i}.json").write_text(
            json.dumps(
                {
                    "question": "Who won the 1998 final?",
                    "ground_truth": "France",
                    # File 3 names an agent that does not act at the annotated step.
                    "mistake_agent": "WebSurfer" if i != 3 else "Coder",
                    "mistake_step": "1",
                    "mistake_reason": "hallucinated a source",
                    "history": [
                        {"role": "assistant", "name": "Manager", "content": "Plan: search, then verify."},
                        {"role": "assistant", "name": "WebSurfer", "content": "Searching the web for the 1998 final result."},
                        {"role": "assistant", "name": "Verifier", "content": "The answer is France."},
                    ],
                }
            )
        )

    hc = root / "whowhen" / "Hand-Crafted"
    hc.mkdir(parents=True)
    for i in range(3):
        (hc / f"hc_{i}.json").write_text(
            json.dumps(
                {
                    "question": "What is the capital of Peru?",
                    "ground_truth": "Lima",
                    "mistake_agent": "WebSurfer",
                    "mistake_step": "1",
                    "mistake_reason": "wrong page",
                    "history": [
                        {"role": "Orchestrator (thought)", "content": "I should ask WebSurfer to look this up."},
                        {"role": "Orchestrator (-> WebSurfer)", "content": "WebSurfer, find the capital of Peru."},
                        {"role": "WebSurfer", "content": "Address: http://example.com\nViewport position: top"},
                        {"role": "Orchestrator (final answer)", "content": "The answer is Lima."},
                    ],
                }
            )
        )

    return root


@pytest.fixture(scope="session")
def paths(data_root):
    from masuq.config import load_paths

    return load_paths(root=data_root)
