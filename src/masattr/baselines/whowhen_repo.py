"""Who&When baseline reproduction (spec v3 Part C §6, Part B.3).

Their repo is a **dependency, not a fork**: point ``--repo-path`` at a checkout
and this module imports their three functions from ``inference.py`` and calls
them. Nothing of theirs is patched except that credentials arrive via CLI flags
(their env-var fallback is documented but not implemented in their code). Their
``evaluate.py`` is *not* imported — the substring scorer is re-implemented in
``eval/scorers.py`` so the comparability row is deliberate.

Each method runs twice: with gpt-4o (their regime) and with our judge model (the
capability control). Without the second arm, any gap between their methods and
ours is confounded with judge capability.

``--impl local`` exists only so the pipeline is runnable without their checkout.
It is a re-prompting of the same three strategies, **not** the reproduction, and
every row it produces is stamped ``impl="local"`` so it can never be mistaken
for one.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..record import Record

MAX_STEP_CHARS = 1500
METHODS = ("all_at_once", "step_by_step", "binary_search")

_STEP_PAT = re.compile(r"step\s*(?:index\s*)?[:#]?\s*(\d+)", re.IGNORECASE)
_AGENT_PAT = re.compile(r"agent\s*[:#]?\s*([A-Za-z_][\w \-]*)", re.IGNORECASE)


def parse_answer(text: str) -> tuple[str | None, int | None]:
    """``(agent, step)`` from a free-form response; ``(None, None)`` if unreadable.

    Unreadable answers are counted as misses, not dropped — dropping them would
    inflate the baseline's accuracy by silently shrinking its denominator.
    """
    if not text:
        return None, None
    lo, hi = text.find("{"), text.rfind("}")
    if lo != -1 and hi > lo:
        try:
            obj = json.loads(text[lo : hi + 1])
            if isinstance(obj, dict):
                agent = obj.get("agent") or obj.get("mistake_agent")
                raw = obj.get("step", obj.get("mistake_step"))
                try:
                    step = int(raw) if raw is not None else None
                except (TypeError, ValueError):
                    step = None
                return (str(agent) if agent else None), step
        except (json.JSONDecodeError, ValueError):
            pass
    a, s = _AGENT_PAT.search(text), _STEP_PAT.search(text)
    return (a.group(1).strip() if a else None), (int(s.group(1)) if s else None)


# --- generators -------------------------------------------------------------


class Generator:
    """Minimal text-generation interface shared by both baseline arms."""

    name = "abstract"

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> str:
        raise NotImplementedError


class MockGenerator(Generator):
    name = "mock"

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> str:
        import hashlib

        h = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big")
        return json.dumps({"agent": "WebSurfer", "step": h % 5})


class OpenAIGenerator(Generator):  # pragma: no cover - network
    """Their regime: gpt-4o, credentials passed explicitly (Part C §6)."""

    def __init__(self, model: str = "gpt-4o", *, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return r.choices[0].message.content or ""


class JudgeGenerator(Generator):
    """The capability control: our judge model driving their methods."""

    def __init__(self, client) -> None:
        self.client = client
        self.name = f"judge:{client.name}"

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> str:
        self.client.reset("")
        text, _ = self.client.generate(prompt, max_new_tokens=max_new_tokens)
        return text


def build_generator(spec: str, *, api_key: str | None = None, device: str | None = None) -> Generator:
    """``mock`` | ``openai:<model>`` | ``judge:<judge-spec>``"""
    if spec == "mock":
        return MockGenerator()
    kind, _, name = spec.partition(":")
    if kind == "openai":
        if not api_key:
            raise ValueError("openai generator needs --api-key (Part C §6: credentials via flags)")
        return OpenAIGenerator(name or "gpt-4o", api_key=api_key)
    if kind == "judge":
        from ..judge.client import build_client

        return JudgeGenerator(build_client(name or "mock", device=device))
    raise ValueError(f"unknown generator spec {spec!r}")


# --- their repo, imported ---------------------------------------------------


def load_repo(repo_path: str | Path) -> Any:
    """Import the Who&When repo's ``inference.py`` as a module."""
    path = Path(repo_path)
    candidates = [path / "inference.py", path / "evaluation" / "inference.py"]
    for c in candidates:
        if c.exists():
            spec = importlib.util.spec_from_file_location("whowhen_inference", c)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["whowhen_inference"] = mod
                spec.loader.exec_module(mod)
                return mod
    raise FileNotFoundError(f"no inference.py under {path} (looked in {[str(c) for c in candidates]})")


def repo_callables(mod: Any) -> dict[str, Callable]:
    missing = [m for m in METHODS if not hasattr(mod, m)]
    if missing:
        raise AttributeError(
            f"Who&When inference.py is missing {missing}; expected {list(METHODS)} — "
            "their API changed, so wrap the new names rather than patching their file"
        )
    return {m: getattr(mod, m) for m in METHODS}


# --- local re-prompting (NOT the reproduction) ------------------------------


def _steps_text(record: Record, lo: int, hi: int) -> str:
    return "\n".join(
        f"[step {s.idx}] {s.agent}: {(s.content or '')[:MAX_STEP_CHARS]}"
        for s in record.steps[lo:hi]
    )


def _header(record: Record) -> str:
    return (
        "A multi-agent system failed to solve the following task.\n\n"
        f"Task: {record.query or '(not recorded)'}\n"
        f"Ground truth: {record.ground_truth or '(not recorded)'}\n"
    )


def local_all_at_once(record: Record, gen: Generator) -> tuple[str | None, int | None, int]:
    prompt = (
        _header(record)
        + "\nFull transcript:\n"
        + _steps_text(record, 0, record.n_steps)
        + '\n\nIdentify the agent that made the decisive mistake and the step index where '
        'it happened. Reply as JSON: {"agent": "...", "step": N}\n'
    )
    return (*parse_answer(gen.generate(prompt)), 1)


def local_step_by_step(record: Record, gen: Generator) -> tuple[str | None, int | None, int]:
    calls = 0
    for i, s in enumerate(record.steps):
        prompt = (
            _header(record)
            + "\nTranscript so far:\n"
            + _steps_text(record, 0, i + 1)
            + f"\n\nIs step {i} by '{s.agent}' the decisive mistake that caused the failure? "
            "Answer Yes or No.\n"
        )
        calls += 1
        if (gen.generate(prompt, max_new_tokens=8) or "").strip().lower().startswith("yes"):
            return s.agent, i, calls
    last = record.steps[-1]
    return last.agent, last.idx, calls


def local_binary_search(record: Record, gen: Generator) -> tuple[str | None, int | None, int]:
    lo, hi, calls = 0, record.n_steps, 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        prompt = (
            _header(record)
            + f"\nSegment A (steps {lo}..{mid - 1}):\n"
            + _steps_text(record, lo, mid)
            + f"\n\nSegment B (steps {mid}..{hi - 1}):\n"
            + _steps_text(record, mid, hi)
            + "\n\nWhich segment contains the decisive mistake? Answer A or B.\n"
        )
        calls += 1
        if (gen.generate(prompt, max_new_tokens=4) or "").strip().upper().startswith("B"):
            lo = mid
        else:
            hi = mid
    return record.steps[lo].agent, lo, calls


LOCAL = {
    "all_at_once": local_all_at_once,
    "step_by_step": local_step_by_step,
    "binary_search": local_binary_search,
}


# --- driver -----------------------------------------------------------------


@dataclass
class BaselineRun:
    subset: str
    method: str
    generator: str
    impl: str
    n: int = 0
    calls: int = 0
    unparsed: int = 0
    preds: dict[str, tuple[str | None, int | None]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subset": self.subset,
            "method": self.method,
            "generator": self.generator,
            "impl": self.impl,
            "n": self.n,
            "llm_calls": self.calls,
            "unparsed": self.unparsed,
        }


def run_method(
    records: Sequence[Record],
    gen: Generator,
    *,
    method: str,
    subset: str,
    impl: str = "local",
    repo_path: str | Path | None = None,
    limit: int | None = None,
) -> BaselineRun:
    if method not in METHODS:
        raise ValueError(f"unknown baseline method {method!r}; known: {METHODS}")
    records = list(records)[: limit or len(records)]
    out = BaselineRun(subset=subset, method=method, generator=gen.name, impl=impl, n=len(records))

    if impl == "repo":
        if repo_path is None:
            raise ValueError("impl='repo' needs --repo-path pointing at the Who&When checkout")
        fn = repo_callables(load_repo(repo_path))[method]
        for rec in records:
            agent, step = parse_answer(str(fn(rec.to_dict())))
            out.preds[rec.key] = (agent, step)
            out.calls += 1
            out.unparsed += int(agent is None and step is None)
        return out

    if impl != "local":
        raise ValueError(f"unknown impl {impl!r}; use 'repo' or 'local'")
    fn = LOCAL[method]
    for rec in records:
        agent, step, calls = fn(rec, gen)
        out.preds[rec.key] = (agent, step)
        out.calls += calls
        out.unparsed += int(agent is None and step is None)
    return out
