"""Who&When baseline reproduction (spec v3 Part C §6, Part B.3).

Their repo is a **dependency, not a fork**: point ``--repo-path`` at a checkout
and this module invokes their ``inference.py`` **as a subprocess**, exactly as
they document it. Nothing of theirs is patched, edited, or imported.

Subprocess rather than import because their ``inference.py`` exposes only
``main()`` behind argparse — there are no ``all_at_once`` / ``step_by_step`` /
``binary_search`` functions to call. Their CLI is::

    python inference.py --method {all_at_once,step_by_step,binary_search}
                        --model <id> --directory_path <dir>
                        --is_handcrafted {True,False}
                        --api_key ... --azure_endpoint ... --api_version ...

Their file still imports the Azure client, which is a stale 2025 dependency
rather than a design choice. Rather than edit their tree, a shim is injected on
``PYTHONPATH`` at the subprocess boundary: it makes ``openai.AzureOpenAI``
construct a standard ``openai.OpenAI``, ignoring the endpoint and api-version
arguments. Their prompt assembly, method logic, retry loop, and output contract
stay byte-identical.

The shim also records the concrete ``model`` string the API returns — the
gpt-4o *snapshot*, not the alias — to a receipt file, because a drifted alias is
the first thing to suspect if reproduction lands off the published numbers.

It reads a *directory of per-trajectory JSON*, not the parquet, and writes
results to ``outputs/{method}_{model}[_handcrafted].txt`` beside itself rather
than to stdout.

Their files are named by ordinal (``1.json``), while our records are keyed by
``question_ID``, so predictions are joined through the ``question_ID`` inside
each of their files. Without that mapping the two sides would line up by
position and quietly score nonsense. Their
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

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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


#: Where ``inference.py`` sits in their tree, newest layout first.
_INFERENCE_CANDIDATES = ("Automated_FA/inference.py", "inference.py", "evaluation/inference.py")


def find_inference(repo_path: str | Path) -> Path:
    path = Path(repo_path)
    for rel in _INFERENCE_CANDIDATES:
        c = path / rel
        if c.exists():
            return c
    raise FileNotFoundError(
        f"no inference.py under {path} (looked for {list(_INFERENCE_CANDIDATES)})"
    )


def repo_command(
    script: Path,
    *,
    method: str,
    model: str,
    directory_path: str | Path,
    is_handcrafted: bool,
    api_key: str | None,
    device: str | None = None,
) -> list[str]:
    """Their documented command line. Credentials are passed explicitly."""
    cmd = [
        sys.executable,
        str(script.name),
        "--method",
        method,
        "--model",
        model,
        "--directory_path",
        str(directory_path),
        "--is_handcrafted",
        "True" if is_handcrafted else "False",
    ]
    if api_key:
        cmd += ["--api_key", api_key]
    if device:
        cmd += ["--device", device]
    return cmd


def run_repo_subprocess(
    repo_path: str | Path,
    *,
    method: str,
    model: str,
    directory_path: str | Path,
    is_handcrafted: bool,
    api_key: str | None = None,
    device: str | None = None,
    snapshot_receipt: str | Path | None = None,
    base_url: str | None = None,
    model_rewrite: str | None = None,
    timeout: int = 60 * 60 * 6,
) -> str:
    """Run their script in its own directory and return stdout.

    Run from the script's directory because their default paths are relative to
    it; anything else would require editing their file.
    """
    script = find_inference(repo_path)
    cmd = repo_command(
        script,
        method=method,
        model=model,
        directory_path=directory_path,
        is_handcrafted=is_handcrafted,
        api_key=api_key,
        device=device,
    )
    env = dict(os.environ)
    shim_dir = write_openai_shim(snapshot_receipt)
    env["PYTHONPATH"] = os.pathsep.join([str(shim_dir), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    if snapshot_receipt:
        env["MASATTR_SNAPSHOT_RECEIPT"] = str(snapshot_receipt)
    if base_url:
        env["MASATTR_OPENAI_BASE_URL"] = base_url
    if model_rewrite:
        env["MASATTR_MODEL_REWRITE"] = model_rewrite
    proc = subprocess.run(
        cmd, cwd=script.parent, capture_output=True, text=True, timeout=timeout, env=env
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"their inference.py exited {proc.returncode}\n"
            f"cmd: {' '.join(cmd)}\n{proc.stderr[-2000:]}"
        )
    return proc.stdout


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
        raise ValueError(
            "impl='repo' runs their script over a whole directory in one "
            "invocation, so it is driven by runs/baselines.py rather than "
            "per-record here — call run_repo_subprocess directly"
        )

    if impl != "local":
        raise ValueError(f"unknown impl {impl!r}; use 'repo' or 'local'")
    fn = LOCAL[method]
    for rec in records:
        agent, step, calls = fn(rec, gen)
        out.preds[rec.key] = (agent, step)
        out.calls += calls
        out.unparsed += int(agent is None and step is None)
    return out


# --- reading their results ---------------------------------------------------

#: Their own contract, re-implemented from evaluate.py rather than imported.
_PRED_BLOCK = re.compile(r"Prediction for ([^:]+\.json):(.*?)(?=Prediction for|\Z)", re.DOTALL)
_PRED_AGENT = re.compile(r"Agent Name:\s*([\w_]+)", re.IGNORECASE)
_PRED_STEP = re.compile(r"Step Number:\s*(\d+)", re.IGNORECASE)


def output_path(repo_path: str | Path, *, method: str, model: str, is_handcrafted: bool) -> Path:
    script = find_inference(repo_path)
    suffix = "_handcrafted" if is_handcrafted else ""
    return script.parent / "outputs" / f"{method}_{model.replace('/', '_')}{suffix}.txt"


def id_map(directory_path: str | Path) -> dict[str, str]:
    """``their filename stem -> question_ID``, read from their own files."""
    out: dict[str, str] = {}
    for p in sorted(Path(directory_path).glob("*.json")):
        try:
            ident = json.loads(p.read_text(encoding="utf-8")).get("question_ID")
        except (json.JSONDecodeError, OSError):
            continue
        if ident:
            out[p.stem] = str(ident)
    return out


def parse_repo_output(text: str, ids: Mapping[str, str], subset: str) -> dict[str, tuple[str, int]]:
    """Their result file -> ``{record key: (agent, step)}``."""
    preds: dict[str, tuple[str, int]] = {}
    for block in _PRED_BLOCK.finditer(text):
        stem = Path(block.group(1).strip()).stem
        body = block.group(2)
        agent, step = _PRED_AGENT.search(body), _PRED_STEP.search(body)
        if not (agent and step):
            continue
        ident = ids.get(stem)
        if ident is None:
            continue
        preds[f"{subset}/{ident}"] = (agent.group(1), int(step.group(1)))
    return preds


# --- transport shim ---------------------------------------------------------

_SHIM = '''"""Injected at the subprocess boundary; their tree is never edited.

Three jobs, all env-driven so their file stays byte-identical:

1. Their inference.py still imports the Azure client, a stale dependency rather
   than a design choice. AzureOpenAI here constructs a standard OpenAI client
   and drops the endpoint/api-version arguments.
2. MASATTR_OPENAI_BASE_URL redirects the client at a local OpenAI-compatible
   server, which is how their three strategies run on our judge — the
   capability control: their prompts and logic, our model.
3. MASATTR_MODEL_REWRITE replaces the model name on every call, because their
   CLI only accepts names from their own hard-coded list.

It also records the concrete model string the API returns, so a drifted alias
or an unexpected served model is visible afterwards.
"""
import os

import openai

_receipt = os.environ.get("MASATTR_SNAPSHOT_RECEIPT")
_base_url = os.environ.get("MASATTR_OPENAI_BASE_URL")
_rewrite = os.environ.get("MASATTR_MODEL_REWRITE")
_seen = set()


def _record(model):
    if not _receipt or not model or model in _seen:
        return
    _seen.add(model)
    with open(_receipt, "a", encoding="utf-8") as fh:
        fh.write(model + "\\n")


class _Completions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, *a, **kw):
        if _rewrite:
            kw["model"] = _rewrite
        resp = self._inner.create(*a, **kw)
        _record(getattr(resp, "model", None))
        return resp


class _Chat:
    def __init__(self, inner):
        self._inner = inner
        self.completions = _Completions(inner.completions)


class _Client:
    def __init__(self, *a, **kw):
        for drop in ("azure_endpoint", "api_version", "azure_deployment"):
            kw.pop(drop, None)
        if _base_url:
            kw["base_url"] = _base_url
            kw.setdefault("api_key", "not-needed")
        self._inner = openai.OpenAI(*a, **kw)
        self.chat = _Chat(self._inner.chat)

    def __getattr__(self, name):
        return getattr(self._inner, name)


openai.AzureOpenAI = _Client
openai.OpenAI = _Client if _base_url else openai.OpenAI
'''


def write_openai_shim(receipt: str | Path | None = None) -> Path:
    """Write the shim to a temp dir and return it, for PYTHONPATH injection."""
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="masattr_shim_"))
    (d / "sitecustomize.py").write_text(_SHIM, encoding="utf-8")
    _ = receipt
    return d
