"""Surrogate-intrinsic baseline (spec v3 Part C §6, experiment E7).

Frozen logs do not carry the generating model's own token distributions, so no
true intrinsic uncertainty signal is recoverable here. What is computable is how
a third-party proxy LM scores the recorded text: per-step mean token logprob and
mean token entropy of ``content``.

It feeds the *same* attribution rules as the judge field — applied identically —
so E7 is a fair comparison of signals, not of pipelines. It is expected to be
weak, and that is the point: it is the control showing the judge signal is not
merely fluency.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

from ..judge.score import StepScore
from ..record import Record


@dataclass(slots=True)
class SurrogateRow:
    key: str
    step_idx: int
    mean_logprob: float
    mean_entropy: float
    n_tokens: int


def _squash(mean_logprob: float) -> float:
    """Monotone map into (0,1) so the surrogate can enter the same rules.

    Purely a rescaling — it adds no information, and the surrogate is never
    calibrated, because there is nothing principled to calibrate it against.
    """
    return 1.0 / (1.0 + math.exp(-(mean_logprob + 2.0)))


class ProxyLM:
    """Base LM scorer over recorded step text."""

    def __init__(self, model_id: str, *, device: str | None = None, dtype: str = "bfloat16",
                 max_tokens: int = 1024) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError("ProxyLM needs the 'judge' extra: pip install -e '.[judge]'") from e
        self._torch = torch
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype) if dev != "cpu" else torch.float32
        ).to(dev)
        self.model.eval()
        self.device = dev

    def score_text(self, text: str) -> tuple[float, float, int]:
        torch = self._torch
        ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_tokens
        ).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return 0.0, 0.0, int(ids.shape[1])
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0, :-1].float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = ids[0, 1:]
        return (
            float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()),
            float((-(lp.exp() * lp).sum(-1)).mean()),
            int(tgt.shape[0]),
        )


class MockProxyLM:
    """Hash-and-length stand-in so E7 is runnable without a GPU."""

    model_id = "mock"

    def score_text(self, text: str) -> tuple[float, float, int]:
        h = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") / 2**32
        return -1.0 - 2.0 * h, 1.0 + 2.0 * (1 - h), max(len(text.split()), 1)


def score_records(records: Sequence[Record], lm) -> tuple[list[SurrogateRow], dict[str, list[StepScore]]]:
    """Score every step and emit rows in the same shape the attribution rules eat."""
    rows: list[SurrogateRow] = []
    by_file: dict[str, list[StepScore]] = {}
    for rec in records:
        for s in rec.steps:
            lp, ent, n = lm.score_text(s.content or "")
            rows.append(SurrogateRow(rec.key, s.idx, lp, ent, n))
            by_file.setdefault(rec.key, []).append(
                StepScore(
                    subset=rec.subset,
                    file_id=rec.file_id,
                    step_idx=s.idx,
                    agent=s.agent,
                    type_norm=s.type_norm,
                    type_source=s.type_source,
                    p_raw=_squash(lp),
                    judge=f"surrogate:{getattr(lm, 'model_id', '?')}",
                    readout="surrogate_logprob",
                    policy="plain",
                )
            )
    return rows, by_file


def entropy_scores(rows: Sequence[SurrogateRow], records: Sequence[Record]) -> dict[str, list[StepScore]]:
    """The entropy arm of the same baseline, as a separate score field.

    Higher entropy should mean less reliable, so it enters as ``1 − normalised
    entropy`` to keep the "higher p is better" convention every rule assumes.
    """
    by_key = {r.key: r for r in records}
    ents = [r.mean_entropy for r in rows] or [0.0]
    lo, hi = min(ents), max(ents)
    span = (hi - lo) or 1.0
    out: dict[str, list[StepScore]] = {}
    for row in rows:
        rec = by_key.get(row.key)
        if rec is None or row.step_idx >= rec.n_steps:
            continue
        step = rec.steps[row.step_idx]
        out.setdefault(row.key, []).append(
            StepScore(
                subset=rec.subset,
                file_id=rec.file_id,
                step_idx=row.step_idx,
                agent=step.agent,
                type_norm=step.type_norm,
                type_source=step.type_source,
                p_raw=1.0 - (row.mean_entropy - lo) / span,
                judge="surrogate:entropy",
                readout="surrogate_entropy",
                policy="plain",
            )
        )
    return out
