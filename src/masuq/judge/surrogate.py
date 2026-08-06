"""Surrogate-intrinsic baseline (spec §3).

The honest position: on frozen logs we cannot recover the generating model's own
token distributions, so no true intrinsic uncertainty signal exists. What *is*
computable is how a third-party base LM scores the recorded text — per-step mean
token logprob and mean token entropy of ``content``. We report it as the only
intrinsic-flavoured signal available on frozen logs, and expect it to be weak.
Reporting it weak is the point: it is the control that shows the judge signal is
not just fluency.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..schema import Record


@dataclass(slots=True)
class SurrogateScore:
    key: str
    step_idx: int
    type_norm: str
    mean_logprob: float
    mean_entropy: float
    n_tokens: int
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def p_proxy(self) -> float:
        """Monotone squash of mean logprob into (0,1) so it can enter the same
        AUROC/noisy-OR machinery as a judge score. Purely a rescaling — it adds
        no information and is never calibrated."""
        return 1.0 / (1.0 + math.exp(-(self.mean_logprob + 2.0)))


class SurrogateLM:
    """Base-LM scorer for recorded step text."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_tokens: int = 1024,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "SurrogateLM needs the 'judge' extra: pip install -e '.[judge]'"
            ) from e
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
        """Return ``(mean_logprob, mean_entropy, n_tokens)`` for ``text``."""
        torch = self._torch
        ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_tokens
        ).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return 0.0, 0.0, int(ids.shape[1])
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0, :-1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        targets = ids[0, 1:]
        tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        entropy = -(logprobs.exp() * logprobs).sum(-1)
        return float(tok_lp.mean()), float(entropy.mean()), int(targets.shape[0])

    def score_record(self, record: Record) -> list[SurrogateScore]:
        out = []
        for s in record.steps:
            lp, ent, n = self.score_text(s.content or "")
            out.append(
                SurrogateScore(
                    key=record.key,
                    step_idx=s.idx,
                    type_norm=s.type_norm,
                    mean_logprob=lp,
                    mean_entropy=ent,
                    n_tokens=n,
                    model=self.model_id,
                )
            )
        return out


class MockSurrogateLM:
    """Length-and-hash based stand-in so the pipeline runs without a GPU."""

    model_id = "mock"

    def score_text(self, text: str) -> tuple[float, float, int]:
        import hashlib

        n = max(len(text.split()), 1)
        h = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") / 2**32
        return -1.0 - 2.0 * h, 1.0 + 2.0 * (1 - h), n

    def score_record(self, record: Record) -> list[SurrogateScore]:
        out = []
        for s in record.steps:
            lp, ent, n = self.score_text(s.content or "")
            out.append(
                SurrogateScore(
                    key=record.key,
                    step_idx=s.idx,
                    type_norm=s.type_norm,
                    mean_logprob=lp,
                    mean_entropy=ent,
                    n_tokens=n,
                    model=self.model_id,
                )
            )
        return out


def score_corpus(records: Sequence[Record], lm) -> list[SurrogateScore]:
    out: list[SurrogateScore] = []
    for r in records:
        out.extend(lm.score_record(r))
    return out
