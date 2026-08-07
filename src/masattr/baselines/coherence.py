"""B4 — judge-free semantic coherence fields (pilot baseline spec).

Two per-step error fields computed without any judge, so the grid can ask
whether the judge's contribution survives a purely semantic control.

* ``embed_divergence`` — cosine distance between step ``t``'s embedding and the
  centroid of steps ``0..t−1``. A step that departs from what the trajectory has
  been about scores high.
* ``nli_contradiction`` — contradiction probability from an off-the-shelf NLI
  model, premise = the last few prefix steps, hypothesis = step ``t``.

Both are strictly prefix-conditional, like W0, so they sit in the same
evaluation grid as the judge field. Both are converted to the same
"higher ``p`` is better" convention every attribution rule assumes, by taking
``p = 1 − error``.

Step 0 has no prefix. Rather than invent a score, it takes the field's neutral
value (0.5) and is flagged, because a fabricated first-step score would feed
straight into first-step-biased rules.
"""

from __future__ import annotations

from typing import Sequence

from ..judge.score import StepScore
from ..record import Record

#: How many prefix steps form the NLI premise. Bounded because the premise is
#: truncated to the model's input limit anyway, and older steps are less
#: relevant to whether step t contradicts the run so far.
NLI_PREMISE_STEPS = 5

#: Neutral score for a step with no prefix to compare against.
NEUTRAL = 0.5

MAX_CHARS = 2000


def _rows(record: Record, values: Sequence[float], field: str, model_id: str) -> list[StepScore]:
    return [
        StepScore(
            subset=record.subset,
            file_id=record.file_id,
            step_idx=s.idx,
            agent=s.agent,
            type_norm=s.type_norm,
            type_source=s.type_source,
            p_raw=float(v),
            judge=f"{field}:{model_id}",
            readout=field,
            policy="plain",
            parse_ok=(s.idx > 0),
        )
        for s, v in zip(record.steps, values)
    ]


class Embedder:
    """Sentence embeddings via transformers, mean-pooled."""

    def __init__(self, model_id: str, *, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).to(dev).eval()
        self.device = dev

    def encode(self, texts: Sequence[str]):
        torch = self._torch
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(emb, dim=-1)


def embed_divergence(record: Record, embedder: Embedder) -> list[StepScore]:
    """``p = 1 − cosine distance from the prefix centroid``."""
    torch = embedder._torch
    emb = embedder.encode([(s.content or "")[:MAX_CHARS] for s in record.steps])
    values = [NEUTRAL]
    for t in range(1, len(record.steps)):
        centroid = torch.nn.functional.normalize(emb[:t].mean(0, keepdim=True), dim=-1)
        cos = float((emb[t : t + 1] * centroid).sum())
        # cosine in [-1,1] -> distance in [0,1] -> p = 1 - distance
        values.append(1.0 - (1.0 - cos) / 2.0)
    return _rows(record, values, "embed_divergence", embedder.model_id)


class NLI:
    """Off-the-shelf cross-encoder NLI; returns P(contradiction)."""

    def __init__(self, model_id: str, *, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_id).to(dev).eval()
        )
        self.device = dev
        labels = getattr(self.model.config, "id2label", {}) or {}
        self.contradiction_idx = next(
            (i for i, name in labels.items() if str(name).lower().startswith("contradiction")),
            0,
        )

    def contradiction(self, premise: str, hypothesis: str) -> float:
        torch = self._torch
        batch = self.tokenizer(
            premise, hypothesis, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**batch).logits[0]
        return float(torch.softmax(logits, dim=-1)[self.contradiction_idx])


def nli_contradiction(record: Record, nli: NLI) -> list[StepScore]:
    """``p = 1 − P(contradiction | last few prefix steps, step t)``."""
    values = [NEUTRAL]
    for t in range(1, len(record.steps)):
        prefix = record.steps[max(0, t - NLI_PREMISE_STEPS) : t]
        premise = "\n".join((s.content or "")[:MAX_CHARS] for s in prefix)
        hypothesis = (record.steps[t].content or "")[:MAX_CHARS]
        values.append(1.0 - nli.contradiction(premise, hypothesis))
    return _rows(record, values, "nli_contradiction", nli.model_id)


def score_records(records: Sequence[Record], field: str, model) -> dict[str, list[StepScore]]:
    fn = {"embed_divergence": embed_divergence, "nli_contradiction": nli_contradiction}[field]
    return {rec.key: fn(rec, model) for rec in records}
