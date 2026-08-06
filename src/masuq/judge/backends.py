"""Judge backends (spec §3).

The judge scores every step of a trajectory *prefix-conditionally*: step ``t`` is
assessed given the query plus steps ``0..t``. Done naively that is O(T²) tokens.
The interface here is therefore a **prefix scorer** — a stateful object whose
shared prefix only ever grows, so the KV cache from step ``t`` is reused for step
``t+1``. On W&W-HC (up to 130 steps) this is the difference between a feasible
and an infeasible pilot, which is why the spec calls prefix sharing mandatory.

Three implementations:

* :class:`MockPrefixScorer` — deterministic, dependency-free; drives tests and
  the smoke run so the pipeline can be exercised without a GPU.
* :class:`HFPrefixScorer` — the real thing: a local causal LM, single-token
  ``True``/``False`` readout at prefill+1, cache cropped back after each readout.
* :class:`VerbalizedPrefixScorer` — the baseline row of spec §3: same prompt,
  but the model *says* a number instead of us reading its logits.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ScoreTrace:
    """Per-assessment bookkeeping for the cost paragraph (spec §3)."""

    prefix_tokens: int = 0
    readout_tokens: int = 0
    seconds: float = 0.0
    cache_hit: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class PrefixScorer(ABC):
    """Scores readout prompts against a monotonically growing shared prefix."""

    #: Human-readable identity written into every result row.
    name: str = "abstract"

    @abstractmethod
    def reset(self, prefix: str) -> None:
        """Start a new trajectory with ``prefix`` as the initial shared context."""

    @abstractmethod
    def extend(self, text: str) -> None:
        """Append ``text`` to the shared prefix, keeping the cache warm."""

    @abstractmethod
    def p_true(self, readout: str) -> tuple[float, ScoreTrace]:
        """Score one readout prompt against the current prefix; do not commit it."""

    # -- optional -----------------------------------------------------------

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self) -> "PrefixScorer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Mock


class MockPrefixScorer(PrefixScorer):
    """Deterministic pseudo-judge.

    Scores are a hash of (prefix tail, readout) squashed to (0, 1), plus an
    optional weak signal from ``answer_key`` so tests can assert that a working
    pipeline separates correct from incorrect trajectories. It is *not* a
    simulator of judge behaviour and must never appear in a reported number.
    """

    name = "mock"

    def __init__(self, *, seed: int = 0, signal: float = 0.0) -> None:
        self.seed = seed
        self.signal = signal
        self._prefix = ""
        self._n_extends = 0

    def reset(self, prefix: str) -> None:
        self._prefix = prefix
        self._n_extends = 0

    def extend(self, text: str) -> None:
        self._prefix += text
        self._n_extends += 1

    def p_true(self, readout: str) -> tuple[float, ScoreTrace]:
        tail = self._prefix[-2000:]
        h = hashlib.sha256(f"{self.seed}|{tail}|{readout}".encode()).digest()
        base = int.from_bytes(h[:8], "big") / 2**64
        if self.signal:
            # Nudge toward 1.0 when the readout context contains an explicit
            # correctness hint; used only by fixtures.
            hint = 1.0 if "[[CORRECT]]" in tail else 0.0
            base = (1 - self.signal) * base + self.signal * hint
        p = min(max(base, 1e-6), 1 - 1e-6)
        return p, ScoreTrace(
            prefix_tokens=len(tail) // 4,
            readout_tokens=len(readout) // 4,
            cache_hit=self._n_extends > 0,
        )


# ---------------------------------------------------------------------------
# HuggingFace


class HFPrefixScorer(PrefixScorer):
    """Local causal-LM judge with a single-token ``True``/``False`` readout.

    The score is ``P("True") / (P("True") + P("False"))`` at the first generated
    position (prefill+1), renormalised over the two readout tokens so that
    probability mass the model spends on formatting never leaks into the signal.
    """

    name = "hf"

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        true_token: str = "True",
        false_token: str = "False",
        max_prefix_tokens: int = 28_000,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "HFPrefixScorer needs the 'judge' extra: pip install -e '.[judge]'"
            ) from e

        self._torch = torch
        self.model_id = model_id
        self.name = f"hf:{model_id}"
        self.max_prefix_tokens = max_prefix_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if resolved_device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(resolved_device)
        self.model.eval()
        self.device = resolved_device

        self.true_ids = self._readout_ids(true_token)
        self.false_ids = self._readout_ids(false_token)
        if not self.true_ids or not self.false_ids:
            raise ValueError(f"{model_id}: cannot resolve single-token readout ids")

        self._cache = None
        self._prefix_len = 0
        self._pending = ""

    def _readout_ids(self, word: str) -> list[int]:
        """All single-token spellings of ``word`` (with and without a leading space)."""
        ids: list[int] = []
        for variant in (word, " " + word, word.lower(), " " + word.lower()):
            toks = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1 and toks[0] not in ids:
                ids.append(toks[0])
        return ids

    # -- prefix management --------------------------------------------------

    def reset(self, prefix: str) -> None:
        self._cache = None
        self._prefix_len = 0
        self._pending = ""
        if prefix:
            self.extend(prefix)

    def extend(self, text: str) -> None:
        if not text:
            return
        torch = self._torch
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=(self._prefix_len == 0)
        ).input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, past_key_values=self._cache, use_cache=True)
        self._cache = out.past_key_values
        self._prefix_len += ids.shape[1]
        if self._prefix_len > self.max_prefix_tokens:
            raise RuntimeError(
                f"prefix grew to {self._prefix_len} tokens, past max_prefix_tokens="
                f"{self.max_prefix_tokens}; truncate the evidence policy rather than "
                "silently dropping context"
            )

    def _crop(self) -> None:
        """Return the cache to exactly the shared prefix after a readout."""
        cache = self._cache
        if cache is None:
            return
        crop = getattr(cache, "crop", None)
        if callable(crop):
            crop(self._prefix_len)
            return
        # Legacy tuple cache.
        self._cache = tuple(
            tuple(t[..., : self._prefix_len, :] for t in layer) for layer in cache
        )

    # -- scoring ------------------------------------------------------------

    def p_true(self, readout: str) -> tuple[float, ScoreTrace]:
        import time

        torch = self._torch
        t0 = time.perf_counter()
        ids = self.tokenizer(
            readout, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, past_key_values=self._cache, use_cache=True)
        logits = out.logits[0, -1].float()
        self._cache = out.past_key_values
        self._crop()

        logprobs = torch.log_softmax(logits, dim=-1)
        lt = torch.logsumexp(logprobs[self.true_ids], dim=0).item()
        lf = torch.logsumexp(logprobs[self.false_ids], dim=0).item()
        m = max(lt, lf)
        p = math.exp(lt - m) / (math.exp(lt - m) + math.exp(lf - m))

        return p, ScoreTrace(
            prefix_tokens=self._prefix_len,
            readout_tokens=int(ids.shape[1]),
            seconds=time.perf_counter() - t0,
            cache_hit=self._prefix_len > 0,
            extra={"logp_true": lt, "logp_false": lf},
        )

    def close(self) -> None:
        self._cache = None


# ---------------------------------------------------------------------------
# Verbalized confidence


_NUM_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*%?")


class VerbalizedPrefixScorer(PrefixScorer):
    """Baseline readout: ask for a confidence number and parse it (spec §3).

    Same evidence, same prompt shape as the logit readout — the only difference
    is where the number comes from. Unparseable generations return ``None``-like
    behaviour via a 0.5 fallback and are counted, because silently dropping them
    would flatter the baseline.
    """

    name = "verbalized"

    def __init__(self, generator: "TextGenerator", *, max_new_tokens: int = 12) -> None:
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.name = f"verbalized:{generator.name}"
        self._prefix = ""
        self.n_unparsed = 0

    def reset(self, prefix: str) -> None:
        self._prefix = prefix

    def extend(self, text: str) -> None:
        self._prefix += text

    def p_true(self, readout: str) -> tuple[float, ScoreTrace]:
        import time

        t0 = time.perf_counter()
        text = self.generator.generate(self._prefix + readout, max_new_tokens=self.max_new_tokens)
        m = _NUM_PAT.search(text or "")
        if not m:
            self.n_unparsed += 1
            return 0.5, ScoreTrace(
                seconds=time.perf_counter() - t0, extra={"raw": text, "parsed": False}
            )
        val = float(m.group(1))
        if val > 1.0:
            val /= 100.0
        p = min(max(val, 0.0), 1.0)
        return p, ScoreTrace(
            seconds=time.perf_counter() - t0, extra={"raw": text, "parsed": True}
        )


class TextGenerator(ABC):
    """Minimal generation interface used by the verbalized baseline and W&W repro."""

    name: str = "abstract"

    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str: ...


class MockGenerator(TextGenerator):
    name = "mock"

    def __init__(self, reply: str = "0.5") -> None:
        self.reply = reply

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        h = hashlib.sha256(prompt.encode()).digest()
        return f"{int.from_bytes(h[:2], 'big') % 101 / 100:.2f}"


class HFGenerator(TextGenerator):  # pragma: no cover - env dependent
    def __init__(self, model_id: str, *, device: str | None = None, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model_id}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype) if dev != "cpu" else torch.float32
        ).to(dev)
        self.model.eval()
        self.device = dev
        self._torch = torch

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        torch = self._torch
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


class OpenAIGenerator(TextGenerator):  # pragma: no cover - network
    """Used for the Who&When baseline reproduction at the README default (gpt-4o).

    Credentials are passed explicitly (spec §6) rather than read from the
    environment, so a run's provenance is visible in its command line.
    """

    def __init__(self, model: str = "gpt-4o", *, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
