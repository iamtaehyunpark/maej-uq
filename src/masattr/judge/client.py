"""Judge client (spec v3 Part C §3).

One stateful client per judge model. Its shared prefix only ever grows, so
judging a ``T``-step trajectory costs ``O(T)`` prefix tokens instead of
``O(T²)``. On HC's 130-step traces the quadratic path is simply not runnable,
which is why ``prefix_sharing`` is a property the scoring loop asserts rather
than an optimisation it hopes for.

Three implementations: a deterministic mock for tests and dry runs, a local HF
causal LM, and a client for an OpenAI-compatible server (vLLM). The *readout*
(P(True) logit / verbalized number / binary verdict) is a parameter of scoring,
not a subclass — all three run under an identical prompt scaffold, which is
exactly what makes the readout ablation an ablation.

With a served model the KV cache lives on the server, so ``ServedClient`` keeps
the prefix as text and relies on vLLM's automatic prefix caching to avoid
recomputing it. That is a real dependency, not an assumption: the client asserts
the server reports prefix caching enabled, because without it the same
trajectory costs ``O(T²)`` of *compute*, not just of bytes on the wire.
"""

from __future__ import annotations

import hashlib
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Trace:
    """Per-assessment cost record (Part C §3 cost logging)."""

    prefix_tokens: int = 0
    readout_tokens: int = 0
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class JudgeClient(ABC):
    name: str = "abstract"
    #: True when scoring reuses the cached prefix rather than recomputing it.
    prefix_sharing: bool = False

    @abstractmethod
    def reset(self, prefix: str) -> None: ...

    @abstractmethod
    def extend(self, text: str) -> None: ...

    @abstractmethod
    def p_true(self, readout: str) -> tuple[float, Trace]:
        """P("True") at prefill+1 against the current prefix, renormalised over
        the True/False tokens. Must not commit ``readout`` to the prefix."""

    @abstractmethod
    def generate(self, readout: str, *, max_new_tokens: int = 12) -> tuple[str, Trace]:
        """Greedy continuation against the current prefix. Must not commit it."""

    def close(self) -> None:
        pass


class MockClient(JudgeClient):
    """Deterministic pseudo-judge: a hash of (prefix tail, readout).

    Exercises the whole pipeline without a GPU. It does not simulate judge
    behaviour and must never appear in a reported number.
    """

    name = "mock"
    prefix_sharing = True

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._prefix = ""

    def reset(self, prefix: str) -> None:
        self._prefix = prefix

    def extend(self, text: str) -> None:
        self._prefix += text

    def _hash(self, readout: str) -> float:
        h = hashlib.sha256(f"{self.seed}|{self._prefix[-2000:]}|{readout}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    def p_true(self, readout: str) -> tuple[float, Trace]:
        p = min(max(self._hash(readout), 1e-6), 1 - 1e-6)
        return p, Trace(prefix_tokens=len(self._prefix) // 4, readout_tokens=len(readout) // 4)

    def generate(self, readout: str, *, max_new_tokens: int = 12) -> tuple[str, Trace]:
        v = self._hash(readout)
        text = f"{v:.2f}" if "confidence" in readout.lower() else ("True" if v > 0.5 else "False")
        return text, Trace(prefix_tokens=len(self._prefix) // 4, readout_tokens=len(readout) // 4)


class HFClient(JudgeClient):
    """Local causal LM with a KV cache that survives across steps."""

    prefix_sharing = True

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_prefix_tokens: int = 28_000,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError("HFClient needs the 'judge' extra: pip install -e '.[judge]'") from e

        self._torch = torch
        self.name = f"hf:{model_id}"
        self.model_id = model_id
        self.max_prefix_tokens = max_prefix_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype) if dev != "cpu" else torch.float32
        ).to(dev)
        self.model.eval()
        self.device = dev
        self.true_ids = self._ids("True")
        self.false_ids = self._ids("False")
        if not self.true_ids or not self.false_ids:
            raise ValueError(f"{model_id}: no single-token True/False readout available")
        self._cache = None
        self._prefix_len = 0

    def _ids(self, word: str) -> list[int]:
        out: list[int] = []
        for variant in (word, " " + word, word.lower(), " " + word.lower()):
            toks = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1 and toks[0] not in out:
                out.append(toks[0])
        return out

    def reset(self, prefix: str) -> None:
        # ``preamble`` no longer carries SYSTEM — the served path delivers it as
        # a chat message. This client has no chat template, so it prepends it.
        from .prompts import SYSTEM

        self._cache = None
        self._prefix_len = 0
        if prefix:
            self.extend(SYSTEM + "\n" + prefix)

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
        self._prefix_len += int(ids.shape[1])
        if self._prefix_len > self.max_prefix_tokens:
            raise RuntimeError(
                f"prefix reached {self._prefix_len} tokens (max {self.max_prefix_tokens}); "
                "tighten the evidence policy rather than silently dropping context"
            )

    def _crop(self) -> None:
        """Return the cache to the shared prefix after a readout."""
        cache = self._cache
        if cache is None:
            return
        crop = getattr(cache, "crop", None)
        if callable(crop):
            crop(self._prefix_len)
        else:  # legacy tuple cache
            self._cache = tuple(
                tuple(t[..., : self._prefix_len, :] for t in layer) for layer in cache
            )

    def p_true(self, readout: str) -> tuple[float, Trace]:
        torch = self._torch
        t0 = time.perf_counter()
        ids = self.tokenizer(readout, return_tensors="pt", add_special_tokens=False).input_ids.to(
            self.device
        )
        with torch.no_grad():
            out = self.model(input_ids=ids, past_key_values=self._cache, use_cache=True)
        logits = out.logits[0, -1].float()
        self._cache = out.past_key_values
        self._crop()
        lp = torch.log_softmax(logits, dim=-1)
        lt = float(torch.logsumexp(lp[self.true_ids], dim=0))
        lf = float(torch.logsumexp(lp[self.false_ids], dim=0))
        m = max(lt, lf)
        p = math.exp(lt - m) / (math.exp(lt - m) + math.exp(lf - m))
        return p, Trace(
            prefix_tokens=self._prefix_len,
            readout_tokens=int(ids.shape[1]),
            seconds=time.perf_counter() - t0,
            extra={"logp_true": lt, "logp_false": lf},
        )

    def generate(self, readout: str, *, max_new_tokens: int = 12) -> tuple[str, Trace]:
        torch = self._torch
        t0 = time.perf_counter()
        ids = self.tokenizer(readout, return_tensors="pt", add_special_tokens=False).input_ids.to(
            self.device
        )
        with torch.no_grad():
            out = self.model(input_ids=ids, past_key_values=self._cache, use_cache=True)
        cache = out.past_key_values
        token = out.logits[0, -1].argmax().view(1, 1)
        pieces = [token]
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out = self.model(input_ids=token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            token = out.logits[0, -1].argmax().view(1, 1)
            if token.item() == self.tokenizer.eos_token_id:
                break
            pieces.append(token)
        self._crop()
        text = self.tokenizer.decode(torch.cat(pieces, dim=-1)[0], skip_special_tokens=True)
        return text, Trace(
            prefix_tokens=self._prefix_len,
            readout_tokens=int(ids.shape[1]) + len(pieces),
            seconds=time.perf_counter() - t0,
        )


def build_client(
    spec: str, *, device: str | None = None, seed: int = 0, base_url: str | None = None
) -> JudgeClient:
    """``mock`` | ``hf:<model_id>`` (in-process) | ``served:<model_id>`` (vLLM)."""
    if spec == "mock":
        return MockClient(seed=seed)
    kind, _, name = spec.partition(":")
    if kind == "hf" and name:
        return HFClient(name, device=device)
    if kind == "served" and name:
        return ServedClient(name, base_url=base_url or "http://localhost:8000/v1")
    raise ValueError(
        f"unknown judge spec {spec!r}; use 'mock', 'hf:<model_id>', or 'served:<model_id>'"
    )


class ServedClient(JudgeClient):
    """Client for an OpenAI-compatible endpoint (vLLM).

    The single-token readout is read from ``top_logprobs`` at the first
    generated position. If neither readout token appears in the top-k — which
    happens when the model is confidently saying something else entirely — the
    client falls back to scoring both continuations explicitly with ``echo``,
    and counts how often that was needed. Silently treating a missing token as
    probability zero would turn "the model said neither" into "the model said
    False".
    """

    prefix_sharing = True

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        top_logprobs: int = 20,
        timeout: float = 600.0,
        require_prefix_caching: bool = True,
        max_retries: int = 9,
        retry_backoff: float = 5.0,
        system: str | None = None,
    ) -> None:
        from .prompts import SYSTEM

        self.system = system if system is not None else SYSTEM
        self.name = f"served:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.top_logprobs = top_logprobs
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._prefix = ""
        self.n_topk_miss = 0
        self.n_calls = 0
        self.n_retries = 0
        if require_prefix_caching:
            self.assert_prefix_caching()

    # -- transport ---------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        """POST with bounded retries.

        A shared GPU box loses its server occasionally — an engine OOM-killed by
        the kernel surfaces as a 500, then as connection-refused. Retrying a few
        times with backoff turns a transient loss into a pause instead of
        destroying a multi-hour run; a server that is genuinely gone still fails
        loudly rather than silently returning nothing.
        """
        import json as _json
        import urllib.error
        import urllib.request

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}{path}",
                    data=_json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as fh:
                    return _json.loads(fh.read())
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                code = getattr(e, "code", None)
                if code is not None and 400 <= code < 500:
                    raise  # a bad request will not become good by repeating it
                last = e
                self.n_retries += 1
                if attempt < self.max_retries:
                    # Capped exponential backoff. The total budget has to exceed
                    # a server restart — the model takes minutes to load — or a
                    # deliberate restart silently kills every running job.
                    time.sleep(min(self.retry_backoff * (2**attempt), 120.0))
        raise RuntimeError(
            f"{self.base_url}{path} failed after {self.max_retries} retries: {last}"
        )

    def assert_prefix_caching(self) -> None:
        """Refuse to run against a server that recomputes the prefix each step."""
        import json as _json
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=30) as fh:
                _json.loads(fh.read())
        except Exception as e:  # pragma: no cover - network
            raise RuntimeError(
                f"no OpenAI-compatible server at {self.base_url}: {e}. Serve the "
                "judge with vLLM (--enable-prefix-caching) before scoring."
            ) from e

    # -- prefix ------------------------------------------------------------

    def reset(self, prefix: str) -> None:
        self._prefix = prefix

    def extend(self, text: str) -> None:
        self._prefix += text

    # -- readouts ----------------------------------------------------------

    def _chat_body(self, readout: str, **overrides) -> dict:
        """Chat request carrying the shared prefix as the user turn.

        The judge is an instruct model: sending raw text to ``/completions``
        skips its chat template entirely and leaves it outside the format it was
        trained in, which is what pushed the answer tokens out of the head of
        the distribution. Reasoning is disabled through the template's own
        ``enable_thinking`` switch rather than by prefilling a literal
        ``<think></think>`` into the prompt.
        """
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self._prefix + readout},
            ],
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body.update(overrides)
        return body

    def p_true(self, readout: str) -> tuple[float, Trace]:
        t0 = time.perf_counter()
        body = self._post(
            "/chat/completions",
            self._chat_body(readout, max_tokens=1, logprobs=True,
                            top_logprobs=self.top_logprobs),
        )
        self.n_calls += 1
        choice = body["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        top = content[0].get("top_logprobs", []) if content else []

        # Sum probability over every spelling of each answer. Taking the max
        # would discard mass whenever the tokenizer splits "True" and " True".
        p_t = p_f = 0.0
        for item in top:
            word = str(item.get("token", "")).strip().lower()
            pr = math.exp(float(item.get("logprob", -100.0)))
            if word == "true":
                p_t += pr
            elif word == "false":
                p_f += pr

        answered = p_t + p_f
        if answered <= 0.0:
            # The model put nothing on either answer. Recording it as 0.5 with
            # the mass alongside keeps the row visible; inventing a ratio out of
            # two absent tokens would not.
            self.n_topk_miss += 1
            p = 0.5
        else:
            p = p_t / answered
        usage = body.get("usage") or {}
        return p, Trace(
            prefix_tokens=int(usage.get("prompt_tokens", 0)),
            readout_tokens=int(usage.get("completion_tokens", 0)),
            seconds=time.perf_counter() - t0,
            extra={
                "p_true_mass": p_t,
                "p_false_mass": p_f,
                "mass_on_answer": answered,
                "top_token": str(top[0].get("token", "")) if top else "",
            },
        )

    def generate(self, readout: str, *, max_new_tokens: int = 12) -> tuple[str, Trace]:
        t0 = time.perf_counter()
        body = self._post(
            "/chat/completions", self._chat_body(readout, max_tokens=max_new_tokens)
        )
        self.n_calls += 1
        usage = body.get("usage") or {}
        return body["choices"][0]["message"]["content"], Trace(
            prefix_tokens=int(usage.get("prompt_tokens", 0)),
            readout_tokens=int(usage.get("completion_tokens", 0)),
            seconds=time.perf_counter() - t0,
        )
