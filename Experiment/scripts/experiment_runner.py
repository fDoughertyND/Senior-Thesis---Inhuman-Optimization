"""Generic LLM experiment runner.

An experiment is a cartesian product of axes (e.g. name, credit_score)
crossed with a set of models, repeated `trials` times. Every resulting
trial is an independent API call, so the runner flattens the whole thing
to a task list and fires it with one thread pool per provider, all
providers in parallel.

Usage:

    from experiment_runner import Experiment, MODELS, run

    PREFIX = "...{full_name}..."
    SUFFIX = "...{credit_score}..."

    exp = Experiment(
        axes={"name": names, "credit_score": list(range(580, 701, 5))},
        models={k: MODELS[k] for k in ("grok", "gpt", "claude", "deepseek", "gemini")},
        trials=1,
        prompt_builder=lambda ax, cfg: (
            PREFIX.format(full_name=ax["name"]),
            SUFFIX.format(credit_score=ax["credit_score"]),
        ),
    )

    results = run(exp)
    # results: list of {"axes": {...}, "model": ..., "trial": i,
    #                   "answer": str, "cached_in": int, "total_in": int}
"""
import os
import itertools
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import openai
import anthropic
from google import genai


# --- Secrets + model registry ----------------------------------------------

os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["XAI_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""


MODELS = {
    "grok": {
        "provider": "openai_compat",
        "model_id": "grok-4-1-fast",
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
    },
    "gpt": {
        "provider": "openai",
        "model_id": "gpt-5.4",
        "api_key_env": "OPENAI_API_KEY",
    },
    "claude": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "deepseek": {
        "provider": "openai_compat",
        "model_id": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "gemini": {
        "provider": "google",
        "model_id": "gemini-2.5-flash-lite",
        "api_key_env": "GEMINI_API_KEY",
    },
}


# --- Provider dispatch -----------------------------------------------------

_client_cache = {}


def _client(model_cfg: dict):
    key = (model_cfg["provider"], model_cfg["model_id"], model_cfg.get("base_url"))
    if key in _client_cache:
        return _client_cache[key]

    api_key = os.environ[model_cfg["api_key_env"]]
    provider = model_cfg["provider"]
    # A per-request timeout so a stuck socket raises instead of blocking the
    # worker thread forever (which would defeat the retry loop entirely).
    # 120 seconds is well above normal completion latency for a Yes/No prompt.
    timeout = 120.0
    if provider in ("openai", "openai_compat"):
        c = openai.OpenAI(api_key=api_key, base_url=model_cfg.get("base_url"),
                          timeout=timeout)
    elif provider == "anthropic":
        c = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    elif provider == "google":
        c = genai.Client(api_key=api_key)  # google SDK uses its own defaults
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    _client_cache[key] = c
    return c


def query_model(model_cfg: dict, prefix: str, suffix: str):
    """One completion. Returns (answer, cached_input_tokens, total_input_tokens).

    prefix is marked as the cache breakpoint on Anthropic and is the portion
    that should be stable across calls in a batch (so prefix caching can fire
    on every provider that supports it).
    """
    provider = model_cfg["provider"]
    c = _client(model_cfg)

    if provider in ("openai", "openai_compat"):
        r = c.chat.completions.create(
            model=model_cfg["model_id"],
            messages=[{"role": "user", "content": prefix + suffix}],
            temperature=1,
        )
        answer = r.choices[0].message.content.strip()
        total = getattr(r.usage, "prompt_tokens", 0) or 0
        details = getattr(r.usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        return answer, cached, total

    if provider == "anthropic":
        # Anthropic rejects empty "text" blocks — skip the suffix block when
        # the caller passed "" (otherwise every call errors out, hits the
        # retry/backoff loop, and looks like a hang).
        content = [
            {"type": "text", "text": prefix,
             "cache_control": {"type": "ephemeral"}},
        ]
        if suffix:
            content.append({"type": "text", "text": suffix})
        r = c.messages.create(
            model=model_cfg["model_id"],
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
            temperature=1,
        )
        answer = r.content[0].text.strip()
        u = r.usage
        inp = getattr(u, "input_tokens", 0) or 0
        crd = getattr(u, "cache_read_input_tokens", 0) or 0
        ccr = getattr(u, "cache_creation_input_tokens", 0) or 0
        return answer, crd, inp + crd + ccr

    if provider == "google":
        r = c.models.generate_content(
            model=model_cfg["model_id"],
            contents=prefix + suffix,
            config=genai.types.GenerateContentConfig(temperature=1),
        )
        answer = r.text.strip()
        u = getattr(r, "usage_metadata", None)
        total = (getattr(u, "prompt_token_count", 0) or 0) if u else 0
        cached = (getattr(u, "cached_content_token_count", 0) or 0) if u else 0
        return answer, cached, total

    raise ValueError(f"Unsupported provider: {provider}")


# --- Preflight check -------------------------------------------------------

def preflight(models_dict: dict, prompt: str = "Respond with just the word: ok"):
    """Fire one trivial prompt at each model, sequentially, before a real run.

    Purpose: fail fast if any provider is broken (auth, out of credits, bad
    model id, connectivity), so we don't burn tokens on working providers
    while one silently eats the retry budget. Prints a one-line status per
    model and raises RuntimeError on the first failure.
    """
    print("preflight check...")
    for name, cfg in models_dict.items():
        try:
            answer, _, _ = query_model(cfg, prompt, "")
        except Exception as e:
            msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            print(f"  ✗ {name}: {type(e).__name__}: {msg}", flush=True)
            raise RuntimeError(
                f"preflight failed for {name!r} — fix before running the "
                f"full experiment so other providers aren't wasted."
            )
        short = answer.strip().splitlines()[0][:40] if answer else "<empty>"
        print(f"  ✓ {name}: {short!r}", flush=True)
    print("preflight ok\n", flush=True)


# --- Experiment + runner ---------------------------------------------------

@dataclass
class Experiment:
    axes: dict                 # {dim_name: [values]}
    models: dict               # {model_name: model_cfg}
    trials: int                # repetitions per (axes, model)
    prompt_builder: Callable   # (axes_dict, model_cfg) -> (prefix, suffix)


def _tasks(exp: Experiment):
    keys = list(exp.axes.keys())
    for combo in itertools.product(*exp.axes.values()):
        ax = dict(zip(keys, combo))
        for model_name in exp.models:
            for i in range(exp.trials):
                yield ax, model_name, i


# --- Concurrency presets ---------------------------------------------------
# Per-provider in-flight worker cap. Tweak these in-file as tier limits or
# transient throttling change; every experiment that calls run(exp) picks the
# new values up automatically. The "default" key covers any provider bucket
# not listed explicitly (used for forward compatibility if a new provider is
# added without updating this dict).

DEFAULT_WORKERS = {
    "openai":        50,   # gpt
    "anthropic":     10,   # claude — easiest to rate-limit; keep conservative
    "google":        50,   # gemini
    "openai_compat": 30,   # grok + deepseek SHARE this bucket (one pool)
    "default":       50,
}


def run(exp: Experiment, max_workers_per_provider=None,
        skip_preflight: bool = False) -> list[dict]:
    """Execute every trial. Providers run in parallel; trials within a
    provider run concurrently in that provider's own thread pool.

    `max_workers_per_provider` can be:
      - None (default): use the DEFAULT_WORKERS preset above.
      - an int: the same cap applies to every provider (overrides preset).
      - a dict: per-provider caps, e.g. {"anthropic": 5, "default": 50}.
        Providers missing fall back to the dict's "default" key, or 50.

    Runs a sequential preflight check against every model first (one trivial
    prompt each) and bails out if any provider is broken — so a missing key
    or empty Anthropic balance doesn't cause the batch to burn tokens
    elsewhere. Pass skip_preflight=True to bypass (not recommended).
    """
    if max_workers_per_provider is None:
        max_workers_per_provider = DEFAULT_WORKERS
    def _workers_for(provider: str) -> int:
        if isinstance(max_workers_per_provider, int):
            return max_workers_per_provider
        return max_workers_per_provider.get(
            provider, max_workers_per_provider.get("default", 50),
        )
    if not skip_preflight:
        preflight(exp.models)
    tasks = list(_tasks(exp))

    by_provider: dict[str, list] = {}
    for ax, model_name, idx in tasks:
        provider = exp.models[model_name]["provider"]
        by_provider.setdefault(provider, []).append((ax, model_name, idx))

    def run_one(ax, model_name, idx, max_attempts=10, base_delay=1.0, max_delay=60.0):
        cfg = exp.models[model_name]
        prefix, suffix = exp.prompt_builder(ax, cfg)
        answer, cached, total = "N/A", 0, 0
        for attempt in range(max_attempts):
            try:
                answer, cached, total = query_model(cfg, prefix, suffix)
                break
            except Exception as e:
                # Print a one-liner so the error is visible — otherwise all
                # API failures look like the run silently hanging. Keep the
                # message short so the console stays readable.
                msg = str(e).splitlines()[0][:160] if str(e) else type(e).__name__
                print(f"[retry {attempt + 1}/{max_attempts}] "
                      f"{model_name} {type(e).__name__}: {msg}",
                      flush=True)
                if attempt == max_attempts - 1:
                    break
                # Exponential backoff with jitter, capped: ~1, 2, 4, 8, 16,
                # 32, then clamped at 60s.
                delay = min(base_delay * (2 ** attempt), max_delay)
                delay *= 0.7 + 0.6 * random.random()
                time.sleep(delay)
        return {
            "axes": ax,
            "model": model_name,
            "trial": idx,
            "answer": answer,
            "cached_in": cached,
            "total_in": total,
        }

    def drain(provider, provider_tasks):
        n = _workers_for(provider)
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(run_one, *t) for t in provider_tasks]
            return [f.result() for f in as_completed(futs)]

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, len(by_provider))) as top:
        fut_to_provider = {top.submit(drain, p, ts): p for p, ts in by_provider.items()}
        for fut in as_completed(fut_to_provider):
            provider = fut_to_provider[fut]
            chunk = fut.result()
            results.extend(chunk)
            cached = sum(r["cached_in"] for r in chunk)
            total = sum(r["total_in"] for r in chunk)
            if total:
                pct = 100 * cached / total
                print(f"{provider}: {cached}/{total} input tokens cached ({pct:.0f}%)")
    return results


# --- Aggregation helper ----------------------------------------------------

def aggregate_yes_rate(results: list[dict], group_by: list[str],
                       positive: str = "yes", min_responses: int = 1,
                       fail_threshold: int = 5) -> dict:
    """Group trials by (axes[group_by...], model) and compute Yes-rate.

    Returns {(ax_val_1, ax_val_2, ..., model_name): rate_or_-1}. A rate of
    -1 means the group had too many failures or not enough valid responses.
    """
    buckets: dict[tuple, dict] = {}
    for r in results:
        key = tuple(r["axes"][k] for k in group_by) + (r["model"],)
        b = buckets.setdefault(key, {"yes": 0, "resp": 0, "fail": 0})
        ans = r["answer"]
        norm = ans.strip().lower().replace("*", "") if isinstance(ans, str) else "n/a"
        if norm == "n/a":
            b["fail"] += 1
        else:
            b["resp"] += 1
            if norm == positive:
                b["yes"] += 1

    out = {}
    for key, b in buckets.items():
        if b["fail"] > fail_threshold or b["resp"] < min_responses:
            out[key] = -1
        else:
            out[key] = b["yes"] / b["resp"]
    return out
