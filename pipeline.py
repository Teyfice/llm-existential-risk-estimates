#!/usr/bin/env python3
"""Ask every text model on OpenRouter the question in PROMPT.txt, one GO at a time.

Commands (see README.md):
    prepare                 download catalogue and freeze the plan (public, no tokens)
    go --number N           show what GO N would do (no tokens)
    go --number N --execute run GO N: one call per model and condition, then stop
    status --number N       show progress and spend of GO N
    export --number N       rebuild the CSV files of GO N from its raw log
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import signal
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
BASE_URL = "https://openrouter.ai/api/v1"
MAX_OUTPUT_TOKENS = 32 * 1024
PROMPT_TOKEN_ALLOWANCE = 1000  # upper estimate of prompt size, used for reserving money
CONTEXT_MARGIN = 1000  # room left for the prompt when a model's context is small
EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
CAUSES = ("malicious_use", "ai_race", "organizational_risks", "rogue_ai")
TEMPLATE_CAUSES = '"<malicious_use | ai_race | organizational_risks | rogue_ai>"'
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
TECHNICAL = "technical_failure"


@dataclass
class Config:
    data_dir: Path = ROOT / "data"
    prompt_file: Path = ROOT / "PROMPT.txt"
    key_file: Path = ROOT / "secret_key_openrouter.txt"
    inclusion_file: Path = ROOT / "model_inclusion.csv"
    base_url: str = BASE_URL
    budget: float = 60.0
    concurrency: int = 40
    call_timeout: float = 2400.0  # seconds for one call; long reasoning can take many minutes
    max_attempts: int = 4
    backoff_base: float = 5.0
    generation_delays: tuple = (2, 4, 8, 15, 30)
    status_interval: float = 15.0
    key_poll_interval: float = 30.0
    final_poll_delay: float = 10.0
    retry_failed: bool = False
    number_of_gos: int = 5  # the study runs GO 1 to GO 5


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def go_dir(cfg: Config, number: int) -> Path:
    return cfg.data_dir / f"go_{number:03d}"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    """Replace a file atomically; a brief Dropbox lock must not stop a run."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    for _ in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    print(f"WARNING: could not update {path}", file=sys.stderr)


def append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    for attempt in range(40):
        try:
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.25)


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_included(cfg: Config) -> set:
    """Models marked included = yes. Anything else, including a model not listed, is never called or exported.

    Accepts the file as written here (commas) or as saved by Excel (semicolons, possibly with a generic
    "Column1;Column2;..." line above the real header). Anything ambiguous stops the run instead of guessing.
    """
    raw = cfg.inclusion_file.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")  # Excel's "CSV (Trennzeichen-getrennt)" on a German system
    lines = text.splitlines()
    for start, line in enumerate(lines[:3]):
        for delimiter in (",", ";", "\t"):
            header = [h.strip().lower() for h in next(csv.reader([line], delimiter=delimiter))]
            if "model" in header and "included" in header:
                break
        else:
            continue
        break
    else:
        raise ValueError(f"{cfg.inclusion_file.name}: no header with 'model' and 'included' in the first lines")
    rows = list(csv.reader(lines[start + 1:], delimiter=delimiter))
    m, inc = header.index("model"), header.index("included")
    problems, seen, included = [], set(), set()
    for n, row in enumerate(rows, start=start + 2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) <= max(m, inc):
            problems.append(f"line {n}: too few columns")
            continue
        model, value = row[m].strip(), row[inc].strip().lower()
        if value not in ("yes", "no"):
            problems.append(f"line {n} ({model}): included = {row[inc]!r}, expected yes or no")
        if model in seen:
            problems.append(f"line {n}: {model} is listed twice")
        seen.add(model)
        if value == "yes":
            included.add(model)
    if problems:
        raise ValueError(f"{cfg.inclusion_file.name} has problems:\n  " + "\n  ".join(problems[:20]))
    return included


# ---------------------------------------------------------------- prompt

def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def build_prompt(base: str, order) -> str:
    """Return the prompt with the causes in `order`, in the list and in the JSON template."""
    matches = list(re.finditer(r'^   - "(\w+)":.*?(?=^   - "|^\s*$)', base, re.M | re.S))
    blocks = {m.group(1): m.group(0) for m in matches}
    if tuple(blocks) != CAUSES or base.count(TEMPLATE_CAUSES) != 1:
        raise ValueError("PROMPT.txt no longer has the expected cause list or JSON template")
    start, end = matches[0].start(), matches[-1].end()
    text = base[:start] + "".join(blocks[c] for c in order) + base[end:]
    return text.replace(TEMPLATE_CAUSES, '"<' + " | ".join(order) + '>"')


def call_id(go: int, model: str, condition: str) -> str:
    return hashlib.sha256(f"{go}|{model}|{condition}".encode()).hexdigest()[:16]


def cause_order(go: int, model: str, condition: str) -> list:
    """Random per call, but reproducible, so a restarted call gets the same prompt."""
    seed = int(hashlib.sha256(f"order|{go}|{model}|{condition}".encode()).hexdigest(), 16)
    return random.Random(seed).sample(list(CAUSES), len(CAUSES))


# ---------------------------------------------------------------- plan

def exclusion_reason(model: dict, all_ids: set):
    mid = model["id"]
    arch = model.get("architecture") or {}
    if "text" not in (arch.get("input_modalities") or []) or "text" not in (arch.get("output_modalities") or []):
        return "not_text"
    if mid.startswith("openrouter/") or float((model.get("pricing") or {}).get("prompt") or 0) < 0:
        return "router"
    if mid.endswith(":batch"):
        return "batch_duplicate"
    if mid.startswith("~") or model.get("alias_target"):
        return "alias_duplicate"
    if mid.endswith(":free") and mid[: -len(":free")] in all_ids:
        return "free_twin_of_paid_model"
    return None


def select_models(models: list) -> tuple:
    """Return (kept, excluded); excluded rows carry the reason."""
    all_ids = {m["id"] for m in models}
    kept, excluded, by_slug = [], [], {}
    for m in sorted(models, key=lambda x: (len(x["id"]), x["id"])):
        reason = exclusion_reason(m, all_ids)
        slug = m.get("canonical_slug")
        if not reason and slug and slug in by_slug:
            reason = f"same_canonical_slug_as:{by_slug[slug]}"
        if reason:
            excluded.append({"id": m["id"], "reason": reason})
            continue
        if slug:
            by_slug[slug] = m["id"]
        kept.append(m)
    return kept, excluded


def conditions_for(model: dict) -> list:
    """List the reasoning conditions of a model, from lowest to highest effort."""
    r = model.get("reasoning") or {}
    if "reasoning" not in (model.get("supported_parameters") or []):
        return [{"condition": "no_reasoning", "reasoning": None}]
    efforts = r.get("supported_efforts")
    if "supported_efforts" in r and efforts is None:  # documented: null means all values accepted
        efforts = EFFORT_ORDER
    if efforts:
        keep = [e for e in EFFORT_ORDER if e in efforts and not (e == "none" and r.get("mandatory"))]
        return [
            {
                "condition": f"effort_{e}",
                "reasoning": {"effort": e},
                "effort": e,
                "effort_rank": rank,
                "is_default_effort": e == r.get("default_effort"),
            }
            for rank, e in enumerate(keep)
        ]
    if r.get("mandatory"):
        return [{"condition": "reasoning_mandatory", "reasoning": None}]
    return [
        {"condition": "reasoning_off", "reasoning": {"enabled": False}},
        {"condition": "reasoning_on", "reasoning": {"enabled": True}},
    ]


def output_limit(model: dict) -> dict:
    """The lower of 32,768 and the model's limits, plus the parameter that carries it."""
    params = model.get("supported_parameters") or []
    top = (model.get("top_provider") or {}).get("max_completion_tokens")
    context = model.get("context_length")
    limits = {"study_limit": MAX_OUTPUT_TOKENS}
    if top:
        limits["provider_limit"] = top
    if context:
        limits["context_limit"] = max(1, context - CONTEXT_MARGIN)
    source = min(limits, key=limits.get)
    field = next((p for p in ("max_tokens", "max_completion_tokens") if p in params), None)
    return {"max_output_tokens": limits[source], "limit_source": source, "limit_parameter": field}


def worst_case_usd(model: dict, endpoints: list, limit: dict) -> float:
    """Most a single call could cost at listed prices, across every provider of the model."""
    prices = [model.get("pricing") or {}] + [e.get("pricing") or {} for e in endpoints]

    def highest(*keys):
        return max(max(float(p.get(k) or 0) for k in keys) for p in prices)

    tokens = limit["max_output_tokens"]
    if not limit["limit_parameter"]:  # the limit cannot be sent, so the model may use all it has
        top = (model.get("top_provider") or {}).get("max_completion_tokens")
        tokens = max(tokens, top or model.get("context_length") or tokens)
    return tokens * highest("completion", "internal_reasoning") + PROMPT_TOKEN_ALLOWANCE * highest("prompt")


def flatten(obj, prefix="") -> dict:
    out = {}
    for key, value in (obj or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        elif isinstance(value, list):
            out[name] = json.dumps(value, ensure_ascii=False)
        else:
            out[name] = value
    return out


def write_csv(path: Path, rows: list) -> None:
    columns = []
    for row in rows:
        columns.extend(c for c in row if c not in columns)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    for _ in range(12):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # e.g. the file is open in Excel; the raw log is unaffected
            time.sleep(0.25)
    print(f"WARNING: {path} is locked, so the new version is in {tmp}. `export` rebuilds it.", file=sys.stderr)


async def fetch_endpoints(client: httpx.AsyncClient, cfg: Config, model: dict, gate: asyncio.Semaphore) -> dict:
    path = (model.get("links") or {}).get("details")
    host = cfg.base_url.rsplit("/api/v1", 1)[0]
    url = host + path if path else f"{cfg.base_url}/models/{model['id']}/endpoints"
    async with gate:
        for attempt in range(4):
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return {"id": model["id"], "data": r.json().get("data")}
                error = f"HTTP {r.status_code}"
            except (httpx.HTTPError, ValueError) as e:
                error = repr(e)
            await asyncio.sleep(cfg.backoff_base * attempt)
    return {"id": model["id"], "data": None, "error": error}


async def prepare(cfg: Config, force: bool = False) -> dict:
    plan_path = cfg.data_dir / "plan.json"
    if plan_path.exists() and not force:
        raise SystemExit(f"{plan_path} already exists. The plan stays frozen between GOs (--force replaces it).")
    if force and list(cfg.data_dir.glob("go_*/raw.jsonl")):
        raise SystemExit("GOs already exist for this plan; move them away before replacing it.")
    base_prompt = load_prompt(cfg.prompt_file)
    build_prompt(base_prompt, CAUSES)
    catalogue_dir = cfg.data_dir / "catalogue"
    catalogue_dir.mkdir(parents=True, exist_ok=True)

    # Public endpoints: no API key is sent, so this cannot use tokens.
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{cfg.base_url}/models")
        r.raise_for_status()
        (catalogue_dir / "models.json").write_text(r.text, encoding="utf-8")
        models = r.json()["data"]
        kept, excluded = select_models(models)
        gate = asyncio.Semaphore(8)
        fetched = await asyncio.gather(*(fetch_endpoints(client, cfg, m, gate) for m in kept))
    endpoints = {e["id"]: e for e in fetched}
    write_json(catalogue_dir / "endpoints.json", endpoints)

    planned, metadata = [], []
    for m in kept:
        ep_data = endpoints[m["id"]].get("data") or {}
        eps = ep_data.get("endpoints") or []
        limit = output_limit(m)
        planned.append({
            "id": m["id"],
            "canonical_slug": m.get("canonical_slug"),
            "conditions": conditions_for(m),
            "reservation_usd": worst_case_usd(m, eps, limit),
            "typical_output_tokens": 3000 if "reasoning" in (m.get("supported_parameters") or []) else 400,
            "completion_price": float((m.get("pricing") or {}).get("completion") or 0),
            "prompt_price": float((m.get("pricing") or {}).get("prompt") or 0),
            **limit,
        })
        metadata.append({
            **flatten(m),
            **limit,
            "n_conditions": len(planned[-1]["conditions"]),
            "n_endpoints": len(eps),
            "providers": json.dumps([e.get("provider_name") for e in eps], ensure_ascii=False),
            "quantizations": json.dumps([e.get("quantization") for e in eps], ensure_ascii=False),
            "endpoints_error": endpoints[m["id"]].get("error"),
        })
    plan = {
        "created_at": now_iso(),
        "prompt_sha256": hashlib.sha256(base_prompt.encode()).hexdigest(),
        "n_catalogue": len(models),
        "models": planned,
    }
    write_json(plan_path, plan)
    write_csv(cfg.data_dir / "models_metadata.csv", metadata)
    write_csv(cfg.data_dir / "models_excluded.csv", excluded)

    calls = planned_calls(cfg)
    print(f"Catalogue entries: {len(models)}; kept: {len(kept)}; excluded: {len(excluded)}")
    print(f"Included through {cfg.inclusion_file.name}: {len({c['model']['id'] for c in calls})}")
    for reason, n in Counter(e["reason"].split(":")[0] for e in excluded).most_common():
        print(f"  excluded, {reason}: {n}")
    missing = sum(1 for e in fetched if not e.get("data"))
    print(f"Provider metadata missing for {missing} model(s)")
    print(f"Conditions (= calls per GO): {len(calls)}")
    print(f"Limit lowered below {MAX_OUTPUT_TOKENS}: {sum(1 for m in planned if m['max_output_tokens'] < MAX_OUTPUT_TOKENS)} model(s)")
    print(f"Limit cannot be sent: {sum(1 for m in planned if not m['limit_parameter'])} model(s)")
    print(f"Plan frozen: {plan_path}")
    return plan


def plan_calls(plan: dict) -> list:
    calls = []
    for m in plan["models"]:
        for c in m["conditions"]:
            calls.append({"model": m, **c})
    return calls


def planned_calls(cfg: Config) -> list:
    """The calls every GO makes: the same in each, so a model that fails is simply tried again next GO."""
    plan_path = cfg.data_dir / "plan.json"
    if not plan_path.exists():
        raise SystemExit("No plan yet. Run: python pipeline.py prepare")
    included = load_included(cfg)
    return [c for c in plan_calls(read_json(plan_path)) if c["model"]["id"] in included]


# ---------------------------------------------------------------- reading answers

def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text") or "" for p in content if isinstance(p, dict))
    return ""


def parse_answer(content: str) -> dict:
    """Read the JSON answer. `strict` means the reply was only JSON, as the prompt demanded."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return {"parse_mode": "strict", "answer": obj}
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    found = []
    for m in re.finditer(r"\{", content):
        try:
            obj, _ = decoder.raw_decode(content[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    wanted = [o for o in found if "probability" in o]
    if wanted or found:
        return {"parse_mode": "lenient", "answer": (wanted or found)[-1]}
    return {"parse_mode": None, "answer": None}


def is_probability(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1


def schema_problems(a: dict) -> list:
    problems = [k for k in ("probability", "lower_bound", "upper_bound") if not is_probability(a.get(k))]
    if a.get("cause") not in CAUSES:
        problems.append("cause")
    if not isinstance(a.get("explanation"), str):
        problems.append("explanation")
    return problems


def interpret(status: int, text: str) -> dict:
    """Turn one HTTP reply into an outcome. `retry` = worth trying again; `fatal` = stop the GO."""
    try:
        body = json.loads(text) if text else None
    except ValueError:
        body = None
    body = body if isinstance(body, dict) else None
    error = (body or {}).get("error")
    error = error if isinstance(error, dict) else ({"message": str(error)} if error else None)
    meta = (error or {}).get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    usage = (body or {}).get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    result = {
        "outcome": TECHNICAL, "retry": False, "fatal": False,
        "generation_id": (body or {}).get("id"),
        "usage_cost": cost if isinstance(cost, (int, float)) else None,
        "error_message": (error or {}).get("message"),
        "error_code": (error or {}).get("code"),
        "finish_reason": None,
    }
    code = status if status != 200 else (error or {}).get("code")

    if code in (401, 402):
        waitable = code == 402 and meta.get("limit_source") == "openrouter_in_flight_budget"
        return {**result, "retry": waitable, "fatal": not waitable}
    if code == 403:
        if meta.get("reasons") or meta.get("flagged_input"):
            return {**result, "outcome": "moderated"}
        return result
    if status != 200:
        return {**result, "retry": status in TRANSIENT_STATUS}
    if body is None:
        return {**result, "retry": True, "error_message": "200 reply was not JSON"}
    if error:
        return {**result, "retry": code in TRANSIENT_STATUS or code is None}
    choices = body.get("choices")
    if not choices:
        return {**result, "retry": True, "error_message": "200 reply had no choices"}
    choice = choices[0]
    if choice.get("error"):
        return {**result, "retry": True, "error_message": json.dumps(choice["error"], ensure_ascii=False)}
    message = choice.get("message") or {}
    content = text_of(message.get("content"))
    finish = choice.get("finish_reason")
    result["finish_reason"] = finish
    refused = bool(message.get("refusal")) or finish == "content_filter"
    if not content.strip():
        if refused:
            return {**result, "outcome": "refusal"}
        if finish == "length":
            return {**result, "outcome": "truncated_no_answer"}
        return {**result, "retry": True, "error_message": "empty answer"}
    parsed = parse_answer(content)
    if parsed["answer"] is None:
        return {**result, "outcome": "refusal" if refused else "invalid_json"}
    outcome = "schema_violation" if schema_problems(parsed["answer"]) else "valid"
    return {**result, "outcome": outcome}


def failure_info(message: str, retry: bool) -> dict:
    return {"outcome": TECHNICAL, "retry": retry, "fatal": False, "generation_id": None, "usage_cost": None,
            "error_message": message, "error_code": None, "finish_reason": None}


# ---------------------------------------------------------------- money

class Ledger:
    """Spend = confirmed cost, or OpenRouter's own usage figure if that is higher.

    Before a call starts, the most it could cost is reserved. A call only starts if
    spend + all reservations + its reservation stay within the budget, however many run in parallel.
    """

    def __init__(self, budget: float, prior: float = 0.0):
        self.budget, self.prior = budget, prior
        self.settled = self.reserved = self.server_delta = 0.0

    @property
    def spent(self) -> float:
        return self.prior + max(self.settled, self.server_delta)

    def fits(self, amount: float) -> bool:
        return self.spent + self.reserved + amount <= self.budget

    def over_budget(self) -> bool:
        return self.spent + self.reserved > self.budget


# ---------------------------------------------------------------- running a GO

class Runner:
    def __init__(self, cfg: Config, number: int, calls: list, prior_spend: float, base_prompt: str):
        self.cfg, self.number, self.base_prompt = cfg, number, base_prompt
        self.dir = go_dir(cfg, number)
        self.raw = self.dir / "raw.jsonl"
        self.calls = calls
        self.total = len(calls)
        self.ledger = Ledger(cfg.budget, prior_spend)
        self.outcomes = Counter()
        self.done = 0
        self.active = set()
        self.background = set()
        self.generation_gate = asyncio.Semaphore(10)
        self.stop_reason = None
        self.fatal_reason = None
        self.baseline = None
        self.started = time.monotonic()

    def request_body(self, call: dict, prompt: str) -> dict:
        m = call["model"]
        body = {
            "model": m["id"],
            "messages": [{"role": "user", "content": prompt}],
            "provider": {"require_parameters": True},
        }
        if m["limit_parameter"]:
            body[m["limit_parameter"]] = m["max_output_tokens"]
        if call["reasoning"] is not None:
            body["reasoning"] = call["reasoning"]
        return body

    async def fetch_generation(self, client, call, generation_id) -> dict:
        url = f"{self.cfg.base_url}/generation"
        record = {
            "type": "generation", "go": self.number, "generation_id": generation_id,
            "call_id": call_id(self.number, call["model"]["id"], call["condition"]),
        }
        async with self.generation_gate:
            for delay in self.cfg.generation_delays:
                await asyncio.sleep(delay)
                try:
                    r = await client.get(url, params={"id": generation_id})
                    record.update(http_status=r.status_code, response_text=r.text, fetched_at=now_iso())
                    if r.status_code == 200:
                        break
                except httpx.HTTPError as e:
                    record.update(error=repr(e))
        append_jsonl(self.raw, record)
        try:
            data = json.loads(record.get("response_text") or "{}").get("data") or {}
            cost = data.get("total_cost")
            return {"total_cost": cost if isinstance(cost, (int, float)) else None}
        except ValueError:
            return {"total_cost": None}

    async def one_attempt(self, client, call, attempt, prompt, order) -> dict:
        m = call["model"]
        body = self.request_body(call, prompt)
        record = {
            "type": "attempt", "go": self.number,
            "call_id": call_id(self.number, m["id"], call["condition"]),
            "model": m["id"], "canonical_slug": m["canonical_slug"], "condition": call["condition"],
            "attempt": attempt, "cause_order": order,
            "reservation_usd": m["reservation_usd"],
            "started_at": now_iso(),
            # The Authorization header is deliberately not logged.
            "request": {"url": f"{self.cfg.base_url}/chat/completions", "json": body},
        }
        t0 = time.monotonic()
        billable_unknown = False
        try:
            r = await asyncio.wait_for(
                client.post(f"{self.cfg.base_url}/chat/completions", json=body),
                timeout=self.cfg.call_timeout,
            )
            record.update(http_status=r.status_code, response_headers=dict(r.headers), response_text=r.text)
            try:
                info = interpret(r.status_code, r.text)
            except (AttributeError, TypeError, IndexError, KeyError) as e:  # reply had an unexpected shape
                info = failure_info(f"unreadable reply: {e!r}", retry=False)
            retry_after = r.headers.get("retry-after")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            # The request never reached OpenRouter, so it cannot have been billed.
            record.update(http_status=None, error=repr(e))
            info = failure_info(repr(e), retry=True)
            retry_after = None
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            # The request may be running and billed. Do not retry; assume the full reservation was spent.
            record.update(http_status=None, error=repr(e))
            info = failure_info(f"connection lost or timed out: {e!r}", retry=False)
            retry_after = None
            billable_unknown = True
        record["elapsed_s"] = round(time.monotonic() - t0, 3)
        record["finished_at"] = now_iso()
        record["outcome"] = info["outcome"]
        record["retryable"] = info["retry"]
        for key in ("generation_id", "error_message", "error_code", "finish_reason"):
            record[key] = info[key]

        cost, source = info["usage_cost"], "usage"
        if cost is None:
            if billable_unknown:
                cost, source = m["reservation_usd"], "reserved_amount_assumed"
            elif record.get("http_status") == 200 and info["generation_id"]:
                g = await self.fetch_generation(client, call, info["generation_id"])
                if g["total_cost"] is not None:
                    cost, source = g["total_cost"], "generation"
                else:
                    cost, source = m["reservation_usd"], "reserved_amount_assumed"
            elif record.get("http_status") == 200:
                cost, source = m["reservation_usd"], "reserved_amount_assumed"
            else:
                cost, source = 0.0, "error_reply_assumed_free"
        elif info["generation_id"]:
            task = asyncio.create_task(self.fetch_generation(client, call, info["generation_id"]))
            self.background.add(task)
            task.add_done_callback(self.background.discard)
        self.ledger.settled += cost
        record.update(cost_usd=cost, cost_source=source)
        return {"record": record, "info": info, "retry_after": retry_after}

    async def run_call(self, client, call) -> None:
        m = call["model"]
        order = cause_order(self.number, m["id"], call["condition"])
        prompt = build_prompt(self.base_prompt, order)
        try:
            for attempt in range(1, self.cfg.max_attempts + 1):
                got = await self.one_attempt(client, call, attempt, prompt, order)
                record, info = got["record"], got["info"]
                if info["fatal"]:
                    self.fatal_reason = f"HTTP {record.get('http_status')}: {info['error_message']}"
                    self.stop_reason = "fatal_api_error"
                    record["final"] = False
                    append_jsonl(self.raw, record)
                    return
                retry = info["retry"] and attempt < self.cfg.max_attempts
                if retry and (self.stop_reason or self.ledger.over_budget()):
                    record["final"] = False  # left for a restart, not counted as a failure
                    append_jsonl(self.raw, record)
                    return
                record["final"] = not retry
                append_jsonl(self.raw, record)
                if not retry:
                    self.done += 1
                    self.outcomes[info["outcome"]] += 1
                    return
                try:
                    wait = min(float(got["retry_after"]), 120.0)
                except (TypeError, ValueError):
                    wait = self.cfg.backoff_base * 3 ** (attempt - 1)
                await asyncio.sleep(wait * random.uniform(1.0, 1.25))
        except Exception as e:  # a bug must not lose a call silently; it stays missing for a restart
            self.outcomes["internal_error"] += 1
            append_jsonl(self.raw, {"type": "internal_error", "go": self.number, "model": m["id"],
                                    "condition": call["condition"], "error": repr(e), "at": now_iso()})
            print(f"INTERNAL ERROR on {m['id']} / {call['condition']}: {e!r}", file=sys.stderr, flush=True)
        finally:
            self.ledger.reserved -= m["reservation_usd"]

    async def poll_key(self, client) -> None:
        try:
            r = await client.get(f"{self.cfg.base_url}/key")
            usage = r.json()["data"]["usage"] if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            usage = None
        if usage is None:
            return
        if self.baseline is None:
            self.baseline = usage
        self.ledger.server_delta = max(self.ledger.server_delta, usage - self.baseline)

    def status(self) -> dict:
        lg = self.ledger
        return {
            "go": self.number, "updated_at": now_iso(),
            "calls_total": self.total, "calls_finished": self.done, "calls_running": len(self.active),
            "budget_usd": lg.budget, "spent_usd": round(lg.spent, 4),
            "earlier_sessions_usd": round(lg.prior, 4),
            "confirmed_this_session_usd": round(lg.settled, 4),
            "openrouter_usage_this_session_usd": round(lg.server_delta, 4),
            "reserved_usd": round(lg.reserved, 4),
            "outcomes": dict(self.outcomes), "stop_reason": self.stop_reason,
            "fatal_reason": self.fatal_reason,
            "elapsed_s": round(time.monotonic() - self.started, 1),
        }

    def report(self) -> None:
        s = self.status()
        append_jsonl(self.dir / "spend.jsonl", s)
        write_json(self.dir / "status.json", s)
        outcomes = ", ".join(f"{k}={v}" for k, v in sorted(s["outcomes"].items())) or "-"
        print(
            f"[{s['elapsed_s'] / 60:6.1f} min] {s['calls_finished']}/{s['calls_total']} done, "
            f"{s['calls_running']} running | spent ${s['spent_usd']:.2f} of ${s['budget_usd']:.2f}, "
            f"reserved ${s['reserved_usd']:.2f} | {outcomes}",
            flush=True,
        )

    async def monitor(self, client, finished: asyncio.Event) -> None:
        last_poll = time.monotonic()
        while not finished.is_set():
            try:
                await asyncio.wait_for(finished.wait(), self.cfg.status_interval)
            except asyncio.TimeoutError:
                pass
            if time.monotonic() - last_poll >= self.cfg.key_poll_interval:
                await self.poll_key(client)
                last_poll = time.monotonic()
            if (self.dir / "STOP").exists() and not self.stop_reason:
                self.stop_reason = "stop_file"
            if self.ledger.spent >= self.ledger.budget and not self.stop_reason:
                self.stop_reason = "budget"
            self.report()

    async def schedule(self, client) -> None:
        pending = deque(self.calls)
        while pending or self.active:
            while pending and not self.stop_reason and len(self.active) < self.cfg.concurrency:
                call = pending[0]
                cost = call["model"]["reservation_usd"]
                if not self.ledger.fits(cost):
                    if not self.active:
                        self.stop_reason = "budget"
                    break  # wait until running calls settle and free their reservations
                pending.popleft()
                self.ledger.reserved += cost
                task = asyncio.create_task(self.run_call(client, call))
                self.active.add(task)
                task.add_done_callback(self.active.discard)
            if self.stop_reason and not self.active:
                break
            if self.active:
                await asyncio.wait(set(self.active), timeout=1, return_when=asyncio.FIRST_COMPLETED)

    async def run(self, key: str) -> dict:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        timeout = httpx.Timeout(connect=30, read=self.cfg.call_timeout, write=60, pool=60)
        limits = httpx.Limits(max_connections=self.cfg.concurrency + 20, max_keepalive_connections=20)
        async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
            await self.poll_key(client)
            finished = asyncio.Event()
            monitor = asyncio.create_task(self.monitor(client, finished))
            try:
                await self.schedule(client)
                if self.background:
                    await asyncio.gather(*list(self.background), return_exceptions=True)
                await asyncio.sleep(self.cfg.final_poll_delay)
                await self.poll_key(client)
            finally:
                finished.set()
                await monitor
        self.report()
        return self.status()


def final_records(records: list) -> dict:
    return {r["call_id"]: r for r in records if r.get("type") == "attempt" and r.get("final")}


def prior_spend(records: list) -> float:
    return sum(r.get("cost_usd") or 0 for r in records if r.get("type") == "attempt")


def remaining_calls(cfg: Config, number: int) -> tuple:
    """Return (calls still to make in random order, all raw records, calls in the GO)."""
    calls = planned_calls(cfg)
    records = read_jsonl(go_dir(cfg, number) / "raw.jsonl")
    finals = final_records(records)
    todo = []
    for c in calls:
        f = finals.get(call_id(number, c["model"]["id"], c["condition"]))
        if f is None or (cfg.retry_failed and f["outcome"] == TECHNICAL):
            todo.append(c)
    random.Random(f"run-order-{number}").shuffle(todo)
    return todo, records, calls


def show_go(cfg: Config, number: int) -> None:
    todo, records, calls = remaining_calls(cfg, number)
    worst = sum(c["model"]["reservation_usd"] for c in todo)
    typical = sum(
        c["model"]["typical_output_tokens"] * c["model"]["completion_price"] + 400 * c["model"]["prompt_price"]
        for c in todo
    )
    print(f"GO {number}: {len(calls)} calls, {len(todo)} still to make, on {len({c['model']['id'] for c in todo})} models")
    print(f"Already spent in this GO: ${prior_spend(records):.2f}; stop at ${cfg.budget:.2f}")
    print(f"Cost of the remaining calls: about ${typical:.2f} if outputs are typical, at most ${worst:.2f}")
    print("Nothing was sent. Add --execute to run this GO.")


async def execute_go(cfg: Config, number: int) -> int:
    todo, records, calls = remaining_calls(cfg, number)
    d = go_dir(cfg, number)
    if not todo:
        print(f"GO {number} has no calls left to make.")
        export_go(cfg, number)
        return 0
    key = cfg.key_file.read_text(encoding="utf-8-sig").strip()
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=30) as client:
        balance = None
        try:
            r = await client.get(f"{cfg.base_url}/credits")
            if r.status_code == 200:
                data = r.json()["data"]
                balance = float(data["total_credits"]) - float(data["total_usage"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            pass
    prior = prior_spend(records)
    if balance is None:
        print("WARNING: could not read the balance. The run will stop if OpenRouter reports no credits.")
    elif balance < cfg.budget - prior:
        print(f"Balance ${balance:.2f} is below the ${cfg.budget - prior:.2f} still allowed for this GO. Not starting.")
        return 3
    else:
        print(f"Balance: ${balance:.2f}")

    d.mkdir(parents=True, exist_ok=True)
    (d / "STOP").unlink(missing_ok=True)
    base_prompt = load_prompt(cfg.prompt_file)
    prompt_hash = hashlib.sha256(base_prompt.encode()).hexdigest()
    if prompt_hash != read_json(cfg.data_dir / "plan.json")["prompt_sha256"]:
        print("PROMPT.txt changed since the plan was frozen. Not starting.")
        return 3
    if records and any(s.get("prompt_sha256") != prompt_hash for s in read_jsonl(d / "sessions.jsonl")):
        print(f"GO {number} was started with a different PROMPT.txt. Not mixing prompts within one GO.")
        return 3
    append_jsonl(d / "sessions.jsonl", {
        "started_at": now_iso(), "prompt_sha256": prompt_hash, "budget_usd": cfg.budget, "concurrency": cfg.concurrency,
        "calls_to_make": len(todo), "balance_usd": balance, "prior_spend_usd": prior,
        "model_inclusion_sha256": hashlib.sha256(cfg.inclusion_file.read_bytes()).hexdigest(),
    })
    runner = Runner(cfg, number, todo, prior, base_prompt)

    def on_interrupt(*_):
        if runner.stop_reason == "interrupted":
            raise KeyboardInterrupt
        runner.stop_reason = "interrupted"
        print("\nStopping: no new calls, waiting for running calls. Press Ctrl+C again to abort them.", flush=True)

    previous = signal.signal(signal.SIGINT, on_interrupt)
    try:
        s = await runner.run(key)
    finally:
        signal.signal(signal.SIGINT, previous)
    export_go(cfg, number)
    included = load_included(cfg)
    finals = {k: r for k, r in final_records(read_jsonl(d / "raw.jsonl")).items() if r["model"] in included}
    print(f"\nGO {number}: {len(finals)} of {len(calls)} calls finished. Spent ${s['spent_usd']:.2f}.")
    for name, n in Counter(r["outcome"] for r in finals.values()).most_common():
        print(f"  {name}: {n}")
    if s["fatal_reason"]:
        print(f"Stopped by the API: {s['fatal_reason']}")
        return 3
    missing = sum(call_id(number, c["model"]["id"], c["condition"]) not in finals for c in calls)
    if missing:
        print(f"Stopped early ({s['stop_reason'] or 'internal errors'}), {missing} calls missing. "
              "The same command resumes them.")
        return 2
    return 0


# ---------------------------------------------------------------- export

def export_go(cfg: Config, number: int) -> None:
    d = go_dir(cfg, number)
    records = read_jsonl(d / "raw.jsonl")  # stays complete: models not included only leave the CSV files
    included = load_included(cfg)
    finals = {k: r for k, r in final_records(records).items() if r["model"] in included}
    conditions = {
        (m["id"], c["condition"]): {**c, "plan": m}
        for m in read_json(cfg.data_dir / "plan.json")["models"] for c in m["conditions"]
    }
    generations = {r["generation_id"]: r for r in records if r.get("type") == "generation"}
    spend, attempts = Counter(), Counter()
    for r in records:
        if r.get("type") == "attempt":
            spend[r["call_id"]] += r.get("cost_usd") or 0
            attempts[r["call_id"]] += 1

    rows = []
    for cid, r in finals.items():
        try:
            body = json.loads(r.get("response_text") or "{}")
        except ValueError:
            body = {}
        body = body if isinstance(body, dict) else {}
        choices = body.get("choices") if isinstance(body.get("choices"), list) else []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = text_of(message.get("content"))
        parsed = parse_answer(content) if content.strip() else {"parse_mode": None, "answer": None}
        a = parsed["answer"] or {}
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        details = usage.get("completion_tokens_details") or {}
        try:
            gen = json.loads((generations.get(r.get("generation_id")) or {}).get("response_text") or "{}").get("data") or {}
        except ValueError:
            gen = {}
        cond = conditions.get((r["model"], r["condition"]), {})
        plan = cond.get("plan", {})
        p, lo, hi = a.get("probability"), a.get("lower_bound"), a.get("upper_bound")
        valid = r["outcome"] == "valid"
        rows.append({
            "go": number, "call_id": cid, "model": r["model"], "canonical_slug": r.get("canonical_slug"),
            "condition": r["condition"], "effort": cond.get("effort"), "effort_rank": cond.get("effort_rank"),
            "is_default_effort": cond.get("is_default_effort"),
            "outcome": r["outcome"], "parse_mode": parsed["parse_mode"],
            "probability": p, "lower_bound": lo, "upper_bound": hi,
            "bounds_consistent": (lo <= p <= hi) if valid else None,
            "cause": a.get("cause"), "explanation": a.get("explanation"),
            "extra_keys": json.dumps(sorted(set(a) - {"probability", "lower_bound", "upper_bound", "cause", "explanation"})) if a else None,
            "cause_order": " | ".join(r.get("cause_order") or []),
            "raw_answer": content, "reasoning_chars": len(message.get("reasoning") or ""),
            "finish_reason": r.get("finish_reason"), "native_finish_reason": gen.get("native_finish_reason"),
            "provider": body.get("provider") or gen.get("provider_name"), "model_served": body.get("model"),
            "max_output_tokens": plan.get("max_output_tokens"), "limit_source": plan.get("limit_source"),
            "limit_parameter": plan.get("limit_parameter"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "response_cache_source_id": gen.get("response_cache_source_id"),
            "cost_usd": spend[cid], "cost_source": r.get("cost_source"), "attempts": attempts[cid],
            "elapsed_s": r.get("elapsed_s"), "started_at": r.get("started_at"), "http_status": r.get("http_status"),
            "generation_id": r.get("generation_id"), "error_code": r.get("error_code"), "error_message": r.get("error_message"),
        })
    rows.sort(key=lambda x: (x["model"], x["effort_rank"] if x["effort_rank"] is not None else -1, x["condition"]))
    if rows:
        write_csv(d / "results.csv", rows)

    usable = {}
    for row in rows:
        u = usable.setdefault(row["model"], {
            "model": row["model"], "conditions_run": 0, "conditions_usable": 0, "valid_answers": 0,
            "technical_failures": [],
        })
        u["conditions_run"] += 1
        if row["outcome"] == TECHNICAL:
            u["technical_failures"].append(row["condition"])
        else:
            u["conditions_usable"] += 1
        u["valid_answers"] += row["outcome"] == "valid"
    if usable:
        write_csv(d / "model_availability.csv", [
            {**u, "technical_failures": " | ".join(u["technical_failures"])} for u in usable.values()
        ])
    print(f"Wrote {d / 'results.csv'} ({len(rows)} rows)")


def show_status(cfg: Config, number: int) -> None:
    path = go_dir(cfg, number) / "status.json"
    if not path.exists():
        raise SystemExit(f"GO {number} has not started.")
    print(json.dumps(read_json(path), indent=2))


# ---------------------------------------------------------------- command line

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="download catalogue, freeze the plan (no tokens)")
    p.add_argument("--force", action="store_true", help="replace an unused plan")
    g = sub.add_parser("go", help="show or run one GO")
    g.add_argument("--number", type=int, required=True)
    g.add_argument("--execute", action="store_true", help="really send requests; this spends money")
    g.add_argument("--budget", type=float, default=Config.budget, help="stop this GO at this many USD")
    g.add_argument("--concurrency", type=int, default=Config.concurrency)
    g.add_argument("--retry-failed", action="store_true", help="try this GO's technical failures again")
    for name in ("status", "export"):
        s = sub.add_parser(name)
        s.add_argument("--number", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config()
    if args.command == "prepare":
        asyncio.run(prepare(cfg, args.force))
    elif args.command == "status":
        show_status(cfg, args.number)
    elif args.command == "export":
        export_go(cfg, args.number)
    elif args.command == "go":
        cfg.budget, cfg.concurrency, cfg.retry_failed = args.budget, args.concurrency, args.retry_failed
        if not 1 <= args.number <= cfg.number_of_gos:
            print(f"The study has GO 1 to GO {cfg.number_of_gos}; GO {args.number} is not one of them.")
            return 3
        if not args.execute:
            show_go(cfg, args.number)
            return 0
        return asyncio.run(execute_go(cfg, args.number))
    return 0


if __name__ == "__main__":
    sys.exit(main())
