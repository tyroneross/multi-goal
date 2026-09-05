#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0
"""Scan a repo for tunable parameters that could be DOE factors.

Stdlib only. Run: python3 suggest_factors.py --workdir <path>

Scans source files (.py / .ts / .tsx / .js / .jsx / .go / .rs / .rb) for
common patterns that indicate a tunable knob:

  1. UPPER_SNAKE_CASE numeric constants near tuning keywords
  2. process.env.X / os.getenv("X") with a numeric default
  3. argparse / commander / clap defaults that are numeric
  4. Numeric literals near keywords: timeout, retry, batch, parallel,
     cache, worker, pool_size, limit, max_*, min_*, chunk, buffer

Output: JSON list of candidates ranked by signal strength:
  [{"name": "BATCH_SIZE", "file": "src/x.ts:42", "current_value": 32,
    "suggested_levels": [16, 32, 64], "confidence": "high",
    "why": "UPPER_SNAKE constant + 'batch' keyword + 5 references"}, ...]

This is a heuristic scanner - it will miss factors and may suggest
non-tunable values. Output is meant to be confirmed by the user before
running agent-doe-engine optimization.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Files we walk
EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs", ".rb"}

# Dirs we skip
SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", ".next", ".nuxt",
    "__pycache__", ".pytest_cache", ".cache", "coverage",
    ".venv", "venv", "env", "target", ".bookmark", ".navgator",
    ".agent-doe-engine", ".multi-goal", ".claude-code-debugger", "vendor",
}

# Keywords that suggest a numeric value is tunable
TUNING_KEYWORDS = {
    "timeout", "retry", "retries", "batch", "batch_size", "batchsize",
    "parallel", "concurrency", "workers", "worker", "pool", "pool_size",
    "cache", "cachesize", "cache_size", "ttl", "expiry",
    "limit", "max", "min", "chunk", "chunk_size", "buffer", "buffer_size",
    "page_size", "pagesize", "rate_limit", "ratelimit", "throttle",
    "interval", "delay", "backoff", "jitter",
    "dim", "dimension", "embed_dim", "n_layers", "hidden_size",
    "lr", "learning_rate", "dropout", "temperature",
    "tokens", "max_tokens", "top_k", "top_p",
}

# Patterns
UPPER_SNAKE_NUM = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var|final|static)?\s*"
    r"([A-Z][A-Z0-9_]{2,})\s*(?::\s*\w+)?\s*=\s*(\d+(?:\.\d+)?)\s*[;,]?\s*$"
)
PY_UPPER_SNAKE_NUM = re.compile(
    r"^\s*([A-Z][A-Z0-9_]{2,})\s*(?::\s*\w+)?\s*=\s*(\d+(?:\.\d+)?)\s*$"
)
ENV_GETENV_PY = re.compile(
    r"""os\.(?:getenv|environ\.get)\(\s*['"]([A-Z][A-Z0-9_]+)['"]\s*,\s*['"]?(\d+(?:\.\d+)?)['"]?\s*\)"""
)
ENV_PROCESS_ENV = re.compile(
    r"""process\.env\.([A-Z][A-Z0-9_]+)\s*\|\|\s*(\d+(?:\.\d+)?)"""
)

NUM_LITERAL = re.compile(r"\b(\d+(?:\.\d+)?)\b")

# Pattern 4: numeric fields inside an exported object literal / dict constant.
#   export const SEARCH_THRESHOLDS = { GROQ_SCORE_WEIGHT: 0.7, ... }
#   SEARCH_THRESHOLDS = { "min_score": 0.3, ... }        (Python)
# The real tuning surface of an app often lives here, not in top-level
# constants, and a scanner that cannot see it ranks PORT above the knob that
# matters. Fields are reported as OBJECT.KEY so the definition site is unique.
OBJ_LITERAL_OPEN = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)?\s*([A-Z][A-Z0-9_]{2,})\s*(?::\s*[^=]+?)?\s*=\s*\{\s*$"
)
PY_OBJ_LITERAL_OPEN = re.compile(r"^\s*([A-Z][A-Z0-9_]{2,})\s*(?::\s*[^=]+?)?\s*=\s*\{\s*$")
OBJ_LITERAL_FIELD = re.compile(
    r"""^\s*['"]?([A-Za-z_][A-Za-z0-9_]*)['"]?\s*:\s*(-?\d+(?:\.\d+)?)\s*,?\s*(?://.*|#.*)?$"""
)
OBJ_LITERAL_CLOSE = re.compile(r"^\s*\}")

HINT_STOPWORDS = {"the", "and", "for", "with", "make", "faster", "improve", "optimize",
                  "reduce", "increase", "better", "more", "less", "time", "page"}


def hint_tokens(hint: str | None) -> set[str]:
    """Lower-case keyword set from an objective hint ("search latency" -> {search, latency})."""
    if not hint:
        return set()
    toks = {t for t in re.split(r"[^a-z0-9]+", hint.lower()) if len(t) >= 3}
    return toks - HINT_STOPWORDS


@dataclass
class Candidate:
    name: str
    file: str
    line: int
    current_value: float
    confidence: str  # high / medium / low
    why: str
    references: int = 0
    suggested_levels: list = field(default_factory=list)
    hint_matches: int = 0


# ---------------------------------------------------------------------------
# Level suggestion heuristics
# ---------------------------------------------------------------------------

def suggest_levels(value: float, name: str) -> list:
    """Pick reasonable [low, high] (and optionally a center) for a numeric factor.

    Heuristic: span 1/2x to 2x of current value, snap to integers if value is integer.
    For values < 1 (rates, dropout) use 0.5x and 2x without integer rounding.
    For values that look like power-of-two batch sizes, snap to 2^n.
    """
    is_int = float(value).is_integer()
    name_lower = name.lower()

    # Probabilistic / rate values: keep float
    if value < 1 and any(k in name_lower for k in ("rate", "drop", "temp", "lr", "p", "k")):
        return [round(value * 0.5, 4), round(value * 2.0, 4)]

    # Power-of-two friendly
    if is_int and value >= 2 and any(k in name_lower for k in
                                      ("batch", "chunk", "buffer", "tokens", "pool", "workers")):
        v = int(value)
        # Find nearest powers of two
        below = 1
        while below * 2 <= v:
            below *= 2
        above = below * 2
        return [below, above] if v == below else [below, v, above]

    # Default: 1/2x and 2x as integers if integer-valued
    if is_int:
        low = max(1, int(value * 0.5)) if value >= 2 else int(value)
        high = max(int(value * 2), low + 1)
        return sorted({low, int(value), high})

    return [round(value * 0.5, 4), round(value, 4), round(value * 2.0, 4)]


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------

def scan_file(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return candidates

    lines = text.splitlines()
    obj_name: str | None = None
    for lineno, line in enumerate(lines, start=1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue

        # Pattern 4: fields of an object-literal constant (tracked by block)
        if obj_name is not None:
            if OBJ_LITERAL_CLOSE.match(line):
                obj_name = None
            else:
                mf = OBJ_LITERAL_FIELD.match(line)
                if mf:
                    key, val = mf.group(1), float(mf.group(2))
                    key_lower = key.lower()
                    kw_match = next((k for k in TUNING_KEYWORDS if k in key_lower), None)
                    confidence = "high" if kw_match else "medium"
                    why = (f"field of object constant {obj_name}"
                           + (f" + '{kw_match}' keyword" if kw_match else ""))
                    candidates.append(Candidate(
                        name=f"{obj_name}.{key}", file=str(path), line=lineno,
                        current_value=val, confidence=confidence, why=why,
                    ))
                continue
        mo = (PY_OBJ_LITERAL_OPEN.match(line) if path.suffix == ".py"
              else OBJ_LITERAL_OPEN.match(line))
        if mo:
            obj_name = mo.group(1)
            continue

        # Pattern 1: UPPER_SNAKE_CASE = N  (Python or JS/TS-ish)
        m = (UPPER_SNAKE_NUM.match(line) if path.suffix != ".py" else PY_UPPER_SNAKE_NUM.match(line))
        if m:
            name, val = m.group(1), float(m.group(2))
            name_lower = name.lower()
            kw_match = next((k for k in TUNING_KEYWORDS if k in name_lower), None)
            confidence = "high" if kw_match else ("medium" if val > 1 else "low")
            why = f"UPPER_SNAKE constant" + (f" + '{kw_match}' keyword" if kw_match else "")
            candidates.append(Candidate(
                name=name, file=str(path), line=lineno,
                current_value=val, confidence=confidence, why=why,
            ))
            continue

        # Pattern 2: os.getenv("X", "N")
        for m2 in ENV_GETENV_PY.finditer(line):
            name, val = m2.group(1), float(m2.group(2))
            candidates.append(Candidate(
                name=name, file=str(path), line=lineno,
                current_value=val, confidence="high",
                why="os.getenv with numeric default",
            ))

        # Pattern 3: process.env.X || N
        for m3 in ENV_PROCESS_ENV.finditer(line):
            name, val = m3.group(1), float(m3.group(2))
            candidates.append(Candidate(
                name=name, file=str(path), line=lineno,
                current_value=val, confidence="high",
                why="process.env with numeric fallback",
            ))

    return candidates


def walk_repo(root: Path, paths: list[str] | None = None) -> list[Candidate]:
    """Scan every source file under `root`; `paths` restricts the walk to files
    matching any of the given globs (relative to root, e.g. "lib/search/**")."""
    candidates: list[Candidate] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in EXTS:
            continue
        if paths:
            rel = path.relative_to(root)
            if not any(rel.match(g) or rel.as_posix().startswith(g.rstrip("*/"))
                       for g in paths):
                continue
        candidates.extend(scan_file(path))
    return candidates


def deduplicate_and_count(candidates: list[Candidate]) -> list[Candidate]:
    """Group by name; pick the highest-confidence definition site, count refs."""
    by_name: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_name[c.name].append(c)
    out = []
    for name, group in by_name.items():
        # Best definition: high confidence first, then earliest file
        group.sort(key=lambda c: ({"high": 0, "medium": 1, "low": 2}[c.confidence], c.file, c.line))
        primary = group[0]
        primary.references = len(group)
        if primary.references > 1:
            primary.why += f" + {primary.references - 1} additional reference(s)"
        primary.suggested_levels = suggest_levels(primary.current_value, primary.name)
        out.append(primary)
    return out


def hint_score(candidate: Candidate, tokens: set[str], root: Path | None = None) -> int:
    """How many hint keywords the candidate's name or file path carries."""
    if not tokens:
        return 0
    hay = candidate.name.lower()
    file_part = candidate.file
    if root is not None:
        try:
            file_part = str(Path(candidate.file).relative_to(root))
        except ValueError:
            pass
    hay += " " + file_part.lower()
    return sum(1 for t in tokens if t in hay)


def rank_candidates(candidates: list[Candidate], top_n: int,
                    hint: str | None = None, root: Path | None = None) -> list[Candidate]:
    """Sort by confidence x references, boosted by objective-hint matches; return top_n.

    A hint match is worth more than a confidence grade: a "medium" knob in
    lib/search/ beats a "high" PORT constant when the goal is search latency.
    """
    confidence_score = {"high": 3, "medium": 2, "low": 1}
    tokens = hint_tokens(hint)
    for c in candidates:
        c.hint_matches = hint_score(c, tokens, root)
    candidates.sort(key=lambda c: -(c.hint_matches * 100
                                    + confidence_score[c.confidence] * 10 + c.references))
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workdir", default=".", help="repo root to scan (default: cwd)")
    p.add_argument("--top", type=int, default=15, help="max candidates to return")
    p.add_argument("--min-confidence", default="medium",
                   choices=["high", "medium", "low"],
                   help="filter out candidates below this confidence")
    p.add_argument("--json", action="store_true", help="JSON output (default: human-readable)")
    p.add_argument(
        "--research-levels", action="store_true",
        help="Mark high-confidence candidates with `needs_research: true` so the "
             "calling skill can opt in to consulting the host's research capability for "
             "best-practice level suggestions. The script itself never calls "
             "research - it only flags candidates worth researching. The "
             "orchestrator decides whether to invoke the research capability based "
             "on user confirmation. Off by default.",
    )
    p.add_argument("--hint", default=None,
                   help="objective text (e.g. 'search latency'); candidates whose name or "
                        "path carry its keywords rank first")
    p.add_argument("--paths", default=None,
                   help="comma-separated globs relative to workdir restricting the scan "
                        "(e.g. 'lib/search/**,app/api/search/**')")
    args = p.parse_args(argv)

    root = Path(args.workdir).resolve()
    if not root.is_dir():
        sys.stderr.write(f"not a directory: {root}\n")
        return 2

    path_globs = [g.strip() for g in args.paths.split(",") if g.strip()] if args.paths else None
    candidates = walk_repo(root, path_globs)
    candidates = deduplicate_and_count(candidates)
    confidence_order = {"high": 3, "medium": 2, "low": 1}
    min_score = confidence_order[args.min_confidence]
    candidates = [c for c in candidates if confidence_order[c.confidence] >= min_score]
    candidates = rank_candidates(candidates, args.top, hint=args.hint, root=root)

    if args.json:
        rows = [asdict(c) for c in candidates]
        if args.research_levels:
            # Mark high-confidence candidates whose name matches a known
            # tuning-keyword pattern as research-eligible. The skill text
            # explains how the orchestrator consumes this.
            for r in rows:
                if r.get("confidence") == "high":
                    name_lower = r.get("name", "").lower()
                    if any(kw in name_lower for kw in TUNING_KEYWORDS):
                        r["needs_research"] = True
                        r["research_topic"] = (
                            f"best-practice levels for {r['name']} "
                            f"(currently {r.get('current_value')})"
                        )
        print(json.dumps(rows, indent=2))
    else:
        if not candidates:
            print("No tunable parameters found.")
            return 0
        print(f"Found {len(candidates)} candidate factor(s) in {root}:\n")
        for i, c in enumerate(candidates, start=1):
            rel = Path(c.file).relative_to(root) if Path(c.file).is_relative_to(root) else c.file
            print(f"  {i}. {c.name}  (currently {c.current_value})")
            print(f"     {rel}:{c.line}  [{c.confidence}]")
            print(f"     suggested levels: {c.suggested_levels}")
            print(f"     {c.why}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
