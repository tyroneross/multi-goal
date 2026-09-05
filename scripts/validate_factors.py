#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0
"""Factor-adjustability validation - prove the optimizer can move it.

Stdlib only. Run:
    python3 validate_factors.py --workdir <path> --candidates <suggest_factors.json>

For each candidate (output from `suggest_factors.py --json`), this script:

  1. Locates every definition site for the named constant/env-var across the
     scanned source tree, using the same patterns as suggest_factors.py.
  2. If the candidate is `adjustable`, applies a test mutation to the value
     at the primary definition site (byte-level rewrite - *not* an AST edit;
     the goal is to prove file-mutation actually changes what downstream
     readers see).
  3. Re-reads the file and confirms the new value is present.
  4. Reverts to the original byte content (verbatim) and verifies byte
     equality with the snapshot taken before mutation.
  5. Classifies the candidate as one of:

       adjustable                       - probe passed end-to-end
       not_adjustable / dead_constant   - zero references outside its own
                                          definition site
       not_adjustable / duplicate_definition
                                        - same name defined at >=2 sites
                                          with conflicting values (which one
                                          wins is ambiguous)
       not_adjustable / mutation_failed - write attempt did not change the
                                          file (read-only FS, locked file)
       not_adjustable / revert_failed   - write succeeded but original bytes
                                          could not be restored; surfaced
                                          loudly because the working tree is
                                          now dirty

Output: JSON list mirroring the input plus `adjustability`, `reason`,
`evidence`, and (for adjustable cases) `definition_site`.

This script never modifies files unless it can also revert them - it
performs a snapshot-write-verify-revert-verify cycle on a SINGLE primary
definition site. On any failure mid-cycle, it attempts to restore the
original bytes and surfaces a `revert_failed` reason if that fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Patterns mirror suggest_factors.py - keep in sync if that file changes.
EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs", ".rb"}

SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", ".next", ".nuxt",
    "__pycache__", ".pytest_cache", ".cache", "coverage",
    ".venv", "venv", "env", "target", ".bookmark", ".navgator",
    ".agent-doe-engine", ".multi-goal", ".claude-code-debugger", "vendor",
}

# Same anchored patterns as suggest_factors.py for definition sites
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

# Reference patterns: any token-bounded occurrence of NAME.
def _reference_pattern(name: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(name) + r"\b")


@dataclass
class DefSite:
    """Where a named constant is defined."""

    file: str
    line: int
    raw_line: str  # the exact source line (for byte-precise rewrite)
    value: float
    pattern: str  # which regex matched ("upper_snake" | "env_getenv" | "env_process")


@dataclass
class Validated:
    name: str
    current_value: float
    adjustability: str  # "adjustable" | "not_adjustable" | "not_probed"
    reason: str  # "ok" | "dead_constant" | "duplicate_definition" | "mutation_failed" | "revert_failed" | "no_definition_site" | "categorical_level"
    evidence: str
    definition_site: dict | None = None  # {"file": ..., "line": ..., "value": ...}
    extras: dict = field(default_factory=dict)  # forwarded from input candidate


# ---------------------------------------------------------------------------
# Definition-site location
# ---------------------------------------------------------------------------

def _is_skipped(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return any(part in SKIP_DIRS for part in rel.parts)


def find_definition_sites(name: str, root: Path) -> list[DefSite]:
    """Walk repo, return every line that DEFINES `name` to a numeric value."""
    sites: list[DefSite] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in EXTS:
            continue
        if _is_skipped(path, root):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        lines = text.splitlines(keepends=False)
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            # Try anchored definition patterns first (the suggest_factors set)
            m = (UPPER_SNAKE_NUM.match(line) if path.suffix != ".py" else PY_UPPER_SNAKE_NUM.match(line))
            if m and m.group(1) == name:
                sites.append(DefSite(
                    file=str(path), line=lineno, raw_line=line,
                    value=float(m.group(2)), pattern="upper_snake",
                ))
                continue
            for m2 in ENV_GETENV_PY.finditer(line):
                if m2.group(1) == name:
                    sites.append(DefSite(
                        file=str(path), line=lineno, raw_line=line,
                        value=float(m2.group(2)), pattern="env_getenv",
                    ))
            for m3 in ENV_PROCESS_ENV.finditer(line):
                if m3.group(1) == name:
                    sites.append(DefSite(
                        file=str(path), line=lineno, raw_line=line,
                        value=float(m3.group(2)), pattern="env_process",
                    ))
    return sites


def count_references(name: str, root: Path, exclude_positions: set[tuple[str, int]]) -> int:
    """Count token-bounded occurrences of `name` across ALL source files,
    excluding only the exact (file, line) positions that are definition sites.

    A match on any other line - including non-definition lines in the same file
    as a definition - counts as a reference. A factor with zero such references
    is a dead constant.

    Previously this excluded entire definition files, which caused constants
    defined AND used in the same module to be falsely classified as dead.
    """
    pat = _reference_pattern(name)
    count = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in EXTS:
            continue
        if _is_skipped(path, root):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        path_str = str(path)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if (path_str, lineno) in exclude_positions:
                continue
            count += len(pat.findall(line))
    return count


# ---------------------------------------------------------------------------
# Mutation probe
# ---------------------------------------------------------------------------

def _probe_value(original: float) -> float:
    """Pick a probe value that is provably different from `original`.

    Avoid 0 (might be the natural default) and avoid the original. Use
    `original + 1` for integers, `original + 0.5` for floats < 1, and
    `original * 2 + 1` as a fallback to dodge accidental equalities.
    """
    if float(original).is_integer():
        return float(int(original) + 1)
    if original < 1:
        return round(original + 0.5, 4)
    return round(original * 2 + 1, 4)


def _format_value(probe: float, original: float) -> str:
    """Match the original's lexical shape (int vs float) so the rewrite is
    byte-stable. If the original was an integer literal, write the probe as
    an integer; otherwise preserve the decimal form."""
    if float(original).is_integer() and float(probe).is_integer():
        return str(int(probe))
    return repr(probe) if "." in str(probe) else f"{probe:.4f}".rstrip("0").rstrip(".")


def _rewrite_line(raw_line: str, name: str, original: float, probe: float, pattern: str) -> str | None:
    """Produce a single-line rewrite of `raw_line` that swaps `original` for
    `probe` at the *named* binding site. Returns None if no substitution
    matched (defense in depth - the caller treats None as mutation_failed).
    """
    new_val_str = _format_value(probe, original)
    # The original value can appear as int ("5") or float ("5.0"); accept
    # both forms in the source.
    if float(original).is_integer():
        orig_patterns = [str(int(original)), f"{int(original)}.0", f"{int(original):.1f}"]
    else:
        orig_patterns = [str(original), f"{original:.4f}".rstrip("0").rstrip(".")]
    # De-dup, preserve order
    seen = set()
    orig_patterns = [p for p in orig_patterns if not (p in seen or seen.add(p))]

    # We rebuild from the matched group rather than risk regex-replacing every
    # numeric token on the line (the line could contain other numerics).
    if pattern in ("upper_snake",):
        rx = re.compile(
            r"^(\s*(?:export\s+)?(?:const|let|var|final|static)?\s*"
            + re.escape(name) + r"\s*(?::\s*\w+)?\s*=\s*)("
            + "|".join(re.escape(p) for p in orig_patterns) + r")(\s*[;,]?\s*)$"
        )
        m = rx.match(raw_line)
        if not m:
            # Try the Python anchored form
            rx_py = re.compile(
                r"^(\s*" + re.escape(name) + r"\s*(?::\s*\w+)?\s*=\s*)("
                + "|".join(re.escape(p) for p in orig_patterns) + r")(\s*)$"
            )
            m = rx_py.match(raw_line)
            if not m:
                return None
        return m.group(1) + new_val_str + m.group(3)
    if pattern == "env_getenv":
        # Replace the numeric default inside os.getenv("NAME", "<val>") or os.getenv("NAME", <val>)
        rx = re.compile(
            r"(os\.(?:getenv|environ\.get)\(\s*['\"]" + re.escape(name) + r"['\"]\s*,\s*['\"]?)("
            + "|".join(re.escape(p) for p in orig_patterns) + r")(['\"]?\s*\))"
        )
        new = rx.sub(lambda m: m.group(1) + new_val_str + m.group(3), raw_line, count=1)
        return new if new != raw_line else None
    if pattern == "env_process":
        rx = re.compile(
            r"(process\.env\." + re.escape(name) + r"\s*\|\|\s*)("
            + "|".join(re.escape(p) for p in orig_patterns) + r")(\b)"
        )
        new = rx.sub(lambda m: m.group(1) + new_val_str + m.group(3), raw_line, count=1)
        return new if new != raw_line else None
    return None


def probe_mutation(site: DefSite) -> tuple[bool, str]:
    """Snapshot → rewrite → verify → revert → verify. Returns (passed, evidence).

    On revert failure, raises RuntimeError - the caller MUST surface the
    dirty-file state loudly. We never silently leave the tree modified.
    """
    path = Path(site.file)
    try:
        original_bytes = path.read_bytes()
    except OSError as e:
        return False, f"could not read file: {e}"

    original_text = original_bytes.decode("utf-8", errors="ignore")
    lines = original_text.splitlines(keepends=True)
    if site.line < 1 or site.line > len(lines):
        return False, f"definition site line {site.line} out of range"

    line_with_ending = lines[site.line - 1]
    # Preserve the line ending the file already uses
    if line_with_ending.endswith("\r\n"):
        line_no_ending = line_with_ending[:-2]
        ending = "\r\n"
    elif line_with_ending.endswith("\n"):
        line_no_ending = line_with_ending[:-1]
        ending = "\n"
    elif line_with_ending.endswith("\r"):
        line_no_ending = line_with_ending[:-1]
        ending = "\r"
    else:
        line_no_ending = line_with_ending
        ending = ""

    probe = _probe_value(site.value)
    new_line = _rewrite_line(line_no_ending, _name_from_site_or_raise(site), site.value, probe, site.pattern)
    if new_line is None:
        return False, "could not construct mutated line (line shape changed since scan?)"

    lines[site.line - 1] = new_line + ending
    new_text = "".join(lines)
    new_bytes = new_text.encode("utf-8")

    if new_bytes == original_bytes:
        return False, "rewritten bytes identical to original (no actual mutation)"

    # Write probe value
    try:
        path.write_bytes(new_bytes)
    except OSError as e:
        return False, f"mutation_failed: write error: {e}"

    # Verify it landed (read back)
    try:
        after_bytes = path.read_bytes()
    except OSError as e:
        # Try to restore
        try:
            path.write_bytes(original_bytes)
        except OSError:
            raise RuntimeError(f"revert_failed: read-back error then restore failed for {path}: {e}")
        return False, f"mutation_failed: read-back error: {e}"

    if after_bytes != new_bytes:
        # Restore and bail
        try:
            path.write_bytes(original_bytes)
        except OSError:
            raise RuntimeError(f"revert_failed: write landed differently and restore failed for {path}")
        return False, "mutation_failed: file content after write did not match what we wrote"

    # Revert
    try:
        path.write_bytes(original_bytes)
    except OSError as e:
        raise RuntimeError(f"revert_failed: could not write original bytes back to {path}: {e}")

    # Confirm revert
    try:
        final_bytes = path.read_bytes()
    except OSError as e:
        raise RuntimeError(f"revert_failed: could not re-read {path} after restore: {e}")
    if final_bytes != original_bytes:
        raise RuntimeError(f"revert_failed: post-restore bytes differ from snapshot for {path}")

    return True, (
        f"probe value {probe} written to {path.name}:{site.line} (verified), "
        f"original bytes restored (verified)"
    )


def _name_from_site_or_raise(site: DefSite) -> str:
    """Re-extract the name from the raw line - we keep a tiny verifier so a
    bad upstream record can't silently corrupt the rewrite."""
    line = site.raw_line
    if site.pattern == "upper_snake":
        m = UPPER_SNAKE_NUM.match(line) or PY_UPPER_SNAKE_NUM.match(line)
        if m:
            return m.group(1)
    elif site.pattern == "env_getenv":
        m = ENV_GETENV_PY.search(line)
        if m:
            return m.group(1)
    elif site.pattern == "env_process":
        m = ENV_PROCESS_ENV.search(line)
        if m:
            return m.group(1)
    raise RuntimeError(f"could not extract name from raw line: {line!r}")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_candidate(name: str, current_value: float, root: Path) -> Validated:
    sites = find_definition_sites(name, root)

    if not sites:
        return Validated(
            name=name, current_value=current_value,
            adjustability="not_adjustable", reason="no_definition_site",
            evidence=f"could not locate any definition for {name} in {root}",
        )

    # Duplicate definition with conflicting values?
    distinct_values = sorted({s.value for s in sites})
    if len(distinct_values) > 1:
        sample = "; ".join(f"{Path(s.file).name}:{s.line}={s.value}" for s in sites[:3])
        return Validated(
            name=name, current_value=current_value,
            adjustability="not_adjustable", reason="duplicate_definition",
            evidence=(
                f"{name} defined at {len(sites)} sites with conflicting values "
                f"{distinct_values}: {sample}"
            ),
        )

    # Pick the primary site (first by file path, then line). The scan order is
    # already deterministic via sorted(rglob) above.
    primary = sites[0]

    # Dead-constant check: are there any references outside the definition lines?
    # Exclude only the exact (file, line) positions of definition sites so that
    # uses of the constant on other lines in the same file are counted.
    def_positions = {(s.file, s.line) for s in sites}
    ref_count = count_references(name, root, exclude_positions=def_positions)
    if ref_count == 0:
        return Validated(
            name=name, current_value=current_value,
            adjustability="not_adjustable", reason="dead_constant",
            evidence=f"{name} has zero references outside its definition line(s); primary site {Path(primary.file).name}:{primary.line}",
            definition_site={"file": primary.file, "line": primary.line, "value": primary.value},
        )

    # Probe: mutate, verify, revert
    try:
        passed, evidence = probe_mutation(primary)
    except RuntimeError as exc:
        # Loud surface - the working tree may be dirty
        return Validated(
            name=name, current_value=current_value,
            adjustability="not_adjustable", reason="revert_failed",
            evidence=str(exc),
            definition_site={"file": primary.file, "line": primary.line, "value": primary.value},
        )

    if not passed:
        return Validated(
            name=name, current_value=current_value,
            adjustability="not_adjustable", reason="mutation_failed",
            evidence=evidence,
            definition_site={"file": primary.file, "line": primary.line, "value": primary.value},
        )

    return Validated(
        name=name, current_value=current_value,
        adjustability="adjustable", reason="ok",
        evidence=f"{ref_count} reference(s) across the source tree; mutation probe round-tripped cleanly. {evidence}",
        definition_site={"file": primary.file, "line": primary.line, "value": primary.value},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_candidates(path: Path | None, stdin_data: str | None) -> list[dict]:
    if path is not None:
        return json.loads(path.read_text())
    if stdin_data is not None:
        return json.loads(stdin_data)
    raise RuntimeError("no candidates supplied (use --candidates or pipe JSON on stdin)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--workdir", default=".", help="repo root the candidates refer to (default: cwd)")
    p.add_argument(
        "--candidates",
        type=Path,
        help="path to suggest_factors.py --json output. If omitted, reads JSON from stdin.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="JSON output (default: human-readable summary)",
    )
    p.add_argument(
        "--reject-non-adjustable", action="store_true",
        help="exit 1 if any candidate is not_adjustable (CI / planner use)",
    )
    args = p.parse_args(argv)

    root = Path(args.workdir).resolve()
    if not root.is_dir():
        sys.stderr.write(f"not a directory: {root}\n")
        return 2

    stdin_data = None
    if args.candidates is None:
        if sys.stdin.isatty():
            sys.stderr.write(
                "error: no --candidates path supplied and stdin is a tty.\n"
                "       pipe suggest_factors.py --json output, or pass --candidates <file>.\n"
            )
            return 2
        stdin_data = sys.stdin.read()

    try:
        candidates = load_candidates(args.candidates, stdin_data)
    except (json.JSONDecodeError, RuntimeError) as e:
        sys.stderr.write(f"could not load candidates: {e}\n")
        return 2

    results: list[Validated] = []
    for c in candidates:
        name = c.get("name")
        if not name:
            continue
        raw_value = c.get("current_value", 0.0)
        try:
            current_value = float(raw_value)
        except (TypeError, ValueError):
            # Categorical factor (e.g. "always" | "on_demand"): the numeric
            # mutation probe does not apply. Report it as not probed rather
            # than crashing, and tell the caller how to prove adjustability.
            v = Validated(
                name=name, current_value=raw_value,
                adjustability="not_probed", reason="categorical_level",
                evidence=(f"current_value {raw_value!r} is not numeric; prove adjustability "
                          f"by applying the other declared level and diffing an observable "
                          f"response (the probe only perturbs numbers)"),
            )
            for k in ("suggested_levels", "why", "confidence", "file", "line", "levels"):
                if k in c:
                    v.extras[k] = c[k]
            results.append(v)
            continue
        v = classify_candidate(name=name, current_value=current_value, root=root)
        # Forward useful fields from suggest_factors (suggested_levels, why, needs_research, ...)
        for k in ("suggested_levels", "why", "confidence", "needs_research", "research_topic", "file", "line"):
            if k in c:
                v.extras[k] = c[k]
        results.append(v)

    any_rejected = any(v.adjustability != "adjustable" for v in results)

    if args.json:
        rows = []
        for v in results:
            d = asdict(v)
            # Flatten extras onto the top level for easy consumption
            extras = d.pop("extras", {}) or {}
            d.update(extras)
            rows.append(d)
        print(json.dumps(rows, indent=2))
    else:
        if not results:
            print("No candidates to validate.")
            return 0
        adj = [v for v in results if v.adjustability == "adjustable"]
        rej = [v for v in results if v.adjustability != "adjustable"]
        print(f"Validated {len(results)} candidate(s): {len(adj)} adjustable, {len(rej)} rejected.\n")
        if adj:
            print("Adjustable (safe to enter DOE):")
            for v in adj:
                site = v.definition_site or {}
                print(f"  + {v.name} = {v.current_value}  ({Path(site.get('file', '?')).name}:{site.get('line', '?')})")
                print(f"      {v.evidence}")
        if rej:
            print("\nRejected (will NOT enter DOE):")
            for v in rej:
                print(f"  - {v.name}  [{v.reason}]")
                print(f"      {v.evidence}")

    if args.reject_non_adjustable and any_rejected:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
