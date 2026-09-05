# SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0
"""
Multi-objective core: scalarization, Derringer-Suich desirability, Pareto dominance.

Public API (contract — do not rename or reshape):
    compute_bounds, normalize, scalarize_run, desirability_run,
    dominates, pareto_front, select_best,
    validate_objectives, is_feasible

The goal contract
-----------------
An objective is minimally `{"name", "direction", "weight"}`. Every field below
is OPTIONAL and the defaults preserve the pre-contract behaviour exactly:

    role            "primary" | "guardrail" | "quality"   (default "primary")
    driver          str   - the product driver this objective serves
    baseline        float - the measured value before the experiment
    min_acceptable  float - the absolute bar, in raw units
    target          float - the absolute goal, in raw units
    min_effect      float - practical-significance threshold, in raw units

`role` is what turns a number into a decision rule, following the multi-metric
decision taxonomy Spotify publishes (success / guardrail / deterioration /
quality): a `primary` objective must IMPROVE (superiority), a `guardrail` must
merely NOT DEGRADE (non-inferiority), and a `quality` objective is reported but
never decides. Guardrails are therefore constraints handled by `is_feasible`,
not extra terms in the score - averaging a guardrail into a weighted sum lets a
large primary win buy its way past a safety bar.

`min_acceptable` / `target` are ABSOLUTE limits, which is what Derringer &
Suich (1980) specify. Without them `desirability_run` min-max normalises across
the batch, so the worst run in any batch scores 0 however good it is in raw
units - a batch-relative verdict wearing an absolute verdict's clothes.
"""

from __future__ import annotations

import math
from typing import Any


VALID_ROLES = ("primary", "guardrail", "quality")
# Score gap below which two runs are reported as contenders rather than a winner.
CONTENDER_TOLERANCE = 0.02
VALID_DIRECTIONS = ("lower", "higher")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_objective(obj: dict) -> dict:
    """Return objective dict with defaults applied (non-mutating)."""
    out = {
        "name": obj["name"],
        "direction": obj.get("direction", "lower"),
        "weight": float(obj.get("weight", 1.0)),
        "role": obj.get("role", "primary"),
    }
    for key in ("driver", "baseline", "min_acceptable", "target", "min_effect"):
        if obj.get(key) is not None:
            out[key] = obj[key]
    return out


def _as_float(value: Any) -> float | None:
    """Coerce an optional contract limit to float; None stays None."""
    if value is None:
        return None
    return float(value)


def _meets(value: float, bar: float, direction: str) -> bool:
    """True when `value` meets `bar` for this direction (a tie meets the bar)."""
    if direction == "higher":
        return value >= bar
    return value <= bar


def scored_objectives(objectives: list[dict]) -> list[dict]:
    """The objectives that enter a score: primaries only. Guardrails are
    constraints (see is_feasible) and quality metrics are reported, never
    scored. When no objective is marked primary every objective scores, which
    is exactly the pre-contract behaviour."""
    primaries = [o for o in objectives if o.get("role", "primary") == "primary"]
    return primaries or list(objectives)


def _normalized_weights(objectives: list[dict]) -> list[float]:
    """Return per-objective weights normalized to sum to 1.0."""
    weights = [float(o.get("weight", 1.0)) for o in objectives]
    total = sum(weights)
    if total == 0.0:
        # Guard: all-zero weights → equal weighting
        n = len(weights)
        return [1.0 / n] * n
    return [w / total for w in weights]


def _is_at_least_as_good(a_val: float, b_val: float, direction: str) -> bool:
    """True if a_val is >= b_val in the context of `direction`."""
    if direction == "higher":
        return a_val >= b_val
    return a_val <= b_val  # "lower" is better


def _is_strictly_better(a_val: float, b_val: float, direction: str) -> bool:
    """True if a_val is strictly better than b_val for `direction`."""
    if direction == "higher":
        return a_val > b_val
    return a_val < b_val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_bounds(runs: list[dict], objectives: list[dict]) -> dict:
    """
    runs = [{"run_id": int, "values": {obj_name: float, ...}}, ...].
    objectives = [{"name": str, "direction": "lower"|"higher", "weight": float}, ...].
    Returns {obj_name: {"min": float, "max": float}} computed across all runs.
    Raises ValueError if a run is missing an objective's value.
    """
    obj_names = [o["name"] for o in objectives]
    obj_by_name = {o["name"]: o for o in objectives}
    bounds: dict[str, dict[str, float]] = {}

    for name in obj_names:
        values: list[float] = []
        for run in runs:
            run_vals = run.get("values", {})
            if name not in run_vals:
                raise ValueError(
                    f"Run {run.get('run_id', '?')} missing value for objective '{name}'"
                )
            values.append(float(run_vals[name]))
        lo, hi = min(values), max(values)
        floor = obj_by_name[name].get("noise_floor")
        if floor is not None and (hi - lo) <= float(floor):
            # The observed spread is inside this objective's noise. Min-max
            # normalisation would stretch that spread to fill [0,1] just as
            # fully as a real one, letting a meaningless wobble veto a run that
            # is better on every other objective - and under `desirability`,
            # zero it outright. Collapse the range instead: normalize() returns
            # a constant 1.0 when hi == lo, so the objective stops swinging the
            # selection while its raw values are still reported.
            mid = (hi + lo) / 2.0
            bounds[name] = {"min": mid, "max": mid, "degenerate": True,
                            "observed_min": lo, "observed_max": hi,
                            "noise_floor": float(floor)}
        else:
            bounds[name] = {"min": lo, "max": hi}

    return bounds


def normalize(value: float, direction: str, lo: float, hi: float) -> float:
    """
    Min-max normalize to [0,1] where 1.0 == best.
    direction 'higher': (value-lo)/(hi-lo).
    direction 'lower':  (hi-value)/(hi-lo).
    Degenerate hi==lo -> return 1.0.
    """
    if hi == lo:
        return 1.0
    if direction == "higher":
        return (value - lo) / (hi - lo)
    # direction == "lower"
    return (hi - value) / (hi - lo)


def scalarize_run(values: dict, objectives: list[dict], bounds: dict) -> float:
    """
    Weighted sum of normalized responses. Weights normalized internally to sum to 1.
    Higher return = better. Range [0,1].
    """
    objectives = scored_objectives(objectives)
    w_norm = _normalized_weights(objectives)
    score = 0.0
    for w, obj in zip(w_norm, objectives):
        name = obj["name"]
        direction = obj.get("direction", "lower")
        lo = bounds[name]["min"]
        hi = bounds[name]["max"]
        d = normalize(float(values[name]), direction, lo, hi)
        score += w * d
    return score


def desirability_limits(obj: dict, bound: dict) -> tuple[float, float] | None:
    """Resolve one objective's absolute Derringer-Suich limits as (bad, good).

    Returns None when the objective declares neither `min_acceptable` nor
    `target` - the signal to fall back to batch min-max normalisation. When
    only one limit is declared the batch bound supplies the other, so a
    half-specified contract still buys an absolute bar on the side it named.

    `bad` is the value where d == 0, `good` the value where d == 1:
      direction "lower":  bad = min_acceptable (or batch max), good = target (or batch min)
      direction "higher": bad = min_acceptable (or batch min), good = target (or batch max)
    """
    min_acceptable = _as_float(obj.get("min_acceptable"))
    target = _as_float(obj.get("target"))
    if min_acceptable is None and target is None:
        return None
    direction = obj.get("direction", "lower")
    lo, hi = float(bound["min"]), float(bound["max"])
    if direction == "higher":
        bad = min_acceptable if min_acceptable is not None else lo
        good = target if target is not None else hi
    else:
        bad = min_acceptable if min_acceptable is not None else hi
        good = target if target is not None else lo
    return bad, good


def desirability_value(value: float, obj: dict, bound: dict) -> float:
    """One objective's desirability d in [0,1]; 1.0 == fully satisfies the goal.

    Absolute (Derringer-Suich 1980) whenever the objective declares
    `min_acceptable` and/or `target`; otherwise the legacy batch min-max
    normalisation, which is relative to whatever else happened to run.
    """
    direction = obj.get("direction", "lower")
    if bound.get("degenerate"):
        # This objective's entire observed spread sits inside its declared
        # noise floor. compute_bounds already collapsed it so it cannot decide
        # the winner; applying an absolute scale here would re-arm it.
        return 1.0
    limits = desirability_limits(obj, bound)
    if limits is None:
        return normalize(float(value), direction, float(bound["min"]), float(bound["max"]))
    bad, good = limits
    if bad == good:
        # Bar and goal coincide: meeting it is full desirability, missing it zero.
        return 1.0 if _meets(float(value), good, direction) else 0.0
    d = (float(value) - bad) / (good - bad)
    return min(1.0, max(0.0, d))


def desirability_run(values: dict, objectives: list[dict], bounds: dict) -> float:
    """
    Derringer-Suich overall desirability D.
    Per objective: d_i = desirability_value(...) - ABSOLUTE one-sided limits when
    the objective declares `min_acceptable` and/or `target`, else the legacy
    batch min-max normalisation.
    D = (prod_i d_i ** w_i) ** (1 / sum_i w_i).
    If any d_i == 0 -> D == 0 (a hard fail on one objective tanks the run).
    Range [0,1].
    """
    objectives = scored_objectives(objectives)
    weights = [float(obj.get("weight", 1.0)) for obj in objectives]
    w_sum = sum(weights)
    if w_sum == 0.0:
        w_sum = float(len(weights))
        weights = [1.0] * len(weights)

    log_sum = 0.0
    for w, obj in zip(weights, objectives):
        name = obj["name"]
        d_i = desirability_value(float(values[name]), obj, bounds[name])
        if d_i == 0.0:
            return 0.0
        log_sum += w * math.log(d_i)

    return math.exp(log_sum / w_sum)


# ---------------------------------------------------------------------------
# Goal contract: validation + feasibility
# ---------------------------------------------------------------------------

def validate_objectives(objs: list[dict] | None) -> dict:
    """Check a goal contract. Returns {"errors": [...], "warnings": [...]}.

    ERRORS are contradictions that make the contract unusable:
      - an objective with no `name`
      - `direction` outside {lower, higher}
      - `role` outside {primary, guardrail, quality}
      - `min_acceptable` strictly BETTER than `target`. The bar is the value at
        which desirability hits 0 and the target the value at which it hits 1,
        so the bar must sit on the bad side of the goal. Inverted, the scale
        runs backwards and every reading is wrong.

    WARNINGS are an incomplete contract - runnable, but the run cannot tell you
    whether it is done:
      - no `primary` objective at all: nothing has to improve
      - a primary with no `driver`: the number is not tied to a product outcome
      - a primary with neither `target` nor `min_effect`: no bar to clear, so
        "best run" is only best-of-batch
      - a guardrail with neither `min_acceptable` nor `baseline`: no bar to
        hold, so it constrains nothing
    """
    errors: list[str] = []
    warnings: list[str] = []
    objs = objs or []

    if not objs:
        warnings.append("No objectives declared: nothing to optimize or protect.")
        return {"errors": errors, "warnings": warnings}

    n_primary = 0
    for idx, obj in enumerate(objs):
        label = obj.get("name") or f"objective[{idx}]"

        if not obj.get("name"):
            errors.append(f"{label}: missing 'name'.")

        direction = obj.get("direction", "lower")
        if direction not in VALID_DIRECTIONS:
            errors.append(
                f"{label}: direction {direction!r} is not one of {list(VALID_DIRECTIONS)}."
            )

        role = obj.get("role", "primary")
        if role not in VALID_ROLES:
            errors.append(f"{label}: role {role!r} is not one of {list(VALID_ROLES)}.")

        min_acceptable = _as_float(obj.get("min_acceptable"))
        target = _as_float(obj.get("target"))
        if (min_acceptable is not None and target is not None
                and direction in VALID_DIRECTIONS
                and min_acceptable != target
                and _meets(min_acceptable, target, direction)):
            errors.append(
                f"{label}: min_acceptable={min_acceptable:g} is better than "
                f"target={target:g} for direction '{direction}'. The bar must sit "
                f"on the bad side of the goal (d=0 at min_acceptable, d=1 at target)."
            )

        if role == "primary":
            n_primary += 1
            if not obj.get("driver"):
                warnings.append(
                    f"{label}: primary objective has no 'driver'. Name the product "
                    f"outcome it serves, or nobody can tell whether moving it matters."
                )
            if target is None and obj.get("min_effect") is None:
                warnings.append(
                    f"{label}: primary objective declares neither 'target' nor "
                    f"'min_effect'. Without an absolute bar the winner is only "
                    f"best-of-batch, not good enough."
                )
        elif role == "guardrail":
            if min_acceptable is None and obj.get("baseline") is None:
                warnings.append(
                    f"{label}: guardrail declares neither 'min_acceptable' nor "
                    f"'baseline', so it constrains nothing."
                )

    if n_primary == 0:
        warnings.append(
            "No 'primary' objective: every objective is a guardrail or quality "
            "metric, so nothing is required to improve."
        )

    return {"errors": errors, "warnings": warnings}


def is_feasible(values: dict, objectives: list[dict]) -> tuple[bool, list[str]]:
    """Check a run's measured values against every guardrail constraint.

    A guardrail is a constraint, not a score term. It is violated when the
    measured value is worse than:
      - its `min_acceptable`, when declared (absolute bar); else
      - its `baseline`, when declared (non-inferiority: do not make it worse
        than it already was).
    A guardrail with neither declares no constraint and can never be violated.
    Ties meet the bar. `primary` and `quality` objectives are never constraints.

    Returns (feasible, violated_objective_names).
    """
    violated: list[str] = []
    for obj in objectives:
        if obj.get("role", "primary") != "guardrail":
            continue
        name = obj["name"]
        if name not in values:
            continue
        direction = obj.get("direction", "lower")
        bar = _as_float(obj.get("min_acceptable"))
        if bar is None:
            bar = _as_float(obj.get("baseline"))
        if bar is None:
            continue
        if not _meets(float(values[name]), bar, direction):
            violated.append(name)
    return (not violated), violated


def dominates(a_values: dict, b_values: dict, objectives: list[dict]) -> bool:
    """
    True if run A Pareto-dominates run B: A is at-least-as-good on EVERY objective
    (respecting each objective's direction) AND strictly better on at least one.
    """
    at_least_as_good_all = True
    strictly_better_one = False

    for obj in objectives:
        name = obj["name"]
        direction = obj.get("direction", "lower")
        a_val = float(a_values[name])
        b_val = float(b_values[name])

        if not _is_at_least_as_good(a_val, b_val, direction):
            at_least_as_good_all = False
            break
        if _is_strictly_better(a_val, b_val, direction):
            strictly_better_one = True

    return at_least_as_good_all and strictly_better_one


def pareto_front(runs: list[dict], objectives: list[dict]) -> list[int]:
    """Return sorted list of run_id values that are non-dominated."""
    non_dominated: list[int] = []

    for i, run_a in enumerate(runs):
        dominated = False
        for j, run_b in enumerate(runs):
            if i == j:
                continue
            if dominates(run_b["values"], run_a["values"], objectives):
                dominated = True
                break
        if not dominated:
            non_dominated.append(run_a["run_id"])

    return sorted(non_dominated)


def select_best(
    runs: list[dict],
    objectives: list[dict],
    method: str = "scalarize",
) -> dict:
    """
    method in {"scalarize","desirability","pareto"}.

    Returns:
    {
      "method": str,
      "bounds": {obj_name: {"min":..,"max":..}},
      "scores": [{"run_id": int, "score": float}, ...]   # score per run by the method
                                                          # (for pareto: desirability),
      "best_run_id": int,        # argmax score; for pareto, max-desirability run within the front
      "best_score": float,
      "pareto_front": [run_id, ...],   # always computed and returned regardless of method
      "best_values": {obj_name: float} # raw measured values of the best run
    }

    Single-objective degenerate case (len(objectives)==1): best_run_id is the run with the
    best raw value for that objective's direction; scores still populated;
    pareto_front == [that run].
    """
    valid_methods = {"scalarize", "desirability", "pareto"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Must be one of: {sorted(valid_methods)}"
        )

    bounds = compute_bounds(runs, objectives)
    front = pareto_front(runs, objectives)
    warnings = [
        (f"Objective '{name}' varied by "
         f"{b['observed_max'] - b['observed_min']:.6g} across these runs, at or "
         f"below its declared noise_floor of {b['noise_floor']:.6g}. It was held "
         f"constant so it cannot decide the winner; its raw values are unchanged.")
        for name, b in bounds.items() if b.get("degenerate")
    ]

    # --- compute per-run scores -------------------------------------------
    if method == "scalarize":
        score_fn = lambda run: scalarize_run(run["values"], objectives, bounds)
    else:
        # desirability for both "desirability" and "pareto" methods
        score_fn = lambda run: desirability_run(run["values"], objectives, bounds)

    scores = [
        {"run_id": run["run_id"], "score": score_fn(run)}
        for run in runs
    ]

    # --- feasibility: guardrails are constraints, not score terms ----------
    feasibility = {run["run_id"]: is_feasible(run["values"], objectives) for run in runs}
    feasible_run_ids = [rid for rid, (ok, _) in feasibility.items() if ok]
    infeasible = {rid: names for rid, (ok, names) in feasibility.items() if not ok}
    feasible_set = set(feasible_run_ids)
    for rid, names in infeasible.items():
        warnings.append(
            f"Run {rid} violates guardrail(s) {names} and was excluded from selection."
        )

    # --- pick best run -------------------------------------------------------
    if method == "pareto":
        # best is the max-desirability run within the Pareto front
        front_set = set(front)
        candidate_scores = [s for s in scores
                            if s["run_id"] in front_set and s["run_id"] in feasible_set]
        if not candidate_scores:
            # Every front member fails a guardrail; fall back to any feasible run.
            candidate_scores = [s for s in scores if s["run_id"] in feasible_set]
    else:
        candidate_scores = [s for s in scores if s["run_id"] in feasible_set]

    result: dict[str, Any] = {
        "method": method,
        "bounds": bounds,
        "scores": scores,
        "best_run_id": None,
        "best_score": None,
        "pareto_front": front,
        "best_values": None,
        "feasible_run_ids": feasible_run_ids,
        "infeasible": infeasible,
        "contenders": [],
        "warnings": warnings,
    }
    if not candidate_scores:
        result["reason"] = (
            "No run satisfies every guardrail; nothing can be selected. "
            "Relax a bar you are willing to relax, or widen the factor ranges."
        )
        return result

    best_entry = max(candidate_scores, key=lambda s: s["score"])
    best_run_id = best_entry["run_id"]
    best_score = best_entry["score"]

    # look up raw values for best run
    best_run = next(r for r in runs if r["run_id"] == best_run_id)
    best_values = {name: float(best_run["values"][name]) for name in best_run["values"]}

    # Near ties: a winner chosen by a hair (often a target that saturates one
    # objective's desirability at 1.0) is a guess wearing a verdict. Report
    # every candidate within CONTENDER_TOLERANCE of the best score so the
    # caller confirms each rather than trusting the argmax.
    contenders = sorted(
        (s["run_id"] for s in candidate_scores
         if best_score - s["score"] <= CONTENDER_TOLERANCE),
        key=lambda rid: -next(s["score"] for s in candidate_scores if s["run_id"] == rid),
    )
    if len(contenders) > 1:
        warnings.append(
            f"Runs {contenders} score within {CONTENDER_TOLERANCE:g} of the best; treat "
            f"them as a tie and confirm each before choosing. If a 'target' was met by "
            f"several, the ranking among them is saturation, not evidence."
        )

    result.update({
        "best_run_id": best_run_id,
        "best_score": best_score,
        "best_values": best_values,
        "contenders": contenders,
    })
    return result


# ---------------------------------------------------------------------------
# Streaming-loop scoring (autoresearch / loop.py)
# ---------------------------------------------------------------------------

# NOTE: baseline_aggregate uses a different normalization strategy than
# select_best's batch min-max.  select_best sees ALL runs simultaneously and
# can normalize each objective to [0,1] across the observed range.  The
# autoresearch loop is *streaming* — it scores one candidate at a time against
# a fixed starting point.  Using batch min-max here would require re-running
# every past candidate on every iteration.  Instead we express each
# objective's score as an improvement RATIO vs the fixed baseline: a ratio > 1
# means the candidate beats the baseline on that objective.  The weighted sum
# of ratios gives a scalar that the loop can compare across streaming
# iterations without ever needing to collect a run set first.

_BASELINE_EPSILON = 1e-9  # guard against division by zero (documented below)


def baseline_aggregate(values: dict, baseline: dict, objectives: list[dict]) -> float:
    """Weighted aggregate of per-objective improvement RATIOS vs a fixed baseline.

    For each objective (weights normalized to sum to 1):
      'lower'  better: r_i = baseline_i / value_i   (>1 = improved)
      'higher' better: r_i = value_i / baseline_i   (>1 = improved)

    Returns sum_i w_i * r_i. Higher = better. Guards against div-by-zero:
    when the denominator is 0 (or smaller than _BASELINE_EPSILON) it is
    replaced with _BASELINE_EPSILON (1e-9).  This makes the ratio very large
    when the baseline was zero and the new value is positive (big improvement)
    or 0 when the new value is also zero (no change).  The epsilon value is
    documented here and in tests so callers know the guard is active.

    This is the loop's keep/revert score; it is distinct from select_best's
    batch min-max normalization because the loop is streaming (one candidate
    at a time against a fixed starting point).
    """
    w_norm = _normalized_weights(objectives)
    total = 0.0
    for w, obj in zip(w_norm, objectives):
        name = obj["name"]
        direction = obj.get("direction", "lower")
        v = float(values[name])
        b = float(baseline[name])

        if direction == "lower":
            # lower is better: ratio > 1 when value < baseline
            denom = v if v >= _BASELINE_EPSILON else _BASELINE_EPSILON
            r_i = b / denom
        else:
            # higher is better: ratio > 1 when value > baseline
            denom = b if b >= _BASELINE_EPSILON else _BASELINE_EPSILON
            r_i = v / denom

        total += w * r_i
    return total
