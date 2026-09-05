# SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0
"""
Tests for scripts/objectives.py

All fixtures are hand-verified numeric examples. No randomness; fully deterministic.
"""

import math
import sys
import os

# Allow importing from scripts/ without an installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest
from objectives import (
    compute_bounds,
    normalize,
    scalarize_run,
    desirability_run,
    dominates,
    pareto_front,
    select_best,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TWO_OBJ = [
    {"name": "latency_ms", "direction": "lower",  "weight": 2.0},
    {"name": "coverage",   "direction": "higher", "weight": 1.0},
]

FOUR_RUNS = [
    {"run_id": 1, "values": {"latency_ms": 100.0, "coverage": 0.9}},
    {"run_id": 2, "values": {"latency_ms": 200.0, "coverage": 0.8}},
    {"run_id": 3, "values": {"latency_ms": 150.0, "coverage": 0.6}},
    {"run_id": 4, "values": {"latency_ms": 120.0, "coverage": 0.7}},
]

# Bounds for FOUR_RUNS / TWO_OBJ (hand-computed):
#   latency_ms: min=100, max=200
#   coverage:   min=0.6, max=0.9
FOUR_RUNS_BOUNDS = {
    "latency_ms": {"min": 100.0, "max": 200.0},
    "coverage":   {"min": 0.6,   "max": 0.9},
}


# ---------------------------------------------------------------------------
# compute_bounds
# ---------------------------------------------------------------------------

class TestComputeBounds:
    def test_correct_bounds(self):
        b = compute_bounds(FOUR_RUNS, TWO_OBJ)
        assert b["latency_ms"] == {"min": 100.0, "max": 200.0}
        assert b["coverage"]   == {"min": 0.6,   "max": 0.9}

    def test_single_run(self):
        runs = [{"run_id": 1, "values": {"latency_ms": 50.0, "coverage": 0.75}}]
        b = compute_bounds(runs, TWO_OBJ)
        assert b["latency_ms"] == {"min": 50.0, "max": 50.0}
        assert b["coverage"]   == {"min": 0.75, "max": 0.75}

    def test_missing_value_raises(self):
        bad_runs = [
            {"run_id": 1, "values": {"latency_ms": 100.0}},  # coverage missing
            {"run_id": 2, "values": {"latency_ms": 200.0, "coverage": 0.8}},
        ]
        with pytest.raises(ValueError, match="coverage"):
            compute_bounds(bad_runs, TWO_OBJ)

    def test_multiple_objectives(self):
        objs = [
            {"name": "a", "direction": "lower"},
            {"name": "b", "direction": "higher"},
        ]
        runs = [
            {"run_id": 1, "values": {"a": 1.0, "b": 10.0}},
            {"run_id": 2, "values": {"a": 3.0, "b":  5.0}},
            {"run_id": 3, "values": {"a": 2.0, "b":  8.0}},
        ]
        b = compute_bounds(runs, objs)
        assert b["a"] == {"min": 1.0, "max": 3.0}
        assert b["b"] == {"min": 5.0, "max": 10.0}


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_higher_best_is_1(self):
        assert normalize(10.0, "higher", 0.0, 10.0) == pytest.approx(1.0)

    def test_higher_worst_is_0(self):
        assert normalize(0.0, "higher", 0.0, 10.0) == pytest.approx(0.0)

    def test_higher_midpoint(self):
        assert normalize(5.0, "higher", 0.0, 10.0) == pytest.approx(0.5)

    def test_lower_best_is_1(self):
        # Lowest value is best for "lower"; value=0 when lo=0, hi=10
        assert normalize(0.0, "lower", 0.0, 10.0) == pytest.approx(1.0)

    def test_lower_worst_is_0(self):
        assert normalize(10.0, "lower", 0.0, 10.0) == pytest.approx(0.0)

    def test_lower_midpoint(self):
        assert normalize(5.0, "lower", 0.0, 10.0) == pytest.approx(0.5)

    def test_degenerate_hi_equals_lo_returns_1(self):
        assert normalize(7.0, "higher", 7.0, 7.0) == 1.0
        assert normalize(7.0, "lower",  7.0, 7.0) == 1.0

    def test_higher_endpoints_explicit(self):
        # lo=2, hi=8: value=2 -> 0.0; value=8 -> 1.0
        assert normalize(2.0, "higher", 2.0, 8.0) == pytest.approx(0.0)
        assert normalize(8.0, "higher", 2.0, 8.0) == pytest.approx(1.0)

    def test_lower_endpoints_explicit(self):
        # lo=2, hi=8: value=8 -> 0.0; value=2 -> 1.0
        assert normalize(8.0, "lower", 2.0, 8.0) == pytest.approx(0.0)
        assert normalize(2.0, "lower", 2.0, 8.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# scalarize_run
# ---------------------------------------------------------------------------

class TestScalarizeRun:
    def test_best_run_scores_1(self):
        # Run 1 is best on latency (100 = min) and best on coverage (0.9 = max)
        result = scalarize_run(
            {"latency_ms": 100.0, "coverage": 0.9},
            TWO_OBJ,
            FOUR_RUNS_BOUNDS,
        )
        assert result == pytest.approx(1.0)

    def test_worst_run_scores_0(self):
        # Run with worst latency (200) and worst coverage (0.6)
        result = scalarize_run(
            {"latency_ms": 200.0, "coverage": 0.6},
            TWO_OBJ,
            FOUR_RUNS_BOUNDS,
        )
        assert result == pytest.approx(0.0)

    def test_hand_checked_weighted_mix(self):
        # latency=150 (mid): lower -> normalize = (200-150)/(200-100) = 0.5
        # coverage=0.75 (mid): higher -> normalize = (0.75-0.6)/(0.9-0.6) = 0.5
        # weights: latency=2, coverage=1 -> normalized: 2/3, 1/3
        # score = (2/3)*0.5 + (1/3)*0.5 = 0.5
        result = scalarize_run(
            {"latency_ms": 150.0, "coverage": 0.75},
            TWO_OBJ,
            FOUR_RUNS_BOUNDS,
        )
        assert result == pytest.approx(0.5)

    def test_asymmetric_weighted_mix(self):
        # latency=100 (best=1.0), coverage=0.6 (worst=0.0)
        # weights 2/3 on latency, 1/3 on coverage
        # score = (2/3)*1.0 + (1/3)*0.0 = 2/3
        result = scalarize_run(
            {"latency_ms": 100.0, "coverage": 0.6},
            TWO_OBJ,
            FOUR_RUNS_BOUNDS,
        )
        assert result == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# desirability_run
# ---------------------------------------------------------------------------

class TestDesirabilityRun:
    def test_zero_on_one_objective_gives_zero(self):
        # latency=200 is worst; direction=lower -> d_i=0 -> D=0
        result = desirability_run(
            {"latency_ms": 200.0, "coverage": 0.9},
            TWO_OBJ,
            FOUR_RUNS_BOUNDS,
        )
        assert result == 0.0

    def test_zero_coverage_gives_zero(self):
        # coverage=0.6 is worst; direction=higher -> d_i=0 -> D=0
        result = desirability_run(
            {"latency_ms": 100.0, "coverage": 0.6},
            TWO_OBJ,
            FOUR_RUNS_BOUNDS,
        )
        assert result == 0.0

    def test_best_run_scores_1(self):
        # Both objectives at best -> D = 1.0
        result = desirability_run(
            {"latency_ms": 100.0, "coverage": 0.9},
            TWO_OBJ,
            FOUR_RUNS_BOUNDS,
        )
        assert result == pytest.approx(1.0)

    def test_two_objective_geometric_mean(self):
        # latency=150: d_latency = 0.5  (direction=lower, lo=100, hi=200)
        # coverage=0.75: d_coverage = 0.5  (direction=higher, lo=0.6, hi=0.9)
        # weights: w_latency=2, w_coverage=1, w_sum=3
        # D = exp((2*ln(0.5) + 1*ln(0.5)) / 3)
        #   = exp(3*ln(0.5)/3) = exp(ln(0.5)) = 0.5
        result = desirability_run(
            {"latency_ms": 150.0, "coverage": 0.75},
            TWO_OBJ,
            FOUR_RUNS_BOUNDS,
        )
        assert result == pytest.approx(0.5)

    def test_equal_weights_two_objectives(self):
        # With equal weights: D = (d1 * d2) ** 0.5
        objs = [
            {"name": "a", "direction": "lower",  "weight": 1.0},
            {"name": "b", "direction": "higher", "weight": 1.0},
        ]
        bounds = {"a": {"min": 0.0, "max": 10.0}, "b": {"min": 0.0, "max": 10.0}}
        # a=2.5: d_a = (10-2.5)/10 = 0.75
        # b=4.0: d_b = 4/10 = 0.4
        # D = (0.75 * 0.4) ** 0.5 = 0.3 ** 0.5
        result = desirability_run({"a": 2.5, "b": 4.0}, objs, bounds)
        assert result == pytest.approx(math.sqrt(0.75 * 0.4))


# ---------------------------------------------------------------------------
# dominates
# ---------------------------------------------------------------------------

class TestDominates:
    def test_clear_domination(self):
        # A better on latency (lower=better) and better on coverage (higher=better)
        a = {"latency_ms": 100.0, "coverage": 0.9}
        b = {"latency_ms": 200.0, "coverage": 0.7}
        assert dominates(a, b, TWO_OBJ) is True

    def test_incomparable_not_dominated(self):
        # A better on latency, worse on coverage -> neither dominates
        a = {"latency_ms": 100.0, "coverage": 0.6}
        b = {"latency_ms": 200.0, "coverage": 0.9}
        assert dominates(a, b, TWO_OBJ) is False
        assert dominates(b, a, TWO_OBJ) is False

    def test_equal_on_all_not_dominated(self):
        a = {"latency_ms": 150.0, "coverage": 0.75}
        b = {"latency_ms": 150.0, "coverage": 0.75}
        assert dominates(a, b, TWO_OBJ) is False

    def test_better_on_one_equal_on_other(self):
        # A strictly better on latency, equal on coverage -> A dominates B
        a = {"latency_ms": 100.0, "coverage": 0.8}
        b = {"latency_ms": 150.0, "coverage": 0.8}
        assert dominates(a, b, TWO_OBJ) is True

    def test_reverse_domination(self):
        a = {"latency_ms": 200.0, "coverage": 0.6}  # worst on both
        b = {"latency_ms": 100.0, "coverage": 0.9}  # best on both
        assert dominates(b, a, TWO_OBJ) is True
        assert dominates(a, b, TWO_OBJ) is False


# ---------------------------------------------------------------------------
# pareto_front
# ---------------------------------------------------------------------------

class TestParetoFront:
    def test_hand_built_four_runs(self):
        # FOUR_RUNS with TWO_OBJ (latency lower=better, coverage higher=better):
        #
        # Run 1: latency=100, coverage=0.9  -> best latency, best coverage -> non-dominated
        # Run 2: latency=200, coverage=0.8  -> worst latency; dominated by Run 1 (1 wins both)
        # Run 3: latency=150, coverage=0.6  -> coverage=0.6 worst; dominated by Run 4
        #                                      (Run4: 120<150 AND 0.7>0.6)
        # Run 4: latency=120, coverage=0.7  -> dominated by Run 1 (100<120 AND 0.9>0.7)
        #
        # Only Run 1 is non-dominated.
        front = pareto_front(FOUR_RUNS, TWO_OBJ)
        assert front == [1]

    def test_two_incomparable_runs_both_on_front(self):
        # A: low latency, low coverage; B: high latency, high coverage -> incomparable
        runs = [
            {"run_id": 10, "values": {"latency_ms": 100.0, "coverage": 0.6}},
            {"run_id": 20, "values": {"latency_ms": 200.0, "coverage": 0.9}},
        ]
        front = pareto_front(runs, TWO_OBJ)
        assert sorted(front) == [10, 20]

    def test_three_runs_one_dominated(self):
        # Run 3 dominates Run 2 (better on both)
        runs = [
            {"run_id": 1, "values": {"latency_ms": 100.0, "coverage": 0.6}},
            {"run_id": 2, "values": {"latency_ms": 200.0, "coverage": 0.7}},
            {"run_id": 3, "values": {"latency_ms": 180.0, "coverage": 0.8}},
        ]
        # Run 1: 100 latency / 0.6 coverage
        # Run 2: 200 latency / 0.7 coverage  -- run 3 has better latency (180<200) AND coverage (0.8>0.7)
        # Run 3: 180 latency / 0.8 coverage
        # Run 1 vs 3: latency 1 wins (100<180); coverage 3 wins (0.8>0.6) -> incomparable
        # Front: [1, 3]
        front = pareto_front(runs, TWO_OBJ)
        assert sorted(front) == [1, 3]

    def test_all_dominated_except_one(self):
        runs = [
            {"run_id": 1, "values": {"latency_ms": 100.0, "coverage": 0.9}},
            {"run_id": 2, "values": {"latency_ms": 110.0, "coverage": 0.85}},
            {"run_id": 3, "values": {"latency_ms": 120.0, "coverage": 0.80}},
        ]
        # Run 1 dominates all others (strictly best latency AND coverage)
        front = pareto_front(runs, TWO_OBJ)
        assert front == [1]


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------

class TestSelectBest:
    def _result_keys(self, result):
        return set(result.keys())

    def test_output_keys_present_scalarize(self):
        result = select_best(FOUR_RUNS, TWO_OBJ, method="scalarize")
        expected_keys = {
            "method", "bounds", "scores", "best_run_id",
            "best_score", "pareto_front", "best_values", "warnings",
            "feasible_run_ids", "infeasible",
        }
        assert expected_keys == self._result_keys(result)

    def test_output_keys_present_desirability(self):
        result = select_best(FOUR_RUNS, TWO_OBJ, method="desirability")
        expected_keys = {
            "method", "bounds", "scores", "best_run_id",
            "best_score", "pareto_front", "best_values", "warnings",
            "feasible_run_ids", "infeasible",
        }
        assert expected_keys == self._result_keys(result)

    def test_output_keys_present_pareto(self):
        result = select_best(FOUR_RUNS, TWO_OBJ, method="pareto")
        expected_keys = {
            "method", "bounds", "scores", "best_run_id",
            "best_score", "pareto_front", "best_values", "warnings",
            "feasible_run_ids", "infeasible",
        }
        assert expected_keys == self._result_keys(result)

    def test_pareto_front_always_returned(self):
        for method in ("scalarize", "desirability", "pareto"):
            result = select_best(FOUR_RUNS, TWO_OBJ, method=method)
            assert "pareto_front" in result
            assert isinstance(result["pareto_front"], list)

    def test_scalarize_best_is_run1(self):
        # Run 1 is best on both objectives -> score=1.0 -> must be selected
        result = select_best(FOUR_RUNS, TWO_OBJ, method="scalarize")
        assert result["best_run_id"] == 1
        assert result["best_score"] == pytest.approx(1.0)
        assert result["method"] == "scalarize"

    def test_desirability_best_is_run1(self):
        # Run 1: both d_i=1.0 -> D=1.0
        result = select_best(FOUR_RUNS, TWO_OBJ, method="desirability")
        assert result["best_run_id"] == 1
        assert result["best_score"] == pytest.approx(1.0)
        assert result["method"] == "desirability"

    def test_pareto_best_is_run1(self):
        result = select_best(FOUR_RUNS, TWO_OBJ, method="pareto")
        assert result["best_run_id"] == 1
        assert result["method"] == "pareto"

    def test_scores_list_has_all_runs(self):
        result = select_best(FOUR_RUNS, TWO_OBJ, method="scalarize")
        run_ids_in_scores = {s["run_id"] for s in result["scores"]}
        assert run_ids_in_scores == {1, 2, 3, 4}

    def test_best_values_are_raw_measurements(self):
        result = select_best(FOUR_RUNS, TWO_OBJ, method="scalarize")
        assert result["best_values"] == {"latency_ms": 100.0, "coverage": 0.9}

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            select_best(FOUR_RUNS, TWO_OBJ, method="invalid")

    def test_single_objective_degenerate_scalarize(self):
        # Single objective: direction=lower -> best run is the one with min value
        objs = [{"name": "latency_ms", "direction": "lower", "weight": 1.0}]
        runs = [
            {"run_id": 1, "values": {"latency_ms": 300.0}},
            {"run_id": 2, "values": {"latency_ms": 100.0}},  # best
            {"run_id": 3, "values": {"latency_ms": 200.0}},
        ]
        result = select_best(runs, objs, method="scalarize")
        assert result["best_run_id"] == 2
        # pareto_front should contain only the best run in a single-objective case
        assert result["pareto_front"] == [2]

    def test_single_objective_degenerate_higher(self):
        objs = [{"name": "coverage", "direction": "higher", "weight": 1.0}]
        runs = [
            {"run_id": 1, "values": {"coverage": 0.7}},
            {"run_id": 2, "values": {"coverage": 0.9}},  # best
            {"run_id": 3, "values": {"coverage": 0.5}},
        ]
        result = select_best(runs, objs, method="scalarize")
        assert result["best_run_id"] == 2
        assert result["pareto_front"] == [2]

    def test_bounds_match_compute_bounds(self):
        result = select_best(FOUR_RUNS, TWO_OBJ, method="scalarize")
        expected_bounds = compute_bounds(FOUR_RUNS, TWO_OBJ)
        assert result["bounds"] == expected_bounds

    def test_default_method_is_scalarize(self):
        result = select_best(FOUR_RUNS, TWO_OBJ)
        assert result["method"] == "scalarize"


# ---------------------------------------------------------------------------
# baseline_aggregate (streaming-loop scoring)
# ---------------------------------------------------------------------------

from objectives import baseline_aggregate  # noqa: E402


class TestBaselineAggregate:
    # Two objectives used across most cases:
    #   lat: lower=better, weight=0.6
    #   cost: lower=better, weight=0.4
    # Normalized weights: lat=0.6, cost=0.4 (already sum to 1)
    TWO_LOWER = [
        {"name": "lat",  "direction": "lower", "weight": 0.6},
        {"name": "cost", "direction": "lower", "weight": 0.4},
    ]

    def test_baseline_equals_values_returns_1(self):
        # Every ratio r_i = baseline_i / value_i = 1 => aggregate = 1.0
        baseline = {"lat": 10.0, "cost": 5.0}
        values   = {"lat": 10.0, "cost": 5.0}
        result = baseline_aggregate(values, baseline, self.TWO_LOWER)
        assert result == pytest.approx(1.0)

    def test_both_improved_returns_greater_than_1(self):
        # lat: 10/8 = 1.25; cost: 5/5 = 1.0
        # aggregate = 0.6*1.25 + 0.4*1.0 = 0.75 + 0.40 = 1.15
        baseline = {"lat": 10.0, "cost": 5.0}
        values   = {"lat":  8.0, "cost": 5.0}
        result = baseline_aggregate(values, baseline, self.TWO_LOWER)
        assert result > 1.0
        assert result == pytest.approx(0.6 * (10.0 / 8.0) + 0.4 * (5.0 / 5.0))

    def test_mixed_improvement_hand_checked(self):
        # lat improved: 10/8 = 1.25; cost worsened: 5/7 ≈ 0.7143
        # aggregate = 0.6*1.25 + 0.4*(5/7) = 0.75 + 0.4*0.714286 = 0.75 + 0.285714 ≈ 1.035714
        baseline = {"lat": 10.0, "cost": 5.0}
        values   = {"lat":  8.0, "cost": 7.0}
        expected = 0.6 * (10.0 / 8.0) + 0.4 * (5.0 / 7.0)
        result = baseline_aggregate(values, baseline, self.TWO_LOWER)
        assert result == pytest.approx(expected)

    def test_direction_higher_better(self):
        # coverage: higher=better, weight=1.0 (only objective)
        objs = [{"name": "cov", "direction": "higher", "weight": 1.0}]
        baseline = {"cov": 0.8}
        # cov improved from 0.8 -> 0.9: ratio = 0.9/0.8 = 1.125 > 1
        result_improved = baseline_aggregate({"cov": 0.9}, baseline, objs)
        assert result_improved == pytest.approx(0.9 / 0.8)
        assert result_improved > 1.0
        # cov worsened from 0.8 -> 0.7: ratio = 0.7/0.8 = 0.875 < 1
        result_worse = baseline_aggregate({"cov": 0.7}, baseline, objs)
        assert result_worse == pytest.approx(0.7 / 0.8)
        assert result_worse < 1.0

    def test_direction_lower_better(self):
        # lat: lower=better, weight=1.0
        objs = [{"name": "lat", "direction": "lower", "weight": 1.0}]
        baseline = {"lat": 10.0}
        # improved: 10/8 = 1.25
        result_improved = baseline_aggregate({"lat": 8.0}, baseline, objs)
        assert result_improved == pytest.approx(10.0 / 8.0)
        assert result_improved > 1.0
        # worsened: 10/12 ≈ 0.833
        result_worse = baseline_aggregate({"lat": 12.0}, baseline, objs)
        assert result_worse == pytest.approx(10.0 / 12.0)
        assert result_worse < 1.0

    def test_div_by_zero_value_zero_lower(self):
        # lower-better: denom = value. If value=0, use epsilon.
        # ratio = baseline / epsilon (very large but finite)
        objs = [{"name": "lat", "direction": "lower", "weight": 1.0}]
        result = baseline_aggregate({"lat": 0.0}, {"lat": 10.0}, objs)
        assert math.isfinite(result)
        assert result > 1.0  # improved (lower is better, value went to 0)

    def test_div_by_zero_baseline_zero_higher(self):
        # higher-better: denom = baseline. If baseline=0, use epsilon.
        # ratio = value / epsilon (very large but finite)
        objs = [{"name": "cov", "direction": "higher", "weight": 1.0}]
        result = baseline_aggregate({"cov": 0.9}, {"cov": 0.0}, objs)
        assert math.isfinite(result)
        assert result > 1.0

    def test_weight_normalization(self):
        # Weights 2 and 2 should normalize to 0.5 each — same as weights 1 and 1.
        objs_2_2 = [
            {"name": "lat",  "direction": "lower", "weight": 2.0},
            {"name": "cost", "direction": "lower", "weight": 2.0},
        ]
        objs_1_1 = [
            {"name": "lat",  "direction": "lower", "weight": 1.0},
            {"name": "cost", "direction": "lower", "weight": 1.0},
        ]
        baseline = {"lat": 10.0, "cost": 5.0}
        values   = {"lat":  8.0, "cost": 4.0}
        r1 = baseline_aggregate(values, baseline, objs_2_2)
        r2 = baseline_aggregate(values, baseline, objs_1_1)
        assert r1 == pytest.approx(r2)


class TestNoiseFloor:
    """A sub-noise wobble on one objective must not decide the winner.

    Regression guard for a demonstrated defect: min-max normalisation stretches
    whatever range it observes to fill [0,1], so a 0.05% wobble on one
    objective normalises exactly as fully as a real spread on another. Under
    `desirability` the geometric mean then scores the batch-minimum run at 0.0,
    tying a run that is best on two of three objectives with the worst run.
    """

    # run 7 is best on latency AND temp, and loses only on a 0.0005 recall wobble
    RUNS = [
        {"run_id": 6, "values": {"latency_ms": 90.0, "recall": 0.9823, "temp": 80.0}},
        {"run_id": 7, "values": {"latency_ms": 45.0, "recall": 0.9818, "temp": 75.0}},
    ]
    OBJS_NO_FLOOR = [
        {"name": "latency_ms", "direction": "lower",  "weight": 0.4},
        {"name": "recall",     "direction": "higher", "weight": 0.4},
        {"name": "temp",       "direction": "lower",  "weight": 0.2},
    ]
    OBJS_WITH_FLOOR = [
        {"name": "latency_ms", "direction": "lower",  "weight": 0.4},
        {"name": "recall",     "direction": "higher", "weight": 0.4,
         "noise_floor": 0.01},
        {"name": "temp",       "direction": "lower",  "weight": 0.2},
    ]

    def test_without_a_floor_desirability_zeroes_the_better_run(self):
        # The defect this exists to correct. Run 7 is better on latency (45 vs
        # 90) and on temp (75 vs 80), and loses only a 0.0005 wobble on recall.
        # Desirability is a geometric mean, so one d_i of 0.0 forces D = 0.0 --
        # scoring the better run identically to the worst run in the batch.
        r = select_best(self.RUNS, self.OBJS_NO_FLOOR, method="desirability")
        by_id = {s["run_id"]: s["score"] for s in r["scores"]}
        assert by_id[7] == 0.0
        assert r["best_run_id"] == 6

    def test_floor_hands_the_win_to_the_genuinely_better_run(self):
        r = select_best(self.RUNS, self.OBJS_WITH_FLOOR, method="scalarize")
        assert r["best_run_id"] == 7

    def test_desirability_no_longer_zeroes_the_better_run(self):
        r = select_best(self.RUNS, self.OBJS_WITH_FLOOR, method="desirability")
        by_id = {s["run_id"]: s["score"] for s in r["scores"]}
        assert by_id[7] > 0.0
        assert r["best_run_id"] == 7

    def test_collapsed_objective_is_reported_not_hidden(self):
        r = select_best(self.RUNS, self.OBJS_WITH_FLOOR, method="scalarize")
        assert r["bounds"]["recall"].get("degenerate") is True
        assert any("recall" in w for w in r["warnings"])
        # raw measurements are untouched
        assert r["best_values"]["recall"] == 0.9818

    def test_real_spread_above_the_floor_still_counts(self):
        runs = [
            {"run_id": 1, "values": {"latency_ms": 90.0, "recall": 0.60, "temp": 80.0}},
            {"run_id": 2, "values": {"latency_ms": 45.0, "recall": 0.99, "temp": 75.0}},
        ]
        r = select_best(runs, self.OBJS_WITH_FLOOR, method="scalarize")
        assert r["bounds"]["recall"].get("degenerate") is not True
        assert r["best_run_id"] == 2
