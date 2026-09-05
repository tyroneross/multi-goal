#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for doe.py - design generation + effects analysis accuracy.

Ported from build-loop/scripts/test_optimize_doe.py; all design-matrix tests
are byte-faithful. Multi-objective analyze tests are appended below.

Tests:
  DesignGeneratorTests    - full / fractional / PB matrices shape + orthogonality
  RoutingTests            - auto-routing picks the right design type by k
  EffectsAccuracyTests    - OLS recovers known ground-truth coefficients
  LevelMappingTests       - ±1 → concrete level mapping
  CliRoundTripTests       - generate → analyze (legacy single-metric) pipeline
  PyDOE3EquivalenceTests  - skipped unless pyDOE3 installed
  MultiObjectiveAnalyzeTests  - NEW: multi-objective analyze via --objectives
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    # Never call sys.exit() at module scope: pytest imports test modules inside
    # its collection hook, where a SystemExit escapes as INTERNALERROR and
    # aborts the WHOLE session ("no tests ran") instead of skipping one module.
    import pytest

    pytest.skip("test_doe.py requires numpy", allow_module_level=True)

# Robust import of doe.py from scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import doe  # noqa: E402
import doe_stats  # noqa: E402

SCRIPT = SCRIPTS_DIR / "doe.py"


# ---------------------------------------------------------------------------
# Direct unit tests on the design generators (ported, byte-faithful)
# ---------------------------------------------------------------------------

class DesignGeneratorTests(unittest.TestCase):
    def test_full_factorial_shape(self) -> None:
        for k in range(1, 6):
            with self.subTest(k=k):
                d = doe.full_factorial_2level(k)
                self.assertEqual(d.shape, (2 ** k, k))
                self.assertTrue(np.all((d == -1) | (d == 1)))

    def test_full_factorial_orthogonal(self) -> None:
        for k in range(2, 6):
            with self.subTest(k=k):
                d = doe.full_factorial_2level(k)
                gram = d.T @ d
                off_diag = gram - np.diag(np.diag(gram))
                self.assertTrue(np.allclose(off_diag, 0))

    def test_fracfact_shape(self) -> None:
        # 2^(5-2) → 8 runs, 5 factors
        d = doe.fracfact("a b c ab ac")
        self.assertEqual(d.shape, (8, 5))

    def test_fracfact_orthogonal(self) -> None:
        for k, gen in doe.FRACFACT_8_RUN.items():
            with self.subTest(k=k):
                d = doe.fracfact(gen)
                gram = d.T @ d
                off_diag = gram - np.diag(np.diag(gram))
                self.assertTrue(np.allclose(off_diag, 0))

    def test_pb_12_shape(self) -> None:
        d = doe.plackett_burman_12()
        self.assertEqual(d.shape, (12, 11))

    def test_pb_12_orthogonal(self) -> None:
        d = doe.plackett_burman_12()
        gram = d.T @ d
        off_diag = gram - np.diag(np.diag(gram))
        self.assertTrue(np.allclose(off_diag, 0),
                        f"PB off-diag max = {np.max(np.abs(off_diag))}")


class RoutingTests(unittest.TestCase):
    def test_select_design(self) -> None:
        cases = {1: "autoresearch", 2: "full", 3: "full",
                 4: "fractional", 5: "fractional", 7: "fractional",
                 8: "pb", 11: "pb"}
        for k, expected in cases.items():
            with self.subTest(k=k):
                self.assertEqual(doe.select_design(k), expected)

    def test_build_design_dispatch(self) -> None:
        m, name = doe.build_design(3, "full")
        self.assertEqual(m.shape, (8, 3))
        self.assertIn("full factorial", name)

        m, name = doe.build_design(5, "fractional")
        self.assertEqual(m.shape, (8, 5))
        self.assertIn("fractional factorial", name)

        m, name = doe.build_design(8, "pb")
        self.assertEqual(m.shape, (12, 8))
        self.assertIn("Plackett-Burman", name)


class EffectsAccuracyTests(unittest.TestCase):
    """Recover known ground-truth coefficients from synthetic measurements."""

    def test_full_factorial_no_noise(self) -> None:
        d = doe.full_factorial_2level(3)
        # y = 10 + 5*x1 + 2*x2 - 0.3*x3 + 0.5*x1*x2
        y = 10 + 5 * d[:, 0] + 2 * d[:, 1] - 0.3 * d[:, 2] + 0.5 * d[:, 0] * d[:, 1]
        e = doe.fit_effects(d, y, include_interactions=True)
        self.assertAlmostEqual(e["intercept"], 10, places=8)
        self.assertAlmostEqual(e["main"][0], 5, places=8)
        self.assertAlmostEqual(e["main"][1], 2, places=8)
        self.assertAlmostEqual(e["main"][2], -0.3, places=8)
        self.assertAlmostEqual(e["interactions"][(0, 1)], 0.5, places=8)

    def test_fractional_with_noise(self) -> None:
        d = doe.fracfact("a b c ab ac")
        rng = np.random.default_rng(42)
        truth_main = [3.0, -1.5, 0.8, 2.2, -0.4]
        y = 20 + sum(t * d[:, i] for i, t in enumerate(truth_main)) + rng.normal(0, 0.1, 8)
        e = doe.fit_effects(d, y, include_interactions=False)
        for i, expected in enumerate(truth_main):
            with self.subTest(factor=f"x{i+1}"):
                self.assertAlmostEqual(e["main"][i], expected, delta=0.2)

    def test_pb_screening_identifies_vital_few(self) -> None:
        d = doe.plackett_burman_12()
        rng = np.random.default_rng(11)
        # Only first 3 factors active
        y = 50 + 4 * d[:, 0] - 2.5 * d[:, 1] + 1.0 * d[:, 2] + rng.normal(0, 0.3, 12)
        e = doe.fit_effects(d, y, include_interactions=False)
        # Top 3 by |effect| should be factors 0, 1, 2
        ranking = sorted(e["main"].items(), key=lambda kv: -abs(kv[1]))
        top3 = {idx for idx, _ in ranking[:3]}
        self.assertEqual(top3, {0, 1, 2})


class LevelMappingTests(unittest.TestCase):
    def test_low_high_mapping(self) -> None:
        d = np.array([[-1, 1], [1, -1]], dtype=float)
        factors = [
            {"name": "x", "low": 16, "high": 64},
            {"name": "y", "low": 1, "high": 5},
        ]
        runs = doe.map_levels(d, factors)
        self.assertEqual(runs[0]["_factors"], {"x": 16, "y": 5})
        self.assertEqual(runs[1]["_factors"], {"x": 64, "y": 1})

    def test_levels_array(self) -> None:
        d = np.array([[-1], [1]], dtype=float)
        factors = [{"name": "x", "levels": ["off", "on"]}]
        runs = doe.map_levels(d, factors)
        self.assertEqual(runs[0]["_factors"], {"x": "off"})
        self.assertEqual(runs[1]["_factors"], {"x": "on"})


class CliRoundTripTests(unittest.TestCase):
    """Exercise the CLI end-to-end: generate matrix, fake measurements, analyze."""

    def test_generate_then_analyze(self) -> None:
        factors = [
            {"name": "batch_size", "low": 16, "high": 64},
            {"name": "retries", "low": 1, "high": 5},
            {"name": "workers", "low": 2, "high": 8},
        ]
        # Step 1: generate
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "generate",
             "--factors", json.dumps(factors), "--design", "auto", "--seed", "1"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        design = json.loads(r.stdout)
        self.assertEqual(design["design"]["type"], "full")
        self.assertEqual(design["design"]["n_runs"], 8)

        # Step 2: synthesize measurements where x1 dominates, x3 mildly opposes
        with tempfile.TemporaryDirectory() as tmp:
            design_path = Path(tmp) / "design.json"
            results_path = Path(tmp) / "results.jsonl"
            design_path.write_text(json.dumps(design))
            matrix = np.array(design["matrix"])
            y = 100 - 8 * matrix[:, 0] + 0.5 * matrix[:, 1] + 1 * matrix[:, 2]
            with open(results_path, "w") as f:
                for run_id in range(len(y)):
                    f.write(json.dumps({"run_id": run_id, "value": float(y[run_id])}) + "\n")

            # Step 3: analyze (legacy single-metric path)
            r2 = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(design_path), "--results", str(results_path),
                 "--direction", "lower"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r2.returncode, 0, r2.stderr)
            analysis = json.loads(r2.stdout)
            top = analysis["ranked_effects"][0]
            self.assertEqual(top["term"], "batch_size")
            self.assertAlmostEqual(top["effect"], -8, delta=0.5)

    def test_analyze_emits_best_factors(self) -> None:
        factors = [
            {"name": "batch_size", "low": 16, "high": 64},
            {"name": "retries", "low": 1, "high": 5},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "generate",
                 "--factors", json.dumps(factors), "--design", "full", "--seed", "0"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            design = json.loads(r.stdout)
            (tmp / "design.json").write_text(json.dumps(design))
            # Fake measurements: y = 100 - 8*x1 - 2*x2 (lower is better)
            matrix = np.array(design["matrix"])
            y = 100 - 8 * matrix[:, 0] - 2 * matrix[:, 1]
            with (tmp / "results.jsonl").open("w") as f:
                for i in range(len(y)):
                    f.write(json.dumps({"run_id": i, "value": float(y[i])}) + "\n")
            # Analyze with direction=lower
            r2 = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(tmp / "design.json"),
                 "--results", str(tmp / "results.jsonl"),
                 "--direction", "lower"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r2.returncode, 0, r2.stderr)
            effects = json.loads(r2.stdout)
            self.assertIn("best_factors", effects, "analyze must emit best_factors block")
            self.assertEqual(effects["best_factors"], {"batch_size": 64, "retries": 5})
            self.assertEqual(effects["direction"], "lower")


class PyDOE3EquivalenceTests(unittest.TestCase):
    """If pyDOE3 happens to be installed, verify our matrices match.

    Usually skipped because agent-doe-engine's stdlib+numpy convention means pyDOE3
    isn't installed.
    """

    def test_equivalence_when_pydoe3_present(self) -> None:
        try:
            from pyDOE3 import fullfact as pd_full, fracfact as pd_frac
        except ImportError:
            self.skipTest("pyDOE3 not installed (expected - agent-doe-engine is stdlib+numpy)")

        # Convert pyDOE3's 0/1 coding to our ±1 coding for comparison
        their_full = pd_full([2, 2, 2]) * 2 - 1
        mine_full = doe.full_factorial_2level(3)
        self.assertTrue(
            np.array_equal(np.array(sorted(map(tuple, their_full))),
                           np.array(sorted(map(tuple, mine_full)))),
            "2^3 full factorial differs from pyDOE3",
        )

        their_frac = pd_frac("a b c ab ac")
        mine_frac = doe.fracfact("a b c ab ac")
        self.assertTrue(np.array_equal(their_frac, mine_frac),
                        "2^(5-2) fractional differs from pyDOE3")


# ---------------------------------------------------------------------------
# Multi-objective analyze tests (new)
# ---------------------------------------------------------------------------

class MultiObjectiveAnalyzeTests(unittest.TestCase):
    """2^2 full-factorial with two objectives; hand-computed best under scalarize."""

    # Design: 2 factors (x1, x2), 4 runs.
    # Coding: run 0=(-1,-1), 1=(-1,+1), 2=(+1,-1), 3=(+1,+1)
    #
    # Objective A (obj_a, lower): driven entirely by x1
    #   obj_a = 10 - 5*x1        (x1=+1 is worse: 5; x1=-1 is better: 15? no:
    #   wait: "lower is better" and obj_a = 10 - 5*x1 means x1=+1 gives 5 (best)
    #   Actually we want x1 to be the dominant factor for obj_a.
    #   obj_a = 10 + 5*x1  → x1=+1 gives 15 (worse), x1=-1 gives 5 (best): lower is better
    #
    # Objective B (obj_b, higher): driven entirely by x2
    #   obj_b = 10 + 5*x2  → x2=+1 gives 15 (best), x2=-1 gives 5 (worse): higher is better
    #
    # Per-objective effects:
    #   obj_a: main effect of x1 = 5 (positive), x2 = 0 → x1 ranks #1
    #   obj_b: main effect of x2 = 5 (positive), x1 = 0 → x2 ranks #1
    #
    # Scalarize (equal weights 0.5 each):
    #   Bounds: obj_a in [5,15], obj_b in [5,15]
    #   normalize obj_a (lower): (15-v)/10; normalize obj_b (higher): (v-5)/10
    #   run 0: obj_a=5, obj_b=5  → norm_a=(15-5)/10=1.0, norm_b=(5-5)/10=0.0  → score=0.5
    #   run 1: obj_a=5, obj_b=15 → norm_a=1.0, norm_b=1.0 → score=1.0  ← BEST
    #   run 2: obj_a=15, obj_b=5 → norm_a=0.0, norm_b=0.0 → score=0.0
    #   run 3: obj_a=15, obj_b=15→ norm_a=0.0, norm_b=1.0 → score=0.5
    #   Best run = 1 (x1=-1, x2=+1)

    FACTORS = [
        {"name": "x1", "low": -1, "high": 1},
        {"name": "x2", "low": -1, "high": 1},
    ]
    OBJECTIVES = [
        {"name": "obj_a", "direction": "lower",  "weight": 0.5},
        {"name": "obj_b", "direction": "higher", "weight": 0.5},
    ]

    def _make_design_and_results(self, tmp: Path) -> tuple[Path, Path]:
        """Generate a 2^2 full-factorial design and hand-built 2-obj results."""
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "generate",
             "--factors", json.dumps(self.FACTORS), "--design", "full", "--seed", "0"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        design = json.loads(r.stdout)
        design_path = tmp / "design.json"
        design_path.write_text(json.dumps(design))

        matrix = np.array(design["matrix"])  # shape (4, 2)
        results_path = tmp / "results.jsonl"
        with results_path.open("w") as f:
            for i in range(4):
                x1, x2 = matrix[i, 0], matrix[i, 1]
                obj_a = 10 + 5 * x1   # lower is better; x1=-1 gives 5
                obj_b = 10 + 5 * x2   # higher is better; x2=+1 gives 15
                f.write(json.dumps({
                    "run_id": i,
                    "values": {"obj_a": float(obj_a), "obj_b": float(obj_b)},
                    "guard_ok": True,
                }) + "\n")

        return design_path, results_path

    def test_per_objective_effects_rank_correct_factor(self) -> None:
        """x1 must be top effect for obj_a; x2 must be top effect for obj_b."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            design_path, results_path = self._make_design_and_results(tmp)
            obj_arg = json.dumps(self.OBJECTIVES)

            r = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(design_path),
                 "--results", str(results_path),
                 "--objectives", obj_arg,
                 "--selection", "scalarize"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout)

            # Structure checks
            self.assertIn("per_objective", out)
            self.assertIn("obj_a", out["per_objective"])
            self.assertIn("obj_b", out["per_objective"])
            self.assertIn("selection", out)

            # x1 is the top ranked effect for obj_a
            top_a = out["per_objective"]["obj_a"]["ranked_effects"][0]["term"]
            self.assertEqual(top_a, "x1",
                             f"Expected x1 to dominate obj_a; got {top_a}")

            # x2 is the top ranked effect for obj_b
            top_b = out["per_objective"]["obj_b"]["ranked_effects"][0]["term"]
            self.assertEqual(top_b, "x2",
                             f"Expected x2 to dominate obj_b; got {top_b}")

    def test_scalarize_picks_hand_computed_best(self) -> None:
        """best_run_id must be run 1 (x1=-1, x2=+1) under equal-weight scalarize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            design_path, results_path = self._make_design_and_results(tmp)
            obj_arg = json.dumps(self.OBJECTIVES)

            r = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(design_path),
                 "--results", str(results_path),
                 "--objectives", obj_arg,
                 "--selection", "scalarize"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout)

            # The design matrix row order from generate with seed=0 may differ from
            # the ±1 order we manually computed. Find the run_id where x1=-1 and x2=+1.
            design = json.loads(design_path.read_text())
            matrix = np.array(design["matrix"])
            expected_run_id = None
            for i, row in enumerate(matrix):
                if row[0] < 0 and row[1] > 0:
                    expected_run_id = i
                    break
            self.assertIsNotNone(expected_run_id, "Could not find x1=-1, x2=+1 row in design")

            actual_best = out["selection"]["best_run_id"]
            self.assertEqual(
                actual_best, expected_run_id,
                f"Expected best_run_id={expected_run_id} (x1=-1,x2=+1); got {actual_best}",
            )

    def test_single_objective_via_objectives_arg_matches_legacy(self) -> None:
        """Single-objective --objectives run gives the same best_run as the legacy path."""
        factors = [
            {"name": "batch_size", "low": 16, "high": 64},
            {"name": "retries", "low": 1, "high": 5},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Generate design
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "generate",
                 "--factors", json.dumps(factors), "--design", "full", "--seed", "0"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            design = json.loads(r.stdout)
            design_path = tmp / "design.json"
            design_path.write_text(json.dumps(design))
            matrix = np.array(design["matrix"])

            # y = 100 - 8*x1 - 2*x2 → lower is better → x1=+1, x2=+1 gives 90 (min)
            y = 100 - 8 * matrix[:, 0] - 2 * matrix[:, 1]

            # Legacy results file: {run_id, value}
            legacy_path = tmp / "results_legacy.jsonl"
            with legacy_path.open("w") as f:
                for i in range(len(y)):
                    f.write(json.dumps({"run_id": i, "value": float(y[i])}) + "\n")

            # Multi-obj results file: {run_id, values:{latency:...}}
            multi_path = tmp / "results_multi.jsonl"
            with multi_path.open("w") as f:
                for i in range(len(y)):
                    f.write(json.dumps({
                        "run_id": i,
                        "values": {"latency": float(y[i])},
                        "guard_ok": True,
                    }) + "\n")

            # Legacy path
            r_leg = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(design_path),
                 "--results", str(legacy_path),
                 "--direction", "lower"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r_leg.returncode, 0, r_leg.stderr)
            legacy_out = json.loads(r_leg.stdout)

            # Multi-objective path with one objective (also accept legacy lines)
            single_obj = [{"name": "latency", "direction": "lower", "weight": 1.0}]
            r_multi = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(design_path),
                 "--results", str(multi_path),
                 "--objectives", json.dumps(single_obj),
                 "--selection", "scalarize"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r_multi.returncode, 0, r_multi.stderr)
            multi_out = json.loads(r_multi.stdout)

            self.assertEqual(
                legacy_out["best_run"],
                multi_out["selection"]["best_run_id"],
                "Single-objective --objectives path must agree with legacy path on best_run",
            )

    def test_legacy_value_lines_accepted_with_single_objective(self) -> None:
        """Legacy {run_id, value} lines are accepted when exactly one objective is declared."""
        factors = [{"name": "x1", "low": 0, "high": 1}, {"name": "x2", "low": 0, "high": 1}]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "generate",
                 "--factors", json.dumps(factors), "--design", "full", "--seed", "0"],
                capture_output=True, text=True, timeout=10,
            )
            design = json.loads(r.stdout)
            design_path = tmp / "design.json"
            design_path.write_text(json.dumps(design))
            matrix = np.array(design["matrix"])
            y = 5 + 2 * matrix[:, 0]

            results_path = tmp / "results.jsonl"
            with results_path.open("w") as f:
                for i in range(len(y)):
                    f.write(json.dumps({"run_id": i, "value": float(y[i])}) + "\n")

            single_obj = [{"name": "metric", "direction": "lower", "weight": 1.0}]
            r2 = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(design_path),
                 "--results", str(results_path),
                 "--objectives", json.dumps(single_obj)],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r2.returncode, 0, r2.stderr)
            out = json.loads(r2.stdout)
            self.assertIn("selection", out)
            self.assertIn("per_objective", out)
            self.assertIn("metric", out["per_objective"])


# ---------------------------------------------------------------------------
# Inference: SE / t / p / CI, replicate pooling, trust verdicts (new)
# ---------------------------------------------------------------------------

class InferenceTests(unittest.TestCase):
    def test_saturated_design_no_error_df(self) -> None:
        # 2^2 with interactions: 4 params, 4 runs → 0 residual df, df==0 verdict.
        d = doe.full_factorial_2level(2)
        y = 10 + 5 * d[:, 0] + 2 * d[:, 1] + 0.5 * d[:, 0] * d[:, 1]
        e = doe.fit_effects(d, y, include_interactions=True)
        self.assertEqual(e["error_df"], 0)
        self.assertIn("saturated", e["inference"])
        self.assertTrue(e["warnings"])
        # p-values are NaN on a saturated fit
        self.assertTrue(math.isnan(e["main_stats"][0]["p_value"]))

    def test_low_power_warning_at_one_df(self) -> None:
        # 2^2 mains-only: 3 params, 4 runs → residual_df == 1 → low-power.
        d = doe.full_factorial_2level(2)
        y = 10 + 5 * d[:, 0] + 2 * d[:, 1]
        e = doe.fit_effects(d, y, include_interactions=False)
        self.assertEqual(e["residual_df"], 1)
        self.assertEqual(e["error_df"], 1)
        self.assertIn("low power", e["inference"])

    def test_se_ci_on_known_orthogonal_design(self) -> None:
        # 2^3 mains-only fit. Put residual SS on terms NOT in the model: each
        # of the three 2-way interaction columns carries coefficient 1, none of
        # which the mains-only model can absorb (orthogonality). Each ±1 column
        # contributes SS = sum(1^2) = 8; three of them → residual SS = 24.
        # residual_df = n - rank = 8 - 4 = 4 → error_var = 24/4 = 6.
        d = doe.full_factorial_2level(3)
        y = (10 + 4 * d[:, 0]
             + 1.0 * d[:, 0] * d[:, 1]
             + 1.0 * d[:, 0] * d[:, 2]
             + 1.0 * d[:, 1] * d[:, 2])
        e = doe.fit_effects(d, y, include_interactions=False)
        self.assertEqual(e["residual_df"], 4)
        self.assertAlmostEqual(e["error_var"], 24.0 / 4.0, places=8)  # = 6
        # Orthogonal design: SE_j = sqrt(error_var / n) = sqrt(6/8).
        se_x1 = e["main_stats"][0]["se"]
        self.assertAlmostEqual(se_x1, math.sqrt(6.0 / 8.0), places=8)
        # t for x1 = effect/SE = 4 / sqrt(0.75)
        self.assertAlmostEqual(e["main_stats"][0]["t"], 4.0 / math.sqrt(0.75), places=6)
        # CI = effect ± t_crit(.975, df=4) * SE; t_crit ≈ 2.776445
        lo, hi = e["main_stats"][0]["ci95"]
        self.assertAlmostEqual(lo, 4.0 - 2.776445 * se_x1, places=4)
        self.assertAlmostEqual(hi, 4.0 + 2.776445 * se_x1, places=4)

    def test_replicate_pooling_uses_pure_error(self) -> None:
        # 2^2 with 3 replicates/cell and a TRUE noise sigma → pure error df = 8.
        d = doe.full_factorial_2level(2)
        rng = np.random.default_rng(99)
        cells = []
        for i in range(4):
            mean = 10 + 5 * d[i, 0] + 2 * d[i, 1]
            cells.append([float(mean + rng.normal(0, 0.5)) for _ in range(3)])
        ymeans = np.array([np.mean(c) for c in cells])
        e = doe.fit_effects(d, ymeans, include_interactions=True, cell_values=cells)
        self.assertEqual(e["pure_error_df"], 8)
        self.assertEqual(e["error_source"], "pure_error")
        self.assertEqual(e["inference"], "ok")
        # x1 (true 5) and x2 (true 2) significant; interaction (true 0) not.
        self.assertTrue(e["main_stats"][0]["significant"])
        self.assertTrue(e["main_stats"][1]["significant"])
        self.assertFalse(e["inter_stats"][(0, 1)]["significant"])

    def test_replicate_se_matches_observation_level_ols(self) -> None:
        # REGRESSION (auditor nay 2026-06-10): replicated SEs must equal the
        # observation-level OLS SE, NOT the √r-inflated cell-level SE.
        d = doe.full_factorial_2level(2)
        rng = np.random.default_rng(1)
        r = 3
        cells = [[float(10 + 5 * d[i, 0] + 2 * d[i, 1] + rng.normal(0, 1.0))
                  for _ in range(r)] for i in range(4)]
        ymeans = np.array([np.mean(c) for c in cells])
        e = doe.fit_effects(d, ymeans, include_interactions=True, cell_values=cells)

        # Ground truth: expand to observation level and run full OLS by hand.
        rows, yobs = [], []
        for i in range(4):
            for v in cells[i]:
                rows.append([1, d[i, 0], d[i, 1], d[i, 0] * d[i, 1]])
                yobs.append(v)
        X = np.array(rows, dtype=float)
        yo = np.array(yobs)
        beta, _, _, _ = np.linalg.lstsq(X, yo, rcond=None)
        resid = yo - X @ beta
        s2 = float(resid @ resid) / (len(yo) - X.shape[1])
        se_true = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
        self.assertAlmostEqual(e["main_stats"][0]["se"], se_true[1], places=10)
        self.assertAlmostEqual(e["main_stats"][1]["se"], se_true[2], places=10)

    def test_unbalanced_replicates_se_correct(self) -> None:
        # Unequal replicate counts: SE = sqrt(pure_error_var · (XᵀX)⁻¹_jj) on
        # the OBSERVATION-level matrix. Point estimates also weight by replicate
        # count (observation-level OLS), not the unweighted cell means.
        d = doe.full_factorial_2level(2)
        cells = [[10.0, 12.0, 11.0], [20.0, 22.0], [5.0, 7.0, 6.0, 8.0], [30.0]]
        ymeans = np.array([np.mean(c) for c in cells])
        e = doe.fit_effects(d, ymeans, include_interactions=False, cell_values=cells)

        # Build the observation-level matrix and the pooled pure-error variance.
        rows, yobs = [], []
        for i in range(4):
            for v in cells[i]:
                rows.append([1, d[i, 0], d[i, 1]])
                yobs.append(v)
        X = np.array(rows, dtype=float)
        yo = np.array(yobs)
        # Point estimates: observation-level OLS.
        beta_true, _, _, _ = np.linalg.lstsq(X, yo, rcond=None)
        self.assertAlmostEqual(e["main"][0], beta_true[1], places=10)
        self.assertAlmostEqual(e["main"][1], beta_true[2], places=10)
        # SEs: pooled pure-error variance on the observation-level info matrix.
        pe_var, pe_df = doe_stats.pooled_pure_error(cells)
        self.assertEqual(e["error_df"], pe_df)
        self.assertEqual(e["error_source"], "pure_error")
        se_true = np.sqrt(pe_var * np.diag(np.linalg.inv(X.T @ X)))
        for j in range(1, 3):
            self.assertAlmostEqual(e["main_stats"][j - 1]["se"], se_true[j], places=10)

    def test_pooled_variance_matches_hand_value(self) -> None:
        # Pure error must equal the hand-pooled within-cell variance.
        d = doe.full_factorial_2level(2)
        cells = [
            [9.0, 11.0],     # mean 10, SS 2, df 1
            [14.0, 16.0],    # mean 15, SS 2, df 1
            [4.0, 6.0],      # mean 5,  SS 2, df 1
            [19.0, 21.0],    # mean 20, SS 2, df 1
        ]
        ymeans = np.array([np.mean(c) for c in cells])
        e = doe.fit_effects(d, ymeans, include_interactions=True, cell_values=cells)
        # pooled SS = 8, pooled df = 4 → pooled var = 2
        self.assertEqual(e["pure_error_df"], 4)
        self.assertAlmostEqual(e["error_var"], 2.0, places=10)

    def test_backward_compatible_call_still_works(self) -> None:
        d = doe.full_factorial_2level(3)
        y = 10 + 5 * d[:, 0]
        e = doe.fit_effects(d, y)  # legacy 2-arg call
        self.assertIn("main", e)
        self.assertIn("inference", e)  # now also carries inference


# ---------------------------------------------------------------------------
# Alias / resolution structure (new)
# ---------------------------------------------------------------------------

class AliasStructureTests(unittest.TestCase):
    def test_full_factorial_no_aliasing(self) -> None:
        d = doe.full_factorial_2level(3)
        a = doe.alias_structure(d, ["x1", "x2", "x3"])
        self.assertFalse(a["aliasing"])
        self.assertEqual(a["resolution"], "Full")
        self.assertIn("no aliasing", a["note"])
        self.assertEqual(a["alias_chains"], [])

    def test_res3_2_5_2_defining_relation_and_chains(self) -> None:
        # 2^(5-2): a b c ab ac → D=AB, E=AC → I = ABD = ACE = BCDE, Res III.
        d = doe.fracfact("a b c ab ac")
        a = doe.alias_structure(d, ["A", "B", "C", "D", "E"])
        self.assertEqual(a["resolution"], "III")
        self.assertEqual(a["resolution_int"], 3)
        rel = a["defining_relation"][0]
        self.assertIn("A·B·D", rel)
        self.assertIn("A·C·E", rel)
        self.assertIn("B·C·D·E", rel)
        # The chain containing main A must include its 2-way aliases BD and CE.
        a_chain = next(c for c in a["alias_chains"] if c[0] == "A")
        self.assertIn("B·D", a_chain)
        self.assertIn("C·E", a_chain)

    def test_res4_2_4_1(self) -> None:
        # 2^(4-1): a b c abc → D=ABC → I = ABCD, Res IV.
        d = doe.fracfact("a b c abc")
        a = doe.alias_structure(d, ["A", "B", "C", "D"])
        self.assertEqual(a["resolution"], "IV")
        self.assertIn("A·B·C·D", a["defining_relation"][0])
        # Mains are clear; 2-ways are aliased in pairs.
        for chain in a["alias_chains"]:
            self.assertTrue(all(len(term.split("·")) == 2 for term in chain))

    def test_pb12_non_regular(self) -> None:
        d, _ = doe.build_design(8, "pb")
        a = doe.alias_structure(d)
        self.assertTrue(a["aliasing"])
        self.assertIn("Non-regular", a["note"])


# ---------------------------------------------------------------------------
# Regression of the user's own under-powered study (the trust-signal proof)
# ---------------------------------------------------------------------------

class UserStudyRegressionTests(unittest.TestCase):
    """The 8-run/3-factor study that previously reported a bare r²≈0.94 must now
    surface residual_df==1 and a LOW-POWER warning - proving the trust signals
    stop over-trusting a high r² on an under-powered design."""

    def test_study_reports_low_power(self) -> None:
        factors = [
            {"name": "tier", "low": "sonnet", "high": "opus"},
            {"name": "plan_first", "low": False, "high": True},
            {"name": "self_verify", "low": False, "high": True},
        ]
        values = [20, 20, 19, 17, 20, 20, 20, 20]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "generate",
                 "--factors", json.dumps(factors), "--design", "full", "--seed", "0"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            (tmp / "design.json").write_text(r.stdout)
            with (tmp / "results.jsonl").open("w") as f:
                for i, v in enumerate(values):
                    f.write(json.dumps({"run_id": i, "value": v}) + "\n")
            r2 = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(tmp / "design.json"),
                 "--results", str(tmp / "results.jsonl"),
                 "--direction", "higher"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r2.returncode, 0, r2.stderr)
            out = json.loads(r2.stdout)
            self.assertEqual(out["summary"]["residual_df"], 1)
            self.assertEqual(out["summary"]["error_source"], "residual")
            self.assertIn("low power", out["summary"]["inference"])
            self.assertTrue(any("Low power" in w for w in out["warnings"]))
            # The previously-trusted r² is still reported but now flanked by the
            # trust signal; the top effect must NOT be flagged significant.
            self.assertIsNotNone(out["summary"]["r2"])
            top = out["ranked_effects"][0]
            self.assertFalse(top["significant"])


class ReplicateCliTests(unittest.TestCase):
    def test_replicated_rows_pool_to_pure_error(self) -> None:
        factors = [{"name": "x1", "low": -1, "high": 1},
                   {"name": "x2", "low": -1, "high": 1}]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "generate",
                 "--factors", json.dumps(factors), "--design", "full", "--seed", "0"],
                capture_output=True, text=True, timeout=10,
            )
            (tmp / "design.json").write_text(r.stdout)
            matrix = np.array(json.loads(r.stdout)["matrix"])
            rng = np.random.default_rng(5)
            with (tmp / "results.jsonl").open("w") as f:
                for i in range(4):
                    mean = 10 + 5 * matrix[i, 0] + 2 * matrix[i, 1]
                    for _ in range(3):  # 3 replicates per cell
                        f.write(json.dumps({
                            "run_id": i, "value": float(mean + rng.normal(0, 0.4))
                        }) + "\n")
            r2 = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(tmp / "design.json"),
                 "--results", str(tmp / "results.jsonl"),
                 "--direction", "higher"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r2.returncode, 0, r2.stderr)
            out = json.loads(r2.stdout)
            self.assertEqual(out["summary"]["n_observations"], 12)
            self.assertEqual(out["summary"]["n_replicated_cells"], 4)
            self.assertEqual(out["summary"]["error_source"], "pure_error")
            self.assertEqual(out["summary"]["pure_error_df"], 8)
            self.assertEqual(out["summary"]["inference"], "ok")

    def test_non_finite_value_rejected(self) -> None:
        # REGRESSION (auditor 2026-06-10): NaN/Infinity in results must fail
        # loudly, not silently contaminate the means/inference.
        factors = [{"name": "x1", "low": 0, "high": 1},
                   {"name": "x2", "low": 0, "high": 1}]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "generate",
                 "--factors", json.dumps(factors), "--design", "full", "--seed", "0"],
                capture_output=True, text=True, timeout=10,
            )
            (tmp / "design.json").write_text(r.stdout)
            # Write a results file with a NaN (json emits NaN literally).
            with (tmp / "results.jsonl").open("w") as f:
                f.write(json.dumps({"run_id": 0, "value": 1.0}) + "\n")
                f.write('{"run_id": 1, "value": NaN}\n')
                f.write(json.dumps({"run_id": 2, "value": 3.0}) + "\n")
                f.write(json.dumps({"run_id": 3, "value": 4.0}) + "\n")
            r2 = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(tmp / "design.json"),
                 "--results", str(tmp / "results.jsonl")],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r2.returncode, 2)
            self.assertIn("non-finite", r2.stderr)

    def test_analyze_emits_aliasing_block(self) -> None:
        # Fractional design → analyze output must carry the alias structure.
        factors = [{"name": f"f{i}", "low": 0, "high": 1} for i in range(5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "generate",
                 "--factors", json.dumps(factors), "--design", "fractional", "--seed", "0"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            design = json.loads(r.stdout)
            self.assertIn("aliasing", design)  # generate emits it too
            self.assertEqual(design["aliasing"]["resolution"], "III")
            (tmp / "design.json").write_text(r.stdout)
            matrix = np.array(design["matrix"])
            y = 50 + 4 * matrix[:, 0]
            with (tmp / "results.jsonl").open("w") as f:
                for i in range(len(y)):
                    f.write(json.dumps({"run_id": i, "value": float(y[i])}) + "\n")
            r2 = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze",
                 "--design", str(tmp / "design.json"),
                 "--results", str(tmp / "results.jsonl")],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(r2.returncode, 0, r2.stderr)
            out = json.loads(r2.stdout)
            self.assertIn("aliasing", out)
            self.assertEqual(out["aliasing"]["resolution"], "III")
            self.assertTrue(out["aliasing"]["alias_chains"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class DegenerateFitTests(unittest.TestCase):
    """A residual at floating-point zero is not an error estimate.

    Regression guard for a real defect: a 5-factor / 8-run fit whose response
    the model reproduces exactly reported standard errors around 1e-16 and
    p-values around 1e-31, flagged `significant: true`. Three of those
    "significant" terms were one aliased estimate wearing three names.
    """

    def _perfect_five_factor(self):
        d = doe.fracfact(doe.FRACFACT_8_RUN[5])
        # Response is an exact linear function of the columns: zero residual.
        y = 0.65 + 0.0875 * d[:, 0] + 0.0875 * d[:, 1] - 0.0875 * d[:, 2]
        return d, y

    def test_degenerate_fit_is_flagged(self) -> None:
        d, y = self._perfect_five_factor()
        e = doe.fit_effects(d, y, include_interactions=False)
        self.assertTrue(e["degenerate_fit"])

    def test_degenerate_fit_withholds_significance(self) -> None:
        d, y = self._perfect_five_factor()
        e = doe.fit_effects(d, y, include_interactions=False)
        for name, st in e["main_stats"].items():
            self.assertIsNone(st["significant"],
                              f"{name} claimed significance off a zero residual")
            self.assertTrue(math.isnan(st["p_value"]), f"{name} reported a p-value")

    def test_degenerate_fit_still_reports_effect_sizes(self) -> None:
        # Withholding inference must not withhold the measurement itself.
        d, y = self._perfect_five_factor()
        e = doe.fit_effects(d, y, include_interactions=False)
        self.assertTrue(any(abs(v) > 1e-6 for v in e["main"].values()))

    def test_real_residual_still_gets_pvalues(self) -> None:
        # The guard must not fire on a genuine, small-but-real error term.
        d = doe.full_factorial_2level(3)
        y = (10 + 4 * d[:, 0]
             + 1.0 * d[:, 0] * d[:, 1]
             + 1.0 * d[:, 0] * d[:, 2]
             + 1.0 * d[:, 1] * d[:, 2])
        e = doe.fit_effects(d, y, include_interactions=False)
        self.assertFalse(e["degenerate_fit"])
        self.assertFalse(math.isnan(e["main_stats"][0]["p_value"]))


class AliasedWithTests(unittest.TestCase):
    """Each ranked row must carry its own alias set.

    A consumer filtering ranked_effects for `significant` decides what to ship
    from that row. In a resolution III design several rows are the same
    estimate, and the alias block lives in a different part of the output.
    """

    def test_ranked_rows_carry_alias_partners(self) -> None:
        d = doe.fracfact(doe.FRACFACT_8_RUN[5])
        names = ["ef_search", "iterative_scan", "work_mem", "parallel", "jit"]
        y = 1.0 + 0.5 * d[:, 0] + np.array([0.01, -0.02, 0.03, -0.01,
                                            0.02, -0.03, 0.01, -0.01])
        e = doe.fit_effects(d, y, include_interactions=False)
        chains = doe.alias_structure(d, factor_names=names)["alias_chains"]
        rows = doe.rank_findings(e, names, alias_chains=chains)
        by_term = {r["term"]: r for r in rows}
        # This design is resolution III: every main effect is aliased.
        for n in names:
            self.assertIn("aliased_with", by_term[n])
            self.assertTrue(by_term[n]["aliased_with"],
                            f"{n} reported no alias partners in a resolution III design")

    def test_alias_field_absent_when_chains_not_supplied(self) -> None:
        d = doe.full_factorial_2level(2)
        y = 10 + 5 * d[:, 0] + 2 * d[:, 1]
        e = doe.fit_effects(d, y, include_interactions=False)
        rows = doe.rank_findings(e, ["a", "b"])
        self.assertIsNone(rows[0]["aliased_with"])
