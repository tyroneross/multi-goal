# SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0
"""Goal contract, feasibility, next-step rules, and the confirmation gate.

These are the behaviours a Fable-tier review found missing (2026-09-05):
absolute bars (Derringer & Suich 1980), guardrails as constraints (Spotify's
success / guardrail split), a measurement-driven next step (sequential
experimentation), and an explicit done criterion (Jensen 2016 confirmation
runs judged against a prediction interval).
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    import pytest
    pytest.skip("numpy not installed", allow_module_level=True)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import doe  # noqa: E402
import objectives  # noqa: E402

SCRIPT = SCRIPTS_DIR / "doe.py"


# ---------------------------------------------------------------------------
# objectives.py: contract validation, absolute desirability, feasibility
# ---------------------------------------------------------------------------

class ValidateObjectivesTests(unittest.TestCase):
    def test_inverted_bar_is_an_error(self):
        out = objectives.validate_objectives(
            [{"name": "lat", "direction": "lower", "min_acceptable": 50, "target": 100}])
        self.assertTrue(any("better than target" in e for e in out["errors"]))

    def test_bad_role_and_direction_are_errors(self):
        out = objectives.validate_objectives(
            [{"name": "x", "direction": "sideways", "role": "vibe"}])
        self.assertEqual(len(out["errors"]), 2)

    def test_incomplete_contract_warns_not_errors(self):
        out = objectives.validate_objectives(
            [{"name": "lat", "direction": "lower"},
             {"name": "acc", "direction": "higher", "role": "guardrail"}])
        self.assertEqual(out["errors"], [])
        joined = " ".join(out["warnings"])
        self.assertIn("no 'driver'", joined)
        self.assertIn("neither 'target' nor 'min_effect'", joined)
        self.assertIn("constrains nothing", joined)

    def test_no_primary_warns(self):
        out = objectives.validate_objectives(
            [{"name": "acc", "direction": "higher", "role": "guardrail", "baseline": 0.9}])
        self.assertTrue(any("No 'primary'" in w for w in out["warnings"]))

    def test_legacy_objective_is_valid(self):
        out = objectives.validate_objectives(
            [{"name": "lat", "direction": "lower", "weight": 1.0, "driver": "page load",
              "target": 80}])
        self.assertEqual(out, {"errors": [], "warnings": []})


class AbsoluteDesirabilityTests(unittest.TestCase):
    def test_absolute_limits_replace_batch_minmax(self):
        obj = {"name": "lat", "direction": "lower", "min_acceptable": 100, "target": 50}
        bound = {"min": 70, "max": 90}   # batch bounds must not matter
        self.assertEqual(objectives.desirability_value(50, obj, bound), 1.0)
        self.assertEqual(objectives.desirability_value(100, obj, bound), 0.0)
        self.assertAlmostEqual(objectives.desirability_value(75, obj, bound), 0.5)

    def test_worst_run_is_not_zero_when_it_clears_the_bar(self):
        # The README claim: desirability makes every goal clear a MINIMUM BAR.
        # With absolute limits the batch's worst run keeps its desirability.
        objs = [{"name": "lat", "direction": "lower", "min_acceptable": 100, "target": 50}]
        runs = [{"run_id": 0, "values": {"lat": 60}}, {"run_id": 1, "values": {"lat": 70}}]
        bounds = objectives.compute_bounds(runs, objs)
        d_worst = objectives.desirability_run(runs[1]["values"], objs, bounds)
        self.assertGreater(d_worst, 0.0)
        self.assertAlmostEqual(d_worst, 0.6)

    def test_without_limits_batch_minmax_still_applies(self):
        objs = [{"name": "lat", "direction": "lower"}]
        runs = [{"run_id": 0, "values": {"lat": 60}}, {"run_id": 1, "values": {"lat": 70}}]
        bounds = objectives.compute_bounds(runs, objs)
        self.assertEqual(objectives.desirability_run(runs[1]["values"], objs, bounds), 0.0)


class FeasibilityTests(unittest.TestCase):
    OBJS = [
        {"name": "lat", "direction": "lower", "role": "primary"},
        {"name": "acc", "direction": "higher", "role": "guardrail", "min_acceptable": 0.9},
        {"name": "cost", "direction": "lower", "role": "guardrail", "baseline": 5.0},
    ]

    def test_guardrail_bar_and_non_inferiority(self):
        ok, violated = objectives.is_feasible({"lat": 10, "acc": 0.95, "cost": 5.0}, self.OBJS)
        self.assertTrue(ok); self.assertEqual(violated, [])
        ok, violated = objectives.is_feasible({"lat": 1, "acc": 0.85, "cost": 5.1}, self.OBJS)
        self.assertFalse(ok); self.assertEqual(violated, ["acc", "cost"])

    def test_primary_never_constrains(self):
        ok, _ = objectives.is_feasible({"lat": 10**9, "acc": 0.99, "cost": 1}, self.OBJS)
        self.assertTrue(ok)

    def test_select_best_skips_infeasible_runs(self):
        runs = [
            {"run_id": 0, "values": {"lat": 1, "acc": 0.80, "cost": 1}},   # fastest, breaks acc
            {"run_id": 1, "values": {"lat": 5, "acc": 0.95, "cost": 4}},
            {"run_id": 2, "values": {"lat": 3, "acc": 0.92, "cost": 6}},   # breaks cost baseline
        ]
        for method in ("scalarize", "desirability", "pareto"):
            res = objectives.select_best(runs, self.OBJS, method)
            self.assertEqual(res["best_run_id"], 1, method)
            self.assertEqual(res["feasible_run_ids"], [1])
            self.assertEqual(res["infeasible"], {0: ["acc"], 2: ["cost"]})

    def test_select_best_with_no_feasible_run(self):
        runs = [{"run_id": 0, "values": {"lat": 1, "acc": 0.5, "cost": 1}}]
        res = objectives.select_best(runs, self.OBJS, "scalarize")
        self.assertIsNone(res["best_run_id"])
        self.assertIn("No run satisfies every guardrail", res["reason"])


# ---------------------------------------------------------------------------
# doe.py: practical significance + next_step rules
# ---------------------------------------------------------------------------

def _full_2x2():
    return doe.full_factorial_2level(2)


def _rows(effects, names, min_effect=None):
    return doe.annotate_practical(doe.rank_findings(effects, names), min_effect)


class PracticalSignificanceTests(unittest.TestCase):
    def test_low_to_high_change_is_twice_the_coefficient(self):
        d = _full_2x2()
        y = 100 - 10 * d[:, 0] + 1 * d[:, 1]
        eff = doe.fit_effects(d, y, include_interactions=False)
        rows = _rows(eff, ["a", "b"], min_effect=5)
        by = {r["term"]: r for r in rows}
        self.assertAlmostEqual(by["a"]["low_to_high_change"], -20.0)
        self.assertTrue(by["a"]["practically_significant"])
        self.assertFalse(by["b"]["practically_significant"])

    def test_none_when_no_threshold(self):
        d = _full_2x2()
        eff = doe.fit_effects(d, 100 - 10 * d[:, 0], include_interactions=False)
        self.assertIsNone(_rows(eff, ["a", "b"])[0]["practically_significant"])


class NextStepTests(unittest.TestCase):
    def _replicated(self, fn, reps=3, seed=1):
        d = _full_2x2()
        rng = random.Random(seed)
        cells = [[fn(r[0], r[1]) + rng.gauss(0, 0.5) for _ in range(reps)] for r in d]
        y = np.array([np.mean(c) for c in cells])
        eff = doe.fit_effects(d, y, include_interactions=True, cell_values=cells)
        return d, eff

    def test_saturated_design_asks_for_replicates_then_confirm(self):
        d = _full_2x2()
        eff = doe.fit_effects(d, 100 - 10 * d[:, 0], include_interactions=False)
        rows = _rows(eff, ["a", "b"], min_effect=5)
        steps = doe.next_step(eff, rows, d, ["a", "b"], best_run_idx=int(np.argmin(100 - 10 * d[:, 0])),
                              min_effect=5)
        actions = [s["action"] for s in steps]
        self.assertEqual(actions[0], "add_replicates")
        self.assertIn("confirm", actions)

    def test_real_effect_yields_confirm_and_extend_range(self):
        d, eff = self._replicated(lambda a, b: 100 - 10 * a)
        rows = _rows(eff, ["a", "b"])
        best = int(np.argmin([100 - 10 * r[0] for r in d]))
        steps = doe.next_step(eff, rows, d, ["a", "b"], best)
        actions = [s["action"] for s in steps]
        self.assertIn("confirm", actions)
        ext = [s for s in steps if s["action"] == "extend_range"]
        self.assertEqual(ext[0]["terms"], ["a"])
        self.assertIn("high level", ext[0]["reason"])
        self.assertNotIn("add_replicates", actions)
        self.assertNotIn("stop_or_widen", actions)

    def test_no_effect_yields_stop_or_widen(self):
        d = _full_2x2()
        cells = [[99.5, 100.5, 100.0] for _ in d]   # pure error, effects exactly zero
        eff = doe.fit_effects(d, np.array([100.0] * 4), include_interactions=True,
                              cell_values=cells)
        rows = _rows(eff, ["a", "b"], min_effect=5)
        steps = doe.next_step(eff, rows, d, ["a", "b"], 0, min_effect=5)
        self.assertEqual([s["action"] for s in steps], ["stop_or_widen"])
        self.assertIn("min_effect=5", steps[0]["reason"])

    def test_aliased_significant_effect_yields_decouple_first(self):
        d = doe.fracfact(doe.FRACFACT_8_RUN[5])
        names = [f"f{i}" for i in range(5)]
        rng = random.Random(4)
        cells = [[50 + 4 * r[0] + rng.gauss(0, 0.3) for _ in range(2)] for r in d]
        y = np.array([np.mean(c) for c in cells])
        eff = doe.fit_effects(d, y, include_interactions=False, cell_values=cells)
        aliasing = doe.alias_structure(d, names)
        rows = doe.annotate_practical(
            doe.rank_findings(eff, names, alias_chains=aliasing["alias_chains"]), None)
        steps = doe.next_step(eff, rows, d, names, int(np.argmin(y)))
        self.assertEqual(steps[0]["action"], "decouple")
        self.assertIn("f0", steps[0]["terms"])

    def test_merge_keeps_stop_only_when_unanimous(self):
        merged = doe.merge_next_steps({
            "lat": [{"action": "confirm", "terms": [], "reason": "r"}],
            "acc": [{"action": "stop_or_widen", "terms": [], "reason": "r"}],
        })
        self.assertEqual([m["action"] for m in merged], ["confirm"])
        merged = doe.merge_next_steps({
            "lat": [{"action": "stop_or_widen", "terms": [], "reason": "r"}],
            "acc": [{"action": "stop_or_widen", "terms": [], "reason": "r"}],
        })
        self.assertEqual(merged[0]["objectives"], ["lat", "acc"])


# ---------------------------------------------------------------------------
# doe.py confirm: the done gate, end to end through the CLI
# ---------------------------------------------------------------------------

class ConfirmCliTests(unittest.TestCase):
    def _setup(self, tmp: Path, guardrail_bar: float, reps: int = 2, seed: int = 3):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "generate", "--factors",
             json.dumps([{"name": "a", "low": 1, "high": 3}, {"name": "b", "low": 10, "high": 20}]),
             "--design", "full", "--seed", "1"], capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        (tmp / "d.json").write_text(r.stdout)
        design = json.loads(r.stdout)
        rng = random.Random(seed)
        with (tmp / "r.jsonl").open("w") as f:
            for i, (a, b) in enumerate(design["matrix"]):
                for _ in range(reps):
                    f.write(json.dumps({"run_id": i, "values": {
                        "lat": 100 - 20 * a + 2 * b + rng.gauss(0, 1.5),
                        "acc": 0.90 + 0.01 * b + rng.gauss(0, 0.003)}}) + "\n")
        (tmp / "o.json").write_text(json.dumps({"objectives": [
            {"name": "lat", "direction": "lower", "role": "primary", "driver": "page load",
             "baseline": 118, "target": 85, "min_effect": 5},
            {"name": "acc", "direction": "higher", "role": "guardrail",
             "min_acceptable": guardrail_bar}], "selection": "desirability"}))
        return design

    def _confirm(self, tmp: Path, conf_rows):
        (tmp / "c.jsonl").write_text("\n".join(json.dumps({"values": v}) for v in conf_rows))
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "confirm", "--design", str(tmp / "d.json"),
             "--results", str(tmp / "r.jsonl"), "--objectives", str(tmp / "o.json"),
             "--confirmation", str(tmp / "c.jsonl")], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_done_when_confirmed_and_bars_hold(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._setup(tmp, guardrail_bar=0.88)
            # best for lat: a=+1, b=-1 -> lat ~78, acc ~0.89
            out = self._confirm(tmp, [{"lat": 78 + d, "acc": 0.89} for d in (-1, 0, 1, 0.5, -0.5)])
            self.assertTrue(out["done"], out)
            self.assertEqual(out["recommendation"], "ship")
            lat = next(c for c in out["criteria"] if c["name"] == "lat")
            self.assertEqual(lat["pi_source"], "pure_error")
            self.assertTrue(lat["mean_in_pi"])
            self.assertTrue(lat["pass"])

    def test_guardrail_break_blocks_done(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._setup(tmp, guardrail_bar=0.88)
            out = self._confirm(tmp, [{"lat": 78, "acc": 0.85}] * 5)
            self.assertFalse(out["done"])
            self.assertEqual(out["recommendation"], "re_plan")
            acc = next(c for c in out["criteria"] if c["name"] == "acc")
            self.assertFalse(acc["pass"])
            self.assertIn("breaks", acc["why"])

    def test_model_disagreement_blocks_done(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._setup(tmp, guardrail_bar=0.88)
            # meets the target but far from the prediction: model not confirmed
            out = self._confirm(tmp, [{"lat": 40, "acc": 0.89}] * 5)
            self.assertFalse(out["done"])
            lat = next(c for c in out["criteria"] if c["name"] == "lat")
            self.assertFalse(lat["mean_in_pi"])
            self.assertIn("outside the prediction interval", lat["why"])

    def test_too_few_runs_is_provisional(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._setup(tmp, guardrail_bar=0.88)
            out = self._confirm(tmp, [{"lat": 78, "acc": 0.89}] * 2)
            self.assertFalse(out["done"])
            self.assertEqual(out["recommendation"], "more_confirmation_runs")

    def test_no_feasible_run_replans(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._setup(tmp, guardrail_bar=0.99)  # nothing reaches 0.99
            out = self._confirm(tmp, [{"lat": 78, "acc": 0.89}] * 5)
            self.assertFalse(out["done"])
            self.assertIsNone(out["best_run"])
            self.assertIn("No run satisfies", out["reason"])

    def test_saturated_single_metric_falls_back_to_confirmation_sd(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            design = self._setup(tmp, guardrail_bar=0.88, reps=1)
            with (tmp / "r1.jsonl").open("w") as f:
                for i, (a, b) in enumerate(design["matrix"]):
                    f.write(json.dumps({"run_id": i, "value": 100 - 20 * a + 2 * b}) + "\n")
            (tmp / "c.jsonl").write_text("\n".join(json.dumps({"value": v}) for v in (77, 79, 78)))
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "confirm", "--design", str(tmp / "d.json"),
                 "--results", str(tmp / "r1.jsonl"), "--confirmation", str(tmp / "c.jsonl"),
                 "--target", "85"], capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout)
            self.assertEqual(out["criteria"][0]["pi_source"], "confirmation_sd")
            self.assertTrue(out["done"])
            self.assertTrue(any("weaker" in w for w in out["warnings"]))

    def test_analyze_carries_contract_and_next_step(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._setup(tmp, guardrail_bar=0.88)
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "analyze", "--design", str(tmp / "d.json"),
                 "--results", str(tmp / "r.jsonl"), "--objectives", str(tmp / "o.json")],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout)
            self.assertEqual(out["objectives_contract"]["errors"], [])
            self.assertIn("next_step", out)
            self.assertIn("confirm", [s["action"] for s in out["next_step"]])
            lat_rows = out["per_objective"]["lat"]["ranked_effects"]
            self.assertIn("practically_significant", lat_rows[0])
            self.assertIn("feasible_run_ids", out["selection"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
