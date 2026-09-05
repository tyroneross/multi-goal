# SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0
"""The factor scanner must see object-literal knobs and rank by objective.

Dogfood evidence (Atomize AI, 2026-09-05): the scanner ranked PORT,
TOAST_REMOVE_DELAY and cron batch sizes "high" for a search-latency goal while
every real search knob sat inside `export const SEARCH_THRESHOLDS = {...}` and
was never listed - 0 of 20 candidates were usable.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import suggest_factors as sf  # noqa: E402

SCRIPT = SCRIPTS_DIR / "suggest_factors.py"

TS_FIXTURE = """\
export const SEARCH_THRESHOLDS = {
  MIN_RESULTS_FOR_RERANKING: 10, // Minimum results to trigger reranking
  GROQ_SCORE_WEIGHT: 0.7,
  MIN_RELEVANCE_SCORE: 0.3,
  LABEL: 'not a number',
};
export const CACHE_TTL_S = 60;
"""

PY_FIXTURE = """\
SEARCH_CONFIG = {
    "batch_size": 32,
    "timeout_s": 5.5,
}
"""


class ObjectLiteralScanTests(unittest.TestCase):
    def test_ts_object_fields_are_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "search-thresholds.ts"
            f.write_text(TS_FIXTURE)
            names = {c.name: c for c in sf.scan_file(f)}
            self.assertIn("SEARCH_THRESHOLDS.MIN_RESULTS_FOR_RERANKING", names)
            self.assertIn("SEARCH_THRESHOLDS.GROQ_SCORE_WEIGHT", names)
            self.assertEqual(names["SEARCH_THRESHOLDS.MIN_RELEVANCE_SCORE"].current_value, 0.3)
            self.assertEqual(names["SEARCH_THRESHOLDS.MIN_RESULTS_FOR_RERANKING"].confidence, "high")
            self.assertIn("CACHE_TTL_S", names)  # top-level constants still found
            self.assertNotIn("SEARCH_THRESHOLDS.LABEL", names)

    def test_python_dict_fields_are_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "config.py"
            f.write_text(PY_FIXTURE)
            names = {c.name for c in sf.scan_file(f)}
            self.assertIn("SEARCH_CONFIG.batch_size", names)
            self.assertIn("SEARCH_CONFIG.timeout_s", names)


class HintRankingTests(unittest.TestCase):
    def _repo(self, td: str) -> Path:
        root = Path(td)
        (root / "lib" / "search").mkdir(parents=True)
        (root / "lib" / "search" / "thresholds.ts").write_text(TS_FIXTURE)
        (root / "server.ts").write_text("export const PORT = 5000;\nexport const TOAST_REMOVE_DELAY = 1000000;\n")
        return root

    def test_hint_ranks_search_knobs_above_port(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            r = subprocess.run([sys.executable, str(SCRIPT), "--workdir", str(root), "--top", "3",
                                "--json", "--hint", "search latency"],
                               capture_output=True, text=True, timeout=20)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = json.loads(r.stdout)
            self.assertTrue(all(row["name"].startswith("SEARCH_THRESHOLDS.")
                                or row["name"] == "CACHE_TTL_S" for row in rows), rows)
            self.assertTrue(rows[0]["name"].startswith("SEARCH_THRESHOLDS."))
            self.assertGreater(rows[0]["hint_matches"], 0)

    def test_paths_restricts_the_walk(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            r = subprocess.run([sys.executable, str(SCRIPT), "--workdir", str(root), "--top", "10",
                                "--json", "--paths", "lib/search/**"],
                               capture_output=True, text=True, timeout=20)
            rows = json.loads(r.stdout)
            self.assertTrue(rows)
            self.assertFalse(any(row["name"] in ("PORT", "TOAST_REMOVE_DELAY") for row in rows))

    def test_without_hint_behaviour_is_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            r = subprocess.run([sys.executable, str(SCRIPT), "--workdir", str(root), "--top", "10", "--json"],
                               capture_output=True, text=True, timeout=20)
            rows = json.loads(r.stdout)
            self.assertTrue(all(row["hint_matches"] == 0 for row in rows))
            self.assertIn("PORT", {row["name"] for row in rows})


if __name__ == "__main__":
    unittest.main(verbosity=2)
