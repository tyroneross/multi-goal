<!-- SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# agent-doe-engine - Planning Phase + Factor Adjustability + Worktree Separation + Research Seam

Status: local branch `feat/planning-phase` in worktree `~/dev/git-folder/agent-doe-engine-planning`. **Not pushed. Not merged.** For review.

## What this adds

Four changes, composed:

1. **Phase 0 (PLAN)** - explicit new phase in `skills/agent-doe-engine/SKILL.md` BEFORE SETUP. Walks scan → host-LLM pick → validate → optional research → write factors.json. Phase 1 is now SETUP-of-objectives only; factors come pre-validated from Phase 0.
2. **`scripts/validate_factors.py`** (new, stdlib-only, SPDX-headed, 13 tests). Snapshot → mutate → re-read → revert → verify byte equality on each candidate's primary definition site. Classifies as `adjustable` or `not_adjustable` with one of six reasons (`ok`, `dead_constant`, `duplicate_definition`, `mutation_failed`, `revert_failed`, `no_definition_site`). `--reject-non-adjustable` exits 1 for planner/CI use.
3. **`scripts/worktree.py`** (new, stdlib-only, SPDX-headed, 16 tests). First-class git-worktree separation: every agent-doe-engine run lives in `<repo>-agent-doe-engine-<slug>` on branch `agent-doe-engine/<slug>`. Subcommands `init` / `info` / `cleanup [--delete-branch]`. Idempotent init, symlink-aware path resolution.
4. **Research seam wired end-to-end** in SKILL.md §0.4 - consumes the existing `--research-levels` output from `suggest_factors.py`. Off by default, host-driven (the host's LLM does the reasoning; the plugin carries structured input/output only). Supports both numeric-level replacement and categorical levels (prompt A/B/C, model variants, etc.). NO vendor API calls inside the plugin.

## How they compose

```
User: /agent-doe "make build faster without blowing up bundle size"
  │
  ├─ Phase 0.0: worktree.py init --target "build-faster-bundle"
  │     → cd into <repo>-agent-doe-engine-build-faster-bundle/
  │
  ├─ Phase 0.1: suggest_factors.py --research-levels  → /tmp/mg-candidates.json
  │     [host LLM reads candidates]
  │
  ├─ Phase 0.2: AskUserQuestion → user confirms 4 candidates
  │
  ├─ Phase 0.3: validate_factors.py --reject-non-adjustable
  │     → 3 adjustable, 1 dead_constant (rejected, surfaced with evidence)
  │
  ├─ Phase 0.4: [host LLM researches levels for the 2 candidates flagged
  │             needs_research=true, calls its own web/research tool]
  │
  ├─ Phase 0.5: write .agent-doe-engine/optimize/factors.json
  │
  ├─ Phase 1: objectives.json + design selection
  ├─ Phase 2: DOE matrix → run each → measure → analyze
  └─ Phase 3: overfitting-reviewer → summarize → worktree.py cleanup
```

## Files added

| Path | Lines | Purpose |
|---|---|---|
| `scripts/validate_factors.py` | 539 | Adjustability classifier + byte-precise mutation probe |
| `scripts/worktree.py` | 235 | git-worktree create/info/cleanup helper |
| `tests/test_validate_factors.py` | 226 | 13 tests covering every classification branch + CLI |
| `tests/test_worktree.py` | 200+ | 16 tests covering slug, init, info, cleanup, CLI round-trip |

## Files modified

| Path | Change |
|---|---|
| `skills/agent-doe-engine/SKILL.md` | New §Phase 0 (PLAN) with subsections 0.0–0.5; §1.2 now points at Phase 0 as canonical; §Phase 3 Review now ends with worktree cleanup |
| `commands/agent-doe.md` | Three-phase → four-phase flow; calls out Phase 0 + worktree-first |

## Verification

| Check | Result |
|---|---|
| `uv run pytest -q` | 105 passed, 1 skipped (was 76 passed, 1 skipped - +29 new tests, all green) |
| `uvx reuse lint` | clean |
| Worktree isolation | confirmed via `git worktree list` - main and `feat/planning-phase` independent |
| Push / merge to main | NOT performed (per ask) |

## How Phase 0 + adjustability validation + research seam compose

The three are designed to layer cleanly:

- **Phase 0 without adjustability validation** would let the optimizer waste runs on dead constants the scanner heuristically picked up. The validator is the gate.
- **Adjustability validation without Phase 0** would still work (it accepts a JSON candidate list from any source) but the orchestration story is muddled - the user wouldn't know when to run it.
- **Research seam without Phase 0** wouldn't have a structured hand-off point - the existing `--research-levels` flag emits the signal but nothing consumed it. Phase 0.4 is the consumption.

The host-LLM-is-the-reasoner contract is preserved end-to-end: all three scripts are deterministic (stdlib only, no network, no LLM calls); the *choices* - which candidates to test, whether to research levels, how to interpret a `revert_failed` - are the host LLM's.

## End-to-end integration with atomize-ai's eval harness (Stage A)

Once atomize-ai's eval harness lands, run agent-doe-engine against it as follows:

1. From atomize-ai's project root: `python3 ${AGENT_DOE_ENGINE_ROOT}/scripts/worktree.py --workdir . --target "eval-quality-and-cost" init`
2. `cd ../atomize-ai-agent-doe-engine-eval-quality-and-cost`
3. Compose `objectives.json` from the eval harness's outputs:
   ```json
   {
     "objectives": [
       {"name": "win_rate", "direction": "higher", "weight": 0.5, "metric_cmd": "uv run python eval/run.py --json | jq .win_rate"},
       {"name": "cost_usd", "direction": "lower",  "weight": 0.3, "metric_cmd": "uv run python eval/run.py --json | jq .cost_usd"},
       {"name": "latency_p95_ms", "direction": "lower", "weight": 0.2, "metric_cmd": "uv run python eval/run.py --json | jq .latency_p95_ms"}
     ],
     "selection": "desirability"
   }
   ```
4. Run Phase 0.1–0.5: scan, pick (`PROMPT_VARIANT`, `MAX_TOKENS`, `TEMPERATURE`, retrieval `TOP_K`), validate (the validator will reject any that aren't actually wired), optionally research best-practice levels for `TEMPERATURE` and `TOP_K`.
5. Generate the DOE (`doe.py generate --design auto --factors @factors.json`).
6. Loop runs (≤8 for a 4-factor full factorial). Each run applies factor values, runs the eval harness, records per-objective values.
7. Analyze (`doe.py analyze --objectives objectives.json`) - get per-objective effects + the best run by desirability.
8. Review: overfitting-reviewer on the kept changes + worktree cleanup.

Stage A is the eval harness providing the `metric_cmd` interface; Stage B (the research seam) is shipped here; Stage 0 (Phase 0 Planning) is shipped here. The whole loop is wire-compatible the day the eval harness ships, because the integration surface is the existing `metric_cmd` string - agent-doe-engine does not need to know anything about how the eval harness runs internally.

## Notes for the reviewer

- **OAR (optimization-auto-research-plugin) untouched**, per the ask. The retirement cleanup is a separate small change.
- **No push, no merge, no marketplace publish.** Worktree is at `~/dev/git-folder/agent-doe-engine-planning` on branch `feat/planning-phase` for direct inspection. To dispose: `git -C ~/dev/git-folder/agent-doe-engine worktree remove ../agent-doe-engine-planning && git -C ~/dev/git-folder/agent-doe-engine branch -D feat/planning-phase`. To land: regular review / merge / push from the user's primary checkout.
- The validator's `revert_failed` outcome is intentionally loud (RuntimeError surfaced as `reason: revert_failed` with the file path). If it ever fires in real use, the run should stop until the user has inspected the dirty file - this is the only failure mode that can leave the tree modified.
- The worktree helper's `info` subcommand returns exit 1 when the worktree doesn't exist, so SKILL.md callers can branch (`if info; then ... else init; fi`).
- Phase 0.4's research seam is intentionally minimal - it documents the contract and points at the existing `needs_research`/`research_topic` fields. No new plugin code; the host LLM owns the call.

## Backlog from the 2026-09-05 Atomize AI dogfood (read path: feed, home, search)

Full log: the run's `lane-b-dogfood-log.md` (13 defects). Fixed the same day: goal contract + guardrail feasibility, `next_step`, `confirm`, object-literal factor scanning with `--hint`/`--paths`, zero-resolution guardrails, categorical replicate advice, categorical `validate_factors`, near-tie `contenders`. Still open:

1. **`worktree.py init` should stage runtime deps** (D4): detect gitignored `node_modules`, `.env*`, `.venv`, generated clients in the source checkout and link/copy them, warning when `node_modules` is shared.
2. **Factor discovery by reachability, not spelling** (D2): take the objective and the entry points (routes) as input and rank by "is this on the awaited path of route X"; two of the three real factors were code-path choices with no named constant.
3. **Categorical adjustability probe** (D3): substitute the other declared level and diff an observable response, instead of `not_probed`.
4. **Nuisance variables and fixture pinning** (D5, D6): a place to declare what is held constant, and a warning when a live data source drifts under the matrix.
5. **Guards that re-run the world** (D7): cache the guard per distinct code state instead of per sample.
6. **Threshold proximity** (D9b): warn when a run sits within one measurement unit of a pass threshold.
7. **`min_effect` from between-run noise** (D11): within-run replicates understate the noise a fresh run sees; recommend measuring the baseline across restarts.
