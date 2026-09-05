---
name: agent-doe-engine
description: Use when the user wants to tune an AI agent or optimize one or more measurable numbers at once - "optimize this", "make X faster without blowing up Y", "reduce latency and cost", "find the best trade-off between A and B", "tune these parameters", "which model/prompt/setup", "speed up my app". Runs a Design of Experiments matrix (up to 11 factors in one pass), measures every objective on every run, and selects the best trade-off by weighted scalarization, Derringer-Suich desirability, or Pareto frontier. Falls back to a single-variable autoresearch loop.
user-invocable: true
---

<!-- SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- agent-doe-engine@tyroneross:canary:agent-doe-engine -->
<!-- canary-end -->

# agent-doe-engine

Optimize numbers you can measure - fast. The core idea is Design of Experiments: **test many input variables at once in a handful of runs** instead of changing one thing at a time. Then, when several *outcomes* compete, pick the setting that best trades them off.

The efficiency is the whole point. One-factor-at-a-time needs a run per variable and still misses interactions. DOE resolves several variables together: 2–3 factors in ≤8 runs, 4–7 in 8 runs, 8–11 in a 12-run screening pass. Build time, latency, token cost, bundle size, coverage, accuracy - anything a one-line command turns into a number.

The metric is the only judge. No "this looks better."

`${CLAUDE_PLUGIN_ROOT}` below is the plugin root; from a clone it's the repo root. Runtime state lives in the **consumer** project under `.agent-doe-engine/optimize/`.

## Three shapes of request

| Shape | Trigger | Path |
|---|---|---|
| **Multi-factor** *(the core - fewer runs)* | one number, several knobs to test together | DOE matrix, single objective |
| **Multi-objective** *(the differentiator)* | ≥2 competing numbers ("faster AND cheaper", "latency vs accuracy") | DOE matrix + an `objectives` list + a `selection` method |
| **Single-factor** | one number, one thing to try | autoresearch greedy loop |

Multi-factor and multi-objective compose: a single DOE run can test many variables *and* score several objectives at once - that is the fastest path to a good trade-off.

## Phase 0: PLAN - pick the right variables before spending runs

Skip when the user already named factors **and** they're known-adjustable in this repo. Otherwise run this phase first - wrong factors burn the whole budget on noise.

### 0.0 - Open a dedicated worktree (mandatory; do NOT run in main)

Every agent-doe-engine run mutates factor values across many DOE runs. Doing that in the user's primary checkout interleaves optimization writes with real work-in-progress and risks leaving the tree dirty if a run is killed. The helper handles create / reuse / cleanup:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/worktree.py \
  --workdir "$TARGET_REPO" --target "<target name>" --json init
```

Use the printed `path` as the worktree from this point on (`cd` into it before any further agent-doe-engine command). The branch is `agent-doe-engine/<slug>`; the worktree is `<repo-name>-agent-doe-engine-<slug>` alongside the repo. Re-running `init` is idempotent. At the end of Phase 3 Review run `worktree.py ... cleanup [--delete-branch]` to remove it.

This is the default path, not an afterthought - there is no "just run in main" shortcut. The helper is stdlib-only and never reaches outside `git worktree` operations.

### 0.1 - Scan for factor candidates

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/suggest_factors.py \
  --workdir "$PWD" --top 12 --json --research-levels > /tmp/mg-candidates.json
```

`--research-levels` flags high-confidence numeric knobs whose names match tuning keywords (`batch`, `timeout`, `lr`, ...) with `needs_research: true` and a `research_topic` string. The script never calls research itself - it just marks which candidates would benefit if the host has a research capability available (see §0.4).

### 0.2 - Host LLM ranks and picks the candidates to test

The host coding agent's LLM reads the candidate list and selects which to take forward. The script is deterministic; the choice is reasoning work. Use the AskUserQuestion path to confirm - candidates **pre-checked**, per existing convention. Surface for each: `name`, `current_value`, `suggested_levels`, `confidence`, file:line, and one-line `why`. Limit the user-facing list to the ~6 highest-signal entries; let the user add/remove.

This is the canonical confirmation point - never auto-run downstream phases on heuristic candidates alone.

### 0.3 - Validate adjustability (REQUIRED before DOE)

For each accepted candidate, prove the optimizer can actually move it before spending DOE runs:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/validate_factors.py \
  --workdir "$PWD" --candidates /tmp/mg-candidates.json --json --reject-non-adjustable \
  > .agent-doe-engine/optimize/validated_factors.json
```

The validator performs a snapshot → mutate → re-read → revert → verify cycle on each candidate's primary definition site. Output classification:

| adjustability | reason | Action |
|---|---|---|
| `adjustable` | `ok` | enters the DOE |
| `not_adjustable` | `dead_constant` | reject - zero references; the optimizer would change a value nothing reads |
| `not_adjustable` | `duplicate_definition` | reject - two sites with conflicting values, which one wins is ambiguous |
| `not_adjustable` | `mutation_failed` | reject - write didn't land (read-only FS, locked file, race with build cache) |
| `not_adjustable` | `revert_failed` | hard-surface - working tree is dirty; **stop the run** and ask the user before continuing |
| `not_adjustable` | `no_definition_site` | reject - name vanished since the scan |

Only `adjustable` candidates enter `factors.json`. For each rejection, show the user the `reason` and `evidence` so they can fix the underlying issue (extract a duplicate to a single config, add a real reference, etc.) or override. `--reject-non-adjustable` exits 1 - surface that to the user with the rejection summary; the user decides whether to drop, fix, or override.

### 0.4 - Research seam (optional, host-driven, off by default)

For any validated candidate that was flagged `needs_research: true` in §0.1, the host LLM may consult its research capability (web search, Exa, Context7, internal docs - whatever the host has available) to propose best-practice levels. The reasoning is the host's; the plugin only carries the structured input/output.

- **Numeric factor**: replace the heuristic `suggested_levels` ([0.5x, 1x, 2x]) with researched levels (e.g. for `BATCH_SIZE = 32` on a Transformer training loop, research may suggest `[8, 16, 32, 64]` based on published GPU memory tradeoffs).
- **Categorical factor**: replace the levels with named variants (`{"name": "prompt_variant", "levels": ["chain-of-thought", "few-shot", "zero-shot"]}`). The DOE machinery treats them as categorical levels - useful for prompt A/B/C, tokenizer choice, model variant, scheduler family, etc.

This step is **off by default**. Enable only when the user explicitly asks ("research good levels for these") OR when the candidate set is small enough (≤3 factors) that the research overhead is worth it. The host invokes its own research tool - there are **no vendor API calls inside this plugin**. Always cite the source the research returned in the `factors.json` `why` field so a future run can audit it.

If the host has no research capability, skip this step silently - the heuristic levels are a working default.

### 0.5 - Compose the factor file

Write the validated (and optionally researched) candidates to `.agent-doe-engine/optimize/factors.json` in the shape `[{name, low, high}]` (numeric two-level) or `[{name, levels:[...]}]` (numeric multi-level OR categorical). From here, the rest of the SETUP phase (objectives, design) proceeds as Phase 1 below.

## Phase 1: SETUP - the goal contract

Wrong metric = Goodhart's Law. Wrong factors = wasted runs. This is the highest-leverage phase, and it is where "done" gets defined - before any run, not after.

### 1.1 - Write the goal contract

Every objective is a number, a direction, and a **role** that turns it into a decision rule. Tie each primary to the product **driver** it serves; a number with no driver cannot tell you whether moving it mattered.

```json
{
  "objectives": [
    {"name": "feed_p50_ms", "direction": "lower", "weight": 0.6, "role": "primary",
     "driver": "feed is the product surface: page load", "metric_cmd": "python3 bench.py --stat p50",
     "baseline": 1630, "target": 800, "min_effect": 100, "validity": "validated"},
    {"name": "search_accuracy", "direction": "higher", "weight": 0, "role": "guardrail",
     "driver": "search answers the question", "metric_cmd": "npx tsx scripts/eval.ts --metric",
     "baseline": 0.83, "min_acceptable": 0.83},
    {"name": "db_grounding", "direction": "higher", "weight": 0, "role": "quality",
     "metric_cmd": "..."}
  ],
  "selection": "desirability"
}
```

| Role | Decision rule | Must carry |
|---|---|---|
| `primary` | must **improve** (superiority): meets `target`, or beats `baseline` by `min_effect` | `driver`, plus `target` or `min_effect` |
| `guardrail` | must **not degrade** (non-inferiority): never worse than `min_acceptable`, else `baseline` | `min_acceptable` or `baseline` |
| `quality` | reported, never decides | - |

Guardrails are constraints, not score terms: a run that breaks one is infeasible and cannot win however good its primaries look. `min_acceptable` / `target` are absolute limits in raw units (Derringer & Suich 1980); without them desirability is only relative to the batch, so the worst run always scores 0 even when it is acceptable. `min_effect` is the smallest change worth shipping - an effect below it is reported as real-but-not-practical.

**Measure the baseline first, at least 3 times.** Record the mean as `baseline` and set `min_effect` to at least 2x the sample SD (noise floor). `doe.py analyze` runs `validate_objectives` and refuses a contradictory contract (bar better than target); an incomplete one runs, with warnings that say what "done" cannot mean yet.

Write it to `.agent-doe-engine/optimize/objectives.json`. One primary objective is the single-metric case - everything below still works (`--min-effect`, `--target`, `--baseline` on the CLI).

**`validity` field** (optional; default `unvalidated`): `validated` = the metric is known to track the user outcome (correlation study, A/B, published benchmark); `unvalidated` = a proxy; `needs_human_ratings` = a proxy with ratings available. The overfitting reviewer treats a winner selected on an unvalidated metric as a `strong_checkpoint` finding (Goodhart risk).

**Choosing `selection`:**

| Method | Picks (among feasible runs) | Use when |
|---|---|---|
| `scalarize` *(default)* | max weighted sum of normalized primaries | you can express priorities as weights |
| `desirability` | max Derringer-Suich D (geometric mean of per-primary desirabilities) | every primary must clear a bar - a zero on one tanks the run |
| `pareto` | the non-dominated trade-off set (single winner = max-desirability point on the front) | you want to see all trade-offs before committing |

### 1.2 - Identify factors

If the user named factors ("optimize workers, batch_size, timeout"), validate the shape `[{name, low, high}]` or `[{name, levels:[...]}]` and run them through Phase 0.3 (adjustability validation) before skipping ahead.

Otherwise the factor inventory comes from **Phase 0 (PLAN)** above: scan → host-LLM picks → adjustability validation → optional research → `.agent-doe-engine/optimize/factors.json`. Phase 0 is the canonical path; this section is the contract for what `factors.json` must contain. Do not auto-run optimization on heuristic candidates that have not passed `validate_factors.py`.

### 1.3 - Pick the design (≥2 factors)

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/doe.py detect <k>
```
Routing: `k=1` → autoresearch (§Single-factor); `2–3` → 2^k full factorial (≤8 runs); `4–7` → fractional factorial 2^(k-p) Res III/IV (8 runs); `8–11` → Plackett-Burman 12-run screening.

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/doe.py generate \
  --factors "$(cat .agent-doe-engine/optimize/factors.json)" \
  --design auto --seed "$RANDOM" \
  > .agent-doe-engine/optimize/doe.json
```

## Phase 2: RUN THE MATRIX

For each row in `.agent-doe-engine/optimize/doe.json` (in randomized `run_order`):

1. Apply the factor values from `runs[i]._factors` to code / config / env.
2. Measure **every objective** - run each objective's `metric_cmd` (use `metric_runner.py` for sampled/aggregated measurement of noisy metrics):
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/metric_runner.py --cmd "<metric_cmd>" --samples 5 --warmups 1 --aggregate p95
   ```
3. Run the guard (must exit 0): `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/metric_runner.py --guard "<guard_cmd>"`.
4. Append to `.agent-doe-engine/optimize/results.jsonl`: `{"run_id": i, "values": {"latency_ms": .., "cost_usd": ..}, "guard_ok": true}`.
5. Revert the factor changes - each DOE run starts from the same baseline; the design does not accumulate.

Then fit effects and select:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/doe.py analyze \
  --design .agent-doe-engine/optimize/doe.json \
  --results .agent-doe-engine/optimize/results.jsonl \
  --objectives .agent-doe-engine/optimize/objectives.json \
  > .agent-doe-engine/optimize/effects.json
```

Output: ranked main effects + interactions **per objective**, the `selection` result (best run, scores, **always** the `pareto_front`), and `best_factors` (concrete winning values). Apply the winning combination as one commit. If `selection: pareto`, present the front and let the user pick the trade-off; default to the max-desirability point.

## Phase 2b: READ THE NEXT STEP - the measurements decide what happens next

`effects.json` carries `next_step`: an ordered list of `{action, terms, reason}` derived from the fit, not from opinion. Follow the first entry.

| Action | Fires when | What to do |
|---|---|---|
| `decouple` | a significant effect shares its column with other terms (Res III/IV alias chain) | fold-over, or one-factor-at-a-time confirmation runs at the aliased terms; do not credit any of them yet |
| `add_replicates` | saturated design (no error df), degenerate fit, or low power | add 3 center points or replicate 2-3 rows; p-values are not evidence until then |
| `confirm` | at least one real effect | go to Phase 2c |
| `extend_range` | a significant main effect with the best run at the edge of its range | the optimum may lie beyond: move that level further out in the next stage (steepest ascent) |
| `stop_or_widen` | nothing beat noise or `min_effect` | the factors do not move the number (record that), or the levels were too close (widen and re-run) |

Sequential experimentation is the method, not an option: plan the first matrix at roughly a quarter of the run budget (Montgomery), keep the rest for decoupling, moving, and confirming. Each ranked effect also carries `low_to_high_change` (2x the coefficient) and `practically_significant` (against `min_effect`).

## Phase 2c: CONFIRM - the done criteria

Nothing is done because a batch had a best row. Run **at least 3 (5-10 per Jensen 2016) confirmation runs at `best_factors`**, one row per run in `confirm.jsonl` as `{"values": {...}}`, then:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/doe.py confirm \
  --design .agent-doe-engine/optimize/doe.json \
  --results .agent-doe-engine/optimize/results.jsonl \
  --objectives .agent-doe-engine/optimize/objectives.json \
  --confirmation .agent-doe-engine/optimize/confirm.jsonl \
  > .agent-doe-engine/optimize/confirm.json
```

`done` is true only when **all** of these hold, and the output says which one failed:

1. every guardrail holds on the confirmation mean;
2. every primary meets its `target`, or beats `baseline` by at least `min_effect`;
3. the confirmation mean of every primary lies inside the model's prediction interval (`mean_in_pi`) - the design predicted this result, so it is not a fluke;
4. at least 3 confirmation runs were measured;
5. Phase 3's overfitting review passes.

`recommendation` is `ship`, `more_confirmation_runs`, or `re_plan`. When the design has no error estimate the interval falls back to the confirmation sample SD and is flagged `pi_source: confirmation_sd` - weaker, and said so.

## Single-factor - autoresearch loop

When there is one factor (or one thing to try), skip DOE.

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/loop.py --init --workdir "$PWD" \
  --target "<name>" --scope "<glob>" \
  --objectives "$(cat .agent-doe-engine/optimize/objectives.json | python3 -c 'import sys,json;print(json.dumps(json.load(sys.stdin)["objectives"]))')" \
  --selection scalarize \
  --metric-cmd "true" --guard-cmd "<cmd>" --budget 20 --direction lower
```

Measure the baseline once, then record it:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/loop.py --set-baseline --workdir "$PWD" \
  --baseline-values '{"latency_ms": 100, "cost_usd": 5}'
```

Then dispatch the `optimize-runner` agent. Each iteration: hypothesize one atomic change → apply → measure every objective → `loop.py --score --values '{...}'` to get the scalar aggregate (improvement ratio vs baseline; >1 = better) → keep if aggregate improves and the guard passes, else `git revert`. Convergence: 5 consecutive discards, regressing trend, or budget exhausted.

Single-objective mode is the original behavior - omit `--objectives` and use `--metric-cmd` directly.

## Phase 3: REVIEW

1. Dispatch `overfitting-reviewer` (read-only): check for removed safety, fragile shortcuts, metric-gaming, scope violations across the kept changes.
2. Summarize: runs, kept/reverted, per-objective improvement, the chosen trade-off.
3. Archive: `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/loop.py --archive --workdir "$PWD"`.
4. Worktree cleanup (Phase 0.0 counterpart). When the user has reviewed the kept changes and is ready to merge/cherry-pick or discard, remove the agent-doe-engine worktree:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/worktree.py \
     --workdir "$TARGET_REPO" --target "<target name>" --json cleanup [--delete-branch]
   ```
   Default keeps the branch (so the user can inspect / merge later); add `--delete-branch` only when the user explicitly discards the run.

## Model tiering (when running under a multi-model host)

| Component | Tier | Why |
|---|---|---|
| Setup (objectives, factors, selection) | Thinking | Wrong metric = Goodhart |
| Hypothesis generation | Code | High volume |
| Metric / guard / analyze | deterministic scripts | no LLM |
| Keep/revert | deterministic | numeric comparison |
| Overfitting review | Code (read-only) | pattern matching |

## State files

```text
.agent-doe-engine/optimize/
├── objectives.json   # objectives + selection method
├── factors.json      # factor definitions
├── doe.json          # generated design matrix
├── results.jsonl     # measured responses per run
├── effects.json      # per-objective effects + selection result
├── experiment.json   # autoresearch config (single/few-factor mode)
├── results.tsv       # autoresearch iteration log
└── experiments/      # archived runs
```

## Profiles

See `profiles.md` for ready-made single-objective presets (simplify, build time, bundle size, latency). Compose them into a multi-objective `objectives.json` when you want to optimize several at once.
