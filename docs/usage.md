<!-- SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Usage

Run as a Claude Code / Codex plugin (the `agent-doe-engine` skill or `/agent-doe` command drives the flow), or call the scripts directly. All scripts are numpy-only; runtime state lives under `.agent-doe-engine/optimize/` in the project being optimized.

## Worked example - optimize latency and cost together

Two factors (`workers`, `batch`), two competing objectives (latency should drop, cost should not balloon).

### 1. Define the goal contract → `.agent-doe-engine/optimize/objectives.json`

Measure the baseline 3 times first; `min_effect` is the smallest change worth shipping (at least 2x the noise SD).

```json
{
  "objectives": [
    {"name": "latency_ms", "direction": "lower", "weight": 0.7, "role": "primary",
     "driver": "request latency", "baseline": 90, "target": 50, "min_effect": 5},
    {"name": "cost_usd",   "direction": "lower", "weight": 0.3, "role": "guardrail",
     "baseline": 5.0}
  ],
  "selection": "scalarize"
}
```

`cost_usd` is a guardrail: a run costing more than the 5.0 baseline is infeasible and cannot win.

### 2. Define factors → `.agent-doe-engine/optimize/factors.json`

```json
[
  {"name": "workers", "low": 2, "high": 8},
  {"name": "batch",   "low": 16, "high": 64}
]
```

### 3. Generate the design

```bash
python3 scripts/doe.py detect 2        # -> "full"
python3 scripts/doe.py generate \
  --factors "$(cat .agent-doe-engine/optimize/factors.json)" \
  --design auto --seed 1 \
  > .agent-doe-engine/optimize/doe.json
```

### 4. Run each row, measure both objectives

For every `run` in `doe.json` (randomized `run_order`): apply `run._factors`, measure each objective, append to `results.jsonl`, then revert:

```jsonl
{"run_id": 0, "values": {"latency_ms": 88.0, "cost_usd": 2.0}, "guard_ok": true}
{"run_id": 1, "values": {"latency_ms": 51.2, "cost_usd": 5.0}, "guard_ok": true}
...
```

Use `metric_runner.py` for noisy metrics:
```bash
python3 scripts/metric_runner.py --cmd "python3 bench.py --stat p95" --samples 5 --warmups 1 --aggregate p95
```

### 5. Analyze

```bash
python3 scripts/doe.py analyze \
  --design .agent-doe-engine/optimize/doe.json \
  --results .agent-doe-engine/optimize/results.jsonl \
  --objectives .agent-doe-engine/optimize/objectives.json
```

Output (abridged):
```json
{
  "per_objective": {
    "latency_ms": {"ranked_effects": [{"term": "workers", "effect": -36.0, ...}], "r2": 0.99},
    "cost_usd":   {"ranked_effects": [{"term": "workers", "effect":   3.0, ...}]}
  },
  "selection": {
    "method": "scalarize",
    "best_run_id": 3,
    "pareto_front": [1, 3],
    "best_values": {"latency_ms": 49.6, "cost_usd": 5.0}
  },
  "best_factors": {"workers": 8, "batch": 64}
}
```

`workers` is the dominant latency driver. The output also carries `objectives_contract` (warnings about an incomplete contract), `selection.feasible_run_ids` / `infeasible` (which runs broke a guardrail), `practically_significant` per effect, and `next_step` - the ordered actions the fit supports (`decouple`, `add_replicates`, `confirm`, `extend_range`, `stop_or_widen`). The `pareto_front` `[1, 3]` shows the rational trade-offs - switch `selection` to `pareto` to choose between them yourself.

### 6. Confirm, then decide

When `next_step` says `confirm`, measure 3-10 runs at `best_factors` into `confirm.jsonl` (`{"values": {"latency_ms": 50.1, "cost_usd": 4.9}}` per line) and judge them:

```bash
python3 scripts/doe.py confirm \
  --design .agent-doe-engine/optimize/doe.json \
  --results .agent-doe-engine/optimize/results.jsonl \
  --objectives .agent-doe-engine/optimize/objectives.json \
  --confirmation .agent-doe-engine/optimize/confirm.jsonl
```

Output: `done` (true only when every guardrail holds, every primary clears its bar, and each primary's confirmation mean lies inside the model's prediction interval), a `criteria` row per objective saying why, and `recommendation` (`ship` / `more_confirmation_runs` / `re_plan`). Single-metric runs pass `--target`, `--baseline`, `--min-effect` instead of `--objectives`.

## Single-objective / single-factor

Omit `--objectives` for the original single-metric behavior, or use the autoresearch loop for one knob:

```bash
python3 scripts/loop.py --init --workdir "$PWD" \
  --target build-time --scope "webpack.config.js" \
  --metric-cmd "/usr/bin/time -p npm run build 2>&1 | grep ^real | awk '{print \$2}'" \
  --guard-cmd "npm test" --direction lower --budget 5
```

Then dispatch the `optimize-runner` agent.

## Tests

```bash
uv run pytest -q     # or: python3 -m pytest -q
```
