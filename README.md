# agent-doe-engine

A design-of-experiments engine for tuning AI agents.

It finds which settings actually change your results, and the best combination when your goals compete. It does this in one small, planned batch of trials, instead of guessing or testing one thing at a time.

## The problem it solves

Two things go wrong with normal tuning:

1. **One-at-a-time testing is slow and blind.** It misses the cases where two settings only matter together (for example, a bigger batch size only helps when you also add workers).
2. **A single run can fool you.** Every run has random variation, so a number that looks better might just be noise, not the setting you changed.

This engine handles both: it varies several settings together, and it tells you which changes are real.

## What it does

You list the settings to test and the results you care about (speed, cost, quality, accuracy). It runs a planned set of trials, then for each result it reports:

- **Which settings moved it, and by how much**, including settings that only matter in combination.
- **Whether that movement is real or just noise** (see "Real effect vs fluke").
- **Which settings can't be told apart** in this design (see "Tangled settings").
- **The single best configuration** when your goals compete.

## Real effect vs fluke

Run the same setting twice and the result won't be identical. That spread is the noise.

- A **real effect** is a change bigger than that noise. Re-run and you would see it again.
- A **fluke** is a change that fits inside the noise. Re-run and it may shrink, vanish, or flip.

For every effect it reports a p-value and a confidence interval, and it warns you when you have too few runs to tell a real effect from a fluke (low power). The point: you do not ship a change that was never real.

## Tangled settings (aliasing)

To test many settings in few runs, the smaller designs leave some effects mathematically inseparable. A result might be caused by setting A, or by the combination of B and C, and this data cannot tell which.

It reports exactly which effects are tangled, so you never credit the wrong setting. Full designs have no tangling; the cost of that clarity is more runs.

## The designs it uses, and why

It picks the smallest design that still answers your question:

| Settings | Design | What it does | Why it matters |
|---|---|---|---|
| 2 to 3 | **Full factorial** (4 to 8 runs) | Tests every combination | Most accurate; finds every interaction; nothing tangled |
| 4 to 7 | **Fractional factorial** (8 runs) | Tests a carefully chosen subset | Far fewer runs; some effects tangled, and it tells you which |
| 8 to 11 | **Plackett-Burman** (12 runs) | A screening design | Quickly finds the few settings that matter out of many, before a deeper test |

(For a single setting, it falls back to a simpler "try a change, measure, keep it if better" loop. That mode is cheaper to set up but cannot see interactions.)

## When your goals compete

When you care about several numbers that fight each other (faster, but cheaper, but more accurate), choose how to balance them:

| Method | What it does | Use when |
|---|---|---|
| **scalarize** | Picks the best weighted blend of your goals | You can rank goals by weight |
| **desirability** | Geometric mean of per-goal desirabilities: a zero on any goal zeroes the run. With `min_acceptable`/`target` declared the bar is absolute (Derringer-Suich); without them it is relative to the batch | No single goal can be sacrificed |
| **pareto** | Shows all the best trade-offs (the options where improving one number can only come by hurting another) | You want to see the choices before committing |

## Goals, guardrails, and knowing when you are done

Each objective carries a `role`. A **primary** must improve (meet `target`, or beat `baseline` by `min_effect`, the smallest change worth shipping). A **guardrail** must not degrade (never worse than `min_acceptable`, else `baseline`); it is a constraint, and a run that breaks it cannot win. A **quality** metric is reported and never decides. Every primary names the product `driver` it serves.

```json
{"objectives": [
  {"name": "latency_ms", "direction": "lower", "role": "primary", "driver": "page load",
   "baseline": 118, "target": 85, "min_effect": 5},
  {"name": "accuracy", "direction": "higher", "role": "guardrail", "min_acceptable": 0.90}
 ], "selection": "desirability"}
```

`analyze` then tells you what the numbers say to do next (`next_step`: decouple an alias chain, add replicates, confirm, extend a range, or stop), and `confirm` judges 3-10 confirmation runs at the best setting against the model's prediction interval and the contract. `done` is true only when every guardrail holds, every primary clears its bar, and the confirmation mean lands where the model said it would.

## Built for agent tuning

Which model (cheap or frontier), which prompt structure, which configuration, scored on accuracy and token cost and latency at the same time. A few planned runs instead of guessing.

> Extracted and extended from [build-loop](https://github.com/tyroneross/build-loop)'s single-metric `optimize` subsystem. agent-doe-engine adds multi-objective selection while keeping the same numpy-only engine.

## Install

**As a Claude Code plugin** (this repo is its own single-plugin marketplace):
```text
/plugin marketplace add tyroneross/agent-doe-engine
/plugin install agent-doe-engine@agent-doe-engine
```
Then `/agent-doe` (guided flow), `/doe` (direct matrix), and `/status` are available. The host coding agent's LLM does the reasoning (hypotheses, factor confirmation); the scripts are deterministic and host-neutral.

**As a Codex plugin** (a `.codex-plugin/plugin.json` manifest ships alongside the Claude one; point Codex at the repo).

**Standalone** (the scripts run on their own):
```bash
uv run python scripts/doe.py detect 4          # which design for 4 factors
uv run pytest -q                                # test suite
```
Requirements: Python >=3.10, numpy. Dev: pytest.

## Quick start

```bash
# 1. which design for k factors
python3 scripts/doe.py detect 2
# 2. generate the matrix
python3 scripts/doe.py generate --factors '[{"name":"workers","low":2,"high":8},{"name":"batch","low":16,"high":64}]' --design auto --seed 1 > doe.json
# 3. run each row, measure every objective into results.jsonl, then:
python3 scripts/doe.py analyze --design doe.json --results results.jsonl \
  --objectives '{"objectives":[{"name":"latency","direction":"lower","weight":0.7},{"name":"cost","direction":"lower","weight":0.3}],"selection":"scalarize"}'
# 4. read next_step; when it says confirm, measure 3-10 runs at best_factors and judge them:
python3 scripts/doe.py confirm --design doe.json --results results.jsonl \
  --objectives objectives.json --confirmation confirm.jsonl
```

Full walkthrough and the method/math: [`docs/usage.md`](docs/usage.md), [`docs/method.md`](docs/method.md).

## Scripts

| Script | Role |
|---|---|
| `scripts/doe.py` | DOE matrix generation plus multi-response effects analysis |
| `scripts/objectives.py` | multi-objective core: scalarize, desirability, Pareto, baseline aggregate |
| `scripts/loop.py` | single/few-variable autoresearch greedy loop |
| `scripts/suggest_factors.py` | codebase scanner for factor candidates |
| `scripts/metric_runner.py` | sampled metric / guard execution |

## License

Apache-2.0 (c) Tyrone Ross, Jr. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
