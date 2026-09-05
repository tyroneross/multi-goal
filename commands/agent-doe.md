---
description: Multi-objective optimization via Design of Experiments - optimize competing metrics (latency + cost + size) at once and find the best trade-off. Falls back to a single-variable autoresearch loop.
argument-hint: "[target or 'latency and cost' or factor list]"
---

<!-- SPDX-FileCopyrightText: 2025-2026 Tyrone Ross, Jr <46267523+tyroneross@users.noreply.github.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Run the agent-doe-engine optimizer for: $ARGUMENTS

Invoke the `agent-doe-engine` skill and follow its five phases:

0. **Plan** - open a dedicated worktree (`scripts/worktree.py init`); scan for factor candidates (`scripts/suggest_factors.py --research-levels`); have the host LLM pick the candidates to test (AskUserQuestion confirmation); validate adjustability (`scripts/validate_factors.py --reject-non-adjustable`); optionally research best-practice levels via the host's research capability (host-driven, off by default - see SKILL.md §0.4); write the validated factors to `.agent-doe-engine/optimize/factors.json`.
1. **Setup** - write the goal contract: each objective with direction, weight, `role` (primary / guardrail / quality), the product `driver` it serves, a one-line metric command, and absolute bars (`target`, `min_acceptable`, `min_effect`) after measuring the baseline 3 times; pick a selection method (`scalarize` / `desirability` / `pareto`). The factors are already on disk from Phase 0.
2. **Run** - generate the DOE matrix (≥2 factors) or initialize the autoresearch loop (1 factor), measure every objective on every run, and analyze with `scripts/doe.py analyze --objectives ...`. Follow `next_step` (decouple / add_replicates / confirm / extend_range / stop_or_widen).
3. **Confirm** - measure 3-10 runs at `best_factors` and run `scripts/doe.py confirm`; nothing is done until it says `done: true`.
4. **Review** - dispatch `overfitting-reviewer`, summarize per-objective improvement and the chosen trade-off, archive, and clean up the worktree (`scripts/worktree.py cleanup`).

If $ARGUMENTS names a single number and a single knob, still do Phase 0 (the validator catches dead-constant single-factor cases too) and then go to the autoresearch loop. If it names competing goals ("faster and cheaper"), set up a multi-objective `objectives.json`. Never auto-run optimization on heuristic factor candidates that have not passed `validate_factors.py`, and never run agent-doe-engine directly in the user's primary checkout - always work inside the Phase 0.0 worktree.
