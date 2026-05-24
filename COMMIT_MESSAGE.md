# Commit Message: Integrate adaptive decision evaluation and branch-derived sequence baselines

## Summary

This change extends the fifth-shot forehand decision pipeline into a more complete
offline decision-evaluation framework while preserving the validated MCP parser as
the single source of rally truth.

The implementation adds match-local exposure features, opponent response and
adaptability diagnostics, adaptive player/match evaluation outputs, and clean
integration points for the useful ideas from the `rnn_implementation` and `sai`
branches. The branch code was not merged wholesale because those branches contain
duplicate parsers, experimental neural/RL scripts, pretrained artifacts, and some
decision-time leakage risks. Instead, the reusable concepts were reimplemented
against the existing parsed shot-level table.

## Major Changes

### Match-local exposure features

- Added pre-decision exposure features for fifth-shot forehand CC/DTL contexts.
- Features are grouped by match, hitter, opponent, and side proxy.
- The current action is excluded from its own exposure features.
- Added recent and cumulative DTL/CC counts and rates.

### Adaptive Q and policy evaluation

- Added CC-facing policy gap and Q-edge metrics alongside the existing DTL metrics.
- Player recommendations can now be interpreted by overall mix and by high/low
  recent exposure buckets.
- Added adaptive pressure and timing-error metrics so player evaluation can answer
  whether a player used more CC when recent DTL exposure made DTL more vulnerable.

### Opponent response and adaptability model

- Added shot-6 opponent response features after fifth-shot forehand decisions.
- Added response outcomes:
  - opponent made next ball
  - opponent won within two shots
  - opponent neutralized
  - hitter forced next-shot error
- Added future-exposure placebo features to help distinguish adaptation from
  match-flow artifacts.
- Added match-level bootstrap uncertainty for opponent adaptability estimates.

### Sequence-model integration from collaborator branches

- Added a shared sequence-model interface inspired by `rnn_implementation`.
- Added parsed-shot sequence builders that consume the validated shot-level table
  instead of duplicate rally parsers.
- Added simple empirical and Markov sequence win-probability baselines.
- Added a sequence-model comparison evaluator with Brier score, log loss, AUC,
  and expected value gap.
- Added a fast next-shot behavior-policy model inspired by `sai`'s next-shot work.
- Kept neural/RL work as future-compatible rather than default pipeline behavior.

### Tests

- Added tests for sequence table construction.
- Added tests that next-shot examples use only prefix tokens.
- Added tests that opponent-response features attach the next shot and keep
  future-exposure placebo separate.
- Existing parser, state-construction, and no-leakage tests continue to pass.

## Validation

The following validation was run before committing:

```text
pytest -q tests/test_parser.py tests/test_state_builder.py tests/test_no_leakage.py tests/test_sequence_integration.py
10 passed

python -m compileall src models/markov_kartik tests
passed
```

Parser validation on filtered hard-court 2022-2024 MCP points:

```text
points_total:            202,622
points_parseable:        202,398
winner_alignment_count:  201,448
winner_alignment_rate:   99.42%
rally_terminal_count:    166,870
serve_terminal_count:     35,752
mean_rally_shot_count:      3.74
```

## Notes

- Generated CSV/model artifacts are intentionally not part of this source-code
  integration commit unless separately staged.
- The full all-shot neural/RL models are not yet promoted into the core decision
  pipeline. The current implementation creates safe interfaces and lightweight
  baselines first.
- The next major step is generalizing the decision table beyond fifth-shot
  forehands to all reliable shot/action contexts.
