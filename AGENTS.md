# AGENTS.md

## Project

Build a tennis shot-selection decision system using Jeff Sackmann Match Charting Project data. The system should estimate shot-level point win probability and recommend player-specific shot mixes for tactical decisions such as forehand down-the-line versus cross-court in a given rally context.

## Core objective

For a decision context like:

- hitter = Player A
- shot number = 5
- stroke = forehand
- court side = deuce-side proxy
- candidate actions = cross-court vs down-the-line

estimate:

1. `Q(s, a)` = probability the hitter eventually wins the point if they choose action `a` in state `s`
2. the player’s observed action policy `mu(a | s)`
3. an optimized or recommended action policy `pi(a | s)`
4. decision quality metrics such as regret, uplift, and deviation from recommended mix

The project is not only predictive. It is also an offline decision-evaluation problem.

---

## What the agent should optimize for

Prioritize these goals in order:

1. **Correct state/action construction** from match-charting codes
2. **Trustworthy counterfactual shot evaluation**, not just outcome prediction
3. **Calibration** of win probabilities
4. **Player-specific estimates** with shrinkage / partial pooling
5. **Interpretability** for coaching use
6. **Reproducibility** and leakage-free evaluation

Do not optimize purely for headline predictive accuracy if it hurts interpretability or decision validity.

---

## Modeling philosophy

Treat each rally as a sequential decision process.

- A rally is an episode.
- Each shot is a decision point.
- Terminal reward is `1` if the focal hitter wins the point and `0` otherwise.
- We care about the value of a shot choice conditional on rally state.

The agent should distinguish between:

- **State value** `V(s)`: probability of winning the point from state `s`
- **Action value** `Q(s, a)`: probability of winning the point if action `a` is chosen in state `s`
- **Behavior policy** `mu(a | s)`: what the player actually chose historically
- **Target policy** `pi(a | s)`: what we recommend

The primary deliverable is `Q(s, a)`, not only `V(s)`.

---

## Dataset assumptions

Primary dataset: Jeff Sackmann Match Charting Project.

Expected structure includes:

- point-level rows
- coded rally strings in `1st` and `2nd`
- point winner in `PtWinner`
- server in `Svr`
- score / game / set context such as `Pts`, `Gm1`, `Gm2`, `Set1`, `Set2`
- player names and match metadata where available

The agent must assume the match-charting strings are the source of shot sequence truth and must parse them carefully.

---

## Non-negotiable rules

### 1. Never skip parsing validation

Before modeling, always validate that:

- point strings are parsed correctly
- shot order is correct
- terminal event assignment is correct
- hitter alternation is correct
- point winner labels align with parsed terminal events

If parsing logic changes, rerun validation summaries before any downstream modeling.

### 2. Do not treat descriptive win rates as causal conclusions

Simple conditional win rates by action are only sanity checks. They are not sufficient for recommendations because of selection bias and missing context.

### 3. Do not recommend a policy without overlap checks

If a player almost never uses one action in a state, flag the recommendation as low-support rather than overconfident.

### 4. Avoid leakage

Do not let future shots in the rally leak into features for a decision at shot `t`.

For a shot-`t` decision, only use information available up to and including the decision state just before action `a_t`.

### 5. Prefer calibrated probabilities

Every probability model used for decision support must be checked for calibration. Report Brier score, log loss, and reliability diagnostics where possible.

---

## Recommended project structure

```text
project/
  data/
    raw/
    interim/
    processed/
  notebooks/
    01_parse_and_validate.ipynb
    02_feature_build.ipynb
    03_baseline_q_models.ipynb
    04_player_policy_eval.ipynb
    05_final_recommendations.ipynb
  src/
    parsing/
      mcp_parser.py
      validation.py
    features/
      state_builder.py
      action_builder.py
      context_filters.py
    models/
      baseline_glm.py
      gbdt_q_model.py
      seq_model.py
      propensity_model.py
      calibration.py
    evaluation/
      offline_policy_eval.py
      overlap.py
      regret.py
      uplift.py
    reporting/
      recommendation_tables.py
      plots.py
  outputs/
    models/
    tables/
    figures/
  tests/
    test_parser.py
    test_state_builder.py
    test_no_leakage.py
  AGENTS.md
```

---

## State construction guidance

The agent should build decision states at the shot level.

### Minimum state fields

- `match_id`
- `point_id`
- `shot_index`
- `hitter`
- `opponent`
- `server`
- `surface` if available
- `score_state` from `Pts`
- `set_score`
- `game_score`
- `rally_length_so_far`
- `previous_shot_type`
- `previous_shot_direction`
- `previous_shot_depth` if inferable
- `current_stroke_type` such as forehand / backhand / slice
- `net_state` if inferable from code
- `serve_context` for early shots

### Important contextual features

The agent should strongly consider engineered features such as:

- shot number buckets: `1`, `2`, `3`, `4-5`, `6+`
- offensive / neutral / defensive proxy from recent shot pattern
- whether the hitter is responding to a wide ball proxy
- whether the opponent was pulled off-court on the previous shot proxy
- server / returner role
- handedness if available from external merge

### Deuce/ad-side approximation

The charting dataset does not directly provide full court coordinates for every shot. Therefore, any “deuce side” or “ad side” label may be partially inferred.

The agent must:

- clearly document the proxy used
- maintain a confidence field for side inference if needed
- avoid overstating certainty
- compare alternative side-inference schemes in sensitivity analysis

---

## Action construction guidance

The first tactical action space should be intentionally narrow and high-quality.

### Phase 1 action space

For forehand groundstrokes in selected contexts:

- `CC` = cross-court-like
- `DTL` = down-the-line-like
- optionally `BODY/MIDDLE` or `OTHER` if needed for coverage

Only include actions when the mapping from code to tactical class is clear enough.

### Recommended rollout order

1. Forehand direction only
2. Backhand direction
3. Serve + first-ball patterns
4. Net-approach decisions
5. Richer action spaces with depth and court-position proxies

Do not expand the action space until the narrow action space is reliable.

---

## Modeling roadmap

The agent should proceed in stages.

### Stage 1: parser + descriptive sanity checks

Build a robust parser that converts charting strings into shot-level event tables.

Deliverables:

- one row per shot decision
- parsed shot type, direction, modifiers, and terminal outcome
- sanity-check tables by player, shot number, and action

### Stage 2: baseline direct `Q(s, a)` model

Train an interpretable baseline that predicts point-win probability from:

- state features
- chosen action
- player identity effects

Preferred first models:

- logistic regression with interactions
- hierarchical / partial-pooling logistic model
- gradient-boosted trees if feature interactions are strong

This stage should establish a stable, calibrated baseline before sequence models.

### Stage 3: behavior policy model `mu(a | s)`

Estimate the probability that the player chose each action in context.

Use this for:

- overlap diagnostics
- inverse propensity weighting
- doubly robust offline policy evaluation

### Stage 4: offline policy evaluation

Evaluate alternative shot policies using off-policy methods.

Preferred sequence:

1. direct method
2. inverse propensity weighting with clipping
3. doubly robust estimator
4. bootstrap confidence intervals

Flag low-support regions.

### Stage 5: player-specific recommendation engine

For each player and context:

- estimate `Q(s, CC)` and `Q(s, DTL)`
- estimate observed `mu(a | s)`
- compute recommended `pi(a | s)`
- quantify uncertainty
- translate into a coaching-style recommendation

### Stage 6: sequence model extension

Only after the baseline pipeline is stable, add a sequence model such as:

- LSTM / GRU
- Transformer encoder over shot history

Use it only if it improves calibrated `Q` estimates or policy evaluation quality. Do not replace a trustworthy baseline with a less interpretable black box unless the gain is meaningful.

---

## Recommendation policy guidance

A purely greedy policy often returns 100 percent on one action. That is usually too brittle for coaching.

The preferred default is a soft recommendation policy:

```text
pi(DTL | s) = sigmoid((Q(s, DTL) - Q(s, CC)) / tau)
```

Where:

- lower `tau` gives a more deterministic recommendation
- higher `tau` gives a more mixed and robust recommendation

The agent should expose `tau` as a tunable robustness parameter.

Default deliverable:

- recommended `DTL%`
- recommended `CC%`
- expected uplift vs observed policy
- uncertainty band

---

## Player evaluation metrics

The agent should score player decision-making using at least these metrics.

### 1. Decision regret

For each observed decision:

```text
regret = max_a Q(s, a) - Q(s, chosen_action)
```

Aggregate by player, context, surface, and opponent class.

### 2. Policy gap

Difference between:

- observed action rate `mu(a | s)`
- recommended action rate `pi(a | s)`

This is the most intuitive “shot selection gap” metric.

### 3. Expected point-win uplift

Estimate the increase in expected point-win probability if the player shifts from observed policy to recommended policy.

### 4. Calibration-aware value score

Do not report value gains without calibrated probability checks.

### 5. Support / trust score

Each recommendation must include a confidence or support tag such as:

- high support
- moderate support
- low support / extrapolation risk

---

## Evaluation protocol

### Splitting

Preferred split hierarchy:

- out-of-match split at minimum
- ideally out-of-time split where feasible
- for player-specific recommendations, ensure evaluation is not polluted by near-duplicate match contexts

### Metrics

For reward / value models:

- log loss
- Brier score
- AUC only as a secondary metric
- calibration curve / expected calibration error

For policy evaluation:

- estimated policy value
- DR estimate with bootstrap interval
- support diagnostics
- proportion of contexts with unstable recommendation

### Heterogeneity checks

Always stratify by:

- player
- surface
- server / returner status
- rally phase
- score pressure if sample size allows

---

## Sensitivity analyses the agent should run

The agent should not stop at one model. It must test whether conclusions are stable to:

1. alternative deuce-side inference rules
2. different action mappings from MCP codes to tactical classes
3. clipping thresholds in importance weighting
4. player pooling strength
5. inclusion or exclusion of rare contexts
6. surface-specific versus pooled models
7. score-context inclusion

A recommendation that changes wildly under minor specification changes should be labeled unstable.

---

## Failure modes to watch for

The agent must proactively check for these issues:

- parsing errors in rally strings
- mis-assigned hitter on odd/even shots
- accidental use of future information
- sparse-action extrapolation
- probabilities that are sharp but miscalibrated
- player identity overfitting
- recommendations driven by tiny samples
- confusing descriptive rates with counterfactual value

---

## Preferred outputs

### Technical outputs

- shot-level modeling table
- parser validation tables
- calibrated `Q(s, a)` models
- policy evaluation summaries
- player-specific recommendation tables

### Coaching-style outputs

For a player and context, generate summaries like:

> On 5th-ball forehands in the deuce-side proxy context, Player A currently chooses down-the-line 31% of the time. The model recommends 44% down-the-line and 56% cross-court. Estimated gain from shifting to this mix is +1.3 percentage points in point-win probability, with moderate support.

### Visual outputs

- action-value difference plots by context
- observed vs recommended shot-mix charts
- regret heatmaps by player
- calibration plots
- support / overlap plots

---

## Coding expectations for the agent

- Write modular Python code.
- Prefer pandas or polars for data wrangling.
- Keep parser logic separate from modeling logic.
- Write tests for parsing and state construction.
- Save intermediate tables so parsing is not recomputed unnecessarily.
- Use deterministic seeds where possible.
- Every model artifact should store training metadata and feature lists.

---

## Priority order for the first working version

The first usable version should do exactly this:

1. parse MCP point strings into shot-level rows
2. isolate a narrow decision context such as 5th-shot forehand direction
3. build an interpretable direct `Q(s, a)` baseline
4. build a behavior-policy model
5. run doubly robust offline policy evaluation
6. output player-specific observed vs recommended shot proportions
7. quantify regret and uplift

Do not begin with a large RNN if these basics are not working.

---

## Definition of done

A task in this project is only done when all of the following are true:

- parsing has been validated
- the state/action definition is documented
- leakage checks passed
- calibration has been evaluated
- overlap/support has been evaluated
- uncertainty has been reported
- the recommendation can be explained in plain English

---

## Short instruction to the agent

Build the simplest trustworthy decision-value pipeline first. Parse carefully, model `Q(s, a)` directly, estimate the logged policy, evaluate alternative policies offline with caution, and only then scale up to richer sequence models.
