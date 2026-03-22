# Experiment Archive

This file contains branches that were tested and are no longer part of the active roadmap.

## Public-Proven Dead Or Overfit

### Graph smoothing

- Reason:
  - holdout looked good
  - public collapsed
- Key result:
  - `0.8006751846`

### Oracle target-wise holdout blend

- Reason:
  - classic holdout overfit
- Key result:
  - public `0.7947312943`

### Broad AutoGluon hard-target replacement

- Reason:
  - extremely strong offline
  - unstable and overfit on public
- Key results:
  - `a0p7 = 0.8075571610`
  - `a1p0 = 0.8031004260`

## Local-Proven Low Value

### Adversarial weighting / train-test shift

- Reason:
  - adversarial AUC `~0.4993`
  - no meaningful shift signal

### Weighted baseline sanity

- Reason:
  - weak tiny gain
  - did not challenge stronger branches

### LGBM specialists as a main family

- Reason:
  - weak family relative to `CatBoost`
  - no compelling route to best public

## Archived Broad Post-Processing

### Broad `error-corrector`

- Reason:
  - on a broad weak block, subset macro got worse
- Note:
  - narrow correction remains alive in the main roadmap

### Broad `object-level gate`

- Reason:
  - broad `spec gate` on the weak block got worse
- Note:
  - narrow gate remains alive only as an endgame overlay

### Wide router tuning

- Reason:
  - useful historically
  - now dominated by `TabM longrun v3`

### `Feature-view` as direct submit-layer over `TabM v3`

- Reason:
  - local transformed-view signal exists
  - but direct online transfer failed even in a very conservative probe
- Key result:
  - `clip_only`, shortlist-6, `alpha=0.01`
  - holdout delta: `+0.0000727280`
  - public: `0.8408613052`
  - delta vs best public `-0.0000813827`
- Note:
  - transformed views remain alive only as inputs to new backbones / richer challengers, not as direct postprocess submissions

## Archived Near-Zero Micro-Uplifts

### Target-wise `TabM + CatBoost` soft-blend over weak targets

- Reason:
  - real signal existed only on a couple of targets
  - full holdout uplift was too small to justify a separate public slot
- Key result:
  - base global `TabM + CatBoost` logit blend (`0.7 / 0.3`): `0.8440603637`
  - best target-wise softened blend: `0.8440851027`
  - delta: `+0.0000247389`
- Artifacts:
  - `artifacts/post_tabm_catboost_weak_local_v1/soft_blend_summary.json`
  - `artifacts/post_tabm_catboost_weak_local_v1/soft_blend_weight_grid.csv`
- Note:
  - keep only as a reference that target-specific blend search was tried

### Weak-target `meta-layer` over `TabM + CatBoost`

- Reason:
  - slightly better than target-wise soft-blend
  - still too small to clearly beat submit noise / public variance
- Key result:
  - base global `TabM + CatBoost` logit blend (`0.7 / 0.3`): `0.8440603637`
  - best weak-target meta-layer: `0.8441422314`
  - delta: `+0.0000818677`
- Artifacts:
  - `artifacts/post_tabm_catboost_weak_local_v1/weak_meta_summary.json`
  - `artifacts/post_tabm_catboost_weak_local_v1/weak_meta_grid.csv`
- Note:
  - archive as a nearly non-working strategy unless later stronger backbones create a much larger weak-target correction opportunity

## Archived Pilot

### `Family-head pilot v1`

- Reason:
  - the cheap pilot infrastructure worked
  - but it did not beat `TabM longrun v3`
- Note:
  - `Family-head v2` remains active only as a later GPU idea with a cleaner architecture hypothesis

### `TabM v4 hybrid numeric path` with hard split policy

- Reason:
  - current tested split was too aggressive and lost clearly to `TabM v3`
- Key result:
  - holdout `0.8335313590`
  - vs `TabM v3` holdout `0.8375924123`
  - delta `-0.0040610533`
- Note:
  - the broader hybrid numeric idea is not dead
  - but this exact `90 ok / 134 bad` variant is archived and not a main path anymore
