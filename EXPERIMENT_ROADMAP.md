# Data Fusion 2026 Task 2 Experiment Roadmap

## Goal

- Target: beat `0.87` public/private-quality level.
- Near-term target: push the current `0.8492067672` line materially closer to `0.86`.
- Optimization target: `macro ROC-AUC` over `41` labels.

## Current Snapshot

Date: `2026-03-22`

### Best public submissions

| Rank | Method | Public score | File |
|---|---|---:|---|
| 1 | `TabM + CatBoost + XGBoost v2lite final polish` | `0.8492067672` | `output/submissions/tabm_cat_xgb_v2lite_final_polish_submission.parquet` |
| 2 | `TabM + CatBoost + XGBoost v2lite greedy target-wise tri-blend` | `0.8491812192` | `output/submissions/tabm_cat_xgb_v2lite_besttri_greedy_submission.parquet` |
| 3 | `TabM + CatBoost + XGBoost v2lite restrained tri-blend` | `0.8490780420752876` | `output/submissions/tabm_cat_xgb_v2lite_tri_logit_055_010_035_restrained_submission.parquet` |
| 4 | `Three-model stack v3 (two-stage core-plus logistic, v2lite XGB)` | `0.848899328782742` | `output/submissions/three_model_stack_v3_01_two_stage_core_plus_logistic_0p05_m0p0_submission.parquet` |
| 5 | `TabM + CatBoost + XGBoost restrained target-wise tri-blend` | `0.8488617303` | `output/submissions/tabm_cat_xgb_tri_logit_055_010_035_restrained_targetwise_submission.parquet` |
| 6 | `Three-model stack v2 (two-stage core-plus ridge)` | `0.8487597` | `output/submissions/three_model_stack_v2_01_two_stage_core_plus_ridge_4p0_m0p0001_submission.parquet` |
| 7 | `TabM + CatBoost global logit blend (0.7 / 0.3)` | `0.8471400693` | `output/submissions/tabm_fs_v1_catboost_multilabel_logit30_submission.parquet` |
| 8 | `TabM longrun v3 fs_v1` | `0.8445399716` | `output/submissions/tabm_longrun_v3_fs_v1_submission.parquet` |
| 9 | `TabM longrun v3 final-only` | `0.8409426879` | `output/submissions/tabm_longrun_v3_finalonly_submission.parquet` |
| 10 | `XGBoost multi-output fs_v1` | `0.8391654278045049` | `output/submissions/xgboost_multioutput_fs_v1_submission.parquet` |

### Best current live system

- `TabM + CatBoost + XGBoost` `v2lite` final polish
  - public: `0.8492067672`
  - construction:
    - base `v2lite` greedy target-wise tri-blend
    - conservative target-wise polish from `three_model_stack_v2` and `three_model_stack_v3`
  - online uplift vs previous best:
    - `0.8491812192 -> 0.8492067672`
    - delta `+0.0000255480`
  - strongest touched targets:
    - `target_2_3`
    - `target_2_8`
    - `target_8_3`
    - `target_10_1`
    - `target_9_7`
    - `target_3_1`
  - artifacts:
    - `artifacts/final_polish_submission_fast_v1`
    - `output/submissions/tabm_cat_xgb_v2lite_final_polish_submission.parquet`

### Best current backbone

- `TabM longrun v3 fs_v1`
  - public: `0.8445399716`
  - holdout: `0.8412405192`
  - delta vs previous best public: `+0.0035972837`
  - artifacts:
    - `artifacts/feature_selection_v1`
    - `output/kaggle-output/tabm-longrun-v3-fs-v1/pull_complete/artifacts/gpu_tabm_longrun_v3_fs_v1_full_gpu`

### Strongest current offline candidate

- `TabM longrun v3 fs_v1`
  - holdout: `0.8412405192`
  - delta vs previous `TabM longrun v3` holdout: `+0.0036481069`
  - dataset: `199` main + `220` selected extra = `419` features
  - artifacts:
    - `artifacts/feature_selection_v1`
    - `output/kaggle-output/tabm-longrun-v3-fs-v1/pull_complete/artifacts/gpu_tabm_longrun_v3_fs_v1_full_gpu`
  - public: `0.8445399716`

### Strongest current facts

- `TabM longrun v3 fs_v1` remains the strongest single backbone.
- The strongest live system is now the `v2lite` final polish line at `0.8492067672` public.
- `CatBoost` is confirmed as a useful second backbone:
  - global `0.7 * TabM + 0.3 * CatBoost` reached `0.8471400693` public
- `XGBoost missing-lite` improved the XGBoost backbone materially:
  - standalone holdout `0.8362671353 -> 0.8383497098`
  - strongest gains hit weak targets such as `target_2_6`, `target_5_1`, `target_5_2`, `target_2_3`
- `XGBoost` is confirmed as a useful third backbone only in ensembles:
  - legacy restrained tri-blend public: `0.8488617303`
  - `v2lite` restrained public: `0.8490780420752876`
  - `v2lite` greedy public: `0.8491812192`
- Endgame polish is still alive, but only at micro scale:
  - `v2lite` final polish reached `0.8492067672`
  - uplift vs `v2lite` greedy: `+0.0000255480`
  - this is a real transfer, but far too small to count as a new main branch by itself
- The tested holdout-only `three-model stack` transferred again, but still did not beat the best rule-based line:
  - holdout `0.8476789914`
  - public `0.8487597`
  - `v3` with `XGBoost missing-lite`: holdout `0.8475000810`, public `0.848899328782742`
  - delta vs best public `-0.0003074384`
- This keeps `full OOF stack` alive as a research path, but again demotes `shared-75k holdout stack` as a direct submit layer.
- `TabM longrun v3` beats `TabM v1` by `+0.0112123406` holdout macro.
- `TabM longrun v3` improved `39 / 41` targets relative to `TabM v1`.
- `TabM longrun v3` is best on `40 / 41` targets against `GANDALF` and `CatBoost specialists`.
- The only target where `GANDALF` still wins is `target_2_8`.
- `TabM v4 hybrid numeric path` in the tested hard split policy lost to `v3` on holdout:
  - `v4 holdout = 0.8335313590`
  - `v3 holdout = 0.8375924123`
  - delta `-0.0040610533`
- Broad `error-corrector` and broad `object-level gate` do not work.
- Narrow `error-corrector / gate` overlays are alive, but currently only as endgame micro-uplifts.
- Post-ensemble weak-target polishing over `TabM + CatBoost` was tested and archived as almost non-working:
  - target-wise soft-blend: `+0.0000247389`
  - weak-target meta-layer: `+0.0000818677`
- `Feature-view` as a direct submit-layer over `TabM v3` did not transfer online:
  - `clip_only`, shortlist-6, `alpha=0.01`
  - public `0.8408613052`
  - delta vs best public `-0.0000813827`

## Main Path

This is the only path that currently deserves primary GPU budget.

### 1. Main backbone

- `TabM longrun v3 fs_v1`
- This is the reference single-backbone base for every new idea.
- The current production submit layer above it is the restrained `TabM + CatBoost + XGBoost` tri-blend.
- No new branch should be evaluated against old `GANDALF` as the primary baseline anymore.

### 2. Main next expensive experiment

- `RealMLP`
- Rationale:
  - `TabM v4` in the current tested form already disappointed
  - `RealMLP` is the strongest near-term challenger with a genuinely different inductive bias
  - it is more relevant now than another wide post-process or another immediate `v4b` rerun

### 3. Next production-strength branch if `RealMLP` disappoints

- `TabM-family ensemble`
- Rationale:
  - keep the best proven `TabM v3` backbone
  - test whether a second backbone complements it better than post-processing did

## Secondary Road To Best

These are confirmed or plausible branches that can still help, but they are not the main GPU priority.

### Narrow correction layer

- Status: confirmed as real, but tiny
- Current best narrow local signal is now conservative and smaller:
  - `raw` narrow error-corrector over `6` targets
  - holdout delta: `+0.0000541621`
- Conservative `clip_only` feature-view correction also exists as a micro-signal:
  - holdout delta: `+0.0000727280`
  - but it failed online as a direct submit-layer
- Use only late, after stronger backbone work is exhausted
- Best current narrow shortlist:
  - `target_2_6`
  - `target_2_4`
  - `target_9_3`
  - `target_3_1`
  - `target_9_7`
  - `target_10_1`

### `GANDALF` fallback / reference branch

- Status: confirmed secondary
- Use cases:
  - `target_2_8` fallback
  - comparison baseline
  - sanity check against overfitting branches
- Not a main backbone anymore

### `TabM + GANDALF` complementarity

- Status: real signal exists on holdout
- Current best cheap architecture blend:
  - `0.8 * TabM + 0.2 * GANDALF`
- Keep as a secondary road, not as the first expensive test

## Active Local Tests

Only ideas that still have a real chance after the latest results.

### 1. Feature-view augmentation

- Status: active local priority
- Goal:
  - find a transformed feature view that helps weak/mid targets and can be promoted into a new backbone or richer challenger
- Views to test:
  - `rank-normalized`
  - `quantile-normalized`
  - `clipped robust`
- Local protocol:
  - fit all transforms on the train fold only
  - first on weak block only
  - weak block start set:
    - `target_9_3`
    - `target_9_6`
    - `target_3_1`
    - `target_6_1`
    - `target_6_2`
    - `target_2_4`
    - `target_10_1`
    - `target_9_7`
    - `target_2_6`
  - compare:
    - `raw_only`
    - `rank_only`
    - `quantile_only`
    - `clipped_only`
    - `raw + rank`
    - `raw + quantile`
    - `raw + clipped`
    - `raw + rank + quantile`
- Success gate:
  - positive deltas on `4+` weak/mid targets
  - beats current best narrow correction-layer
- Important correction:
  - do not use feature-view as a direct `TabM v3` submit-layer anymore
  - `clip_only` probe already failed online despite a tiny holdout gain
  - require signal on a group of targets, not on one isolated label

### 2. Narrow correction layer refinement

- Status: active local secondary
- Keep only as a narrow overlay branch
- Current broad versions are rejected
- What remains alive:
  - `error-corrector`
  - `object-level gate`
- What to refine:
  - smaller target set only
  - stronger meta features / better views
  - compare against current hybrid `+0.0001561225`

### 3. `TabM v1 -> v3` decomposition follow-up

- Status: completed first pass, keep as reference
- Key outputs:
  - `artifacts/tabm_v1_vs_v3_analysis_v1/per_target_delta.csv`
  - `artifacts/tabm_v1_vs_v3_analysis_v1/family_delta.csv`
  - `artifacts/tabm_v1_vs_v3_analysis_v1/tier_delta.csv`
- Current conclusion:
  - gain is broad
  - stronger on `weak + mid` than on `strong`
  - family structure exists, but the primary effect is still backbone / numeric recipe

## Active GPU Tests

These are the only unverified GPU ideas that still deserve a slot.

### 1. `RealMLP`

- Status: top GPU priority
- Why it stays alive:
  - different tabular inductive bias
  - next strongest challenger after `TabM v4` disappointment
  - better use of GPU than immediate `v4b`
- Success gate:
  - must materially beat the old `TabM v1` holdout
  - ideally challenge `0.8375924123`

### 2. `TabM-family ensemble`

- Status: active GPU secondary
- Goal:
  - keep `TabM v3` as the main backbone
  - test whether another backbone complements it enough to move public score

### 3. `TabM v4: hybrid numeric path v4b`

- Status: secondary GPU research path
- Goal:
  - split numeric block into `piecewise_ok` and `piecewise_bad`
- Numeric partition table must include:
  - `feature`
  - `group`
  - `n_unique_train`
  - `n_unique_sampled_for_bins`
  - `effective_bin_edges_count_sampled`
  - `effective_bin_edges_count_full`
  - `iqr`
  - `std`
  - `missing_ratio`
  - `one_bin_warning`
- Partition rule:
  - `piecewise_bad` if any:
    - `effective_bin_edges_count_sampled <= 2`
    - `n_unique_sampled_for_bins <= 4`
    - `iqr <= 1e-6` on scaled values
    - explicit `one-bin` warning
  - otherwise `piecewise_ok`
- Validation rule:
  - derive the partition from train-only statistics inside each split
  - never use validation/test rows to assign `piecewise_ok` vs `piecewise_bad`
- Model change:
  - `piecewise_ok` -> `PiecewiseLinearEmbeddings(version="B")`
  - `piecewise_bad` -> simple learnable linear projection
- Keep fixed from `v3`:
  - same trunk
  - same seeds
  - same scheduler/recipe family
- Current tested result:
  - failed in the tested hard split policy with `90 ok / 134 bad`
  - holdout `0.8335313590`
- Success gate for any return:
  - `+0.0015` holdout or better over `v3`
  - preferably uplift on `20+` targets
- Important correction:
  - track both sampled and full bin behavior
  - keep full-train fallback for bin construction

### 4. GPU promotion of feature-view augmentation

- Status: conditional GPU priority
- Only run if local ablation confirms a living view
- First GPU version:
  - keep `TabM v3` recipe unchanged
  - add one extra transformed numeric block only
- Do not combine:
  - hybrid numeric path
  - multiple transformed views
  - family-head
  in one run

### 5. `Family-head v2`

- Status: second-echelon GPU idea
- This is no longer main path
- Keep only as:
  - shared trunk
  - small family-specific heads
  - no extra auxiliary loss on first run
- Use same preprocessing and numeric recipe as `v3`
- Success gate:
  - `+0.0015` strong success
  - `+0.0007` acceptable if seeds stable
- Current judgment:
  - worth one clean check later
  - not worth jumping ahead of `TabM v4` or `RealMLP`

## Specification Corrections

These corrections are applied to the current run-spec.

### `TabM v4`

- The spec is valid.
- Corrections:
  - record both sampled and full effective bin counts
  - do not rely only on warning presence
  - keep full-train bin fallback mandatory
  - keep `piecewise_bad` path simple on first run
  - derive feature groups from train-only statistics per split

### Feature-view augmentation

- The spec is valid.
- Corrections:
  - do local proof first
  - use sampled/approximate quantile fitting if exact transforms are too slow
  - do not promote a view to GPU unless it wins consistently over `raw_only`
  - require wins on multiple weak/mid targets, not on one isolated label

### `Family-head v2`

- The spec is valid as a later experiment.
- Corrections:
  - keep heads shallow
  - do not change trunk and head recipe simultaneously
  - compare only against the exact `v3` recipe
  - keep it behind `TabM v4` and `RealMLP` in GPU order

### `RealMLP`

- Add to near-term live ideas as the main challenger backbone after `TabM v4`
- Start simple:
  - no large architecture sweep
  - one clean baseline run with a stable recipe first

## Current EntryPoints

- Local:
  - `scripts/run_tabm_error_corrector.py`
  - `scripts/run_object_level_gate.py`
  - `scripts/run_feature_view_rank_quantile.py`
  - `scripts/run_family_head_pilot.py`
- Notebook builders:
  - `scripts/build_gpu_tabm_notebook.py`
  - `scripts/build_gpu_tabm_longrun_v3_notebook.py`
  - `scripts/build_gpu_tabm_final_only_notebook.py`

## Archived / Removed

- Tested dead or dominated ideas moved to:
  - `EXPERIMENT_ARCHIVE.md`
- Untested low-confidence ideas that no longer justify roadmap space were removed from the main plan.

## Experiment Log

Use this as the operational journal for confirmed results.

| Date | Run ID | Branch | Scope | Resource | Offline result | Public result | Decision |
|---|---|---|---|---|---:|---:|---|
| 2026-03-17 | gandalf_full_gpu | GANDALF | full 750k | Kaggle GPU | `0.8100` holdout | `0.8047705681` | confirmed baseline |
| 2026-03-18 | graphsmooth_95_05 | graph smoothing | post-process | local | `0.8150` holdout | `0.8006751846` | reject |
| 2026-03-18 | oracle_targetwise | oracle blend | holdout-picked | local | `0.8186` holdout | `0.7947312943` | reject |
| 2026-03-19 | adversarial_validation_v1 | train/test shift | train+test | local | AUC `0.4993` | n/a | kill as main path |
| 2026-03-19 | weighted_baseline_sanity_v1 | weighted baseline | hardest 12 | local | weak gain only | n/a | low priority |
| 2026-03-19 | cb_plain_h12_fast | CatBoost specialists | hardest 12 | local | `0.81500` vs `0.81004` | n/a | confirmed as old best specialist branch |
| 2026-03-19 | cb_submission_build_v1 | CatBoost specialists | full test-time | local | built submissions | `0.8091390663 / 0.8085486193` | confirmed public but superseded |
| 2026-03-19 | top1_recovery_ag_extract_v1 | AutoGluon hard replacement | hardest 12 | Kaggle + local | `0.85724 / 0.86200 / 0.86656` proxy | `0.8075571610 / 0.8031004260` | overfit |
| 2026-03-19 | lgbm_specialists_h12_v1 | LGBM specialists | hardest 12 | local | `0.81176` vs `0.81004` | n/a | weak family |
| 2026-03-19 | targetwise_router_catboost_public_v1 | CatBoost router | hardest 12 | public | `0.81492` validation | `0.8091478440` | small but real |
| 2026-03-19 | tabm_kaggle_full_v1 | TabM v1 | full GPU | Kaggle GPU | `0.8263800717` | `0.8287721641` | first real backbone jump |
| 2026-03-20 | tabm_longrun_v3_finalonly | TabM longrun v3 | final-only full-train | Kaggle GPU | precursor `0.8375924123` | `0.8409426879` | new main backbone |
| 2026-03-20 | tabm_error_corrector_v2_longrun | broad error-corrector | weak block | local | subset worsened | n/a | archive broad version |
| 2026-03-20 | object_level_gate_v2_spec | broad spec gate | weak block | local | subset worsened | n/a | archive broad version |
| 2026-03-20 | tabm_error_gate_hybrid_scan_v1 | narrow correction scan | selected weak targets | local | `+0.0001561225` full macro | n/a | keep as endgame overlay |
| 2026-03-21 | tabm_catboost_targetwise_softblend_v1 | target-wise soft-blend | weak shortlist | local | `+0.0000247389` over global blend | n/a | archive as almost non-working |
| 2026-03-21 | tabm_catboost_weakmeta_v1 | weak-target meta-layer | weak shortlist | local | `+0.0000818677` over global blend | n/a | archive as almost non-working |
| 2026-03-21 | tabm_catboost_blend_v1 | `TabM + CatBoost` global blend | full exact holdout | local + public | `0.8440603637` holdout | `0.8471400693` | confirmed second backbone |
| 2026-03-21 | xgboost_multioutput_fs_v1 | `XGBoost` multi-output | full exact holdout | Kaggle GPU + public | `0.8362671353` holdout | `0.8391654278045049` | weak standalone, keep only as ensemble component |
| 2026-03-21 | tabm_cat_xgb_tri_restrained_v1 | restrained tri-blend | full exact holdout | local + public | `0.8467476918` holdout | `0.8488617303` | previous best public line |
| 2026-03-21 | three_model_meta_stack_v2_01 | two-stage three-model stack | shared 75k OOS holdout | local + public | `0.8476789914` holdout | `0.8487597` | near-best but not enough to replace restrained tri-blend |
| 2026-03-22 | xgboost_multioutput_fs_v2_missing_lite | XGBoost missing-lite | full exact holdout | Kaggle GPU + local | `0.8383497098` holdout | n/a | better than XGB fs_v1 by `+0.0020825746`, useful as stronger ensemble donor |
| 2026-03-22 | tabm_cat_xgb_v2lite_restrained | restrained tri-blend with XGB missing-lite | full exact holdout | local + public | `0.8469056933` holdout | `0.8490780420752876` | clean positive transfer over previous best line |
| 2026-03-22 | tabm_cat_xgb_v2lite_greedy | greedy target-wise tri-blend with XGB missing-lite | full exact holdout | local + public | `0.8468836986` holdout | `0.8491812192` | previous best public line despite slightly weaker holdout than restrained |
| 2026-03-22 | three_model_meta_stack_v3_01 | two-stage three-model stack with XGB missing-lite | shared 75k OOS holdout | local + public | `0.8475000810` holdout | `0.848899328782742` | positive transfer, still below best rule-based line |
| 2026-03-22 | tabm_cat_xgb_v2lite_final_polish | conservative endgame polish over best v2lite rule-based line | public-tested polish layer | public only | n/a | `0.8492067672` | current best public line; validates that stack polish still transfers, but only at micro scale |

## Reference Files

- main roadmap:
  - `EXPERIMENT_ROADMAP.md`
- archived ideas:
  - `EXPERIMENT_ARCHIVE.md`
- public submission index:
  - `output/submissions/README.md`
- best public submission:
  - `output/submissions/tabm_cat_xgb_v2lite_final_polish_submission.parquet`
