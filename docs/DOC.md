# Data Fusion 2026 — план на максимальную метрику

Ниже — рабочий пайплайн для выжимания максимального Macro ROC-AUC и быстрой итерации на сервере.

## 1) Окружение (обязательно через `uv`)

```bash
uv sync
```

## 2) EDA и гипотезы

```bash
uv run python scripts/eda.py --data-dir downloads --out-dir artifacts/eda --sample-size 250000
```

Скрипт сохраняет:
- `artifacts/eda/summary.json`
- `artifacts/eda/missing_stats.csv`
- `artifacts/eda/target_rate.csv`
- `artifacts/eda/target_corr.csv`
- `artifacts/eda/EDA_REPORT.md`

### Ключевые гипотезы
1. **Дисбаланс таргетов высокий** → нужна отдельная модель на каждый target и `scale_pos_weight`.
2. **Много NaN + выбросы** → бустинги по деревьям (LGBM/CatBoost) лучше линейных моделей.
3. **Корреляции между таргетами** → можно добавить второй слой (stacking / blending OOF).
4. **`extra_features` важны, но шумные** → нужен отбор фич (по importance, null-ratio, стабильности по fold).

## 3) Базовый сильный CV-бейзлайн (LGBM, per-target)

### Быстрый локальный прогон (для проверки)
```bash
uv run python scripts/train_lgbm_cv.py \
  --data-dir downloads \
  --out-dir artifacts/lgbm_cv_quick \
  --feature-set all \
  --folds 3 \
  --subsample-rows 200000 \
  --n-estimators 500 \
  --learning-rate 0.05 \
  --num-leaves 96
```

### Полноценный прод-запуск (сервер)
```bash
uv run python scripts/train_lgbm_cv.py \
  --data-dir downloads \
  --out-dir artifacts/lgbm_cv_full \
  --feature-set all \
  --folds 5 \
  --n-estimators 2500 \
  --learning-rate 0.02 \
  --num-leaves 192 \
  --feature-fraction 0.7 \
  --bagging-fraction 0.85 \
  --bagging-freq 1
```

Артефакты обучения:
- `metrics.json` (OOF macro AUC)
- `fold_scores.csv`
- `oof_predictions.parquet`
- `models_index.json` + `model_target_*_fold*.pkl`

## 4) Инференс / сабмит

```bash
uv run python scripts/predict_lgbm.py \
  --data-dir downloads \
  --models-dir artifacts/lgbm_cv_full \
  --feature-set all \
  --output artifacts/submission_lgbm_full.parquet
```

## 5) Как дожать ещё выше (рекомендуемая стратегия)

1. **Двухмодельный ансамбль**:
   - LGBM (all features)
   - CatBoost (main + top-N из extra по importance)
   - blend по OOF весами (поиск весов через Optuna).

2. **Per-target tuning**:
   - По 41 target отдельно подбирать `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `lambda_l1/l2`.
   - Ограничить бюджет: 40–80 trial на target.

3. **Feature pruning**:
   - убрать колонки с экстремальным missing ratio (например > 99.8%),
   - убрать константные / почти константные,
   - оставить stable features (высокая пересекаемость top importance между fold).

4. **Калибровка предиктов**:
   - isotonic/platt по OOF для каждого таргета,
   - применять к test после усреднения fold-моделей.

5. **Pseudo-labeling (осторожно)**:
   - брать только high-confidence test-предикты,
   - добавлять с небольшим весом в обучение.

---

Если вычислений мало локально, запускай «полноценный прод-запуск» выше на сервере: этот код уже готов для длинных прогонов.

## 6) Локальный smoke-result (уже проверено)

Команда:
```bash
uv run python scripts/train_lgbm_cv.py --data-dir downloads --out-dir artifacts/lgbm_cv_smoke --feature-set main --folds 2 --subsample-rows 10000 --n-estimators 60 --learning-rate 0.1 --num-leaves 32
```

Получено:
- OOF Macro ROC-AUC: **0.625265**
- Это лишь sanity-check на 10k строках и урезанных параметрах; для реального leaderboard нужен full run из раздела 3.
