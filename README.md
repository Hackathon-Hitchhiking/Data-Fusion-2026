# Data-Fusion-2026

Для решения задачи 2 “Киберполка” предлагается два наборов данных, а также пример решения задачи.

Важные напоминания:

Расшифровка названий признаков, как и расшифровка названий целевых переменных, не предоставляется.
Решения принимаются в формате .parquet. Форма файлов, названия и порядок столбцов должны строго соответствовать таковым в sample_submit.parquet.

## Основные материалы соревнования
### train_target.parquet
https://storage.yandexcloud.net/data-fusion-2026/2%20CyberShelf/train_target.parquet
4.8MB, разметка целевых переменных для тренировочных данных

### sample_submit.parquet
https://storage.yandexcloud.net/data-fusion-2026/2%20CyberShelf/sample_submit.parquet
43.9MB, пример базового решения на основе Catboost

### baseline_catboost.ipynb
https://storage.yandexcloud.net/data-fusion-2026/2%20CyberShelf/baseline_catboost.ipynb
231.5KB, ноутбук с реализацией базового решения на основе Catboost

## Тренировочные данные 

Для обучения моделей участникам представляются 2 файла с различными группами признаков для моделирования. 
В качестве ключа для объединения данных выступает идентификатор клиента customer_id.

### train_main_features.parquet
https://storage.yandexcloud.net/data-fusion-2026/2%20CyberShelf/train_main_features.parquet
119.0MB, основная группа признаков (категорийные и числовые)

### train_extra_features.parquet
https://storage.yandexcloud.net/data-fusion-2026/2%20CyberShelf/train_extra_features.parquet
959.3MB, дополнительные признаки для решения задачи (числовые признаки)

## Тестовые данные 

Аналогичные 2 файла, необходимые для подготовки ваших предсказаний:

### test_main_features.parquet
https://storage.yandexcloud.net/data-fusion-2026/2%20CyberShelf/test_main_features.parquet
42.4MB, основная группа признаков (категорийные и числовые)

### test_extra_features.parquet
https://storage.yandexcloud.net/data-fusion-2026/2%20CyberShelf/test_extra_features.parquet
320.4MB, дополнительные признаки для решения задачи (числовые признаки)
