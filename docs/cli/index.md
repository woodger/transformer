# Справочник CLI

> Type: Reference. Локальные команды Transformer, их arguments, output и
> границы поведения.

Все команды выполняются из корня working copy через project interpreter:

```bash
./.venv/bin/python ./app/main.py <command> [args] [options]
```

CLI help — публичный contract для точного набора options и defaults:

```bash
./.venv/bin/python ./app/main.py --help
./.venv/bin/python ./app/main.py <command> --help
./.venv/bin/python ./app/main.py --version
```

`--version` выводит только статическую версию package и не инициализирует
Torch, CUDA или worker runtime. Версии ML runtime публикует worker
`inspect` через Flight capabilities.

## Карта команд

| Команда | Назначение |
| --- | --- |
| `fit INPUT` | Обучить Transformer на Arrow IPC file |
| `predict INPUT` | Выполнить prediction из checkpoint в Arrow IPC file |
| `fit-stream` | Читать framed Arrow payloads из stdin и обучать модель |
| `predict-stream` | Читать framed Arrow payloads из stdin и писать predictions в stdout |
| `gmark` | Нагрузить CUDA синтетическим training с контролем integrity и температуры |
| `plot-metrics METRICS_FILE` | Построить SVG-графики по metrics JSONL |
| `flight serve` | Запустить durable Arrow Flight job service |
| `auth tokens issue\|list\|revoke` | Управлять API access tokens в PostgreSQL |
| `models list\|delete` | Просматривать и удалять опубликованные model generations |
| `db migrations status\|apply\|rollback` | Управлять схемой PostgreSQL |

`flight serve`, `auth tokens`, `models` и `db migrations` требуют настройки
PostgreSQL. Lifecycle `auth tokens` описан в
[руководстве по управлению API access tokens](../operations/api-access-tokens.md),
published models — в
[руководстве по управлению моделями](../operations/published-models.md), а
schema — в [руководстве по migrations](../operations/database-migrations.md).
Service operations находятся в [Flight runbook](../operations/flight-service.md).
Public remote API не является обёрткой над local CLI: его нормативный contract
находится в [`app/contracts/flight/v11`](../../app/contracts/flight/v11/README.md).

## File commands

Обучение из Arrow file:

```bash
./.venv/bin/python ./app/main.py fit ./data/train.arrow \
  --device=cpu \
  --checkpoint-out=model_weights.pth \
  --model-contract=./data/model-contract.json \
  --epochs=25 \
  --batch-size=256
```

Prediction из checkpoint:

```bash
./.venv/bin/python ./app/main.py predict ./data/test.arrow \
  --device=cpu \
  --checkpoint=model_weights.pth \
  --output=/tmp/preds.arrow \
  --pred-col=out
```

Checkpoint v6 содержит полный ModelContract, включая model config, target
layout и Objective. Поэтому prediction не принимает отдельные model options и
восстанавливает точную конфигурацию из checkpoint.

Предыдущие checkpoint formats и raw `state_dict` не интерпретируются. Полный
текущий формат определяет [training reference](../training-runtime.md).

## Stream commands

`fit-stream` и `predict-stream` не принимают путь к файлу данных: лишний
positional argument отклоняется parser. Оба читают последовательность framed
Arrow IPC payloads из stdin. Формат frame, schema, limits и правила stdout
определены в [локальном Arrow и stream contract](../local-arrow-protocol.md).

`predict-stream` загружает checkpoint один раз на первом непустом frame и
пишет один result frame на каждый input frame, включая пустой.

Пример standalone training:

```bash
./.venv/bin/python ./app/main.py fit-stream \
  --device=cpu \
  --checkpoint-out=model_weights.pth \
  --model-contract=./data/model-contract.json
```

Пример prediction:

```bash
./.venv/bin/python ./app/main.py predict-stream \
  --device=cpu \
  --checkpoint=model_weights.pth \
  --pred-col=out \
  > /tmp/predictions.framed
```

В `predict-stream` stdout — бинарный результат; diagnostics идут в stderr.
`fit-stream` печатает training progress в текстовый stdout. Semantics обучения
на отдельных frames, objective и early stopping определены в
[training reference](../training-runtime.md).

## GPU stress test

`gmark` выполняет синтетические optimizer steps через production
`TransformerModel`: forward, consumer-neutral SmoothL1 objective, backward,
gradient clipping и Adam. Входы, targets и параметры модели имеют `float32`; `--use-amp` включает
тот же CUDA autocast и `GradScaler`, что и production worker. Команда проверяет
loss, gradient norm, model parameters и optimizer state на `NaN` и `Inf`.
Динамические метрики температуры, utilization и power читаются через системный
`nvidia-smi`.

При AMP начальные gradients могут переполниться на высоком dynamic scale.
`gmark` допускает штатный backoff `GradScaler`, продолжает после первого
успешного optimizer update и завершает тест ошибкой, если scale не
стабилизируется.

Короткая проверка:

```bash
./.venv/bin/python ./app/main.py gmark \
  --duration=60 \
  --use-amp \
  --max-temperature=80
```

Не запускайте `gmark` одновременно с training или prediction jobs на том же
GPU: команда намеренно потребляет compute capacity. Дополнительный VRAM ballast
по умолчанию отключён; его можно включить через `--memory-fraction`. Отсутствие
показания температуры считается ошибкой, потому что порог нельзя гарантировать.
`--max-temperature=0` отключает эту защиту и допустим только при внешнем
мониторинге.

| Аргумент | Описание | По умолчанию |
| --- | --- | --- |
| `--duration SECONDS` | Продолжительность после warm-up | `60` |
| `--device INDEX` | Логический индекс CUDA-устройства | `0` |
| `--seq-len LENGTH` | Длина входной последовательности | `10` |
| `--feature-dim COUNT` | Число features в timestep | `891` |
| `--batch-size COUNT` | Число строк в optimizer step | `256` |
| `--hidden SIZE` | Hidden dimension модели | `256` |
| `--layers COUNT` | Число Transformer encoder layers | `5` |
| `--nhead COUNT` | Число attention heads | `8` |
| `--use-amp` | Production CUDA autocast и `GradScaler` | выключено |
| `--memory-fraction FRACTION` | Доля свободной после warm-up VRAM, `0..0.9`; `0` отключает ballast | `0` |
| `--max-temperature CELSIUS` | Температурный порог; `0` отключает | `80` |
| `--status-interval SECONDS` | Интервал статуса и integrity check | `2` |
| `--warmup-steps COUNT` | Optimizer steps до замера | `3` |
| `--seed SEED` | Seed параметров модели и synthetic data | `42` |

Коды завершения:

| Код | Значение |
| --- | --- |
| `0` | Продолжительность завершена, integrity checks пройдены |
| `1` | Ошибка CUDA, конфигурации, VRAM, monitoring или integrity |
| `3` | Нагрузка остановлена температурным порогом |
| `130` | Нагрузка остановлена пользователем через `Ctrl+C` |

Датчики VRAM и VRM могут отсутствовать даже при доступном core-temperature
sensor. Поэтому температурный порог по умолчанию защищает по температуре GPU
core, но не является полной гарантией температур VRAM и power circuitry.

## Параметры

Точный набор параметров всегда показывает `COMMAND --help`; таблицы ниже
суммируют стабильные options.

### Runtime и output

| Аргумент | Команды | Описание | По умолчанию |
| --- | --- | --- | --- |
| `--device` | `fit`, `predict`, `fit-stream`, `predict-stream` | `cpu`, `gpu` или `auto`; `auto` выбирает GPU при наличии | `cpu` |
| `--checkpoint-out` | `fit`, `fit-stream` | checkpoint output | `model_weights.pth` |
| `--checkpoint` | `predict`, `predict-stream` | checkpoint input | `model_weights.pth` |
| `--output` | `predict` | Arrow output file | `/tmp/preds.arrow` |
| `--pred-col` | `predict`, `predict-stream` | имя единственной prediction-колонки | `out` |
| `--metrics-out` | `fit`, `fit-stream` | metrics JSONL | не задан |
| `--model-contract` | `fit`, `fit-stream` | self-contained ModelContract JSON file | обязательный |
| `--max-frame-bytes` | `fit-stream`, `predict-stream` | максимальный размер одного payload | `536870912` (512 MiB) |
| `--use-amp` | все fit/predict варианты | GPU mixed precision | выключено |
| `--plots-dir` | `plot-metrics` | каталог для SVG | `metrics_plots` |

Для совместимости сохранены aliases: `--model-name` для checkpoint input/output,
`--preds-path` для `--output` и `--metrics-name` для `--metrics-out`.
`--pred-col` доступен в обоих prediction-вариантах, `--use-amp` — во всех
четырёх fit/predict командах.

### Модель

`fit` и `fit-stream` получают architecture, tensor geometry, ordered target
slots и Objective только из обязательного `--model-contract`. Документ
валидируется до построения модели. `predict` и `predict-stream` используют его
точную checkpoint-owned копию. Отдельных CLI options для `seqLen`, `hidden`,
`layers`, `nhead`, `dropout`, `mode`, targets или direct operators нет.

### Обучение

| Аргумент | Описание | По умолчанию |
| --- | --- | --- |
| `--lr` | Learning rate | `0.0005` |
| `--weight-decay` | Adam weight decay | `0.00001` |
| `--batch-size` | Размер mini-batch | `256` |
| `--epochs` | Эпохи для file fit / максимум на stdin frame | `25` |
| `--[no-]select-best-checkpoint` | Выбирать best checkpoint по direct losses завершённой epoch | выключено |
| `--selection-min-delta` | Минимальное улучшение selection score | `0.0` |
| `--selection-patience` | Число неулучшающихся epochs; `0` не останавливает обучение | `0` |
| `--seed` | Seed `0..4294967295` для Python, NumPy, PyTorch и CUDA | `42` |
| `--deterministic` | Включить deterministic PyTorch algorithms | выключено |

Training options доступны только у `fit` и `fit-stream`. `--use-amp` на CPU
явно отключается и для training, и для prediction; `--device=gpu` завершается
ошибкой, если GPU недоступен. Значение `--device=cuda` не поддерживается.
Deterministic mode может быть медленнее и может
сообщить об операции, для которой PyTorch не имеет deterministic implementation.

## Опубликованные модели

Точные administrative commands:

```bash
./.venv/bin/python ./app/main.py models list
./.venv/bin/python ./app/main.py models list --deleted
./.venv/bin/python ./app/main.py models delete \
  mdl_ead8077a4cba4455920d718532551248
```

`models delete` принимает только точный `MODEL_REF`, а не alias. Полный
необратимый lifecycle, состояния и порядок проверки описывает
[`руководство по управлению опубликованными моделями`](../operations/published-models.md).

## Пути и запись артефактов

Относительные checkpoint paths, `--metrics-out` и `METRICS_FILE` для
`plot-metrics` разрешаются от каталога `models/` в корне проекта независимо от
текущего working directory. Абсолютные пути разрешены. Относительный путь,
который после нормализации выходит из `models/` через `..`, отклоняется.

Checkpoint и file prediction записываются через временный файл рядом с целью и
атомарный replace после успешной записи; существующий target заменяется только
в конце. Metrics JSONL аналогично атомарно очищается в начале нового запуска,
после чего строки эпох дописываются. `--output` и `--plots-dir` не привязаны к
`models/`.

## Metrics charts

`--metrics-out` записывает одну JSONL-строку на training pass. Per-target
метрики, selection score и пример `plot-metrics` описаны в
[training reference](../training-runtime.md).
