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
| `models list\|delete` | Просматривать и удалять опубликованные model generations |
| `db migrations status\|apply\|rollback` | Управлять схемой PostgreSQL |

`flight serve`, `models` и `db migrations` требуют настройки PostgreSQL.
`flight serve` дополнительно требует базовый адрес Ory Hydra
Admin API;
Transformer не содержит CLI для выдачи OAuth credentials.
Их lifecycle и безопасный порядок операций описаны в
[Flight runbook](../flight-operations.md). Public remote API не является
обёрткой над local CLI: его нормативный contract находится в
[`app/contracts/flight/v5`](../../app/contracts/flight/v5/README.md).

## File commands

Обучение из Arrow file:

```bash
./.venv/bin/python ./app/main.py fit ./data/train.arrow \
  --device=cpu \
  --checkpoint-out=model_weights.pth \
  --seq-len=20 \
  --mode=relaxed \
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

Checkpoint v3 содержит model config и `feature_dim`, поэтому при prediction
model options передавать не требуется. Явно переданные `--seq-len`, `--hidden`,
`--layers`, `--dropout`, `--nhead` и `--mode` — это проверка: значение должно
совпасть с checkpoint, иначе команда завершится, например, ошибкой
`--seq-len=30 conflicts with checkpoint value 20`.

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
  --seq-len=20 \
  --mode=relaxed
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
на отдельных frames, loss schedule и early stopping определены в
[training reference](../training-runtime.md).

## GPU stress test

`gmark` выполняет синтетические optimizer steps через production
`TransformerModel`: forward, полный loss stage, backward, gradient clipping и
Adam. Входы, targets и параметры модели имеют `float32`; `--use-amp` включает
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
| `--device` | `fit`, `predict`, `fit-stream`, `predict-stream` | `cpu`, `cuda` или `auto`; `auto` выбирает CUDA при наличии | `cpu` |
| `--checkpoint-out` | `fit`, `fit-stream` | checkpoint output | `model_weights.pth` |
| `--checkpoint` | `predict`, `predict-stream` | checkpoint input | `model_weights.pth` |
| `--output` | `predict` | Arrow output file | `/tmp/preds.arrow` |
| `--pred-col` | `predict`, `predict-stream` | имя единственной prediction-колонки | `out` |
| `--metrics-out` | `fit`, `fit-stream` | metrics JSONL | не задан |
| `--max-frame-bytes` | `fit-stream`, `predict-stream` | максимальный размер одного payload | `536870912` (512 MiB) |
| `--use-amp` | все fit/predict варианты | CUDA mixed precision | выключено |
| `--plots-dir` | `plot-metrics` | каталог для SVG | `metrics_plots` |

Для совместимости сохранены aliases: `--model-name` для checkpoint input/output,
`--preds-path` для `--output` и `--metrics-name` для `--metrics-out`.
`--pred-col` доступен в обоих prediction-вариантах, `--use-amp` — во всех
четырёх fit/predict командах.

### Модель

| Аргумент | Описание | Fit default |
| --- | --- | --- |
| `--seq-len` | Длина последовательности | обязательный для `fit` и `fit-stream` |
| `--hidden` | Размер скрытого слоя | `256` |
| `--layers` | Количество Transformer layers | `5` |
| `--nhead` | Количество attention heads | `8` |
| `--dropout` | Dropout | `0.1` |
| `--mode` | Как обрабатывать `NaN` в context timesteps: `strict`, `relaxed` | `relaxed` |

Для prediction эти options по умолчанию не заданы и читаются из checkpoint.
Если option передан явно, он должен совпасть с checkpoint. Положительные
размеры и `--dropout` в диапазоне `[0, 1)` проверяются parser; `hidden` должен
делиться на `nhead`.

### Обучение

| Аргумент | Описание | По умолчанию |
| --- | --- | --- |
| `--lr` | Learning rate | `0.0005` |
| `--weight-decay` | Adam weight decay | `0.00001` |
| `--batch-size` | Размер mini-batch | `256` |
| `--epochs` | Эпохи для file fit / максимум на stdin frame / эпохи всего Flight job | `25` |
| `--loss-stage` | Максимальный этап target-aligned objective; зафиксирован в `4` | `4` |
| `--loss-schedule` | Как двигать этап loss: `none`, `epoch`, `step` | `epoch` |
| `--stage-size` | Сколько epoch/optimizer steps держать один этап | `5` |
| `--direct-loss-weights` | Шесть положительных весов `L0…L5` через запятую | `1,1,1,1,1,1` |
| `--[no-]select-best-checkpoint` | Выбирать best checkpoint только по direct losses полной epoch stage 4 | выключено |
| `--selection-min-delta` | Минимальное улучшение selection score | `0.0` |
| `--selection-patience` | Число неулучшающихся stage-4 epochs; `0` не останавливает обучение | `0` |
| `--seed` | Seed `0..4294967295` для Python, NumPy, PyTorch и CUDA | `42` |
| `--deterministic` | Включить deterministic PyTorch algorithms | выключено |

Training options доступны только у `fit` и `fit-stream`. `--use-amp` на CPU
явно отключается и для training, и для prediction; `--device=cuda` завершается
ошибкой, если CUDA недоступна. Deterministic mode может быть медленнее и может
сообщить об операции, для которой PyTorch не имеет deterministic implementation.

## Опубликованные модели

Список доступных model generations включает точный `modelRef`, owner, label,
generation и lifecycle state:

```bash
./.venv/bin/python ./app/main.py models list
```

Удаление запрашивается только по точному `modelRef`; alias команда не
принимает:

```bash
./.venv/bin/python ./app/main.py models delete \
  mdl_ead8077a4cba4455920d718532551248
```

Команда атомарно переводит доступную модель в `DELETING` и снимает alias,
если он всё ещё указывает на эту generation. Новые prediction jobs после этого
модель не видят. Maintenance-процесс Flight service удаляет каталог, затем
физически удаляет строку модели из PostgreSQL. После завершения модель исчезает
из `models list`, а повторный запрос возвращает `model generation not found`.

Минимальная identity и timestamps доступны отдельно:

```bash
./.venv/bin/python ./app/main.py models list --deleted
```

Опция показывает модели в процессе удаления (`DELETING`) и завершённые
удаления (`DELETED`). Для `DELETING` колонка `DELETED AT` остаётся пустой; она
заполняется после физической очистки и переноса identity в archive. Checkpoint
paths, hashes, contracts и metadata в archive не сохраняются. Generation
остаётся монотонным и не используется повторно.

Удаление отклоняется, пока на модель ссылается незавершённый prediction job.
Run telemetry имеет собственный lifecycle: удаление модели не отменяет её
доставку и не удаляет уже созданные документы OpenSearch.

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
