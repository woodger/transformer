# Transformer

Python-проект для обучения и инференса PyTorch Transformer на датасетах в
формате Apache Arrow.

Проект умеет работать в двух режимах:

- читать готовый Arrow-файл с диска (`fit`, `predict`)
- принимать поток micro-batch Arrow payloads через stdin (`fit-stream`,
  `predict-stream`)

Checkpoint v2 сохраняет веса, model/train config и размер входной фичи
`feature_dim`. Поэтому `predict` и `predict-stream` восстанавливают архитектуру
и проверяют вход по metadata checkpoint.

## Требования

- Python 3.11+
- PyTorch
- NumPy
- PyArrow
- PostgreSQL
- SQLAlchemy 2.x
- Psycopg 3
- Alembic
- python-dotenv
- CUDA опционально

Установка зависимостей:

```bash
pip install torch numpy pyarrow SQLAlchemy 'psycopg[binary]' alembic python-dotenv pytest
```

Этого достаточно для обычного запуска на CPU и NVIDIA GPU. При работающем
NVIDIA driver отдельно устанавливать CUDA Toolkit, cuDNN или NCCL через
system package manager не нужно.

Проверка CUDA:

```bash
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

Подробная production-настройка Flight service находится в
[`docs/flight-operations.md`](docs/flight-operations.md).
Установка и управление процессом на Fedora через systemd описаны в
[`docs/deployment/systemd.md`](docs/deployment/systemd.md).

## Remote Arrow Flight service

Проект содержит single-instance Arrow Flight v1 job service поверх существующих
`fit-stream`/`predict-stream`. Нормативный wire contract находится в
[`contracts/flight/v1`](contracts/flight/v1/README.md), а конфигурация, TLS/mTLS,
запуск, recovery и retention — в
[`docs/flight-operations.md`](docs/flight-operations.md).

```bash
python ./app/main.py db migrations apply
python ./app/main.py auth tokens issue --subject=inventory-production

python ./app/main.py flight serve \
  --host=0.0.0.0 \
  --tls-cert-file=/run/secrets/transformer/tls.crt \
  --tls-key-file=/run/secrets/transformer/tls.key
```

Параметры PostgreSQL читаются из `.env` или окружения: `POSTGRES_HOST`,
`POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER` и `POSTGRES_PASSWORD`.
Bearer authentication требуется при любом transport; токены выпускаются и
отзываются через `auth tokens`, а Flight service держит активные credentials в
оперативной памяти и обновляет cache через PostgreSQL `LISTEN/NOTIFY`. Без TLS
сервер запускается только с явным `--allow-plaintext`. Один DoPut остаётся одним
semantic stream frame, checkpoint принадлежит Transformer, а клиент получает
только непрозрачный `modelRef`. DoExchange и PollFlightInfo в v1 не входят.

## CLI

Запуск выполняется из корня проекта. CLI использует обязательные subcommands с
собственными positional arguments и наборами options:

```text
python ./app/main.py fit INPUT [options]
python ./app/main.py predict INPUT [options]
python ./app/main.py fit-stream [options]
python ./app/main.py predict-stream [options]
python ./app/main.py flight serve [options]
python ./app/main.py plot-metrics METRICS_FILE [options]
python ./app/main.py auth tokens issue|list|revoke [options]
python ./app/main.py db migrations status|apply|rollback
```

Общий help показывает только global options и список команд. Command-specific
help содержит применимые к выбранной команде аргументы, их defaults и, где это
полезно, отдельный блок с примерами:

```bash
python ./app/main.py --help
python ./app/main.py fit --help
python ./app/main.py predict-stream --help
python ./app/main.py flight serve --help
```

Версию можно посмотреть через `--version` или его короткую форму `-v`:

```bash
python ./app/main.py --version
```

Доступные действия:

- `fit` — обучить модель на Arrow-файле
- `predict` — загрузить модель и сохранить предсказания в Arrow-файл
- `fit-stream` — читать framed Arrow payloads из stdin и обучать модель batch за batch
- `predict-stream` — читать framed Arrow payloads из stdin и писать framed
  predictions в stdout
- `flight serve` — запустить durable Arrow Flight job service
- `auth tokens issue|list|revoke` — управлять API access tokens в PostgreSQL
- `db migrations status|apply|rollback` — управлять схемой PostgreSQL
- `plot-metrics` — построить SVG-графики из metrics JSONL

## Обучение из файла

```bash
python ./app/main.py fit ./data/train.arrow \
  --device=cpu \
  --checkpoint-out=model_weights.pth \
  --seq-len=20 \
  --mode=relaxed \
  --epochs=25 \
  --batch-size=256
```

## Предсказание из файла

```bash
python ./app/main.py predict ./data/test.arrow \
  --device=cpu \
  --checkpoint=model_weights.pth \
  --output=/tmp/preds.arrow \
  --pred-col=out
```

Для checkpoint v2 model options при prediction можно не передавать. Явно
переданные `--seq-len`, `--hidden`, `--layers`, `--dropout`, `--nhead` и
`--mode` рассматриваются как проверка: значение должно совпасть с checkpoint,
иначе запуск завершится ошибкой вида
`--seq-len=30 conflicts with checkpoint value 20`.

Checkpoint v1 и raw legacy `state_dict` по-прежнему читаются. Если в legacy
checkpoint нет model config, `--seq-len` обязателен, а отличающиеся от defaults
параметры архитектуры нужно передать вручную.

## Потоковое обучение

`fit-stream` не принимает путь к файлу данных: лишний positional argument
отклоняется parser. Режим читает stdin до terminator или EOF:

```bash
python ./app/main.py fit-stream \
  --device=cpu \
  --checkpoint-out=model_weights.pth \
  --seq-len=20 \
  --mode=relaxed
```

Transformer обучается на каждом непустом входящем frame и сохраняет модель
после terminator или EOF. Пустые frames пропускаются; если непустых frames не
было, запуск завершается ошибкой и модель не сохраняется.

## Параметры

Точный набор параметров показывает `COMMAND --help`; следующие таблицы —
краткая сводка.

### Runtime и output

| Аргумент | Команды | Описание | По умолчанию |
| --- | --- | --- | --- |
| `--device` | все, кроме `plot-metrics` | `cpu`, `cuda` или `auto`; `auto` выбирает CUDA при наличии | `cpu` |
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

Для prediction эти options являются legacy/checkpoint overrides и по умолчанию
не заданы. Положительные размеры и `--dropout` в диапазоне `[0, 1)` проверяются
parser; `hidden` должен делиться на `nhead`.

### Обучение

| Аргумент | Описание | По умолчанию |
| --- | --- | --- |
| `--lr` | Learning rate | `0.0005` |
| `--weight-decay` | Adam weight decay | `0.00001` |
| `--batch-size` | Размер mini-batch | `256` |
| `--epochs` | Эпохи для file fit / максимум на stdin frame / эпохи всего Flight job | `25` |
| `--loss-stage` | Максимальный этап loss: `1..4` | `4` |
| `--loss-schedule` | Как двигать этап loss: `none`, `epoch`, `step` | `epoch` |
| `--stage-size` | Сколько epoch/optimizer steps держать один этап | `5` |
| `--patience` | Early stopping patience | `5` |
| `--monitor` | Training-pass monitor: `loss`, `ret_mae`, `ret_mae_skill` | `ret_mae_skill` |
| `--monitor-min-improvement` | Доля улучшения `[0, 1)` относительно zero-return baseline | `0.0` |
| `--seed` | Seed `0..4294967295` для Python, NumPy, PyTorch и CUDA | `42` |
| `--deterministic` | Включить deterministic PyTorch algorithms | выключено |
| `--[no-]save-best-checkpoint` | Сохранить лучший checkpoint по monitor вместо последних весов | включено |

Training options доступны только у `fit` и `fit-stream`. `--use-amp` на CPU
явно отключается и для training, и для prediction; `--device=cuda` завершается
ошибкой, если CUDA недоступна.
Deterministic mode может быть медленнее и может сообщить об операции, для
которой PyTorch не имеет deterministic implementation.

## Пути и запись артефактов

Относительные checkpoint paths, `--metrics-out` и `METRICS_FILE` для
`plot-metrics` разрешаются от каталога `models/` в корне проекта, независимо от
текущего рабочего каталога. Абсолютные пути разрешены. Относительный путь,
который после нормализации выходит из `models/` через `..`, отклоняется.

Checkpoint и file prediction записываются через временный файл рядом с целью и
атомарный replace после успешной записи; существующий target заменяется только
в конце. Metrics JSONL аналогично атомарно очищается в начале нового запуска,
после чего строки эпох дописываются. `--output` и `--plots-dir` не
привязаны к `models/`.

## Checkpoint contract

Новый формат `transformer-checkpoint-v2` содержит `state_dict`, версию
приложения, model config, train config, `data_schema` и metadata выбора best
checkpoint. Model config включает `seq_len`, `hidden`, `layers`, `dropout`,
`nhead`, `context_mode`, `out_dim` и фактический `feature_dim` training input.
`data_schema` фиксирует имена и ширину `src`/`tgt`, допустимые element types,
`float32` tensor dtype, исходный и подготовленный model input dimension,
отсутствие normalization и missing policy. При prediction входной
`feature_dim` должен совпасть с сохранённым.

`fit` и `fit-stream` всегда создают новую модель; продолжение обучения из
checkpoint не реализовано. `--checkpoint-out` задаёт только конечную цель:
существующий checkpoint остаётся нетронутым во время обучения и атомарно
заменяется лишь при успешном save.

Loader принимает v2, предыдущий wrapped v1 и raw legacy `state_dict`;
неизвестный wrapped format отклоняется. Правила CLI overrides для v2 и ручных
legacy model options описаны выше в разделе «Предсказание из файла» и одинаковы
для `predict-stream`.

## Обучение, loss и early stopping

Критерий early stopping и выбора checkpoint задаётся `--monitor`; требуемая для
MAE monitors доля улучшения относительно zero-return baseline —
`--monitor-min-improvement`. Их defaults определены в `app/config.py`:

```python
TRAIN_MONITOR = "ret_mae_skill"          # loss | ret_mae | ret_mae_skill
TRAIN_MONITOR_MIN_IMPROVEMENT = 0.0      # 0.01 означает лучше baseline на 1%
SAVE_BEST_CHECKPOINT = True
```

Для `ret_mae_skill` формула такая:

```text
ret_mae_skill = ret_mae / ret_mae_baseline
```

Значение `< 1.0` означает, что модель лучше нулевого прогноза `meanR=0`.
Monitor вычисляется по тому же training pass, на котором обновлялись веса;
отдельного validation split этот CLI не создаёт. Early stopping начинает
останавливать только на максимальном настроенном loss stage, сбрасывает свой
счётчик при смене stage и отключён при `--patience=0`. Best checkpoint
обновляется только если baseline пройден, monitor конечен и улучшился; если
baseline ни разу не пройден, сохраняются текущие веса последней эпохи.

В standalone `fit-stream`, читающем stdin, каждый непустой Arrow frame
обучается отдельным циклом `epoch=1..--epochs` до срабатывания `--patience`.
Веса модели при этом не сбрасываются между frames; optimizer step также
остаётся глобальным, а per-frame early stopping начинается заново. Flight fit
использует другой внутренний режим этой команды: каждая job-wide эпоха читает
все sealed payloads из durable spool по ordinal, с едиными loss schedule,
optimizer, checkpoint selection и early stopping на весь job.

Loss stage соответствует следующим компонентам (точные формулы находятся в
`docs/losses.md`):

| Stage | Компоненты |
| --- | --- |
| `1` | Gaussian NLL для return |
| `2` | stage 1 + BCE для TP/SL logits |
| `3` | stage 2 + Bayesian EV/risk |
| `4` | stage 3 + log-volatility loss |

Stage 1 NLL может быть отрицательным — это допустимое значение Gaussian NLL,
а не признак сломанного обучения.

Loss schedule можно зафиксировать вручную:

```bash
python ./app/main.py fit-stream \
  --seq-len=20 \
  --loss-schedule=none \
  --loss-stage=1
```

Или включить автоматический curriculum по эпохам:

```bash
python ./app/main.py fit-stream \
  --seq-len=20 \
  --loss-stage=4 \
  --loss-schedule=epoch \
  --stage-size=4
```

Для schedule по optimizer steps:

```bash
python ./app/main.py fit-stream \
  --seq-len=20 \
  --loss-stage=4 \
  --loss-schedule=step \
  --stage-size=100
```

При step schedule счётчик проверяется перед каждым optimizer step, поэтому
активный stage может смениться посреди epoch.

## Контекстные пропуски

`src` может содержать `NaN` в отдельных фичах контекстного timestep. Режим
`--mode` задаёт, как такие timesteps попадают в Transformer:

- `strict` — timestep маскируется, если хотя бы одна фича `NaN`; значения
  `NaN` заменяются на `0.0`, per-feature flags не добавляются.
- `relaxed` — timestep маскируется, только если все фичи `NaN`; значения `NaN`
  заменяются на `0.0`, а к каждой фиче добавляется бинарный missing-флаг.
  Поэтому `0` остаётся численным placeholder, а информация о частичном
  пропуске не теряется.

Текущий default — `relaxed`. При таком контракте producer должен передавать
`NaN` для отсутствующих context candles; Transformer строит mask и, в relaxed
mode, missing-флаги из исходных `NaN` до любых tensor ops, и только затем
заменяет `NaN -> 0`.

Trading head получает фактический последний незамаскированный timestep, а не
просто число валидных токенов: ведущие, внутренние и хвостовые пропуски поэтому
обрабатываются корректно. Для полностью пустой последовательности первый
timestep с заполненными нулями значениями временно остаётся unmasked как
безопасный placeholder; relaxed missing-флаги при этом сохраняются. Это не даёт
attention создать non-finite значения.

## Метрики обучения

`fit` и `fit-stream` один раз печатают конфигурацию запуска, а затем компактную
summary-строку для каждого epoch. Standalone `fit-stream` дополнительно пишет
номер входного frame; Flight fit пишет одну агрегированную строку на job-wide
эпоху без `frame`. Loss schedule продвигается по выбранному `--loss-schedule`:

- `none` — всегда используется `--loss-stage`
- `epoch` — stage считается от epoch внутри текущего standalone frame или
  всего Flight job
- `step` — stage считается от глобального optimizer step и не сбрасывается
  между frames

```text
frame=1 epoch=2 monitor_value=3.82703 loss=-3.149016 mae=0.0225603 baseline=0.00589499 skill=3.82703x status=WORSE sigma=0.0231593 grad=476.013 rows=67249 batches=263 time=181.7s stage=1/4
```

Summary показывает основной результат эпохи, сравнение с baseline, среднюю
`sigmaR`, gradient norm до clipping, объём данных, время и активный loss stage.
`status=BETTER` означает `skill < 1.0`, `status=WORSE` — что baseline пока
лучше модели. Для non-finite skill выводятся `skill=n/a status=N/A`.

Полный набор метрик доступен в JSONL:

- `loss` — итоговый loss после всех весов компонентов
- `batch_size`, `loss_schedule`, `stage_size`, `max_loss_stage`, `hidden`,
  `layers`, `seq_len`, `device` — параметры запуска, записываются в каждую
  строку JSONL
- `loss_ret`, `loss_prob`, `loss_ev`, `loss_vol` — компоненты loss
- `sigma_min`, `sigma_p05`, `sigma_mean` — статистика предсказанного `sigmaR`
  для диагностики Gaussian NLL
- `ret_mae`, `ret_rmse` — ошибка прогноза `meanR` против target `meanR`
- `ret_mae_baseline` — MAE нулевого прогноза `meanR=0`; полезно сравнивать с
  `ret_mae`, чтобы видеть, лучше ли модель простой нулевой гипотезы
- `ret_mae_skill` — отношение `ret_mae / ret_mae_baseline`; меньше `1.0`
  означает лучше baseline
- `ret_mae_improvement` — `1 - ret_mae_skill`
- `monitor_value`, `best_monitor`, `baseline_passed`, `checkpoint_best` —
  состояние критерия early stopping и выбора checkpoint
- `grad_norm` — gradient norm до clipping
- `rows`, `batches` — объём данных в проходе
- `nan_ratio` — доля NaN во входном `src`
- `masked_token_ratio` — доля timesteps, скрытых от attention текущим
  `--mode`
- `complete_token_ratio` — доля timesteps без `NaN`
- `partial_token_ratio` — доля timesteps с частью заполненных фичей и `NaN`
- `empty_token_ratio` — доля timesteps, где все фичи `NaN`
- `step` — глобальный номер optimizer step к концу строки метрик
- `lr` — текущий learning rate
- `loss_stage` — активный этап функции потерь
- `elapsed_ms` — время training pass

Чтобы сохранять полный набор метрик для каждой эпохи в JSONL:

```bash
python ./app/main.py fit ./data/train.arrow \
  --seq-len=20 \
  --metrics-out=train.jsonl
```

Для `fit-stream` используется тот же аргумент:

```bash
python ./app/main.py fit-stream \
  --seq-len=20 \
  --metrics-out=train-stream.jsonl
```

Каждая строка — один JSON object с полным набором числовых полей, параметрами
запуска и контекстом `epoch`; standalone stream также добавляет `frame`.
Non-finite значения сериализуются как JSON `null`.

Построить SVG-графики по JSONL:

```bash
python ./app/main.py plot-metrics train.jsonl \
  --plots-dir=./metrics/plots
```

`plot-metrics` создаёт отдельные SVG-файлы для `loss`, компонентов loss,
`sigma_min`, `sigma_p05`, `sigma_mean`, `grad_norm`, `nan_ratio`, token ratios,
`rows`, `batches`, `step`, `lr`, `loss_stage` и `elapsed_ms`.
Output directory создаётся автоматически; существующие одноимённые SVG
перезаписываются. Невалидная JSON-строка в `METRICS_FILE` прерывает команду.

## Потоковое предсказание

`predict-stream` не принимает путь к файлу данных. Он читает framed Arrow
payloads из stdin, загружает модель один раз на первом непустом frame и пишет
по одному framed Arrow result на каждый input frame, включая пустой:

```bash
python ./app/main.py predict-stream \
  --device=cpu \
  --checkpoint=model_weights.pth \
  --pred-col=out
```

stdout в этом режиме является бинарным протоколом результата. Диагностические
сообщения пишутся в stderr, поэтому stdout можно безопасно
перенаправлять в файл или следующему процессу:

```bash
python ./app/main.py predict-stream \
  --checkpoint=model_weights.pth \
  > /tmp/predictions.framed
```

## Arrow contract

Каждый file input и каждый payload stream — самостоятельный Arrow IPC file.
Поддерживаются `List`, `LargeList` и `FixedSizeList`, но child type должен быть
строго `float32` или `float64`:

```text
src: list<float32|float64>  # flattened [seq_len * feature_dim]
tgt: list<float32|float64>  # width 6, required for training
```

В обеих колонках запрещены Arrow null rows и null elements; длина list должна
быть одинаковой у всех строк. `src` допускает IEEE `NaN` для пропусков, но
отклоняет `+inf` и `-inf`. `tgt` должен быть полностью finite, иметь ровно шесть
значений, неотрицательный `tgt[4]` и `tgt[5]` в диапазоне `[0, 1]`. У непустого
input ширина `src` должна быть больше нуля.

Текущий loss использует target positions так:

| Позиция | Назначение |
| --- | --- |
| `tgt[0]` | target return для Gaussian NLL |
| `tgt[4]` | target next volatility |
| `tgt[5]` | hit probability для TP/SL BCE |
| `tgt[1:4]` | присутствуют в формате, но текущим loss не используются |

После валидации `float64` и `float32` input преобразуется в PyTorch
`float32`; finite `float64`, который выходит за диапазон `float32`,
отклоняется до cast. `src` reshaped следующим образом:

```text
[rows, seq_len * feature_dim] -> [rows, seq_len, feature_dim]
```

Ширина `src` должна делиться на `--seq-len`. В stream feature dimension не
может меняться между frames; при prediction с checkpoint v2 он также должен
совпасть с сохранённым `feature_dim`.

Для prediction `tgt` не требуется. File и stream prediction output содержит
только одну колонку `--pred-col` типа `list<float32>`; input columns в output не
копируются. Prediction runtime проверяет форму `[rows, 6]` и finite значения;
row count и исходный порядок строк сохраняются. `predict` отклоняет одинаковый
canonical input/output path, чтобы не уничтожить входной файл.

Порядок шести выходов:

```text
[meanR, sigmaR, logitTP, logitSL, volNext, logitHit]
```

`sigmaR` и `volNext` положительны, позиции TP/SL/hit являются raw logits.
Пустой file input для `predict` создаёт типизированный пустой output;
`fit` отклоняет training input без строк.

## Framed stdin protocol

`fit-stream` и `predict-stream` читают последовательность payloads из stdin:

```text
8 bytes unsigned big-endian payload length
Arrow file payload
8 bytes unsigned big-endian payload length
Arrow file payload
...
0 as 8-byte unsigned big-endian length, or clean EOF
```

Каждый payload — самостоятельный Arrow IPC file с колонкой `src`. Для
`fit-stream` также нужна колонка `tgt`; для `predict-stream` `tgt` не
требуется. Declared length `0` является terminator и не Arrow frame. Чистый EOF
между frames также нормально завершает stream. Частичный 8-byte header и EOF
внутри объявленного payload считаются ошибками.

Невалидный Arrow/schema, запрещённый non-finite или malformed/oversized frame
прерывает весь stream. В `predict-stream` соответствие input/output `1:1`
гарантируется только для успешно провалидированных frames до ошибки.

Declared size проверяется до чтения payload. Default limit — `512 MiB`
(`536870912` bytes); положительный `--max-frame-bytes` меняет его только для
`fit-stream` и `predict-stream`.

В standalone `fit-stream` каждый непустой stdin frame получает собственный
цикл до `--epochs` с per-frame early stopping; пустая Arrow table пропускается.
Flight worker не применяет эту per-frame семантику к training job: `--epochs`
охватывает весь sealed набор, а payloads обходятся по ordinal внутри каждой
эпохи. В `predict-stream` пустому input frame соответствует пустой output frame
типа `list<float32>`. Zero terminator не порождает output frame. При clean EOF
или terminator `predict-stream` завершает stdout обычным EOF и не добавляет
свой zero terminator.

`predict-stream` пишет в stdout тот же framed protocol. Каждый output payload —
самостоятельный Arrow IPC file с одной колонкой `--pred-col`; writer flushes
stdout после каждого frame. Весь текстовый progress и diagnostics этого режима
идут в stderr. `fit-stream`, напротив, пишет training progress в текстовый
stdout, поскольку его stdout не является output data protocol.

## Тестирование

PostgreSQL integration tests требуют отдельную базу, имя которой начинается с
`transformer_test`. Внутри неё pytest создаёт миграциями одноразовую schema
`transformer_test_<uuid>` и удаляет её после test session. Production-база для
тестов намеренно отвергается.

```bash
python -m pip install pytest
POSTGRES_DB=transformer_test python3.11 -m pytest -q
```

Правила разработки и review собраны в
[`docs/policy/index.md`](docs/policy/index.md).

## Основные файлы

```text
app/main.py          # тонкий CLI entrypoint
app/cli/             # argparse и форматированный --help/--version
app/commands/        # реализации fit/predict/stream/plot команд
app/data/            # Arrow file/framed protocol, reshape и shape validation
app/model/           # Transformer, positional encoding, context masking
app/training/        # Trainer, configs, loss stages, scheduler, early stopping
app/metrics/         # TrainMetrics, JSONL writer/reader, SVG-графики
app/storage/         # checkpoint с весами и конфигурацией модели
app/runtime/         # device selection и версия приложения
app/flight/          # durable Arrow Flight service, ledger, spool и worker
contracts/flight/v1/ # нормативные JSON Schemas и golden fixtures
app/config.py        # project defaults
app/utils.py         # небольшие совместные runtime helpers
```
