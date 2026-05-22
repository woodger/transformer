# Transformer

Python-проект для обучения и инференса PyTorch Transformer на датасетах в
формате Apache Arrow.

Проект умеет работать в двух режимах:

- читать готовый Arrow-файл с диска (`fit`, `predict`)
- принимать поток micro-batch Arrow payloads через stdin (`fit-stream`,
  `predict-stream`)

`fit-stream` используется командой `trainTransformer` из проекта
`inventory`.

Новые checkpoint-файлы сохраняют не только веса, но и конфигурацию модели
(`seq_len`, `hidden`, `layers`, `nhead`, `mode`). Поэтому `predict` и
`predict-stream` могут восстановить архитектуру из checkpoint, если
соответствующие CLI-аргументы не переданы.

## Требования

- Python 3.10+
- PyTorch
- NumPy
- PyArrow
- CUDA опционально

Установка зависимостей:

```bash
pip install torch numpy pyarrow pytest
```

## CLI

Запуск выполняется из корня проекта:

```bash
python ./app/main.py <action> [data] [options]
```

Версию можно посмотреть так:

```bash
python ./app/main.py --version
```

Доступные действия:

- `fit` — обучить модель на Arrow-файле
- `predict` — загрузить модель и сохранить предсказания в Arrow-файл
- `fit-stream` — читать framed Arrow payloads из stdin и обучать модель batch за batch
- `predict-stream` — читать framed Arrow payloads из stdin и писать framed
  predictions в stdout

## Обучение из файла

```bash
python ./app/main.py fit ./data/train.arrow \
  --device=cpu \
  --model-name=model_weights.pth \
  --seq-len=20 \
  --mode=relaxed \
  --epochs=25 \
  --batch-size=256
```

## Предсказание из файла

```bash
python ./app/main.py predict ./data/test.arrow \
  --device=cpu \
  --model-name=model_weights.pth \
  --preds-path=/tmp/preds.arrow \
  --pred-col=out
```

Если модель была сохранена новой версией приложения, `--seq-len`, `--hidden`,
`--layers`, `--nhead` и `--mode` для `predict` можно не передавать: они будут
прочитаны из checkpoint. Для legacy-файлов, где сохранён только `state_dict`,
эти параметры всё ещё нужно передать вручную.

## Потоковое обучение

`fit-stream` не принимает путь к файлу данных. Если передать positional
`data`, запуск завершится ошибкой. Режим читает stdin до EOF:

```bash
python ./app/main.py fit-stream \
  --device=cpu \
  --model-name=model_weights.pth \
  --seq-len=20 \
  --mode=relaxed
```

Обычно этот режим запускается не вручную, а из `inventory`:

```bash
yarn build
node dist/index.js trainTransformer \
  --figi=BBG0013HJJ31 \
  --context=BBG000B9XRY4,BBG004730N88 \
  --from=2020-09-09T21:00:00.000Z \
  --to=2021-08-27T21:00:00.000Z \
  --interval=1day \
  --chunk-days=30 \
  --lookback=10 \
  --horizon=5 \
  --seq-len=20 \
  --mode=relaxed \
  --device=cpu \
  --model-name=model_weights.pth
```

В этом сценарии `inventory`:

1. Загружает `getFrame()` чанками.
2. Собирает temporal windows.
3. Пишет Arrow payloads в stdin transformer.
4. Закрывает stdin после последнего чанка.

Transformer обучается на каждом непустом входящем frame и сохраняет модель
после EOF. Пустые frames пропускаются; если непустых frames не было, модель не
сохраняется.

## Параметры

### Runtime

| Аргумент | Описание | По умолчанию |
| --- | --- | --- |
| `--device` | Устройство: `cpu` или `gpu` | `cpu` |
| `--model-name` | Путь к файлу весов | `model_weights.pth` |
| `--preds-path` | Куда сохранить предсказания | `/tmp/preds.arrow` |
| `--pred-col` | Имя колонки с предсказаниями | `out` |
| `--metrics-name` | Имя JSONL-файла с метриками в `models/` | не задан |
| `--plots-dir` | Каталог для SVG-графиков `plot-metrics` | `metrics_plots` |

### Модель

| Аргумент | Описание | По умолчанию |
| --- | --- | --- |
| `--seq-len` | Длина последовательности | обязательный для `fit`; для `predict` может браться из checkpoint |
| `--hidden` | Размер скрытого слоя | `256` |
| `--layers` | Количество Transformer layers | `5` |
| `--nhead` | Количество attention heads | `8` |
| `--dropout` | Dropout | `0.1` |
| `--mode` | Как обрабатывать `NaN` в context timesteps: `strict`, `relaxed` | `relaxed` |

### Обучение

| Аргумент | Описание | По умолчанию |
| --- | --- | --- |
| `--lr` | Learning rate | `0.0005` |
| `--batch-size` | Размер mini-batch | `256` |
| `--epochs` | Количество эпох для `fit` | `25` |
| `--loss-stage` | Максимальный этап loss: `1..4` | `4` |
| `--loss-schedule` | Как двигать этап loss: `none`, `epoch`, `step` | `epoch` |
| `--stage-size` | Сколько epoch/optimizer steps держать один этап | `5` |
| `--patience` | Early stopping patience | `5` |
| `--use-amp` | Включить AMP, если используется CUDA | выключено |

В `fit-stream` каждый непустой Arrow frame обучается отдельным циклом
`epoch=1..--epochs` до срабатывания `--patience`. Веса модели при этом не
сбрасываются между frames.

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

## Контекстные пропуски

`src` может содержать `NaN` в отдельных фичах контекстного timestep. Режим
`--mode` задаёт, как такие timesteps попадают в Transformer:

- `strict` — старая схема: timestep маскируется, если хотя бы одна фича `NaN`.
- `relaxed` — timestep маскируется только если все фичи `NaN`; после этого
  `NaN` заменяются на `0.0`, а к входу добавляются бинарные missing-флаги.
  Поэтому `0` остаётся только численным placeholder, а информация о пропуске
  не теряется.

Текущий default — `relaxed`. При таком контракте `inventory` должен передавать
`NaN` для отсутствующих context candles; Transformer строит masks и missing-флаги
из исходных `NaN` до любых tensor ops, и только затем заменяет `NaN -> 0`.

## Метрики обучения

`fit` и `fit-stream` печатают компактную строку `TrainMetrics` для каждого
epoch. В `fit-stream` строка дополнительно содержит номер входного frame.
Loss schedule продвигается по выбранному `--loss-schedule`:

- `none` — всегда используется `--loss-stage`
- `epoch` — stage считается от epoch внутри текущего frame
- `step` — stage считается от глобального optimizer step и не сбрасывается
  между frames

```text
epoch=1 norm=183 batch_size=256 loss_schedule=epoch stage_size=5 max_loss_stage=4 device=cpu hidden=256 layers=5 seq_len=20 loss=0.384000 ret=0.184000 prob=0.092000 ev=-0.011000 vol=0.000000 sigma_min=0.0800 sigma_p05=0.1200 sigma_mean=0.4200 ret_mae=0.018 ret_rmse=0.024 ret_mae_baseline=0.031 grad=0.830 rows=256 batches=1 nan=0.0300 masked_tokens=0.1200 complete_tokens=0.7600 partial_tokens=0.1200 empty_tokens=0.1200 step=1 lr=0.0005 loss_stage=1 ms=42
```

Поля:

- `loss` — итоговый loss после всех весов компонентов
- `batch_size`, `loss_schedule`, `stage_size`, `max_loss_stage`, `hidden`,
  `layers`, `seq_len`, `device` — параметры запуска, записываются в каждую
  строку JSONL
- `ret`, `prob`, `ev`, `vol` — вклад компонентов loss
- `sigma_min`, `sigma_p05`, `sigma_mean` — статистика предсказанного `sigmaR`
  для диагностики Gaussian NLL
- `ret_mae`, `ret_rmse` — ошибка прогноза `meanR` против target `meanR`
- `ret_mae_baseline` — MAE нулевого прогноза `meanR=0`; полезно сравнивать с
  `ret_mae`, чтобы видеть, лучше ли модель простой нулевой гипотезы
- `grad` — gradient norm до clipping
- `rows`, `batches` — объём данных в проходе
- `nan` — доля NaN во входном `src`
- `masked_tokens` — доля timesteps, которые скрыты от attention текущим
  `--mode`
- `complete_tokens` — доля timesteps без `NaN`
- `partial_tokens` — доля timesteps с частью заполненных фичей и частью `NaN`
- `empty_tokens` — доля timesteps, где все фичи `NaN`
- `step` — глобальный номер optimizer step к концу строки метрик
- `lr` — текущий learning rate
- `loss_stage` — активный этап функции потерь
- `ms` — время обучения прохода

Чтобы дополнительно писать каждую строку метрик в JSONL:

```bash
python ./app/main.py fit ./data/train.arrow \
  --seq-len=20 \
  --metrics-name=train.jsonl
```

Для `fit-stream` используется тот же аргумент:

```bash
python ./app/main.py fit-stream \
  --seq-len=20 \
  --metrics-name=train-stream.jsonl
```

Файл сохраняется в `models/` и перезаписывается в начале нового `fit` /
`fit-stream` запуска. Каждая строка — один JSON object с теми же числовыми
полями и контекстом `epoch` или `frame`.

Построить SVG-графики по JSONL:

```bash
python ./app/main.py plot-metrics train.jsonl \
  --plots-dir=./metrics/plots
```

`plot-metrics` создаёт отдельные SVG-файлы для `loss`, компонентов loss,
`sigma_min`, `sigma_p05`, `sigma_mean`, `grad_norm`, `nan_ratio`, token ratios,
`rows`, `batches`, `step`, `lr`, `loss_stage` и `elapsed_ms`.

## Потоковое предсказание

`predict-stream` не принимает путь к файлу данных. Он читает framed Arrow
payloads из stdin, загружает модель один раз на первом непустом frame и пишет
framed Arrow payloads с предсказаниями в stdout:

```bash
python ./app/main.py predict-stream \
  --device=cpu \
  --model-name=model_weights.pth \
  --pred-col=out
```

stdout в этом режиме является бинарным протоколом результата. Диагностические
сообщения пишутся в stderr.

## Arrow-файл

Для `fit` и `predict` входной файл должен быть Arrow IPC file с колонками:

```text
src: list<float>  # flattened [seq_len * feature_dim]
tgt: list<float>  # target vector, required for fit
```

После чтения `src` преобразуется в тензор:

```text
[rows, seq_len * feature_dim] -> [rows, seq_len, feature_dim]
```

Если ширина `src` не делится на `--seq-len`, запуск завершится ошибкой.

Для `predict` колонка `tgt` не требуется. `predict` сохраняет Arrow IPC file с
одной колонкой `--pred-col`.
`predict-stream` пишет такую же таблицу в каждом output frame.

## Framed stdin protocol

`fit-stream` и `predict-stream` читают последовательность payloads из stdin:

```text
8 bytes unsigned big-endian payload length
Arrow file payload
8 bytes unsigned big-endian payload length
Arrow file payload
...
EOF
```

Каждый payload — самостоятельный Arrow IPC file с колонкой `src`. Для
`fit-stream` также нужна колонка `tgt`; для `predict-stream` `tgt` не
требуется.

`predict-stream` пишет в stdout тот же framed protocol. Каждый output payload —
самостоятельный Arrow IPC file с одной колонкой `--pred-col`.

## Тестирование

```bash
. .venv/bin/activate
pytest -v
```

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
app/config.py        # project defaults
app/utils.py         # небольшие совместные runtime helpers
```
