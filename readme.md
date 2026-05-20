# Transformer

Python-проект для обучения и инференса PyTorch Transformer на датасетах в
формате Apache Arrow.

Проект умеет работать в двух режимах:

- читать готовый Arrow-файл с диска (`fit`, `predict`)
- принимать поток micro-batch Arrow payloads через stdin (`fit-stream`,
  `predict-stream`)

`fit-stream` используется командой `trainTransformer` из проекта
`inventory`.

## Требования

- Python 3.9+
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
PYTHONPATH=./app python ./app/main.py <action> [data] [options]
```

Доступные действия:

- `fit` — обучить модель на Arrow-файле
- `predict` — загрузить модель и сохранить предсказания в Arrow-файл
- `fit-stream` — читать framed Arrow payloads из stdin и обучать модель batch за batch
- `predict-stream` — читать framed Arrow payloads из stdin и писать framed
  predictions в stdout

## Обучение из файла

```bash
PYTHONPATH=./app python ./app/main.py fit ./data/train.arrow \
  --device=cpu \
  --model-name=model_weights.pth \
  --seq-len=20 \
  --epochs=25 \
  --batch-size=256
```

## Предсказание из файла

```bash
PYTHONPATH=./app python ./app/main.py predict ./data/test.arrow \
  --device=cpu \
  --model-name=model_weights.pth \
  --seq-len=20 \
  --preds-path=/tmp/preds.arrow \
  --pred-col=out
```

## Потоковое обучение

`fit-stream` не принимает путь к файлу данных. Если передать positional
`data`, запуск завершится ошибкой. Режим читает stdin до EOF:

```bash
PYTHONPATH=./app python ./app/main.py fit-stream \
  --device=cpu \
  --model-name=model_weights.pth \
  --seq-len=20
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
| `--seq-len` | Длина последовательности | обязательный |
| `--hidden` | Размер скрытого слоя | `256` |
| `--layers` | Количество Transformer layers | `5` |
| `--nhead` | Количество attention heads | `8` |
| `--dropout` | Dropout | `0.1` |

### Обучение

| Аргумент | Описание | По умолчанию |
| --- | --- | --- |
| `--lr` | Learning rate | `0.0005` |
| `--batch-size` | Размер mini-batch | `256` |
| `--epochs` | Количество эпох для `fit` | `25` |
| `--per-week` | Сколько epoch/frame считаются одной неделей в loss schedule | `5` |
| `--patience` | Early stopping patience | `5` |
| `--use-amp` | Включить AMP, если используется CUDA | выключено |

В `fit-stream` параметр `--epochs` сейчас не повторяет входящий поток несколько
раз. Каждый Arrow frame обучается как один streaming batch pass.

## Метрики обучения

`fit` и `fit-stream` печатают компактную строку `TrainMetrics` для каждого
epoch/frame:

```text
epoch=1 norm=183 loss=0.384000 ret=0.184000 prob=0.092000 ev=-0.011000 vol=0.000000 grad=0.830 rows=256 batches=1 nan=0.0300 valid_tokens=0.8800 lr=0.0005 week=1 ms=42
```

Поля:

- `loss` — итоговый loss после всех весов компонентов
- `ret`, `prob`, `ev`, `vol` — вклад компонентов loss
- `grad` — gradient norm до clipping
- `rows`, `batches` — объём данных в проходе
- `nan` — доля NaN во входном `src`
- `valid_tokens` — доля timesteps без NaN, совпадает с текущей padding mask
- `lr` — текущий learning rate
- `week` — номер loss schedule week с учётом `--per-week`
- `ms` — время обучения прохода

Чтобы дополнительно писать каждую строку метрик в JSONL:

```bash
PYTHONPATH=./app python ./app/main.py fit ./data/train.arrow \
  --seq-len=20 \
  --metrics-name=train.jsonl
```

Для `fit-stream` используется тот же аргумент:

```bash
PYTHONPATH=./app python ./app/main.py fit-stream \
  --seq-len=20 \
  --metrics-name=train-stream.jsonl
```

Файл сохраняется в `models/` и перезаписывается в начале нового `fit` /
`fit-stream` запуска. Каждая строка — один JSON object с теми же числовыми
полями и контекстом `epoch` или `frame`.

Построить SVG-графики по JSONL:

```bash
PYTHONPATH=./app python ./app/main.py plot-metrics train.jsonl \
  --plots-dir=./metrics/plots
```

`plot-metrics` создаёт отдельные SVG-файлы для `loss`, компонентов loss,
`grad_norm`, `nan_ratio`, `valid_token_ratio`, `rows`, `batches`, `lr`, `week`
и `elapsed_ms`.

## Потоковое предсказание

`predict-stream` не принимает путь к файлу данных. Он читает framed Arrow
payloads из stdin, загружает модель один раз на первом непустом frame и пишет
framed Arrow payloads с предсказаниями в stdout:

```bash
PYTHONPATH=./app python ./app/main.py predict-stream \
  --device=cpu \
  --model-name=model_weights.pth \
  --seq-len=20 \
  --pred-col=out
```

stdout в этом режиме является бинарным протоколом результата. Диагностические
сообщения пишутся в stderr.

## Arrow-файл

Для `fit` и `predict` входной файл должен быть Arrow IPC file с колонками:

```text
src: list<float>  # flattened [seq_len * feature_dim]
tgt: list<float>  # target vector
```

После чтения `src` преобразуется в тензор:

```text
[rows, seq_len * feature_dim] -> [rows, seq_len, feature_dim]
```

Если ширина `src` не делится на `--seq-len`, запуск завершится ошибкой.

`predict` сохраняет Arrow IPC file с одной колонкой `--pred-col`.
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
PYTHONPATH=./app pytest -v
```

## Основные файлы

```text
app/main.py        # CLI entrypoint
app/args.py        # argparse contract
app/arrow_io.py    # Arrow file и framed stdin protocol
app/trainer.py     # fit, fit_batch, predict
app/transformer.py # модель
```
