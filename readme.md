# Transformer

Python-проект для обучения и инференса PyTorch Transformer на датасетах в
формате Apache Arrow.

Проект умеет работать в двух режимах:

- читать готовый Arrow-файл с диска (`fit`, `predict`)
- принимать поток micro-batch Arrow payloads через stdin (`fit-stream`)

`fit-stream` используется командой `trainTransformer` из соседнего проекта
`../inventory`.

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
cd ../inventory
npm run build
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
| `--patience` | Early stopping patience | `5` |
| `--use-amp` | Включить AMP, если используется CUDA | выключено |

В `fit-stream` параметр `--epochs` сейчас не повторяет входящий поток несколько
раз. Каждый Arrow frame обучается как один streaming batch pass.

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

## Framed stdin protocol

`fit-stream` читает последовательность payloads из stdin:

```text
8 bytes unsigned big-endian payload length
Arrow file payload
8 bytes unsigned big-endian payload length
Arrow file payload
...
EOF
```

Каждый payload — самостоятельный Arrow IPC file с колонками `src` и `tgt`.

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
