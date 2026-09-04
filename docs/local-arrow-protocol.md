# Локальный Arrow и stream contract

> Тип: справочник. Формат данных local file/stream CLI. Этот документ не
> переопределяет public Arrow Flight v10 contract.

`fit`, `predict`, `fit-stream` и `predict-stream` используют самостоятельные
Arrow IPC files. Для remote API нормативны schemas и fixtures в
[`app/contracts/flight/v10`](../app/contracts/flight/v10/README.md); local CLI
сохраняет отдельный dense `src`/`tgt` contract и не принимает compact
`indexedFeatureBlocks` Flight v10.

## Arrow IPC input

Каждый file input и каждый stream payload — самостоятельный Arrow IPC file.
Поддерживаются `List`, `LargeList` и `FixedSizeList`, но child type должен быть
строго `float32` или `float64`:

```text
src: list<float32|float64>  # flattened [seq_len * feature_dim]
tgt: list<float32|float64>  # width 6, required for training
```

В обеих колонках запрещены Arrow null rows и null elements; длина list должна
быть одинаковой у всех строк. `src` допускает IEEE `NaN` для пропусков, но
отклоняет `+inf` и `-inf`. `tgt` должен быть полностью finite и иметь ровно
шесть значений: `tgt[0]` находится в `[-1, 1]`, остальные координаты — в
`[0, 1]`. У непустого input ширина `src` должна быть больше нуля.

Target и prediction используют одинаковый порядок:

| Позиция | Назначение |
| --- | --- |
| `tgt[0]` | `MeanReturn` |
| `tgt[1]` | `SigmaReturn` |
| `tgt[2]` | `ProbTP` |
| `tgt[3]` | `ProbSL` |
| `tgt[4]` | `VolatilityNext` |
| `tgt[5]` | `HittingProbTP` |

Каждая координата имеет прямой supervised path с первого optimizer step.
Private Gaussian scale не входит в этот вектор.

После валидации `float64` и `float32` input преобразуется в PyTorch `float32`;
finite `float64`, который выходит за диапазон `float32`, отклоняется до cast.
`src` reshaped следующим образом:

```text
[rows, seq_len * feature_dim] -> [rows, seq_len, feature_dim]
```

Ширина `src` должна делиться на `--seq-len`. В stream `feature_dim` не может
меняться между frames; при prediction с checkpoint v5 она также должна совпасть
с сохранённым значением. Missing-data semantics для `NaN` определяет
[training reference](./training-runtime.md).

## Prediction output

Для prediction `tgt` не требуется. File и stream prediction output содержит
только одну колонку `--pred-col` типа `list<float32>`; input columns в output
не копируются. Prediction runtime проверяет форму `[rows, 6]` и finite
значения; row count и исходный порядок строк сохраняются. `predict` отклоняет
одинаковый canonical input/output path, чтобы не уничтожить входной файл.

Порядок шести output values:

```text
[MeanReturn, SigmaReturn, ProbTP, ProbSL, VolatilityNext, HittingProbTP]
```

`MeanReturn` находится в `[-1, 1]`, остальные координаты — в `[0, 1]`.
`ProbTP` и `ProbSL` независимы. Raw logits не пересекают output boundary.
Пустой file input для `predict` создаёт типизированный пустой output; `fit`
отклоняет training input без строк.

## Framed stdin/stdout protocol

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
`fit-stream` также нужна колонка `tgt`; для `predict-stream` `tgt` не требуется.
Declared length `0` является terminator и не Arrow frame. Чистый EOF между
frames также нормально завершает stream. Частичный 8-byte header и EOF внутри
объявленного payload считаются ошибками.

Невалидный Arrow/schema, запрещённый non-finite или malformed/oversized frame
прерывает весь stream. В `predict-stream` соответствие input/output `1:1`
гарантируется только для успешно провалидированных frames до ошибки.

Declared size проверяется до чтения payload. Default limit — `512 MiB`
(`536870912` bytes); положительный `--max-frame-bytes` меняет его только для
`fit-stream` и `predict-stream`.

В `predict-stream` пустому input frame соответствует пустой output frame типа
`list<float32>`. Zero terminator не порождает output frame. При clean EOF или
terminator процесс завершает stdout обычным EOF и не добавляет свой zero
terminator.

`predict-stream` пишет в stdout тот же framed protocol. Каждый output payload
— самостоятельный Arrow IPC file с одной колонкой `--pred-col`; writer flushes
stdout после каждого frame. Весь текстовый progress и diagnostics этого режима
идут в stderr. `fit-stream`, напротив, пишет training progress в текстовый
stdout, поскольку его stdout не является output data protocol.

Semantics обучения на frame, schedule и early stopping находятся в
[training reference](./training-runtime.md); команда и аргументы — в
[справочнике CLI](./cli/index.md).
