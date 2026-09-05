# Локальный Arrow и stream contract

> Тип: справочник. Формат данных local file/stream CLI. Этот документ не
> переопределяет public Arrow Flight v11 contract.

`fit`, `predict`, `fit-stream` и `predict-stream` используют самостоятельные
Arrow IPC files. Для remote API нормативны schemas и fixtures в
[`app/contracts/flight/v11`](../app/contracts/flight/v11/README.md); local CLI
сохраняет отдельный dense `src`/`tgt` contract и не принимает compact
`indexedFeatureBlocks` Flight v11.

## Arrow IPC input

Каждый file input и каждый stream payload — самостоятельный Arrow IPC file.
Поддерживаются `List`, `LargeList` и `FixedSizeList`, но child type должен быть
строго `float32` или `float64`:

```text
src: list<float32|float64>  # flattened [seq_len * feature_dim]
tgt: list<float32|float64>  # targetContract.slots.length, required for fit
```

В обеих колонках запрещены Arrow null rows и null elements; длина list должна
быть одинаковой у всех строк. `src` допускает IEEE `NaN` для пропусков, но
отклоняет `+inf` и `-inf`. `tgt` должен быть полностью finite, иметь ширину
ordered `targetContract.slots` и удовлетворять объявленному для каждой позиции
`observedConstraint`. У непустого input ширина `src` должна быть больше нуля.

Local fit получает полный consumer-neutral `ModelContract` через обязательный
`--model-contract=FILE`. Порядок `tgt` задаёт порядок opaque slot identities в
этом документе. Каждая координата имеет ровно один direct supervised component
с первого optimizer step. Private resources Objective не входят в `tgt`.

После валидации `float64` и `float32` input преобразуется в PyTorch `float32`;
finite `float64`, который выходит за диапазон `float32`, отклоняется до cast.
`src` reshaped следующим образом:

```text
[rows, seq_len * feature_dim] -> [rows, seq_len, feature_dim]
```

Ширина `src` должна делиться на `--seq-len`. В stream `feature_dim` не может
меняться между frames; при prediction с checkpoint v6 она также должна совпасть
с сохранённым значением. Missing-data semantics для `NaN` определяет
[training reference](./training-runtime.md).

## Prediction output

Для prediction `tgt` не требуется. File и stream prediction output содержит
только одну колонку `--pred-col` типа `list<float32>`; input columns в output
не копируются. Prediction runtime проверяет форму
`[rows, targetContract.slots.length]`, declared public transformations и finite
значения; row count и исходный порядок строк сохраняются. `predict` отклоняет
одинаковый canonical input/output path, чтобы не уничтожить входной файл.

Порядок output values совпадает с ordered slots сохранённого ModelContract.
Transformer не интерпретирует их identities. Для каждой raw model coordinate
применяется объявленная `publicPredictionTransformation`; private resources не
пересекают output boundary.
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
