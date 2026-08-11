# Контракт Transformer Arrow Flight v3

Этот каталог содержит нормативный контракт обмена данными между Inventory и
Transformer, не зависящий от языка реализации. JSON Schema, Arrow-схемы и
эталонные фикстуры версионируются вместе. Обоснование решений о надёжности и
восстановлении приведено в
[ADR 0005](../../../../docs/adr/0005-durable-streaming-flight-v3.md).

## Конверт и аутентификация

Каждый запрос и результат action представляет собой JSON-объект в кодировке
UTF-8, содержащий:

```json
{"contract":"transformer-flight","version":3,"requestId":"UUID"}
```

Для каждого RPC требуется заголовок `authorization: Bearer TOKEN`. Мутации
также содержат `idempotencyKey`. Канонический хеш запроса — SHA-256 от
компактного JSON с отсортированными ключами после удаления `requestId` и
`idempotencyKey`. Точный повтор возвращает зафиксированный результат;
повторное использование ключа для другого запроса отклоняется.

## Actions

Сервер объявляет строго следующий список:

- `transformer.v3.capabilities`
- `transformer.v3.health`
- `transformer.v3.job.create`
- `transformer.v3.job.acquire`
- `transformer.v3.job.status`
- `transformer.v3.job.inputs.list`
- `transformer.v3.job.input.close`
- `transformer.v3.job.outputs.list`
- `transformer.v3.job.cancel`
- `transformer.v3.model.describe`

Этот список полностью определяет доступную поверхность actions.
`job.input.close` обозначает EOF. `DoExchange` и `PollFlightInfo` не входят в
контракт.

Схемы запросов и результатов закрыты: неизвестные поля отклоняются.
`action-result.schema.json` содержит закрытое объединение всех результатов
actions. Эталонные документы находятся в `fixtures/json/`.

## Состояние

V3 предоставляет две независимые оси состояния:

```text
input.state     OPEN | CLOSED | ABORTED
execution.state WAITING_INPUT | QUEUED | RUNNING | RETRYING |
                CANCELLING | SUCCEEDED | FAILED | CANCELLED
```

Первый зафиксированный непустой непрерывный префикс входных данных
автоматически ставит job в очередь. Worker может работать, пока вход остаётся
открытым. `job.input.close` фиксирует EOF и неизменяемый манифест receipt-ов,
но не является командой запуска. Для успешного завершения и публичного
доступа к результатам вход должен быть закрыт.

Попытка закрыть пустой fit возвращает `EMPTY_INPUT`, не закрывая вход. Пустой
predict допустим: отсутствие payload-ов даёт отсутствие outputs, а один
типизированный пустой payload — один типизированный пустой output.

## Стабильная идентичность и fencing

Inventory создаёт и сохраняет `jobId` до вызова create. После удаления тяжёлых
данных job Transformer сохраняет компактную, ограниченную owner-ом запись об
идентичности. Благодаря этому потерянный ответ create восстанавливается точным
повтором запроса, а UUID нельзя позднее использовать для другого запроса.

Каждая публичная мутация защищена парой `clientExecutionId` и выданным сервером
монотонным `fencingToken`, закодированным десятичной строкой. `job.acquire`
атомарно сравнивает предыдущее владение и увеличивает token. DoPut проверяет
fence перед приёмом данных и повторно — в транзакции фиксации входа. Устаревшая
загрузка может оставить только свой уникальный candidate без ссылок на него;
она не может заменить победивший input artifact.

## Загрузка

Descriptor входных данных:

```text
pathDescriptor("transformer", "v3", "jobs", jobId, "inputs", ordinal)
```

Metadata соответствует `upload-metadata.schema.json`. Один DoPut представляет
один семантический payload; границы RecordBatch используются только для
транспортного разбиения. Сервер надёжно публикует неизменяемый candidate,
фиксирует его receipt и только после этого отправляет один PutResult. PutResult
содержит `inputRevision`, первый отсутствующий `nextInputOrdinal` и признак
того, привела ли эта фиксация к автоматической постановке в очередь.

Фиксация не по порядку разрешена. Worker получает только непрерывный префикс
ordinal, поэтому время поступления не влияет на логический порядок строк.

## Закрытие входа и digest манифеста

При закрытии передаются счётчики постоянного размера и `manifestSha256`, а не
полный массив. Digest — SHA-256 от компактного JSON с отсортированными ключами
для следующих полей server receipt, упорядоченных по ordinal:

```text
payloadId, ordinal, schemaId, dataContractSha256, rows, batches, bytes,
sha256, schemaFingerprint
```

`commitRevision`, временные метки, состояние очереди и порядок поступления
исключены. Перед фиксацией `CLOSED` сервер проверяет непрерывность ordinal,
итоговые значения, единую физическую Arrow-схему и единый
`dataContractSha256`.

## Пагинация

Страницы входных данных упорядочены по `commitRevision` конкретного job. Первый
запрос передаёт `afterRevision` и фиксирует возвращённый `snapshotRevision`.
Следующие страницы повторно используют этот snapshot и удовлетворяют условию:

```text
cursor < commitRevision <= snapshotRevision
```

После завершения обхода следующий начинается с предыдущего `snapshotRevision`
в качестве `afterRevision`. Поэтому payload с малым ordinal, зафиксированный
позже, получает новую revision и не может потеряться. Размер страницы не
превышает 100 записей.

Status содержит только ограниченные по размеру сводные данные. Input receipts
и output descriptors возвращаются соответствующими list actions.

## Arrow-схемы

Физические схемы заданы точно:

```text
inventory.sequence.fit.v2
  src: non-null FixedSizeList<Float32>[seqLen * featureDim]
  tgt: non-null FixedSizeList<Float32>[6]

inventory.sequence.predict.v2
  src: non-null FixedSizeList<Float32>[seqLen * featureDim]

transformer.prediction.v2
  <predictionColumn>: non-null FixedSizeList<Float32>[6]
```

Каждое поле верхнего уровня имеет `nullable=false`. У каждого `FixedSizeList`
есть дочернее поле `item` типа `Float32` с `nullable=true`, что соответствует
каноническому представлению PyArrow. Это свойство физической схемы не разрешает
null-значения в ML-данных: ingress-валидация и валидация output отклоняют любую
строку или дочернее значение с null. Metadata схемы не имеет семантического
значения и исключается из точного сравнения и `schemaFingerprint`.

Неканоническая входная схема отклоняется в DoPut с `INVALID_ARGUMENT` до
создания input reservation, durable artifact или запуска worker. Вход job
остаётся открытым, поэтому тот же ordinal можно загрузить повторно с
канонической схемой.

Строки и элементы с null, а также бесконечности отклоняются. `src` может
содержать NaN; значения target и prediction должны быть конечными. Волатильность
target с индексом 4 не может быть отрицательной, а вероятность попадания с
индексом 5 должна находиться в диапазоне `[0, 1]`. Фиксированная ширина списков
однозначно задаёт типизированный пустой payload без batch-ей.

## ML-контракт данных и модели

Inventory владеет семантическим документом набора данных. Create передаёт его
идентичность, `dataContractSha256`, `seqLen`, `featureDim` и идентичность схемы
target. Transformer хранит и возвращает эти поля, не воспроизводя семантику
features Inventory. Predict create отклоняется с `MODEL_SCHEMA_MISMATCH` до
загрузки, если выбранная неизменяемая модель не сертифицирована для того же
хеша.

`modelAlias` ограничен owner-ом и во время create атомарно разрешается в
`resolvedModelRef`. `model.describe` возвращает неизменяемую generation, digest
checkpoint, конфигурацию модели и идентичность контракта данных без пути на
сервере. У опубликованных моделей нет автоматического TTL. Стабильные ошибки:
`NOT_FOUND`, `MODEL_UNAVAILABLE`, `MODEL_CORRUPT` и
`MODEL_SCHEMA_MISMATCH`.

## Доступ к результатам

Клиенты получают output descriptors через `job.outputs.list`. GetFlightInfo
использует:

```text
pathDescriptor("transformer", "v3", "jobs", jobId, "outputs", ordinal)
```

Операция разрешена только после `execution.state = SUCCEEDED`. Tickets остаются
случайными, непрозрачными, ограниченными owner-ом и сроком действия. Prediction
artifacts могут подготавливаться при открытом входе, но все outputs публикуются
одновременно с terminal success; частичный успех никогда не виден.

## Потоковая обработка и восстановление

Нулевая epoch читает открытый непрерывный поток. Optimizer batches и
ограниченные shuffle windows пересекают границы RecordBatch и payload. На
текущей границе доступных данных worker сообщает об ожидании; EOF сбрасывает
последнее окно и завершает epoch. Последующие epochs повторно читают закрытый
durable dataset.

Сбой до EOF перезапускает незавершённую epoch с начала. Recovery checkpoints
остаются снимками на границах полных global epochs. `inputIdleTimeout`
отсчитывается только после подтверждения worker-ом ожидания следующего
непрерывного ordinal; фиксация вне порядка не продлевает timeout.

При одинаковых упорядоченных данных, seed, deterministic-конфигурации и
hardware/runtime отложенная потоковая подача и полностью закрытый вход должны
давать одинаковый порядок строк и shuffle, optimizer steps, ML-state после
каждой epoch, семантически одинаковый checkpoint и итоговую модель. Wall-clock
telemetry и digest сериализованного файла не являются критериями
эквивалентности.
