# ADR 0005: долговечный потоковый Flight v3

- Статус: принято
- Дата: 2026-08-10
- Заменяет: решения о lifecycle Flight v2 из ADR 0003 и контракт процесса
  worker v1 из ADR 0004

## Контекст

Flight v2 принимает набор долговечных payload-ов `DoPut`, фиксирует их полный
manifest и запускает worker только после получения всех входных данных. Это
закрывает окна повторной передачи transport-а, но оставляет для Inventory две
системные проблемы:

- ответ create может потеряться после фиксации созданного Transformer-ом
  `jobId`, но до его сохранения в Inventory;
- lease в PostgreSQL Inventory не защищает границу Transformer от поздней
  мутации прежнего владельца.

Запечатанный worker manifest также делает обучение строго пакетным. Нулевая
epoch не может начаться после первого долговечного payload, пока продолжают
поступать следующие payload-ы. Долговечный потоковый протокол не должен
привязывать job к одному Flight-соединению: каждый `DoPut` независимо фиксирует
прикладной payload, а worker может быть заменён без потери committed inputs.

Протокол образует breaking boundary. Любой v2 compatibility surface в
production runtime v3 сделал бы неоднозначными семантику состояния,
восстановления и владения.

## Решение

Transformer Flight v3 является долговечным потоковым job-протоколом
прикладного уровня. Он не использует `DoExchange`, а job не принадлежит одному
сетевому соединению. Каждый успешный `DoPut` надёжно публикует один
неизменяемый semantic payload и фиксирует его receipt в PostgreSQL до возврата
`PutResult` сервером.

Целевой flow fit:

```text
job.create
-> input OPEN / execution WAITING_INPUT
-> первый непустой непрерывный committed input
-> execution QUEUED
-> worker RUNNING при по-прежнему OPEN input
-> input.close фиксирует EOF и неизменяемый manifest
-> worker завершает epoch 0
-> последующие epochs перечитывают полный долговечный dataset
-> SUCCEEDED атомарно публикует modelRef
```

Реализация могла поставляться серией проверяемых изменений в ветке разработки.
Production обновлялся одной breaking-границей:

```text
Inventory v2 + Transformer v2
             -> атомарное переключение
Inventory v3 + Transformer v3
```

Runtime v3 не имеет v2 compatibility surface: отсутствуют v2 dispatcher,
имена actions, descriptor paths, aliases схем, fallback и одновременная
поддержка двух протоколов.

## Публичные состояния

Input и execution являются независимыми осями:

```text
input.state:
  OPEN | CLOSED | ABORTED

execution.state:
  WAITING_INPUT
  QUEUED
  RUNNING
  RETRYING
  CANCELLING
  SUCCEEDED
  FAILED
  CANCELLED
```

Следующие связующие правила являются нормативными:

- `OPEN` однократно переходит в `CLOSED` или `ABORTED`;
- `CLOSED` и `ABORTED` запрещают новые uploads;
- `SUCCEEDED` требует `CLOSED`, и terminal publication проверяет это в той же
  transaction PostgreSQL;
- отмена или ошибка без retry при `OPEN` переводит input в `ABORTED`;
- ошибка с retry сохраняет `OPEN` или `CLOSED`;
- `OPEN + RUNNING` допустимо и является штатным потоковым состоянием;
- первый committed непустой непрерывный prefix автоматически переводит
  execution из `WAITING_INPUT` в `QUEUED`;
- `job.input.close` фиксирует EOF, но не запускает execution;
- закрытие пустого fit отклоняется с `EMPTY_INPUT`, а input остаётся `OPEN`;
- пустой predict допустим. Ноль payload-ов даёт ноль outputs, а typed-empty
  input payload даёт соответствующий typed-empty output.

## Публичные actions и доступ к output

Закрытый набор actions v3:

```text
transformer.v3.capabilities
transformer.v3.health
transformer.v3.job.create
transformer.v3.job.acquire
transformer.v3.job.status
transformer.v3.job.inputs.list
transformer.v3.job.input.close
transformer.v3.job.outputs.list
transformer.v3.job.cancel
transformer.v3.model.describe
```

`job.seal` и `job.start` не существуют. `GetFlightInfo` разрешён только после
`execution.state = SUCCEEDED`: закрытие input необходимо, но недостаточно для
публикации. Пока input открыт, prediction outputs могут готовиться в workspace
attempt, однако все outputs становятся видимыми в одной terminal transaction.
Частичный успешный результат никогда не публикуется.

## Стабильная идентификация job и tombstones

Inventory создаёт UUID `jobId` и фиксирует его локально до первого сетевого
вызова. `job.create` содержит:

```json
{
  "contract": "transformer-flight",
  "version": 3,
  "requestId": "uuid",
  "idempotencyKey": "inventory-job-create:...",
  "jobId": "uuid",
  "clientExecutionId": "uuid",
  "operation": "fit"
}
```

Transformer атомарно сохраняет owner, canonical hash create-запроса, начальное
внешнее владение и idempotent result. Компактная identity сохраняется после
удаления тяжёлых job artifacts и изменяемой строки job. Повтор с теми же owner
и hash запроса разрешается через сохранённую identity; другой create-запрос не
может повторно использовать `jobId`. Ответ другому owner не раскрывает
существование identity.

Сохранённая identity закрывает окно потерянного create-response и исключает
повторное использование идентификатора. Отдельный action `job.resolve` не
вводится.

## Межсистемный fencing

Внешнее владение состоит из:

```json
{
  "clientExecutionId": "uuid",
  "fencingToken": "7"
}
```

`fencingToken` — положительное монотонное целое число, выданное сервером и
закодированное canonical decimal string. Для takeover вызывается `job.acquire`
с прежней identity владельца, ожидаемым token и новым `clientExecutionId`.
Transformer атомарно сравнивает текущую пару, увеличивает token и возвращает
его. Точный idempotent replay, включая повтор после потери acquire-response,
возвращает уже зафиксированное новое владение без нового увеличения.

Текущий внешний fence проверяется:

- до приёма данных `DoPut`;
- повторно в transaction PostgreSQL непосредственно перед фиксацией receipt
  `DoPut`;
- в `job.input.close` и `job.cancel`;
- в каждой другой публичной мутации.

`DoPut`, начатый до takeover, но не прошедший вторую проверку, удаляет только
свой temporary или unreferenced candidate artifact и возвращает
`STALE_FENCE`. Read-only status, списки inputs и outputs, а также описание
модели не требуют fence.

Существующий UUID `attemptId` остаётся внутренним equality fence worker-а.
Владение внутренней attempt никогда не передаётся, поэтому дополнительный
`attemptFence` не вводится. Монотонный fence существует только на границе
Inventory → Transformer.

## Долговечный upload и порядок

Descriptor имеет вид:

```text
pathDescriptor("transformer", "v3", "jobs", jobId, "inputs", ordinal)
```

Metadata содержит внешний fence и semantic identity:

```json
{
  "contract": "transformer-flight",
  "version": 3,
  "jobId": "uuid",
  "clientExecutionId": "uuid",
  "fencingToken": "8",
  "payloadId": "uuid",
  "ordinal": 0,
  "schemaId": "inventory.sequence.fit.v2",
  "dataContractSha256": "hex",
  "rows": 1820
}
```

Один `DoPut` соответствует одному semantic payload. Границы RecordBatch нужны
только для transport chunking. Завершение не по порядку разрешено, но workers
получают только непрерывный prefix ordinals. Логический порядок задаётся
ordinal, затем номером строки внутри payload; время поступления его не меняет.

Каждый upload записывает уникальный неизменяемый candidate path, например:

```text
recovery/jobs/{jobId}/inputs/{ordinal}-{payloadId}-{uploadToken}.arrow
```

Uploads не перезаписывают один общий конечный path для ordinal. После проверки
ordinal, payload identity, текущего внешнего fence, лимитов и состояния job
PostgreSQL выбирает единственную победившую ссылку на artifact. Проигравший
candidate является orphan, доступным для reconciliation; он не может заменить
победителя, выбранного последующим owner.

`PutResult` содержит receipt и дополнительные поля:

```text
inputRevision
nextInputOrdinal
queued
```

`inputRevision` — монотонная commit revision в пределах job.
`nextInputOrdinal` — первый отсутствующий ordinal. Дублирующиеся worker
notifications и повторное восстановление `PutResult` не должны приводить к
двукратному использованию одного ordinal.

## Закрытие input и digest manifest

`job.input.close` передаёт итоговые значения вместо полного manifest:

```json
{
  "jobId": "uuid",
  "clientExecutionId": "uuid",
  "fencingToken": "8",
  "payloadCount": 5,
  "totalRows": 9100,
  "totalBytes": 324625520,
  "manifestSha256": "hex"
}
```

Transformer проверяет ordinals `0..payloadCount-1`, отсутствие пропусков,
количество payload-ов, rows и bytes, единую physical Arrow schema и единый
`dataContractSha256`. Digest является SHA-256 от определённого контрактом
canonical упорядоченного списка server receipts. Receipts сортируются по
ordinal. Digest не включает порядок поступления, timestamps,
`commitRevision`, состояние queue/execution и любые wall-clock данные, поэтому
для одного committed dataset он стабилен.

Успешная close transaction сохраняет итоговые значения и переводит input в
`CLOSED`. Повторное закрытие с теми же данными idempotent; отличающиеся данные
считаются конфликтом.

## Стабильная pagination

`job.inputs.list` и `job.outputs.list` используют pagination. Inputs обходятся
по монотонному `commitRevision`, а не по ordinal. Первый запрос фиксирует
текущий `inputRevision` как `snapshotRevision`; каждая страница использует:

```text
cursor < commitRevision <= snapshotRevision
```

Значения полей:

- `afterRevision` — watermark завершённого предыдущего обхода;
- `snapshotRevision` — включительная верхняя граница, зафиксированная для
  текущего обхода;
- `cursor` — последний прочитанный `commitRevision` текущего обхода.

Новый обход начинается с:

```text
afterRevision = previousSnapshotRevision
snapshotRevision = currentInputRevision
cursor = afterRevision
```

Поэтому input с малым ordinal, зафиксированный позднее, получает больший
`commitRevision` и не может быть пропущен. Размер страницы ограничен
контрактом; status и terminal results не содержат неограниченные массивы
inputs или outputs.

## Семантика потокового fit

Логический порядок данных:

```text
ordinal -> строка внутри payload -> ограниченное shuffle window -> optimizer batch
```

Границы payload и RecordBatch не являются границами optimizer batch, shuffle
window или epoch. Нулевая epoch читает открытый непрерывный поток. Полное
shuffle window немедленно поступает в обучение; на текущем input frontier
worker ожидает следующий ordinal. EOF сбрасывает последнее неполное shuffle
window, и только после этого нулевая epoch считается завершённой. Первая и
последующие epochs повторно читают полный закрытый неизменяемый dataset из
долговечного хранилища.

До commit input artifact и его неизменяемого receipt сервис выполняет полную
проверку physical schema и значений. Перед первым использованием worker
attempt проверяет identity receipt, byte count и SHA-256. Последующие чтения
того же receipt используют fast replay: worker по-прежнему разбирает IPC и
проверяет точную physical schema и число строк, но не повторяет digest и value
scans на каждой epoch. При закрытом input replay заранее готовит не более
одного следующего CPU batch, пока текущий batch обучается. Для открытой нулевой
epoch prefetch не используется, поэтому порядок долговечного control channel
и EOF остаётся синхронным.

Изменение разбиения payload-ов или времени upload не должно менять ML
trajectory. Критерий go/no-go:

```text
одинаковый ordered dataset
+ seed
+ конфигурация с deterministic=true
+ одинаковые hardware/runtime
-> одинаковый порядок rows и shuffle
-> одинаковые optimizer steps
-> одинаковое ML-state после каждой epoch
-> семантически эквивалентный checkpoint
-> одинаковая итоговая модель
```

Сравниваются ML-метрики и optimizer steps; wall-clock telemetry, включая
timestamps, elapsed time и latency, исключается. Checkpoint сравнивается после
десериализации: model, optimizer, scaler, training state, RNG state, shuffle
state, early-stopping state и checkpoint selection. SHA-256 файла checkpoint
не является критерием эквивалентности, поскольку сериализация не обязана быть
canonical.

## Восстановление и idle timeout

Внутри открытой global epoch нет безопасной границы checkpoint. Если worker
падает до EOF, незавершённая нулевая epoch целиком повторяется с начала.
Committed input artifacts сохраняются. Model, optimizer и random state
восстанавливаются к началу незавершённой epoch, поэтому optimizer step не
применяется дважды. После закрытия input восстановление выполняется на
границах полных global epochs.

`inputIdleTimeout` защищает GPU от заброшенного открытого потока. Он активен
только когда worker явно подтвердил ожидание следующего непрерывного ordinal.
Timer сбрасывается только при продвижении этого непрерывного frontier.
Out-of-order commit не продлевает timeout. Успешный takeover владения даёт
ограниченный grace period, но не превращает посторонние мутации в input
activity. Этот timeout не зависит от жёстких сроков выполнения subprocess и
отмены.

## ML-контракт данных и схемы Arrow

Inventory владеет документом semantic data contract. Create передаёт его
стабильную identity:

```json
{
  "dataContract": {
    "id": "inventory.learning-dataset",
    "version": 1,
    "dataContractSha256": "hex",
    "seqLen": 10,
    "featureDim": 891,
    "targetSchemaId": "inventory.target.v1"
  }
}
```

Canonical документ Inventory включает упорядоченные identities features,
семантику targets, normalization, missing-value policy и версию профиля.
Transformer хранит и возвращает identity и hash, но не воспроизводит и не
интерпретирует семантику features Inventory. Везде используется термин
`dataContractSha256`. Predict обязан передать hash, сертифицированный выбранной
моделью; иначе create до upload завершается с `MODEL_SCHEMA_MISMATCH`.

Breaking Arrow schemas:

```text
inventory.sequence.fit.v2
  src: FixedSizeList<Float32>[seqLen * featureDim]
  tgt: FixedSizeList<Float32>[6]

inventory.sequence.predict.v2
  src: FixedSizeList<Float32>[seqLen * featureDim]

transformer.prediction.v2
  <predictionColumn>: FixedSizeList<Float32>[6]
```

Фиксированные типы определяют dimensions даже для typed-empty payload-ов.

## Жизненный цикл модели

Опубликованные `modelRef` и generation неизменяемы и не имеют автоматического
TTL. `modelAlias` остаётся owner-scoped и во время `job.create` атомарно
разрешается в один `resolvedModelRef`. `predictionColumn` является параметром
job, а не свойством модели.

`model.describe` возвращает неизменяемую identity модели, generation, digest
checkpoint, конфигурацию модели и identity сертифицированного data contract.
Стабильные model errors:

- неизвестная видимая owner identity: `NOT_FOUND`;
- строка модели существует, но checkpoint отсутствует: `MODEL_UNAVAILABLE`;
- digest или checkpoint некорректен: `MODEL_CORRUPT`;
- несовместимый или несертифицированный data contract:
  `MODEL_SCHEMA_MISMATCH`.

Модели, созданные до v3, не считаются совместимыми автоматически, поскольку у
них нет сертифицированного `dataContractSha256`. Они должны быть переобучены
или сертифицированы отдельной аудируемой offline migration. Сертификация не
переписывает незаметно исторические неизменяемые metadata модели.

## Контракт процесса worker v2

Worker v2 получает неизменяемый base manifest с identity job, attempt, model,
training и data contract, а также snapshot уже committed непрерывных inputs.
Новые непрерывные inputs и явный EOF передаются через ограниченный control
channel service → worker. Worker не обращается к PostgreSQL и не следит за
каталогом.

Каждое control-сообщение содержит `jobId`, числовой `attempt`, `attemptId`,
строго возрастающий номер сообщения и, когда применимо, ordinal input. Worker
подтверждает наибольший непрерывный ordinal. Повторные control-сообщения
безопасны: уже принятые inputs проверяются как точные дубликаты и никогда не
используются дважды. После восстановления service или worker новая attempt
получает построенный по PostgreSQL свежий snapshot, включая `inputClosed`;
notification delivery является оптимизацией, а не источником истины.

Все progress, checkpoint, output и terminal events продолжают содержать
`attempt`, `attemptId` и sequence отдельного процесса. `attemptId` достаточно
для отклонения поздних сообщений, поскольку владение внутри attempt никогда не
передаётся.

## Обязательные acceptance tests

Автоматические тесты должны покрывать:

- первый committed непустой payload ставит worker в queue при открытом input;
- время upload, out-of-order completion и разбиение payload-ов сохраняют
  логический порядок строк;
- optimizer batches и shuffle windows пересекают границы payload-ов;
- закрытие input завершает нулевую epoch;
- сбой до EOF повторяет незавершённую epoch без двойного применения optimizer;
- takeover отклоняет поздние `DoPut`, close и cancel прежнего owner;
- stale `DoPut` не может перезаписать input artifact следующего owner;
- потерянные create- и acquire-responses повторно возвращают committed
  identity;
- close отклоняет пропуск, неверные итоги и неверный digest;
- несовпадение model schema отклоняется до upload;
- output и model нельзя опубликовать до закрытия input;
- `GetFlightInfo` недоступен до успешного завершения execution;
- pagination возвращает поздний малый ordinal в следующем revision traversal;
- out-of-order commit не продлевает input idle timeout;
- tombstoned `jobId` нельзя использовать повторно;
- потеря notification после commit input и после close восстанавливается из
  PostgreSQL;
- дубли worker control-сообщений не приводят к повторному использованию
  данных;
- deterministic запуски с закрытым input и задержанным потоком удовлетворяют
  указанному выше условию семантической ML-эквивалентности.

Race tests используют явную синхронизацию на границах reserve, publication
artifact, повторной проверки fence, commit input, notification и terminal
publication; wall-clock sleeps не являются механизмом арбитража.

## Переключение

Операционное переключение выполнялось так:

1. остановить workers Inventory и Transformer v2;
2. применить breaking migration PostgreSQL Transformer;
3. удалить v2 jobs, inputs, attempts, tickets, idempotency и recovery state;
4. сохранить API access tokens;
5. оставить записи прежних моделей несертифицированными до явной offline
   certification либо переобучить модели;
6. развернуть только Flight v3 и worker v2;
7. рекламировать ровно `protocolVersions: [3]`;
8. запустить Inventory с обязательным v3 без fallback;
9. до восстановления traffic проверить health, fencing create/acquire,
   streaming fit и доступ к terminal output.

Очистка filesystem выполняется как reconciliation после database boundary, а
не внутри transaction PostgreSQL. Старые и неуспешные candidate artifacts
могут временно оставаться orphan, но PostgreSQL никогда не ссылается на
частичный файл.

## Последствия

- Обучение может перекрываться с долговечным upload без привязки
  восстановления к сетевой сессии.
- Внешнее владение становится безопасным для нескольких workers Inventory.
- Состояния, pagination и совместимость моделей становятся явными публичными
  контрактами.
- Supervision worker получает ограниченный двунаправленный прикладной протокол
  и обязан восстанавливать потерянные notifications из snapshots PostgreSQL.
- Реализация существенно сложнее v2, а semantic deterministic equivalence
  становится release gate.
