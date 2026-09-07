# Интеграция Consumer-ов с Transformer Arrow Flight v13

> Тип: интеграционное руководство. Durable workflow Consumer-а поверх
> текущего consumer-neutral протокола.

Нормативный wire-контракт находится в
[`app/contracts/flight/v13`](../app/contracts/flight/v13/README.md), а
математический язык — в
[`app/contracts/semantic/v1`](../app/contracts/semantic/v1/README.md). Их JSON
Schemas и golden fixtures имеют приоритет над этим руководством. Эксплуатация
сервиса описана в [Flight runbook](operations/flight-service.md), credentials —
в [операционном руководстве](operations/api-access-tokens.md).

Flight v13 — единственный remote API. Actions v11, dense fallback и
compatibility layer отсутствуют. Model aliases сохраняются только как selector
predict; catalog detail принимает exact `modelRef`.

## Транспорт и аутентификация

Используйте Arrow Flight `DoAction`, `DoPut`, `GetFlightInfo` и `DoGet`. Каждый
RPC содержит:

```text
authorization: Bearer a.<base64url>
```

Credential представляет owner subject; job и model resources изолированы по
этому owner-у. Не записывайте bearer credential или непрозрачный output ticket
в логи. Job workflow JSON envelope содержит `contract=transformer-flight`,
`version=13` и canonical UUID `requestId`. Model Catalog actions используют
независимые `contract=transformer-model-catalog`, `revision=1` и тот же формат
`requestId`. Training Telemetry actions аналогично используют независимые
`contract=transformer-training-telemetry` и `revision=1`.

Для мутации заранее сохраните стабильный `idempotencyKey` и неизменный request.
При потере ответа повторяйте то же действие с тем же содержимым; ключ нельзя
переиспользовать для другой операции.

## Capabilities и device

До create вызовите `transformer.v13.capabilities` и проверьте как минимум:

- `protocolVersions == [13]` и `workerProtocolVersion == 12`;
- checkpoint/recovery v6 и оба metrics format v5;
- `sourceEncodings == ["indexedFeatureBlocks"]`;
- objective language revision, advertised primitives, architecture и capacity
  limits внутри `semantic`;
- `fitInitializations == ["publishedModel", "random"]`;
- capability document `modelCatalog` с revision, actions и limits;
- capability document `trainingTelemetry` с revision, actions, outcomes и limits;
- effective upload/job limits и доступные devices.

Consumer не должен копировать capacity defaults из repository. Create принимает
`device: "cpu"`, `"gpu"` или `"auto"`. Явный `gpu` требует доступный GPU и не
подменяется CPU; `auto` выбирает GPU при наличии. Public значение `cuda`
недопустимо: concrete CUDA device является внутренней деталью worker runtime.

## Materialization contract до create

До Flight boundary Consumer формирует self-contained документы без server-side
lookup в Consumer catalog:

- `dataContract` — opaque identity/revision/profile, Consumer-owned digest и
  точные `seqLen/featureDim`;
- `modelContract` — objective language revision, ordered TargetContract,
  Objective и model configuration;
- `sourceEncoding` — физическая geometry ordered feature blocks.

Target slot содержит opaque identity, observed constraint, loss-input и public
prediction transformations. Objective явно содержит direct/auxiliary
components, typed role bindings, weights, parameters и abstract private
resources. Transformer валидирует закрытый язык и geometry, но не
интерпретирует target identity, profile, instruments, intervals или formulas.

`seqLen/featureDim` в data и model contracts должны совпадать, а feature blocks
без разрывов покрывают `featureDim`. Transformer вычисляет и возвращает D1
layers `dataContractSha256`, `targetContractSha256`, `objectiveSha256` и
`modelContractSha256`, а также resolved `jobConfigSha256`. Полные canonical
documents, а не digests, остаются источниками смысла.

`sourceEncoding` входит в job configuration, receipts и recovery fencing, но
не в target/objective/model identity. Consumer по-прежнему владеет всеми
feature values и causal projections; Transformer только детерминированно
восстанавливает logical `x`.

Training policy и diagnostics передаются отдельно от Objective. Diagnostics
включает best-effort наблюдения и не меняет loss или model compatibility.

## Инициализация fit

Независимое обучение использует:

```json
{"kind":"random"}
```

Weights-only warm start использует owner-scoped immutable reference:

```json
{"kind":"publishedModel","modelRef":"mdl_..."}
```

`publishedModel` требует точного совпадения data, target, objective и model D1
layers. Другой временной период допустим только если его границы не изменяют
Consumer data digest. Transformer загружает полный parent `state_dict`, но
создаёт новые optimizer, AMP scaler, RNG, progress и selection state. Parent
остаётся неизменным; lineage хранит parent model/checkpoint identities.

## Создание job и fencing

До первого сетевого запроса Consumer надёжно сохраняет:

- client-generated `jobId` логического job;
- `clientExecutionId` текущего claim;
- create `idempotencyKey` и точный canonical request.

Fit create содержит `modelLabel`, `initialization` и optional resolved
training/diagnostics policy. Predict содержит ровно один `modelRef` или
`modelAlias`; Transformer атомарно возвращает immutable `resolvedModelRef`.
Начальное состояние:

```text
input.state     = OPEN
execution.state = WAITING_INPUT
fencingToken    = "1"
```

Сохраните create result вместе с resolved contracts, D1 digests, initialization,
ownership и limits. При takeover вызовите `transformer.v13.job.acquire` с
предыдущими execution identity/token и новым `clientExecutionId`. Token —
положительная десятичная строка. Старый claim получает `STALE_FENCE`.

## Durable compact upload

Один DoPut — один physical payload; каждая top-level Arrow row — self-contained
chunk одного range. Для каждого ordered feature block Consumer передаёт:

- рассчитанные native Float32 rows с необходимым halo;
- локальные `observationOffsets` всех sequence positions;
- для fit — `tgt` по одному ordered vector на logical example.

Назначайте `rangeOrdinal` плотно от нуля только успешно переданным ranges. При
split одного range продолжайте `exampleOffset` без пропусков и дубликатов.
Sequence не пересекает range, а offsets не выходят за локальные native rows.
Application metadata объявляет `chunks`, `logicalRows` и `nativeRows[]`.

RecordBatch, chunk и payload boundaries не меняют logical order, shuffle,
optimizer batches или values. `maxJobBytes` применяется к durable compact IPC
bytes, а не к восстановленному dense tensor. Точные Arrow schemas и формула
reconstruction находятся в
[Flight contract](../app/contracts/flight/v13/README.md#arrow-и-indexedfeatureblocks),
а provider-neutral fixtures — в `app/contracts/flight/v13/fixtures/`.

PutResult появляется только после durable artifact и PostgreSQL receipt.
Завершение DoPut не по ordinal разрешено. `nextInputOrdinal` показывает первый
пропуск непрерывного префикса, `inputRevision` — commit revision job.

## Сверка и закрытие input

После потерянного PutResult перечислите receipts через
`transformer.v13.job.inputs.list`. Первая страница фиксирует
`snapshotRevision`; продолжайте с тем же snapshot и cursor. Повторный exact
payload возвращает существующий receipt, а другое содержимое для занятой
identity/ordinal является конфликтом.

`transformer.v13.job.input.close` обозначает EOF, а не команду запуска. Close
передаёт canonical digest всех server receipts в ordinal order. До перехода в
`CLOSED` сервер проверяет непрерывность payload/range/example sequence, counts,
digest, единую schema и data identity. Пустой fit возвращает `EMPTY_INPUT`;
пустой predict допустим.

Fit может уже быть `RUNNING` при `input.state=OPEN`: первый непустой contiguous
prefix ставит execution в очередь, а epoch 0 ожидает последующие committed
payloads до EOF.

## Каталог опубликованных моделей

Catalog truth определяется PostgreSQL registry, а не OpenSearch telemetry или
filesystem scan. Вызов `transformer.model-catalog.v1.list` возвращает только
`AVAILABLE` generations аутентифицированного owner в порядке
`createdAt DESC, modelRef ASC`. Первая страница передаёт `cursor=null`; для
продолжения повторите выданный `nextCursor` и тот же `pageSize`. Cursor действует
900 секунд от первой страницы и не продлевается.

List summary содержит exact `modelRef`, label/generation, D1 digests, resolved
model configuration, ordered opaque target identities, initialization,
producing run и checkpoint summary. List не читает checkpoint bytes.

`transformer.model-catalog.v1.detail` принимает только exact `modelRef` и
возвращает canonical data/model contracts, training/diagnostics configuration,
selection, terminal progress, lineage и `jobConfigSha256`. Перед успешным
ответом Transformer проверяет metadata и полный SHA-256 checkpoint. Unknown,
foreign, deleted model и удаление между list/detail возвращают одинаковый
`MODEL_NOT_FOUND`; начните новый traversal после обновления UI. Полные schemas,
limits и structured outcomes задаёт
[Model Catalog Query v1](../app/contracts/model_catalog/v1/README.md).

## Training telemetry опубликованной модели

`transformer.training-telemetry.v1.report` принимает exact `modelRef` и
возвращает owner-scoped training report. Transformer сам разрешает
authoritative producing run из registry; owner и run identity в request не
передаются. Unknown, foreign и deleted model неразличимы.

Report возвращает `pending`, окончательный `unavailable` либо только
полностью проверенный `available`. Epochs идут по возрастанию и
выдаются signed cursor-ом с TTL 900 секунд. Метрики описывают
training-проход эпохи, а не повторную оценку published checkpoint.

`transformer.training-telemetry.v1.gradient-interactions` лениво возвращает
components и sparse oriented pairs одной exact epoch. Отсутствующие
observations являются обычным result. Точные schemas, outcome precedence,
pagination и structured errors задаёт
[`Training Telemetry Query v1`](../app/contracts/training_telemetry/v1/README.md).

## Polling, recovery и results

Status возвращает независимые input/execution axes. Уважайте `pollAfterMs` и
считайте успехом только `CLOSED + SUCCEEDED` после атомарной публикации.
Recovery checkpoints создаются на границах завершённых global epochs и fenced
по exact semantic digests, job config, input manifest/revision и progress.
Сбой открытой epoch 0 повторяет её с начала; durable inputs сохраняются.

После успешного predict:

1. Получите страницы `transformer.v13.job.outputs.list`.
2. Для каждого ordinal вызовите `GetFlightInfo` с нормативным descriptor.
3. Используйте новый opaque ticket в `DoGet` до его expiration.

Prediction имеет finite Float32 width, равную ordered target slots checkpoint-а.
Координаты уже прошли declared public transformations. Consumer сопоставляет их
по позиции и ordered opaque identities из catalog detail; private resources
наружу не выходят. `predictionColumn` принадлежит predict job, а не модели.

## Ошибки и retries

Human-readable Flight message начинается с application error code, а
`FlightError.extra_info` содержит structured v13 error detail. Consumer должен
ветвиться по `code`, `reason` и typed fields, а не по тексту message. Invalid
contract, unavailable primitive, target value violation, compatibility
mismatch, stored corruption и recovery fencing имеют разные reasons.

| Потерянный или неуспешный шаг | Поведение Consumer-а |
| --- | --- |
| Ответ create/acquire/close/cancel | Повторить ту же мутацию с тем же idempotency key |
| Ответ DoPut | Сверить receipts и повторить exact payload, если он отсутствует |
| Status/job list/catalog list или detail | Повторить read-only request с учётом cursor expiration |
| Истечение ticket | Получить новый ticket после terminal success |
| Прерывание DoGet | Получить новый ticket и начать данный output заново |
| `STALE_FENCE` | Прекратить мутации старого claim |

`requestId` может меняться между транспортными попытками. Job, payload,
ownership и idempotency identities меняться не должны.

## Проверка интеграции

1. Проверьте exact v13 capabilities и отказ для v12 actions.
2. Пересчитайте JCS/D1 golden fixtures независимо от Transformer runtime.
3. Проверьте generic regression, probability, shared-resource objective,
   target reorder и новый opaque target без изменения Transformer.
4. Проверьте invalid target values, unavailable primitive и отдельные digest
   mismatch layers до upload.
5. Проверьте идемпотентный create, takeover и reconciliation потерянного
   PutResult.
6. На cross-project fixtures сравните logical tensor для single/multi-block,
   hour boundary, split range, missing observation и missing native prefix.
7. Проверьте random fit, strict publishedModel warm start, recovery и новый
   immutable modelRef.
8. Проверьте positional decoding prediction по checkpoint-owned target layout.
9. Проверьте owner isolation, high-water traversal, cursor expiration и
   физическую detail verification по Model Catalog fixtures.
10. Проверьте available/pending/unavailable telemetry, epoch pagination,
    deletion during traversal и sparse gradient pairs по Training Telemetry fixtures.
