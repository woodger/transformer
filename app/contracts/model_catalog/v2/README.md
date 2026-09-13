# Контракт owner-scoped Model Catalog Query v2

> CONTRACT DOCUMENT. Этот package задаёт нормативные JSON documents,
> pagination semantics, capabilities и structured errors read-only каталога
> опубликованных моделей.

JSON Schemas Draft 2020-12 и перечисленные manifest-ом golden fixtures являются
источником истины для формы документов. Этот README задаёт семантические
инварианты, которые JSON Schema выразить не может.

## Версия и Flight binding

Query имеет независимые identity `transformer-model-catalog` и immutable
`revision=2`. Он активирован вместе с Flight v14. Query revision не обязана
меняться синхронно с Flight workflow.

Revision 2 связывается с двумя `DoAction`:

```text
transformer.model-catalog.v2.list
transformer.model-catalog.v2.detail
```

Hosting Flight capabilities включает document из
`schemas/capabilities.schema.json`. Отсутствующий query либо запрошенная
неподдерживаемая revision возвращают capability-unavailable outcome, а не
fallback к admin CLI, OpenSearch или filesystem scan.

Batch detail, filters и server-side comparison в revision 2 отсутствуют.
Добавление batch detail после первой интеграции требует новой query revision;
оно не меняет семантику single detail и не вводит сравнение на стороне
Transformer.

## Authentication и источник истины

Owner scope всегда выводится из authenticated subject Flight-соединения.
Request не содержит owner identity. List читает только `AVAILABLE` generations
текущего owner из model registry. OpenSearch, telemetry и checkpoint directory
scan не определяют membership.

Unknown, foreign, deleted и недоступный current alias не различаются. Detail
для любого такого `modelRef` возвращает одинаковый `MODEL_NOT_FOUND`; list не
показывает foreign rows. Parent reference в lineage не даёт права dereference:
родитель открывается отдельным owner-scoped detail.

## List request и result

`list-request.schema.json` требует явные `pageSize` и `cursor`. Первая страница
передаёт `cursor=null`; continuation передаёт неизменённый server-issued token.
`pageSize` находится в диапазоне `1..100` и связывается с cursor первого
запроса. Другое значение на continuation классифицируется как
`INVALID_CATALOG_CURSOR`.

Canonical traversal order:

```text
createdAt DESC, modelRef ASC
```

`createdAt` публикуется в UTC с шестью знаками микросекунд. На первой странице
service фиксирует owner-scoped high-water publication boundary. Generations,
зафиксированные после boundary, не входят в traversal. Внутреннее представление
high-water mark не является wire contract.

Cursor имеет форму `mc2.<opaque-base64url>`, integrity-protected service-ом и
связан с authenticated owner, query revision, page size, ordering, high-water
boundary, последним sort key и expiration. Consumer не декодирует token.
Повторное использование чужим owner, подмена, неизвестный key либо malformed
payload дают одинаковый `INVALID_CATALOG_CURSOR`.

TTL равен 900 секундам от формирования первой страницы и не продлевается при
continuation. Result возвращает `nextCursor` и тот же абсолютный
`cursorExpiresAt`, пока traversal продолжается. Terminal page возвращает для
обоих полей `null`. В момент expiration и после него корректно подписанный
cursor возвращает `CATALOG_CURSOR_EXPIRED` с `restartRequired=true`; пустая
страница или автоматический restart запрещены.

Concurrent publication не пересекает high-water boundary. Concurrent deletion
может исключить ещё не возвращённую generation, поэтому строгая snapshot
membership не обещается. Неизменившаяся generation не возвращается повторно.
Consumer после deletion gap либо expiration начинает новый traversal.

List summary содержит:

- `modelRef`, label, generation и creation time;
- четыре D1 semantic digests;
- полный resolved `modelConfig`;
- ordered opaque target identities;
- catalog initialization `source` и optional `parentModelRef`;
- `producingRunId`;
- committed checkpoint format, SHA-256 и byte count.

Полные contracts, training configuration и selection в list отсутствуют.
Checkpoint summary отражает committed registry state: list не читает и не
хэширует checkpoint artifact. Если registry metadata не позволяет построить
валидный summary, вся страница завершается `STORED_MODEL_METADATA_INVALID` без
partial result.

## Single detail

Detail принимает только exact immutable `modelRef`; label и alias не являются
selector-ами. Result содержит summary и дополнительно:

- canonical `dataContract` и `ModelContract`;
- resolved training и diagnostics configuration;
- final selection summary и terminal training progress;
- exact resolved initialization lineage;
- `jobConfigSha256`.

Перед result service проверяет stored canonical documents, D1 digests и их
согласованность с summary, затем проверяет checkpoint path внутри managed
storage, format, exact byte count и полный SHA-256. Summary внутри успешного
detail поэтому относится к физически проверенному artifact. Checkpoint path,
optimizer/scaler/RNG state, private resource values и metric points наружу не
передаются.

Дополнительные semantic invariants detail:

- `summary.modelConfig` равен `modelContract.modelConfig`;
- `summary.targetIdentities` равен ordered identities
  `modelContract.targetContract.slots`;
- `summary.semanticDigests` точно соответствует data и D1 model layers;
- `selection.modelContractSha256` равен model digest summary;
- `CatalogInitializationSummary` является точной проекцией checkpoint-owned `ResolvedInitialization`;
- `summary.producingRunId` соответствует immutable producing job identity;
- `progress.trainingComplete=true` и published checkpoint принадлежит этому
  terminal fit.

## Bounded work

Revision 2 фиксирует следующие пределы:

```text
maxPageSize                         100
cursorTtlSeconds                    900
maxResponseBytes                8388608  (8 MiB)
maxCheckpointVerificationsPerDetail  1
maxCheckpointVerificationBytes 1073741824 (1 GiB)
```

List выполняет bounded keyset query `pageSize + 1` и не читает checkpoint
bytes. Single detail проверяет не более одного artifact. Registered byte count
сначала сравнивается с 1 GiB budget; превышение возвращает
`CHECKPOINT_VERIFICATION_BUDGET_EXCEEDED` до filesystem read. Runtime,
активирующий catalog, обязан применять тот же верхний предел при publication,
чтобы новая `AVAILABLE` generation не оказалась недоступной для detail.

JSON result сериализуется до commit ответа и не превышает 8 MiB. Превышение
возвращает `CATALOG_RESPONSE_BUDGET_EXCEEDED`, не усечённый document.

## Structured errors

Application code и `reason` передаются в `FlightError.extra_info` как exact
UTF-8 JSON по `error-detail.schema.json`. Consumer ветвится по ним, а не по
`message`.

Основные outcomes:

| Ситуация | `code` | `reason` |
| --- | --- | --- |
| Невалидный request/page size | `INVALID_ARGUMENT` | `INVALID_CATALOG_QUERY` |
| Подменённый, foreign или malformed cursor | `INVALID_ARGUMENT` | `INVALID_CATALOG_CURSOR` |
| Истёкший valid cursor | `FAILED_PRECONDITION` | `CATALOG_CURSOR_EXPIRED` |
| Query revision не advertised | `FAILED_PRECONDITION` | `CATALOG_QUERY_REVISION_UNAVAILABLE` |
| Unknown/foreign/deleted model | `NOT_FOUND` | `MODEL_NOT_FOUND` |
| Повреждённая stored metadata | `MODEL_CORRUPT` | `STORED_MODEL_METADATA_INVALID` |
| Artifact отсутствует | `MODEL_CORRUPT` | `MODEL_CHECKPOINT_UNAVAILABLE` |
| Размер или digest не совпал | `MODEL_CORRUPT` | `MODEL_CHECKPOINT_SIZE_MISMATCH` / `MODEL_CHECKPOINT_DIGEST_MISMATCH` |
| Verification/response budget превышен | `RESOURCE_EXHAUSTED` | профильный budget reason |
| Registry временно недоступен | `UNAVAILABLE` | `MODEL_REGISTRY_UNAVAILABLE` |

Detail не возвращает partial document. Поскольку batch detail отсутствует,
per-item outcomes в revision 2 не определены.

## Cross-project fixtures

`fixtures/manifest.json` перечисляет byte-identical request, result,
capabilities и error documents. `fixtures/manifest.sha256` фиксирует SHA-256
самого manifest. Paths отсортированы ASCII; manifest не включает сам себя.

Inventory хранит offline byte-identical copy bundle и проверяет schemas,
ordering, cursor propagation, summary/detail projection и все structured
outcomes. Fixture cursor является только синтаксическим golden token и не
считается криптографически действительным runtime credential.

## Initialization documents

List summary принимает только closed `CatalogInitializationSummary`: `source=random` либо `source=publishedModel` с `parentModelRef`. Detail дополнительно возвращает полный checkpoint-owned `ResolvedInitialization` v7. Requested Flight initialization здесь не принимается. Совпадение bytes random variants не делает эти contract types взаимозаменяемыми.
