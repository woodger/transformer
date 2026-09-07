# Контракт owner-scoped Training Telemetry Query v1

> CONTRACT DOCUMENT. Этот package задаёт нормативные JSON documents,
> pagination semantics, capabilities и structured errors read-only доступа к
> telemetry успешного fit, активированного Flight v13.

JSON Schemas Draft 2020-12 и перечисленные manifest-ом golden fixtures являются
источником истины для формы документов. Этот README задаёт семантические
инварианты, которые JSON Schema выразить не может. Активация package добавляет
два Flight actions; PostgreSQL schema, metrics v5 и OpenSearch projection не
изменяются.

## Назначение и версия

Query имеет независимые identity `transformer-training-telemetry` и immutable
`revision=1`. Он возвращает только полностью проверенную training telemetry
одной опубликованной model generation:

- total, direct и auxiliary losses по эпохам;
- training MAE/RMSE каждого ordered target slot;
- first, best и published epoch milestones;
- optimizer/AMP/gradient health counters;
- optional gradient interactions отдельным ленивым запросом.

Query не возвращает model-test observations, out-of-sample quality, telemetry
незавершённых jobs, произвольный analytics query или каталог моделей. Он не
проверяет физический checkpoint и не доказывает его целостность.

Revision 1 использует две `DoAction` identity:

```text
transformer.training-telemetry.v1.report
transformer.training-telemetry.v1.gradient-interactions
```

Первичная активация выполнена Flight v13. Query revision не обязана меняться
вместе с job workflow, metrics projection, Worker или checkpoint format.

Batch query отсутствует. Inventory ограничивает сравнение четырьмя моделями и
выполняет bounded single-model запросы самостоятельно.

## Authentication и model lookup

Request принимает exact immutable `modelRef`, но не owner и не run identity.
Transformer выводит owner scope из authenticated subject, разрешает только
`AVAILABLE` generation и сам получает authoritative `producingRunId`.

Unknown, foreign и deleted model дают security-equivalent `MODEL_NOT_FOUND`.
Сохранившаяся telemetry удалённой модели не восстанавливает доступ к ней.
Каждый initial или continuation request сначала повторяет owner-scoped registry
lookup; удаление модели имеет приоритет над действующим cursor.

После lookup Transformer проверяет canonical stored model metadata и D1
digests. Ошибка этой проверки возвращает `STORED_MODEL_METADATA_INVALID` до
обращения к telemetry backend. Полный SHA-256 checkpoint для telemetry query не
считается.

## Семантика epoch-pass metrics

Training telemetry не является повторной оценкой опубликованного checkpoint.
Для каждой партии loss и target errors вычислены на состоянии модели до
соответствующего optimizer update. Наблюдения затем агрегированы по всем
training rows одного прохода эпохи.

Следовательно:

- MAE/RMSE сравнивают observed `y` с
  `publicPredictionTransformation(raw)` в момент обработки каждой партии;
- loss values описывают тот же проход с изменяющимися внутри эпохи weights;
- milestone epoch связывает observation прохода с checkpoint, выбранным после
  завершения этой эпохи, но не является отдельной checkpoint evaluation;
- для `published` milestone UI использует термин «метрики прохода
  опубликованной эпохи», а не «метрики опубликованного checkpoint».

Это выражено capability
`epochObservation.kind=PreOptimizerUpdateEpochPass` и
`checkpointReevaluation=false`.

## Порядок outcomes

Report применяет следующий нормативный порядок:

1. Owner-scoped registry lookup и canonical model metadata validation.
2. Telemetry completion marker ещё не достигнут и продолжение доставки либо
   materialization достоверно ожидается —
   `pending / MATERIALIZATION_PENDING`.
3. Telemetry completion marker отсутствует и продолжение достоверно не
   ожидается —
   `unavailable / NO_COMPLETE_REPORT`.
4. Completion marker существует, но epochs или observations после provider
   consistency handling неполны либо противоречивы —
   `TRAINING_TELEMETRY_INTEGRITY_FAILED`.
5. Только полностью проверенная telemetry возвращается как `available`.

`pending` относится только к построению report уже опубликованной generation и
не открывает telemetry незавершённого fit. Временная недоступность необходимого
query backend является structured RPC error, а не `pending` или `unavailable`.
`RETENTION_EXPIRED` допустим только
если Transformer может доказать retention outcome; иначе окончательное
отсутствие обозначается `NO_COMPLETE_REPORT`. `FORMAT_UNSUPPORTED` означает,
что report найден, но revision 1 не имеет reader-а его immutable format.

Partial numerical report запрещён. Ошибка сбора отдельной эпохи не меняет fit
outcome, но делает весь report недоступным, если complete projection так и не
была сформирована.

## Report request и pagination

Initial request содержит `cursor=null`. Continuation повторяет exact
`modelRef` и `pageSize` и передаёт provider-issued opaque cursor. Epochs
возвращаются строго по `epoch ASC`; initial page начинается с epoch 1, а
continuation — со следующей epoch после cursor boundary.

Epoch cursor integrity-protected и связан как минимум с:

- authenticated owner;
- model и authoritative producing run;
- query revision и immutable report identity;
- page size и последней возвращённой epoch;
- абсолютным expiration timestamp.

TTL равен 900 секундам от первой страницы и не продлевается. Нетерминальная
страница возвращает `nextCursor` и `cursorExpiresAt`; terminal page возвращает
для обоих полей `null`. Provider удерживает underlying immutable report до
expiration выданного cursor. Исключение — model deletion: continuation
возвращает `MODEL_NOT_FOUND`.

Report page содержит не более 100 epochs. Полный serialized response не
превышает 8 MiB. Превышение возвращает
`TELEMETRY_RESPONSE_BUDGET_EXCEEDED`, а не усечённый result; Consumer может
повторить initial traversal с меньшим `pageSize`.

## Available report

Успешный report повторяет на каждой странице immutable header:

- exact `modelRef` и authoritative `producingRunId`;
- четыре D1 semantic digests;
- полную coverage `1..completedEpochs`;
- selection и milestone anchors;
- health totals по всем epochs;
- gradient-interaction descriptor.

`semanticDigests` должны точно совпадать с registry-owned model metadata.
Telemetry не добавляет новый D1 layer и не участвует в predict или warm-start
compatibility.

### Selection и anchors

Transformer не вычисляет `bestEpoch` повторно из telemetry:

- при включённой selection `bestEpoch` берётся из model metadata,
  `publishedEpoch=bestEpoch`, а `publishedSource` равен
  `best_direct_selection_score`;
- при выключенной selection `bestEpoch=null`,
  `publishedEpoch=completedEpochs`, а source равен `last_epoch`.

`anchors` содержит полные epoch metrics для уникальных first, best и published
epochs. Совпадающие milestones объединяются в один anchor. Anchors идут по
`epoch ASC`; roles внутри anchor идут в порядке `first`, `best`, `published` с
пропуском неприменимых ролей. Epoch также остаётся на своём обычном месте в
paginated sequence — anchor не удаляет и не заменяет page item.

### Одна epoch

Каждый epoch item содержит:

- `epoch`, job-wide `globalStep` и фактический `attempt`;
- authoritative observation weighted `totalLoss`;
- optional authoritative observation `selectionScore`;
- unweighted `GlobalRowMean` каждого direct и auxiliary component;
- MAE/RMSE каждого target;
- optimizer/AMP/gradient health counters;
- признак наличия gradient-interaction observations.

Инварианты:

- epochs образуют плотный диапазон `1..completedEpochs`, включая recovery;
- `globalStep` строго возрастает;
- direct losses и target metrics следуют ordered target slot layout;
- auxiliary losses следуют canonical Objective order;
- component/operator/target references точно разрешаются в checkpoint-owned
  ModelContract;
- direct, auxiliary, total, selection, MAE и RMSE finite; auxiliary loss может
  быть отрицательным;
- `selectionScore` семантически является weighted sum только direct losses при
  включённой selection и равен `null` при выключенной;
- `totalLoss` семантически является weighted total Objective;
- все direct и auxiliary component values являются unweighted means.

`totalLoss`, `selectionScore` и component values являются отдельными
authoritative observations, агрегированными Worker-ом для одной epoch. Query
проверяет их наличие и finite representation, но не восстанавливает один
aggregate из других и не применяет к ним cross-field equality predicate.
Причина — total и component means накапливаются независимо, а порядок и
округление floating-point операций являются implementation detail. Их
различие на последнем binary64 разряде само по себе не означает telemetry
corruption. Revision 1 поэтому не задаёт `atol`, `rtol`, dtype внутреннего
accumulator-а или межъязыковой порядок сложения.

Health каждой epoch и `healthTotals` удовлетворяют:

```text
trainingBatchesCompleted
  = optimizerUpdatesApplied + optimizerUpdatesSkipped
  = finiteGradientBatches + nonFiniteGradientBatches

ampOverflowBatches <= optimizerUpdatesSkipped
ampOverflowBatches <= nonFiniteGradientBatches
```

`healthTotals` является точной покомпонентной суммой всех epochs, а не только
текущей страницы. AMP enablement читается Inventory из Model Catalog detail;
нулевой overflow сам по себе не доказывает, что AMP был выключен.

Inventory вычисляет абсолютную разницу milestone values как
`laterValue - earlierValue`. Процентная разница не входит в revision 1.

## Gradient interactions

Основной report не переносит потенциально квадратичный список component pairs.
Он возвращает одно из состояний:

- `notConfigured` — diagnostics не были включены;
- `configuredWithoutObservations` — diagnostics включены, но observations нет
  ни для одной epoch;
- `available` — хотя бы одна epoch имеет observations.

При `available`:

- `collectedEpochCount` равен числу epochs с
  `gradientInteractionsCollected=true`;
- `publishedEpochCollected` показывает наличие observations published epoch;
- `defaultEpoch` выбирается как published epoch, если она собрана; иначе как
  `max(collectedEpoch < publishedEpoch)`; если предыдущей нет — как
  `min(collectedEpoch > publishedEpoch)`.

Отдельный request выбирает exact epoch. Components возвращаются полностью в
Objective execution order: direct components в slot order, затем auxiliary в
canonical Objective order. Direct component содержит target identity/index;
auxiliary component их не выдумывает.

Для каждой unordered пары objective components ориентация определяется один
раз: `left` — component, расположенный раньше в Objective execution order,
`right` — расположенный позже. Self-pairs и reversed duplicates запрещены;
обе identities обязаны разрешаться в возвращённом ordered `components`.

Gradient pair result является разреженным. Пара публикуется, только если в
epoch для неё существует хотя бы одна finite cosine observation. `meanCosine`
и `negativeCosineFraction` агрегируются только по этим finite observations.
Если norm одного из gradients равен нулю, cosine этой observation не определён
и не подменяется нулём. Отсутствующая пара означает отсутствие finite cosine
observations и не является ни нулевым cosine, ни telemetry corruption.

Каждая присутствующая oriented пара уникальна. После определения orientation
пары сортируются по ASCII tuple
`(leftComponentIdentity, rightComponentIdentity)` и выдаются страницами до
1000 элементов. Pair cursor дополнительно связан с exact epoch и последней
pair identity. Components повторяются на каждой pair page и не меняются.

Запрос epoch без observations возвращает обычный result `notCollected`, а не
RPC error. Reasons revision 1: `DIAGNOSTICS_NOT_CONFIGURED` и
`NO_OBSERVATIONS_FOR_EPOCH`. UI вызывает action только после раскрытия
доступной diagnostics section и использует per-epoch collected flag.

## Полная проверка available

До выдачи любой страницы available Transformer обязан:

1. проверить model/run/D1 identities;
2. доказать наличие всех epochs `1..completedEpochs`;
3. исключить duplicate epoch и metric observations;
4. проверить ordered target/component references по ModelContract;
5. проверить finite authoritative observations и health invariants, не
   выполняя обратный arithmetic proof между aggregates;
6. проверить anchors, health totals и gradient descriptor по полной
   последовательности;
7. построить bounded public projection.

Completion marker не отменяет эту проверку. Marker с пропущенной epoch,
неверным digest, run или component reference приводит к telemetry corruption.
Способ хранения и чтения report остаётся внутренним: index names, mappings,
document IDs, filesystem paths и retention topology публичную границу не
пересекают.

## Structured errors

Application `code` и `reason` передаются как exact JSON `extra_info`. Consumer
ветвится по ним, а не по тексту `message`.

| Ситуация | `code` | `reason` |
| --- | --- | --- |
| Invalid request | `INVALID_ARGUMENT` | `INVALID_TELEMETRY_QUERY` |
| Invalid/foreign cursor | `INVALID_ARGUMENT` | `INVALID_TELEMETRY_CURSOR` |
| Expired cursor | `FAILED_PRECONDITION` | `TELEMETRY_CURSOR_EXPIRED` |
| Unsupported revision | `FAILED_PRECONDITION` | `TELEMETRY_QUERY_REVISION_UNAVAILABLE` |
| Unknown/foreign/deleted model | `NOT_FOUND` | `MODEL_NOT_FOUND` |
| Invalid stored model metadata | `MODEL_CORRUPT` | `STORED_MODEL_METADATA_INVALID` |
| Contradictory complete telemetry | `TELEMETRY_CORRUPT` | `TRAINING_TELEMETRY_INTEGRITY_FAILED` |
| Backend unavailable | `UNAVAILABLE` | `TRAINING_TELEMETRY_BACKEND_UNAVAILABLE` |
| Response exceeds 8 MiB | `RESOURCE_EXHAUSTED` | `TELEMETRY_RESPONSE_BUDGET_EXCEEDED` |

Pending, final absence of telemetry and absence of gradient observations are
normal result documents.

## Capabilities

`capabilities.schema.json` задаёт закрытый block с query identity/revision,
action identities, metric families, availability states, epoch observation
semantics, outcome precedence, limits и supported features. Hosting Flight
capabilities включает этот block только после фактической runtime activation.

Отсутствующий block означает, что query не поддержан. Consumer не угадывает
action version, не читает OpenSearch напрямую и не использует metrics v5 как
fallback API.

## Cross-project fixtures

`fixtures/manifest.json` перечисляет byte-identical golden documents, их
SHA-256 и conformance cases. Paths и case identities отсортированы ASCII;
manifest не включает сам себя. `manifest.sha256` содержит digest exact bytes
manifest-а.

Fixtures покрывают:

- selection enabled/disabled и объединённые milestone roles;
- pre-update epoch-pass semantics без checkpoint reevaluation;
- independently accumulated authoritative aggregates без cross-field equality;
- opaque multi-target layout, negative auxiliary loss и recovery attempts;
- AMP overflow/skipped/non-finite counters;
- все gradient availability states, lazy pair pagination, zero-norm с
  отсутствующими sparse pairs и три ветви `defaultEpoch`;
- pending и доказанное final unavailable;
- completion marker с missing epoch и invalid D1/run/component references;
- security-equivalent unknown/foreign model и deletion during traversal;
- cursor expiration и response budget.

Inventory хранит byte-identical offline copy и независимо проверяет schemas,
semantic invariants и manifest digests до runtime implementation.
