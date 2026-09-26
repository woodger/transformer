# Запрос телеметрии обучения в области владельца v4

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет telemetry только для чтения одной
> опубликованной generation модели. Он активируется Flight v20.

Telemetry описывает проход обучения, создавший epoch, а не повторную оценку
весов финального checkpoint-а. Для каждого batch loss и ошибки target-ов
наблюдаются до optimizer update, а затем агрегируются для этой epoch. Это не
данные качества model test или out-of-sample и не доказательство целостности
checkpoint-а.

## Операции и scope owner-а

```text
transformer.training-telemetry.v4.report
transformer.training-telemetry.v4.gradient-interactions
```

Requests принимают точный `modelRef`, но никогда identity owner-а или run-а.
Transformer выводит scope owner-а из authentication и получает авторитетный
producing run. Неизвестные, чужие и удалённые models возвращают одинаковый
`MODEL_NOT_FOUND`. Сохранённый документ telemetry не может сделать удалённую
model видимой.

Перед выдачей report Transformer валидирует принадлежащие registry metadata
model, semantic identities, completion marker, coverage epoch, layout target и
component, конечность чисел и инварианты health counters. Частичный числовой
report никогда не возвращается.

Порядок outcomes: lookup model/validation metadata; `pending` только пока
materialization доказуемо ожидается; `unavailable`, когда полный report не
появится; failure целостности telemetry, когда completion marker противоречит
observations; затем `available`. Недоступность backend-а — RPC error, а не
`pending`.

## Report

Header report-а содержит identities model и run, digests data/model definition
и один `layout` с target identities и identities direct/auxiliary components.
Массивы epoch затем содержат только числовые values в порядке этого layout. Это
исключает повторение семантических labels в каждой epoch.

Available report включает coverage, milestones selection, anchors first/best/
published, health totals и одну ограниченную страницу epochs в порядке
возрастания. `totalLoss`, `selectionScore` и means components — авторитетные
наблюдения Worker, накапливаемые независимо. Они проверяются на наличие и
конечное представление, но не пересчитываются друг через друга между языками.

`selectionScore` описывает взвешенные direct losses при включённом selection;
иначе он равен null. `bestEpoch` берётся из принадлежащего checkpoint-у state
selection, а не из нового вычисления argmin по telemetry. При включённом
selection опубликованные weights принадлежат лучшей epoch; иначе — последней.

Для `PositiveClassWeightedBinaryCrossEntropyWithLogits` target MAE/RMSE
вычисляются по checkpoint-owned public prediction
`sigmoid(rawLogit - log(positiveClassWeight))`. Direct losses,
`selectionScore` и `totalLoss` сохраняют наблюдаемую взвешенную objective и не
сравниваются как абсолютная метрика между разными весами класса.

Health counters каждой epoch удовлетворяют:

```text
trainingBatchesCompleted
  = optimizerUpdatesApplied + optimizerUpdatesSkipped
  = finiteGradientBatches + nonFiniteGradientBatches
```

`ampOverflowBatches` не превышает ни skipped updates, ни batches с non-finite
gradient. Health totals — точные покомпонентные суммы по всем epochs.

## Gradient interactions

Gradient interactions загружаются отдельно для одной epoch. Components идут в
порядке исполнения objective. У опубликованной пары уникальная ориентация:
левый component предшествует правому в порядке исполнения objective. Self и
reversed pairs отсутствуют. Pairs появляются только при наличии хотя бы одного
finite cosine observation; поэтому zero-norm pair без finite cosine observation
отсутствует, а не представляется sentinel-ом. Страницы сортируют pairs по ASCII
`(leftComponentIdentity, rightComponentIdentity)`.

Report сообщает, выключены ли diagnostics, настроены ли без observations или
доступны. Default epoch diagnostics — опубликованная, если она собрана; иначе
ближайшая собранная epoch до неё, затем самая ранняя после неё.

## Cursors, limits и restart

Страницы epoch допускают не более 100 items; страницы gradient — не более 1 000
pairs. Обе используют подписанные cursors, связанные с owner-ом, с TTL 900
секунд. Continuation повторяет точные model, page size и, где требуется,
запрошенную epoch.

Non-terminal pages удерживают immutable валидированный snapshot в общем
ограниченном локальном для process pool: не более 64 entries, 64 MiB всего и
16 MiB на snapshot. Admission атомарен до выдачи cursor; неистёкшие entries не
вытесняются. Terminal response не требует admission. Если capacity не может
удержать snapshot, операция возвращает
`RESOURCE_EXHAUSTED / TELEMETRY_SNAPSHOT_CAPACITY_EXHAUSTED`.

Snapshots намеренно локальны для process. Валидный неистёкший cursor от
предыдущего успешного старта сервиса возвращает
`FAILED_PRECONDITION / TELEMETRY_CURSOR_INVALIDATED`; истёкший cursor
возвращает `TELEMETRY_CURSOR_EXPIRED`. Удаление model имеет приоритет и
возвращает `MODEL_NOT_FOUND` даже во время traversal.

Response budget — 8 MiB. Документы ошибок определяют structured outcomes для
некорректных query, cursor, metadata, backend, integrity, capacity и budget.
Clients ветвятся по `code` и `reason`, а не по тексту ошибки.

## Чистый переход и fixtures

Revision 4 не читает telemetry предыдущих revisions. Migration 0028 удаляет
старые models и telemetry; новые reports создаются только новыми fits Worker
v18 и принадлежащей provider-у projection metrics v10.

`fixtures/` содержит компактные примеры pending/available report и lazy
gradient. Его manifest хеширует только bundle fixtures для офлайн-проверки; он
не входит в identity совместимости telemetry, model или checkpoint.
