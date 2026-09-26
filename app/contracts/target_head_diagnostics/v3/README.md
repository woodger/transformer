# Диагностика выходных головок и обучения encoder v3

> ДОКУМЕНТ КОНТРАКТА. Подготовленный пакет для межпроектной проверки. Он не
> активирует runtime и не изменяет действующий Flight v19.

## Назначение

```text
transformer.target-head-diagnostics.v3.report
```

Запрос только для чтения в области владельца возвращает явно включённые
наблюдения обучения конкретной опубликованной версии модели. Наряду с v2
`rawLogit`, `publicPrediction`, gradients target head и `representationFlow`,
v3 показывает обучение каждого encoder layer.

Запрос принимает только точный `modelRef`, `requestId`, `pageSize` и
непрозрачный cursor. Владелец выводится из аутентифицированного subject.
Неизвестная, чужая и удалённая version модели возвращают одинаковый
`MODEL_NOT_FOUND`.

Наружу не выдаются строки, значения признаков и целей, тензоры и имена
параметров, байты и пути checkpoint, сведения об устройстве или топологии
хранения.

`layout` один раз связывает каждую запись target head с точными
`targetIdentity`, `targetIndex` и `directComponentIdentity`. Все `targetHeads`
в observation epoch идут строго в этом порядке. Перед выдачей `available`
проверяется соответствие layout direct components и ordered target slots
checkpoint-owned `ModelContract`, плотный диапазон epoch `1..completedEpochs`,
строгое возрастание `globalStep`, конечность чисел и согласованность row counts
с `sampleRowCount`.

## Явная runtime-настройка

Полная диагностика включается так:

```json
{
  "schemaVersion": 3,
  "gradientInteractions": null,
  "targetHead": "fullCommittedArtifact",
  "encoderLayerDiagnostics": "directComponentPerBatch"
}
```

`encoderLayerDiagnostics` допускает только `null` или
`directComponentPerBatch`; второе значение требует
`targetHead: "fullCommittedArtifact"`. Оно означает сбор direct-component
gradients на каждом training batch и parameter updates после каждого успешного
optimizer step. `null` сохраняет v2 scope: report содержит target-head и
representation-flow observations, но не `encoderLearning`.

Configuration входит в `jobConfigSha256` и recovery fence. Она не входит в
Semantic v4, `modelDefinitionSha256`, D1, warm-start compatibility, objective
или public predict definition.

## Наблюдения encoder

Для каждой epoch `encoderLayerLearning.layers` идёт в execution order и точно
совпадает с `representationFlow.encoderLayers`; `layerIndex` начинается с 0.
`directComponentGradients` идёт в порядке прямых компонентов layout. Каждая
запись имеет три закрытые группы, определённые provider-ом:

- `attention` — self-attention parameters;
- `feedForward` — feed-forward path parameters;
- `normalization` — параметры обеих нормализаций encoder layer.

Для component `c` и группы параметров `P` batch observation определён так:

```text
gradientL2(c, P) = sqrt(Σₚ∈P ||∂c / ∂p||²)
```

`c` — `GlobalRowMean` именованного direct component после его declared
`weight`, но до aggregation с остальными direct и auxiliary components.
Observation не включает total loss, AMP scaling, global gradient clipping или
`optimizer.step()`. Epoch aggregates содержат `l2Mean`, `l2Maximum` и
`finiteBatchCount`; при count `0` оба L2 значения равны `null`. При ненулевом
count все L2 значения конечны и `l2Mean <= l2Maximum`.

`parameterUpdates` не связываются с отдельным component. Для группы `P` и
успешного optimizer update:

```text
parameterUpdateL2(P) = sqrt(Σₚ∈P ||p_after - p_before||²)
```

Это наблюдение описывает полный optimizer path, включая total objective,
auxiliary components, global clipping, Adam и weight decay. Пропущенный AMP
update не входит в aggregate. При `appliedBatchCount = 0` L2 значения равны
`null`; при ненулевом count значения конечны и `l2Mean <= l2Maximum`. Global
skipped/non-finite counters остаются в Training Telemetry.

`representationFlow` сохраняет v2 semantics. Его абсолютная L2-величина не
самостоятельное доказательство потери информации между boundary с разной
normalization; она интерпретируется вместе с encoder gradients, updates и raw
logits.

`rawLogit` остаётся значением до публичного преобразования.
`publicPrediction` вычисляется строго через checkpoint-owned определение
предсказания; для
`PositiveClassWeightedBinaryCrossEntropyWithLogits` это
`sigmoid(rawLogit - log(positiveClassWeight))`, а не weighted score.

## Сбор и availability

Worker собирает gradient observations до `backward`/clipping и snapshots
parameter updates вокруг `optimizer.step()`. Post-update `representationFlow`,
logits и public predictions по-прежнему измеряются на полном committed artifact
в `eval()`/`no_grad()`.

Сбор остаётся best-effort: он не меняет weights, optimizer/scaler state, RNG,
batch order, selection, fit terminal outcome или prediction values. `available`
выдаётся только для плотной полной последовательности epochs. Частичный или
противоречивый artifact даёт
`TARGET_HEAD_DIAGNOSTICS_INTEGRITY_FAILED`.

Градиент direct component получается наблюдательным вызовом autograd без
записи в `.grad`; после измерения training path продолжает обычный `backward`.

Модель без `targetHead` configuration возвращает
`notConfigured / TARGET_HEAD_DIAGNOSTICS_NOT_CONFIGURED`. Модель с
`encoderLayerDiagnostics: null` может вернуть обычный target-head v3 report
без `encoderLayerLearning`. V2 artifact не достраивается из checkpoint и в v3
возвращает `unavailable / FORMAT_UNSUPPORTED`.

Порядок outcomes: поиск модели в области владельца и проверка metadata, затем
`notConfigured`, затем `pending` только пока materialization доказуемо
продолжается, `unavailable`, structured integrity error для противоречивого
artifact и `available` только для полной проверенной проекции. Pagination
сохраняет epoch ASC, owner/model/run/page-size binding и TTL 900 seconds.
Не terminal страница удерживает проверенную immutable projection в bounded
snapshot cache; удаление модели имеет приоритет над cursor. Точные structured
errors определены в `schemas/error-detail.schema.json`; новых categories v3 не
вводит.

## Версии и границы

Подготовленный v3 требует Target Head Diagnostics v3, Worker v18,
checkpoint/recovery v11, Model Catalog v6 и Flight v20. Semantic v4, Metrics
v9, Training Telemetry v4, Model Topology v2, PostgreSQL и Arrow data plane
не меняются.

`fixtures/manifest.json` предназначен только для офлайн conformance review.
Он не является runtime fence и не участвует в compatibility или prediction.
