# Интеграция с Transformer Flight v22

> Тип: руководство. Практический порядок работы вызывающей системы с действующей
> provider boundary. JSON Schema в `app/contracts/` имеет приоритет над этим
> пояснением.

## Предварительные условия

Вызывающая система вызывает Transformer только со своей аутентифицированной
bearer credential. Браузер никогда не вызывает Transformer или OpenSearch.
Scope owner-а выводится из аутентифицированного subject; вызывающая система не
передаёт owner, path checkpoint-а или identity storage provider-а.

Перед интеграцией используйте точные актуальные packages:

- [Семантическая модель v5](../app/contracts/semantic/v5/README.md)
- [Flight v22](../app/contracts/flight/v22/README.md)
- [Запрос каталога моделей v7](../app/contracts/model_catalog/v7/README.md)
- [Запрос topology модели v3](../app/contracts/model_topology/v3/README.md)
- [Запрос телеметрии обучения v4](../app/contracts/training_telemetry/v4/README.md)
- [Диагностика выходных головок v5](../app/contracts/target_head_diagnostics/v5/README.md)

Путь compatibility предыдущих Flight revisions отсутствует. Existing generations, созданные до
migration 0029, удалены и не могут использоваться для predict или warm start.

## Fit

1. Materialize self-contained Semantic v5 `ModelContract` из локальной
   семантики target/catalog/profile. Не передавайте paths profile, keys lookup
   target-ов или исполняемый loss code.
2. Постройте `dataBinding` из непрозрачного `dataContractSha256`, geometry
   tensor-а и `inputLayout.featureBlocks`.
3. Вызовите `transformer.v22.fit.create` с UUID `requestId`, стабильным
   `idempotencyKey`, label, запрошенным устройством, model contract,
   configuration training и diagnostics и запрошенной initialization.
4. Сохраните возвращённые `jobId`, `mutationLease` и resolved definition.
5. Загрузите упорядоченные compact Arrow payloads, используя возвращённый path
   descriptor-а и публичную schema metadata DoPut.
6. Вычислите digest manifest receipts, вызовите `job.input.close`, затем
   опрашивайте `job.status` до terminal state.

Server рассчитывает фактические physical receipts и totals. Не передавайте
повторно IDs schema provider-а, digest data, identity client execution, values
fencing или close totals в metadata upload. `expectedLogicalRows` необязателен
и используется только для раннего обнаружения EOF.

Fit может начаться после durable availability input-а согласно scheduling
service; input close отмечает EOF и завершает input manifest. Вызывающая
система всё равно должна дождаться terminal result job до признания модели
published.

## Predict

Вызовите `transformer.v22.predict.create` с точным `modelRef`, запрошенным
устройством и текущим `dataBinding`. Не передавайте повторно TargetContract,
Objective или model tuning. Create result передаёт `predictionDefinition` до
upload: `seqLen`, output width и упорядоченные непрозрачные targets с public
prediction transformations. Используйте их для построения и decoding Arrow
data plane.

Несовпадение data/model definition отклоняется до input upload. Coordinates
prediction всегда следуют принадлежащему checkpoint-у порядку target-ов;
неявное remapping не выполняется.

## Mutation, status и errors

`job.acquire` заменяет устаревший непрозрачный mutation lease. `job.cancel` и
`job.input.close` требуют текущего lease. `job.status` намеренно является
mutable projection; вызывающая система хранит immutable create response вместо
ожидания API job-detail provider-а.

Ветвитесь по structured error `code` и `reason`, но никогда по тексту message.
`MODEL_NOT_FOUND` security-equivalent для неизвестных, чужих и удалённых
models. Stored corruption, несовместимые requests, unavailable dependencies и
validation errors имеют разные structured outcomes.

## Model catalog, topology и telemetry

Используйте `transformer.model-catalog.v7.list` для owner-scoped discovery и
`.detail` для точного выбранного `modelRef`. List — bounded high-water/keyset
traversal; параллельное deletion может заставить detail вернуть
`MODEL_NOT_FOUND`, тогда вызывающая система обновляет своё представление.

Используйте `transformer.training-telemetry.v4.report` для observations
завершённого training pass и запрашивайте gradient interactions только после
раскрытия этой секции UI. Отсутствие telemetry не скрывает и не делает
опубликованную model некорректной.

Используйте `transformer.model-topology.v3.detail` только для точного
опубликованного `modelRef` и загружайте response лениво при открытии схемы
модели. Topology уже привязана к `modelDefinitionSha256`; не выводите graph из
model tuning или target names. Для объединения с telemetry сначала сравните
точные `modelRef` и `modelDefinitionSha256`.

Используйте `transformer.target-head-diagnostics.v5.report` только для модели,
созданной с явной runtime-настройкой `diagnostics.targetHead`.
Для learning observations encoder также задайте
`diagnostics.encoderLayerDiagnostics: "directComponentPerBatch"`.
Загружайте эту диагностику лениво: она описывает ход обучения выходных головок,
а не качество модели на новых данных.

В каждом observation `representationFlow` содержит
`encoderInput`, упорядоченный `encoderLayers` и `targetHeadInput`. Во всех
трёх случаях `rowCenteredL2Mean` измерен тем же post-update проходом полного
committed artifact; по последовательности значений UI показывает первую
границу, на которой различие между строками сжалось. `encoderBlockFlow`
уточняет это место named boundaries encoder в порядке, объявленном
`normalizationOrder`, и `OutputHead.shared`.

Все четыре query packages определяют собственные schemas и rules outcomes. Их
availability объявляется capabilities Flight v22, но их семантика не встроена в
generic job actions.
