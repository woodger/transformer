# Интеграция с Transformer Flight v23

> Тип: руководство. Практический порядок работы вызывающей системы с действующей
> provider boundary. JSON Schema в `app/contracts/` имеет приоритет над этим
> пояснением.

## Предварительные условия

Вызывающая система вызывает Transformer только со своей аутентифицированной
bearer credential. Браузер никогда не вызывает Transformer или OpenSearch.
Scope owner-а выводится из аутентифицированного subject; вызывающая система не
передаёт owner, path checkpoint-а или identity storage provider-а.

Перед интеграцией используйте точные актуальные packages:

- [Семантическая модель v6](../app/contracts/semantic/v6/README.md)
- [Flight v23](../app/contracts/flight/v23/README.md)
- [Запрос каталога моделей v8](../app/contracts/model_catalog/v8/README.md)
- [Запрос topology модели v4](../app/contracts/model_topology/v4/README.md)
- [Запрос телеметрии обучения v4](../app/contracts/training_telemetry/v4/README.md)
- [Диагностика выходных головок v5](../app/contracts/target_head_diagnostics/v5/README.md)

Путь compatibility предыдущих Flight revisions отсутствует. Migration 0030
удаляет existing generations предыдущей Semantic revision; после перехода
их нельзя использовать для predict или warm start.

## Fit

1. Materialize self-contained Semantic v6 `ModelContract` из локальной
   семантики target/catalog/profile. Не передавайте paths profile, keys lookup
   target-ов или исполняемый loss code.
2. Постройте `dataBinding` из непрозрачного `dataContractSha256`, geometry
   tensor-а и `inputLayout.featureBlocks`.
3. Вызовите `transformer.v23.fit.create` с UUID `requestId`, стабильным
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

## Confidence penalty в objective

Для probability target-а можно явно добавить auxiliary component:

```json
{
  "identity": "auxiliary.confidence",
  "operator": "BernoulliConfidencePenalty",
  "weight": 0.1,
  "roles": {"probability": "Opaque.Probability"}
}
```

`Opaque.Probability` должен разрешаться в существующий target с public
`Sigmoid`. Коэффициент в примере не является default или рекомендацией для
profile. Operator добавляет negative entropy публичной вероятности; для
weighted binary BCE учитывается class-weight correction. При direct weight
`a` и auxiliary weight `b` получается `a * BCE - b * H(p)`.

Без компонента penalty отсутствует. Вызывающая система проверяет Semantic v6
и `auxiliaryOperators` в capabilities, разрешает declaration в своём profile
и материализует тот же objective для всего job. Изменение coefficient меняет
definition и несовместимо с recovery/warm start. Формулы target и features не
меняются. Коэффициент следует проверять на held-out данных: entropy может
завышать вероятность редкого события.

## Predict

Вызовите `transformer.v23.predict.create` с точным `modelRef`, запрошенным
устройством и текущим `dataBinding`. Не передавайте повторно TargetContract,
Objective или model tuning. Create result передаёт `predictionDefinition` до
upload: `seqLen`, output width и упорядоченные непрозрачные targets с public
prediction transformations. Используйте их для построения и decoding Arrow
data plane.

Несовпадение data/model definition отклоняется до input upload. Coordinates
prediction всегда следуют принадлежащему checkpoint-у порядку target-ов;
неявное remapping не выполняется.

Predict остаётся в `WAITING_INPUT`, пока вызывающая система не загрузит все
payloads и не зафиксирует manifest через `job.input.close`. В отличие от fit,
Worker для predict не запускается для открытого input.

После `job.input.close` опрашивайте `job.status` до terminal state. Для
`SUCCEEDED` получите predictions следующим образом:

1. Вызовите `transformer.v23.job.outputs.list` с `jobId` и UUID `requestId`.
   Обойдите все страницы через `nextCursor`, пока `hasMore` не станет `false`.
2. Для каждого output в порядке `ordinal` передайте возвращённый
   `descriptorPath` как path descriptor в `GetFlightInfo`.
3. Возьмите непрозрачный ticket из endpoint результата `GetFlightInfo` и
   передайте его в `DoGet`. Каждый RPC предъявляет Bearer credential того же
   owner-а.
4. Прочитайте Arrow stream и декодируйте prediction coordinates согласно
   сохранённой `predictionDefinition`.

Tickets ограничены по сроку действия и привязаны к owner-у. Если ticket истёк,
получите новый через `GetFlightInfo`, пока output ещё доступен по retention
job. Для `FAILED` или `CANCELLED` получение output не выполняется.

## Mutation, status и errors

`job.acquire` заменяет устаревший непрозрачный mutation lease. `job.cancel` и
`job.input.close` требуют текущего lease. `job.status` намеренно является
mutable projection; вызывающая система хранит immutable create response вместо
ожидания API job-detail provider-а.

Для ошибок с structured detail в `FlightError.extra_info` ветвитесь по
`code` и `reason`; текст `message` служит только пояснением. Обычные ошибки
jobs без detail передают стабильный application code в префиксе сообщения
`CODE: ...`, без поля `reason`. Используйте этот code, а не поясняющий текст.
Класс PyArrow exception и transport status не всегда совпадают с application
code; ограничения описаны в
[справочнике PyArrow Flight](./flight-dependency-note.md#отсутствующие-server-status-codes).

`MODEL_NOT_FOUND` security-equivalent для неизвестных, чужих и удалённых
models. Stored corruption, несовместимые requests, unavailable dependencies и
validation errors имеют разные structured outcomes.

## Model catalog, topology и telemetry

Используйте `transformer.model-catalog.v8.list` для owner-scoped discovery и
`.detail` для точного выбранного `modelRef`. List — bounded high-water/keyset
traversal; параллельное deletion может заставить detail вернуть
`MODEL_NOT_FOUND`, тогда вызывающая система обновляет своё представление.

Используйте `transformer.training-telemetry.v4.report` для observations
завершённого training pass и запрашивайте gradient interactions только после
раскрытия этой секции UI. Отсутствие telemetry не скрывает и не делает
опубликованную model некорректной.

Используйте `transformer.model-topology.v4.detail` только для точного
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
availability объявляется capabilities Flight v23, но их семантика не встроена в
generic job actions.
