# Запрос каталога моделей в области владельца v8

> ДОКУМЕНТ КОНТРАКТА. Запрос Model Catalog только для чтения для активного
> Flight v23.

Операции сохраняют owner-scoped discovery и точный detail:

```text
transformer.model-catalog.v8.list
transformer.model-catalog.v8.detail
```

List не меняет summary-проекцию. Detail по-прежнему возвращает Semantic v6
`ModelContract`, checkpoint-owned `predictionDefinition`, D1, training
configuration, selection, progress и initialization summary.

`summary.modelTuning` и `detail.modelContract.modelTuning` обязаны содержать
одинаковый `encoderNormalizationOrder`. Значение принадлежит immutable
generation и объясняет порядок blocks в Model Topology v4; это не live
настройка, не current alias и не команда Worker.

Поле `diagnostics` detail соответствует checkpoint/recovery v13 configuration
v3 и заранее сообщает, запрашивались ли learning observations encoder:

```json
{
  "schemaVersion": 3,
  "gradientInteractions": null,
  "targetHead": "fullCommittedArtifact",
  "encoderLayerDiagnostics": "directComponentPerBatch"
}
```

`encoderLayerDiagnostics: null` означает отсутствие layer-learning
observations. Значение configuration не доказывает готовность report:
`pending`, `unavailable` и integrity outcomes остаются свойствами Target Head
Diagnostics v5.

Registry остаётся единственным источником существования model. Неизвестный,
чужой и удалённый `modelRef` security-equivalent. Checkpoint details, physical
hashes, paths, Worker version и observations не добавляются в catalog.

`fixtures/` предназначен только для офлайн-проверки и не участвует в runtime
или semantic compatibility.

Detail v8 раскрывает сохранённый Semantic v6 objective, включая optional
`BernoulliConfidencePenalty` и его weight. Это неизменяемое свойство generation.
