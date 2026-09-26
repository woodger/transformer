# Запрос каталога моделей в области владельца v7

> ДОКУМЕНТ КОНТРАКТА. Запрос Model Catalog только для чтения для активного
> Flight v22.

Операции сохраняют owner-scoped discovery и точный detail:

```text
transformer.model-catalog.v7.list
transformer.model-catalog.v7.detail
```

List не меняет summary-проекцию. Detail по-прежнему возвращает Semantic v5
`ModelContract`, checkpoint-owned `predictionDefinition`, D1, training
configuration, selection, progress и initialization summary.

`summary.modelTuning` и `detail.modelContract.modelTuning` обязаны содержать
одинаковый `encoderNormalizationOrder`. Значение принадлежит immutable
generation и объясняет порядок blocks в Model Topology v3; это не live
настройка, не current alias и не команда Worker.

Поле `diagnostics` detail соответствует checkpoint/recovery v12 configuration
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
