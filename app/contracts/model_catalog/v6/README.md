# Запрос каталога моделей в области владельца v6

> ДОКУМЕНТ КОНТРАКТА. Текущий запрос Model Catalog только для чтения для
> Flight v21.

Операции сохраняют owner-scoped discovery и точный detail:

```text
transformer.model-catalog.v6.list
transformer.model-catalog.v6.detail
```

List не меняет summary-проекцию. Detail по-прежнему возвращает Semantic v4
`ModelContract`, checkpoint-owned `predictionDefinition`, D1, training
configuration, selection, progress и initialization summary.

Поле `diagnostics` detail соответствует checkpoint/recovery v11 configuration
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
Diagnostics v4.

Registry остаётся единственным источником существования model. Неизвестный,
чужой и удалённый `modelRef` security-equivalent. Checkpoint details, physical
hashes, paths, Worker version и observations не добавляются в catalog.

`fixtures/` предназначен только для офлайн-проверки и не участвует в runtime
или semantic compatibility.
