# Запрос каталога моделей в области владельца v5

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет действующую revision read-only
> Model Catalog Query для Flight v19.

Операции сохраняют owner-scoped discovery и точный detail:

```text
transformer.model-catalog.v5.list
transformer.model-catalog.v5.detail
```

List не меняет свою summary-проекцию. Detail по-прежнему возвращает полный
Semantic v4 `ModelContract`, checkpoint-owned `predictionDefinition`, D1,
training configuration, selection, progress и initialization summary.

Единственное изменение v5 — поле `diagnostics` detail теперь соответствует
checkpoint/recovery v10 configuration v2 и всегда содержит `targetHead`:

```json
{
  "schemaVersion": 2,
  "gradientInteractions": null,
  "targetHead": "fullCommittedArtifact"
}
```

Так вызывающая система до lazy запроса Target Head Diagnostics v2 видит, была
ли диагностика запрошена для generation. Значение не доказывает доступность
полного отчёта: `pending`, `unavailable` и integrity outcomes принадлежат
самой диагностической поверхности.

Для сохранённой metadata checkpoint v9 Transformer проецирует прежнюю
diagnostics configuration v1 в данную форму v2, добавляя `targetHead: null`.
Это единственный compatibility mapping: он не изменяет checkpoint, не
пересчитывает observations и не делает старую generation диагностированной.
Поэтому сохранённая v9 модель остаётся видимой в Catalog v5, а Target Head
Diagnostics v2 возвращает ей `notConfigured`.

Registry остаётся единственным источником существования model. Неизвестный,
чужой и удалённый `modelRef` security-equivalent. Checkpoint details, physical
hashes, paths, Worker version и сами observations не добавляются в catalog.

`fixtures/` содержит обновлённый detail с diagnostics v2. Его manifest служит
только офлайн-проверке и не участвует в runtime или semantic compatibility.
