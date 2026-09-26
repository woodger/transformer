# Transformer checkpoint и recovery v10

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет действующие внутренние metadata
> checkpoint/recovery для Semantic v4, Flight v19 и Worker v17.

`transformer-checkpoint-v10` сохраняет разрешённую configuration diagnostics
v2 вместе с остальными fences задания: `jobConfigSha256`, input manifest,
semantic identities, progress и selection. Новая configuration имеет форму:

```json
{
  "schemaVersion": 2,
  "gradientInteractions": null,
  "targetHead": "fullCommittedArtifact"
}
```

`targetHead` — runtime-настройка сбора, а не часть semantic document. Она
входит в `jobConfigSha256` и recovery fence, но не меняет `modelDefinitionSha256`
или допустимость warm start. `targetHead: null` сохраняется явно и означает,
что запрос Target Head Diagnostics v2 для этой generation вернёт
`notConfigured`.

`transformer-recovery-v10` не несёт отдельную копию configuration diagnostics:
её точность обеспечивается уже обязательным `jobConfigSha256`. Worker
проверяет тот же fence до продолжения attempt внутри работающего сервиса.

Полные observations диагностики не встраиваются в checkpoint и не участвуют в
его checksum. Они являются отдельным immutable артефактом Worker v17.

Уже опубликованные checkpoint v9 не переписываются. Model Catalog v5 имеет
единственную узкую provider-owned проекцию их diagnostics v1 в v2:
`gradientInteractions` сохраняется, а отсутствующий `targetHead` становится
явным `null`. Она служит только для результата `notConfigured`, не изменяет
bytes checkpoint-а и не создаёт compatibility reader старого Flight action.
