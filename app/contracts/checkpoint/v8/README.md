# Transformer checkpoint и recovery v8

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутренние для provider-а metadata
> checkpoint и recovery для Semantic v3, Flight v16 и Worker v14.

`transformer-checkpoint-v8` встраивает валидированное намерение модели semantic
v3, разрешённую Transformer configuration модели, identities D1,
configuration training и diagnostics, state selection, разрешённую
initialization, fence configuration job, fence input manifest и progress.
Параметры tensor-а, state optimizer/scaler, state RNG, path checkpoint-а и hash
физического artifact остаются данными реализации.

Запрошенная и разрешённая initialization — разные документы:

- Flight принимает `{ "source": "random" }` или
  `{ "source": "publishedModel", "modelRef": ... }` как запрошенное
  намерение.
- Metadata checkpoint-а сохраняют разрешённый source; source published model
  также сохраняет parent model/checkpoint и точные identities совместимости
  parent/current.
- Model Catalog раскрывает только краткое summary catalog: random либо видимую
  reference parent model.

`transformer-recovery-v8` связывает recovery artifact с точными job, revision
input-а, semantic identities, разрешённой configuration job, input manifest и
progress. До загрузки tensors Transformer проверяет managed artifact и все
metadata fences. Несовместимость не исправляется переназначением targets или
частичной загрузкой state.

Checkpoint/recovery v8 не имеет reader v6/v7. Migration 0027 Flight v15
удаляет старые jobs и generations до активации v8; новые модели обучаются через
новую публичную границу.
