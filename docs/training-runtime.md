# Runtime обучения и checkpoint

> Тип: справочник. Принадлежащие provider-у обучение Worker v14, checkpoint
> v8 и поведение recovery за Flight v15.

## Model definition и обучение

Fit Flight v15 получает документ target/objective Semantic v3 и geometry
данных. Transformer валидирует закрытый язык, материализует внутреннюю
configuration модели и записывает identities D1. Упорядоченные непрозрачные
target slots задают output coordinates; ни одна ветвь model/loss не зависит от
имён target-ов Consumer.

Worker применяет loss-input и public-prediction transformations каждого slot к
одной raw output coordinate. Direct и auxiliary operators, private resource
classes и gradient semantics документированы в [losses](./losses.md) и Semantic
v3. Training policy отделена от identity model definition. При включённом
selection используется weighted direct-loss score; auxiliary components не
входят в этот score.

Feature values могут содержать NaN согласно `missingValuePolicy`:

- `strict`: timestep mask-ируется, когда отсутствует любой feature;
- `relaxed`: он mask-ируется только когда отсутствуют все features, а
  indicators missingness передаются внутренней модели.

Masking выполняется до преобразования NaN в zero. Model использует последний
valid timestep; полностью masked sequence использует безопасный zero
placeholder.

## Input, epochs и recovery

Worker восстанавливает компактный input Flight v15 `indexedFeatureBlocks` в
ограниченные slices `[rows, seqLen, featureDim]`. Границы payload/chunk не
являются batches optimizer-а, границами shuffle или границами epoch. Durable
fit может начаться после появления input-а; close отмечает EOF и фиксирует
input manifest для последующих полных epochs.

Checkpoints recovery создаются только на завершённых границах global epoch после
EOF. Checkpoint v8 хранит model, optimizer, AMP scaler, RNG, shuffle, selection,
progress, semantic identities, resolved configuration job и fences input
manifest. Recovery проверяет их до загрузки state. Другая definition data/model
или manifest отклоняются; remapping target-ов и частичная загрузка state не
выполняются.

Временные artifacts attempt принадлежат service. Startup reconciliation удаляет
только не имеющие ссылок managed artifacts в собственном locked runtime
directory service-а; он не проверяет и не удаляет artifacts другого instance
сервиса Transformer.

## Telemetry

Telemetry epoch — observation её training pass до каждого optimizer update. Она
включает values objective, training MAE/RMSE, health counters и optional
gradient interactions. Её best-effort persistence не меняет исполнение
optimizer-а, selection, успех fit или публикацию модели.

OpenSearch получает принадлежащую provider-у projection v7. Consumer получает
валидированный, нормализованный report через Training Telemetry Query v3, а не
напрямую из OpenSearch. Отсутствие report не делает опубликованную model
некорректной.

## Чистый переход

Checkpoint/recovery v8 не имеет reader для прежнего state checkpoint-а.
Migration 0027 удаляет старые jobs и generations перед активацией Flight v15;
после deployment обучите новые generations.
