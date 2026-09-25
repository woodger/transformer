# Runtime обучения и checkpoint

> Тип: справочник. Принадлежащие provider-у обучение Worker v17, checkpoint
> v10 и поведение recovery за Flight v19.

## Model definition и обучение

Fit Flight v19 получает документ target/objective Semantic v4 и geometry
данных. Transformer валидирует закрытый язык, материализует внутреннюю
configuration модели и записывает identities D1. Упорядоченные непрозрачные
target slots задают output coordinates; ни одна ветвь model/loss не зависит от
имён target-ов внешней предметной области.

Worker применяет loss-input и public-prediction transformations каждого slot к
одной raw output coordinate. Direct и auxiliary operators, private resource
classes и gradient semantics документированы в [losses](./losses.md) и Semantic
v4. Training policy отделена от identity model definition. При включённом
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

Worker восстанавливает компактный input Flight v19 `indexedFeatureBlocks` в
ограниченные slices `[rows, seqLen, featureDim]`. Границы payload/chunk не
являются batches optimizer-а, границами shuffle или границами epoch. Durable
fit может начаться после появления input-а; close отмечает EOF и фиксирует
input manifest для последующих полных epochs.

Checkpoints recovery создаются только на завершённых границах global epoch после
EOF. Checkpoint v10 хранит model, optimizer, AMP scaler, RNG, shuffle, selection,
progress, semantic identities, resolved configuration job и fences input
manifest. Recovery проверяет их до загрузки state. Другая definition data/model
или manifest отклоняются; remapping target-ов и частичная загрузка state не
выполняются.

Recovery используется только для retryable сбоя Worker, пока service продолжает
работать. При любом restart service все незавершённые jobs завершаются до
запуска WorkerPool: `WAITING_INPUT`, `QUEUED`, `RUNNING` и `RETRYING` получают
`FAILED / EXECUTION_INTERRUPTED`, а `CANCELLING` — `CANCELLED`. Checkpoint не
возобновляет такой job после старта нового process service.

Временные artifacts attempt принадлежат service. Startup reconciliation удаляет
только не имеющие ссылок managed artifacts в собственном locked runtime
directory service-а; он не проверяет и не удаляет artifacts другого instance
сервиса Transformer.

## Telemetry

Telemetry epoch — observation её training pass до каждого optimizer update. Она
включает values objective, training MAE/RMSE, health counters и optional
gradient interactions. Её best-effort persistence не меняет исполнение
optimizer-а, selection, успех fit или публикацию модели.

OpenSearch получает принадлежащую provider-у projection v9. Вызывающая система
получает валидированный, нормализованный report через Training Telemetry Query v4, а не
напрямую из OpenSearch. Отсутствие report не делает опубликованную model
некорректной.

При явной `diagnostics.targetHead = "fullCommittedArtifact"` Worker v17 после
каждой завершённой эпохи выполняет отдельный наблюдательный проход по полному
committed input artifact в `eval()` и `torch.no_grad()`. Он сохраняет только
агрегаты raw logit, public prediction, представления до encoder, после каждого
encoder layer и перед финальной target head, градиента direct component и
параметров target head. Этот best-effort artifact не меняет
веса, optimizer, RNG, telemetry epoch или результат `predict`; без настройки
он не создаётся.

## Исторический чистый переход

Model Catalog v5 читает metadata checkpoint v9 только для безопасной проекции:
такая legacy generation возвращает `notConfigured` в Target Head Diagnostics
Query. Worker не возобновляет state v9. Migration 0028 удаляет старые jobs и
generations перед первоначальной активацией Semantic v4; после перехода на
Flight v19 замените OpenSearch indices v8 на v9 и обучите новые generations
для opt-in diagnostics.
