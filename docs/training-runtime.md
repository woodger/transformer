# Runtime обучения и checkpoint

> Тип: справочник. Принадлежащие provider-у обучение Worker v21, checkpoint
> v13 и поведение recovery за Flight v23.

## Model definition и обучение

Fit Flight v23 получает документ target/objective Semantic v6 и geometry
данных. Transformer валидирует закрытый язык, материализует внутреннюю
configuration модели и записывает identities D1. Упорядоченные непрозрачные
target slots задают output coordinates; ни одна ветвь model/loss не зависит от
имён target-ов внешней предметной области.

Positional encoding поддерживает нечётную `hiddenWidth` и весь объявленный
`seqLen`. Для `seqLen <= 5000` сохраняется прежний размер checkpoint buffer;
для более длинных sequences buffer создаётся по `seqLen`.

Worker применяет loss-input и public-prediction transformations каждого slot к
одной raw output coordinate. Direct и auxiliary operators, private resource
classes и gradient semantics документированы в [losses](./losses.md) и Semantic
v5. Training policy отделена от identity model definition. При включённом
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

Worker восстанавливает компактный input Flight v23 `indexedFeatureBlocks` в
ограниченные slices `[rows, seqLen, featureDim]`. Границы payload/chunk не
являются batches optimizer-а, границами shuffle или границами epoch. Durable
fit может начаться после появления input-а; close отмечает EOF и фиксирует
input manifest для последующих полных epochs.

Checkpoints recovery создаются только на завершённых границах global epoch после
EOF. Checkpoint v13 хранит model, optimizer, AMP scaler, RNG, shuffle, selection,
progress, semantic identities, resolved configuration job и fences input
manifest. Recovery проверяет их до загрузки state. Другая definition data/model
или manifest отклоняются; remapping target-ов и частичная загрузка state не
выполняются.

Состояния CPU RNG и shuffle восстанавливаются на CPU и при исполнении на CUDA;
состояния CUDA RNG также передаются генераторам как CPU ByteTensor.

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

OpenSearch получает принадлежащую provider-у projection v11. Вызывающая система
получает валидированный, нормализованный report через Training Telemetry Query v4, а не
напрямую из OpenSearch. Отсутствие report не делает опубликованную model
некорректной.

При явной `diagnostics.targetHead = "fullCommittedArtifact"` Worker v21 после
каждой завершённой эпохи выполняет отдельный наблюдательный проход по полному
committed input artifact в `eval()` и `torch.no_grad()`. Он сохраняет только
агрегаты raw logit, public prediction, представления до encoder, после каждого
encoder layer и перед финальной target head, градиента direct component и
параметров target head. В v5 artifact также сохраняет границы `input`,
`attentionResidual`, `norm1`, `feedForwardResidual`, `norm2` каждого encoder
layer и `linear`, `gelu`, `layerNorm` блока `OutputHead.shared`; их execution
order определяется immutable `encoderNormalizationOrder`. Этот
best-effort artifact не меняет
веса, optimizer, RNG, telemetry epoch или результат `predict`; без настройки
он не создаётся.

При дополнительной `diagnostics.encoderLayerDiagnostics =
"directComponentPerBatch"` artifact также содержит direct-component gradients
и нормы обновлений групп `attention`, `feedForward` и `normalization` каждого
encoder layer. Точный состав групп, границ прямого прохождения и правила
обработки отсутствующих или non-finite observations определяет Target Head
Diagnostics v5.

## Переход на Semantic v6

Migration 0030 удаляет jobs и generations прежней Semantic revision перед
активацией Semantic v6. Worker не читает
прежние checkpoints или recovery state. Flight v23 использует OpenSearch
indices Metrics v12; после migration обучите новые generations для
opt-in diagnostics.

## Confidence penalty

Optional auxiliary `BernoulliConfidencePenalty` использует public probability
существующего target-а. Он даёт gradient к той же head без нового resource
или mutable state. Declaration и weight принадлежат objective Semantic v6,
не runtime defaults или настройке optimizer-а. Recovery и warm start
проверяют точную definition; смена weight означает другой objective.

Penalty наблюдается в auxiliary losses и gradient interactions. Selection
и direct-component diagnostics сохраняют текущую семантику.
