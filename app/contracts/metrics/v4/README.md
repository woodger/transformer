# Контракт централизованных training metrics v4

Этот каталог содержит нормативный контракт между Transformer и общей
платформой OpenSearch. Он не является частью Flight v9: сбой доставки метрик
не меняет lifecycle job и не влияет на публикацию уже проверенной модели.

## Durable artifact

Успешный fit run при доступной telemetry содержит неизменяемый файл:

```text
telemetry/{jobId}/metrics.jsonl
```

Одна строка соответствует одной завершённой global epoch. Строка проверяется
по `training-record.schema.json` и содержит фактический `targets` subset,
total/direct/auxiliary losses, per-target MAE/RMSE, checkpoint-selection
context, missing-data ratios и wall-clock phase durations. Target-bound
значения представлены структурно через точную пару
`target.index`/`target.name`; index относится к выбранному target vector.
Старые fixed-width и camelCase формы не являются aliases. Строка также
различает training batches, реальные
optimizer updates, skipped updates, AMP overflow и finite/non-finite gradient
batches. Для конечных pre-clip gradient norms сохраняются mean, max и
nearest-rank P95; если конечных norms нет, все три значения равны `null`.
`selection_score` отсутствует как OpenSearch point, пока его значение в
artifact равно `null`; остальные обязательные показатели должны быть
конечными. Optional `gradientInteractions` хранит число sampled steps, средние
нормы objective components и средний pairwise cosine. Отсутствие diagnostics
представлено `null`.

Поле `step` — монотонный global training step, то есть число завершённых
training batches. При AMP оно не заявляется как точное число реально
выполненных optimizer updates: `GradScaler` может пропустить update при
overflow. Payload и RecordBatch boundaries на `step` не влияют.

Recovery checkpoint становится видимым независимо от строки epoch. Метрика
сохраняется best effort отдельной транзакцией PostgreSQL; новая attempt может
использовать уже зафиксированные строки прежних attempts. Итоговый run-owned artifact
собирается по generation `1..N` только после model publication. Если полный
набор недоступен или повреждён, telemetry отбрасывается, а модель остаётся
успешно опубликованной.

Новые epoch fields проецируются в OpenSearch так:

| Artifact field | Point metric | Unit |
| --- | --- | --- |
| `trainingBatchesCompleted` | `training.batches.completed` | `count` |
| `optimizerUpdatesApplied` | `training.optimizer.updates.applied` | `count` |
| `optimizerUpdatesSkipped` | `training.optimizer.updates.skipped` | `count` |
| `ampOverflowBatches` | `training.amp.overflow_batches` | `count` |
| `finiteGradientBatches` | `training.gradient.batches.finite` | `count` |
| `nonFiniteGradientBatches` | `training.gradient.batches.non_finite` | `count` |
| `preClipGradientNormMean` | `training.gradient.pre_clip_norm.mean` | `1` |
| `preClipGradientNormMax` | `training.gradient.pre_clip_norm.max` | `1` |
| `preClipGradientNormP95` | `training.gradient.pre_clip_norm.p95` | `1` |
| `directLosses[].value` | `training.loss.direct` + `target` | `1` |
| `auxiliaryLosses[].value` | `training.loss.auxiliary` + `operator` | `1` |
| `gradientInteractions.components[].meanNorm` | `training.gradient.component.norm` | `1` |
| `gradientInteractions.pairs[].meanCosine` | `training.gradient.pair.cosine` | `1` |

## OpenSearch projection

Широкая строка artifact преобразуется в закрытый набор узких документов
`inventory.metrics.point.v4`, проверяемых по `point.schema.json`. Metadata
локального artifact нужна только outbox для проверки целостности и отдельно в
OpenSearch не публикуется. Bytes и filesystem paths наружу не передаются.
Direct loss, MAE и RMSE используют стабильные общие имена metric и отдельный
объект `target`; semantic identity не кодируется в имени поля или metric.

`eventId` вычисляется как SHA-256 RFC 8785/JCS-массива:

```text
schema, source, deploymentId, runId, attemptId,
frame-or-null, epoch, step, metric.name, targetIndex-or-null,
operator-or-null, gradientComponent-or-null, gradientPair-or-null
```

`eventId` используется как OpenSearch `_id`. `documentSha256` вычисляется над
каноническим документом без самого `documentSha256`. Повторный `create` с тем
же `_id` считается успешным только после чтения существующего документа и
точного совпадения `documentSha256`.

Golden identity находится в `fixtures/event-identity.json`; независимый
Node.js-скрипт `fixtures/event_id_sha256.mjs` фиксирует межъязыковую
канонизацию.

## Индексы OpenSearch

Оператор заранее устанавливает strict templates из `opensearch/` и создаёт
один обычный versioned index для epoch points:

```text
metrics-points-v4
```

Template задаёт `number_of_replicas: 0`: текущая платформа является
одиночным узлом и не должна оставлять индексы в состоянии `yellow` из-за
невозможной replica allocation.

Publisher не создаёт indices, templates или mappings. Data streams и rollover
в v4 не используются: запись в data stream направляется в текущий write index,
поэтому после rollover повтор потерянного Bulk response может создать тот же
semantic event в новом backing index. Обычный index сохраняет глобальную
уникальность `_id` внутри versioned projection и позволяет проверить конфликт
через `_mget`.

Будущее разбиение по времени требует отдельной версии delivery contract:
конкретный index должен детерминированно вычисляться по immutable `recordedAt`,
чтобы исходная доставка и любой retry всегда попадали в один partition.

## Outbox

После terminal transaction model generation отдельная best-effort transaction
фиксирует metadata run-owned metrics artifact и outbox pointer. Publisher отправляет
только документы, детерминированно восстановленные из неизменяемого artifact.
Network errors, HTTP 408/429/5xx и потерянный response повторяются в пределах
конечного retry budget; schema, mapping и integrity errors переводят запись в
terminal `BLOCKED` до retention cleanup.

В v4 централизуются только метрики успешно опубликованных fit runs. Метрики
окончательно `FAILED` или `CANCELLED` attempts не публикуются и удаляются
вместе со штатным job retention.

Итоговые lifecycle durations и counters успешного fit принадлежат контракту
[`fit_run/v4`](../fit_run/v4/README.md) и индексу `metrics-runs-v4`. Epoch
points и terminal run summary доставляются одной outbox projection
`inventory.metrics.v6`.
