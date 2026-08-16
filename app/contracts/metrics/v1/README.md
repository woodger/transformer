# Контракт централизованных training metrics v1

Этот каталог содержит нормативный контракт между Transformer и общей
платформой OpenSearch. Он не является частью Flight v4: сбой доставки метрик
не меняет lifecycle job и не влияет на публикацию уже проверенной модели.

## Durable artifact

Каждая успешно опубликованная fit generation содержит неизменяемый файл:

```text
models/{modelRef}/metrics.jsonl
```

Одна строка соответствует одной завершённой global epoch. Строка проверяется
по `training-record.schema.json` и содержит training losses, шесть пар
per-target MAE/RMSE, checkpoint-selection context, missing-data ratios и
wall-clock phase durations. `selection_score` отсутствует как OpenSearch point,
пока его значение в artifact равно `null`; остальные обязательные показатели
должны быть конечными.

Поле `step` — монотонный global training step, то есть число завершённых
training batches. При AMP оно не заявляется как точное число реально
выполненных optimizer updates: `GradScaler` может пропустить update при
overflow. Payload и RecordBatch boundaries на `step` не влияют.

Recovery checkpoint и строка epoch становятся видимыми в одной транзакции
PostgreSQL. Поэтому новая attempt сохраняет уже зафиксированные строки прежних
attempts. Итоговый artifact собирается по generation `1..N`, проверяется и
публикуется вместе с model generation. Artifact живёт не меньше модели и не
зависит от terminal job spool.

## OpenSearch projection

Широкая строка artifact преобразуется в закрытый набор узких документов
`inventory.metrics.point.v1`, проверяемых по `point.schema.json`. Metadata
artifact публикуется отдельно по `artifact.schema.json`; bytes и абсолютные
filesystem paths в OpenSearch не передаются.

`eventId` вычисляется как SHA-256 RFC 8785/JCS-массива:

```text
schema, source, deploymentId, runId, attemptId,
frame-or-null, epoch, step, metric.name, targetIndex-or-null
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
два обычных versioned index:

```text
metrics-points-v1
metrics-artifacts-v1
```

Оба template задают `number_of_replicas: 0`: текущая платформа является
одиночным узлом и не должна оставлять индексы в состоянии `yellow` из-за
невозможной replica allocation.

Publisher не создаёт indices, templates или mappings. Data streams и rollover
в v1 не используются: запись в data stream направляется в текущий write index,
поэтому после rollover повтор потерянного Bulk response может создать тот же
semantic event в новом backing index. Обычный index сохраняет глобальную
уникальность `_id` внутри versioned projection и позволяет проверить конфликт
через `_mget`.

Будущее разбиение по времени требует отдельной версии delivery contract:
конкретный index должен детерминированно вычисляться по immutable `recordedAt`,
чтобы исходная доставка и любой retry всегда попадали в один partition.

## Outbox

Terminal PostgreSQL transaction атомарно фиксирует model generation, metadata
metrics artifact и outbox pointer. Publisher отправляет только документы,
детерминированно восстановленные из неизменяемого artifact. Network errors,
HTTP 408/429/5xx и потерянный response повторяются; schema, mapping и integrity
errors блокируют запись outbox для оператора.

В v1 централизуются только метрики успешно опубликованных fit runs. Метрики
окончательно `FAILED` или `CANCELLED` attempts остаются attempt-local и не
публикуются.

Итоговые lifecycle durations и counters успешного fit принадлежат отдельному
контракту [`fit_run/v1`](../fit_run/v1/README.md) и индексу
`metrics-runs-v1`. Epoch points и их два существующих индекса не меняются.
