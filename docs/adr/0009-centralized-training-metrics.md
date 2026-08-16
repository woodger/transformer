# ADR 0009: Централизованные training metrics через immutable artifact и outbox

## Статус

Принято, 2026-08-15. Уточнено 2026-08-16: projection использует обычные
versioned indices вместо data streams.

## Контекст

Worker уже вычисляет полные epoch-level training metrics, но attempt-local
`metrics.jsonl` находится в runtime spool и удаляется вместе с terminal job.
Чтение stdout, отправка из optimizer loop или непосредственный сетевой доступ
worker-а не дают надёжной связи с опубликованной model generation и делают
OpenSearch частью критического пути обучения.

OpenSearch на `hp260g9.home` является общей аналитической проекцией Inventory и
Transformer. Он не заменяет PostgreSQL как источник истины job/model lifecycle
и filesystem как хранилище checkpoint и исходного metrics artifact.

## Решение

Принимается отдельный интеграционный контракт
`transformer.training-metrics.v1` → `inventory.metrics.v1`. Публичный Flight v4
не меняется. Внутренний worker process contract повышается до v4, поскольку
событие `checkpoint` теперь обязательно содержит полную метрику той же global
epoch.

Durable flow:

```text
worker завершил global epoch
  → fsync recovery checkpoint
  → checkpoint event + полная epoch metric
  → одна PostgreSQL transaction
       recovery generation + metric interval + job progress
  → terminal worker checkpoint
  → models/{modelRef}/checkpoint.pth
     models/{modelRef}/metadata.json
     models/{modelRef}/metrics.jsonl
  → одна PostgreSQL transaction
       model generation + artifact metadata + outbox + SUCCEEDED
  → background publisher
  → OpenSearch Bulk create
```

Filesystem и PostgreSQL не образуют распределённую транзакцию. Файлы сначала
публикуются через fsync и atomic rename, затем становятся видимыми в БД.
Падение до database commit оставляет только orphan model directory, который
удаляет startup reconciliation. После database commit outbox можно повторить
из immutable artifact.

## Identity и данные

`runId` равен Transformer `jobId`. Каждая строка artifact содержит `jobId`,
фактические `attemptId` и `attempt` epoch, `modelRef`, immutable ML digests,
checkpoint format, версию приложения и Git commit. Итоговая artifact metadata
содержит attempt, опубликовавшую модель; отдельные строки могут принадлежать
предыдущим attempts после recovery.

В OpenSearch публикуются только закрытые имена metrics. Per-target points
получают `targetIndex`; arbitrary dynamic attributes запрещены JSON Schema и
strict mapping. Wall-clock phase metrics нужны для диагностики, но не входят в
deterministic identity модели.

`recordedAt` фиксируется сервисом при durable commit epoch. `step` означает
число завершённых training batches, а не гарантированное число optimizer
updates при AMP overflow. Эта семантика сохраняет существующий training state
и не изменяет loss scheduling.

## Доставка

Publisher не запускается, если OpenSearch полностью не настроен. Частичная или
небезопасная конфигурация блокирует startup как ошибка deployment. При штатной
конфигурации недоступность OpenSearch не блокирует startup, fit или model
publication.

Bulk request ограничен 500 документами и 2 MiB. Доставка использует `create` и
детерминированный `_id`. HTTP 409 проверяется через `_mget` и
`documentSha256`. Повторяются network errors и HTTP 408/429/5xx с exponential
backoff и jitter от одной секунды до пяти минут. Остальные ошибки переводят
outbox entry в `BLOCKED` и требуют вмешательства оператора.

Pending/blocked outbox entries не удаляются. Доставленные записи хранятся семь
дней. Health metrics показывают число, bytes и возраст backlog; пороги
10 000 entries, 10 GiB или 30 дней создают явное событие, но не меняют
результат fit.

## Индексы OpenSearch

Templates и обычные versioned indices `metrics-points-v1` и
`metrics-artifacts-v1` создаёт оператор до включения publisher-а. Data streams
не используются: они направляют запись в текущий write index, поэтому rollover
позволяет повторно создать тот же `_id` в новом backing index, а exact `_mget`
по имени data stream не обеспечивает проверку существующего документа.

Один concrete index на versioned projection гарантирует конфликт повторного
`create` и пакетную проверку `documentSha256` через `_mget`. Будущее временное
разбиение допустимо только как отдельное изменение delivery contract с
детерминированным выбором partition по immutable `recordedAt`.

## Безопасность

Transformer использует отдельного non-admin writer-а, доверенный CA и только
HTTPS. Writer может выполнять Bulk create и exact `_mget` только для двух
metrics indices. Он не управляет indices, templates, mappings, lifecycle
policies или cluster settings. Password, Arrow data, stack traces и filesystem
paths не попадают в документы и логи.

## Ограничения v1

- публикуются только успешно завершённые fit runs;
- `experimentRunId` не добавляется без согласованного межсистемного identity;
- checkpoints и исходный NDJSON не отправляются в OpenSearch;
- независимые test metrics и baselines остаются ответственностью Inventory;
- централизованные logs, traces и host metrics не входят в решение.

## Последствия

Metrics artifact становится обязательной частью новой model publication.
Повреждённая или неполная epoch metric завершает fit ошибкой до model
generation. Уже опубликованная модель остаётся доступной, если OpenSearch или
metrics projection позже недоступны. Migration `0007` добавляет durable epoch
intervals, model artifact metadata и outbox без изменения Flight v4 schema.
