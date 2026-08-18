# ADR 0009: Централизованные training metrics через immutable artifact и outbox

## Статус

Принято, 2026-08-15. Уточнено 2026-08-16: projection использует обычные
versioned indices вместо data streams, а текущее развёртывание подключается к
OpenSearch по trusted-LAN HTTP-профилю.
Контракт v1 расширен без изменения durability boundary в
[ADR 0012](0012-gradient-and-target-telemetry.md).
Уточнено 2026-08-17: telemetry является строго best effort и не входит в
transaction прикладного checkpoint или model generation.

## Контекст

Worker уже вычисляет epoch-level training metrics, но attempt-local
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
не меняется. Внутренний worker process contract был повышен до v4. Core
checkpoint event остаётся строгим, а метрика той же global epoch передаётся
как необязательная telemetry. Terminal fit summary и timing boundaries развиваются отдельным
[ADR 0011](0011-terminal-fit-run-summary.md), который повышает внутренний
контракт до v5 без изменения Flight v4.

Durable flow:

```text
worker завершил global epoch
  → fsync recovery checkpoint
  → checkpoint event
  → PostgreSQL transaction: recovery generation + compact job progress
  → best-effort PostgreSQL transaction: metric interval
  → terminal worker checkpoint
  → models/{modelRef}/checkpoint.pth
     models/{modelRef}/metadata.json
  → PostgreSQL transaction: model generation + SUCCEEDED
  → best-effort telemetry/{jobId}/metrics.jsonl + metadata + outbox
  → background publisher
  → OpenSearch Bulk create
```

Filesystem и PostgreSQL не образуют распределённую транзакцию. Файлы сначала
публикуются через fsync и atomic rename, затем становятся видимыми в БД.
Падение до database commit оставляет только orphan model directory, который
удаляет startup reconciliation. После прикладного commit модель остаётся
успешно опубликованной даже при потере telemetry. Зарегистрированный outbox
можно повторить из immutable artifact.

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

Publisher не запускается, если OpenSearch полностью не настроен. Ошибочная
конфигурация отключает только publisher, регистрируется в operational
logs/counters и не блокирует startup. При штатной конфигурации недоступность
OpenSearch не блокирует startup, fit или model publication.

Bulk request ограничен 500 документами и 2 MiB. Доставка использует `create` и
детерминированный `_id`. HTTP 409 проверяется через `_mget` и
`documentSha256`. Повторяются network errors и HTTP 408/429/5xx с exponential
backoff и jitter от одной секунды до пяти минут. Остальные ошибки переводят
outbox entry в `BLOCKED` и требуют вмешательства оператора.

Одна запись повторяется не более 288 раз и не дольше 24 часов, после чего
переходит в `CANCELLED`. `BLOCKED`, `CANCELLED` и `DELIVERED` записи хранятся
семь дней. Admission новых записей ограничен 10 000 entries и 10 GiB; при
исчерпании бюджета telemetry отбрасывается после публикации модели. Health
metrics показывают число, bytes и возраст backlog, а превышение диагностических
порогов создаёт operational event.

## Индексы OpenSearch

Templates и обычные versioned indices `metrics-points-v1` и
`metrics-artifacts-v1` создаёт оператор до включения publisher-а. Companion
projection `metrics-runs-v1` определена ADR 0011. Data streams
не используются: они направляют запись в текущий write index, поэтому rollover
позволяет повторно создать тот же `_id` в новом backing index, а exact `_mget`
по имени data stream не обеспечивает проверку существующего документа.

Один concrete index на versioned projection гарантирует конфликт повторного
`create` и пакетную проверку `documentSha256` через `_mget`. Будущее временное
разбиение допустимо только как отдельное изменение delivery contract с
детерминированным выбором partition по immutable `recordedAt`.

Оба template задают `number_of_replicas: 0`, поскольку текущий OpenSearch
работает как одиночный узел.

## Подключение

Текущее развёртывание находится в полностью доверенной локальной сети и
использует HTTP без REST TLS, но с существующей OpenSearch Basic Auth.
Runtime-конфигурация содержит endpoint, полную пару username/password и
стабильный `deploymentId`; CA для HTTP не задаётся. Анонимный HTTP остаётся
допустимым профилем для доверенных deployments. Templates и indices по-прежнему
создаёт оператор до запуска publisher-а; publisher выполняет только Bulk create
и exact `_mget` для versioned metrics indices.

Transformer сохраняет прежний HTTPS-профиль для других сред, но он не является
частью текущего deployment. Arrow data, stack traces и filesystem paths не
попадают в документы и логи независимо от transport profile.

## Ограничения v1

- публикуются только успешно завершённые fit runs;
- `experimentRunId` не добавляется без согласованного межсистемного identity;
- checkpoints и исходный NDJSON не отправляются в OpenSearch;
- независимые test metrics и baselines остаются ответственностью Inventory;
- централизованные logs, traces и host metrics не входят в решение.

## Последствия

Metrics artifact является необязательной run-owned telemetry. Повреждённая,
неполная или отсутствующая epoch metric логируется и отбрасывается, но не
меняет recovery, fit outcome или model generation. Migration `0007` добавляет
best-effort epoch intervals, model artifact metadata и outbox без изменения
Flight v4 schema.

Структурные владельцы core state и telemetry уточнены в
[ADR 0013](0013-telemetry-ownership-boundaries.md).
