# ADR 0011: Итоговая telemetry успешного fit

## Статус

Принято, 2026-08-16.

## Контекст

Checkpoint-aligned epoch metrics описывают обучение, но не позволяют увидеть
полный критический путь remote fit: ожидание input/EOF, queue, запуск worker-а,
recovery, публикацию checkpoint и модели. Восстанавливать эти значения в
Inventory из `job.status` нельзя, а отправка событий из hot loop или worker-а
сделала бы OpenSearch частью training path.

Часть контрольных данных уже находится в PostgreSQL. Это не случайное
аналитическое хранилище: job, attempts, inputs, recovery checkpoints и epoch
intervals участвуют в fencing, recovery, проверке полноты и terminal
транзакции. Их перенос в OpenSearch нарушил бы прикладную atomicity.

## Решение

Для каждой успешно опубликованной модели создаётся второй неизменяемый
model-owned artifact:

```text
models/{modelRef}/metrics.jsonl
models/{modelRef}/run-summary.json
```

`run-summary.json` имеет формат `transformer.fit-run-summary.v1`. Он содержит
один terminal summary, а не ряд событий:

- ожидание первого input и EOF;
- сумму queue wait и worker startup по attempts;
- сумму времени завершённых global epochs;
- сумму сериализации и durable publication recovery/terminal checkpoints;
- model publication и полный remote fit;
- attempts, recoveries, input payloads, rows и bytes.

Duration-поля могут перекрываться. В частности, streaming epoch 0 может идти
одновременно с ожиданием EOF. Полный критический путь задаёт `remoteFitMs`, а
не сумма частных duration.

Внутрипроцессные checkpoint intervals измеряются monotonic clock. Границы
между service/worker attempts и recovery берутся из durable PostgreSQL
timestamps. `publishedAt` — логическая граница terminal publication,
зафиксированная перед одной PostgreSQL-транзакцией; время фактического commit в
immutable artifact не включается.

Terminal sequence:

```text
worker completed
  → durable checkpoint + metrics.jsonl + run-summary.json + metadata.json
  → одна PostgreSQL transaction
       model generation
       оба artifact metadata
       outbox inventory.metrics.v2
       job SUCCEEDED
  → post-commit OpenSearch publisher
       epoch points + artifact metadata + fit run summary
```

Внутренний worker contract повышается до v5. Публичный Flight v4 не меняется.
Epoch artifact и `inventory.metrics.v1` остаются совместимыми. Новая terminal
проекция имеет schema `inventory.metrics.fit-run.v1`, deterministic
`summaryId`, `documentSha256` и отдельный обычный index `metrics-runs-v1` без
rollover.

## PostgreSQL и OpenSearch

PostgreSQL остаётся источником истины для lifecycle и recovery. Epoch
intervals удаляются вместе с terminal job по действующей retention policy
только после того, как из них опубликован model-owned artifact. OpenSearch
хранит производную аналитическую проекцию и не используется для принятия
lifecycle-решений.

Недоступность OpenSearch не влияет на `SUCCEEDED`; outbox повторяет доставку.
Повреждённый или неполный локальный summary, напротив, блокирует model
publication как нарушение внутреннего контракта.

## Последствия

Migration `0009` добавляет timing boundaries attempts, checkpoint durations и
metadata terminal summary. Перед migration следует завершить или отменить
уже активные fit attempts: прежний worker contract не создаёт обязательные
timing values.

Публикуются только успешные fit runs. Failed/cancelled attempt telemetry,
row/batch events, host metrics и изменение Flight остаются вне решения.
