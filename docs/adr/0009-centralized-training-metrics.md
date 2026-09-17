# ADR 0009: best-effort telemetry обучения через artifact и outbox

- Статус: Принято
- Дата решения: 2026-08-15

> Историческая запись решения; не является актуальным описанием системы. См.
> [указатель ADR](index.md).

## Контекст

Worker вычисляет epoch observations, но attempt-local output не является
долговечной run identity. Прямая отправка из optimizer loop или зависимость
успешного fit от OpenSearch сделали бы аналитическую систему частью critical
training path и смешали observability с model/job lifecycle.

Filesystem и PostgreSQL не образуют distributed transaction, а OpenSearch не
должен становиться источником истины Transformer.

## Решение

Training telemetry является отдельным best-effort observer. После durable
checkpoint или успешной model publication service формирует immutable
run-owned artifacts и регистрирует PostgreSQL outbox отдельной операцией.
Background publisher доставляет deterministic documents в OpenSearch
идемпотентным create flow.

Worker не подключается к OpenSearch и не выполняет network delivery. Ошибка
сбора, artifact publication, outbox admission или доставки наблюдаема, но не
меняет recovery generation, model publication или terminal outcome fit.

Telemetry принадлежит run identity, а не lifecycle модели. Удаление модели не
отменяет delivery и не удаляет run telemetry. OpenSearch остаётся производной
аналитической projection; PostgreSQL и immutable local artifacts обеспечивают
bounded retry lifecycle.

## Рассмотренные альтернативы

- Отправлять metrics непосредственно из worker/hot loop. Отклонено из-за
  network dependency, latency и смешения process responsibilities.
- Парсить stdout без versioned artifact contract. Отклонено из-за слабой
  durability и неявной identity.
- Делать OpenSearch transactionally обязательным для `SUCCEEDED`. Отклонено:
  observability не должна определять business outcome.
- Использовать OpenSearch как источник lifecycle state. Отклонено в пользу
  PostgreSQL authority.
- Владеть telemetry через model foreign keys. Отклонено, потому что model и
  run имеют независимые deletion/retention lifecycles.

## Последствия

- Fit и model publication продолжаются при недоступном OpenSearch.
- Возможна потеря части telemetry без потери ML-state; это осознанная
  best-effort граница.
- Outbox требует bounded capacity, retry и retention policies.
- Contracts telemetry версионируются независимо от Flight и worker lifecycle.

## Текущая документация

- [Политика metrics и OpenSearch](../policy/metrics-policy.md)
- [Runtime обучения](../training-runtime.md)
- [Эксплуатация OpenSearch](../deployment/opensearch.md)
- [Контракты metrics](../../app/contracts/metrics/)
