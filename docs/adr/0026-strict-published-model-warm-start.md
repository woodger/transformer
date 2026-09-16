# ADR 0026: строгий warm start опубликованной модели

- Статус: Принято
- Дата решения: 2026-09-02
- Заменяет: [ADR 0025](0025-unified-published-model-initialization.md)

> Историческая запись решения; не является актуальным описанием системы. См.
> [указатель ADR](index.md).

## Контекст

ADR 0025 разрешил `publishedModel` загружать weights при различающихся
data-contract digests, если публичная структура tensor-ов оставалась
совместимой. Это неявно объединило дообучение той же модели и
cross-instrument transfer learning под одним initialization kind.

Экспериментальная гипотеза о пользе такого transfer не подтвердилась. При этом
ослабленная проверка лишила `dataContractSha256` роли точного fence
Consumer-owned семантики target/context slots и сделала смысл
`publishedModel` неоднозначным.

## Решение

Сохранить во Flight v9 два initialization kinds: `random` и `publishedModel`.
Для `publishedModel` требовать точного совпадения parent и current
`dataContractSha256` вместе с существующими проверками model config и всего ML
contract. Несовпадение возвращает `MODEL_SCHEMA_MISMATCH` до создания job.

Равный digest означает ту же Consumer-owned семантическую привязку
target/context slots. Границы временного периода `from`/`to` в digest не входят,
поэтому новый fit может обучать ту же модель на другом периоде.

Worker загружает полный parent `state_dict`, но создаёт с нуля optimizer, AMP
scaler, RNG, progress, checkpoint selection и recovery state. Lineage сохраняет
parent model и checkpoint digests, а также parent/current data-contract
digests; для принятого warm start последние равны.

Flight v9, worker v10, terminal fit telemetry v4, checkpoint/recovery v5 и
PostgreSQL revision `0021` сохраняются. Новая migration не требуется.

## Рассмотренные альтернативы

- Сохранить structural compatibility внутри `publishedModel`. Отклонено:
  подтверждённой пользы cross-instrument transfer нет, а действие остаётся
  семантически неоднозначным.
- Вернуть отдельный `publishedModelTransfer`. Отклонено: неподтверждённый
  экспериментальный сценарий не оправдывает дополнительный public contract и
  compatibility policy.
- Добавить optional флаг ослабления digest. Отклонено: флаг снова скрыл бы
  transfer learning внутри обычного warm start.

## Последствия

- `publishedModel` однозначно означает weights-only дообучение той же модели,
  в том числе на другом временном периоде.
- Cross-instrument initialization с другим data-contract digest отклоняется как
  `MODEL_SCHEMA_MISMATCH`.
- Predict сохраняет точную проверку всего data contract.
- Wire schema, initialization lineage, checkpoint format, telemetry schema и
  persistence не меняются.
- Если cross-instrument transfer снова станет нужен, он потребует отдельного
  подтверждённого решения и явной public semantics.

## Текущая документация

- [Контракт Arrow Flight v14](../../app/contracts/flight/v14/README.md)
- [Процессный контракт Worker v13](../../app/contracts/worker/v13/README.md)
- [Интеграция Consumer-ов](../consumer-flight-integration.md)
- [Runtime обучения](../training-runtime.md)
