# ADR 0025: unified published-model initialization

- Status: Superseded
- Superseded by: [ADR 0026](0026-strict-published-model-warm-start.md)
- Decision date: 2026-09-02
- Supersedes: [ADR 0024](0024-cross-instrument-transfer-initialization.md)

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Strict warm start и перенос weights на структурно совместимый data contract
исполняются одинаково: Transformer разрешает immutable parent model, загружает
весь `state_dict` и создаёт новое training state. Разные public initialization
kinds описывали не разные способы исполнения, а только разную строгость
проверки одного digest, усложняя capabilities и интеграцию Consumer-а.

При этом полное совпадение data contract необходимо для predict, но для нового
fit Consumer может осознанно изменить предметное наполнение тех же
структурно совместимых slots. Transformer не владеет этой семантикой и не
выполняет remapping.

## Решение

Оставить во Flight v9 два initialization kinds: `random` и `publishedModel`.
`publishedModel` принимает точный owner-scoped immutable `modelRef`, требует
совпадения model config, всего ML contract и структурных полей data contract:
identity, version, profile, sequence length, feature width и target schema.
Отличаться разрешено только полному data-contract digest.

Worker загружает полный parent `state_dict`, но создаёт с нуля optimizer, AMP
scaler, RNG, progress, checkpoint selection и recovery state. Resolved lineage
сохраняет parent model и checkpoint digests, а также parent и current
data-contract digests. Дочерняя generation получает точный data contract
текущего fit; predict по-прежнему требует его полного совпадения.

Flight v9, worker v10, terminal fit telemetry v4, checkpoint/recovery v5 и
PostgreSQL revision `0021` сохраняются. Отдельная migration не требуется.

## Рассмотренные альтернативы

- Сохранить отдельные strict и structural initialization kinds. Отклонено:
  способ загрузки weights и lifecycle нового fit у них одинаковы, а lineage
  уже явно фиксирует оба data-contract digests.
- Добавить boolean-флаг к `publishedModel`. Отклонено: optional policy-флаг
  усложняет закрытую schema и создаёт ещё одну семантическую комбинацию.
- Сохранить обязательное равенство полного data contract. Отклонено: оно
  исключает осознанное повторное использование weights при неизменной
  структуре tensor-ов.
- Разрешить изменение profile, shape, targets, objective либо частичную
  загрузку weights. Отклонено: это другой класс совместимости, требующий
  явного сопоставления параметров и предметной семантики.

## Последствия

- Consumer выбирает между независимым fit и единым weights-only
  `publishedModel` initialization.
- Равные и различающиеся data-contract digests проходят одну структурную
  проверку; несовместимые model, ML или structural data contracts возвращают
  `MODEL_SCHEMA_MISMATCH`.
- Consumer отвечает за корректность нового предметного наполнения совместимых
  slots.
- Parent остаётся immutable и блокируется от удаления до terminal state
  зависимого fit; опубликованная child generation после этого физически
  независима.
- Третий initialization kind не принимается schemas или runtime.

## Текущая документация

- [Контракт Arrow Flight v10](../../app/contracts/flight/v10/README.md)
- [Worker process contract v11](../../app/contracts/worker/v11/README.md)
- [Terminal fit telemetry v4](../../app/contracts/metrics/fit_run/v4/README.md)
- [Интеграция Consumer-ов](../consumer-flight-integration.md)
- [Training runtime](../training-runtime.md)
- [Управление опубликованными моделями](../operations/published-models.md)
