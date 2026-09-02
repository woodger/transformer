# ADR 0024: explicit cross-instrument transfer initialization

- Status: Accepted
- Decision date: 2026-09-02

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Strict weights-only warm start требует точного совпадения data contract. Это
защищает модель от неявной смены семантики входа, но исключает осознанный
transfer между инструментами, когда размеры tensor-ов, порядок feature slots,
архитектура, targets и objective остаются неизменными, а конкретная привязка
инструментов к этим slots меняется.

Ослабление существующего `publishedModel` сделало бы прежнее безопасное
действие неоднозначным. Transformer при этом не владеет FIGI и не может
интерпретировать либо переставлять принадлежащие Consumer-у feature blocks.

## Решение

Ввести отдельный initialization kind `publishedModelTransfer`, сохранив
`publishedModel` strict. Transfer принимает точный owner-scoped immutable
parent `modelRef`, загружает весь `state_dict` и создаёт с нуля optimizer,
scaler, RNG, progress, selection и recovery state.

Для transfer должны совпадать model config, ML contract и структурные поля data
contract: identity, version, profile, sequence length, feature width и target
schema. Отличаться может только полный data-contract digest. Этим выбором
Consumer явно принимает новое предметное наполнение совместимых slots;
Transformer не выполняет semantic remapping.

Resolved lineage фиксирует parent model и checkpoint digests, а также parent и
current data-contract digests. Он входит в job identity и recovery fence и
сохраняется с результатом, checkpoint, дочерней моделью и terminal telemetry.
Predict дочерней модели снова требует точного совпадения её нового data
contract. Parent остаётся неизменным и защищён от удаления только до terminal
state зависимого fit.

Public contract переходит на Flight v9, process manifest — на worker v10,
terminal fit telemetry — на v4. Checkpoint format остаётся v5, поскольку
структура weights и самого checkpoint не меняется. Flight v8 aliases и
compatibility layer не поддерживаются.

## Рассмотренные альтернативы

- Ослабить `publishedModel`. Отклонено: strict warm start потерял бы точную и
  проверяемую семантику.
- Игнорировать несовпадение digest по флагу. Отклонено: не создаёт отдельной
  identity действия и скрывает transfer в общем режиме.
- Добавить отдельный structural/profile digest. Отклонено: закрытый data
  contract уже содержит полный набор структурных полей, которые Transformer
  способен проверить.
- Переставлять feature blocks по FIGI. Отклонено: FIGI и slot semantics
  принадлежат Consumer-у и не должны пересекать ML implementation boundary.
- Частично загружать backbone или совместимые heads. Отклонено: это другой
  класс transfer learning с неоднозначным сопоставлением параметров.

## Последствия

- Strict warm start и cross-instrument transfer остаются разными явными
  действиями с разными compatibility rules.
- Смена profile, shape, targets, objective или architecture отклоняется до
  создания job как `MODEL_SCHEMA_MISMATCH`.
- Дочерняя generation физически независима после публикации и сохраняет точный
  новый data contract.
- Existing checkpoint-v5 generation можно использовать как transfer parent без
  преобразования artifact и без изменения PostgreSQL schema.
- Consumer отвечает за корректность нового семантического наполнения
  структурно совместимых slots.

## Текущая документация

- [Контракт Arrow Flight v9](../../app/contracts/flight/v9/README.md)
- [Worker process contract v10](../../app/contracts/worker/v10/README.md)
- [Terminal fit telemetry v4](../../app/contracts/metrics/fit_run/v4/README.md)
- [Интеграция Consumer-ов](../consumer-flight-integration.md)
- [Training runtime](../training-runtime.md)
- [Управление опубликованными моделями](../operations/published-models.md)
