# ADR 0015: единая public identity индикаторов

- Status: Accepted
- Decision date: 2026-08-20

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Inventory использовал две строковые формы одной indicator identity: имя enum
member и отдельное wire value. Аббревиатуры особенно легко расходились по
регистру. В результате target мог иметь разные имена в objective, checkpoint,
telemetry и Consumer code.

Одновременно один digest data contract не позволял явно определить ML profile
модели, поэтому Consumer восстанавливал profile по косвенным признакам.

## Решение

Одна регистрозависимая public semantic identity используется во всех wire,
objective, checkpoint, recovery и telemetry documents. Numeric enum ordinal
остаётся только внутренним dispatch detail Consumer-а и не пересекает contract
boundary. Legacy spellings не являются aliases.

Data contract содержит явный opaque profile вместе с остальными semantic
parameters и покрывает его canonical digest. Transformer хранит и возвращает
весь документ без интерпретации profile; predict и recovery требуют точного
совпадения.

Изменение public identity выполнено как breaking Flight version boundary без
совместного runtime старой и новой semantics.

## Рассмотренные альтернативы

- Поддерживать enum name и wire alias одновременно. Отклонено, потому что это
  сохраняет два источника semantic identity.
- Нормализовать casing на каждой границе. Отклонено из-за неоднозначности
  abbreviations и скрытых mappings.
- Определять profile только по digest или model metadata heuristics.
  Отклонено: profile должен быть явной частью Consumer-owned contract.
- Расширить прежнюю protocol version. Отклонено как незаметная breaking
  semantic change.

## Последствия

- Indicator identity не требует cross-language mapping.
- Objective, model metadata и telemetry используют одинаковые names.
- Consumer может выбирать ML profile по явному полю, не интерпретируя digest.
- Старые jobs, checkpoints и models не получают неявной compatibility.

## Текущая документация

- [Контракт Arrow Flight v5](../../app/contracts/flight/v5/README.md)
- [Интеграция Consumer-ов](../consumer-flight-integration.md)
- [Функция потерь](../losses.md)
- [Политика metrics](../metrics.md)
