# ADR 0027: compact indexed feature-block input

- Status: Accepted
- Decision date: 2026-09-04

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Flight v9 принимал полностью развёрнутый tensor каждого sequence example.
Перекрывающиеся native windows повторялись между соседними observations, а
observations повторялись между соседними sequences. Для реального набора с
длинными окнами это превращало логически неизменную выборку в несколько
терабайт transport/storage data и исчерпывало существующий `maxJobBytes` задолго
до завершения загрузки.

Один однородный window descriptor не решает задачу: Consumer использует
ordered multi-timeframe feature blocks с разными window sizes, native row
widths, instrument scopes и независимо меняющимися causal projections.
Transformer при этом не должен становиться владельцем расчёта признаков или
интерпретации instruments и intervals.

## Решение

Перевести единственный public API на Flight v10 с compact-only
`indexedFeatureBlocks`. Create объявляет ordered геометрию всех feature blocks.
Каждый Arrow range chunk передаёт рассчитанные Consumer-ом native feature rows,
локальные offsets sequence observations и fit targets. Chunk self-contained:
весь необходимый window halo находится в нём, а offsets не ссылаются на другой
artifact.

Transformer проверяет block layout, range/example continuity, bounds,
физические и логические counters и immutable manifest. Worker детерминированно
восстанавливает прежний block-major dense tensor ограниченными срезами перед
существующим batching/shuffle path, не материализуя полный dense dataset.

`sourceEncoding` входит в create hash и recovery fencing, но не меняет
`dataContractSha256`, objective, checkpoint или model identity при численно
точном восстановлении logical examples. Public status и receipts разделяют
physical chunks/native rows/bytes и logical training rows.

Flight v9 dense input и compatibility aliases не поддерживаются. Внутренний
worker contract становится v11, PostgreSQL schema — revision `0023`.
Checkpoint/recovery v5, prediction schema и metrics v4 сохраняются.

## Рассмотренные альтернативы

- Увеличить `maxJobBytes`. Отклонено: это сохраняет многократную
  материализацию, terabyte-scale transport и disk pressure.
- Уменьшить window, sequence length, число instruments или обучающий период.
  Отклонено: эти значения принадлежат ML-семантике эксперимента.
- Рассчитывать indicators и multi-timeframe projections в Transformer.
  Отклонено: это дублирует Consumer-owned feature semantics и переносит её
  через неверную component boundary.
- Описать один общий native block. Отклонено: он не выражает независимую
  геометрию и causal projection multi-timeframe blocks.
- Поддерживать dense и compact schemas параллельно. Отклонено: согласовано
  maintenance window без active jobs, поэтому второй runtime path не имеет
  эксплуатационной ценности.

## Последствия

- Физический input масштабируется по native rows и offsets, а не по полностью
  повторённым sequence windows.
- Consumer сохраняет полное владение feature computation, causal alignment и
  логическим порядком examples; Transformer выполняет только проверяемую
  реконструкцию.
- Payload и range boundaries могут потребовать повторения небольшого halo, но
  не меняют shuffle, batching, losses или prediction semantics.
- Обе стороны обязаны проверять cross-project fixtures для single-block,
  multi-timeframe, hour-boundary, split-range и missing-data случаев.
- Upgrade требует terminal state всех jobs прежнего контракта. Старые dense
  artifacts не преобразуются и не возобновляются worker-ом v11.
- Ошибка в offsets теперь является contract violation до durable commit, а
  различие physical и logical counters становится видимой частью operations и
  telemetry.

## Текущая документация

- [Контракт Arrow Flight v11](../../app/contracts/flight/v11/README.md)
- [Worker process contract v12](../../app/contracts/worker/v12/README.md)
- [Интеграция Consumer-ов](../consumer-flight-integration.md)
- [Архитектура Transformer](../architecture.md)
- [Управление migrations](../operations/database-migrations.md)
