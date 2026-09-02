# ADR 0022: declarative target objectives

- Status: Accepted
- Decision date: 2026-08-31

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Прежний Flight contract всегда обучал шесть target-координат и передавал только
digest objective, семантика которого независимо воспроизводилась Consumer-ом и
Transformer. Это не позволяло изолировать отдельные target-задачи для
экспериментов и создавало неявное дублирование loss composition на границе
компонентов. Loss stages дополнительно связывали objective identity с порядком
включения компонентов во время обучения.

Для исследования negative transfer и дисбаланса loss scales требуется менять
target subset, direct/auxiliary losses и их статические веса независимо от
training policy. При этом исполняемый код и autograd не должны пересекать
Flight boundary.

## Решение

Consumer передаёт в fit create каноническое непустое подмножество `targets` и
закрытый декларативный objective. Transformer владеет допустимой target
identity, семантикой operators, tensor implementation и autograd; он валидирует
пару `{targets, objective}`, вычисляет её canonical digest и сохраняет полный
документ.

Выбранные `targets` определяют физическую ширину `tgt`, набор public model heads
и prediction. Private heads создаются только когда этого требует objective.
Objective, training policy и observational diagnostics остаются разными
контрактами. Все objective components активны с первого optimizer step; stage
schedule отсутствует. `ExpectedValue` и `RiskAdjustedExpectedValue` являются
разными operators.

Checkpoint навсегда связан с точными targets и objective. Predict с другим
набором или objective отклоняется как `MODEL_SCHEMA_MISMATCH`.

## Рассмотренные альтернативы

- Оставить фиксированные шесть targets и проводить только полные multitask
  эксперименты. Отклонено: такой эксперимент одновременно меняет несколько
  факторов и не изолирует target-задачу.
- Передавать только digest и дублировать objective composition в Consumer-е и
  Transformer. Отклонено: digest не раскрывает исполняемую декларацию и не
  устраняет два источника semantic truth.
- Передавать исполняемый loss code. Отклонено из-за security boundary,
  language coupling и невозможности закрытой schema validation.
- Считать diagnostics частью objective identity. Отклонено: наблюдение не
  должно менять математическую задачу или совместимость checkpoint.
- Поддерживать Flight v6 и v7 параллельно. Отклонено, поскольку переход получил
  согласованное maintenance window и активные jobs отсутствовали.

## Последствия

- Consumer может определять single-target и multitask эксперименты без
  изменения Transformer code.
- Transformer остаётся единственным владельцем численной реализации operators
  и проверяет их зависимости до создания job.
- Model shape, Arrow schemas, metrics и checkpoint metadata становятся
  target-dependent.
- Изменение targets или objective создаёт другую model identity и требует
  нового обучения.
- Checkpoint/recovery, worker и metrics contracts получают новые версии;
  compatibility layer для прежних форматов отсутствует.
- Gradient-interaction diagnostics могут измерять нормы и попарные cosine
  gradients, не входя в objective digest.

## Текущая документация

- [Контракт Arrow Flight v9](../../app/contracts/flight/v9/README.md)
- [Функция потерь](../losses.md)
- [Training runtime](../training-runtime.md)
- [Training metrics v4](../../app/contracts/metrics/v4/README.md)
