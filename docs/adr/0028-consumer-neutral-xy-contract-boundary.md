# ADR 0028: consumer-neutral граница `x/y`

- Status: Accepted
- Decision date: 2026-09-05
- Supersedes: [ADR 0022](0022-declarative-target-objectives.md)

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Declarative objective устранил передачу исполняемого loss code, но Transformer
продолжил владеть фиксированными target identities, их порядком,
transformations и отображением target в operator. Та же предметная семантика
дублировалась в Consumer-е и provider-е, поэтому новый target на уже
поддержанных численных primitives требовал синхронного изменения Transformer.

Физическое compact-представление `x` не является причиной этой связи:
`indexedFeatureBlocks` передаёт только Consumer-computed values, block geometry
и offsets. Требуется изменить semantic boundary `y` и objective, не перенося в
Transformer formulas признаков или targets.

## Решение

Consumer materializer формирует self-contained ordered target contract и
objective до Flight boundary. Target identity остаётся устойчивой opaque identity;
constraint наблюдаемого `y`, loss-input transformation, public prediction
transformation, operators, weights и role bindings задаются явно и не
выводятся из значения identity.

Transformer владеет закрытым mathematical language: семантикой и tensor-
реализацией constraints, transformations, operators и private resource kinds,
а также autograd, model, checkpoint и recovery. Private model resource
объявляется абстрактно через local identity и Transformer-owned kind; детали
layer, parameterization, tensor layout и storage не пересекают boundary.
Исполняемый Consumer code и hidden server-side target presets запрещены.

Canonical full documents остаются источниками смысла. Compatibility и
diagnostics используют layered D1 digests для Consumer data, target contract,
objective, model contract, job configuration и physical checkpoint. Operator
language имеет собственную immutable revision и advertised capabilities и не
обязан эволюционировать синхронно с Flight workflow.

`indexedFeatureBlocks` сохраняется как physical encoding `x`, входит в job и
recovery fencing, но не в target, objective или model identity. Переход к
новому contract выполняется clean cut: aliases, legacy runtime и conversion
старых models отсутствуют.

## Рассмотренные альтернативы

- Оставить фиксированный target catalog в Transformer. Отклонено: это
  сохраняет дублирование Consumer semantics и требует provider release для
  нового target на существующих primitives.
- Сделать target identity opaque, но продолжить выбирать transformations и
  operators по её значению. Отклонено: behavioral coupling останется скрытым.
- Передавать только positional target indices. Отклонено: перестановка slots
  становится опасной для bindings, compatibility и diagnostics.
- Передавать Consumer-owned executable objective. Отклонено из-за security,
  reproducibility и runtime-language coupling.
- Вернуть dense `x`. Отклонено: это не решает semantic boundary и возвращает
  многократную физическую материализацию overlapping windows.

## Последствия

- Новый target на поддержанных primitives не требует изменения Transformer.
- Transformer валидирует generic numerical contract без branching по
  Consumer-owned target names.
- Target order, transformations, objective graph и private resource
  declarations становятся exact model compatibility identity.
- Consumer и Transformer обязаны проверять общие canonicalization, digest и
  numerical fixtures независимо.
- Следующий public/process/metrics contract является breaking boundary;
  существующие models, checkpoints, runtime metadata и historical metrics не
  сохраняют исполняемую совместимость.
- Точные JSON/Arrow schemas, protocol и format versions, persistence layout и
  PyTorch implementation определяются отдельно и не закрепляются этим ADR.

## Текущая документация

- [Нормативный semantic contract v1](../../app/contracts/semantic/v1/README.md)
- [Контракт Arrow Flight v11](../../app/contracts/flight/v11/README.md)
- [Worker process contract v12](../../app/contracts/worker/v12/README.md)
- [Checkpoint/recovery contract v6](../../app/contracts/checkpoint/v6/README.md)
- [Training metrics contract v5](../../app/contracts/metrics/v5/README.md)
- [Terminal fit metrics contract v5](../../app/contracts/metrics/fit_run/v5/README.md)
- [Архитектура Transformer](../architecture.md)
- [ADR 0027: compact indexed feature-block input](0027-compact-indexed-feature-block-input.md)
