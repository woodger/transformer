# Журнал архитектурных решений

> Тип: указатель. Этот документ задаёт правила допуска и lifecycle исторических
> Architecture Decision Records Transformer.

> **ADR — запись решения, а не описание системы.**

ADR отвечает только на вопрос, почему в конкретный момент было принято
архитектурное решение. Он не является contract, policy, runbook или описанием
текущего устройства системы.

Общие роли и источники документации задаёт
[политика ведения документации](../policy/documentation-policy.md).

## Приоритет источников

При расхождении документов действуют следующие роли:

| Источник | Роль |
| --- | --- |
| `app/contracts/` и contract documents | текущий обязательный contract |
| `docs/policy/` | долгосрочные нормативные правила |
| docs architecture, reference, operations и deployment | текущее состояние системы |
| `docs/adr/` | историческое обоснование решения |

ADR может объяснять происхождение действующего правила, но никогда не
определяет и не переопределяет текущее behavior, configuration, wire format,
persistence schema или deployment procedure. Ссылка на ADR со статусом
`Accepted` не доказывает, что описанные в нём implementation details всё ещё
актуальны.

## Допуск

Новый ADR создаётся только тогда, когда:

- существуют как минимум две реалистичные альтернативы;
- решение трудно или дорого отменить либо оно задаёт долгосрочное ограничение;
- решение пересекает component/process boundary или существенно влияет на
  security, durability, persistence либо compatibility;
- rationale нельзя надёжно восстановить через полгода только по коду и
  текущей документации.

ADR не создаётся для:

- описания существующего компонента или текущего data flow;
- API, DTO, schema, CLI и configuration reference;
- структуры каталогов и реализации отдельного class/module;
- migration или deployment instructions;
- acceptance checklist, implementation plan или отчёта о выполнении;
- временной идеи без принятого архитектурного решения.

Практический тест: если удалить historical context, alternatives и саму
decision delta, а оставшийся текст всё ещё полезен как самостоятельный
справочник по системе, этот текст принадлежит living documentation, а не ADR.

## Lifecycle и изменения

Допустимые статусы: `Предложено`, `Принято`, `Отклонено`, `Заменено`.

После статуса `Принято` содержательная часть ADR immutable. Разрешены только:

- исправление опечатки;
- исправление broken link;
- изменение статуса;
- ссылка на заменяющий ADR;
- обновление ссылок в разделе «Текущая документация» без копирования
  текущего содержимого.

Изменившееся решение получает новый ADR. Старый record не переписывается под
новую систему. Номера монотонны, пропуски сохраняются и повторно не
используются.

Записи ниже были однократно нормализованы при введении этой политики:
справочные и procedural части перенесены в living docs, а прежние полные версии
остались в Git history. После этой границы на них действует правило
immutability.

## Состав ADR

Каждая запись начинается с role banner о ненормативной исторической природе и
содержит только:

```text
Статус
Дата решения
Контекст
Решение
Рассмотренные альтернативы
Последствия
Текущая документация
```

Раздел «Текущая документация» содержит ссылки на действующие contracts, policies и
references. Он не воспроизводит их значения, команды или структуры.

## Текущие записи

| ADR | Статус | Решение |
| --- | --- | --- |
| [0001](0001-arrow-flight-job-service.md) | Принято | Выделить Transformer в Arrow Flight job service |
| [0003](0003-durable-resumable-training-and-device-aware-execution.md) | Принято | Хранить recovery fit между attempts и привязывать CUDA attempt к device |
| [0004](0004-clean-architecture-process-boundaries.md) | Принято | Разделить service, worker и admin process boundaries |
| [0005](0005-durable-streaming-flight-v3.md) | Принято | Использовать durable streaming lifecycle и cross-system fencing |
| [0007](0007-target-aligned-flight-v4.md) | Принято | Совместить public prediction с target-space |
| [0009](0009-centralized-training-metrics.md) | Принято | Доставлять training telemetry best effort через artifact и outbox |
| [0015](0015-unified-indicator-identity-flight-v5.md) | Принято | Использовать единую public identity индикаторов |
| [0016](0016-hard-delete-published-models.md) | Принято | Физически удалять модели с минимальным audit archive |
| [0020](0020-local-opaque-api-access-tokens.md) | Принято | Использовать local opaque database-backed access tokens |
| [0021](0021-provider-neutral-gpu-device-interface.md) | Принято | Использовать provider-neutral public identity `gpu` |
| [0022](0022-declarative-target-objectives.md) | Заменено | Передавать выбранные targets и declarative objective через Flight |
| [0023](0023-weights-only-published-model-initialization.md) | Принято | Инициализировать новый fit только weights опубликованной модели |
| [0024](0024-cross-instrument-transfer-initialization.md) | Заменено | Отделить cross-instrument transfer от strict warm start |
| [0025](0025-unified-published-model-initialization.md) | Заменено | Объединить structural weights-only initialization в `publishedModel` |
| [0026](0026-strict-published-model-warm-start.md) | Принято | Ограничить `publishedModel` точным data-contract digest |
| [0027](0027-compact-indexed-feature-block-input.md) | Принято | Передавать Flight input как compact indexed feature blocks |
| [0028](0028-consumer-neutral-xy-contract-boundary.md) | Принято | Использовать consumer-neutral target и objective contract |
| [0029](0029-owner-scoped-model-catalog-query.md) | Принято | Публиковать owner-scoped каталог immutable model generations |
| [0030](0030-owner-scoped-training-telemetry-query.md) | Принято | Публиковать owner-scoped training telemetry опубликованной generation |
| [0031](0031-subject-specific-contract-vocabulary.md) | Заменено | Использовать предметный vocabulary shared contracts без `kind` |
| [0032](0032-public-contract-simplification-clean-cut.md) | Принято | Упростить публичную границу и удалить несовместимое durable state |

Отсутствующие номера принадлежат документам, не прошедшим admission при
нормализации. Их содержание доступно в Git history, но не является частью
текущего ADR-журнала.
