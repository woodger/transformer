# Журнал архитектурных решений

> Тип: указатель. Этот документ задаёт admission и lifecycle исторических
> Architecture Decision Records Transformer.

> **ADR is a decision record, not a system reference.**

ADR отвечает только на вопрос, почему в конкретный момент было принято
архитектурное решение. Он не является contract, policy, runbook или описанием
текущего устройства системы.

Общие роли и источники документации задаёт
[политика ведения документации](../policy/documentation-policy.md).

## Authority

При расхождении документов действуют следующие роли:

| Источник | Роль |
| --- | --- |
| `app/contracts/` и contract documents | текущий обязательный contract |
| `docs/policy/` | долгосрочные нормативные правила |
| architecture, reference, operations и deployment docs | текущее состояние системы |
| `docs/adr/` | исторический rationale решения |

ADR может объяснять происхождение действующего правила, но никогда не
определяет и не переопределяет current behavior, configuration, wire format,
persistence schema или deployment procedure. Ссылка на `Accepted` ADR не
доказывает, что описанные в нём implementation details всё ещё актуальны.

## Admission

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

Допустимые статусы: `Proposed`, `Accepted`, `Rejected`, `Superseded`.

После `Accepted` содержательная часть ADR immutable. Разрешены только:

- исправление опечатки;
- исправление broken link;
- изменение статуса;
- ссылка `Superseded by`;
- обновление ссылок в разделе `Current documentation` без копирования
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
Status
Decision date
Context
Decision
Alternatives considered
Consequences
Current documentation
```

`Current documentation` содержит ссылки на действующие contracts, policies и
references. Он не воспроизводит их значения, команды или структуры.

## Текущие записи

| ADR | Status | Decision |
| --- | --- | --- |
| [0001](0001-arrow-flight-job-service.md) | Accepted | Выделить Transformer в Arrow Flight job service |
| [0003](0003-durable-resumable-training-and-device-aware-execution.md) | Accepted | Хранить recovery fit между attempts и привязывать CUDA attempt к device |
| [0004](0004-clean-architecture-process-boundaries.md) | Accepted | Разделить service, worker и admin process boundaries |
| [0005](0005-durable-streaming-flight-v3.md) | Accepted | Использовать durable streaming lifecycle и cross-system fencing |
| [0007](0007-target-aligned-flight-v4.md) | Accepted | Совместить public prediction с target-space |
| [0009](0009-centralized-training-metrics.md) | Accepted | Доставлять training telemetry best effort через artifact и outbox |
| [0015](0015-unified-indicator-identity-flight-v5.md) | Accepted | Использовать единую public identity индикаторов |
| [0016](0016-hard-delete-published-models.md) | Accepted | Физически удалять модели с минимальным audit archive |
| [0020](0020-local-opaque-api-access-tokens.md) | Accepted | Использовать local opaque database-backed access tokens |
| [0021](0021-provider-neutral-gpu-device-interface.md) | Accepted | Использовать provider-neutral public identity `gpu` |
| [0022](0022-declarative-target-objectives.md) | Superseded | Передавать выбранные targets и declarative objective через Flight |
| [0023](0023-weights-only-published-model-initialization.md) | Accepted | Инициализировать новый fit только weights опубликованной модели |
| [0024](0024-cross-instrument-transfer-initialization.md) | Superseded | Отделить cross-instrument transfer от strict warm start |
| [0025](0025-unified-published-model-initialization.md) | Superseded | Объединить structural weights-only initialization в `publishedModel` |
| [0026](0026-strict-published-model-warm-start.md) | Accepted | Ограничить `publishedModel` точным data-contract digest |
| [0027](0027-compact-indexed-feature-block-input.md) | Accepted | Передавать Flight input как compact indexed feature blocks |
| [0028](0028-consumer-neutral-xy-contract-boundary.md) | Accepted | Использовать consumer-neutral target и objective contract |
| [0029](0029-owner-scoped-model-catalog-query.md) | Accepted | Публиковать owner-scoped каталог immutable model generations |
| [0030](0030-owner-scoped-training-telemetry-query.md) | Accepted | Публиковать owner-scoped training telemetry опубликованной generation |

Отсутствующие номера принадлежат документам, не прошедшим admission при
нормализации. Их содержание доступно в Git history, но не является частью
текущего ADR-журнала.
