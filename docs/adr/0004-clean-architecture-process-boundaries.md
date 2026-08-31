# ADR 0004: Clean Architecture по process boundaries

- Status: Accepted
- Decision date: 2026-08-06

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Transformer имеет три разных runtime: долговечный Flight service,
короткоживущий ML worker одной execution attempt и административный CLI.
Flight является inbound transport, а не владельцем job lifecycle,
persistence, artifact storage или ML execution. Импорт worker implementation в
service также связывал бы control plane с Torch, CUDA и process-local state.

## Решение

Clean Architecture применяется независимо внутри каждого process boundary:

- service разделяет domain, application, inbound/outbound adapters и
  composition root;
- worker является изолированным ML runtime одной attempt;
- admin CLI создаёт только необходимые короткоживущие resources;
- service и worker обмениваются данными через отдельный versioned process
  contract, не импортируя implementations друг друга.

Application ports называются по capabilities, а concrete database,
filesystem, transport и subprocess concerns остаются в adapters. Worker пишет
только attempt artifacts; проверку и публикацию public outputs, recovery и
models выполняет service.

## Рассмотренные альтернативы

- Один глобальный набор слоёв и bootstrap для всех процессов. Отклонено,
  потому что он смешивает разные resource lifecycles и import graphs.
- Импортировать worker implementation в service. Отклонено из-за coupling с
  Torch/CUDA и потери process isolation.
- Разрешить worker доступ к PostgreSQL и public job state. Отклонено, чтобы
  сохранить единственного владельца lifecycle и fencing.
- Ввести generic repository, unit of work и DI container. Отклонено без
  нескольких implementations и конкретного consumer-а abstractions.

## Последствия

- Каждый process имеет собственный composition root и ограниченный import
  graph.
- Worker crash, CUDA state и pipes принадлежат одной attempt.
- Public contract и внутренний worker contract версионируются независимо.
- Artifact publication требует явной границы между worker staging и
  service-owned durable commit.

## Текущая документация

- [Архитектурная политика](../policy/architecture.md)
- [Worker process contract v8](../../app/contracts/worker/v8/README.md)
- [Контракт Arrow Flight v7](../../app/contracts/flight/v7/README.md)
