# Политики проекта

> Тип: указатель. Рабочие правила изменения Transformer Arrow Flight service.

Политики помогают сохранять архитектуру, публичные контракты и эксплуатационные
свойства проекта. Они применяются вместе с текущим кодом, тестами, профильной
документацией и нормативным Flight contract, а не заменяют их.
Исторический rationale прошедших admission решений находится в
[`docs/adr/`](../adr/index.md), но ADR не определяет текущие правила.

## Основные политики

- [Политика внесения изменений](./change-policy.md)
- [Архитектурная политика](./architecture.md)
- [Политика тестирования](./testing-policy.md)
- [Политика ведения документации](./documentation-policy.md)
- [Нефункциональные требования](./nonfunctional.md)
- [Политика metrics и OpenSearch](./metrics-policy.md)

## Политики реализации

- [Политика выделения абстракций](./abstraction-policy.md)
- [Политика жизненности кода](./code-liveness-policy.md)
- [Политика комментариев](./comment-style.md)
- [Политика зависимостей](./dependencies-policy.md)
- [Политика именования](./naming-policy.md)
- [Политика типов и runtime-контрактов](./typing-policy.md)
- [Политика Python runtime и виртуальных окружений](./python-runtime-policy.md)
- [Политика скриптов и точек запуска](./scripts-policy.md)

## Когда обращаться к политикам

- перед изменением CLI, Arrow/Flight contract, checkpoint или persistence;
- перед архитектурным переносом, удалением кода или добавлением зависимости;
- перед изменением тестового запуска, entrypoint или deployment wiring;
- при code review и подготовке релиза.
