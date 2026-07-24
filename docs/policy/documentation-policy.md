# Политика ведения документации

> Type: Policy. Этот документ задаёт роли документов и правила их обновления.

Документация должна помогать запустить Transformer, понять его contracts и
безопасно изменить систему. Один факт должен иметь один основной источник
правды; в остальных местах используется краткое резюме и ссылка.

## Текущая структура

| Тема | Основной источник |
| --- | --- |
| Назначение, установка и общий CLI | `readme.md` |
| Environment example | `.env.example` |
| ML architecture и training decisions | профильные документы в `docs/` |
| Flight service behavior и operations | `docs/flight-operations.md` |
| Inventory integration | `docs/inventory-flight-handoff.md` |
| Архитектурные решения | `docs/adr/` |
| Ручное production deployment | `docs/deployment/` |
| Правила разработки | `docs/policy/` |
| Нормативный Flight v2 contract | `contracts/flight/v2/` |
| История релизов | `CHANGELOG.md` |

В проекте нет отдельного `docs/index.md`; навигационной входной точкой остаётся
`readme.md`, а для политик — `docs/policy/index.md`.

## README

`readme.md` отвечает на вопросы:

- что делает проект;
- какие runtime dependencies нужны;
- как запустить основные команды;
- где находятся подробные contracts и operations.

Новые длинные объяснения recovery, deployment, protocol edge cases и
архитектурных решений следует помещать в профильный документ, оставляя в README
краткую ссылку.

## CLI help

CLI help является публичным контрактом.

Источники:

- `app/cli/help.py` — parser metadata, descriptions и rendering;
- `app/cli/args.py` — parsing entry;
- `app/commands/` — command behavior;
- `app/main.py` — dispatch и lifecycle.

Глобальный help остаётся компактным: Usage, global options и список command
groups. Полные arguments/options находятся в leaf command help:

```text
transformer <command> --help
```

Блок `Examples` добавляется только когда показывает неочевидную комбинацию,
формат значения, transport mode или side effect. Пример, дублирующий `Usage`,
не нужен.

При изменении command path, option, default или output одновременно
обновляются help tests и относящаяся пользовательская документация.

## Environment

`.env.example` содержит безопасный рабочий образец и русские комментарии к
группам переменных. Фактические правила parsing и defaults находятся в
`app/database/config.py` и `app/flight/config.py`.

Документация не должна:

- помещать реальные credentials в repository;
- обещать default, которого нет в коде;
- дублировать удалённые environment names;
- добавлять `POSTGRES_SSLMODE`, локальную SQLite или token file, которых нет в
  текущем contract.

## Нормативные contracts

JSON Schemas и golden fixtures в `contracts/flight/v2/` нормативны для wire
format. README или operations guide не могут переопределять их.

Изменение Flight contract требует синхронно проверить:

- schemas и fixtures;
- parser/serializer;
- contract tests;
- version/compatibility policy;
- Inventory handoff.

Checkpoint format, Arrow columns и framed protocol также должны описываться в
одном основном месте и проверяться тестами.

## ADR

ADR фиксирует значимое принятое решение, альтернативы и последствия. ADR нужен
для изменения service boundary, persistence ownership, durability model,
protocol или multi-process architecture.

ADR не используется как пошаговый runbook и не переписывается так, будто
предыдущее решение никогда не существовало. Новое решение оформляется новым
ADR или явным изменением статуса.

## Deployment и operations

`docs/deployment/` содержит короткие ручные инструкции установки и запуска.
`docs/flight-operations.md` содержит подробные lifecycle, recovery, security и
storage semantics.

Deployment guide не должен дублировать всю архитектуру сервиса или
автоматизировать изменение production-хоста.

## Policy documents

Документы в `docs/policy/` должны быть прямыми и применимыми на review.
Примеры обязаны использовать Python и реальные concepts Transformer, но не
должны превращать общую policy в описание конкретной модели или consumer.

## Имена и оформление

- постоянные filenames — английский `kebab-case`, если путь уже не закреплён;
- H1 — понятный русский заголовок;
- metadata `> Type: Policy|Reference` рекомендуется для policy/reference;
- code identifiers и wire fields сохраняются в исходной форме;
- relative links должны разрешаться из текущего файла;
- путь документа считается стабильным контрактом и не меняется без причины.

## Добавление документа

Перед созданием файла нужно ответить:

1. Какой вопрос он закрывает?
2. Почему существующий источник не подходит?
3. Где на него будет ссылка?
4. Не дублирует ли он contract, ADR или README?

Временная идея без устойчивой роли остаётся issue/plan, а не постоянным
документом.

## Обновление

Если code change меняет запуск, CLI, environment, schema, persistence,
protocol, checkpoint или архитектурную границу, соответствующая документация
обновляется в том же change set.

Устаревший текст удаляется, а не сохраняется «на всякий случай». Перед
завершением проверяются links, команды, paths, version markers и отсутствие
секретов.
