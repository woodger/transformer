# Операционные руководства

> Тип: указатель. Текущие operator-facing процедуры для работающего
> Transformer и управляемых им ресурсов.

Operational Guide отвечает на вопрос, что оператор должен сделать, в каком
порядке проверить результат и какие ограничения учитывать. Текущее устройство
системы описывают профильные references, обязательное внешнее поведение —
contracts, а причины архитектурных решений — ADR.

Один lifecycle имеет один основной operational source. CLI reference и
deployment guides могут содержать краткий пример или prerequisite, но должны
ссылаться на соответствующее руководство вместо копирования процедуры.

## Управление ресурсами

| Ресурс | Руководство | Операции |
| --- | --- | --- |
| API access tokens | [Управление API access tokens](api-access-tokens.md) | issue, list, передача, ротация, revoke |
| PostgreSQL schema | [Управление схемой PostgreSQL](database-migrations.md) | status, apply, compatibility, rollback boundary |
| Published models | [Управление опубликованными моделями](published-models.md) | list, delete, наблюдение `DELETING` и `DELETED` |

## Эксплуатация service и deployment

- [Flight runbook](flight-service.md) описывает runtime requirements,
  storage, запуск, recovery, shutdown, health и ограничения сервиса.
- [Systemd deployment contract](../deployment/systemd.md) задаёт единственный
  ручной production deployment на Fedora.
- [OpenSearch deployment guide](../deployment/opensearch.md) описывает
  подготовку projection, настройку publisher-а и проверку доставки metrics.

Quick start находится отдельно в [`getting-started.md`](../getting-started.md),
а точный command tree и options — в [`cli/index.md`](../cli/index.md).
