# Управление опубликованными моделями

> Тип: операционное руководство. Просмотр и необратимое удаление published
> model generations Transformer.

Это руководство задаёт текущую операторскую процедуру для server-owned моделей
в `models/` и их metadata в PostgreSQL. Rationale двухфазного hard delete
сохранён в [ADR 0016](../adr/0016-hard-delete-published-models.md), а storage и
maintenance boundaries описаны в
[`Flight runbook`](flight-service.md#хранение-и-ошибки-хранилища).

Публичные discovery и detail выполняются owner-scoped actions из
[Model Catalog Query v4](../../app/contracts/model_catalog/v4/README.md).
Каталог является read-only и не заменяет описанные здесь административные
команды удаления.

Команды требуют актуальной PostgreSQL schema. Порядок её проверки находится в
[`руководстве по migrations`](database-migrations.md).

## Просмотреть доступные модели

```bash
./.venv/bin/python ./app/main.py models list
```

Команда показывает только доступные generations в состоянии `AVAILABLE`:
точный `MODEL REF`, owner, label, generation, state и время создания.
`MODEL REF` является management identity конкретной immutable generation.

Та же working registry определяет membership публичного Model Catalog. List
catalog не читает checkpoint; detail дополнительно проверяет canonical metadata
и полный SHA-256 artifact в пределах contract budget. OpenSearch telemetry и
filesystem scan не добавляют generation в каталог.

Перед удалением зафиксируйте точный `MODEL REF`. Alias или label команда
`models delete` не принимает, чтобы ротация alias не могла изменить target
административной операции.

## Запросить удаление

```bash
./.venv/bin/python ./app/main.py models delete <MODEL_REF>
```

При успехе команда выводит выбранную identity и состояние:

```text
Model: <MODEL_REF>
State: DELETING
```

Транзакция запроса:

- блокирует новые prediction jobs и `publishedModel` fit для этой generation;
- отклоняется, если на модель ссылается любой незавершённый prediction или
  `publishedModel` fit job;
- переводит модель в `DELETING`.

Поддерживаемой команды undo нет, поэтому перед `models delete` убедитесь, что
выбрана точная generation и вызывающая система больше не должна использовать
её.

## Дождаться физического удаления

После commit модель сразу исчезает из обычного `models list`. Физическое
удаление выполняет maintenance работающего Flight service:

1. удаляет каталог `models/{modelRef}`;
2. физически удаляет working row модели из PostgreSQL;
3. создаёт минимальную запись `DELETED` с identity и timestamps.

Пока операция не завершена, состояние видно отдельной командой:

```bash
./.venv/bin/python ./app/main.py models list --deleted
```

Она показывает `DELETING` и завершённые `DELETED` records. У `DELETING`
колонка `DELETED AT` остаётся пустой; после успешной очистки она получает
timestamp.

Filesystem error оставляет модель в `DELETING`, и maintenance повторяет
очистку. Если состояние не меняется, проверьте logs сервиса и доступность
`models/`; не удаляйте PostgreSQL row вручную.

## Повторные операции и история

Повторный `models delete` до завершения очистки возвращает текущее состояние
`DELETING`. После завершения working row отсутствует, поэтому повторный delete
возвращает `model generation not found`.

Archive сохраняет только owner, label, generation, model reference и
timestamps. Checkpoint paths, hashes, contracts и полная metadata не
сохраняются, поэтому восстановить удалённую модель из archive невозможно.
Generation остаётся монотонным и не используется повторно.

Удаление модели не отменяет pending OpenSearch delivery и не удаляет уже
принятые OpenSearch documents. Run telemetry имеет собственный retention
lifecycle.

## Проверить полный lifecycle

Для проверки выбранной generation:

1. Найдите точный `MODEL REF` через `models list`.
2. Убедитесь, что у generation нет незавершённых prediction либо
   `publishedModel` fit jobs.
3. Выполните `models delete <MODEL_REF>` и проверьте `State: DELETING`.
4. Убедитесь, что generation исчезла из обычного `models list`.
5. Наблюдайте `models list --deleted`, пока state не станет `DELETED` и не
   появится `DELETED AT`.

Эта проверка необратима и должна выполняться только для generation, которую
разрешено физически удалить.
