# Управление схемой PostgreSQL

> Тип: операционное руководство. Проверка, применение и compatibility boundary
> migrations schema Transformer.

Это руководство задаёт текущую операторскую процедуру для PostgreSQL schema
`transformer`. Параметры подключения и роль PostgreSQL в runtime описаны в
[`Flight runbook`](flight-service.md#настройка-postgresql), а точный
command tree — в [`справочнике CLI`](../cli/index.md).

Flight service не применяет migrations при запуске. `flight serve`, команды
управления API access tokens и published models требуют schema на текущем
Alembic head и отказываются работать при несовпадении revisions.

## Предварительные условия

- Выполняйте команды из корня working copy через project `.venv`.
- Настройте CLI на ту же PostgreSQL database, что использует Flight service.
- Не помещайте PostgreSQL credentials в repository, command output или логи.
- Перед migration, изменяющей существующую schema, сохраните согласованную
  резервную копию PostgreSQL и связанных model artifacts.

Процедура создания и восстановления backup принадлежит deployment-среде и не
автоматизируется Transformer. Не рассматривайте `rollback` как замену backup.

## Проверить revisions

```bash
./.venv/bin/python ./app/main.py db migrations status
```

`status` выполняет только чтение и выводит:

```text
Current revision: <revision-or-none>
Head revision: <revision>
Pending migrations: <yes-or-no>
```

`Pending migrations: no` означает, что набор current revisions совпадает с
heads текущего checkout. `Current revision: none` означает, что schema ещё не
создана либо не содержит применённой Alembic revision.

## Применить migrations

Сначала выполните `status`, затем примените все pending revisions:

```bash
./.venv/bin/python ./app/main.py db migrations apply
```

`apply` выполняет upgrade до текущего Alembic head и после завершения печатает
тот же status. Успешный результат должен содержать `Pending migrations: no`.
Повторный `apply` при актуальной schema не меняет её.

Revision `0020` является baseline. Revision `0021` добавляет persisted fit
initialization, а текущий head `0022` удаляет дублирующий путь model metadata:

- новая пустая database сначала создаётся baseline, затем получает последующие
  revisions;
- database на revision `0020` или `0021` имеет прямой поддерживаемый upgrade до
  `0022`;
- перед применением `0022` Flight service должен быть остановлен: предыдущий
  executable ещё записывает удаляемую колонку при публикации модели;
- revisions ниже `0020` текущим checkout не поддерживаются.

Для legacy database ниже `0020` сначала разверните tag `0.1.15`, примените его
полную migration chain до `0020` и только затем переходите на текущую версию.
То же правило действует для старых backups. Прежние revisions и data-cutover
instructions сохранены в tag `0.1.15`, Git history и release notes.

## Rollback boundary

Команда rollback запрашивает downgrade последней применённой revision:

```bash
./.venv/bin/python ./app/main.py db migrations rollback
```

Downgrade `0022 → 0021` восстанавливает обязательный `metadata_path` по
`modelRef`, но не создаёт удалённые filesystem sidecars. Downgrade
`0021 → 0020` допустим только при отсутствии fit jobs с
`publishedModel` initialization; иначе migration останавливается без изменения
schema. Он удаляет persisted initialization у остальных jobs. Baseline `0020`
необратима: rollback на ней завершается ошибкой и не удаляет schema или данные.

Не выполняйте rollback как пробный или штатный способ проверки production
schema. Для восстановления после неуспешного изменения используйте заранее
подготовленный backup и процедуру deployment-среды.

## Проверить результат

После `apply` повторно выполните `db migrations status` и убедитесь, что
current и head revisions совпадают, а `Pending migrations` имеет значение
`no`. Только после этого запускайте Flight service или административные
команды, требующие актуальной schema.
