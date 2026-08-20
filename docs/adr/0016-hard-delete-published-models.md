# ADR 0016: безвозвратное удаление опубликованных моделей

- Статус: принято
- Дата: 2026-08-20
- Заменяет хранение tombstone из ADR 0010

## Контекст

ADR 0010 сохранял строку каждой удалённой модели в состоянии `DELETED`. Это
гарантировало монотонный номер generation, но превращало административный
каталог моделей в историю уже несуществующих объектов. `models list` со
временем накапливал tombstones, хотя checkpoint, metadata и каталог модели
были безвозвратно удалены.

Run-owned telemetry и terminal jobs имеют собственные lifecycle и retention.
Они не являются частью опубликованной модели и не должны неявно менять свою
политику хранения при удалении model generation.

## Решение

Сохраняется безопасная двухфазная граница:

```text
AVAILABLE → DELETING → строка отсутствует
```

Команда `models delete MODEL_REF` в PostgreSQL transaction блокирует строку,
проверяет отсутствие активных predict jobs, снимает текущий alias и переводит
модель в `DELETING`. Новые predict и `model.describe` после commit не видят
модель.

Maintenance удаляет server-owned каталог `models/{modelRef}`. Только после
успешного удаления каталога он физически удаляет строку `models`; отсутствие
каталога также считается успешным результатом. Ошибка filesystem оставляет
`DELETING` для следующей попытки. Foreign key модели удаляет любой случайно
оставшийся alias каскадно.

Состояние `DELETED`, поле `deleted_at` и model tombstone удаляются. После
завершения maintenance модель отсутствует в `models list`, повторная команда
получает `model generation not found`, а PostgreSQL не хранит её identity или
metadata.

Revision `0012` удаляет существующие строки `DELETED`, убирает `deleted_at` и
сужает check constraint до `AVAILABLE | DELETING`. Migration необратима,
поскольку восстановить удалённые model metadata невозможно.

## Generation

`modelRef` остаётся единственной долговечной identity существующей модели и
при каждой публикации создаётся заново. Generation остаётся порядковым номером
среди сохранённых generations одного owner/label. После hard delete номер
может быть использован новой моделью повторно; исторической гарантии
монотонности после удаления больше нет.

Consumer не должен сохранять ссылку на удалённую модель или использовать
`owner + label + generation` как глобальную identity. Для корреляции run
telemetry используется точный `modelRef`.

## Граница удаления

Hard delete удаляет только принадлежащие модели данные:

- каталог с checkpoint и metadata;
- строку `models`;
- alias, указывающий на модель.

Исторические terminal jobs и run-owned telemetry не являются model-owned
хвостами. Они продолжают очищаться по собственным retention policies.
OpenSearch documents также не удаляются командой модели.

## Последствия

- `models list` содержит только существующие и ожидающие удаления модели;
- PostgreSQL не накапливает историю удалённых model generations;
- активный predict по-прежнему блокирует удаление;
- filesystem failure остаётся безопасно восстанавливаемым;
- повторный delete после завершения больше не идемпотентен на уровне CLI и
  возвращает `not found`;
- generation нельзя использовать как историческую identity после удаления.
