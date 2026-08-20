# ADR 0016: безвозвратное удаление моделей и архив identity

- Статус: принято
- Дата: 2026-08-20
- Заменяет хранение полного model tombstone из ADR 0010

## Контекст

ADR 0010 оставлял удалённую модель в основной таблице `models` со всеми
checkpoint paths, hashes, ML contracts и metadata. Обычный `models list`
смешивал рабочий каталог с историей удалений и со временем накапливал строки
несуществующих моделей.

Полностью отказаться от deletion record также нежелательно: оператору нужен
явный `deleted_at`, а allocator generation не должен повторно использовать
номер уже опубликованной модели.

Run-owned telemetry и terminal jobs имеют собственные lifecycle и retention.
Они не являются частью опубликованной модели и не должны неявно менять свою
политику хранения при удалении model generation.

## Решение

Сохраняется безопасная двухфазная граница:

```text
models:          AVAILABLE → DELETING → строка отсутствует
deleted_models:                         минимальный audit record
```

Команда `models delete MODEL_REF` в PostgreSQL transaction блокирует строку,
проверяет отсутствие активных predict jobs, снимает текущий alias и переводит
модель в `DELETING`. Новые predict и `model.describe` после commit не видят
модель.

Maintenance удаляет server-owned каталог `models/{modelRef}`. Только после
успешного удаления каталога одна transaction:

1. записывает минимальный audit record в `deleted_models`;
2. физически удаляет рабочую строку `models`.

Отсутствие каталога также считается успешным результатом. Ошибка filesystem
оставляет `DELETING` для следующей попытки. Foreign key модели удаляет любой
случайно оставшийся alias каскадно.

Archive содержит только:

- `modelRef`;
- owner и label;
- generation;
- `created_at`;
- `deletion_requested_at`;
- `deleted_at`.

Checkpoint paths, hashes, model/data/ML contracts, metadata и producing job в
archive не переносятся. Это audit удаления, а не доступная или восстановимая
модель.

## CLI

Обычная команда показывает только строки основной таблицы `models`:

```bash
./.venv/bin/python ./app/main.py models list
```

Архив удалений запрашивается отдельно:

```bash
./.venv/bin/python ./app/main.py models list --deleted
```

Опция `--deleted` является фильтром и показывает только audit records. Колонка
`DELETED AT` заполнена только для архивных записей. После завершения deletion
повторная команда `models delete MODEL_REF` возвращает
`model generation not found`.

## Generation

Публикация вычисляет следующий generation по максимуму основной таблицы и
архива под тем же advisory lock. Поэтому generation одного owner/label остаётся
монотонным и не используется повторно после удаления.

`modelRef` остаётся identity модели, а не архива. Архивная строка не разрешается
через Flight и не может быть использована для predict.

## Миграция

Revision `0012` уже физически удалила прежние строки `DELETED` из `models` и
поле `models.deleted_at`. Их timestamps восстановить невозможно.

Revision `0013` создаёт пустой `deleted_models` для последующих удалений.
Downgrade разрешён только пока archive пуст; после появления первой audit
record migration становится необратимой.

## Граница удаления

Hard delete удаляет принадлежащие модели данные:

- каталог с checkpoint и metadata;
- рабочую строку `models`;
- alias, указывающий на модель.

Исторические terminal jobs и run-owned telemetry продолжают очищаться по
собственным retention policies. OpenSearch documents также не удаляются
командой модели.

## Последствия

- `models list` содержит только существующие и ожидающие удаления модели;
- `models list --deleted` даёт явный минимальный audit с `deleted_at`;
- PostgreSQL не хранит payload или metadata удалённых моделей;
- generation остаётся монотонным;
- активный predict по-прежнему блокирует удаление;
- filesystem failure остаётся безопасно восстанавливаемым;
- восстановить удалённую модель по audit record невозможно.
