# ADR 0010: Штатное удаление опубликованных моделей

## Статус

Принято, 2026-08-16.

## Контекст

Успешный fit публикует immutable generation одновременно в PostgreSQL и
`models/{modelRef}`. Прямое удаление каталога оставляет доступную identity с
отсутствующим checkpoint и превращает ожидаемое административное действие в
`MODEL_UNAVAILABLE`. Удаление строки PostgreSQL, напротив, может переиспользовать
generation, потерять историю и каскадно уничтожить ещё не доставленный metrics
outbox.

Удаление также конкурирует с `job.create` prediction: разрешённый create должен
либо целиком зафиксировать ссылку на доступную модель, либо увидеть её уже
недоступной.

## Решение

Вводится lifecycle:

```text
AVAILABLE → DELETING → DELETED
```

Локальная admin-команда принимает только точный `modelRef`. В одной
PostgreSQL transaction она блокирует строку generation, проверяет отсутствие
незавершённых prediction jobs, снимает alias, если тот всё ещё указывает на
удаляемую generation, и фиксирует `DELETING`. Разрешение модели при predict
использует row lock той же строки и возвращает только `AVAILABLE`, поэтому
create и deletion имеют однозначный порядок commit.

Filesystem не удаляется из краткоживущего admin-процесса. Service maintenance
периодически выбирает `DELETING`, удаляет server-owned каталог и затем отдельной
transaction фиксирует `DELETED`. Отсутствующий каталог считается уже удалённым.
Filesystem error оставляет `DELETING` и повторяется. Startup reconciliation
сохраняет каталоги `AVAILABLE` и `DELETING`, но может удалить остаток уже
зафиксированного `DELETED`.

Строка модели остаётся tombstone со всеми identity fields. Она продолжает
участвовать в `max(generation)`, поэтому ни generation, ни `modelRef` не
переиспользуются. Alias не получает fallback на предыдущую generation.

## Training metrics

Outbox в `PENDING` или `BLOCKED` по умолчанию блокирует удаление. Явный option
`--discard-undelivered-metrics` переводит запись в `CANCELLED`; publisher больше
не принимает её как pending, а финализация удаления каскадно удаляет локальную
artifact metadata и outbox. Уже доставленные OpenSearch documents не удаляются.
Bulk request, начатый до commit отмены, может завершиться; option означает
отказ от гарантии полной projection, а не распределённое удаление документов.

## Граница API

Flight v4 не меняется. Удаление является operator-owned локальной операцией:

```text
models list
models delete MODEL_REF [--discard-undelivered-metrics]
```

Удаление по alias намеренно отсутствует: mutable alias недостаточно точен для
destructive operation.

## Последствия

- активная prediction защищает модель от удаления;
- после `DELETING` новые create и `model.describe` получают `NOT_FOUND`;
- сбой между filesystem deletion и PostgreSQL finalization безопасно
  восстанавливается следующей maintenance attempt;
- tombstones растут по числу опубликованных generations и являются небольшой
  ценой за стабильную identity и monotonic generation;
- удаление не очищает исторические job records и централизованную telemetry.
