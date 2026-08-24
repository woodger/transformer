# ADR 0016: hard delete моделей с минимальным audit archive

- Status: Accepted
- Decision date: 2026-08-20

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Полный model tombstone сохранял checkpoint paths, hashes, ML contracts и
metadata после физического удаления artifacts. Рабочая таблица смешивала
доступные models с историей и неограниченно накапливала metadata
несуществующих generations.

Полное отсутствие deletion record также нежелательно: operator должен видеть
завершение удаления, а allocator не должен повторно использовать generation.
Filesystem и PostgreSQL нельзя изменить одной transaction.

## Решение

Удаление имеет durable двухфазную boundary. Сначала PostgreSQL запрещает новые
predict operations и фиксирует pending deletion. Затем maintenance удаляет
server-owned model artifacts. После успеха отдельная transaction физически
удаляет рабочую model row и создаёт минимальный audit record identity и
timestamps.

Audit archive не содержит checkpoint paths, hashes, ML/data contracts или
payload metadata и не разрешается как model. Allocation generation учитывает
рабочие rows и archive, поэтому номер не используется повторно. Run-owned
telemetry и terminal jobs сохраняют независимые retention lifecycles.

## Рассмотренные альтернативы

- Сохранять полный tombstone в основной model table. Отклонено из-за роста
  metadata и смешения working set с history.
- Не сохранять deletion record. Отклонено из-за потери operator-visible
  completion и риска повторного generation.
- Удалять filesystem и database row одним admin command. Отклонено: crash
  между несогласованными resources нельзя сделать atomic.
- Использовать вечный soft delete без физической очистки. Отклонено, потому что
  server-owned model artifacts должны освобождаться.

## Последствия

- После hard delete модель невозможно восстановить из audit record.
- Filesystem failure оставляет повторяемое pending deletion, а не частичную
  доступную model.
- Working model list не накапливает удалённые metadata.
- Generation остаётся монотонным, а telemetry deletion не получает скрытых
  side effects.

## Текущая документация

- [Управление опубликованными моделями](../operations/published-models.md)
- [Архитектурная политика](../policy/architecture.md)
