# ADR 0003: durable recovery и device-aware execution

- Status: Accepted
- Decision date: 2026-07-24

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Долгий fit не должен полностью теряться при restart сервиса, хоста или
подтверждённой потере GPU. Временный runtime storage и одна логическая CUDA
lane не позволяли восстановить training state и не связывали attempt с
конкретным физическим device.

Training state можно надёжно зафиксировать на границе завершённой global epoch.
Checkpoint внутри произвольного batch потребовал бы существенно более сложной
координации data order, optimizer updates и random state.

## Решение

Committed fit inputs и внутренние recovery checkpoints сохраняются независимо
от временного attempt workspace. PostgreSQL остаётся authority их видимости и
retry lifecycle, а файлы содержат объёмные immutable payloads.

Recovery checkpoint фиксируется только после полной global epoch и включает
состояние model, optimizer, scaler, training progress, selection и random
generators. Прерванная незавершённая epoch повторяется целиком. Отсутствующий,
повреждённый или несовместимый зарегистрированный checkpoint вызывает явную
recovery error, а не скрытый restart с нуля.

Каждая CUDA attempt получает lease одного обнаруженного физического device и
запускается с привязкой к нему. Подтверждённо потерянный device quarantined до
следующей загрузки host; running attempt не переносится между devices. Retry
разрешён для service/host interruption и подтверждённой потери device, но не
для произвольной ошибки обучения.

## Рассмотренные альтернативы

- Хранить fit целиком во временном runtime storage. Отклонено из-за потери
  многочасовой работы при restart.
- Сохранять checkpoint внутри epoch или batch. Отложено из-за сложности точного
  восстановления optimizer, shuffle и RNG state.
- Хранить Arrow payloads и checkpoint blobs в PostgreSQL. Отклонено: БД владеет
  visibility и lifecycle, но не bulk artifacts.
- Автоматически повторять любую ошибку либо переключать CUDA job на CPU.
  Отклонено, потому что это скрывает deterministic failure и меняет execution
  semantics.

## Последствия

- После interruption теряется не больше незавершённой global epoch.
- Persistent recovery storage требует capacity monitoring и retention.
- Resume сохраняет логическое training state, но не обещает bitwise equality
  nondeterministic CUDA kernels.
- Device loss создаёт новую attempt с новой identity, а не переносит владение
  существующего process.

## Текущая документация

- [Training runtime и recovery](../training-runtime.md)
- [Операционное руководство Flight](../operations/flight-service.md)
- [Архитектурная политика](../policy/architecture.md)
