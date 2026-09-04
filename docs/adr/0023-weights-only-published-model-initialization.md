# ADR 0023: weights-only initialization from a published model

- Status: Accepted
- Decision date: 2026-09-01

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Flight fit создавал модель только со случайной инициализацией. Для продолжения
экспериментов требуется использовать обученные weights опубликованной
generation, не теряя воспроизводимую identity run и не превращая immutable
model artifact в изменяемое состояние.

Понятие «дообучить модель» неоднозначно: оно может означать загрузку только
weights либо точное продолжение optimizer, scaler, RNG, progress и selection
state. Published checkpoint содержит production model artifact, тогда как
полное состояние незавершённого training run принадлежит отдельному recovery
lifecycle.

## Решение

Flight v8 вводит обязательный закрытый fit `initialization` с режимами
`random` и `publishedModel`. Второй режим принимает только точный owner-scoped
immutable `modelRef`. Transformer разрешает generation и требует полного
совпадения model config, data contract, targets и objective.

Published model используется только как источник weights. Новый fit создаёт
собственные optimizer, AMP scaler, RNG, progress, checkpoint selection и
recovery state, обучает все параметры и при успехе публикует новую immutable
generation с новым `modelRef`. Parent не изменяется.

Resolved lineage содержит parent model reference и checkpoint SHA-256. Он
сохраняется в job, model metadata и checkpoint metadata, но не входит в
objective digest. Удаление parent generation запрещено, пока зависящий от неё
job не завершён.

Public contract переходит на Flight v8, внутренний process contract — на
worker v9. Checkpoint format остаётся v5: ранее опубликованный корректный
checkpoint-v5 без lineage считается обученным с `random`. Параллельный Flight
v7 runtime и aliases не поддерживаются.

## Рассмотренные альтернативы

- Восстанавливать полное training state parent-а. Отклонено: это exact resume,
  который смешивает immutable published model с recovery незавершённого job и
  не позволяет независимо задать новую training policy.
- Изменять parent generation на месте. Отклонено: нарушает immutable model
  identity, воспроизводимость prediction и безопасную ротацию alias.
- Принимать alias, label или filesystem path. Отклонено: target мог бы
  измениться между retry, а внутренний storage path пересёк бы public boundary.
- Разрешить частичную загрузку при отличающихся heads, targets или objective.
  Отклонено: неявное сопоставление параметров создаёт другую семантику transfer
  learning и не даёт точного compatibility contract.

## Последствия

- Consumer может явно выбрать независимое обучение или weights-only warm
  start без изменения Transformer code.
- Warm start остаётся новым reproducible run, а recovery продолжает только
  этот run.
- Parent lineage доступен через create/status, model describe и checkpoint
  metadata.
- Несовместимый parent отклоняется до загрузки input; отсутствующий или чужой
  reference не раскрывается и возвращает `NOT_FOUND`.
- Parent нельзя физически удалить, пока незавершённый dependent job может
  потребовать его artifact при запуске или recovery.

## Текущая документация

- [Контракт Arrow Flight v10](../../app/contracts/flight/v10/README.md)
- [Worker process contract v11](../../app/contracts/worker/v11/README.md)
- [Интеграция Consumer-ов](../consumer-flight-integration.md)
- [Training runtime](../training-runtime.md)
- [Управление опубликованными моделями](../operations/published-models.md)
