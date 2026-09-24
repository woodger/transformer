# Процессный контракт Transformer Worker v16

> ДОКУМЕНТ КОНТРАКТА. Этот staged-пакет определяет внутренний процессный
> протокол между сервисом и Worker для Flight v18. Он не является публичным
> wire contract и не изменяет действующий Worker v15.

Worker v16 получает `diagnostics` v2 из checkpoint/recovery v10. При
`targetHead: "fullCommittedArtifact"` он после каждой завершённой эпохи
собирает observations Target Head Diagnostics v1. Градиенты именованного
прямого компонента измеряются до шага оптимизатора; статистики logit,
публичного prediction, входа головки и параметров головки — в отдельном
post-update `eval()`/`no_grad()` проходе.

`result-manifest.schema.json` содержит обязательное для fit поле
`targetHeadDiagnostics`. При выключенной настройке оно равно `null`; при
включённой содержит закрытый внутренний артефакт
`transformer-target-head-diagnostics-v1`. Worker связывает его с `jobId`,
attempt, закрытым input manifest, model definition и job configuration, а также
с sample, layout и полной последовательностью эпох. `modelRef` ещё не
существует в момент работы Worker: сервис присваивает его при публикации модели
и затем связывает service-side сохранённый артефакт с публичной generation.
Сервис валидирует Worker-артефакт до создания публичной проекции. Все
дублирующиеся fences (`jobId`, attempt, `attemptId`, input revision/manifest,
`jobConfigSha256` и `modelDefinitionSha256`) должны совпасть с result manifest
и checkpoint metadata; расхождение не создаёт public report.

Наблюдательный проход не меняет weights, optimizer/scaler state, RNG trajectory
или данные `predict`. Ошибка best-effort сбора не заменяет terminal outcome
fit. Структура result manifest не означает, что сырые observations или пути
артефакта пересекают публичную Flight-границу.

Внутренние форматы staged revision:

```text
transformer-worker protocol 16
transformer-checkpoint-v10
transformer-recovery-v10
transformer-target-head-diagnostics-v1
```

Semantic v4 и Arrow input layout Flight v18 сохраняются без изменения.
