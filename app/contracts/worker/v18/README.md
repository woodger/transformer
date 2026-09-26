# Процессный контракт Transformer Worker v18

> ДОКУМЕНТ КОНТРАКТА. Текущий внутренний протокол между сервисом и Worker для
> Flight v20. Он не является публичным API.

Worker v18 получает `diagnostics` v3 из checkpoint/recovery v11. Режим
`encoderLayerDiagnostics: "directComponentPerBatch"` допустим только вместе с
`targetHead: "fullCommittedArtifact"`.

После каждого training batch Worker собирает для каждой группы параметров
encoder градиент именованного прямого компонента до clipping и optimizer step.
После успешного step он отдельно измеряет фактическое изменение групп
параметров. В конце epoch оба набора агрегатов записываются вместе с
существующими observations target head и `representationFlow`.

Измерение component gradient не записывает `.grad`; затем обычный training
path выполняет свой `backward` без изменения порядка операций.

`result-manifest.schema.json` содержит поле `targetHeadDiagnostics`. При
выключенной настройке оно равно `null`; при включённой указывает на закрытый
immutable artifact. Артефакт привязан к `jobId`, attempt, `attemptId`,
`inputRevision`, `manifestSha256`, `modelDefinitionSha256` и
`jobConfigSha256`. Неполный artifact не является доступным report.

Наблюдения не меняют weights, optimizer/scaler state, RNG trajectory или
данные `predict`. Ошибка best-effort сбора не заменяет terminal outcome fit.
Сырые gradients, parameter names, tensors, feature rows и пути артефакта не
пересекают публичную Flight-границу.

```text
transformer-worker protocol 18
transformer-checkpoint-v11
transformer-recovery-v11
transformer-target-head-diagnostics-v3
```

Semantic v4 и Arrow input layout Flight v20 сохраняются без изменения.
