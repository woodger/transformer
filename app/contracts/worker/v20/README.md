# Процессный контракт Transformer Worker v20

> ДОКУМЕНТ КОНТРАКТА. Внутренний протокол между сервисом и Worker для
> активного Flight v22; это не публичный API.

Worker v20 получает `diagnostics` v3 из checkpoint/recovery v12. Режим
`encoderLayerDiagnostics: "directComponentPerBatch"` допустим только вместе с
`targetHead: "fullCommittedArtifact"`.

После каждого training batch Worker собирает для каждой группы параметров
encoder градиент именованного прямого компонента до clipping и optimizer step.
После успешного step он отдельно измеряет фактическое изменение групп
параметров. В конце epoch оба набора агрегатов записываются вместе с observations
target head, `representationFlow` и `encoderBlockFlow`. Последний собирается в
отдельном post-update `eval()`/`no_grad()` проходе полного committed artifact
и показывает named boundaries encoder и `OutputHead.shared`. Для
`postNorm` порядок boundaries равен `attentionResidual → norm1 →
feedForwardResidual → norm2`; для `preNorm` — `norm1 → attentionResidual →
norm2 → feedForwardResidual`.

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
transformer-worker protocol 20
transformer-checkpoint-v12
transformer-recovery-v12
transformer-target-head-diagnostics-v5
```

Resolved `ModelConfig` обязательно содержит `encoderNormalizationOrder`.
Значение входит в command/recovery manifest и определяет `norm_first` при
создании encoder; Worker не выводит его из глубины либо параметров задания.
Arrow input layout Flight v22 сохраняется без изменения.
