# Процессный контракт Transformer Worker v22

> ДОКУМЕНТ КОНТРАКТА. Внутренний протокол между сервисом и Worker для
> активного Flight v24; это не публичный API.

Worker v22 получает `diagnostics` v3 из checkpoint/recovery v14. Режим
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
transformer-worker protocol 22
transformer-checkpoint-v14
transformer-recovery-v14
transformer-target-head-diagnostics-v5
```

Resolved `ModelConfig` обязательно содержит `encoderNormalizationOrder`.
Значение входит в command/recovery manifest и определяет `norm_first` при
создании encoder; Worker не выводит его из глубины либо параметров задания.
Arrow input layout Flight v24 сохраняется без изменения.

Worker v22 исполняет Semantic v7 `BernoulliConfidencePenalty` как negative
entropy и `BernoulliEntropyPenalty` как positive entropy публичной вероятности.
Оба оператора не добавляют resource или coordinate,
сохраняет weighted direct-loss checkpoint selection и пишет penalty в
auxiliary losses и gradient interactions. Без декларации objective обучение
не получает дополнительного регуляризатора.
