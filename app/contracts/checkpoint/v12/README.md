# Transformer checkpoint и recovery v12

> ДОКУМЕНТ КОНТРАКТА. Внутренний пакет metadata для активных Flight v22 и
> Worker v20.

`transformer-checkpoint-v12` сохраняет resolved runtime diagnostics
configuration вместе с `jobConfigSha256`, input manifest, semantic identities,
progress и selection:

```json
{
  "schemaVersion": 3,
  "gradientInteractions": null,
  "targetHead": "fullCommittedArtifact",
  "encoderLayerDiagnostics": "directComponentPerBatch"
}
```

Resolved `modelConfig` обязательно содержит
`encoderNormalizationOrder: "postNorm" | "preNorm"`. Это immutable
архитектурное значение, участвующее в provider-issued `modelDefinitionSha256`;
checkpoint/recovery не выводит его из числа encoder layers и не применяет
неявный default.

`encoderLayerDiagnostics` допускает `null` или `directComponentPerBatch`.
Ненулевое значение требует `targetHead: "fullCommittedArtifact"` и включает
сбор learning observations encoder в отдельный Worker v20 artifact.

Configuration является runtime/recovery fence: она входит в
`jobConfigSha256`, но не изменяет ModelContract, Semantic v5,
`modelDefinitionSha256`, D1, warm-start compatibility или public prediction.

`transformer-recovery-v12` не дублирует configuration: её точность
обеспечивается `jobConfigSha256`. Observations diagnostics не встраиваются в
checkpoint, не влияют на checksum и не переписывают ранее опубликованные
checkpoint artifacts.
