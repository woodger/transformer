# Transformer checkpoint и recovery v11

> ДОКУМЕНТ КОНТРАКТА. Подготовленный внутренний пакет metadata для Flight v20
> и Worker v18. Он не меняет действующий checkpoint/recovery v10.

`transformer-checkpoint-v11` сохраняет resolved runtime diagnostics
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

`encoderLayerDiagnostics` допускает `null` или `directComponentPerBatch`.
Ненулевое значение требует `targetHead: "fullCommittedArtifact"` и включает
сбор learning observations encoder в отдельный Worker v18 artifact.

Configuration является runtime/recovery fence: она входит в
`jobConfigSha256`, но не изменяет ModelContract, Semantic v4,
`modelDefinitionSha256`, D1, warm-start compatibility или public prediction.

`transformer-recovery-v11` не дублирует configuration: её точность
обеспечивается `jobConfigSha256`. Observations diagnostics не встраиваются в
checkpoint, не влияют на checksum и не переписывают ранее опубликованные
checkpoint artifacts.
