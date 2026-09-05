# Контракт процесса Transformer worker v12

> CONTRACT DOCUMENT. Этот каталог задаёт будущий внутренний process contract
> consumer-neutral runtime. Действующий service продолжает запускать worker
> v11 до отдельного implementation change.

Worker v12 получает уже validated `dataContract`, полный `modelContract`, D1
digests и `jobConfigSha256`. Он повторно валидирует semantic documents и
geometry, но не интерпретирует target identities. Public Flight device `gpu`
разрешается service-ом; worker manifest по-прежнему содержит конкретный
внутренний device kind `cpu` или `cuda`.

Запуск остаётся короткоживущим и shell-free:

```text
transformer-worker run --contract-version=12 \
  --job-id=<uuid> --attempt=<positive-integer> \
  --attempt-id=<uuid> --manifest=<service-controlled-path>
```

`inspect --contract-version=12` возвращает `capabilities.schema.json`.
Command manifest, bounded NDJSON control/events и immutable Arrow/result
manifests определены закрытыми schemas. Worker не обращается к PostgreSQL и не
публикует model generation.

## Semantic и recovery fences

Fit manifest содержит resolved training/diagnostics/initialization contracts.
Для `publishedModel` service дополнительно передаёт verified parent checkpoint
artifact. Это weights-only initialization; optimizer, AMP scaler, RNG,
progress и selection создаются заново. Predict получает immutable model
checkpoint.

Recovery descriptor содержит checkpoint artifact, все D1 layers,
`jobConfigSha256`, input `manifestSha256`, input revision и global-epoch
progress. Worker проверяет их до loading state. Format identities:

```text
transformer-worker protocol 12
transformer-checkpoint-v6
transformer-recovery-v6
```

## Inputs, output и telemetry

`indexedFeatureBlocks` и physical Arrow schema IDs не меняются. Worker
memory-map-ит artifacts и reconstruct-ит bounded dense slices; full dense
dataset не materialize-ится. Target width выводится из ordered slots, а
constraint validation выполняется generic.

Prediction output содержит только public transformed coordinates в slot order.
Private resources никогда не входят в Arrow output. Training metrics ссылаются
на component identity и для direct observations сохраняют opaque target
identity плюс derived physical index. Dynamic field names из identities
запрещены.

Worker telemetry arrays разрешаются против exact ModelContract: direct losses
следуют slot order, target identity/index образуют одну пару, auxiliary и
gradient component identities существуют в Objective. Все numerical values
finite; нарушение validated plan является process protocol violation.
