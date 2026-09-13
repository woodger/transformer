# Контракт процесса Transformer worker v13

> CONTRACT DOCUMENT. Этот каталог задаёт process contract предметного словаря
> без `kind`.

Worker v13 переносит validated Semantic v2 documents без изменения математической
семантики. Target identities остаются opaque, physical indices выводятся из
ordered slots, а device выбирается предметным полем `backend`.

Запуск остаётся короткоживущим и shell-free:

```text
transformer-worker run --contract-version=13 \
  --job-id=<uuid> --attempt=<positive-integer> \
  --attempt-id=<uuid> --manifest=<service-controlled-path>
```

`inspect --contract-version=13` возвращает `capabilities.schema.json`. Command,
control/event, Arrow и result manifests остаются closed documents. Существующие
поля `type` у control/event сообщений не являются discriminator-ом shared
semantic vocabulary и сохраняют прежнюю process semantics.

## Semantic и recovery fences

Fit command содержит exact ModelContract v2, D1 digests и checkpoint-owned
`ResolvedInitialization` из checkpoint v7. Requested initialization из Flight
сюда не переносится. Predict получает immutable model checkpoint.

```text
transformer-worker protocol 13
transformer-checkpoint-v7
transformer-recovery-v7
```

Recovery descriptor по-прежнему фиксирует checkpoint artifact, все D1 layers,
`jobConfigSha256`, input `manifestSha256`, input revision и global-epoch progress.
v6 recovery state не является совместимым с v7.

## Неизменный data plane

`indexedFeatureBlocks` использует поле `encoding`, но physical Arrow schema IDs,
columns, offsets и reconstruction semantics остаются прежними. Worker
создаёт тот же logical Float32 tensor и тот же target-aligned prediction v3.
Training observations сохраняют прежнюю numerical semantics и typed references.

## Fixtures

`fixtures/manifest.json` фиксирует SHA-256 всех Worker v13 fixtures. Эти hashes
нужны только для cross-project conformance package и не участвуют в runtime
compatibility или dispatch.
