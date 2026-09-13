# Контракт checkpoint и recovery v7

> CONTRACT DOCUMENT. Этот каталог задаёт нормативные metadata checkpoint/recovery
> format предметного contract vocabulary для Flight v14 и Worker v13.

`checkpoint-metadata.schema.json` описывает metadata, встроенную в binary
checkpoint `transformer-checkpoint-v7`. Она содержит Semantic v2 ModelContract,
D1, training/diagnostics/selection state и точный `ResolvedInitialization`.
Tensor state, optimizer, AMP scaler и RNG остаются binary implementation data,
но обязаны соответствовать metadata и architecture revision.

## Resolved initialization

Checkpoint/model-generation v7 владеет shared типом
`ResolvedInitialization`. Он отличается от Flight-requested intent и Model
Catalog summary.

Random form:

```json
{"source":"random"}
```

Published-model form содержит `source=publishedModel`, exact
`parentModelRef`, `parentCheckpointSha256` и parent/current пары data, target,
objective и model digests. Все поля обязательны; `modelRef` requested form и
сокращённый catalog summary не принимаются.

Flight v14 result/status, Worker v13 command, fit-run v6 и Model Catalog v2
detail используют это единственное shared definition. Оно входит в resolved
`jobConfigSha256` и checkpoint metadata, но не входит в D1 preimages.

## Physical identity и recovery

Physical `checkpointSha256` не находится внутри хэшируемого checkpoint, чтобы
не создавать самоссылку. Внешний `checkpoint-artifact.schema.json` хранит exact
format, byte count и SHA-256.

`recovery-metadata.schema.json` связывает artifact с job, input revision,
Semantic v2 D1, `jobConfigSha256`, input `manifestSha256` и progress. Recovery
v6 не принимается v7 Worker-ом.

До загрузки tensor state recovery проверяет:

1. exact checkpoint bytes и `transformer-checkpoint-v7` format;
2. structural validity embedded metadata и `ResolvedInitialization`;
3. повторно вычисленные Semantic v2 D1;
4. `jobConfigSha256` и `manifestSha256`;
5. job/generation/progress fences;
6. architecture-owned state layout.

Component/resource identities не разрешают remapping или partial loading.
Несовместимость recovery возвращает `RECOVERY_CHECKPOINT_INCOMPATIBLE`, а
повреждённый stored checkpoint — `MODEL_CORRUPT`.

## Clean cut и fixtures

Checkpoint v6 и recovery v6 не конвертируются и не читаются новым runtime.
Старые generations, recovery state и checkpoints удаляются до активации;
модели обучаются заново.

Fixtures содержат random/published resolved forms, полный published-model
checkpoint metadata, physical artifact и recovery metadata. Manifest фиксирует
exact bytes cross-project bundle.
