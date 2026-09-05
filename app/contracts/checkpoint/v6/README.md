# Контракт checkpoint и recovery v6

> CONTRACT DOCUMENT. Этот каталог задаёт normative metadata нового
> consumer-neutral checkpoint/recovery format. Текущий runtime продолжает
> читать и писать checkpoint v5 до отдельного clean-cut implementation change.

`checkpoint-metadata.schema.json` описывает metadata, встроенную в binary
checkpoint `transformer-checkpoint-v6`. Tensor state, optimizer, AMP scaler,
RNG и selection state остаются binary implementation data, но обязаны
соответствовать metadata и architecture revision.

Physical `checkpointSha256` не находится внутри хэшируемого checkpoint, чтобы
не создавать самоссылку. Его вместе с exact byte count содержит внешний
`checkpoint-artifact.schema.json`. `recovery-metadata.schema.json` связывает
этот artifact с job, input manifest и progress.

До загрузки tensor state recovery проверяет в порядке:

1. exact checkpoint bytes и format;
2. structural validity embedded metadata;
3. повторно вычисленные D1 digests;
4. `jobConfigSha256` и `manifestSha256`;
5. job/generation/progress fences;
6. architecture-owned state layout.

Component/resource identities не разрешают remapping или partial loading.
Несовместимость recovery возвращает `RECOVERY_CHECKPOINT_INCOMPATIBLE`, а
повреждённый stored checkpoint — `MODEL_CORRUPT`.
