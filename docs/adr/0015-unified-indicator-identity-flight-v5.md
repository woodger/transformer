# ADR 0015: единая identity индикаторов и Flight v5

- Статус: принято
- Дата: 2026-08-20
- Заменяет semantic contract ADR 0007; lifecycle и durable streaming ADR 0005
  сохраняются

## Контекст

Inventory использовал две строковые формы одной identity индикатора: имя
TypeScript enum member и отдельное camelCase wire-значение. Особенно заметно
расхождение на аббревиатурах: `ProbTp` и `probTP`. Такая модель требовала
параллельных mappings и позволяла одному target иметь разные имена в objective,
checkpoint, telemetry и consumer code.

Одновременно `dataContractSha256` скрывал профиль данных. По одному digest
нельзя было понять, к какому Inventory profile относится модель, поэтому
Consumer пытался восстанавливать profile по косвенным признакам.

Изменение observable semantic identity является breaking contract change и не
может быть внесено под номером Flight v4.

## Решение

Transformer публикует только Flight v5:

```text
protocolVersions: [5]
actions:           transformer.v5.*
descriptor path:  transformer/v5/jobs/...
```

V4 compatibility surface, aliases и fallback отсутствуют. Numeric ordinal
Inventory enum используется только для dispatch внутри одного Node.js process
и никогда не передаётся, не сохраняется и не включается в digest.

### Target identity

`inventory.target.v2` однозначно задаёт ширину, порядок и регистрозависимые
имена:

| Индекс | Имя | Диапазон |
| --- | --- | --- |
| `0` | `MeanReturn` | `[-1, 1]` |
| `1` | `SigmaReturn` | `[0, 1]` |
| `2` | `ProbTP` | `[0, 1]` |
| `3` | `ProbSL` | `[0, 1]` |
| `4` | `VolatilityNext` | `[0, 1]` |
| `5` | `HittingProbTP` | `[0, 1]` |

`targetIndex` остаётся координатой tensor. Старые camelCase формы не являются
aliases и отклоняются contract validation. Python identifiers внутри worker-а
могут оставаться `snake_case`, поскольку не пересекают contract boundary.

Текущие semantic IDs:

```text
data contract        inventory.learning-dataset / version 2
target schema        inventory.target.v2
prediction schema    transformer.prediction.target-aligned.v2
objective            transformer.objective.target-aligned.v2
checkpoint           transformer-checkpoint-v4
training recovery    transformer-training-recovery-v4
worker process       v7
epoch metrics        transformer.training-metrics.v3
OpenSearch point     inventory.metrics.point.v3
outbox projection    inventory.metrics.v4
```

Физические Arrow layout identifiers
`inventory.sequence.fit.v2` и `inventory.sequence.predict.v2` сохраняются:
они описывают только неизменившиеся columns, types и widths, а не target
semantic identity. Prediction output получает новый semantic schema ID.

Objective configuration использует только PascalCase semantic names и
`schemaVersion: 2`. Она канонизируется по RFC 8785/JCS; нормативный digest
fixture равен
`ae695d62d643a636d5daaef31275c83eb8baa75e354e70368061c1997b39c2fb`.

### Data profile

Каждый fit и predict create передаёт полный Inventory-owned contract:

```json
{
  "id": "inventory.learning-dataset",
  "version": 2,
  "profile": "research-dividend-events-v2",
  "dataContractSha256": "<sha256>",
  "seqLen": 10,
  "featureDim": 1242,
  "targetSchemaId": "inventory.target.v2"
}
```

`profile` — обязательная ограниченная непустая строка. Transformer не
интерпретирует, не перечисляет и не нормализует её. Поле входит в canonical
документ Inventory и покрывается его digest. Transformer сохраняет весь
`dataContract` в PostgreSQL, command manifest, checkpoint, recovery и metadata
модели; status и `model.describe` возвращают его без преобразования. Predict и
recovery требуют точного совпадения всего документа.

Capabilities не перечисляет profiles. `model.describe` не возвращает
дублирующий массив targets: `targetSchemaId` и `targetWidth` являются
единственными нормативными указателями на порядок target.

### Telemetry

Target-bound epoch telemetry использует структурную identity:

```json
{"target":{"index":2,"name":"ProbTP"},"mae":0.01,"rmse":0.02}
```

Direct losses имеют такую же пару `target.index`/`target.name`. OpenSearch
points используют общие metric names `training.loss.direct`,
`training.target.mae` и `training.target.rmse`; имя target не кодируется в
metric name. Это повышает metrics contract до v3, но не добавляет
`dataContract.profile` в telemetry: такое расширение требует отдельной
эксплуатационной задачи.

### Cutover durable state

Alembic revision `0011` применяется только после штатного удаления всех
экспериментальных моделей текущей v4 командой и завершения их lifecycle в
состоянии `DELETED`, до обновления runtime на v5.
Migration сохраняет API access tokens и tombstones удалённых model generations,
но удаляет несовместимые v4 jobs, attempts, recovery, inputs/outputs,
idempotency results, aliases, run-owned telemetry и outbox. Downgrade не
поддерживается.

Следующая revision `0012` и ADR 0016 удаляют сохранённые model tombstones;
это не меняет предусловие cutover revision `0011`.

После cutover требуется новый fit и новый `modelRef`. Checkpoint v3 и recovery
v3 не сертифицируются и завершаются стабильной ошибкой несовместимости.

## Последствия

- У каждого индикатора существует одна public semantic identity без mappings
  регистра.
- Inventory может выбирать и проверять ML profile по явному полю модели.
- Flight lifecycle, fencing, pagination, durable streaming и Arrow layout не
  меняются.
- Inventory v5 и Transformer v5 переключаются атомарно; смешанная пара версий
  не поддерживается.
- Текущие модели и незавершённые jobs нельзя использовать после cutover.
