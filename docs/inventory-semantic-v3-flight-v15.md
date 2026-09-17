# Техническая записка для Inventory: Semantic v3 и Flight v15

> Тип: пояснительная записка. Описывает реализованную границу для команды
> Inventory. Нормативными источниками остаются JSON Schema и README пакетов
> `app/contracts/`.

> Историческая записка Flight v15. Текущая граница определена
> [Flight v16](../app/contracts/flight/v16/README.md); этот файл сохранён для
> контекста первоначального clean cut и не описывает активный action surface.

## Статус развертывания

На 16 сентября 2026 года Transformer развернут с commit
`e9935de68227255907e7797ab0e2d2911c6c394d`. Сервис `transformer.service`
активен. PostgreSQL находится на Alembic revision `0027`; pending migrations
нет.

Revision `0027_public_contract_simplification` уже выполнила destructive clean
cut. Модели, job-ы, recovery state и telemetry предыдущей публичной границы
удалены намеренно. Старые `modelRef` нельзя использовать для predict, warm
start, recovery, catalog или telemetry query. Новые models следует обучать
через Flight v15.

## Активный набор версий

| Область | Версия | Значение для Inventory |
| --- | --- | --- |
| Semantic language | v3 | Материализуемый `ModelContract` и его semantic identities. |
| Arrow Flight workflow | v15 | Единственная публичная workflow revision. |
| Model Catalog Query | v3 | Owner-scoped discovery и detail опубликованных models. |
| Training Telemetry Query | v3 | Read-only report обучения и lazy gradient interactions. |
| Worker | v14 | Внутренний process protocol Transformer. |
| Checkpoint/recovery | v8 | Внутренние artifacts Transformer. |
| Metrics projection | v7 | Внутреннее хранилище telemetry Transformer. |

Inventory интегрируется только с первыми четырьмя строками. Версии Worker,
checkpoint/recovery, OpenSearch indices, artifact paths и внутренние hashes
не являются входом или результатом публичной границы.

## Семантическая граница v3

Inventory materializes полный self-contained `ModelContract`:

```text
ModelContract
├── targetContract
├── objective
└── modelTuning
```

`targetContract` задаёт упорядоченные opaque target identities, observed
constraints, transformation для входа loss и transformation публичного
prediction. Порядок slots является физическим порядком output vector.

`objective` содержит явно выбранные direct/auxiliary operators, weights,
typed roles и объявления private resources. Transformer исполняет только
закрытый набор объявленных primitives; он не выводит operator из target name и
не интерпретирует FIGI, profile, `inventory.*` либо иную предметную identity.

В активных Semantic v3 и Flight v15 документах нет универсального
discriminator-а `kind`. Вместо него используются предметные формы: например,
`constraint`, scalar transformation, `resourceClass`, `source` initialization
и `inputLayout.featureBlocks`. Это изменение формы, а не изменение formulas,
roles, resource lifecycle или порядка targets.

`dataBinding` передаётся отдельно от `ModelContract`. В нём находятся:

- opaque `dataContractSha256`;
- `tensorGeometry`;
- `inputLayout`.

После validation Transformer самостоятельно разрешает внутреннюю definition
модели и выпускает `modelDefinitionSha256`. Это provider-issued opaque
identity: Inventory не вычисляет его preimage и не должен пытаться восстановить
внутреннюю architecture configuration.

`targetContractSha256` и `objectiveSha256` вычисляются по Semantic v3 JCS/D1;
`dataContractSha256` остаётся opaque identity, вычисляемой Inventory. Полный
набор identities и `ModelContract` доступен через Model Catalog detail.

## Flight v15: что должен делать Inventory

Flight v15 является clean cut: v14 actions, aliases и compatibility reader
отсутствуют. Актуальный action surface:

```text
transformer.v15.capabilities
transformer.v15.health
transformer.v15.fit.create
transformer.v15.predict.create
transformer.v15.job.acquire
transformer.v15.job.status
transformer.v15.job.inputs.list
transformer.v15.job.input.close
transformer.v15.job.outputs.list
transformer.v15.job.cancel
transformer.model-catalog.v3.list
transformer.model-catalog.v3.detail
transformer.training-telemetry.v3.report
transformer.training-telemetry.v3.gradient-interactions
```

### Fit

1. Inventory materializes Semantic v3 `ModelContract` и строит `dataBinding`.
2. Вызывает `transformer.v15.fit.create` с UUID `requestId`, устойчивым
   `idempotencyKey`, label, requested device, training/diagnostics settings и
   requested initialization.
3. Сохраняет immutable create result: выданные Transformer `jobId`,
   `mutationLease` и resolved model definition.
4. Загружает compact Arrow payloads и закрывает input через
   `transformer.v15.job.input.close` с `manifestSha256` receipts.
5. Получает status до terminal result.

После любого restart service незавершённый job не возобновляется: status станет
`FAILED / EXECUTION_INTERRUPTED` (либо `CANCELLED`, если отмена уже была
запрошена). Для повторной попытки Inventory создаёт новый fit job и повторно
загружает input.

Вызовы mutation требуют текущий `mutationLease`; `job.acquire` выдаёт новый
lease при необходимости. Inventory не передаёт собственный job ID, execution
ID, fence counter, Worker schema ID, close totals или checkpoint metadata.

`expectedLogicalRows` при close необязателен и нужен только для ранней
диагностики EOF. Фактические receipts, totals и physical schema identity
выводит и сохраняет Transformer.

### Predict

`transformer.v15.predict.create` получает только точный `modelRef`, requested
device и текущий `dataBinding`. Повторно передавать target contract, objective
или model tuning нельзя.

Create result до upload возвращает checkpoint-owned `predictionDefinition`:
sequence length, output width и ordered target identities с public prediction
transformations. Inventory использует именно это описание для построения
sequence payload и декодирования coordinate prediction.

## Данные X и `indexedFeatureBlocks`

`indexedFeatureBlocks` остаётся единственным input layout v15. Его логическая
семантика не изменилась: Transformer восстанавливает blocks, разворачивает
окна и конкатенирует их в logical tensor `[rows, seqLen, featureDim]`.

Inventory передаёт для каждого block только `windowRows` и `nativeRowWidth`.
Positions блоков выводит Transformer. Сумма ширин blocks должна совпадать с
`featureDim`. Flight v15 не обещает byte-identical Arrow IPC serialization;
инвариантом является корректное logical reconstruction и validation geometry.

## Model Catalog и Training Telemetry

Catalog и telemetry — две независимые read-only поверхности, активируемые
Flight v15 actions.

- `transformer.model-catalog.v3.list` и `.detail` работают только в scope
  аутентифицированного owner-а. Unknown, foreign и deleted `modelRef`
  возвращают одинаковый `MODEL_NOT_FOUND`.
- Registry, а не telemetry, определяет существование модели.
- `transformer.training-telemetry.v3.report` возвращает observations прохода
  обучения по epoch, а не повторную оценку итогового checkpoint-а.
- Gradient interactions загружаются отдельным ленивым action только при
  необходимости UI.

Inventory не передаёт owner или producing run ID и не обращается напрямую к
OpenSearch, его index names или document IDs. Ветвление должно идти по
structured `code` и `reason`, а не по тексту error message.

## Практический чек-лист Inventory

1. Переключить adapter на точные schemas и action names Flight v15.
2. Materialize только Semantic v3 documents; не переносить wire types
   Transformer в Inventory domain/profile configuration.
3. Хранить response fit/predict create как immutable контекст сессии.
4. Для predict использовать возвращённый `predictionDefinition`, а не
   локально предполагать target order или public transformation.
5. Для discovery использовать Model Catalog v3, а для training UI — Training
   Telemetry v3; не использовать metrics storage как public registry.
6. Не пытаться читать, мигрировать или переиспользовать v14/v2 generations:
   они исключены clean cut и должны быть созданы заново.
7. Проверять advertised capabilities перед включением optional query surfaces
   и корректно обрабатывать structured availability/error outcomes.

## Нормативные источники

- [Semantic v3](../app/contracts/semantic/v3/README.md)
- [Flight v15](../app/contracts/flight/v15/README.md)
- [Model Catalog Query v3](../app/contracts/model_catalog/v3/README.md)
- [Training Telemetry Query v3](../app/contracts/training_telemetry/v3/README.md)
- [Практическая интеграция Flight v15](flight-integration.md)

Эта записка не заменяет schemas. Перед изменением adapter-а Inventory должен
проверять точную форму requests, responses, errors и capabilities по активным
пакетам `app/contracts/` и их fixtures.
