# Предметный словарь shared Transformer contracts без `kind`

> Тип: Design Note. Transformer-side предложение новой формы shared canonical
> documents и public capabilities для совместного review с Consumer-ами. Это
> не ADR, не нормативная JSON Schema, не назначение production migration и не
> описание уже реализованного поведения.

- Статус: архитектурно согласовано; normative staged packages подготовлены к
  точному cross-project review
- Срез: 2026-09-13, техническая исходная точка Flight v13, Objective Language
  v1, Worker v12, checkpoint/recovery v6, Model Catalog Query v1, Training
  Telemetry Query v1 и metrics v5
- Входной материал: Consumer-side запрос Inventory о предметном словаре без
  универсального `kind`
- Область изменения: будущие shared canonical documents и public capabilities;
  действующие contracts, schemas, runtime, migrations и deployment не меняются

## Вывод

Универсальный discriminator `kind` не создаёт машинной неоднозначности внутри
закрытого tagged union и сам по себе не является дефектом. Проблема проявляется
на уровне языка долгоживущих межпроектных документов: одно имя сейчас выбирает
mathematical primitive, namespace ссылки, lifecycle private resource,
initialization source, physical encoding, transport, consistency model, device
backend и measurement point. Это ухудшает читаемость и облегчает перенос wire
формы в Consumer domain model.

Предлагается новая revision со следующими свойствами:

- в shared canonical documents и public capabilities отсутствует свойство с
  точным именем `kind`;
- вместо него не появляется универсальный discriminator `type`;
- каждое различие вариантов выражается предметным полем либо scalar primitive
  в уже однозначно именованном enclosing field;
- закрытость объектов, обязательность полей, namespace references и отсутствие
  неявных defaults сохраняются;
- изменение выполняется clean cut без v1 readers, aliases или digest
  equivalence;
- математика, tensor runtime и Arrow data plane не меняются.

Цена изменения существенна: новая JSON representation меняет canonical bytes,
D1 и все зависимые immutable artifacts. Польза оправдывает эту цену только как
один согласованный переход всего shared surface, а не как серия локальных
rename-ов.

## Граница scope

### Входят в изменение

- `TargetContract`, `Objective`, `ResourceDeclaration`, `ModelContract` и D1
  semantic envelope;
- Flight job documents, source encoding, initialization и capabilities;
- только зависимые Worker, checkpoint, recovery и metrics carrier documents,
  поскольку они встраивают либо фиксируют shared semantic model;
- checkpoint/recovery metadata;
- Model Catalog list/detail и capability documents;
- Training Telemetry query capability и report identity documents;
- training metrics и terminal fit-run records, которые фиксируют semantic и
  checkpoint identities;
- versioned OpenSearch templates для этих metrics formats;
- normative schemas, README, golden fixtures и fixture manifests перечисленных
  packages.

### Не входят в изменение

- внутренние Python dictionaries, cache records, ORM models и log events,
  которые не пересекают versioned contract boundary;
- local CLI и local Arrow framed protocol;
- Consumer-owned ML Profile, target catalog и Inventory domain types;
- ML formulas, operators, training defaults, architecture и tensor layout;
- naming cleanup других generic unions только ради единообразия;
- PostgreSQL schema и конкретная destructive migration до принятия canonical
  packages.

Требование относится к свойству `kind` в versioned documents. Оно не запрещает
слово `kind` в prose или внутреннем implementation и не требует переименовывать
уже существующие несвязанные поля `type`.

## Инвентаризация действующего `kind`

| Область | Нормативное назначение в текущей revision |
| --- | --- |
| Semantic v1 | `Finite`/`ClosedInterval` constraint |
| Semantic v1 | `Identity`/`Tanh`/`Sigmoid` transformation |
| Semantic v1 | target и resource reference namespace |
| Semantic v1 | `PositiveScalarPerObservation` resource lifecycle |
| Semantic v1 | `WeightedSum` aggregation и `GlobalRowMean` reduction |
| Semantic v1 | resolved random/published-model initialization |
| Flight v13 | `indexedFeatureBlocks` source encoding |
| Flight v13 | requested random/published-model initialization |
| Worker v12 | `cpu`/`cuda` execution device; остальные значения наследуются из semantic/Flight documents |
| Checkpoint v6 | resolved initialization и встроенный ModelContract v1 |
| Model Catalog Query v1 | initialization summary, `ArrowFlightDoAction` transport и `LiveHighWater` consistency |
| Training Telemetry Query v1 | `ArrowFlightDoAction` transport и `PreOptimizerUpdateEpochPass` measurement point |
| Fit-run metrics v5 | resolved initialization и его OpenSearch projection |

Training metrics points v5 не содержат собственного `kind`, но ссылаются на D1
Semantic v1 и `transformer-checkpoint-v6`. Model Catalog detail также возвращает
полный ModelContract v1, поэтому проблема не ограничивается capability
documents.

Внутренние snapshot selectors `report`/`gradientInteractions`, local-fit
descriptors и runtime dispatch dictionaries нормативными shared fields не
являются и остаются за границей предложения.

## Правила нового словаря

### Предметный discriminator

Поле, выбирающее вариант, должно одновременно называть смысл выбора. Например,
`constraint` выбирает числовое ограничение, `source` — источник initialization,
а `protocol` — transport. Имя не переиспользуется как универсальный механизм
во всех документах.

### Scalar primitives

Parameterless primitive представляется scalar string, когда enclosing field
однозначно задаёт его категорию. Поэтому transformation, aggregation и
reduction не требуют одноэлементного object wrapper.

Появление parameterized transformation не меняет старую scalar semantics in
place. Оно требует новой Objective Language revision, которая может добавить
предметную object form.

### Closed unions

Объекты остаются закрытыми. В reference union ровно одно namespace field:
`targetIdentity` или `resourceIdentity`. Mixed reference, неизвестное поле,
отсутствующий target `view` и старая `kind`-форма отклоняются до semantic
lookup.

Выбор operator, transformation или resource class никогда не выводится из
Consumer-owned target, component или resource identity.

## Предлагаемая форма

| Понятие | Форма новой revision |
| --- | --- |
| Finite constraint | `{"constraint":"Finite"}` |
| Closed interval | `{"constraint":"ClosedInterval","minimum":0,"maximum":1}` |
| Loss transformation | `"lossInputTransformation":"Identity"` |
| Public transformation | `"publicPredictionTransformation":"Sigmoid"` |
| Aggregation | `"aggregation":"WeightedSum"` |
| Reduction | `"reduction":"GlobalRowMean"` |
| Target reference | `{"targetIdentity":"probability","view":"observed"}` |
| Resource reference | `{"resourceIdentity":"scale"}` |
| Resource declaration | `{"identity":"scale","resourceClass":"PositiveScalarPerObservation"}` |
| Requested initialization | `{"source":"publishedModel","modelRef":"mdl_..."}` |
| Resolved initialization | `{"source":"publishedModel","parentModelRef":"mdl_...","parentCheckpointSha256":"...",...}` |
| Catalog initialization summary | `{"source":"publishedModel","parentModelRef":"mdl_..."}` |
| Physical input encoding | `{"encoding":"indexedFeatureBlocks",...}` |
| Transport capability | `{"protocol":"ArrowFlightDoAction",...}` |
| Catalog consistency | `{"consistencyModel":"LiveHighWater",...}` |
| Epoch observation | `{"measurementPoint":"PreOptimizerUpdateEpochPass",...}` |
| Worker device | `{"backend":"cuda","opaqueId":"..."}` |

`resourceClass` намеренно используется вместо `valueShape`.
`PositiveScalarPerObservation` задаёт не только shape и numerical domain, но и
private differentiable model output, принадлежащий checkpoint. Transformer
по-прежнему скрывает PyTorch layer, positive parameterization, tensor layout,
storage и batching implementation.

Для capabilities список `resourceKinds` предлагается переименовать в
`resourceClasses`. Остальные primitive catalogs сохраняют предметные имена:
`constraints`, `transformations`, `directOperators`, `auxiliaryOperators`,
`aggregations` и `reductions`.

## Три initialization documents

Requested intent, resolved lineage и catalog projection являются разными
документами. Они не используют один общий union только потому, что имеют поле
`source`. Каждая форма получает отдельную schema identity, closed-object
validation и собственное место применения.

### `RequestedInitialization`

Consumer передаёт только способ создания fit model:

```json
{
  "source": "random"
}
```

либо exact parent selector:

```json
{
  "source": "publishedModel",
  "modelRef": "mdl_0123456789abcdef0123456789abcdef"
}
```

Документ принадлежит Flight v14 create request. `modelRef` ещё является
запрошенной ссылкой: Consumer не передаёт checkpoint digest, parent/current D1
или результат provider lookup. Service разрешает reference внутри
authenticated owner scope и не сохраняет requested form как authoritative
lineage.

### `ResolvedInitialization`

После owner-scoped lookup, проверки parent metadata и compatibility Transformer
materialize-ит immutable lineage:

```json
{
  "source": "publishedModel",
  "parentModelRef": "mdl_0123456789abcdef0123456789abcdef",
  "parentCheckpointSha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "parentDataContractSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "dataContractSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "parentTargetContractSha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "targetContractSha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "parentObjectiveSha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "objectiveSha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "parentModelContractSha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "modelContractSha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
}
```

Random initialization имеет отдельный closed variant:

```json
{
  "source": "random"
}
```

Resolved document входит в resolved `jobConfigSha256`, Worker command,
checkpoint metadata, fit-run record, Flight status/result и Model Catalog
detail. Он является источником exact initialization lineage. Для strict
published-model warm start parent/current digests обязаны удовлетворять
действующей compatibility policy; наличие обеих сторон делает эту проверку и
диагностику явной.

Общее определение `ResolvedInitialization` публикуется в shared definitions
checkpoint/model-generation v7 и повторно используется Flight, Worker,
fit-run metrics и Model Catalog detail. Оно описывает checkpoint-owned lineage,
а не Objective Language, поэтому не входит в `TargetContract`, `Objective`,
`ModelContract` или D1 preimages и не принадлежит Semantic v2.

### `CatalogInitializationSummary`

Bounded list projection содержит только сведения, необходимые для отображения
lineage без дублирования checkpoint metadata:

```json
{
  "source": "publishedModel",
  "parentModelRef": "mdl_0123456789abcdef0123456789abcdef"
}
```

Для random generation:

```json
{
  "source": "random"
}
```

Summary принадлежит Model Catalog Query v2. Оно не принимается как input, не
участвует в D1, job/recovery fencing или warm-start validation и не заменяет
resolved lineage в detail. Model Catalog detail возвращает одновременно
summary как часть model summary и полный `ResolvedInitialization`; Transformer
проверяет, что summary является точной проекцией resolved document.

Несмотря на одинаковые bytes random variants, `RequestedInitialization`,
`ResolvedInitialization` и `CatalogInitializationSummary` остаются тремя
разными contract types. Их schemas не ссылаются друг на друга и могут
эволюционировать только в версии владеющего package.

## Пример semantic documents

Один target slot:

```json
{
  "identity": "ConsumerDefined.EventProbability",
  "observedConstraint": {
    "constraint": "ClosedInterval",
    "minimum": 0,
    "maximum": 1
  },
  "lossInputTransformation": "Identity",
  "publicPredictionTransformation": "Sigmoid"
}
```

Direct component:

```json
{
  "identity": "direct.event-probability",
  "operator": "BinaryCrossEntropyWithLogits",
  "weight": 1,
  "roles": {
    "logit": {
      "targetIdentity": "ConsumerDefined.EventProbability",
      "view": "lossEstimate"
    },
    "probability": {
      "targetIdentity": "ConsumerDefined.EventProbability",
      "view": "observed"
    }
  },
  "parameters": {}
}
```

Shared private resource и использующий его auxiliary component:

```json
{
  "resources": [
    {
      "identity": "sharedScale",
      "resourceClass": "PositiveScalarPerObservation"
    }
  ],
  "auxiliaryComponents": [
    {
      "identity": "aux.location-nll",
      "operator": "GaussianNLL",
      "weight": 0.1,
      "roles": {
        "locationEstimate": {
          "targetIdentity": "Location",
          "view": "lossEstimate"
        },
        "observedLocation": {
          "targetIdentity": "Location",
          "view": "observed"
        },
        "scale": {
          "resourceIdentity": "sharedScale"
        }
      },
      "parameters": {}
    }
  ]
}
```

Точные enclosing field names auxiliary operators определит Semantic v2
canonical package. Пример фиксирует vocabulary и namespace distinction, но не
заменяет будущую JSON Schema.

## Неизменная semantic model

Objective Language v2 меняет canonical representation, но сохраняет:

- ordered opaque target slots и derivation physical target index из позиции;
- constraints `Finite` и `ClosedInterval` с прежней числовой семантикой;
- transformations `Identity`, `Tanh` и `Sigmoid`;
- direct operators `SmoothL1`, `BinaryCrossEntropyWithLogits`, `LogMSE`;
- auxiliary operators `GaussianNLL`, `ExpectedValue` и
  `RiskAdjustedExpectedValue`;
- operator formulas, constants, typed roles, component weights и execution
  order;
- `WeightedSum` и `GlobalRowMean`;
- resource sharing по exact identity и structural gradient reachability;
- stop-gradient semantics конкретных operator roles;
- отсутствие behavioral dispatch по Consumer-owned identities;
- derivation public target width из ordered slots;
- current model architecture semantics и validation limits.

Semantic v2 не является расширением primitive catalog. Новая revision нужна
из-за type system и canonicalization, а не из-за новой математики.

## Матрица versions

| Слой | Текущая версия | Предлагаемая версия | Причина изменения |
| --- | --- | --- | --- |
| Semantic / Objective Language | v1 | v2 | Новая canonical form, reference schemas и D1 domains |
| Arrow Flight workflow | v13 | v14 | Новые job documents, capabilities и закрытый action surface query v2 |
| Worker process contract | v12 | v13 | Semantic v2, source encoding, initialization и device vocabulary |
| Checkpoint | v6 | v7 | Встроенные ModelContract v2, initialization и D1 |
| Recovery | v6 | v7 | Exact fencing нового checkpoint/job/semantic contract без v6 resume |
| Model Catalog Query | v1 | v2 | Detail возвращает ModelContract v2; summary initialization и capabilities меняются |
| Training Telemetry Query | v1 | v2 | Capability vocabulary и semantic digest generation меняются; observations сохраняют смысл |
| Training metrics | v5 | v6 | Новые semantic/checkpoint identities и format fence |
| Terminal fit-run metrics | v5 | v6 | Новые initialization, semantic/checkpoint identities и OpenSearch mapping |
| Prediction Arrow | target-aligned v3 | без изменения | Координаты и ordered target decoding не меняются |
| Fit/Predict Arrow input | indexed-feature-blocks v1 | без изменения | Physical columns, buffers и reconstruction не меняются |

Предлагаемые query actions:

```text
transformer.model-catalog.v2.list
transformer.model-catalog.v2.detail
transformer.training-telemetry.v2.report
transformer.training-telemetry.v2.gradient-interactions
```

Flight v14 рекламирует только новые actions. Flight v13 aliases и fallback не
добавляются. Независимый lifecycle query revisions сохраняется: совпадение их
первой clean-cut активации во Flight v14 не связывает последующие revisions с
каждым выпуском workflow.

## Canonicalization и D1

Semantic v2 сохраняет RFC 8785/JCS, UTF-8 и SHA-256. Preimages имеют прежнюю
слоистую структуру, но содержат revision `2` и новые documents:

```text
targetContractSha256 = SHA256(JCS({
  objectiveLanguageRevision: 2,
  targetContract
}))

objectiveSha256 = SHA256(JCS({
  objectiveLanguageRevision: 2,
  objective
}))

modelContractSha256 = SHA256(JCS({
  objectiveLanguageRevision: 2,
  modelConfig,
  targetContractSha256,
  objectiveSha256
}))
```

Новая representation меняет:

- `targetContractSha256`;
- `objectiveSha256`;
- `modelContractSha256`;
- `jobConfigSha256`;
- physical `checkpointSha256` из-за новой embedded metadata;
- package и fixture manifest digests.

`dataContractSha256` остаётся Consumer-owned и не меняется по требованию
Transformer. Inventory может отдельно изменить свой canonical data document,
но это не является частью данного proposal.

Input `manifestSha256` не обязан меняться, если Arrow schemas, payload bytes,
receipts и Consumer data digest совпадают. Prediction artifact digest также не
меняется только из-за vocabulary.

### Representation-independent hashing не вводится

Transformer не нормализует v1 и v2 в скрытый общий AST для вычисления одного
digest. Математически эквивалентные v1 и v2 documents имеют разные canonical
bytes и разные D1. Не вводятся translation tables, digest aliases или
compatibility exceptions для warm start/predict.

Manifest SHA-256 доказывает целостность versioned package и перечисленных
fixtures. Он не является дополнительным model compatibility layer и не
участвует в runtime dispatch, если это прямо не определено отдельным
операционным contract.

## Clean-cut transition

Старые generation нельзя переписать in place. Перевод ModelContract v1 в новую
JSON form изменит D1, checkpoint metadata и physical checkpoint SHA-256, то
есть нарушит immutable model identity даже при тех же tensor weights.

Предпочтительный переход:

1. остановить создание новых Flight v13 jobs и дождаться согласованного окна;
2. остановить либо явно завершить все non-terminal jobs до destructive cut;
3. удалить v1 published generations, checkpoint v6, recovery v6 и связанные
   job/input artifacts;
4. удалить прежнюю v5 telemetry, fit-run projections и pending delivery
   records, чтобы старые документы не выглядели данными нового contract;
5. синхронно активировать Transformer Flight v14 и Inventory adapter новой
   версии без compatibility layer;
6. обучить новые model generations с нуля.

Старые models не поддерживаются для predict, warm start, recovery или Model
Catalog detail. Legacy reader и offline weight conversion в production runtime
не создаются.

Если сохранение конкретных weights когда-либо станет обязательным, отдельный
offline import должен создавать новую generation с явно спроектированной
lineage и equivalence proof. Это самостоятельная задача и не входит в
предлагаемый переход.

Точная PostgreSQL migration, порядок удаления filesystem artifacts и
OpenSearch index names определяются только после принятия canonical packages.

## Arrow data plane и `indexedFeatureBlocks`

JSON envelope меняется с `kind` на `encoding`, но physical Arrow contract
остаётся прежним:

- schema identities fit/predict v1 сохраняются;
- названия, порядок и types Arrow fields не меняются;
- feature block order, positions, widths, offsets, halo и continuity rules не
  меняются;
- payload chunking и reconstruction создают тот же logical Float32 tensor;
- prediction schema target-aligned v3 и ordered coordinate decoding не
  меняются.

При одинаковых native values physical Arrow schemas и сериализованные payload
bytes должны побайтово совпасть с Flight v13. Сами JSON fixtures различаются,
поскольку v14 использует новое поле `encoding`; их chunk data и ожидаемый
logical tensor остаются теми же. Отдельный test сравнивает reconstruction для
v13 и v14 envelope вне production runtime. Такой test доказывает неизменность
data plane, но не является compatibility layer.

## Capabilities

Hybrid lifecycle сохраняется:

- immutable `objectiveLanguage.revision=2` задаёт type system,
  canonicalization и semantics существующих primitives;
- capabilities перечисляют реализованные constraints, transformations,
  `resourceClasses`, operators, aggregations, reductions и model
  architectures;
- добавление Consumer target на существующих primitives не требует выпуска
  Transformer;
- новое primitive implementation может быть additive capability только когда
  его роли и lifecycle уже выражены Semantic v2;
- изменение type system, reference model, resource lifecycle или semantics
  существующей primitive требует следующей language revision.

Public capability documents используют `protocol`, `consistencyModel`,
`measurementPoint` и другие предметные поля из таблицы выше. Unknown fields и
старая `kind`-форма отклоняются их closed schemas.

## Canonical package и fixtures

Packages готовятся и проверяются в порядке зависимостей. До принятия всего
набора runtime не меняется.

1. `semantic/v2`: README semantics, closed schemas, language capabilities,
   positive/negative fixtures, literal JCS bytes, D1 golden vectors и manifest.
2. `checkpoint/v7` и recovery v7: shared `ResolvedInitialization`, metadata
   schemas, integrity/fencing fixtures и manifest.
3. `worker/v13`: command/result/capability schemas, recovery references и
   process fixtures.
4. `metrics/v6` и `metrics/fit_run/v6`: records, OpenSearch templates и
   projection fixtures.
5. `model_catalog/v2` и `training_telemetry/v2`: query schemas, capabilities,
   `CatalogInitializationSummary`, pagination/error fixtures и manifests.
6. `flight/v14`: closed action surface, job documents, aggregate capabilities,
   `RequestedInitialization`, unchanged Arrow fixtures и cross-contract
   manifest.

Каждый package содержит собственный manifest с SHA-256 каждого нормативного
файла. Inventory хранит byte-identical offline copy и независимо проверяет
schemas, references, JCS и D1 до реализации.

Минимальный cross-language conformance set:

- scalar aggregation, reduction и transformations без object wrapper;
- оба constraints и rejection старой `kind`-формы;
- однозначные target/resource references, mixed-reference rejection и
  обязательный target `view`;
- `resourceClass` lifecycle и sharing по resource identity;
- отдельные requested, resolved и catalog-summary initialization schemas;
- запрет `modelRef` в resolved/summary lineage, запрет parent/checkpoint/D1
  fields в requested form и запрет полного lineage в catalog summary;
- точная проекция `ResolvedInitialization` в
  `CatalogInitializationSummary`;
- byte-identical Python/TypeScript JCS и D1;
- одинаковая математика v1/v2 при разных canonical bytes и digests;
- digest change при перестановке slots/components или resource binding;
- Model Catalog v2 detail с точным ModelContract v2;
- kind-free Flight/query/Worker capabilities;
- byte-identical Arrow payloads и одинаковый reconstructed logical tensor;
- отсутствие `inventory.*`, известных target literals и фиксированной target
  width в provider schemas.

После точного Consumer review package архитектурное решение фиксируется ADR.
Только затем Transformer и Inventory реализуют согласованный clean cut и
готовят destructive operational procedure.

## Статус canonical package

Staged packages опубликованы в согласованном порядке зависимостей:

- `app/contracts/semantic/v2`;
- `app/contracts/checkpoint/v7` и recovery v7;
- `app/contracts/worker/v13`;
- `app/contracts/metrics/v6` и `app/contracts/metrics/fit_run/v6`;
- `app/contracts/model_catalog/v2` и `app/contracts/training_telemetry/v2`;
- `app/contracts/flight/v14`.

Каждый package содержит closed schemas, cross-project fixtures и собственный
fixture manifest. Fixture bundle даёт Consumer-у материал для независимой
проверки `$ref`, literal JCS/D1, initialization boundaries, query
projections и неизменности Arrow geometry.
Пакеты не подключены к runtime и не меняют migrations либо действующий Flight
v13 surface.

## Критерии готовности к canonical package

Design Note готов перейти в staged contracts, когда обе стороны подтверждают:

- граница scope не превращается в repository-wide naming cleanup;
- словарь `constraint`, `resourceClass`, `source`, `encoding`, `protocol`,
  `consistencyModel`, `measurementPoint` и `backend` однозначен;
- scalar form transformations/aggregation/reduction принимается как часть
  Semantic v2;
- вся version matrix принимается одним clean cut;
- старые generations удаляются, а не мигрируют под прежними identities;
- v1 и v2 не объявляются digest-equivalent;
- Arrow data plane и numerical runtime сохраняют прежнюю семантику.
