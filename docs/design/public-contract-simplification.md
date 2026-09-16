# Упрощение публичной границы Inventory — Transformer

> Тип: Design Note. Transformer-side предложение пересмотреть shared contract
> как единую границу, а не как набор локальных переименований. Это не ADR, не
> нормативная schema, не назначение следующей версии и не описание уже
> реализованного поведения.

- Статус: согласовано для подготовки canonical package; clean cut требует
  отдельного решения по существующим generations
- Срез: 2026-09-15, техническая исходная точка Flight v14, Semantic v2,
  Model Catalog Query v2 и Training Telemetry Query v2
- Область: будущая публичная граница Inventory — Transformer
- Действующие contracts, runtime, migrations и deployment не меняются

## Вывод

Текущая граница строго защищает target semantics, recovery и model
compatibility, но materialize-ит больше сведений, чем требуется Consumer-у для
выражения intent. В одном request смешаны:

- Consumer-owned постановка обучения;
- tensor geometry и physical input layout;
- Transformer-owned model implementation;
- derived значения и повторные consistency assertions;
- operational fencing;
- metadata, нужная только checkpoint, recovery или observability.

Из-за этого Inventory вынужден знать provider literals, повторять geometry и
переносить wire-типы в собственные configuration/domain models. Пример с
обязательной архитектурой
`{"identity":"transformer.sequence-model","revision":1}` — только один
симптом этой проблемы.

Предпочтительное направление:

- Consumer передаёт только свои semantic choices, training intent и данные,
  необходимые для построения logical tensor;
- Transformer однократно разрешает implementation, defaults и operational
  metadata;
- вычисляемые значения не передаются как обязательные assertions;
- immutable definition не повторяется в каждом status и telemetry item;
- Worker, checkpoint, recovery и metrics versions не публикуются как часть
  Consumer capabilities;
- typed objective остаётся явным: упрощение не возвращает скрытые
  target-to-operator mappings или server-side presets.

Это breaking пересмотр canonical documents. Удалять отдельные поля из
действующего contract до согласования всей модели не предлагается.

## Проблема

Текущий contract проектировался вокруг защиты от нескольких классов ошибок:

- несовпадение data, target, objective и model semantics;
- неявное сопоставление Consumer target с Transformer operator;
- перепутанный target order;
- повреждённый checkpoint или recovery state;
- повторная либо пропущенная часть durable upload;
- stale Consumer после takeover;
- утечка чужих jobs и models.

Эти свойства нужны и дальше. Избыточность возникла не в самих проверках, а в
том, где лежат их входные данные. Сейчас публичный документ нередко содержит
поле только потому, что Transformer хочет сохранить его в другом внутреннем
документе либо проверить собственную константу.

Упрощение не измеряется числом удалённых JSON keys. Поле следует сохранять,
если оно выражает независимый выбор Consumer-а или защищает реальную границу.
Поле следует вывести из request, если оно:

1. имеет ровно одно допустимое значение;
2. однозначно вычисляется из других значений;
3. принадлежит внутренней реализации Transformer;
4. не позволяет Consumer-у принять решение;
5. повторяет immutable document только ради удобства projection.

## Scope

### Входит в review

- fit/predict create и durable job lifecycle Flight;
- Consumer-owned data identity и tensor geometry;
- `indexedFeatureBlocks` JSON declaration, но не Arrow columns;
- TargetContract, Objective и model configuration;
- requested initialization;
- public capabilities и structured errors;
- Model Catalog list/detail projection;
- Training Telemetry report и gradient-interaction projection;
- D1 и граница между semantic и physical digests;
- public visibility Worker/checkpoint/recovery/metrics identities.

### Не входит в review

- изменение loss formulas, transformations или gradient semantics;
- изменение фактической model architecture и training algorithm;
- изменение Arrow columns, dtypes, offsets, chunking либо reconstruction;
- ослабление authentication, owner scope, idempotency, fencing или recovery;
- OpenSearch storage, PostgreSQL schema и filesystem layout;
- Consumer-owned ML Profile и target catalog;
- локальный CLI;
- переименование `source` в `weightsSource`: requested initialization имеет
  реальный выбор `random`/`publishedModel`, а вопрос имени уже вынесен за рамки
  этой работы.

Worker, checkpoint, recovery и metrics contracts рассматриваются только как
внутренние зависимости. Их точная будущая форма не должна определять public
request.

## Проверенные источники

Текущее состояние сопоставлено с:

- [Semantic model v2](../../app/contracts/semantic/v2/README.md);
- [Flight v14](../../app/contracts/flight/v14/README.md);
- [Worker v13](../../app/contracts/worker/v13/README.md);
- [checkpoint/recovery v7](../../app/contracts/checkpoint/v7/README.md);
- [Model Catalog Query v2](../../app/contracts/model_catalog/v2/README.md);
- [Training Telemetry Query v2](../../app/contracts/training_telemetry/v2/README.md);
- [Consumer integration guide](../consumer-flight-integration.md).

Нормативные schemas перечисленных packages имеют приоритет над пересказом
текущей формы в этом Design Note.

## Критерии поля публичного contract

Для каждого поля применяются пять вопросов.

| Вопрос | Если ответ «нет» |
| --- | --- |
| Consumer владеет значением или осознанно его выбирает? | Поле принадлежит provider resolution либо response metadata |
| Значение способно независимо меняться между двумя корректными requests? | Константа фиксируется revision-ом, а не повторяется в каждом document |
| Consumer может действовать на основании значения? | Поле не должно находиться в capabilities |
| Значение нельзя вывести без потери информации? | Transformer вычисляет его и при необходимости сохраняет внутри |
| Поле защищает отдельную compatibility или security axis? | Оно не должно становиться самостоятельным digest/fence |

Versioned schema сама является контекстом. Фиксированное значение не
становится более строгим от того, что каждый Consumer обязан повторить его в
body.

## Карта действующей границы

```text
Inventory
  data identity/profile/revision/digest
  tensor geometry
  feature-block geometry
  target slots
  objective graph
  model and training parameters
  request/job/execution identities
             |
             v
Flight v14 public documents
  provider constants and internal format versions
  repeated geometry and immutable contracts
  operational fences and runtime limits
             |
             v
Transformer
  semantic validation and D1
  model implementation
  durable jobs and inputs
  Worker/checkpoint/recovery
  registry and telemetry projections
```

Предлагаемая граница:

```text
Inventory intent
  data digest
  tensor geometry and ordered input layout
  targets and explicit objective
  selectable model/training parameters
  requested initialization
             |
             v
Minimal public contract
  exact semantic documents
  durable operation identities
  owner-scoped references
             |
             v
Transformer resolution
  model implementation and internal revisions
  derived geometry, indices and digests
  Worker/checkpoint/recovery/metrics
  compact public status, catalog and telemetry projections
```

## Инвентаризация избыточности

### Semantic и geometry

| Текущая форма | Наблюдение | Предпочтительное направление |
| --- | --- | --- |
| `modelConfig.architecture` | Transformer принимает ровно одну собственную пару identity/revision | Удалить из Consumer input; реализацию связывает revision model definition и внутренний checkpoint fence |
| `modelArchitectures` capabilities | Consumer фактически не выбирает архитектуру | Не публиковать каталог из одного элемента |
| `objectiveLanguageRevision` внутри каждого ModelContract | Flight action принимает одну semantic revision | Version-bound parsing; revision остаётся domain separator digest-а, но не обязательным business field |
| `aggregation=WeightedSum` | Других aggregation нет | Зафиксировать semantics revision-ом |
| `reduction=GlobalRowMean` | Других reduction нет | Зафиксировать semantics revision-ом |
| `parameters={}` | Parameterless operator вынуждает передать пустой объект | Поле существует только у operator с параметрами |
| `resources=[]`, `auxiliaryComponents=[]` | Пустые arrays не выражают выбор | В новой canonical form отсутствие означает нормативно пустое множество |
| target role одновременно задаёт role и `view` | Typed role уже определяет `Observed<T>`, `LossEstimate<T>` либо `PublicPrediction<T>` | Binding хранит только identity; namespace/view задаёт role schema |
| `seqLen/featureDim` в data и model contracts | Одна geometry повторяется и сравнивается | Один `TensorGeometry`, используемый data binding и model digest |
| `sourceEncoding.encoding=indexedFeatureBlocks` | В revision нет другого encoding | Version-bound input layout без discriminator из одного значения |
| `featureBlocks.position` | Runtime требует точную накопительную сумму предыдущих blocks | Вычислять позицию из ordered `windowRows * nativeRowWidth` |
| `dataContract.identity/revision/profile` | Transformer хранит opaque labels, но compatibility задаёт Consumer digest | Передавать digest и geometry; readable metadata остаётся в Inventory |
| `modelConfig.mode` | Имя не сообщает semantics `strict`/`relaxed` missing values | Заменить предметной policy, описывающей mask/indicator behavior |

### Operational workflow

| Текущая форма | Наблюдение | Предпочтительное направление |
| --- | --- | --- |
| `contract` + `version/revision` в каждом body | Version уже находится в exact action identity и schema | Action выбирает parser; body не повторяет dispatch constants |
| Один `job.create` с `operation=fit/predict` | Две команды имеют разные обязательные поля и результаты | Рассмотреть отдельные fit-create и predict-create commands |
| Predict повторяет полный ModelContract | Exact `modelRef` уже разрешает checkpoint-owned definition | Predict передаёт `modelRef`, current data digest и input layout; Transformer разрешает model definition |
| Predict принимает alias | Catalog и persisted selection уже используют exact `modelRef` | Удалить alias; Inventory подтверждает отсутствие такого use case |
| Consumer выбирает `predictionColumn` | Это transport presentation, а не model semantics | Фиксированное output field; локальное имя выбирает Inventory adapter |
| Consumer передаёт `jobId` | Idempotency key уже позволяет повторить потерянный create | Рассмотреть server-issued job reference |
| `clientExecutionId` и `fencingToken` | Вместе представляют один mutation lease | Рассмотреть один opaque provider-issued lease token |
| Full definition в create result и каждом status | Immutable payload многократно передаётся при polling | Create возвращает resolved definition один раз; status содержит только mutable state |
| `dataContractSha256` и schema identity в каждом upload metadata | Они уже immutable для job | Привязывать upload к job/lease; сервер использует сохранённую definition |
| Много физических totals в input close | Большая часть вычисляется по durable receipts | Consumer объявляет только сведения, необходимые для доказательства EOF; provider вычисляет physical summary |

### Capabilities и query projections

| Текущая форма | Наблюдение | Предпочтительное направление |
| --- | --- | --- |
| Worker/checkpoint/recovery/metrics versions | Consumer не выбирает и не вызывает эти contracts | Удалить из public capabilities |
| PyArrow/PyTorch versions | Это deployment diagnostics | Оставить health/admin diagnostics, не Consumer protocol |
| Static schema IDs и feature booleans | Они уже определены versioned action contract | Не повторять в capabilities |
| Full Model Catalog/Telemetry capability documents внутри Flight capabilities | Action identity и query package уже задают semantics | Advertise availability и только runtime-variable limits |
| Exact primitive catalogs | Deployment реализует один принятый package | Предпочтительно closed language per revision; отказаться от hybrid addition без нового package |
| Queue capacities | Это текущая нагрузочная конфигурация, не contract feature | Health/availability, если Consumer действительно использует |
| Telemetry metadata повторяется в каждой epoch | Target/component/operator layout immutable для report | Один report layout плюс compact numerical observations |
| `targetIndex`, operator и identities повторяются в ordered arrays | Значения выводятся из report layout и позиции | Проверять внутри Transformer, наружу передавать один раз |
| `coverage.firstEpoch=1` и `lastEpoch=completedEpochs` | Два поля являются инвариантами revision | Достаточно `completedEpochs` |

## Предпочтительная semantic model

Ниже показана логическая форма, а не предложение точных JSON field names.

### Data binding и tensor geometry

Fit требует ровно одну declaration, содержащую:

- Consumer-owned exact data digest;
- sequence length;
- feature width;
- ordered feature-block dimensions.

Readable data identity, revision и profile остаются у Inventory. Они могут
показываться Terminal-ом по локальному target/data catalog, но Transformer не
должен возвращать их обратно как источник истины.

В частности, readable `configurationLabel` — включая различие между profile и
direct configuration — не входит в canonical data preimage. Если итоговые
Consumer-owned data semantics и geometry одинаковы, способ их локальной
materialization не должен менять `dataContractSha256`. Inventory хранит label
для observability и собственного configuration lookup вне provider boundary.

Ordered feature block содержит только:

```json
{
  "windowRows": 100,
  "nativeRowWidth": 648
}
```

Позиция первого block равна нулю, позиция следующего вычисляется как сумма
размеров предыдущих blocks, а feature width — как сумма всех block sizes.
Transformer продолжает проверять вычисленную width против tensor geometry.

Physical Arrow schema, `nativeRows`, `observationOffsets`, range/example order
и reconstruction formula не меняются. Design Note обещает сохранение logical
tensor semantics; требование byte-identical IPC не вводится без отдельной
проверки конкретного canonical package.

### Targets

Сохраняются:

- ordered opaque identity;
- optional finite closed range observed `y`;
- transformation raw coordinate в direct-loss estimate;
- transformation raw coordinate в public prediction.

Finite является общим Arrow-boundary invariant. Поэтому unbounded finite slot
не обязан передавать отдельный `Finite` wrapper. Closed range остаётся явным,
поскольку меняет validation и compatibility.

Обе transformations сохраняются даже при `Identity`: они выражают независимую
model semantics и не должны выводиться из operator либо target identity.

### Direct objective

Каждый target по-прежнему имеет ровно один direct component. Минимальная
declaration содержит:

- stable component identity;
- target identity;
- operator;
- weight;
- parameters только для parameterized operator.

Для текущих direct operators typed role schema однозначно означает один target
и его observed/loss-estimate views. Повторять два reference objects не нужно.
Direct operator остаётся явным в boundary document и никогда не выводится
Transformer-ом из target identity. Это не запрещает Inventory локально выбрать
operator по своей policy при materialization profile: к границе всё равно
поступает полный explicit component, а не только target identity.

Иллюстрация:

```json
{
  "identity": "direct.event-probability",
  "targetIdentity": "EventProbability",
  "operator": "BinaryCrossEntropyWithLogits",
  "weight": 1
}
```

### Auxiliary objective и resources

Auxiliary component сохраняет stable identity, operator, weight, typed role
bindings и непустые parameters. Значение binding является opaque identity;
operator role определяет, относится identity к target или private resource и
какое target representation используется.

```json
{
  "identity": "aux.risk-adjusted",
  "operator": "RiskAdjustedExpectedValue",
  "weight": 1,
  "bindings": {
    "positiveOutcomeProbability": "PositiveOutcome",
    "negativeOutcomeProbability": "NegativeOutcome",
    "uncertaintyScale": "sharedScale"
  },
  "parameters": {
    "riskPenalty": 0.1
  }
}
```

Abstract resource declaration сохраняется. Она выражает
checkpoint-owned private differentiable output и sharing, а не только shape.
Удалять declaration можно только если все стороны согласятся, что lifecycle
полностью и однозначно задаётся typed consumer roles. Одна строковая ссылка на
resource identity по-прежнему означает sharing.

`WeightedSum`, `GlobalRowMean`, operator constants и execution order задаются
semantic revision. Они не повторяются в каждом Objective.

### Model parameters

Consumer выбирает только поддержанные tuning parameters. Tensor geometry
хранится отдельно и входит в model compatibility, но не копируется в model
configuration.

Текущие generic/PyTorch-oriented имена следует рассмотреть предметно:

| Текущее имя | Выражаемый смысл |
| --- | --- |
| `hidden` | hidden representation width |
| `layers` | encoder layer count |
| `nhead` | attention head count |
| `dropout` | training/model dropout probability |
| `mode` | missing-token handling and optional missingness indicators |

Точная vocabulary требует отдельного review. Цель — назвать доступный tuning
parameter, а не раскрывать PyTorch class или внутренний state layout.

Provider implementation, positional encoding, feed-forward multiplier, output
head construction и private tensor layout не передаются Consumer-ом. В том
числе Inventory не хранит Transformer-owned architecture literal в direct
defaults. Их семантика связана с revision model definition. Несовместимое
изменение требует новой revision и checkpoint fence, а не молчаливой замены
реализации под тем же contract.

### Training policy и initialization

Training policy остаётся отдельной от Objective и model digest. Публичные
поля должны называть training intent (`learningRate`, batch size, epochs,
mixed precision, reproducibility и selection), а не внутренний optimizer
object.

Requested initialization остаётся отдельным реальным выбором:

- random initialization;
- exact published parent model.

Три documents не взаимозаменяемы, даже если случайные variants имеют
одинаковые JSON fields:

- requested initialization в fit intent: `source=random` либо
  `source=publishedModel` с exact `modelRef`;
- resolved initialization в checkpoint/recovery: `source=random` либо
  `source=publishedModel` с `parentModelRef` и
  `parentCheckpointSha256`;
- catalog initialization summary: `source=random` либо
  `source=publishedModel` с owner-visible `parentModelRef`.

Resolved lineage, parent checkpoint digest и parent/current semantic layers
вычисляет Transformer. Consumer не materialize-ит resolved form, а Catalog не
раскрывает checkpoint digest.

## Предпочтительный job workflow

### Fit create

Fit intent содержит:

- correlation и idempotency identity;
- model label;
- requested device class;
- data binding и tensor/input geometry;
- targets, objective и model tuning parameters;
- training/diagnostics policy;
- requested initialization.

Transformer валидирует definition, вычисляет semantic digests, materialize-ит
operational defaults и возвращает server job reference и mutation lease.

### Predict create

Predict intent содержит:

- exact immutable `modelRef`;
- current Consumer data digest;
- ordered input-block layout;
- requested device class.

Target layout, objective, model parameters, tensor geometry и public output
representation разрешаются из registry/checkpoint metadata выбранной model
generation. Consumer не пересылает полный ModelContract обратно Transformer-у.
Mismatch data digest либо input layout отклоняется до upload.

До первого payload predict-create result возвращает минимальную
checkpoint-owned prediction definition:

- `seqLen`;
- ordered target identities и transformation каждой public prediction;
- `outputWidth`.

Этого достаточно Inventory для построения sequence payload и декодирования
output. Полный TargetContract, Objective и model configuration в predict
request/result для этого не нужны.

`modelAlias` и custom prediction column удаляются при clean cut. Inventory
может разрешить human label/current selection и переименовать локальную column
до public boundary; persisted execution всегда использует exact `modelRef`.

### Job identity и fencing

Минимально необходимы три разные семантики:

- request correlation;
- idempotent mutation replay;
- exclusive mutation lease после create/takeover.

Предпочтительная модель:

- `requestId` остаётся correlation identity одной RPC attempt;
- `idempotencyKey` остаётся stable identity логической mutation;
- `jobId` выдаёт Transformer и повторно возвращает при replay create;
- один opaque lease token заменяет пару `clientExecutionId` + decimal fencing
  token;
- takeover атомарно выдаёт новый lease и делает старый недействительным.

Если Consumer-у нужен заранее созданный `jobId` для собственной durability,
Inventory должен показать конкретный сценарий, который не покрывает durable
idempotency key.

### Upload, close и receipts

Сохраняются durable payload receipts, resumability, exact-payload replay,
continuous logical order и возможность начать fit до EOF.

Upload не должен повторять immutable job data digest или Arrow schema identity,
если они однозначно определены созданным job и operation. Consumer передаёт
job, lease, payload ordinal/identity и данные, необходимые для раннего bounded
admission. Transformer вычисляет фактические counts, SHA-256 и schema
fingerprint.

Close обязан доказать, что Consumer и Transformer согласны с полным набором
payload receipts. `manifestSha256` может остаться таким доказательством.
Physical totals, однозначно вычисляемые из receipts, не обязаны одновременно
быть обязательными входами close. Какие expected logical totals действительно
нужны для обнаружения преждевременного EOF, решается минимально: manifest
digest покрывает receipts, а отдельно передаётся только independently computed
expected logical count, если он нужен для этого detection.

### Status и results

Две оси `input` и `execution` сохраняются: streaming fit может обучаться при
открытом input. Status содержит только mutable projection:

- job reference и server revision;
- input/execution states;
- progress;
- terminal error либо result;
- polling hint;
- сведения, необходимые для lease/takeover.

Full data/model definition, source layout, resolved training configuration и
semantic digests не повторяются при каждом poll. Они возвращаются resolved
create result и сохраняются Consumer-ом вместе с исходным idempotent request.
Inventory подтверждает, что отдельный provider-side immutable job detail не
нужен: сохранённого immutable create result достаточно.

## Model Catalog

List/detail разделение в целом соответствует публичной задаче и не требует
радикального пересмотра.

В list остаются сведения, которые Inventory уже использует для выбора и
сравнения:

- exact `modelRef`, label, generation и creation time;
- Consumer data digest и provider-issued model digest;
- selectable model parameters;
- ordered target identities;
- initialization/parent summary;
- producing run identity.

Checkpoint format, byte count и SHA-256 не входят в Catalog projection:
Inventory не использует их для решений, а Transformer проверяет их как
provider-internal integrity metadata. Target/objective digests остаются
доступны в detail для diagnostics, но не обязательны в list.

Detail возвращает semantic definition, resolved training policy, selection,
progress и lineage. Provider-internal `jobConfigSha256` не должен быть public,
если Inventory не принимает на его основании решения.

Catalog не возвращает повторно Consumer profile/target descriptions: Inventory
разрешает их по собственным identities.

## Training Telemetry

Семантика epoch-pass observations, availability outcomes, owner lookup,
snapshot cursor и sparse gradient pairs сохраняется.

Текущая форма повторяет invariant metadata в каждой epoch:

- component identity и operator;
- target identity и physical index;
- ordered target/component layout.

Предпочтительная projection содержит один report layout, проверенный против
model definition, и ordered numerical arrays epochs. Например, report header
один раз фиксирует targets, direct components и auxiliary components, а epoch
передаёт только values в этом порядке. Inventory materialize-ит читаемые rows
в adapter-е.

Transformer продолжает проверять identities, order и cardinality до public
projection. Упрощение wire не ослабляет telemetry integrity.

Direct component identity сохраняется один раз в Objective/report layout, но
не повторяется в каждой epoch observation.

`completedEpochs` достаточно для coverage `1..N`; `firstEpoch=1` и
`lastEpoch=N` выводятся. Health projection должна содержать только независимые
counters. Derived applied/finite totals могут вычисляться после provider-side
validation.

Milestone anchors сохраняются, поскольку позволяют построить initial UI без
полного traversal. Gradient pairs остаются отдельным lazy query.

## Capabilities

Capabilities должны описывать только deployment facts, которые действительно
могут отличаться при одной contract revision и меняют поведение Consumer-а.

Предпочтительно оставить:

- доступные device classes;
- effective upload/job quotas;
- runtime availability optional query surfaces, если `ListActions`
  недостаточно;
- runtime-variable pagination limits, только если они не фиксированы contract.

Предлагается убрать:

- Worker protocol version;
- checkpoint/recovery/metrics formats;
- PyTorch и PyArrow versions;
- единственную model architecture;
- fixed Arrow schema IDs;
- static feature booleans и outcome descriptions;
- action names, уже полученные через `ListActions`;
- queue implementation details;
- primitive catalogs, если language становится closed per revision.

Предпочтительна закрытая semantic language revision: новый operator или
resource lifecycle выпускается новым versioned package. Hybrid additive
capabilities не используются: все корректные deployments одной revision имеют
один exact closed language package.

Consumer проверяет наличие обязательных actions, но не требует, чтобы
`ListActions` был равен закрытому списку. Добавление независимого read-only
action не должно ломать fit/predict старого Consumer-а.

## Structured errors

Сохраняются machine-readable `code`, предметный `reason`, path и typed details.
Unknown/foreign/deleted resources остаются security-equivalent.

Упрощение касается packaging:

- общий error envelope используется всеми public actions;
- action package задаёт допустимые reasons и typed details;
- capabilities не дублирует полный список outcomes;
- human-readable message не участвует в branching.

Не предлагается сворачивать invalid input, capability unavailable,
compatibility mismatch, corruption и temporary backend failure в один общий
код.

## Digests и compatibility

Сохраняется разделение смысловых и физических identities.

### Semantic identities

- Consumer data digest — exact identity Consumer-owned data semantics;
- target digest — ordered target declarations;
- objective digest — operators, weights, resources и bindings;
- model digest — opaque identity, выпускаемая Transformer-ом после resolution
  tensor geometry, selectable model parameters, target/objective declarations
  и domain-separated Transformer-owned model-definition revision.

Architecture literal и другие provider constants не входят в Consumer
document. Transformer не может молча изменить model mathematics либо
implementation semantics под той же model-definition revision.

Inventory вычисляет и передаёт только принадлежащий ему data digest. После
validation и provider resolution Transformer выпускает model digest; Inventory
не вычисляет и не подтверждает его preimage самостоятельно. Target/objective
digests могут оставаться detail-level diagnostic identities, но не являются
обязательным input либо list-level compatibility assertion.

### Operational и physical identities

- job configuration hash;
- input manifest digest;
- checkpoint format/digest;
- recovery fences;
- telemetry projection format.

Эти identities остаются внутри Transformer, кроме тех read-only значений,
которые подтверждённо нужны Catalog UI. Consumer не обязан возвращать
Transformer-computed hash вместе с исходным full document.

Новая canonical form создаст новые target/objective/model digests. Не
предлагаются representation-independent hashing, digest aliases или скрытая
нормализация старой и новой формы в один preimage.

## Внутренние contracts

Worker, checkpoint, recovery и metrics должны точно сохранять resolved job и
обеспечивать integrity. Это не делает их версиями публичного Consumer API.

Public capabilities не обязана сообщать Worker protocol либо OpenSearch
projection format. Service отвечает за совместимость собственных процессов и
storage до объявления готовности Flight endpoint.

Model implementation revision, binary state layout и resolved initialization
могут находиться в checkpoint metadata. Они не становятся полями fit request
только потому, что checkpoint обязан их сохранить.

## Рассмотренные стратегии

### S1. Локально удалить несколько constants

Удалить `architecture`, `aggregation`, `reduction` и `encoding`, сохранив
остальную форму.

Плюсы: небольшой implementation diff. Минусы: geometry, status, capabilities и
query duplication остаются. Проблема ownership решается частично.

### S2. Перерезать публичную границу целиком

Отделить intent, derived values, provider metadata и mutable projections по
правилам этого документа.

Плюсы: простая устойчивая модель и меньше cross-project coupling. Минусы:
breaking change во всех adapters и canonical digests.

Это предпочтительная стратегия.

### S3. Consumer передаёт только profile/preset identity

Transformer сам разрешает targets, objective, architecture и training policy.

Отклоняется: возвращает hidden server-side presets, требует знания Inventory
profiles и ломает consumer-neutral boundary.

### S4. Оставить public contract без изменений

Технически допустимо: текущая форма строгая и работоспособная. Цена —
сохранение лишнего materialization и дальнейшее распространение provider wire
types в Inventory.

## Переход

Design Note не назначает номера будущих revisions. Если стратегия S2 будет
принята, изменение затронет semantic documents, job workflow, checkpoint
metadata и query projections. Менять их in place нельзя.

Варианты:

1. clean cut с удалением существующих jobs/generations и переобучением;
2. read-only retention старого Catalog без predict/warm-start;
3. временный translation layer.

Предпочтителен clean cut, поскольку translation layer сохранит именно ту
двойную модель, от которой выполняется упрощение. Но судьба существующих
generations определяется отдельным решением после проверки реально
существующих generations; из исходного кода это решение не выводится.

Arrow input/prediction formats могут сохранить текущие physical identities,
если canonical package подтвердит неизменность schema и reconstruction. JSON
изменения сами по себе не требуют менять Arrow buffers.

## Сценарии проверки вариантов

Будущая модель должна пройти следующие сценарии без provider-specific данных в
Inventory domain:

1. Новый fit с одним regression target и random initialization.
2. Probability target с raw logit direct loss и public probability.
3. Несколько targets, explicit auxiliary bindings и shared private resource.
4. Target reorder меняет target/model identity и prediction order.
5. Новый opaque target на существующем operator не требует Transformer change.
6. Predict exact modelRef не требует повторной передачи Objective.
7. Published-model warm start проверяет data/target/objective/model exactness.
8. Lost create/DoPut/close response восстанавливается идемпотентно.
9. Takeover делает старый mutation lease недействительным.
10. Streaming fit продолжает работать до input close и после recovery.
11. Catalog list/detail остаётся owner-scoped и пригодным для сравнения.
12. Telemetry report декодируется по единому layout без повторения identities в
    каждой epoch.
13. Provider implementation либо checkpoint format меняется без добавления
    provider literal в Consumer request.

## Согласованные позиции Inventory

| Область | Решение |
| --- | --- |
| Readable data metadata | Profile/revision/configuration label остаются у Inventory и не входят в semantic digest. |
| Job identity | Transformer выдаёт `jobId`; Inventory хранит независимый idempotency key до create. |
| Mutation fencing | Один opaque mutation lease заменяет `clientExecutionId` и fencing token. |
| Predict reference/output | `modelAlias` и custom prediction column удаляются; используется exact `modelRef` и fixed transport output. |
| Input close | Receipts покрываются manifest digest; отдельно допускается только expected logical count для early-EOF detection. |
| Job read | Отдельный job detail не нужен: Consumer сохраняет immutable create result. |
| Semantic language | Каждая revision публикует exact closed package без additive primitive capabilities. |
| Telemetry layout | Direct component identity сохраняется в Objective/report layout; epoch передаёт compact numerical arrays. |
| Private resources | Explicit resource declaration сохраняется как описание lifecycle и sharing. |
| Checkpoint metadata | Checkpoint format, bytes и SHA-256 не нужны Inventory для решений и остаются provider-internal. |
| Catalog digests | В list достаточно data/model digests; target/objective digests доступны в detail для diagnostics. |
| Clean cut | Требует отдельного решения после проверки существующих jobs и model generations. |

Эти решения не меняют действующие contracts. Они задают scope будущего
canonical package и clean-cut implementation.

## Критерии согласования

Упрощённая граница считается состоятельной, если одновременно выполняются
следующие свойства:

- Inventory materialize-ит полную математическую постановку без target-name
  branching Transformer-а;
- Consumer не передаёт Transformer-owned implementation identities и internal
  format versions;
- geometry объявляется один раз, а derived positions/widths проверяются
  Transformer-ом;
- predict использует checkpoint-owned model definition по exact `modelRef`;
- objective bindings остаются явными и typed;
- status, Catalog и Telemetry не дублируют invariant metadata без
  Consumer-facing причины;
- compatibility mismatch, corruption, idempotency, fencing и recovery не
  ослабляются;
- `indexedFeatureBlocks` восстанавливает тот же logical tensor;
- нет executable Consumer code, hidden provider presets и compatibility
  guessing;
- новый contract можно объяснить через ownership, а не через историю старых
  wire fields.

## Следующий этап

1. Обе стороны подтверждают эту уточнённую Design Note как архитектурную
   основу S2.
2. Transformer готовит schema-neutral форму минимальных fit, predict, status,
   Catalog и Telemetry documents, включая provider-issued model digest.
3. Обе стороны согласуют digest preimages и отдельно принимают transition
   policy по существующим generations.
4. Только затем выпускается staged canonical package со schemas и небольшим
   набором behavioral cross-project fixtures.
5. Runtime, migrations и deployment меняются после точного review package.

Conformance bundle должен доказывать boundary behavior и digest
canonicalization. Отдельный тест на каждое слово словаря не требуется.
