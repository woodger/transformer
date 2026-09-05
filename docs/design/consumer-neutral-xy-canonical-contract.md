# Canonical contract proposal для consumer-neutral границы `x/y`

> Тип: Design Note. Документ предлагает точные canonical documents и правила
> их интерпретации для совместного согласования с Consumer-ами. Это не ADR, не
> нормативная JSON Schema, не Flight или Worker contract и не описание
> реализованного поведения.

- Статус: согласовано Transformer и Inventory; основа будущего versioned contract
- Срез: 2026-09-05, техническая исходная точка Flight v10 / Worker v11
- Основание: [согласованная schema-neutral semantic model](./consumer-neutral-xy-semantic-model.md)
- Область изменения: будущая consumer-neutral модель; действующие contracts и
  runtime не изменяются

## Назначение

Документ переводит согласованную semantic model в однозначные JSON documents,
но ещё не назначает им Flight action, protocol version, schema ID или
persistence migration. Предлагаемая структура должна быть одинаково
реализуема в Python и TypeScript и не требовать от Transformer знания значений
`MeanReturn`, `ProbTP`, FIGI, interval, profile или других понятий Consumer-а.

Согласованное предложение фиксирует:

- точную структуру `TargetContract`, `Objective`, `ResourceDeclaration` и
  `ModelContract`;
- закрытый начальный mathematical language;
- canonicalization и D1 layered digests;
- hybrid lifecycle language revision и capabilities;
- validation order и стабильные категории ошибок;
- состав cross-project fixtures;
- место `indexedFeatureBlocks` относительно semantic model.

Не входят в предложение:

- номер следующей Flight, Worker, checkpoint, recovery или metrics version;
- имена будущих action и Arrow schema IDs;
- физическая структура PyTorch modules и `state_dict`;
- PostgreSQL columns, migration plan и OpenSearch index names;
- implementation plan и временный compatibility code.

До выпуска нового versioned contract Flight v10 остаётся замороженной
технической исходной точкой. Само принятие proposal не меняет код, нормативные
schemas, migrations или runtime contracts.

## Термины и общие типы

Все объекты ниже закрыты: неизвестное поле является ошибкой. Поле обязательно,
если явно не сказано обратное. Canonical documents не используют `null` для
обозначения отсутствующего значения.

### Идентичности

`TargetIdentity`, `ComponentIdentity` и `ResourceIdentity` имеют одну wire
форму:

```text
^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$
```

Они:

- регистрозависимы;
- сравниваются как exact ASCII strings;
- не нормализуются и не разбираются Transformer-ом;
- уникальны в своём namespace;
- являются contract identities, а не descriptions.

Target, component и resource образуют разные namespaces. Одинаковая строка в
двух разных namespaces не создаёт reference между ними. Ограничение ASCII
устраняет межъязыковую неоднозначность Unicode normalization, не раскрывая
предметный смысл identity.

`Sha256` — строка из 64 lowercase hexadecimal symbols:

```text
^[0-9a-f]{64}$
```

### Числа

Все числа в semantic documents являются JSON numbers. `NaN`, `Infinity` и
`-Infinity` недопустимы уже на JSON boundary. Weight и `riskPenalty` должны
быть finite и строго больше нуля. Bounds `ClosedInterval` должны быть finite и
удовлетворять `minimum <= maximum`.

Integer fields должны находиться в exact interoperable диапазоне
`0..9007199254740991`; конкретное поле может задавать более узкий positive
range. Это исключает потерю точности в TypeScript/ECMAScript до JCS hashing.
JSON number должен иметь одно и то же IEEE 754 binary64 значение на обеих
сторонах boundary.

`y` и public prediction на Arrow boundary остаются Float32. JSON numbers не
задают внутренний tensor dtype: raw coordinates, loss estimates и private
resources могут исполняться с Transformer-owned dtype, включая AMP.

### Objective language revision

Поле:

```json
{"objectiveLanguageRevision":1}
```

задаёт immutable type system, canonicalization, role schemas, resource
lifecycle и общие gradient rules. Каждая primitive identity внутри revision
получает отдельную immutable formula и semantics. `TargetContract` и
`Objective` не интерпретируются отдельно от revision. Начальная proposal
revision равна `1`; она не является продолжением текущего
`objective.schemaVersion=1` и не назначает версию Flight workflow.

Identity primitive определяется парой:

```text
(objectiveLanguageRevision, primitive category, primitive name)
```

Семантика существующей пары не меняется in place.

## `TargetContract`

### Точная структура

```json
{
  "slots": [
    {
      "identity": "MeanReturn",
      "observedConstraint": {
        "kind": "ClosedInterval",
        "minimum": -1,
        "maximum": 1
      },
      "lossInputTransformation": {
        "kind": "Tanh"
      },
      "publicPredictionTransformation": {
        "kind": "Tanh"
      }
    }
  ]
}
```

`slots` — непустой ordered array. Его длина ограничивается advertised
`maxTargetSlots`. `identity` каждого slot уникальна. Позиция в array является
physical target index, но semantic references используют identity.

Каждый slot содержит ровно четыре свойства:

| Поле | Значение |
| --- | --- |
| `identity` | Consumer-owned opaque target identity |
| `observedConstraint` | допустимое множество входного `y` |
| `lossInputTransformation` | преобразование raw coordinate для direct/auxiliary loss roles |
| `publicPredictionTransformation` | преобразование той же raw coordinate для public prediction |

`targetWidth` и `outDim` в документе отсутствуют. Public width однозначно
равна `slots.length`; internal output width дополнительно зависит от private
resources и Transformer-ом наружу не публикуется.

### Constraints revision 1

Допустимы только две формы:

```json
{"kind":"Finite"}
```

```json
{"kind":"ClosedInterval","minimum":0,"maximum":1}
```

`Finite` принимает любое finite Float32 `y`. `ClosedInterval` принимает finite
Float32 в inclusive диапазоне. Bounds являются частью target identity и
canonical digest.

### Transformations revision 1

Допустимы только:

```json
{"kind":"Identity"}
{"kind":"Tanh"}
{"kind":"Sigmoid"}
```

Их mathematical semantics:

```text
Identity(z) = z
Tanh(z)     = tanh(z)
Sigmoid(z)  = 1 / (1 + exp(-z))
```

Для finite logical raw coordinate codomain равен соответственно finite,
`(-1, 1)` и `(0, 1)`. Конкретная stable tensor implementation принадлежит
Transformer.

Public transformation должна статически доказывать совместимость с observed
constraint:

- любая из трёх transformations совместима с `Finite`;
- `Tanh` совместима с `ClosedInterval(minimum, maximum)`, только если весь
  codomain `(-1, 1)` содержится в interval, то есть `minimum <= -1` и
  `maximum >= 1`;
- `Sigmoid` совместима с interval, только если весь codomain `(0, 1)` в нём
  содержится, то есть `minimum <= 0` и `maximum >= 1`;
- `Identity` не доказывает совместимость ни с одним bounded interval.

Это правило проверяет declaration. Фактический `y` всё равно проверяется на
ingress, а public prediction — перед публикацией.

## References

Target value reference имеет точную форму:

```json
{
  "kind": "target",
  "identity": "MeanReturn",
  "view": "lossEstimate"
}
```

`view` принимает ровно одно значение:

- `observed` — `Observed<T>`;
- `lossEstimate` — `LossEstimate<T>`;
- `publicPrediction` — `PublicPrediction<T>`.

Resource reference имеет форму:

```json
{
  "kind": "resource",
  "identity": "returnScale"
}
```

Target reference разрешается через `TargetIdentity -> physical index` только
после полной validation ordered layout. Resource reference разрешается только
в `Objective.resources`. Лишний `view` у resource или отсутствующий `view` у
target является структурной ошибкой.

## `ResourceDeclaration`

### Точная структура

```json
{
  "identity": "returnScale",
  "kind": "PositiveScalarPerObservation"
}
```

Начальная revision содержит один resource kind:

```text
PositiveScalarPerObservation
```

Он означает private differentiable model output со следующими properties:

- один scalar на logical observation;
- значение finite и строго больше нуля;
- resource вычисляется моделью для той же observation;
- trainable parameters и state принадлежат model/checkpoint;
- один `ResourceIdentity` обозначает один shared resource;
- разные identities обозначают независимые resources даже при одинаковом
  kind;
- значение не входит в `y` и public prediction.

Kind не задаёт PyTorch layer, positive parameterization, tensor layout,
physical checkpoint key, batching или storage. Эти детали остаются внутри
Transformer. Resource kind и identity влияют на compatibility, потому что
меняют private model graph.

## `Objective`

### Точная структура

```json
{
  "aggregation": {
    "kind": "WeightedSum"
  },
  "reduction": {
    "kind": "GlobalRowMean"
  },
  "resources": [
    {
      "identity": "returnScale",
      "kind": "PositiveScalarPerObservation"
    }
  ],
  "directComponents": [
    {
      "identity": "direct.mean-return",
      "operator": "SmoothL1",
      "weight": 1,
      "roles": {
        "estimate": {
          "kind": "target",
          "identity": "MeanReturn",
          "view": "lossEstimate"
        },
        "observed": {
          "kind": "target",
          "identity": "MeanReturn",
          "view": "observed"
        }
      },
      "parameters": {}
    }
  ],
  "auxiliaryComponents": [
    {
      "identity": "aux.gaussian-nll",
      "operator": "GaussianNLL",
      "weight": 1,
      "roles": {
        "locationEstimate": {
          "kind": "target",
          "identity": "MeanReturn",
          "view": "lossEstimate"
        },
        "observedLocation": {
          "kind": "target",
          "identity": "MeanReturn",
          "view": "observed"
        },
        "scale": {
          "kind": "resource",
          "identity": "returnScale"
        }
      },
      "parameters": {}
    }
  ]
}
```

Все пять top-level fields обязательны. `resources` и
`auxiliaryComponents` могут быть пустыми arrays. `directComponents` непуст и
содержит ровно один component на каждый target slot.

Каждый component имеет ровно:

- `identity`;
- `operator`;
- positive finite `weight`;
- закрытый object `roles` точной формы operator-а;
- закрытый object `parameters`, в том числе `{}` для operator без parameters.

Component identities уникальны совместно между direct и auxiliary arrays.
Resource identities уникальны внутри `resources`.

### Порядок arrays

Array order является частью canonical document:

- `directComponents[i]` обязан supervising target `slots[i]`;
- `resources` обязан быть отсортирован по ASCII `identity` по возрастанию;
- `auxiliaryComponents` обязан быть отсортирован по ASCII `identity` по
  возрастанию.

Transformer не сортирует неканонический вход молча: нарушение порядка
отклоняется. Это гарантирует одинаковый document и deterministic floating-point
accumulation в Python и Consumer implementation.

Execution order равен `directComponents` по slot order, затем
`auxiliaryComponents` по canonical identity order. Для каждого component
сначала рассчитывается per-observation vector, затем `GlobalRowMean`, затем
weight. `WeightedSum` складывает weighted scalars в execution order. Все
components активны с первого optimizer step.

### Direct operators revision 1

| Operator | Точная форма roles | Parameters | Preconditions и per-observation loss |
| --- | --- | --- | --- |
| `SmoothL1` | `estimate: LossEstimate<T>`, `observed: Observed<T>` | `{}` | Оба refs указывают на один target; fixed `beta=1`; `0.5*d²` при `abs(d)<1`, иначе `abs(d)-0.5` |
| `BinaryCrossEntropyWithLogits` | `logit: LossEstimate<T>`, `probability: Observed<T>` | `{}` | Один target; loss transformation `Identity`; constraint доказывает subset `[0,1]`; `max(logit,0)-logit*probability+ln(1+exp(-abs(logit)))` |
| `LogMSE` | `estimate: LossEstimate<T>`, `observed: Observed<T>` | `{}` | Один target; estimate transformation доказывает `>0`; constraint доказывает `>=0`; `[ln(estimate+1e-6)-ln(observed+1e-6)]²` |

`d = estimate - observed`. `SmoothL1` и BCE formulas используют semantics
соответствующих PyTorch operators, но identity закрепляет математический
результат, а не Python symbol. Изменение `beta`, epsilon или reduction не
является implementation detail и требует новой immutable primitive identity
либо language revision.

### Auxiliary operators revision 1

#### `GaussianNLL`

```text
roles:
  locationEstimate : LossEstimate<T>
  observedLocation : Observed<T>
  scale            : Private<PositiveScalarPerObservation>
parameters: {}
```

Обе target roles ссылаются на один target. Для `mu`, `y` и positive `s`:

```text
variance = s² + 1e-6
loss = 0.5 * ((y - mu)² / variance + ln(variance))
```

Role `scale` передаёт gradient в resource.

#### `ExpectedValue`

```text
roles:
  positiveOutcomeProbability : PublicPrediction<TPositive>
  negativeOutcomeProbability : PublicPrediction<TNegative>
parameters: {}
```

Target identities должны различаться, а обе public transformations должны
доказывать codomain в `[0,1]`:

```text
delta = positiveOutcomeProbability - negativeOutcomeProbability
loss = -delta
```

#### `RiskAdjustedExpectedValue`

```text
roles:
  positiveOutcomeProbability : PublicPrediction<TPositive>
  negativeOutcomeProbability : PublicPrediction<TNegative>
  uncertaintyScale           : Private<PositiveScalarPerObservation>
parameters:
  riskPenalty                : positive finite number
```

Probability requirements совпадают с `ExpectedValue`. Для positive scale `s`:

```text
delta = positiveOutcomeProbability - negativeOutcomeProbability
loss = -(delta - riskPenalty * stopGradient(s) * abs(delta))
```

Stop-gradient относится только к `uncertaintyScale` role этого operator.
Probability branches продолжают участвовать в autograd. Изменение gradient
flow требует новой semantic identity.

`ExpectedValue` и `RiskAdjustedExpectedValue` могут одновременно находиться в
objective как независимые components.

Objective не содержит `balancing`, stage schedule или checkpoint selection.
Наличие только positive static `weight` и `WeightedSum` полностью задаёт
начальную balancing semantics. Selection остаётся Transformer-owned training
policy: score строится из weighted epoch-mean direct losses, auxiliary
components в него не входят.

### Resource graph

Objective принимается, только если:

- каждый resource используется хотя бы одним component;
- kind resource соответствует каждой bound role;
- каждый trainable resource имеет структурный путь к total loss через
  positive-weight component и gradient-producing role;
- отсутствие численного gradient на отдельном batch не считается ошибкой;
- один resource с consumers `GaussianNLL` и
  `RiskAdjustedExpectedValue` валиден;
- resource, используемый только stop-gradient role
  `RiskAdjustedExpectedValue.uncertaintyScale`, невалиден.

## `ModelContract`

### Точная структура

```json
{
  "objectiveLanguageRevision": 1,
  "targetContract": {
    "slots": [
      {
        "identity": "MeanReturn",
        "observedConstraint": {
          "kind": "ClosedInterval",
          "minimum": -1,
          "maximum": 1
        },
        "lossInputTransformation": {
          "kind": "Tanh"
        },
        "publicPredictionTransformation": {
          "kind": "Tanh"
        }
      }
    ]
  },
  "objective": {
    "aggregation": {
      "kind": "WeightedSum"
    },
    "reduction": {
      "kind": "GlobalRowMean"
    },
    "resources": [],
    "directComponents": [
      {
        "identity": "direct.mean-return",
        "operator": "SmoothL1",
        "weight": 1,
        "roles": {
          "estimate": {
            "kind": "target",
            "identity": "MeanReturn",
            "view": "lossEstimate"
          },
          "observed": {
            "kind": "target",
            "identity": "MeanReturn",
            "view": "observed"
          }
        },
        "parameters": {}
      }
    ],
    "auxiliaryComponents": []
  },
  "modelConfig": {
    "architecture": {
      "identity": "transformer.sequence-model",
      "revision": 1
    },
    "seqLen": 10,
    "featureDim": 64848,
    "hidden": 256,
    "layers": 5,
    "dropout": 0.1,
    "nhead": 8,
    "mode": "relaxed"
  }
}
```

`ModelContract` содержит только identity-bearing model semantics. Derived
digests хранятся рядом с ним, но не в нём, чтобы full document не хэшировал
сам себя.

`modelConfig` всегда полностью materialized до hashing. Все его поля
обязательны:

| Поле | Правило |
| --- | --- |
| `architecture.identity` | Transformer-owned opaque architecture identity; proposal value `transformer.sequence-model` |
| `architecture.revision` | positive integer immutable architecture revision; proposal value `1` |
| `seqLen` | positive integer, равен `dataContract.seqLen` |
| `featureDim` | positive integer, равен `dataContract.featureDim` и ширине reconstructed `x` |
| `hidden` | positive integer, делится на `nhead` |
| `layers` | positive integer |
| `dropout` | finite number в `[0,1)` |
| `nhead` | positive integer |
| `mode` | `strict` или `relaxed` |

Provider defaults не остаются неявными внутри canonical `ModelContract`.
Local или Flight adapter может materialize defaults до boundary, но stored,
returned и hashed document всегда полный.

`outDim`, target names и private head count в `modelConfig` отсутствуют:

- public output width выводится из `targetContract.slots.length`;
- private model resources выводятся из validated `objective.resources`;
- architecture revision определяет совместимость physical parameter layout.

Model contract не содержит `dataContractSha256`. Exact Consumer data identity
является отдельным D1 layer. Это позволяет отличить несовпадение данных от
несовпадения model semantics, не ослабляя strict predict и warm-start policy.

Также в `ModelContract` не входят requested device, AMP, optimizer/training
configuration, diagnostics, selection state, initialization lineage и
`sourceEncoding`. Они не меняют target/objective/model layout, но входят в
`jobConfigSha256` и необходимые recovery fences.

### Authority

Consumer materializer формирует полный `TargetContract`, `Objective` и
requested complete `modelConfig`. Transformer:

1. проверяет language revision и advertised primitives;
2. выполняет structural и semantic validation;
3. принимает document без target-specific remapping и hidden defaults;
4. вычисляет D1 digests;
5. сохраняет и возвращает canonical full `ModelContract` вместе с digests.

Consumer может независимо пересчитать digests, но provider-computed значения
являются authoritative для model/checkpoint metadata. Digest никогда не
заменяет full document.

## Canonicalization

### Алгоритм

Для всех Transformer-owned semantic digests применяется один алгоритм:

1. Принять UTF-8 JSON object в пределах advertised document-size limit.
2. Отклонить invalid UTF-8, duplicate object keys, non-JSON numbers и
   non-object top-level value.
3. Проверить закрытую structural schema выбранной language revision.
4. Выполнить semantic validation references, types, ordering, graph и model
   geometry.
5. Убедиться, что все identity-bearing поля materialized; defaults и aliases
   отсутствуют.
6. Построить указанную ниже digest preimage без derived digest fields.
7. Сериализовать preimage по RFC 8785 JSON Canonicalization Scheme.
8. Вычислить SHA-256 от точных JCS UTF-8 bytes и вывести lowercase hex.

JCS сортирует object keys и использует ECMAScript serialization JSON numbers.
Поэтому `1` и `1.0`, а также `0` и `-0`, дают одинаковые canonical bytes.
Array order не меняется. String normalization JCS не выполняет; identities
ограничены ASCII именно поэтому.

Transformer не выполняет:

- сортировку target slots;
- semantic remapping identity;
- замену одного primitive другим;
- заполнение пропущенных identity-bearing fields;
- удаление неизвестных fields;
- canonicalization через обычный `json.dumps(sort_keys=True)` вместо JCS.

### Canonical equality

Два documents canonical-equal, если их JCS bytes равны после успешной
validation. Порядок keys и spelling эквивалентных JSON numbers не влияют на
equality. Порядок arrays, spelling identities и любое значение contract field
влияют.

## D1 layered digests

В model/job metadata четыре semantic layers представлены закрытым object:

```json
{
  "dataContractSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "targetContractSha256": "112f7ecf5fbeb7e70ad4b4657dcd753a1694bbc3cb6ae2d78dfdb91facad03ce",
  "objectiveSha256": "7fba42999d866cc109ea709e55b0c668c4ac7379ecbe47f00b830b51190733e3",
  "modelContractSha256": "a71d514e590fe1eef708a73b4fd08ac7f80b4e090392d5231ed863972315a49d"
}
```

`jobConfigSha256` принадлежит job/recovery metadata, а
`checkpointSha256` — physical artifact metadata. Они не добавляются в
`ModelContract` или этот semantic-layer object.

### `dataContractSha256`

`ConsumerDataDigest` сохраняет wire name `dataContractSha256`. Его вычисляет
Consumer от собственного exact data semantic document. Transformer:

- проверяет lowercase SHA-256 syntax;
- хранит Consumer-provided `dataContract` envelope и digest;
- не пытается реконструировать FIGI, profile, feature formulas или временной
  диапазон;
- сравнивает digest exact для predict и `publishedModel` warm start.

Canonical preimage data digest остаётся Consumer-owned contract и не
переопределяется этим proposal.

Точная замена current `dataContract` не входит в четыре проектируемых
documents. Однако следующий Flight schema не должен проверять литералы
`inventory.learning-dataset` или `inventory.target.v2` как Transformer-owned
constants. Нужные Transformer поля — exact digest, `seqLen` и `featureDim`;
Consumer `id`, `version`, `profile` и аналогичная metadata могут сохраняться
только как opaque values. Их окончательная envelope и участие в Consumer data
digest требуют отдельного подтверждения при wire design.

### `targetContractSha256`

```text
SHA256(JCS({
  "objectiveLanguageRevision": ModelContract.objectiveLanguageRevision,
  "targetContract": ModelContract.targetContract
}))
```

Digest покрывает ordered identities, constraints и обе transformations.

### `objectiveSha256`

```text
SHA256(JCS({
  "objectiveLanguageRevision": ModelContract.objectiveLanguageRevision,
  "objective": ModelContract.objective
}))
```

Digest покрывает array order, component/resource identities, resource kinds,
operators, typed bindings, weights, parameters, aggregation и reduction.

### `modelContractSha256`

```text
SHA256(JCS({
  "objectiveLanguageRevision": ModelContract.objectiveLanguageRevision,
  "modelConfig": ModelContract.modelConfig,
  "targetContractSha256": targetContractSha256,
  "objectiveSha256": objectiveSha256
}))
```

Model digest связывает Transformer architecture/configuration с exact target
и objective layers. `dataContractSha256` намеренно не входит в него и
проверяется отдельным layer.

### `jobConfigSha256`

`JobConfigHash` получает wire name `jobConfigSha256`. Service строит preimage
из уже resolved immutable job configuration.

Для fit preimage содержит ровно следующие поля и canonical values:

```text
operation:                 "fit"
requestedDevice:           canonical requested device policy
modelLabel:                validated publication label
sourceEncoding:            full canonical source-encoding document
dataContractSha256:        exact Consumer digest
modelContractSha256:       exact provider-computed model digest
trainingConfig:            full resolved training configuration
diagnostics:               full resolved diagnostics configuration
initialization:            full resolved initialization lineage
```

Для predict:

```text
operation:                 "predict"
requestedDevice:           canonical requested device policy
resolvedModelRef:          immutable owner-scoped model reference
predictionColumn:          validated public output column
sourceEncoding:            full canonical source-encoding document
dataContractSha256:        exact Consumer digest
modelContractSha256:       exact provider-computed model digest
```

Эти списки задают exact object fields; wire spelling находится слева, а
справа указан источник полного значения. Никакие дополнительные поля в digest
preimage не входят.
`trainingConfig`, `diagnostics` и resolved `initialization` должны быть
полностью materialized до hashing. Для `publishedModel` resolved initialization
покрывает как минимум parent `modelRef`, checkpoint digest, parent/current data
digests и parent/current model-contract digests.

`requestId`, `idempotencyKey`, `jobId`, `clientExecutionId`, `fencingToken`,
timestamps, selected physical device, progress, input receipts и attempts в
`jobConfigSha256` не входят. Их identity/fencing имеют отдельные contracts.
Request idempotency hash и `jobConfigSha256` — разные значения и не должны
переиспользоваться друг вместо друга.

### `checkpointSha256`

`CheckpointDigest` сохраняет wire name `checkpointSha256` в semantic model и
равен SHA-256 exact physical checkpoint bytes. Full checkpoint metadata
проверяется отдельно через D1 documents. Input `manifestSha256` также остаётся
отдельным durable-input fence и не входит в D1 model identity.

### Golden vector proposal

Для приведённого выше single-regression `ModelContract` ожидаются:

```text
targetContractSha256 = 112f7ecf5fbeb7e70ad4b4657dcd753a1694bbc3cb6ae2d78dfdb91facad03ce
objectiveSha256      = 7fba42999d866cc109ea709e55b0c668c4ac7379ecbe47f00b830b51190733e3
modelContractSha256  = a71d514e590fe1eef708a73b4fd08ac7f80b4e090392d5231ed863972315a49d
```

Значения вычислены RFC 8785/JCS + SHA-256 и становятся normative только после
переноса этого example в versioned contract fixtures. На стадии Design Note
они служат cross-language review vector.

## Compatibility policies

| Operation | Exact comparisons до upload |
| --- | --- |
| New fit с `random` | self-consistency `dataContract`, `ModelContract`, geometry и capabilities; parent отсутствует |
| Predict | selected model owner/ref; `dataContractSha256`, `targetContractSha256`, `objectiveSha256` и `modelContractSha256` exact |
| Fit с `publishedModel` | те же exact data, target, objective и model layers; checkpoint integrity; новая training state |
| Recovery | все D1 layers, `jobConfigSha256`, input `manifestSha256`, checkpoint/recovery format и attempt fencing |

Target и objective comparisons остаются явными, даже если равенство
`modelContractSha256` логически следует из них. Это нужно для точной mismatch
diagnostics; full documents всё равно revalidate для обнаружения corruption.

Различие `dataContractSha256` остаётся несовместимым для warm start. Временные
`from`/`to`, не входящие в Consumer digest, могут отличаться. Consumer-neutral
target identities не вводят cross-instrument transfer semantics.

## Hybrid language capabilities

### Точная capability projection

Будущая Flight capabilities response содержит отдельный closed block:

```json
{
  "objectiveLanguage": {
    "revision": 1,
    "constraints": [
      "ClosedInterval",
      "Finite"
    ],
    "transformations": [
      "Identity",
      "Sigmoid",
      "Tanh"
    ],
    "resourceKinds": [
      "PositiveScalarPerObservation"
    ],
    "directOperators": [
      "BinaryCrossEntropyWithLogits",
      "LogMSE",
      "SmoothL1"
    ],
    "auxiliaryOperators": [
      "ExpectedValue",
      "GaussianNLL",
      "RiskAdjustedExpectedValue"
    ],
    "aggregations": [
      "WeightedSum"
    ],
    "reductions": [
      "GlobalRowMean"
    ]
  },
  "semanticLimits": {
    "maxTargetSlots": "<positive integer>",
    "maxObjectiveComponents": "<positive integer>",
    "maxPrivateResources": "<positive integer>"
  },
  "modelArchitectures": [
    {
      "identity": "transformer.sequence-model",
      "revision": 1
    }
  ]
}
```

Placeholders обозначают runtime-advertised integer values, а не strings в
будущем wire document. Все primitive arrays сортируются по ASCII identity.
Consumer рассматривает их как advertised sets после проверки canonical order.

Capabilities не повторяет role schemas и formulas: их определяет immutable
language revision. `maxTargetSlots` является provider capacity, а не доменной
константой `6`. `maxObjectiveComponents` применяется к сумме direct и
auxiliary components. `modelArchitectures` сообщает Transformer-owned model
implementations, допустимые в полном `modelConfig`; array имеет canonical
порядок `(identity, revision)`.

Каждый advertised primitive должен иметь Transformer-owned immutable semantic
definition и numerical golden fixtures. Одного имени в capabilities
недостаточно для определения новой formula; capabilities сообщает availability
уже опубликованной semantics.

### Evolution rules

- Flight workflow version и language revision независимы.
- Новый target на уже advertised primitives не требует Transformer release.
- Новый primitive с roles, выражаемыми revision 1 type system, может быть
  добавлен Transformer release и advertised capabilities без новой Flight
  workflow version.
- Consumer использует новый primitive только после собственного release,
  который знает его immutable semantics, и capability negotiation.
- Удаление advertised primitive является breaking deployment change для jobs,
  которые на него рассчитывают.
- Изменение существующей formula, role schema, parameter set, resource
  lifecycle, canonicalization или gradient semantics требует новой primitive
  identity либо language revision.
- Изменение type system или reference/resource model всегда требует новой
  language revision.
- Clean cut позволяет production runtime поддерживать одну revision; aliases
  старой revision не требуются.

Structural Flight schema проверяет envelope и lexical identities. Доступность
конкретного primitive проверяется semantic registry после выбора language
revision. Таким образом additive operator не вынуждает выпускать новый Flight
workflow только ради расширения JSON enum.

## Validation и error model

### Порядок

Validation выполняется до durable input upload:

1. UTF-8/JSON/duplicate-key и document-size checks.
2. Closed structural shape и lexical fields.
3. Language revision и primitive availability.
4. Target order, uniqueness, constraints и transformations.
5. Component/resource identities, canonical array order и references.
6. Direct coverage и operator-specific typed roles.
7. Auxiliary roles, parameters и probability/location semantics.
8. Resource graph, sharing и gradient reachability.
9. Model geometry и Consumer data geometry.
10. Canonicalization и D1 computation.
11. Operation-specific compatibility comparison.

Ingress Arrow validation идёт позже, но до durable commit конкретного
payload: schema/geometry, Float32/finite `tgt`, per-slot constraint и
`indexedFeatureBlocks` bounds. Runtime отдельно проверяет finite transformed
values, resources, component losses и public predictions.

### Logical error detail

Каждая semantic failure имеет логическую форму:

```json
{
  "code": "INVALID_ARGUMENT",
  "reason": "INVALID_TARGET_CONTRACT",
  "path": "/modelContract/targetContract/slots/0/publicPredictionTransformation",
  "message": "public transformation does not satisfy observed constraint"
}
```

`code` остаётся стабильным Transformer application code. `reason` и `path`
предназначены для machine-actionable diagnostics; `message` — bounded human
text и не участвует в branching. `path` является RFC 6901 JSON Pointer для
JSON document. Для Arrow value error detail вместо `path` содержит:

```json
{
  "targetIdentity": "MeanReturn",
  "targetIndex": 0,
  "logicalRow": 42
}
```

Способ переноса detail через конкретный Flight error envelope определяется на
wire-design этапе. Application code и reason semantics уже не должны зависеть
от текста message.

### Категории

| Ситуация | `code` | `reason` или layer |
| --- | --- | --- |
| Malformed JSON, closed-schema violation, invalid identity/order/reference/role/graph | `INVALID_ARGUMENT` | `INVALID_TARGET_CONTRACT`, `INVALID_OBJECTIVE`, `INVALID_RESOURCE_GRAPH` или `INVALID_MODEL_CONTRACT` |
| Primitive неизвестен для выбранной revision | `INVALID_ARGUMENT` | `UNKNOWN_PRIMITIVE` |
| Revision или известный primitive не поддержан этим deployment | `FAILED_PRECONDITION` | `LANGUAGE_REVISION_UNAVAILABLE` или `PRIMITIVE_UNAVAILABLE` |
| `tgt` не finite или нарушает declared constraint | `INVALID_ARGUMENT` | `TARGET_VALUE_INVALID` с target identity/index/row |
| Два корректных contracts несовместимы для predict/warm start | `MODEL_SCHEMA_MISMATCH` | layer `data`, `target`, `objective` или `model` |
| Stored full document не проходит validation либо не соответствует stored digest | `MODEL_CORRUPT` | layer соответствующего document |
| Recovery checkpoint относится к другому job/model/config/input | `RECOVERY_CHECKPOINT_INCOMPATIBLE` | layer `jobConfig`, `data`, `target`, `objective`, `model` или `inputManifest` |
| Runtime создал non-finite/private/public output или нарушил validated plan | `WORKER_PROTOCOL_VIOLATION` либо `MALFORMED_OUTPUT` | provider-owned runtime reason |

Для mismatch detail содержит:

```json
{
  "code": "MODEL_SCHEMA_MISMATCH",
  "reason": "DIGEST_MISMATCH",
  "layer": "target",
  "expectedSha256": "<model digest>",
  "actualSha256": "<request digest>",
  "message": "target contract does not match the selected model"
}
```

Self-inconsistent новый fit является `INVALID_ARGUMENT`, а не
`MODEL_SCHEMA_MISMATCH`. Correct-but-incompatible request не является
`MODEL_CORRUPT`. Corrupt stored model не маскируется как обычный mismatch.

## Checkpoint, recovery и model description

### Checkpoint metadata

Новый checkpoint metadata должен сохранять:

- полный Consumer `dataContract` и `dataContractSha256`;
- полный canonical `ModelContract`;
- `targetContractSha256`, `objectiveSha256` и
  `modelContractSha256`;
- полный resolved training/diagnostics/selection contracts;
- `jobConfigSha256` и input `manifestSha256`;
- resolved initialization lineage;
- checkpoint format/generation и physical checkpoint digest.

Private resource values и parameters находятся только в checkpoint state.
Metadata хранит abstract `ResourceDeclaration`; physical state keys не входят
в public contract. Architecture revision обязана однозначно восстановить
target slots и private resources из canonical documents.

### Recovery

Recovery сначала проверяет metadata integrity и D1 digests, затем exact
`jobConfigSha256` и input manifest, и только после этого загружает model,
optimizer, AMP scaler, RNG, progress и selection state. ComponentIdentity и
ResourceIdentity используются для semantic validation, но не разрешают
частичную загрузку или remapping.

### `model.describe`

Model description возвращает полный checkpoint-owned `dataContract`,
`ModelContract`, все D1 digests, initialization lineage и checkpoint summary.
Он не возвращает Consumer formulas, descriptions, FIGI interpretation или
profile-file paths. Ordered slots позволяют Consumer сопоставить prediction
coordinates со своим catalog без lookup со стороны Transformer.

Prediction Arrow schema остаётся одним non-null
`FixedSizeList<Float32>[slots.length]`. Target metadata не дублируется в каждом
batch: её authoritative source — model/job contract.

## Telemetry, metrics и OpenSearch

Новая metrics identity потребуется, потому что current metrics v4 кодирует
старые target/operator assumptions. Конкретный version/index name назначается
только на implementation design.

Общий record сохраняет:

- `jobId`, attempt, epoch/step и optional `modelRef`;
- `dataContractSha256`;
- `targetContractSha256`;
- `objectiveSha256`;
- `modelContractSha256`;
- checkpoint/job identities, необходимые конкретному event.

Direct observation имеет bounded nested form:

```json
{
  "componentIdentity": "direct.mean-return",
  "operator": "SmoothL1",
  "targetIdentity": "MeanReturn",
  "targetIndex": 0,
  "value": 0.0123
}
```

Auxiliary observation содержит `componentIdentity`, `operator`, numerical
value и при необходимости resolved role summary. Gradient diagnostics
ссылается на `componentIdentity`; direct component дополнительно сохраняет
target identity/index. Pair observations используют две component identities,
а не concatenated target-specific field names.

OpenSearch mappings должны использовать bounded arrays/nested objects и
`keyword` для opaque identities. Dynamic field names из target или component
identity запрещены. Full `TargetContract` и `Objective` не копируются в каждый
point: run/model metadata содержит canonical documents, а points — digests и
references.

При согласованном clean cut прежние models, runtime metadata и historical
metrics удаляются. Legacy reader, mixed metrics index и conversion старых
component names не требуются. Удаление и установка нового template являются
отдельной будущей operational procedure, а не частью этого proposal.

## Cross-project fixtures

### Один источник и offline copies

После принятия contract Transformer владеет provider-neutral normative fixture
bundle рядом с versioned language contract. Inventory хранит byte-identical
offline copy bundle и его manifest digest, чтобы tests не зависели от сети или
наличия второго repository.

Inventory дополнительно владеет своими materialization fixtures:

```text
target catalog + ML profile -> exact ModelContract JSON
```

Transformer не копирует target catalog, profile file или formulas Inventory.
Cross-project equality проверяется по canonical bytes и expected D1 digests.

Каждый semantic fixture имеет форму:

```json
{
  "fixtureId": "consumer-neutral.single-regression.v1",
  "modelContract": {},
  "expected": {
    "validation": "accepted",
    "targetContractSha256": "<sha256>",
    "objectiveSha256": "<sha256>",
    "modelContractSha256": "<sha256>"
  }
}
```

Rejected fixture вместо digests содержит exact `code`, `reason`, optional
`path` и mismatch `layer`. Fixture wrapper не является runtime document и в
D1 digests не входит.

### Обязательный semantic bundle

| Fixture | Что доказывает |
| --- | --- |
| `single-regression` | `Tanh/Tanh`, `SmoothL1`, raw `0.5`, prediction `~0.462117`, observed `0.25`, три golden D1 digests |
| `single-probability` | `Identity/Sigmoid`, BCE-with-logits, raw `0`, prediction `0.5`, observed `1` |
| `multi-target-shared-resource` | direct coverage, Gaussian NLL и RiskAdjusted EV делят один positive private resource; numerical risk loss `-0.49` |
| `target-reorder-a-b` и `target-reorder-b-a` | refs сохраняют identities, physical indices и target/model digests меняются |
| `new-opaque-target` | `EventProbability` принимается только по primitives, без target-name registry |
| `predict-warm-start-mismatch` | data, target, objective и model mismatch классифицируются раздельно до upload |

Numerical fixtures передают logical raw coordinates, observed values и
resource values отдельно от model implementation. Expected values задаются с
обоснованной Float32 tolerance; canonical JSON/digest assertions остаются
exact.

### Обязательный negative bundle

| Fixture | Ожидаемая ошибка |
| --- | --- |
| `resource-stop-gradient-only` | `INVALID_RESOURCE_GRAPH` |
| `bounded-public-identity` | `INVALID_TARGET_CONTRACT` |
| `same-kind-distinct-resource-identities` | принимаются как два resources; fixture проверяет отсутствие sharing |
| `expected-and-risk-adjusted-coexist` | оба components принимаются и independently weighted |
| `changed-transformation` | новый target digest и target-layer mismatch |
| `changed-resource-binding` | новый objective/model digest и objective-layer mismatch |
| `unknown-target-name-existing-primitives` | принимается; отсутствие target-specific dispatch |
| `unknown-primitive` | `UNKNOWN_PRIMITIVE` |
| `known-but-unavailable-primitive` | `PRIMITIVE_UNAVAILABLE` |
| `duplicate-json-key` | request отклоняется до schema/semantic hashing |

### Cross-language digest fixture

Минимум один fixture, приведённый в разделе `ModelContract`, вычисляется
независимыми Python и dependency-free Node.js implementations RFC 8785/JCS.
Обе стороны проверяют exact canonical bytes и три expected digests. Обычный
sorted-key JSON не принимается как замена JCS.

## Оценка `indexedFeatureBlocks`

### Результат

`indexedFeatureBlocks` рекомендуется сохранить как единственный compact input
encoding следующего contract. Он находится на physical `x` boundary и не
противоречит consumer-neutral target/objective model.

Encoding содержит только:

- ordered block position;
- native window row count;
- native feature-row width;
- Consumer-computed native Float32 rows;
- local observation offsets;
- generic range/example continuity и physical/logical counters.

Он не содержит FIGI, interval, indicator identity, target name, target formula
или profile-specific dispatch. Consumer продолжает вычислять features, causal
projection, missing-data policy и offsets. Transformer только проверяет bounds
и детерминированно восстанавливает logical
`[rows, seqLen, featureDim]` tensor.

### Identity placement

`sourceEncoding`:

- не входит в `dataContractSha256`, если Consumer подтверждает ту же logical
  data semantics;
- не входит в target, objective или model digests;
- входит в `jobConfigSha256`, worker manifest и recovery fencing;
- фиксируется в input receipts и physical/logical telemetry;
- не меняет prediction schema.

`targetWidth` для fit `tgt` теперь выводится из
`ModelContract.targetContract.slots.length`, а per-slot validation — из
generic TargetContract. Это единственное необходимое semantic соединение с
новой model; само compact reconstruction не меняется.

### Что нужно очистить при реализации

Следующие current-v10 хвосты не принадлежат encoding и не должны переноситься:

- fixture identities `production-core-v2` и `production-core-v6` в
  Transformer normative bundle;
- target-range validation по имени `MeanReturn`;
- canonical global target order и maximum width `6`;
- target-specific `tgt`/prediction checks.

Provider-neutral fixtures должны называться по геометрии, например
`single-block` и `heterogeneous-multi-block`. Inventory отдельно проверяет, что
конкретный profile materialize ту же geometry.

### Обязательные encoding fixtures

Сохраняются действующие классы сценариев:

1. один feature block;
2. два heterogeneous blocks с разными `windowRows` и `nativeRowWidth`;
3. независимое изменение offset второго блока на causal boundary;
4. split одного range между self-contained chunks с повторённым halo;
5. пропущенная observation без изменения logical order;
6. отсутствующий native prefix и out-of-bounds offset как rejection;
7. bitwise-equivalent reconstructed Float32 `x` относительно dense oracle;
8. равенство logical row order независимо от RecordBatch/payload boundaries.

`rangeOrdinal` трактуется как dense ordinal успешно переданных generic
sequence groups. Он не означает торговый период и не создаёт Inventory domain
dependency.

### Почему отказ нецелесообразен

Возврат к dense input снова материализует перекрывающиеся native windows и
sequences и восстанавливает terabyte-scale transport/storage problem, не
улучшая boundary ownership. Расчёт feature semantics в Transformer нарушил бы
границу сильнее. Оснований проектировать третий encoding до обнаружения
дефекта `indexedFeatureBlocks` нет.

Окончательное включение encoding в следующий versioned contract требует
успешной cross-project equivalence fixture, но не повторного проектирования
target/objective model.

## Clean transition

Сохраняется согласованный T1:

- production принимает только новый consumer-neutral contract;
- старые modelRef, checkpoints, recovery state, runtime metadata и historical
  metrics удаляются согласованной operational procedure;
- old predict/warm-start readers, aliases и dual-write отсутствуют;
- weights не конвертируются;
- Consumer переобучает models и получает новый metrics baseline;
- временный local equivalence bridge допустим только для development fixtures
  и удаляется до release.

`indexedFeatureBlocks` может быть перенесён без dense compatibility path:
сохраняется его математическая reconstruction semantics, а public namespace и
schema identity назначаются следующей Flight version.

## Результат cross-project approval

Transformer и Inventory подтвердили:

1. exact spelling и lexical pattern трёх identity namespaces;
2. точные field names четырёх canonical documents и reference forms;
3. обязательный canonical order resources и auxiliary components;
4. formulas, constants и gradient semantics initial operators;
5. proposal architecture identity `transformer.sequence-model` revision `1`;
6. отсутствие `outDim` и derived target width в `modelConfig`;
7. D1 preimages, JCS rules и authoritative digest ownership;
8. разделение request idempotency hash, `jobConfigSha256`, input manifest и
   physical checkpoint digest;
9. capability block, ASCII sorting и additive primitive lifecycle;
10. error codes/reasons/layers и минимальный structured detail;
11. fixture ownership и byte-identical offline copy policy;
12. сохранение `indexedFeatureBlocks` как job/recovery concern;
13. consumer-neutral replacement envelope для current `dataContract` без
    Transformer-owned literals `inventory.*`;
14. T1 deletion boundary и отсутствие legacy runtime.

Cross-project approval завершён. Следующий этап — проектирование versioned
Flight/Worker schemas, назначение format identities и подготовка implementation
plan. До реализации этот document остаётся ненормативной основой будущего
contract и не изменяет действующую систему.
