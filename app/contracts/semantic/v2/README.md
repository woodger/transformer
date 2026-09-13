# Staged contract consumer-neutral semantic model v2

> CONTRACT DOCUMENT. Этот каталог задаёт нормативный staged-язык target slots,
> objective, model identity, capabilities и D1 digests для будущих Flight v14,
> Worker v13, checkpoint/recovery v7 и metrics v6. Пакет не является
> действующим runtime contract до согласованного clean cut.

JSON Schemas Draft 2020-12 и перечисленные manifest-ом golden fixtures являются
источником истины для формы документов. Этот README задаёт межполевые и
математические правила, которые JSON Schema выразить не может. Semantic v1 не
изменяется и не считается совместимым с v2 по representation.

## Scope

Revision 2 меняет только предметный JSON vocabulary:

- universal property `kind` отсутствует;
- constraints выбираются полем `constraint`;
- transformations, aggregation и reduction являются scalar primitives;
- target/resource namespaces задаются `targetIdentity` и `resourceIdentity`;
- private resource lifecycle задаётся `resourceClass`.

Primitive formulas, typed roles, gradient semantics, model architecture и
capacity limits совпадают с revision 1. `ResolvedInitialization` не является
частью Objective Language и принадлежит checkpoint/model-generation v7.

## Общие правила

- Все объекты закрыты; неизвестные поля запрещены. Динамические role и
  parameter maps additive operators также запрещают property с точным
  именем `kind`, включая nested parameter objects.
- Все identity регистрозависимы, ASCII и соответствуют
  `^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`.
- `TargetIdentity`, `ComponentIdentity` и `ResourceIdentity` образуют разные
  namespaces.
- JSON numbers являются конечными IEEE 754 binary64 значениями. Integer fields
  ограничены диапазоном `0..9007199254740991`.
- `objectiveLanguageRevision=2` задаёт immutable type system,
  canonicalization, operator roles, resource lifecycle и gradient semantics.
- `y` и public prediction на Arrow boundary являются finite Float32.
  Внутренний dtype raw coordinates, loss estimates и private resources
  принадлежит Transformer.
- Отсутствующие значения не заменяются неявными defaults.

## Target slots

`TargetContract.slots` — непустой ordered array. Identity slots уникальны;
позиция является physical target index. Semantic references используют
identity, а public output width равна числу slots.

Каждый slot содержит:

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

Revision 2 поддерживает constraints:

```json
{"constraint":"Finite"}
{"constraint":"ClosedInterval","minimum":0,"maximum":1}
```

Для `ClosedInterval` выполняется `minimum <= maximum`. Допустимые scalar
transformations: `Identity`, `Tanh`, `Sigmoid`. Public transformation должна
статически доказывать совместимость с observed constraint:

- любая transformation совместима с `Finite`;
- `Tanh` совместима с interval при `minimum <= -1` и `maximum >= 1`;
- `Sigmoid` совместима с interval при `minimum <= 0` и `maximum >= 1`;
- `Identity` не доказывает совместимость с bounded interval.

`lossInputTransformation` и `publicPredictionTransformation` применяются к
одной logical raw coordinate независимо. Direct operator не выводится из
target identity.

## References и resources

Target references имеют закрытую форму:

```json
{"targetIdentity":"EventProbability","view":"observed"}
```

`view` принимает `observed`, `lossEstimate` или `publicPrediction` согласно
typed role operator-а. Resource reference имеет форму:

```json
{"resourceIdentity":"sharedScale"}
```

Объект не может одновременно содержать target и resource namespace. Target
reference всегда содержит `view`; resource reference его не содержит.

Resource declaration:

```json
{
  "identity": "sharedScale",
  "resourceClass": "PositiveScalarPerObservation"
}
```

`PositiveScalarPerObservation` — private differentiable model output: один
finite strictly-positive scalar на observation, checkpoint-owned и не
публикуемый в prediction. Одна resource identity означает sharing; разные
identities — независимые resources. Transformer владеет layer,
parameterization, internal tensor layout, storage и batching.

Каждый resource используется и имеет structural gradient-producing path к
total loss. Resource, используемый только stop-gradient roles, невалиден.

## Objective

Objective явно содержит:

```json
{
  "aggregation": "WeightedSum",
  "reduction": "GlobalRowMean",
  "resources": [],
  "directComponents": [],
  "auxiliaryComponents": []
}
```

Каждый slot имеет ровно один direct component; `directComponents[i]`
supervise-ит `slots[i]`. Resources и auxiliary components отсортированы по
ASCII identity. Component identities уникальны совместно в обоих arrays.

Execution order: direct components в slot order, затем auxiliary components в
canonical identity order. Каждый component вычисляет per-observation vector,
применяет `GlobalRowMean` и weight; `WeightedSum` складывает scalar values в
этом порядке. Weights finite и строго положительны.

Operator roles и semantics revision 2:

- `SmoothL1`: `estimate=LossEstimate<T>`, `observed=Observed<T>` одного slot,
  fixed `beta=1`;
- `BinaryCrossEntropyWithLogits`: `logit=LossEstimate<T>`,
  `probability=Observed<T>` одного slot; loss transformation `Identity`,
  observed constraint является subset `[0,1]`;
- `LogMSE`: `estimate=LossEstimate<T>`, `observed=Observed<T>` одного slot;
  estimate строго положителен, observed неотрицателен, epsilon `1e-6`;
- `GaussianNLL`: location estimate и observed одного slot плюс private positive
  scale; `variance=scale^2+1e-6`; scale получает gradient;
- `ExpectedValue`: две различные public probability roles;
  `loss=-(positive-negative)`;
- `RiskAdjustedExpectedValue`: те же probability roles и private scale;
  `loss=-(delta-riskPenalty*stopGradient(scale)*abs(delta))`.

`ExpectedValue` и `RiskAdjustedExpectedValue` могут сосуществовать.

## ModelContract и D1

`ModelContract` содержит language revision, полный `TargetContract`, полный
`Objective` и полностью materialized `modelConfig`. Device, AMP, training
policy, diagnostics, initialization, data digest и source encoding в него не
входят. `seqLen` и `featureDim` совпадают с data contract; `hidden` делится на
`nhead`.

После полной validation preimages сериализуются RFC 8785/JCS в UTF-8 и
хэшируются SHA-256:

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

Full documents остаются источниками смысла; digests являются производными.
Semantic v1 и v2 не нормализуются в общий representation-independent preimage.
Даже при одинаковой математике их canonical bytes и D1 различаются.

Consumer вычисляет `dataContractSha256` по собственному нормативному data
document. `jobConfigSha256`, input `manifestSha256` и physical
`checkpointSha256` являются отдельными fences.

Duplicate JSON keys, invalid UTF-8, non-JSON numbers и неmaterialized defaults
отклоняются до hashing. Arrays не сортируются молча.

## Hybrid capabilities

`language-capabilities.schema.json` объявляет revision 2, constraints,
transformations, `resourceClasses`, operators, aggregations, reductions,
architectures и capacity limits. Primitive arrays имеют canonical ASCII order.

Новый Consumer target на доступных primitives не требует выпуска Transformer.
Новая primitive с ролями, уже выражаемыми revision 2, может быть additive
capability. Изменение type system, reference/resource lifecycle, существующей
formula или gradient semantics требует новой primitive identity либо language
revision.

## Validation order

После structural schema выполняются:

1. availability language revision и primitives;
2. uniqueness/order slots, components и resources;
3. resolution typed target/resource references;
4. direct coverage и operator-specific roles/domains;
5. auxiliary roles, parameters и probability semantics;
6. resource sharing и structural gradient reachability;
7. model/data geometry;
8. JCS/D1 computation;
9. operation-specific exact compatibility.

Predict и published-model initialization требуют exact equality data, target,
objective и model digest layers до upload. Correct-but-incompatible contract
возвращает `MODEL_SCHEMA_MISMATCH`; corrupt stored contract — `MODEL_CORRUPT`;
self-inconsistent new fit — `INVALID_ARGUMENT`.

## Fixtures

`fixtures/manifest.json` перечисляет byte-identical cross-project fixture
bundle. `manifest.sha256` фиксирует exact bytes manifest-а. В
`fixtures/jcs-golden.json` лежат буквальные JCS strings и D1 vectors для каждого
positive semantic fixture; dependency-free Node.js scripts независимо их
вычисляют.

Positive fixtures покрывают все primitives, shared resource, одновременные
ExpectedValue/RiskAdjustedExpectedValue, slot reorder и новый opaque target.
Negative fixtures проверяют legacy `kind` form, mixed reference, unknown
subject discriminator, duplicate JSON key, resource graph и capability
outcomes. Legacy `kind` встречается только внутри специально отклоняемого
negative fixture и не является допустимым v2 document.

Inventory хранит byte-identical offline copy и независимо проверяет schemas,
JCS bytes, D1 и semantic cases до реализации.
