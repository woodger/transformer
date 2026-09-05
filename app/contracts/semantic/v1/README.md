# Контракт consumer-neutral semantic model v1

> CONTRACT DOCUMENT. Этот каталог задаёт нормативный общий язык target slots,
> objective, model identity, capabilities и D1 digests для Flight v11, worker
> v12, checkpoint/recovery v6 и metrics v5.

JSON Schemas Draft 2020-12 и golden fixtures являются источником истины для
формы документов. Семантические правила, которые JSON Schema выразить не
может, заданы ниже. Runtime Flight v10 и worker v11 этот пакет не используют.

Пакет состоит из этого общего semantic language, публичного
[`Flight v11`](../../flight/v11/README.md), process-контракта
[`worker v12`](../../worker/v12/README.md),
[`checkpoint/recovery v6`](../../checkpoint/v6/README.md) и двух metrics v5
контрактов: [epoch points](../../metrics/v5/README.md) и
[terminal fit run](../../metrics/fit_run/v5/README.md). Версии независимы, но
production включает их только одним clean-cut implementation change.

## Общие правила

- Все документы закрыты; неизвестные поля запрещены.
- Все identity регистрозависимы, ASCII и соответствуют
  `^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`.
- `TargetIdentity`, `ComponentIdentity` и `ResourceIdentity` образуют разные
  namespaces.
- JSON numbers обязаны быть конечными IEEE 754 binary64 значениями. Integer
  fields ограничены диапазоном `0..9007199254740991`.
- `objectiveLanguageRevision=1` задаёт immutable type system, canonicalization,
  operator roles, resource lifecycle и gradient semantics.
- `y` и public prediction на Arrow boundary — finite Float32. Внутренний dtype
  raw coordinates, loss estimates и private resources принадлежит Transformer.

## Target slots и transformations

`TargetContract.slots` — непустой ordered array. Identity slots уникальны;
позиция является physical target index. Semantic references используют
identity, а не index. Public output width равна числу slots.

Revision 1 поддерживает constraints `Finite` и `ClosedInterval` и
transformations `Identity`, `Tanh`, `Sigmoid`. Для `ClosedInterval` выполняется
`minimum <= maximum`. Public transformation должна статически доказывать
совместимость с observed constraint:

- любая transformation совместима с `Finite`;
- `Tanh` совместима с interval только при `minimum <= -1` и `maximum >= 1`;
- `Sigmoid` совместима с interval только при `minimum <= 0` и `maximum >= 1`;
- `Identity` не доказывает совместимость с bounded interval.

`lossInputTransformation` и `publicPredictionTransformation` применяются к
одной логической raw coordinate независимо. Direct operator не выводится из
target identity.

## Objective

Каждый slot имеет ровно один direct component. `directComponents[i]`
supervise-ит `slots[i]`. `resources` и `auxiliaryComponents` отсортированы по
ASCII identity; component identities уникальны совместно в обоих arrays.
Execution order: direct components в slot order, затем auxiliary components в
canonical identity order. Каждый component вычисляет per-observation vector,
`GlobalRowMean`, weight, затем `WeightedSum` складывает scalars в этом порядке.
Все weights finite и строго положительны; все components активны с первого
optimizer step.

Operator roles и semantics revision 1:

- `SmoothL1`: `estimate=LossEstimate<T>`, `observed=Observed<T>` одного slot,
  fixed `beta=1`.
- `BinaryCrossEntropyWithLogits`: `logit=LossEstimate<T>`,
  `probability=Observed<T>` одного slot; loss transformation `Identity`,
  observed constraint является subset `[0,1]`.
- `LogMSE`: `estimate=LossEstimate<T>`, `observed=Observed<T>` одного slot;
  estimate строго положителен, observed неотрицателен, epsilon равен `1e-6`.
- `GaussianNLL`: location estimate и observed одного slot плюс private positive
  scale; `variance=scale^2+1e-6`; scale получает gradient.
- `ExpectedValue`: две различные public probability roles;
  `loss=-(positive-negative)`.
- `RiskAdjustedExpectedValue`: те же probability roles и private scale;
  `loss=-(delta-riskPenalty*stopGradient(scale)*abs(delta))`.

`ExpectedValue` и `RiskAdjustedExpectedValue` могут сосуществовать.
`PositiveScalarPerObservation` — private differentiable model output: один
finite strictly-positive scalar на observation, checkpoint-owned и не
публикуемый в prediction. Одна resource identity означает sharing; разные
identities — независимые resources. Каждый resource используется и имеет
структурный gradient-producing path к total loss. Resource, используемый
только stop-gradient roles, невалиден.

## Model и D1 digests

`ModelContract` содержит language revision, полный `TargetContract`, полный
`Objective` и полностью materialized `modelConfig`. `outDim`, device, AMP,
training policy, diagnostics, initialization, data digest и source encoding в
него не входят. `seqLen` и `featureDim` совпадают с data contract; `hidden`
делится на `nhead`.

После полной validation preimages сериализуются RFC 8785/JCS в UTF-8 и
хэшируются SHA-256:

```text
targetContractSha256 = SHA256(JCS({
  objectiveLanguageRevision,
  targetContract
}))

objectiveSha256 = SHA256(JCS({
  objectiveLanguageRevision,
  objective
}))

modelContractSha256 = SHA256(JCS({
  objectiveLanguageRevision,
  modelConfig,
  targetContractSha256,
  objectiveSha256
}))
```

Consumer вычисляет `dataContractSha256` по своему нормативному data document.
Transformer проверяет syntax, хранит digest и сравнивает его exact, не
интерпретируя Consumer semantics. Full documents остаются источниками смысла;
digests являются производными. `jobConfigSha256`, input `manifestSha256` и
physical `checkpointSha256` — отдельные fences.

JCS не заменяется обычной сортировкой keys. Duplicate JSON keys, invalid UTF-8,
non-JSON numbers и неmaterialized defaults отклоняются до hashing. Arrays не
сортируются молча.

## Hybrid capabilities

Language revision и Flight workflow версионируются независимо. Capabilities
перечисляют доступные primitives, resource kinds, model architectures и
deployment limits в canonical ASCII order. Новый target на уже доступных
primitives не требует выпуска Transformer. Новая primitive с ролями,
выражаемыми revision 1, может быть additive capability; Consumer использует её
только после собственного выпуска и negotiation. Изменение type system,
reference/resource lifecycle, существующей formula или gradient semantics
требует новой identity либо language revision.

`model-contract-envelope.schema.json` проверяет только закрытый общий envelope.
После выбора revision Transformer обязан применить её exact schema и
Transformer-owned semantic registry. Для additive operator-а revision 1
generic component shape разрешает lexical identity, typed value references и
object parameters, но только опубликованная immutable primitive definition
задаёт точные roles, parameters и formula. Неизвестная identity возвращает
`UNKNOWN_PRIMITIVE`, известная, но не advertised — `PRIMITIVE_UNAVAILABLE`.

## Semantic validation

После structural schema выполняются по порядку:

1. availability language revision и primitives;
2. uniqueness/order slots, components и resources;
3. resolution всех typed references;
4. direct coverage и operator-specific roles/domains;
5. auxiliary roles, parameters и probability semantics;
6. resource sharing и structural gradient reachability;
7. model и data geometry;
8. JCS/D1 computation;
9. operation-specific exact compatibility.

Predict и `publishedModel` требуют exact equality data, target, objective и
model digest layers до upload. Recovery дополнительно требует exact
`jobConfigSha256`, input `manifestSha256`, checkpoint/recovery format и attempt
fences. Correct-but-incompatible contract — `MODEL_SCHEMA_MISMATCH`, corrupt
stored contract — `MODEL_CORRUPT`, self-inconsistent new fit —
`INVALID_ARGUMENT`.

## Fixtures

`fixtures/manifest.json` перечисляет byte-identical cross-project bundle, а
`fixtures/manifest.sha256` фиксирует identity самого manifest.
`d1_sha256.mjs` и общий `jcs_sha256.mjs` являются dependency-free Node.js
реализациями для независимой проверки Python/TypeScript canonical bytes и
digests. Accepted fixtures покрывают все advertised v1 operators, probability
output, shared private resource, одновременные ExpectedValue/RiskAdjustedEV,
slot reorder и новый opaque target. `negative-cases.json` фиксирует graph,
capability и mismatch classifications; raw duplicate-key vector проверяет
отказ до hashing. Wrapper fixture не входит в D1 preimage.

Inventory хранит offline byte-identical copy этого bundle и отдельно проверяет
materialization своего target catalog и ML profile. Transformer не копирует
Consumer formulas или profile files.
