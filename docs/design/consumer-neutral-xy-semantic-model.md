# Schema-neutral semantic model границы `x/y`

> Тип: Design Note. Transformer-side предложение математической модели для
> совместного обсуждения с Consumer-ами. Это не ADR, не Flight или Worker
> schema, не план миграции и не описание реализованного поведения.

- Статус: к совместному архитектурному обсуждению
- Срез: 2026-09-05, техническая исходная точка Flight v10 / Worker v11
- Основание: [аудит consumer-neutral границы](./consumer-neutral-xy-boundary.md)
- Входные материалы: два Consumer-side аудита и ответы Inventory от 2026-09-05

## Назначение и границы

Документ конкретизирует semantic model после первого раунда согласования с
Inventory. Он должен позволить проверить математическую полноту границы до
выбора JSON fields, Arrow schemas, protocol versions и способа реализации.

Имена типов и operators ниже являются концептуальной нотацией. Они не
назначают wire spelling, nesting, serialization или номер будущей версии.

Во время обсуждения:

- дальнейшее развёртывание и Consumer-интеграция Flight v10 остановлены;
- код, schemas, migrations и contracts не изменяются;
- текущая реализация остаётся источником проверяемых фактов;
- `indexedFeatureBlocks` рассматривается отдельно как физическое представление
  уже вычисленного `x` и не утверждается частью будущего контракта;
- исполняемый Consumer-код, server-side target presets и lookup в Inventory не
  допускаются.

## Согласованное распределение ответственности

| Понятие | Inventory или другой Consumer | Transformer |
| --- | --- | --- |
| Feature и target formulas | Определяет и вычисляет | Не интерпретирует |
| Target identity | Определяет как устойчивую opaque identity | Проверяет равенство и уникальность, не разбирает значение |
| Target catalog | Хранит семантику `y`, constraint и устойчивое public representation | Не получает ссылку на catalog и не выполняет lookup |
| ML profile | Выбирает ordered targets, direct operators, weights и loss-input representations | Не интерпретирует profile identity |
| Materialization | Собирает полный self-contained target contract и objective | Принимает только materialized document |
| Transformations и operators | Выбирает из поддержанного закрытого языка | Определяет семантику, валидирует и исполняет |
| Private resources | Объявляет abstract resource и связывает roles | Определяет resource kinds, создаёт и сохраняет private model state |
| Model runtime | Не определяет PyTorch implementation | Владеет model, autograd, optimizer, checkpoint и recovery |
| Compatibility | Предоставляет точную Consumer-owned data identity | Определяет и исполняет model/checkpoint/recovery fencing |

Target catalog самостоятельно не пересекает границу. Inventory materializer
объединяет catalog и profile до вызова Transformer. В materialized request нет
ordinal `Indicator`, пути к profile-файлу или имени preset-а, требующего
дополнительного разрешения.

## Семантические сущности

### Target layout

Target contract является непустой ordered последовательностью уникальных
slots:

```text
TargetLayout = OrderedUnique(TargetSlot...)

TargetSlot = (
  TargetIdentity,
  ObservedConstraint,
  LossInputTransformation,
  PublicTransformation
)
```

Один slot задаёт четыре независимых свойства:

1. opaque identity;
2. constraint наблюдаемого `y`;
3. transformation логической raw coordinate в estimate для direct loss;
4. transformation той же raw coordinate в public prediction.

```text
                                      +--> L(raw) --> loss estimate
raw model coordinate for target T ---+
                                      +--> P(raw) --> public prediction

observed y[T] ----------------------------> observed constraint
```

Raw coordinate — логическое понятие model contract. Она не публикуется и не
предписывает Transformer конкретный head, layer или tensor layout.

Все значения `y`, raw model coordinates, loss estimates и public predictions
обязаны оставаться finite Float32. Float32 является общим tensor invariant, а
не настраиваемым свойством каждого slot.

### Objective

Objective состоит из:

```text
Objective = (
  ResourceDeclarations,
  DirectComponents,
  AuxiliaryComponents,
  Aggregation,
  Reduction
)
```

Каждый component имеет локальную opaque identity, operator из закрытого языка,
typed role bindings, положительный finite weight и поддержанные operator
parameters. Component identity нужна для graph references, telemetry и
diagnostics; она не выбирает numerical implementation.

Для исходной модели сохраняются только:

- статические component weights;
- `WeightedSum` aggregation;
- `GlobalRowMean` reduction;
- активация всех объявленных components с первого optimizer step.

Adaptive balancing, stage schedule и исполняемые formulas в эту модель не
добавляются.

### References и namespaces

Semantic binding использует два непересекающихся вида references:

- `TargetRef(TargetIdentity)`;
- `ResourceRef(ResourceIdentity)`.

Для них действуют следующие правила:

- identity сравнивается как opaque value; prefix и предметный смысл не
  интерпретируются;
- target identities уникальны во всём ordered layout;
- resource и component identities уникальны в своих локальных objective
  namespaces;
- reference обязан разрешиться ровно в один объект соответствующего kind;
- target reference в objective задаётся identity, а не physical index;
- после полной проверки Transformer строит неизменяемое соответствие
  `TargetIdentity -> physical index`;
- Arrow, tensor operations и prediction используют вычисленный index;
- telemetry и diagnostics сохраняют одновременно opaque identity и index;
- повторная ссылка на одну ResourceIdentity означает sharing одного private
  resource между operators.

ComponentIdentity остаётся локальной стабильной identity objective component
для telemetry и diagnostics. Нужны ли semantic references на components
внутри самого objective, пока не решено и в начальную модель не добавляется.

Отдельный stable slot ID не вводится. При текущем invariant уникальности target
identity он не добавляет разрешающей способности. Если появится сценарий двух
slots с одной semantic identity, он должен быть рассмотрен отдельно до
ослабления уникальности.

Переименование target identity или перестановка ordered slots меняет target
contract и делает прежнюю model generation несовместимой. Transformer не
выполняет неявный remapping weights или prediction coordinates.

## Минимальный закрытый язык constraints

Предлагаемый начальный каталог содержит только два kinds:

| Constraint | Семантика |
| --- | --- |
| `Finite` | Любое finite Float32 значение |
| `ClosedInterval(minimum, maximum)` | Finite Float32 в inclusive диапазоне; границы finite и `minimum <= maximum` |

Этого достаточно для текущих и согласованных проверочных сценариев:

- bounded regression использует `ClosedInterval(-1, 1)`;
- probability target использует `ClosedInterval(0, 1)`;
- unbounded regression может использовать `Finite`.

`Probability`, `Return`, `Volatility` и другие предметные kinds не вводятся.
`NonNegative`, one-sided ranges и дополнительные constraints могут быть
добавлены только при появлении подтверждённого сценария, который нельзя точно
выразить начальным набором.

Constraint описывает допустимый observed `y`. Mathematical preconditions
operator-а остаются отдельной Transformer-owned проверкой и могут быть строже.

## Минимальный закрытый язык transformations

Начальный каталог содержит три element-wise differentiable transformations:

| Transformation | Логическая семантика | Codomain для finite raw value |
| --- | --- | --- |
| `Identity` | `f(z) = z` | finite Float32 |
| `Tanh` | hyperbolic tangent | `(-1, 1)` |
| `Sigmoid` | logistic sigmoid | `(0, 1)` |

Этого достаточно для действующих representations:

- bounded regression: loss `Tanh`, public `Tanh`;
- positive bounded regression: loss `Sigmoid`, public `Sigmoid`;
- probability with logits: loss `Identity`, public `Sigmoid`.

Transformation identity закрепляет математическую операцию, но не Python
function или PyTorch module. Clipping, scaling, custom parameters и Consumer
callbacks отсутствуют. Положительная parameterization private scale также не
становится target transformation: она скрыта за resource kind.

Public transformation должна гарантировать значения, совместимые с observed
constraint этого slot. Поэтому:

- `Tanh` совместим с `ClosedInterval(-1, 1)`;
- `Sigmoid` совместим с `ClosedInterval(0, 1)`;
- `Identity` без дополнительных гарантий совместим с `Finite`, но не доказывает
  соблюдение bounded interval.

Ingress отдельно проверяет фактический `y`; Worker отдельно проверяет finite
model outputs. Статическая совместимость declaration не заменяет runtime
validation.

## Typed operator roles

### Общая type model

Roles используют не tensor names, а логические views:

```text
Observed<T>          = y target slot T
LossEstimate<T>      = L_T(raw_T)
PublicPrediction<T>  = P_T(raw_T)
Private<K>           = private resource of kind K
```

`T` является TargetIdentity, `K` — Transformer-owned ResourceKind. Role schema
является частью неизменяемой семантики operator identity. Operator получает
ровно объявленные roles; лишние, отсутствующие или несовместимые bindings
отклоняются до upload.

Несколько instances одного operator допустимы при разных ComponentIdentity или
bindings. Это необходимо, например, для нескольких independent regression
slots. Уникальность operator name сама по себе не является objective
invariant.

### Direct operators

Каждый public target slot обязан иметь ровно один direct component. Его
estimate и observed roles ссылаются на один и тот же target.

| Operator | Typed roles | Дополнительные preconditions |
| --- | --- | --- |
| `SmoothL1` | `estimate: LossEstimate<T>`, `observed: Observed<T>` | Одинаковая shape, finite operands |
| `BinaryCrossEntropyWithLogits` | `logit: LossEstimate<T>`, `probability: Observed<T>` | Loss transformation `Identity`; observed constraint доказывает subset `[0, 1]` |
| `LogMSE` | `estimate: LossEstimate<T>`, `observed: Observed<T>` | Estimate строго положителен; observed неотрицателен; epsilon policy принадлежит operator semantics |

В начальном языке strictly-positive loss estimate для `LogMSE` выражается
`Sigmoid`. Его действующий bounded target выражается
`ClosedInterval(0, 1)`. Поддержка unbounded non-negative target потребует
отдельно обоснованного constraint или transformation kind.

Direct operator выбирается materialized profile и никогда не выводится из
TargetIdentity.

### `GaussianNLL`

Typed roles:

```text
locationEstimate : LossEstimate<T>
observedLocation : Observed<T>
scale            : Private<PositiveScalarPerObservation>
```

`locationEstimate` и `observedLocation` обязаны ссылаться на один target.
`scale` является standard-deviation-like положительным значением для той же
logical observation. Построение variance и numerical epsilon являются
семантикой operator-а, а не resource kind или Consumer parameter.

Scale role передаёт gradient в private resource. Поэтому `GaussianNLL` может
калибровать shared scale.

### `ExpectedValue`

Typed roles:

```text
positiveOutcomeProbability : PublicPrediction<TPositive>
negativeOutcomeProbability : PublicPrediction<TNegative>
```

Target references обязаны быть различными. Обе public transformations должны
доказывать probability codomain, совместимый с `[0, 1]`. Порядок направленных
roles является частью operator semantics; Transformer не выводит его из target
names.

### `RiskAdjustedExpectedValue`

Typed roles:

```text
positiveOutcomeProbability : PublicPrediction<TPositive>
negativeOutcomeProbability : PublicPrediction<TNegative>
uncertaintyScale           : Private<PositiveScalarPerObservation>
```

Operator также принимает положительный finite `riskPenalty`. Probability roles
имеют те же требования, что и `ExpectedValue`.

Использование `uncertaintyScale` имеет stop-gradient semantics. Это свойство
конкретного operator identity, а не resource kind. Изменение gradient flow
требует новой operator или language identity; Consumer не переключает detach
произвольным флагом.

Поскольку этот operator не обучает scale через свою role, objective обязан
содержать хотя бы один gradient-producing consumer того же resource. В
начальном каталоге таким consumer является `GaussianNLL`. Это формулируется
через gradient reachability resource graph, а не через target name.

### Objective-level сочетания

Каждый component возвращает per-observation loss; `GlobalRowMean` materialize
его scalar, после чего `WeightedSum` применяет static weights.

Типовая модель не запрещает два instances одного operator. Она также не даёт
математического основания автоматически считать `ExpectedValue` и
`RiskAdjustedExpectedValue` взаимоисключающими. Текущий запрет Flight v10
является отдельной objective policy. Перед будущей schema нужно совместно
решить, сохраняется ли он как явное правило language revision или удаляется
как скрытый preset constraint.

## Abstract private resource

### Declaration

Минимальная declaration содержит:

```text
AbstractResource = (ResourceIdentity, ResourceKind)
```

Shape, domain, lifecycle и gradient capability задаёт закрытая Transformer-owned
семантика ResourceKind. Они не дублируются как произвольно комбинируемые
Consumer fields. Capabilities и model description могут раскрывать эти
логические свойства как описание kind.

### Начальный resource kind

Для воспроизведения подтверждённых objectives требуется один kind:

```text
PositiveScalarPerObservation
```

Здесь observation означает один logical model example, который создаёт одну
строку public prediction; batch axis не входит в логическую shape.

Kind гарантирует:

- один private scalar на observation;
- finite значение строго больше нуля;
- differentiable model output, вычисляемый для той же observation;
- параметры и state являются частью model и checkpoint;
- одинаковая ResourceIdentity во всех bindings обозначает одно и то же
  значение и один и тот же private model resource;
- resource не входит в `y` или public prediction.

Kind не раскрывает:

- PyTorch layer или module path;
- способ positive parameterization;
- расположение private parameters;
- внутренний tensor layout и batching;
- physical keys или storage checkpoint;
- конкретный autograd graph за пределами объявленной operator semantics.

Runtime dispatch по ResourceKind допустим: kind принадлежит Transformer.
Runtime dispatch по TargetIdentity запрещён: identity принадлежит Consumer.

Resource identity, kind и все bindings входят в objective/model compatibility.
Model description возвращает abstract declaration, но не private values.

### Resource graph invariants

- каждый ResourceRef разрешается в одну declaration;
- неиспользуемая declaration отклоняется;
- один resource может использоваться несколькими components;
- kind обязан удовлетворять typed role каждого consumer;
- каждый trainable resource имеет хотя бы один gradient-producing путь от
  objective component;
- stop-gradient задаётся consuming operator-ом, а не resource declaration;
- добавление нового kind требует явной Transformer-owned semantic definition и
  capability.

Composite operator остаётся допустимой будущей альтернативой только если
abstract resource не сможет выразить independent weights, diagnostics или
sharing без раскрытия implementation. Для текущих сценариев такая
необходимость не показана.

## Порядок полной validation

Transformer валидирует materialized model в следующем логическом порядке.
Это порядок зависимостей, а не требование к коду.

1. **Target layout**
   - список непуст;
   - identities непусты и уникальны;
   - порядок фиксирован;
   - target width согласован с model geometry.
2. **Slot declarations**
   - constraints и transformations принадлежат поддержанному language;
   - parameters finite и допустимы;
   - public transformation совместима с observed constraint.
3. **Objective identities и references**
   - component и resource identities уникальны;
   - все TargetRef и ResourceRef разрешаются;
   - reference kind совпадает с ожидаемым namespace.
4. **Direct supervision**
   - каждый target имеет ровно один direct component;
   - estimate и observed указывают на один target;
   - transformation и constraint удовлетворяют operator preconditions.
5. **Auxiliary operators**
   - присутствуют ровно требуемые typed roles и parameters;
   - probability, location и resource domains совместимы;
   - directed roles не переставляются неявно.
6. **Private resource graph**
   - kinds поддержаны;
   - sharing однозначен;
   - каждый resource используется и имеет gradient-producing path;
   - resulting private model layout однозначен.
7. **Canonicalization и compatibility**
   - validated documents канонизируются;
   - вычисляются выбранные semantic digests;
   - predict, warm start и recovery применяют свои exact policies.
8. **Ingress runtime**
   - `y` имеет shape `[rows, targetWidth]`, Float32 и finite values;
   - каждый observed value удовлетворяет своему constraint;
   - проверка не ветвится по TargetIdentity.
9. **Training и prediction runtime**
   - raw, transformed и resource values finite;
   - objective исполняется по resolved indices;
   - наружу выходят только ordered public predictions;
   - private resources не попадают в Arrow output.

Любая semantic несовместимость должна обнаруживаться до durable input upload.
Corrupt persisted metadata и несовместимый корректный contract остаются
разными классами ошибок.

## Проверка на шести schema-neutral сценариях

Target names в примерах нужны только для читаемости и остаются opaque values.

### 1. Один regression target

```text
slot A:
  constraint = ClosedInterval(-1, 1)
  loss       = Tanh
  public     = Tanh

direct-A:
  SmoothL1(
    estimate = LossEstimate<A>,
    observed = Observed<A>
  )
```

Для raw `0.5` оба transformed values равны приблизительно `0.462117`; observed
`0.25` принимается. TargetIdentity не участвует в выборе `Tanh` или
`SmoothL1`.

### 2. Один probability target

```text
slot B:
  constraint = ClosedInterval(0, 1)
  loss       = Identity
  public     = Sigmoid

direct-B:
  BinaryCrossEntropyWithLogits(
    logit       = LossEstimate<B>,
    probability = Observed<B>
  )
```

Raw `0` поступает в loss как logit `0`, а public prediction равна `0.5`.
Observed `1.2` отклоняется ingress validation по constraint независимо от
identity slot-а.

### 3. Несколько targets и auxiliary loss

Layout содержит regression location `L` и две probability coordinates `P+` и
`P-`. Каждый slot имеет свой direct component. Objective объявляет:

```text
resource U:
  kind = PositiveScalarPerObservation

GaussianNLL:
  locationEstimate = LossEstimate<L>
  observedLocation = Observed<L>
  scale             = Resource<U>

RiskAdjustedExpectedValue:
  positiveOutcomeProbability = PublicPrediction<P+>
  negativeOutcomeProbability = PublicPrediction<P->
  uncertaintyScale           = Resource<U>
  riskPenalty                = 0.1
```

Один ResourceIdentity обеспечивает sharing. `GaussianNLL` создаёт gradient path
для scale; risk-adjusted operator использует тот же scale со stop-gradient.
При probabilities `0.75` и `0.25`, scale `0.2` и `riskPenalty=0.1` действующая
формула risk-adjusted component даёт `-0.49`.

### 4. Перестановка target slots

Layout `[A, B]` заменяется `[B, A]`. Semantic references продолжают указывать
на identities, поэтому binding не начинает молча ссылаться на другой target.
Однако physical indices, ordered target contract и его digest меняются.
Прежняя model generation несовместима; remapping weights запрещён.

### 5. Новый target на существующих primitives

Новый opaque target `EventProbability` использует
`ClosedInterval(0, 1)`, loss `Identity`, public `Sigmoid` и
`BinaryCrossEntropyWithLogits`. Его принимает generic validation без изменения
target catalog Transformer и без ветвления по строке `EventProbability`.

Нужны новая Consumer profile/model generation, но не новый Transformer
operator. Требование к release или language revision определяется только
наличием используемых primitives, а не target identity.

### 6. Несовместимый predict или warm start

Модель сохранена для slot `[A]` с `Tanh`, а запрос содержит `[A]` с другой
transformation, `[B]` с `Tanh`, другой binding или другой exact Consumer data
identity. Одинаковая tensor width не делает contracts совместимыми.

Transformer возвращает semantic mismatch до upload. Текущая strict policy для
warm start сохраняет требование exact Consumer data digest; её изменение не
является частью этой модели.

## Digest decomposition

### Общие инварианты

Независимо от выбранного варианта:

- canonical full documents остаются источником смысла;
- digest является equality/fencing key, а не заменой document;
- ordered TargetIdentity, constraints и обе transformations покрываются
  model-compatible identity;
- resource identities, kinds, operators, role bindings, weights и parameters
  покрываются objective/model identity;
- component identities присутствуют в canonical objective для references и
  observability; включать ли их буквально в mathematical digest или
  канонизировать graph независимо от локальных labels, пока не решено;
- derived physical target index отдельно не хэшируется: его однозначно задаёт
  ordered target layout;
- source encoding не становится model identity при неизменном logical `x`, но
  входит в exact job/recovery fencing;
- checkpoint digest продолжает идентифицировать physical artifact bytes;
- descriptive labels не должны случайно менять mathematical identity.

### Вариант D1: layered digests

```text
ConsumerDataDigest
TargetContractDigest
ObjectiveDigest
ModelContractDigest
JobConfigHash
CheckpointDigest
```

- Consumer вычисляет exact data semantic digest.
- Transformer вычисляет target и objective digests от validated canonical
  documents.
- Model digest связывает model geometry/configuration, target contract,
  objective-dependent private layout и operator language identity.
- Job hash дополнительно покрывает execution, source encoding и recovery state.

Преимущество — точная mismatch diagnostics. Недостаток — больше public fields и
необходимость исключить противоречия между component digests и composite
identity.

### Вариант D2: data digest и один composite ML digest

Public model identity содержит Consumer data digest и один Transformer-computed
digest canonical model/target/objective document. Component digests могут
вычисляться только для diagnostics.

Преимущество — компактная граница. Недостаток — один digest снова может
смешать разные причины incompatibility и затруднить объяснение mismatch.

### Вариант D3: content-addressed semantic documents

Target, objective и model documents имеют собственные content identities;
composite document ссылается на них по digest.

Преимущество — независимая эволюция и явная композиция. Недостатки — наиболее
сложные canonicalization, reference resolution и lifecycle. Для текущего
масштаба вариант может оказаться избыточным.

До выбора варианта нужно совместно определить:

- кто канонизирует и вычисляет каждый digest;
- какие Consumer metadata являются semantic, а какие descriptive;
- нужен ли Consumer-у component-level mismatch response;
- входят ли ComponentIdentity в mathematical equality или только в
  observability identity;
- какие digests являются public, а какие остаются internal fences.

## Независимое версионирование operator language и Flight

Constraints, transformations, resource kinds, operator role schemas,
parameters, reduction и gradient semantics вместе образуют закрытый
Transformer-owned mathematical language.

Meaning существующей identity не изменяется in place. Изменение formula,
role schema, resource lifecycle или gradient flow требует новой operator,
resource или language identity.

### Вариант V1: lockstep с Flight workflow

Каждое изменение mathematical language выпускает новую Flight version.

Преимущество — один closed-world schema bundle. Недостаток — новый operator
связывает Consumer release с изменением job transport, даже если lifecycle и
Arrow channels не меняются.

### Вариант V2: embedded independently versioned language

Flight переносит self-contained objective с собственной language identity.
Capabilities объявляет поддерживаемые language revisions; exact schema и
semantic validator выбираются по этой identity.

Преимущество — operator language эволюционирует отдельно от Flight workflow.
Недостаток — Transformer может временно поддерживать несколько language
revisions, если compatibility policy этого потребует.

### Вариант V3: stable language envelope и negotiated catalog

Language envelope остаётся стабильным, а capabilities публикует поддержанные
constraints, transformations, resource kinds и operators. Unknown identity
отклоняется semantic validation.

Преимущество — additive primitive не требует новой Flight schema. Недостаток —
полный closed contract больше не выражается одной статической enum schema;
нужны строгие правила negotiation и запрет изменения semantics существующих
identities.

### Вариант V4: hybrid

Immutable language revision задаёт role model и canonicalization, а
capabilities внутри revision объявляет additive implementations. Изменение
types или semantics создаёт новую revision; добавление operator-а с уже
выразимой role schema может быть capability addition.

Преимущество — разделяет type-system evolution и runtime availability.
Недостаток — самая сложная compatibility classification.

Выбор language lifecycle не назначает номер будущей Flight version. Сначала
нужно определить, какие additions Consumer должен переживать без синхронного
transport release.

## Варианты чистого перехода

Все варианты исключают постоянные aliases, dual-write и параллельное создание
jobs по старой и новой semantic model. Активные jobs отсутствуют, а Flight v10
integration заморожена, поэтому clean cut технически доступен.

### T1. Новые models с нуля

- прежние jobs и models не исполняются новым runtime;
- target-named checkpoint layout не конвертируется;
- Consumer переобучает модели;
- новые metrics получают отдельную contract identity;
- старые runtime paths удаляются до release.

Это минимальная постоянная сложность, но weights и сравнимый metrics baseline
нужно получить заново.

### T2. Одноразовая offline conversion без runtime compatibility

- до cutover отдельный инструмент однозначно преобразует target-named weights
  в positional layout и добавляет explicit semantic documents;
- converted artifact получает новую checkpoint/model identity;
- незавершённый optimizer/recovery state не переносится;
- production runtime после cutover понимает только новую модель.

Так сохраняются weights без compatibility layer, но требуется доказать
numerical equivalence и atomic provenance conversion. Ошибка conversion опаснее
переобучения.

### T3. Clean runtime с read-only historical metadata

- новый runtime исполняет только новую semantic model;
- старые persisted model metadata и OpenSearch indices остаются read-only
  history;
- predict, warm start и recovery старых artifacts недоступны;
- historical records не переписываются под новую semantics.

Вариант может сочетаться с T1 или T2 и отдельно определяет retention, а не
runtime compatibility. Должен ли новый `model.describe` читать historical
metadata, решается отдельно: такая поддержка потребует bounded legacy reader и
не возникает автоматически из retention.

### T4. Временный development bridge

- сначала generic internal model проверяется через одноразовый adapter из
  текущего v10 document;
- bridge используется только для equivalence tests и не развёртывается как
  public compatibility layer;
- перед release bridge и target-specific paths удаляются;
- Consumer переключается одним clean cut.

Так можно отделить numerical refactoring от public schema, но временный bridge
должен иметь заранее заданное условие удаления.

Перед выбором transition нужны решения владельца о ценности существующих
weights, historical metrics и возможности повторного обучения. Номер будущих
checkpoint, recovery и metrics formats до этого не назначается.

## Вопросы следующего согласования

1. Достаточны ли `Finite` и `ClosedInterval` для ближайших Consumer profiles?
2. Достаточны ли `Identity`, `Tanh` и `Sigmoid` для ближайших objectives?
3. Достаточен ли один начальный resource kind
   `PositiveScalarPerObservation` для ближайших objectives?
4. Принимается ли правило gradient reachability для каждого trainable private
   resource?
5. Должны ли `ExpectedValue` и `RiskAdjustedExpectedValue` оставаться
   взаимоисключающими, и если да, чем обосновано это language rule?
6. Нужен ли selection score явный ComponentRef или он остаётся фиксированной
   Transformer policy?
7. Какой вариант digest decomposition D1–D3 нужен для diagnostics без
   дублирования источников правды?
8. Какой lifecycle V1–V4 сохраняет закрытый operator language и допускает новый
   target без Transformer release?
9. Какой clean transition T1–T4 допустим для существующих models, checkpoints и
   metrics?
10. Нужны ли дополнительные schema-neutral scenarios до проектирования wire
    format?

После согласования этих вопросов стороны могут подготовить один canonical
semantic document, затем выбрать wire representation. Только после этого
обоснованно назначать Flight, Worker, checkpoint, recovery и metrics versions.
