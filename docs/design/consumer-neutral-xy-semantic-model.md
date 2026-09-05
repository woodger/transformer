# Schema-neutral semantic model границы `x/y`

> Тип: Design Note. Transformer-side предложение математической модели для
> совместного обсуждения с Consumer-ами. Это не ADR, не Flight или Worker
> schema, не план миграции и не описание реализованного поведения.

- Статус: schema-neutral semantic model согласована сторонами; wire design не начат
- Срез: 2026-09-05, техническая исходная точка Flight v10 / Worker v11
- Основание: [аудит consumer-neutral границы](./consumer-neutral-xy-boundary.md)
- Входные материалы: два Consumer-side аудита, review semantic model и ответы
  владельца от 2026-09-05

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

Значения `y` и public predictions на Arrow boundary обязаны быть finite
Float32. Raw model coordinates, loss estimates и private resources должны быть
finite согласно operator semantics, но их внутренний dtype остаётся
Transformer-owned implementation detail. В частности, semantic model не
запрещает AMP и не фиксирует dtype внутренних tensor operations.

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
для telemetry и diagnostics. Semantic references на components внутри
начального objective не вводятся.

Отдельный stable slot ID не вводится. При текущем invariant уникальности target
identity он не добавляет разрешающей способности. Если появится сценарий двух
slots с одной semantic identity, он должен быть рассмотрен отдельно до
ослабления уникальности.

Переименование target identity или перестановка ordered slots меняет target
contract и делает прежнюю model generation несовместимой. Transformer не
выполняет неявный remapping weights или prediction coordinates.

## Минимальный закрытый язык constraints

Согласованный начальный каталог содержит только два kinds:

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

Согласованный начальный каталог содержит три element-wise differentiable
transformations:

| Transformation | Логическая семантика | Codomain для finite raw value |
| --- | --- | --- |
| `Identity` | `f(z) = z` | finite numeric value |
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

Типовая модель не запрещает два instances одного operator. `ExpectedValue` и
`RiskAdjustedExpectedValue` также могут одновременно присутствовать как два
независимых components с собственными identities и weights. Их сумма
математически определена, поэтому скрытое взаимоисключение Flight v10 не
переносится в начальный language invariant.

### Checkpoint selection

Selection остаётся Transformer-owned training policy и не становится частью
Consumer-owned objective. Начальная generic semantics сохраняет действующее
правило:

```text
selectionScore = sum(
  directComponent.weight * epochMean(directComponent.loss)
)
```

Auxiliary components в selection score не входят. Явный ComponentRef для
selection не вводится. Selection configuration и state участвуют в exact job
и recovery fencing, но не меняют mathematical objective digest.

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
- каждый trainable resource имеет структурный путь к total loss через активный
  component с положительным weight и role, operator semantics которой
  передаёт gradient в resource;
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
   - каждый trainable resource структурно достижим из total loss через
     gradient-producing role активного component с положительным weight;
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
   - raw, transformed и resource values finite согласно operator semantics;
   - objective исполняется по resolved indices;
   - наружу выходят только ordered public predictions;
   - private resources не попадают в Arrow output.

Любая semantic несовместимость должна обнаруживаться до durable input upload.
Corrupt persisted metadata и несовместимый корректный contract остаются
разными классами ошибок.

Gradient reachability является структурной проверкой graph. Она подтверждает
возможность прохождения gradient согласно operator semantics, но не требует
ненулевого численного gradient на каждом batch.

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

### Отрицательные подслучаи

Основные сценарии дополняются следующими обязательными проверками:

| Подслучай | Ожидаемый результат |
| --- | --- |
| Private resource имеет только stop-gradient consumers | Objective отклоняется: resource не имеет структурного gradient-producing path к total loss |
| Slot с `ClosedInterval(0, 1)` использует public `Identity` | Declaration отклоняется: `Identity` не доказывает bounded public codomain |
| Два resources имеют один kind, но разные ResourceIdentity | Они materialize как два независимых private model outputs; совпадение kind не означает sharing |
| Objective содержит `ExpectedValue` и `RiskAdjustedExpectedValue` | Оба components принимаются и независимо входят в `WeightedSum`, если все их roles валидны |
| Изменена loss/public transformation | Меняется TargetContractDigest; predict и warm start получают target-layer mismatch до upload |
| Изменён ResourceRef в operator binding | Меняется ObjectiveDigest и вслед за ним ModelContractDigest; возвращается objective-layer mismatch до upload |

Одинаковый ResourceKind с разными identities может привести и к последующей
ошибке reachability, если один из созданных ресурсов используется только через
stop-gradient role. Это отдельная graph validation, а не правило sharing.

## Digest decomposition

### Согласованная модель D1: layered digests

```text
ConsumerDataDigest
TargetContractDigest
ObjectiveDigest
ModelContractDigest
JobConfigHash
CheckpointDigest
```

Canonical full documents остаются единственными источниками смысла. Все
digests являются производными equality/fencing keys:

- `ConsumerDataDigest` вычисляет Consumer от принадлежащего ему exact data
  semantic document;
- `TargetContractDigest` вычисляет Transformer после validation ordered slots;
  он покрывает TargetIdentity, порядок, constraints и обе transformations;
- `ObjectiveDigest` вычисляет Transformer от validated resources и components;
  он покрывает ComponentIdentity, ResourceIdentity, resource kinds, operators,
  role bindings, weights, parameters, aggregation и reduction;
- `ModelContractDigest` вычисляет Transformer; он связывает model geometry и
  configuration, language identity, target и objective digests и abstract
  private resource declarations;
- `JobConfigHash` дополнительно покрывает execution policy, source encoding и
  exact recovery configuration;
- `CheckpointDigest` идентифицирует physical artifact bytes.

ComponentIdentity и ResourceIdentity являются contract identities, а не
descriptive labels. Они входят в `ObjectiveDigest`. Отдельные public digests
каждого component не вводятся.

Derived physical target index отдельно не хэшируется: его однозначно задаёт
ordered target layout. Source encoding не становится model identity при
неизменном logical `x`, но входит в job/recovery fencing. Descriptive metadata
не должно случайно менять mathematical identities.

Mismatch diagnostics должна различать как минимум data, target, objective и
model layers. Точное wire placement digests и canonicalization bytes будут
определены на следующем этапе.

### Рассмотренные альтернативы

D2 с одним composite ML digest не выбран, потому что снова смешивает причины
несовместимости. D3 с content-addressed semantic documents не выбран из-за
лишней для текущего масштаба сложности references и lifecycle.

## Lifecycle operator language и Flight

Constraints, transformations, resource kinds, operator role schemas,
parameters, reduction и gradient semantics вместе образуют закрытый
Transformer-owned mathematical language.

Meaning существующей identity не изменяется in place.

### Согласованная модель V4: hybrid lifecycle

- immutable language revision задаёт type system, canonicalization, operator
  role schemas, resource lifecycle и gradient semantics;
- capabilities перечисляет фактически доступные primitives внутри revision;
- новый target на уже доступных primitives не требует Transformer release;
- новая implementation operator-а с уже выражаемыми types и roles может быть
  capability addition: она требует установки поддерживающего Transformer
  release, но не новой language revision или Flight workflow version;
- изменение type system, role schema, resource lifecycle, formula или gradient
  semantics создаёт новую language revision либо новую immutable primitive
  identity;
- Flight workflow не версионируется синхронно с каждым новым primitive;
- clean cut не требует параллельной поддержки нескольких language revisions.

Capabilities описывает наличие immutable semantics, а не переопределяет её.
Unknown или недоступный primitive отклоняется до upload.

### Рассмотренные альтернативы

V1 lockstep не выбран из-за связывания mathematical evolution с transport.
V2 с обязательной поддержкой нескольких embedded revisions не нужен при clean
cut. V3 с одним stable envelope недостаточно явно отделяет additive
implementation от изменения type system. V4 сохраняет закрытый язык и при этом
разделяет language identity и runtime availability.

Это решение не назначает номер будущей Flight version и не задаёт wire shape
capabilities.

## Согласованный чистый переход

Выбран T1 с переобучением models:

- новый production runtime принимает только новую semantic model;
- постоянные aliases, dual-write и параллельная поддержка старого contract
  отсутствуют;
- существующие modelRef, checkpoints и связанная runtime metadata удаляются;
- старые models не используются для predict, warm start или recovery;
- offline conversion weights не создаётся;
- historical metrics и metadata не сохраняются как read-only contract и также
  удаляются при переходе;
- Consumer обучает новые model generations и получает новый metrics baseline.

T2 с offline conversion и T3 с legacy reader не выбраны. T4 допустим только как
временный локальный development bridge для equivalence tests. Он не
развёртывается, не становится public compatibility layer и удаляется вместе с
target-specific paths до release.

Эта запись определяет transition policy, но не выполняет удаление. Точный scope,
порядок, transactional boundaries и operational verification удаления должны
быть определены вместе с будущим implementation plan.

Активные jobs отсутствуют, а Flight v10 integration заморожена. Номер будущих
checkpoint, recovery и metrics formats до wire design не назначается.

## Итоговая матрица согласования

| Область | Итог |
| --- | --- |
| Materialization | Inventory materializer передаёт self-contained TargetLayout и Objective без внешнего lookup |
| Target references | Semantic binding использует opaque TargetIdentity; physical index выводится после validation |
| Slot model | Observed constraint, loss-input transformation и public transformation объявлены независимо |
| Constraints | Начальный закрытый набор: `Finite`, `ClosedInterval` |
| Transformations | Начальный закрытый набор: `Identity`, `Tanh`, `Sigmoid` |
| Direct и auxiliary roles | Typed roles согласованы; dispatch по Consumer-owned target name запрещён |
| Private resources | ResourceIdentity + Transformer-owned ResourceKind; начальный kind `PositiveScalarPerObservation` |
| Resource sharing | Только повторная ссылка на одну ResourceIdentity означает sharing |
| Dtype boundary | Arrow `y` и prediction — finite Float32; внутренние dtypes принадлежат Transformer |
| Gradient reachability | Проверяется структурный путь trainable resource к total loss, а не ненулевой gradient каждого batch |
| EV operators | `ExpectedValue` и `RiskAdjustedExpectedValue` могут быть независимыми components одного objective |
| Selection | Transformer-owned weighted sum epoch-mean direct losses; auxiliary losses и ComponentRef не участвуют |
| Digests | D1 layered decomposition; canonical documents остаются источниками смысла |
| Language lifecycle | V4 hybrid: immutable revision и capability-advertised primitives |
| Scenarios | Шесть основных и шесть отрицательных подслучаев согласованы |
| Transition | T1 clean cut; models, checkpoints, runtime metadata и historical metrics удаляются; T4 возможен только локально и временно |
| Transformer internals | Layers, parameterization, internal tensor layout, checkpoint storage и autograd implementation не входят в совместное решение |

Schema-neutral semantic model не имеет оставшихся межпроектных блокеров.

## Следующий этап

Предложение с точными структурами, правилами canonicalization, D1 digests,
capabilities, error model и fixtures вынесено в
[canonical contract proposal](./consumer-neutral-xy-canonical-contract.md).

До изменения кода стороны должны согласовать в этом proposal:

1. canonical target, objective и model documents;
2. точный canonicalization algorithm и покрытие каждого D1 digest;
3. wire representation language revision и capabilities;
4. mismatch error model для data, target, objective и model layers;
5. влияние на checkpoint, recovery, metrics и OpenSearch contracts;
6. окончательную судьбу `indexedFeatureBlocks` в новом contract;
7. безопасную operational procedure удаления прежних artifacts.

Только после согласования этого design можно назначать Flight, Worker,
checkpoint, recovery и metrics versions и переходить к реализации.
