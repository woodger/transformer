# Восстановление consumer-neutral границы `x/y` в Transformer

> Тип: Design Note. Transformer-side аудит, модель ответственности и варианты
> для совместного обсуждения с Consumer-ами. Это не ADR, не Flight schema, не
> план реализации и не описание принятого изменения.

- Статус: к совместному архитектурному обсуждению
- Срез аудита: 2026-09-05, Transformer Flight v10 / Worker v11
- Входные материалы: Consumer-side аудит Inventory от 2026-09-05

## Статус работ

Дальнейшее развёртывание и интеграция Flight v10 остановлены владельцем до
согласования границы. Активные jobs отсутствуют. Текущая реализация сохраняется
как техническая исходная точка и источник проверяемых фактов, но этот документ
не утверждает её будущим public contract.

`indexedFeatureBlocks` предварительно находится вне спорной target-семантики:
он описывает только физическое представление уже вычисленного `x`. Окончательное
решение о его включении в следующий контракт также не принимается этой
запиской.

Во время подготовки Design Note код, schemas, migrations, public и process
contracts не изменяются.

## Вопрос и критерий границы

Первоначальная роль Transformer формулировалась как сервис, который принимает
числовые `x` и `y`, исполняет Transformer model и функцию потерь и корректно
реализует Arrow Flight transport. Consumer должен владеть предметным смыслом
features и targets.

Flight v10 нарушает эту границу на стороне `y`: публичные schemas и runtime
Transformer знают target identities Inventory и выбирают поведение по их
значениям. Зависимость существует не только в adapter-е, поэтому удаление строк
`inventory.*` или переименование wire fields само по себе проблему не решает.

Рабочий критерий consumer-neutral границы:

> Transformer интерпретирует tensor geometry, математические representations,
> operators и их явные bindings, но не ветвится по значению Consumer-owned
> target identity.

Consumer-neutral не означает domain-free. Transformer продолжает владеть
ML-доменом: model, differentiable transformations, operator semantics,
autograd, optimizer, device runtime, checkpoint и recovery. Нейтральность
относится к предметной target ontology конкретного Consumer-а.

## Что остаётся за границей проблемы

Этот Design Note не предлагает:

- переносить вычисление features или targets в Transformer;
- интерпретировать FIGI, indicators, intervals, horizons или profile;
- передавать исполняемый Consumer-код;
- ослаблять finite, shape, compatibility, recovery или ownership checks;
- выбирать окончательную JSON/Arrow schema;
- назначать номер следующей версии Flight, Worker, checkpoint или metrics;
- выбирать transition strategy для существующих models;
- менять `indexedFeatureBlocks` во время архитектурного анализа.

## Проверенные источники Transformer

Аудит выполнен по текущим источникам:

- общие ML identities — [`app/contracts/ml.py`](../../app/contracts/ml.py);
- Flight schemas, Arrow validation и capabilities —
  [`app/contracts/flight/v10/`](../../app/contracts/flight/v10/);
- inbound semantic mapping —
  [`app/service/adapters/inbound/flight/`](../../app/service/adapters/inbound/flight/);
- Worker objective и process schemas —
  [`app/contracts/worker/v11/`](../../app/contracts/worker/v11/);
- model heads —
  [`app/worker/model/transformer.py`](../../app/worker/model/transformer.py);
- loss execution —
  [`app/worker/training/losses.py`](../../app/worker/training/losses.py);
- checkpoint и recovery —
  [`app/worker/checkpoints/`](../../app/worker/checkpoints/);
- Worker telemetry —
  [`app/worker/telemetry/`](../../app/worker/telemetry/);
- metrics artifacts, projections и OpenSearch templates —
  [`app/contracts/metrics/`](../../app/contracts/metrics/);
- model publication и compatibility —
  [`model_contract.py`](../../app/service/application/services/model_contract.py)
  и
  [`publication.py`](../../app/service/adapters/outbound/artifacts/publication.py).

## Текущий поток target-семантики

```text
Flight create
  targets + objective + inventory.* identities
          |
          v
Flight validation / canonical_targets
          |
          v
mlContract + objectiveConfigSha256 + ModelConfig.out_dim
          |
          v
Worker v11 manifest
          |
          +--> target-named ModuleDict heads
          +--> target-specific transformations
          +--> target-specific direct/auxiliary losses
          |
          v
checkpoint / recovery / model metadata
          |
          v
target-specific telemetry / metrics / plotting
```

`sourceEncoding` и реконструкция `x` идут параллельным техническим путём. Они
используют dimensions и offsets, но не входят в описанную цепочку выбора
target-specific поведения.

## Инвентаризация target-specific зависимостей

### Public Flight

| Место | Текущее поведение | Вид связи |
| --- | --- | --- |
| `common.schema.json` | Закрытый enum из шести target names, максимум `6`, `inventory.learning-dataset` и `inventory.target.v2` как `const` | Public contract непосредственно зависит от Inventory ontology |
| `objective-config.schema.json` | Direct loss ссылается на тот же закрытый target enum | Новый target невозможно выразить без изменения Transformer schema |
| `arrow.py` | Нижняя граница `y` выбирается условием `target == "MeanReturn"` | Числовая validation зависит от предметного имени |
| `coordinator.py` | Capabilities публикует фиксированные identities, maximum width и выведенный каталог direct operators | Provider объявляет Consumer-owned universe |
| JSON fixtures и README | Повторяют target catalog и `inventory.*`; compact fixtures используют `production-core-v2/v6` | Нормативные fixtures создают cross-project release coupling |

Public schemas корректно проверяют width, порядок и exact contracts, но
смешивают структурные правила с предметными значениями identities.

### Service application и persistence

Service сохраняет `dataContract`, `mlContract`, model configuration и request
hash в PostgreSQL. Большая часть ledger-а работает с JSON documents и не
интерпретирует target names. Это благоприятная граница для будущего решения.

Target-specific поведение появляется в следующих местах:

- inbound adapter строит `mlContract` через Worker-owned objective parser;
- input upload извлекает target subset через тот же parser для Arrow
  validation;
- model compatibility сравнивает полные `dataContract` и `mlContract`, а затем
  повторно materialize target-specific objective;
- publication строит checkpoint metadata и содержит отдельный hard-coded
  `inventory.target.v2` в canonical data schema;
- telemetry publication снова разбирает target-specific `mlContract`.

PostgreSQL JSONB columns сами по себе не требуют catalog из шести targets.
Возможная новая digest decomposition или новые обязательные metadata fields
могут потребовать schema evolution, но такое решение пока не принято.

### Worker process contract

Worker v11 повторяет public ограничения:

- target enum и предел `6` находятся в process schemas;
- `ObjectiveConfig` вызывает общий `canonical_targets`;
- `DIRECT_LOSS_OPERATORS` выбирает direct operator по target name;
- validation запрещает operator, отличный от закреплённого за конкретным
  target;
- `GaussianNLL`, `ExpectedValue` и `RiskAdjustedExpectedValue` проверяют
  наличие конкретных names;
- default objective и local defaults materialize все шесть targets;
- `ModelConfig.out_dim` ограничен Consumer-owned максимумом.

Worker contract правильно является defense-in-depth границей, но сейчас он
повторно защищает domain-specific policy, а не только provider-owned numerical
contract.

### Model heads и public prediction

`TradingHead` создаёт `nn.ModuleDict`, ключами которого являются target
identities. Следствия:

- target name выбирает public head;
- target name входит в ключи PyTorch `state_dict`;
- переименование opaque identity меняет physical checkpoint keys даже при
  неизменных shape и operator semantics;
- порядок heads отдельно восстанавливается через canonical target universe.

Две функции выбирают поведение по именам:

- `MeanReturn` получает `tanh` до direct loss и prediction;
- `SigmaReturn` и `VolatilityNext` получают `sigmoid` до direct loss;
- `ProbTP`, `ProbSL` и `HittingProbTP` остаются logits для loss, но проходят
  `sigmoid` для public prediction.

Private `returnScale` создаётся одним boolean `include_return_scale`, который
выводится из наличия domain-bound auxiliary operators. Resource не является
public target, но его архитектура и checkpoint shape зависят от objective.

### Loss execution

Dispatch самих direct primitives уже преимущественно нейтрален:

- `SmoothL1` принимает два tensor-а;
- `BinaryCrossEntropyWithLogits` принимает logit и probability target;
- `LogMSE` принимает положительные operands.

Связь появляется до и после dispatch:

- target index строится по opaque name;
- direct operator разрешается только если совпадает с
  `DIRECT_LOSS_OPERATORS[target]`;
- `GaussianNLL` всегда берёт target и prediction `MeanReturn`;
- `ExpectedValue` и `RiskAdjustedExpectedValue` всегда вычитают prediction
  `ProbSL` из `ProbTP`;
- risk-adjusted operator неявно использует тот же `returnScale`;
- gradient component `GaussianNLL` принудительно объединяется с
  `target:MeanReturn`.

Таким образом, provider-owned numerical implementations пригодны для
нейтрального языка, но current bindings являются domain-specific.

### Checkpoint и recovery

Checkpoint v5 сохраняет:

- target-named `state_dict`;
- `model_config` с `out_dim`;
- полный `data_contract`;
- полный `ml_contract`;
- отдельную копию objective;
- canonical data schema с `inventory.target.v2`.

Validation требует exact текущие `DATA_CONTRACT_ID`, version и
`TARGET_SCHEMA_ID`, а objective parser повторно требует текущий target catalog.
Recovery v5 сохраняет те же contracts, objective digest и полный trainer
state. Model, optimizer и best-selection state используют physical parameter
layout текущей модели.

Это создаёт два разных compatibility вопроса:

1. Можно ли однозначно описать старые weights новой явной slot declaration?
2. Можно ли безопасно восстановить незавершённый старый trainer state после
   изменения model parameter layout?

Первый вопрос потенциально допускает metadata conversion. Второй существенно
строже из-за optimizer state, RNG, selection state и recovery fencing.
Ни возможность conversion, ни необходимость нового checkpoint format здесь не
считаются решёнными.

### Telemetry и diagnostics

Часть Worker telemetry уже использует ordered arrays с `{index, name}` и может
переносить opaque identity. Однако остаются target-specific ограничения:

- `EpochObservations`, target errors и training epoch проходят через
  `canonical_targets`;
- direct loss и target metrics индексируются фиксированными names;
- gradient components получают строковые identities вида `target:<name>` и
  `auxiliary:<operator>`;
- `GaussianNLL` сливается с domain-specific direct component;
- plot заранее перечисляет все возможные `direct.<target>` и
  `target.<target>.<metric>` series.

Строка, построенная конкатенацией, не является надёжной identity objective
component: она неоднозначно смешивает presentation label, operator instance и
target binding.

### Metrics contracts и OpenSearch

Metrics v4 и fit-run v4 содержат:

- закрытые enums тех же шести target names;
- максимальный target index `5` и maximum array length `6`;
- auxiliary operator enum;
- `objectiveConfigSha256`, но не отдельную target/model identity;
- schema identities с префиксом `inventory.metrics.*` при owner
  `transformer`.

Generators повторно вызывают `canonical_targets`, поэтому замена только Flight
schema не позволит публиковать новый opaque target.

OpenSearch mappings в основном уже нейтральны: target name, operator и
component references хранятся как `keyword`, а targets — как bounded arrays,
не как dynamic field names. Основной blocker находится в JSON Schemas,
generators и semantic identities документов. Из этого не следует, что
существующий index/template можно сохранить: compatibility contract metrics и
стратегия исторических данных ещё не выбраны.

### Local CLI

Local fit/predict и `gmark` используют тот же default objective, target list,
Arrow validation и model factory. Хотя local CLI не пересекает Flight
boundary, после нейтрализации он должен либо:

- использовать тот же generic target/objective model как локальный Consumer;
- либо получить явно отдельный локальный preset boundary.

Сохранение скрытого domain-specific default только в local path вернуло бы
второй источник семантики.

## Выводы аудита

1. `x` data plane и `indexedFeatureBlocks` не содержат runtime branching по
   Inventory domain.
2. Связь `y` распределена по public Flight, Worker process contract, model,
   losses, checkpoint, recovery и metrics.
3. Удаление литералов `inventory.*` недостаточно: target names входят в
   numerical dispatch и physical `state_dict` keys.
4. Общие numerical loss primitives можно сохранить как Transformer-owned
   implementations.
5. PostgreSQL storage и OpenSearch mappings ближе к consumer-neutral форме,
   чем versioned schemas и generators.
6. Future design должен различать predict compatibility, weights-only warm
   start compatibility и exact recovery compatibility.

### Карта прямых и производных зависимостей

Таблица фиксирует source-level охват аудита без перечисления каждой копии в
golden fixtures и tests.

| Source | Зависимость |
| --- | --- |
| `app/contracts/ml.py` | `inventory.*`, шесть identities, порядок и maximum width |
| `app/contracts/flight/v10/schemas/common.schema.json` | Те же constants, enum, order и bounds |
| `app/contracts/flight/v10/schemas/objective-config.schema.json` | Target enum внутри direct bindings |
| `app/contracts/flight/v10/arrow.py` | Name-dependent target range validation |
| `app/service/adapters/inbound/flight/coordinator.py` | Fixed target capabilities и direct operator projection |
| `app/service/adapters/inbound/flight/validation.py` | Materialization domain-bound objective и `mlContract` |
| `app/service/adapters/inbound/flight/upload_session.py` | Target-dependent Arrow validation через objective parser |
| `app/contracts/worker/v11/config.py` | Default/max `out_dim` и weights от fixed target width |
| `app/contracts/worker/v11/objective.py` | Direct mapping, auxiliary dependencies, catalog/order validation |
| `app/contracts/worker/v11/schemas/` | Повтор public target/objective enums и width bounds |
| `app/worker/model/transformer.py` | Name-keyed heads и name-dependent loss/public transformations |
| `app/worker/training/losses.py` | Name lookup direct/auxiliary roles и component identities |
| `app/worker/training/factory.py` и `trainer.py` | Private scale shape и prediction path из domain-bound objective |
| `app/worker/training/epoch.py` и `run_config.py` | Fixed defaults и maximum target width |
| `app/worker/data/arrow.py` | Canonical target catalog как default для local dense path |
| `app/worker/checkpoints/model.py` | `inventory.*`, target schema и objective validation в checkpoint v5 |
| `app/worker/checkpoints/recovery.py` | Target schema и domain-bound objective в recovery v5 |
| `app/worker/telemetry/epoch.py` | Target-indexed loss/error records |
| `app/worker/telemetry/epoch_observations.py` | Canonical catalog и name-keyed aggregates |
| `app/worker/telemetry/target_errors.py` | Target list определяет positional error pairs |
| `app/worker/telemetry/plot.py` | Статически перечисленные target series |
| `app/contracts/metrics/v4/` | Target enums, index `0..5`, maximum `6` и auxiliary catalog |
| `app/contracts/metrics/fit_run/v4/` | Fixed targets в summary и OpenSearch documents |
| `app/service/adapters/outbound/artifacts/publication.py` | Hard-coded target schema и exact target/objective metadata |
| `app/service/adapters/outbound/artifacts/telemetry/` | Повторная materialization targets для metrics artifacts |
| `app/local/gmark.py` | Semantic значения генерируются через фиксированные координаты `0`, `4`, `5` |

Dependencies в application records, PostgreSQL mapping и worker plan в
основном переносят уже проверенные documents и не ветвятся по target identity.
Их всё равно потребуется проверить при изменении shape/digest, но они не
являются первичным источником domain coupling.

## Модель пространств одного target slot

Текущий термин `transformation` скрывает несколько разных операций. Для slot
`i` полезно рассматривать следующую модель:

```text
shared representation
        |
        v
raw public head z[i]
        |                         observed y[i]
        |                              |
        v                              v
loss projection L[i](z[i]) ----> direct loss operator
        |
        +-- отдельный путь, не обязанный совпадать
        v
public projection P[i](z[i]) ----> prediction[i]

constraint C[i] проверяет observed y[i]
и, при необходимости, public prediction[i]
```

Требуется различать как минимум:

- `raw head` — необработанный model output;
- `loss input` — представление estimate, получаемое direct operator;
- `observed target` — Consumer-computed `y`;
- `public prediction` — значение, пересекающее Flight boundary;
- `constraint` — объявленное допустимое множество `y`/prediction;
- `operator domain` — математические preconditions реализации Transformer.

Примеры текущего различия:

| Случай | Loss input | Public prediction | Constraint `y` |
| --- | --- | --- | --- |
| Bounded regression | `Tanh(z)` | `Tanh(z)` | closed interval |
| Probability with logits | raw `z` | `Sigmoid(z)` | `[0, 1]` |
| Positive regression | positive projection of `z` | та же projection | non-negative или bounded interval |

Consumer declaration может быть строже operator domain. Transformer должен
проверять совместимость declaration с выбранным operator, но Consumer не может
переопределять математическую семантику operator-а.

Открытым остаётся вопрос, является ли public projection свойством target slot,
direct binding или отдельного prediction contract.

## Варианты generic target-slot declaration

Примеры ниже иллюстративны. Имена fields, точная vocabulary и nesting не
являются предложенной wire schema.

### Вариант S1: полностью явный slot

Каждый ordered slot несёт opaque identity и полное описание трёх пространств:

```json
{
  "identity": "consumer.target.alpha",
  "observed": {
    "dtype": "float32",
    "constraint": {"kind": "ClosedInterval", "minimum": -1, "maximum": 1}
  },
  "lossInput": {"transformation": "Tanh"},
  "prediction": {"transformation": "Tanh"}
}
```

Преимущества:

- нет скрытого target → representation mapping;
- digest покрывает все свойства slot;
- новый target на существующей vocabulary не требует Transformer release.

Риски:

- многословность и повторение;
- нужно строго определить совместимые combinations;
- Consumer видит больше деталей model representation.

### Вариант S2: provider-owned representation kinds

Slot выбирает закрытый математический kind:

```json
{
  "identity": "consumer.target.alpha",
  "representation": "BoundedRegressionTanh",
  "constraint": {"minimum": -1, "maximum": 1}
}
```

Kind определяет loss/public views, но не direct loss или target semantics.

Преимущества:

- компактная schema;
- Transformer может гарантировать допустимую комбинацию views;
- private model details остаются внутри provider-а.

Риски:

- kind легко превращается в скрытый preset catalog;
- изменение семантики kind требует новой identity/version;
- некоторые эксперименты потребуют нового Transformer kind вместо композиции
  существующих primitives.

### Вариант S3: slot хранит только observed contract

Slot задаёт identity и constraint. Loss projection объявляется в direct
binding, а public projection — в отдельном prediction layout:

```json
{
  "targetSlots": [
    {"identity": "consumer.target.alpha", "constraint": "Probability"}
  ],
  "directLosses": [
    {"operator": "BinaryCrossEntropyWithLogits", "slot": 0}
  ],
  "predictionSlots": [
    {"slot": 0, "transformation": "Sigmoid"}
  ]
}
```

Преимущества:

- явно разделены observed, loss и prediction spaces;
- высокая композиционность;
- один slot может иметь разные views для разных operators.

Риски:

- cohesive target representation распределяется между документами;
- сложнее доказать, что каждый slot имеет ровно один public output и direct
  supervision;
- выше риск противоречивых references.

### Вариант S4: Consumer-owned reusable declarations

Consumer передаёт definitions и ссылается на них из ordered slots:

```json
{
  "representations": {
    "probability-logit-v1": {
      "constraint": "Probability",
      "lossInput": "Identity",
      "prediction": "Sigmoid"
    }
  },
  "targetSlots": [
    {"identity": "consumer.target.alpha", "representationRef": "probability-logit-v1"}
  ]
}
```

Преимущества:

- сокращает повторение в больших layouts;
- Consumer может переиспользовать release-owned declarations.

Риски:

- reference resolution и canonicalization становятся частью контракта;
- внешняя ссылка без materialized content создаёт скрытую зависимость;
- Transformer всё равно должен получить canonical definition, а не доверять
  одному имени preset-а.

Варианты допускают комбинации. Например, Consumer-owned catalog может
materialize полностью явный S1 document до Flight boundary.

## Explicit bindings для operators

### Общие свойства binding

Возможная модель binding должна выражать:

- стабильную identity конкретного objective component;
- operator identity и language version;
- явные role → slot/view references;
- weight и operator parameters;
- dependencies на другие components или private resources;
- отсутствие циклов;
- типовую совместимость roles;
- ровно одну direct supervision для каждого public target slot.

Открыты три способа ссылки на slot:

1. физический index;
2. opaque target identity;
3. отдельный локальный slot ID при сохранении index и semantic identity.

Index однозначен для tensor operations, но хуже читается. Opaque identity
удобна для диагностики, но её rename становится структурным изменением.
Отдельный ID снимает часть противоречия, но добавляет ещё одну identity.

### Direct binding

Иллюстративная форма:

```json
{
  "componentId": "direct-0",
  "operator": "SmoothL1",
  "roles": {
    "estimate": {"slot": 0, "view": "loss"},
    "observed": {"slot": 0}
  },
  "weight": 1
}
```

Transformer проверяет role types и реализует `SmoothL1`; значение opaque
identity slot-а не участвует в dispatch.

### Auxiliary binding

Иллюстративные роли:

```text
GaussianNLL
  locationEstimate -> public slot loss/public view
  observedLocation -> y того же slot
  scale            -> Transformer-owned positive private resource

ExpectedValue
  positiveOutcomeProbability -> public view одного slot
  negativeOutcomeProbability -> public view другого slot

RiskAdjustedExpectedValue
  positiveOutcomeProbability
  negativeOutcomeProbability
  uncertaintyScale
  riskPenalty
```

Названия ролей являются частью Transformer-owned operator semantics. Значения
target identities остаются Consumer-owned opaque labels.

Objective graph должен однозначно показать, откуда получено значение каждой
роли. Изменение binding, component ID, weight или operator parameter должно
быть рассмотрено при определении objective identity.

## Operator roles и private resources

Consumer выбирает поддержанный operator и связывает его public/observed roles.
Consumer не должен задавать:

- имя Python class или module path;
- PyTorch layer type;
- размер и расположение private layer;
- имя tensor-а в `state_dict`;
- device placement;
- detach/autograd implementation произвольным кодом.

Transformer владеет:

- формулой operator-а;
- shape и numerical domain его roles;
- необходимостью private resource;
- construction и initialization private parameters;
- gradient flow;
- serialization и recovery private state;
- compatibility влиянием operator semantics.

Рассматриваются три модели private resources.

### Вариант R1: неявный operator-owned resource

Binding `GaussianNLL` с location slot автоматически требует provider-owned
positive scale. Другой operator может ссылаться на результат component-а, не
на physical head.

Плюсы: минимальная утечка model internals. Минусы: sharing и dependency могут
быть недостаточно явными для Consumer-а.

### Вариант R2: абстрактный resource в objective graph

Objective объявляет логический resource с provider-owned kind, например
`PositiveScale`, и несколько components ссылаются на его ID. Transformer сам
решает, как materialize resource.

Плюсы: явное sharing и reproducibility. Минусы: Consumer начинает частично
проектировать внутренний model graph.

### Вариант R3: composite operator

Calibration и risk adjustment объединяются в один provider-owned operator.
Private state полностью скрыт.

Плюсы: сильная инкапсуляция. Минусы: растёт каталог composite operators и
снижается экспериментальная композиционность.

Текущий `detach(returnScale)` влияет на gradient flow и потому не является
implementation detail без последствий. Возможные политики:

- закрепить stop-gradient как семантику существующего operator ID;
- сделать gradient policy закрытым operator parameter;
- выделить другой operator ID для иной gradient semantics.

Решение требует совместного обсуждения: Consumer владеет постановкой
эксперимента, а Transformer — корректной autograd-семантикой.

## Варианты digest и compatibility identity

### Текущее состояние

- `dataContractSha256` предоставляет Consumer; Transformer сохраняет и
  сравнивает его вместе с полным `dataContract`;
- `objectiveConfigSha256` вычисляет Transformer от canonical
  `{targets, objective}`;
- request/config hash включает полный immutable create request;
- manifest hash fencing защищает exact committed input;
- checkpoint SHA-256 идентифицирует физические bytes;
- model integrity дополнительно сравнивает дублированные полные документы.

Target transformations сейчас не являются самостоятельным документом: они
скрыто выводятся из names. Поэтому один objective digest не выражает всю
будущую model-compatible семантику.

### Разные виды совместимости

| Проверка | Минимально значимые свойства |
| --- | --- |
| Predict | exact data semantics, tensor geometry, ordered target layout, representations, model configuration, objective/private model shape |
| Weights-only warm start | действующая strict policy требует те же свойства и exact Consumer data digest |
| Recovery | всё выше плюс training configuration, objective execution semantics, input manifest, source encoding, optimizer/scaler/RNG/progress layout |
| Telemetry correlation | immutable model/job identities и стабильные references на target/objective components |

Один digest может быть удобным fence, но не отменяет различий этих policies.

### Вариант D1: несколько видимых component digests

Возможные логические компоненты:

- Consumer-owned `dataSemanticDigest`;
- Transformer-computed `targetContractDigest` от ordered slots,
  representations и constraints;
- Transformer-computed `objectiveDigest` от operators, bindings и parameters;
- Transformer-computed `modelContractDigest` от tensor geometry, target
  contract, objective-dependent private layout и model configuration;
- `jobConfigHash` для exact execution/recovery configuration;
- physical checkpoint digest.

Плюсы: точная диагностика mismatch и ясное ownership. Минусы: больше fields и
риск противоречий между полными documents и digests.

### Вариант D2: Consumer data digest и один composite ML digest

Transformer хранит Consumer-owned data digest и один canonical digest полного
validated model/target/objective document. Component digests могут
вычисляться только для diagnostics.

Плюсы: компактная public identity. Минусы: mismatch менее объясним и разные
compatibility policies могут снова начать сравнивать один слишком широкий
digest.

### Вариант D3: content-addressed documents

Каждый canonical document имеет собственную identity; composite model contract
ссылается на content digests компонентов. Public API возвращает документы и
их digests.

Плюсы: явная композиция и независимая эволюция. Минусы: сложнее wire model,
canonicalization и lifecycle документов.

Во всех вариантах необходимо определить:

- кто канонизирует каждый документ;
- какой digest authoritative;
- проверяет ли Consumer provider-computed digest независимо;
- входят ли descriptive `id`, `version` и `profile` в semantic identity;
- какая ошибка различает corrupt document и incompatible document;
- остаётся ли full document источником смысла, а digest только equality key.

## Независимое версионирование operator DSL и Flight

Текущий Flight contract непосредственно перечисляет operators в JSON Schema.
При строгой immutable schema добавление enum value меняет public contract,
даже если transport workflow не изменился.

### Вариант V1: lockstep с Flight

Каждое расширение transformations/operators выпускает новую Flight version.

Плюсы: простой closed-world contract и одни golden fixtures. Минусы: высокий
cross-project coordination cost и смешение transport evolution с ML language.

### Вариант V2: отдельно версионируемый embedded DSL

Flight переносит objective document с собственной `languageVersion`.
Capabilities объявляет поддерживаемые versions и operators. Flight envelope и
job lifecycle не меняются при добавлении новой independently versioned
language revision.

Плюсы: transport и mathematics эволюционируют независимо. Минусы: Flight
schema должна валидировать envelope, а exact DSL schema выбирается отдельным
dispatch; одновременно поддерживаемые revisions увеличивают runtime surface.

### Вариант V3: стабильная DSL version с capability-negotiated additions

Wire schema допускает bounded operator identity, а capabilities публикует
поддерживаемый каталог. Неизвестный operator отклоняется semantic validation.

Плюсы: новый operator может быть additive capability. Минусы: contract
перестаёт быть полностью перечисленным одной JSON Schema; требуется строго
определить backward compatibility и запрет изменения семантики существующего
operator ID.

### Вариант V4: schema identities по content digest

Capabilities возвращает поддерживаемые objective schema identities/digests, а
request ссылается на одну из них.

Плюсы: точная negotiation и immutable language documents. Минусы: сложнее
client tooling, discovery и offline validation.

Независимо от варианта semantic change существующего operator-а не может
считаться обычным additive capability. Оно требует новой operator или language
identity и влияет на model/recovery compatibility.

## Влияние возможного изменения

| Область | Потенциальное изменение | Что пока не решено |
| --- | --- | --- |
| Public Flight | Opaque ordered slots, representations, constraints, explicit bindings, новые digests/capabilities | Shape документов, namespace/version, compatibility |
| Arrow input/output | Width следует slot count; target validation следует declarations/operators | Нужна ли target metadata в schema или достаточно job contract |
| Worker contract | Generic objective parser, private resource plan, dynamic target width | Номер версии и допустимо ли переиспользовать DSL schema |
| Model | Positional heads вместо name-based dispatch либо другое neutral mapping | `ModuleList` или keyed structure, naming state, maximum width |
| Losses | Role-based inputs вместо lookup известных names | Точная vocabulary roles и composition rules |
| Checkpoint | Explicit target/model contracts и private resource metadata | Новый format или однозначная conversion v5 |
| Recovery | Exact new job/model/objective identities и parameter layout | Возможна ли recovery старого state; вероятно строже model conversion |
| Model publication | Полный validated contract и digests в PostgreSQL metadata | Какие поля authoritative и какие дублирования удалить |
| PostgreSQL | JSONB способен хранить новые документы | Нужны ли columns/constraints и новая migration |
| Worker telemetry | Stable component IDs и opaque slot metadata | Формат component reference и diagnostics identity |
| Metrics artifacts | Dynamic target arrays, component bindings, model/target digests | Новая metrics version и судьба historical artifacts |
| OpenSearch | Mappings уже принимают arbitrary keyword labels | Новый template/index или совместимое расширение текущего mapping |
| Plotting | Series обнаруживаются из records, а не global target catalog | Поведение при неизвестных operator/component kinds |
| Local CLI | Явный generic objective либо отдельный local preset | Является ли local CLI Consumer-ом того же internal contract |
| Tests/fixtures | Provider-neutral slots и mathematical cases | Где живёт cross-project end-to-end fixture |

## Варианты перехода

Ни один вариант не выбран. Во всех вариантах допустимо сохранить
`indexedFeatureBlocks`, если последующее согласование подтвердит его
consumer-neutral характер.

### Переход T1: чистая граница

- новая public/process/metrics identity;
- старые jobs не возобновляются;
- старые models не используются новым predict/warm start;
- compatibility aliases отсутствуют;
- Consumer переобучает модели.

Преимущества: минимальная постоянная сложность и отсутствие двойной
семантики. Недостатки: потеря runtime-доступа к прежним generations и
необходимость нового baseline metrics.

### Переход T2: одноразовая metadata/checkpoint conversion

- v10 target names однозначно materialize в новый explicit slot contract;
- state keys конвертируются из names в согласованный positional layout;
- converted model получает новую format/contract identity;
- незавершённый recovery state может быть исключён даже при conversion models.

Преимущества: возможно сохранить weights. Недостатки: conversion должен
доказать numerical и structural equivalence; ошибка mapping опаснее
переобучения.

### Переход T3: read-only legacy execution

- новые fit создаются только по consumer-neutral contract;
- отдельный ограниченный legacy path читает существующие v10 models;
- новые objective/targets в legacy path не допускаются.

Преимущества: сохраняется prediction старых models. Недостатки: фактический
compatibility layer, дополнительный attack/test surface и неопределённый срок
удаления.

### Переход T4: поэтапный внутренний refactoring до public cutover

- сначала вводится нейтральная internal target/objective model;
- текущий v10 adapter переводит domain-specific документ в неё;
- после доказательства эквивалентности публикуется новый public contract;
- временный adapter удаляется при cutover.

Преимущества: можно отдельно проверить numerical equivalence. Недостатки:
временная двойная модель и риск превратить bridge в постоянный слой.

### Переход T5: новый contract с сохранением исторической telemetry

Новые jobs/models используют новый contract и metrics identities, но старые
OpenSearch indices остаются read-only historical datasets без conversion.
Этот вариант может сочетаться с T1, T2 или T4 и отдельно решает observability
lifecycle.

## Вопросы Transformer к совместному решению

### Target declaration

1. Какая из моделей S1–S4 лучше отделяет явность от избыточной детализации?
2. Является ли opaque identity частью structural model identity или только
   semantic layout identity, если physical head остаётся на той же позиции?
3. Нужен ли отдельный стабильный slot ID помимо array index и Consumer label?
4. Где должен находиться public prediction projection: в slot, objective или
   отдельном prediction layout?
5. Какой минимальный закрытый набор constraints и transformations покрывает
   подтверждённые сценарии без speculative DSL?
6. Остаётся ли maximum target width provider capability, и чем он
   обосновывается: architecture, resources или implementation limit?

### Operators и private resources

7. Какой способ bindings достаточно читаем для Consumer-а и достаточно строг
   для type validation Transformer?
8. Как ссылаться на slots и objective components в telemetry без строковой
   конкатенации?
9. Какая из моделей R1–R3 сохраняет нужную композиционность auxiliary losses,
   не раскрывая model internals?
10. Является ли текущий stop-gradient uncertainty scale обязательной
    семантикой `RiskAdjustedExpectedValue`?
11. Должен ли selection score ссылаться на component IDs или оставаться
    автоматически определённой суммой direct losses?

### Identities и versioning

12. Какие Consumer metadata входят в data semantic digest, а какие только
    сохраняются для observability?
13. Нужен ли отдельный target contract digest или достаточно composite model
    contract digest?
14. Какие полные documents являются source of truth при наличии нескольких
    digests?
15. Какая mismatch диагностика нужна Consumer-у: один общий digest или
    component-level причины?
16. Может ли operator DSL иметь независимый lifecycle, и какой вариант V1–V4
    соответствует требованию закрытого языка?
17. Какие capability additions считаются backward-compatible, если schemas
    immutable?

### Persistence и переход

18. Должны ли opaque target names оставаться в physical `state_dict` keys или
    checkpoint хранит positional weights отдельно от semantic metadata?
19. Требует ли neutral model layout нового checkpoint/recovery format даже при
    совпадающих tensor shapes?
20. Какой transition вариант T1–T5 допустим для существующих models и metrics?
21. Нужно ли сохранять возможность только описывать старую model generation,
    если исполнять её новым runtime нельзя?
22. Должны ли старые OpenSearch indices оставаться отдельным historical
    contract без переписывания документов?

### Cross-project ownership

23. Где Inventory materialize полную slot declaration: в ML profile, target
    catalog или adapter projection?
24. Кто задаёт direct operator: profile напрямую или Consumer-owned
    materialization policy?
25. Где живут provider-neutral fixtures и где проверяется mapping конкретного
    Inventory profile?
26. Какая сторона вычисляет каждый canonical digest и какая только независимо
    его проверяет?

## Условия завершения архитектурного этапа

Перед проектированием versioned schemas стороны должны согласовать:

- ownership каждого поля target, objective и model contract;
- различие observed, raw, loss и public prediction spaces;
- slot/reference model и private resource boundary;
- compatibility inputs для predict, warm start и recovery;
- digest authority и canonicalization;
- DSL evolution policy;
- transition policy для jobs, models, checkpoints и metrics;
- provider-neutral и Consumer-specific fixture ownership.

Архитектурная граница считается достигнутой, если новый target на уже
поддержанных mathematical primitives не требует behavioral branch или release
Transformer, а exact model/recovery fencing при этом остаётся проверяемым без
понимания предметного значения target identity.

Только после этого может быть подготовлено отдельное schema proposal. Само
принятое решение при прохождении ADR admission фиксируется новым ADR; Design
Note не становится историческим или нормативным справочником системы.
