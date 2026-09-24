# Семантическая модель v4

> ДОКУМЕНТ КОНТРАКТА. Этот каталог определяет закрытый язык target slots,
> objectives, model tuning и семантических идентичностей D1, используемый
> Flight v17 и принадлежащим provider-у runtime Worker v15/checkpoint v9.

Схемы являются авторитетной формой документов. Этот файл определяет
семантические правила, которые нельзя выразить JSON Schema. Ревизия 4 —
чистый переход: документы, digests и compatibility readers предыдущих
семантических ревизий не принимаются.

## Граница

`ModelContract` содержит только заявленное вызывающей стороной намерение
обучения:

```json
{
  "targetContract": {"slots": []},
  "objective": {"directComponents": []},
  "modelTuning": {}
}
```

Вызывающая сторона передаёт непрозрачный digest данных и геометрию tensor-а в
Flight `dataBinding`. Transformer разрешает своё model definition из этой
геометрии,
ревизии закрытого языка, target layout, objective и tuning. Он выпускает
получившийся model-definition digest. Реализация архитектуры, private heads,
parameterization, tensor layout, байты checkpoint-а и политика устройства
остаются собственностью Transformer.

Target identities непрозрачны. Transformer не должен ветвиться по имени
target-а, profile, FIGI или другому значению внешней предметной семантики.

## Targets и objective

`TargetContract.slots` — непустой упорядоченный массив уникальных identities.
Его порядок является физическим порядком target vector и определяет output
width. У каждого slot есть transformation входа loss и transformation
публичного prediction. `observedConstraint` необязателен: отсутствие означает
конечные observations; явный закрытый интервал имеет вид:

```json
{
  "identity": "Opaque.Probability",
  "observedConstraint": {
    "constraint": "ClosedInterval",
    "minimum": 0,
    "maximum": 1
  },
  "lossInputTransformation": "Identity",
  "publicPredictionTransformation": "Sigmoid"
}
```

Поддерживаемые transformations: `Identity`, `Tanh` и `Sigmoid`.
`ClosedInterval` — единственный явный primitive constraint-а. Все JSON numbers
являются конечными значениями IEEE-754 binary64. Arrow-значения `y` и
публичные predictions — конечные Float32; внутренний dtype tensor-а,
принадлежащий Transformer, не предписан.

У каждого target slot ровно один direct component в том же порядке, что и
slots. Direct components задают `identity`, `operator`, `weight` и
`targetIdentity`. Auxiliary components задают именованные roles, специфичные
для их operator-а. Поэтому target roles и resource roles структурно различимы
без универсального discriminator-а references. Resources объявляют
непрозрачную identity и `resourceClass`:

```json
{"identity":"sharedScale","resourceClass":"PositiveScalarPerObservation"}
```

`PositiveScalarPerObservation` — private, положительный, дифференцируемый
scalar на observation. Он принадлежит checkpoint-у, может совместно
использоваться повторением resource identity и никогда не является
координатой публичного prediction. Его реализацией владеет Transformer.
Объявленный resource должен использоваться и иметь структурный путь,
порождающий gradient к total loss; использование только через stop-gradient
недопустимо.

Direct operators: `SmoothL1`, `BinaryCrossEntropyWithLogits`,
`PositiveClassWeightedBinaryCrossEntropyWithLogits` и `LogMSE`.
Auxiliary operators: `GaussianNLL`, `ExpectedValue` и
`RiskAdjustedExpectedValue`. Их formulas, role constraints, constants и
gradient semantics реализуются и валидируются Transformer. Выбор direct
operator-а явно указан в документе; Transformer никогда не выводит его из
target identity. `ExpectedValue` и `RiskAdjustedExpectedValue` могут
сосуществовать.

`PositiveClassWeightedBinaryCrossEntropyWithLogits` — отдельный бинарный
примитив с обязательным `positiveClassWeight > 0`. Он не расширяет и не
меняет прежний `BinaryCrossEntropyWithLogits`: тот по-прежнему допускает
дробные observations в `[0, 1]`. Новый примитив допустим только для точно
`ClosedInterval [0,1]`, `Identity` loss input и `Sigmoid` public prediction.
Его target не может участвовать во вспомогательных ролях; каждое принятое
Arrow Float32 observation должно быть ровно `0` или `1` (`-0.0` равен `0`).

Для raw logit `z`, бинарного наблюдения `y` и `w = positiveClassWeight`:

```text
L(z, y) = -w * y * log(sigmoid(z))
          - (1 - y) * log(1 - sigmoid(z))
```

Вес класса применяется до `GlobalRowMean`; обычный component `weight`
применяется затем в `WeightedSum`. Публичная вероятность такой координаты
всегда выводится как `sigmoid(z - log(w))`. Это вероятность невзвешенного
распределения представленных строк, а не обещание статистической калибровки
за пределами этого набора.

Components имеют строго положительные weights. Direct components идут в
порядке target slots; resources и auxiliary components — в порядке ASCII
identity. Язык фиксирует reduction `GlobalRowMean` и aggregation
`WeightedSum`, поэтому эти избыточные поля не передаются в каждом документе.

## Идентичности D1

После структурной и семантической validation Transformer вычисляет SHA-256
для UTF-8 preimages RFC 8785/JCS:

```text
targetContractSha256 = SHA256(JCS({
  objectiveLanguageRevision: 4,
  targetContract
}))

objectiveSha256 = SHA256(JCS({
  objectiveLanguageRevision: 4,
  objective
}))

```

`dataContractSha256` остаётся переданным вызывающей стороной непрозрачным
значением для Transformer. Полные документы — источник смысла; digests —
производные identities. Напротив, `modelDefinitionSha256` выпускается
provider-ом после разрешения Transformer своей внутренней реализации модели и
`ModelConfig`. Он связывает digests target и objective, но его preimage не
является контрактом внешней стороны и не вычисляется повторно через границу.
Хеширование,
независимое от представления, и конвертация старых semantic revisions в v4
отсутствуют. Input manifests, job configuration hashes и physical checkpoint
hashes — отдельные операционные fences, а не слои D1.

## Validation и capabilities

Порядок validation: закрытая schema; конечные JSON numbers; target layout и
constraints; порядок и уникальность components/resources; direct coverage;
operator roles/domains; resource reachability; допустимость model tuning;
затем вычисление D1. Несовместимые сохранённые или запрошенные definitions не
переназначаются.

Неверная declaration нового primitive (неподходящий target, auxiliary role,
неположительный или не конечный class weight) возвращает structured
`INVALID_OBJECTIVE` с путём в semantic document. Небинарное значение target в
Flight input возвращает `TARGET_VALUE_INVALID` с `expectedDomain: "Binary"`,
target identity/index и номером строки. Worker повторяет эту проверку при
чтении сохранённого Arrow input; повреждённый artifact не интерпретируется как
допустимый binary target.

`language-capabilities.schema.json` объявляет ревизию 4, закрытый язык,
доступные direct operators и семантические limits. Вызывающая сторона может ввести новую непрозрачную
target identity с advertised primitives, не вызывая ветвления Transformer по
имени target-а.
Изменение formula primitive-а, type system, role model или resource lifecycle
требует новой revision.

## Fixtures

`fixtures/` содержит небольшой межпроектный набор проверочных примеров:
случаи одного target-а, нескольких targets с общим resource, нового
непрозрачного target-а и изменённого порядка layout. Manifest fixtures
хеширует только файлы этого набора; он служит для офлайн-проверки, но не
является входом runtime, capability или compatibility fence.
