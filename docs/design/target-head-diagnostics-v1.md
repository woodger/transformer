# Проектная записка: Target Head Diagnostics v1

> Тип: Design Note. Предложение отдельной диагностической проекции в области
> владельца для наблюдения финальной проекции target-а по завершённым эпохам.
> Не определяет текущую реализацию, сетевую схему, хранилище телеметрии или
> математическую семантику модели.

- Статус: согласовано для подготовки канонического пакета
- Срез: 2026-09-24; техническая исходная точка — Semantic v4 и Flight v17
- Инициатива: контролируемые проверки переобучения бинарного target-а с
  различением состояния финальной target head, общего представления, логитов и
  публичных предсказаний

## 1. Проблема

Обычный отчёт [Training Telemetry v4](../../app/contracts/training_telemetry/v4/README.md)
содержит `loss`, MAE/RMSE, `health` optimizer-а и, если они настроены,
gradient interactions на общем представлении. Этого недостаточно, чтобы по завершённым
эпохам различить следующие наблюдаемые состояния:

- финальная аффинная проекция target-а остаётся около смещения,
  соответствующего доле положительного класса;
- к target head приходит или не приходит градиент прямого сигнала обучения;
- параметры target head меняются или остаются неизменными;
- логит меняется, но публичная вероятность после калибровочной поправки имеет
  малую дисперсию;
- input финальной аффинной проекции почти одинаков для всех строк или меняется
  между строками.

Текущие target metrics измеряются во время пакета обучения до optimizer update.
Diagnostics gradient interactions измеряются по общему представлению, а не по
параметрам target head. Финальный checkpoint хранит только одно состояние и не
содержит исторических градиентов head, весов, смещений или логитов после update
предыдущих эпох.

Следовательно, существующие модели нельзя ретроспективно объявить
диагностированными. Для модели без новой настройки корректный результат
будущего запроса — `notConfigured`; проверку нужно выполнить повторно.

## 2. Цель и граница

Предлагается самостоятельная ленивая поверхность в области владельца:

```text
Target Head Diagnostics v1
transformer.target-head-diagnostics.v1.report
```

Она принимает только точный неизменяемый `modelRef`, `requestId`, ограничение
страницы и непрозрачный cursor. Идентичность владельца, идентичность запуска,
путь checkpoint-а и путь набора строк вызывающая система не передаёт.
Transformer выводит их из аутентифицированного субъекта, registry и metadata
опубликованной модели.

Поверхность не является проверкой модели, оценкой вне обучающей выборки,
повторным процессом `predict` или доказательством обобщающей способности. Она
даёт ограниченные скалярные
наблюдения о состоянии модели в ходе того же обучения, которое создало
опубликованную generation.

Flight v17 имеет закрытый набор действий. Новый action активируется Flight v18
вместе с отдельным блоком возможностей. Точные JSON Schemas, preimage курсора,
лимиты страниц, TTL и лимит размера ответа принадлежат последующему каноническому
пакету, а не этой записке.

### 2.1 Явное включение

Наблюдения target head собираются только при явной настройке исполнения
diagnostics в `fit.create`. В этой записке она обозначена как
`diagnostics.targetHead`; точное закрытое представление определит канонический
пакет.

В первой revision эта настройка означает только полный committed artifact.
Отсутствие настройки не меняет обучение и приводит к `notConfigured` при
последующем запросе. Настройка сохраняется в metadata checkpoint-а и доступна
через Model Catalog v5, но не входит в `ModelContract` или semantic digest.

## 3. Неизменяемые свойства

Target Head Diagnostics v1 не меняет:

- Semantic v4, `TargetContract`, `Objective`, `ModelContract`, D1 digests и
  `modelDefinitionSha256`;
- формулу `PositiveClassWeightedBinaryCrossEntropyWithLogits`, binary
  validation и derived correction;
- публичные outputs `predict`, training selection, optimizer, AMP либо порядок
  training operations;
- Arrow data plane, `indexedFeatureBlocks`, committed input receipts и
  reconstruction tensor-а;
- существующие report/gradient-interactions Training Telemetry v4 и Model
  Topology v2.

Новая настройка diagnostics — настройка исполнения fit-задания. Она входит в
`jobConfigSha256` и recovery fence, но не входит в семантическую совместимость
и не влияет на допустимость инициализации опубликованной моделью.

В ответе отсутствуют training rows, feature values, targets конкретных строк,
тензоры параметров, имена Python parameters, checkpoint bytes/paths, device и
детали хранения.

## 4. Состав результата

Успешный ответ содержит:

- точный `modelRef`;
- точный `modelDefinitionSha256` опубликованного поколения модели;
- `producingRunId` для внутренней корреляции телеметрии;
- `sampleIdentity` и `sampleRowCount`;
- одну упорядоченную раскладку пар `targetIdentity` /
  `directComponentIdentity`;
- observations эпох в порядке `epoch ASC`.

Идентичности target/component задаются один раз в раскладке. Массивы эпох не
повторяют непрозрачные идентичности, а содержат значения в порядке slots этой
раскладки. Только прямые компоненты имеют запись target head; вспомогательные
компоненты в эту проекцию не входят.

Для каждой эпохи заголовок дополнительно содержит одно наблюдение `headInput`.
В текущей архитектуре input финальной affine projection общий для всех target
slots, поэтому его повторение в каждой записи target было бы вводящим в
заблуждение.

Концептуально доступный результат имеет вид:

```text
modelRef, modelDefinitionSha256, producingRunId
sample: sampleIdentity, sampleRowCount
layout: ordered (targetIdentity, directComponentIdentity)
epochs:
  epoch, globalStep, headInput
  targetHeads[] in layout order
```

Это намеренно не является сетевой схемой.

## 5. Семантика одного target head

В текущей модели финальная проекция target-а — один аффинный слой с матрицей
`W` и вектором смещений `b`. Для target slot `t` его target head означает
строку `W[t]` и координату `b[t]`. Их имена и отдельные значения наружу не
выдаются.

### 5.1 Логит и публичное предсказание

Для строк фиксированного набора после update `i = 1..N`:

```text
zᵢ = raw output target slot до любой transformation
pᵢ = public prediction для того же zᵢ
```

`rawLogit` и `publicPrediction` публикуют:

```text
rowCount = N
minimum = min(xᵢ)
maximum = max(xᵢ)
mean = (1 / N) × Σxᵢ
standardDeviation = sqrt((1 / N) × Σ(xᵢ - mean)²)
```

Здесь `x` равен соответственно `z` или `p`; `standardDeviation` является
стандартным отклонением генеральной совокупности, а не оценкой по sample.
Все опубликованные числа конечны.

`publicPrediction` обязан вычисляться через ту же принадлежащую Transformer
проекцию, что и `predict`. В частности, для
`PositiveClassWeightedBinaryCrossEntropyWithLogits`:

```text
pᵢ = sigmoid(zᵢ - log(positiveClassWeight))
```

Взвешенный score `sigmoid(zᵢ)` не является публичной вероятностью и не может
выдаваться вместо `pᵢ`. Вызывающая система не применяет correction локально.

### 5.2 Градиент именованного прямого компонента

Пусть `c(t, b)` — вклад именованного прямого компонента target-а `t` для
пакета обучения `b`:

```text
c(t, b) = directComponent.weight × GlobalRowMean(operator(rawLogit, target))
```

`operator` уже включает `positiveClassWeight`, когда выбран
`PositiveClassWeightedBinaryCrossEntropyWithLogits`. Величина не включает
вспомогательные компоненты, другие прямые компоненты или aggregate total loss.

Для каждого пакета до optimizer update Transformer вычисляет:

```text
g(t, b) = ∇c(t, b) по (W[t], b[t])
gradientL2(t, b) = sqrt(||∂c/∂W[t]||₂² + (∂c/∂b[t])²)
```

Градиент получается непосредственно из немасштабированного именованного
компонента до глобального ограничения нормы градиента и до optimizer step.
Поэтому AMP loss scaling не меняет его значение. Он показывает сигнал обучения
конкретного target-а, а не суммарный градиент objective.

В `targetHead` публикуются:

- `gradientL2Mean` — среднее конечных `gradientL2(t, b)`;
- `gradientL2Maximum` — максимум конечных `gradientL2(t, b)`;
- `gradientBatchCount` — число конечных градиентов пакетов, вошедших в обе
  величины;
- `weightL2AfterEpoch = ||W[t]||₂`;
- `biasAfterEpoch = b[t]`.

`Health` обычной Training Telemetry остаётся авторитетным источником общего
числа skipped и non-finite batches. Если конечных observations градиента нет,
канонический пакет должен определить форму с `null`, а не подменять отсутствие
нулём.

### 5.3 Input финальной affine projection

Пусть `hᵢ` — output общего блока, непосредственно подаваемый на final affine
projection, на тех же `N` строках набора после update. Пусть:

```text
h̄ = (1 / N) × Σhᵢ
rowCenteredL2Mean = (1 / N) × Σ||hᵢ - h̄||₂
```

`headInput.rowCenteredL2Mean` — один скаляр для эпохи. Он не раскрывает vector
`h`, его ширину или значения coordinates. Величина отделяет почти неподвижный
output encoder/общего блока от случая, когда representation меняется между
строками, но final affine projection производит почти постоянный логит.

Она не является доказательством, что representation содержит обобщающую
предсказательную информацию. Она различает только наблюдаемую дисперсию input
target head.

## 6. Момент измерения и отсутствие побочных эффектов

Каждое наблюдение эпохи имеет две намеренно разные точки измерения.

```text
пакеты обучения
  ├─ градиенты именованного прямого компонента: до optimizer update
  └─ последний optimizer update эпохи
       └─ eval + no_grad на полном committed artifact
            ├─ дисперсия headInput
            ├─ статистики raw logit
            ├─ статистики public prediction
            └─ W/b после эпохи
```

Проход после update обязан:

1. Выполняться после последнего optimizer update этой эпохи.
2. Переводить модель в `eval()` только на время наблюдения и восстанавливать
   предшествующий режим до возврата.
3. Использовать `torch.no_grad()` и не создавать optimizer/scaler/model state.
4. Читать committed Arrow artifact последовательно, без `PayloadBatcher`,
   shuffle generator или случайного отбора.
5. Не materialize targets повторно и не менять input receipts.
6. Не менять CPU/CUDA RNG trajectory, порядок пакетов обучения, веса модели,
   optimizer state, scaler state или публичные outputs обучаемой модели.

При одинаковых input, seed и deterministic configuration включение diagnostics
должно давать те же итоговые weights модели и значения `predict`. Различаться
могут только `jobConfigSha256`, recovery metadata и diagnostics artifact.

Diagnostics являются необязательной наблюдательной операцией относительно fit:
ошибка сбора не отменяет обучение и не маскирует его terminal outcome. Однако
публичный запрос никогда не выдаёт partial epoch sequence как available report.

Если checkpoint selection публикует weights лучшей, а не последней эпохи,
наблюдение `epoch = bestEpoch` описывает состояние опубликованных weights.
Последнее наблюдение остаётся результатом фактически выполненного последнего
update, а не неявной переоценкой published checkpoint.

## 7. Полный artifact в первой revision

Первая контролируемая проверка не нуждается в выборке. Target Head Diagnostics v1
поддерживает только полный committed artifact:

- `sampleRowCount` равен числу logical rows полного закрытого input;
- `sampleIdentity` стабильно связывает exact committed artifact и revision
  диагностической процедуры;
- все эпохи одного поколения используют один и тот же набор и порядок строк;
- ограничение на число строк объявляется в capabilities Transformer.

Для проверки из 1 479 строк response обязан содержать `sampleRowCount = 1479`.
Если будущей revision потребуется выборка большого artifact, она должна
отдельно определить алгоритм отбора, лимит, identity и recovery behavior.
Нельзя неявно заменить полный artifact выборкой в v1.

## 8. Область владельца, доступность и целостность

Порядок обработки запроса сохраняет принципы Model Catalog и Training
Telemetry:

1. Transformer разрешает `modelRef` через registry в области владельца и
   проверяет сохранённые metadata.
2. Неизвестная, чужая и удалённая модель возвращают один `MODEL_NOT_FOUND`.
3. Модель без opt-in diagnostics configuration возвращает обычный
   `notConfigured`, а не corruption или отсутствие модели.
4. Пока доставка проверенного diagnostics artifact доказуемо продолжается,
   результат может быть `pending`.
5. Настроенная, но не собранная полная последовательность возвращает обычный
   unavailable outcome; partial report не выдаётся.
6. Маркер завершения при пропущенной эпохе, duplicate observation, другом
   `modelRef`, `modelDefinitionSha256`, run или layout означает integrity
   failure.
7. Временная недоступность хранилища Transformer/OpenSearch — `UNAVAILABLE`,
   а не `pending` или terminal unavailable.

Канонический пакет определит точные structured code/reason, cursor errors,
snapshot capacity и лимит размера ответа. Cursor, если он нужен, удерживает
проверенную immutable projection только объявленный TTL и не сохраняет право на
удалённую generation модели.

## 9. Граница хранения

Новые observations нельзя вписывать в закрытый training metrics v8 document:
он описывает pre-update epoch-pass telemetry и является существующим внутренним
artifact Transformer. Смешивание двух моментов измерения сделало бы его смысл
неоднозначным.

Предпочтителен отдельный immutable Target Head Diagnostics artifact/projection
v1 со своим маркером завершения. Он связывается с точными `modelRef`,
`producingRunId`, `modelDefinitionSha256`, job configuration и epoch sequence,
но не становится новым D1 layer.

Публичный запрос проверяет полный непрерывный диапазон `1..completedEpochs`,
layout, конечность чисел и точные identities до выдачи available response.
Internal index names, document IDs, templates, artifact paths и delivery
topology не пересекают публичную границу.

## 10. Версии и переход

Предлагаемая матрица намеренно разделяет semantic model и наблюдаемость:

| Область | Предлагаемое состояние | Причина |
| --- | --- | --- |
| Semantic v4 | Сохраняется | Формула objective и identity модели не меняются. |
| Flight v18 | Новая active revision | Flight v17 закрыт; нужен новый action и capabilities. |
| Target Head Diagnostics v1 | Новый public package | Самостоятельная ленивая поверхность только для чтения. |
| Worker v16 | Новая revision | Worker получает opt-in configuration и записывает observations. |
| Checkpoint/recovery v10 | Новая revision | Checkpoint metadata сохраняет расширенную diagnostics configuration. |
| Model Catalog v5 | Новая revision | Detail раскрывает новую runtime diagnostics configuration. |
| Training Telemetry v4 | Сохраняется | Её epoch-pass semantics и actions не меняются. |
| Metrics v8 | Сохраняется | Новый момент измерения хранится отдельным internal artifact. |
| Model Topology v2 | Сохраняется | Topology уже связывает target/component identities; нового графа не требуется. |

Изменение не требует разрушительного удаления поколений Semantic v4.
Существующие поколения остаются видимыми; при отсутствии новой configuration
их результат Target Head Diagnostics v1 равен `notConfigured`. Transformer не
конвертирует старые checkpoint-и и не создаёт synthetic исторические
observations из финального checkpoint-а.

Конкретная PostgreSQL migration и порядок развёртывания нового internal artifact
выбираются вместе с каноническим пакетом. Они не являются частью этой Design
Note.

## 11. Отклонённые варианты

### Расширить существующий Training Telemetry v4 report

Не подходит: report v4 нормативно описывает pre-optimizer epoch-pass metrics.
Post-update evaluation имеет другой момент измерения, стоимость и lifecycle.
Смешивание сделало бы одну запись эпохи семантически неоднородной.

### Вычислять observations из final published checkpoint

Не подходит: checkpoint содержит только одно состояние — обычно final или
selected best epoch. Он не содержит 200 исторических head gradients, weights,
biases и post-update raw logits.

### Использовать total-loss gradient target head

Не подходит: auxiliary roles и другие components могут добавлять собственные
пути к параметрам. Такой градиент не отвечает на вопрос о supervision
конкретного direct component.

### Выдавать raw tensor или parameter names

Не подходит: это раскрывает implementation details и создаёт неограниченный
response. Для controlled probe достаточно ограниченных scalar aggregates.

### Не собирать head-input dispersion

Не подходит: постоянный raw logit сам по себе не отличает неподвижное общее
представление от случая, когда final affine projection игнорирует меняющийся
input.

### Неявно выбирать sample большого artifact

Не подходит: это меняет состав observations между runs и делает A/B
неоднозначным. Первая revision использует только полный committed artifact.

## 12. Проверка следующего канонического пакета

Канонический пакет должен содержать закрытые schemas, error-detail,
capabilities, negative fixtures, manifest и межъязыковые эталонные responses
как минимум для:

1. `notConfigured` существующей generation без новой diagnostics configuration.
2. Полного artifact из 1 479 строк и 200 непрерывных observations эпох.
3. Двух одинаковых binary runs с `positiveClassWeight = 1` и `28`, где
   `rawLogit` и corrected `publicPrediction` различимы по формуле.
4. Ненулевого градиента named direct component при отсутствии auxiliary roles.
5. Auxiliary component, который не меняет published target-head gradient
   прямого компонента.
6. Неподвижного и меняющегося `headInput.rowCenteredL2Mean`.
7. Изменения `weightL2AfterEpoch` и `biasAfterEpoch` без публикации tensor
   values.
8. Отсутствия конечных gradients с явным `gradientBatchCount` и nullable
   aggregates.
9. Пропущенной или duplicate epoch, неверной model/run/model-definition
   identity и non-finite value как integrity failures.
10. Unknown, foreign и deleted `modelRef`, pending delivery, backend failure,
    cursor expiration и capacity/budget outcomes.
11. Детерминированного regression scenario: включение diagnostics не меняет
    итоговые weights или значения `predict` при тех же inputs, seed и
    deterministic configuration.

До принятия пакета Transformer не меняет runtime, PostgreSQL migrations,
OpenSearch templates, Flight v17 contracts или adapter вызывающей системы.
