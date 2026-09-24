# Диагностика выходных головок целей v1

> ДОКУМЕНТ КОНТРАКТА. Этот staged-пакет задаёт отдельную owner-scoped
> проекцию наблюдений выходных головок целей опубликованной модели. Он будет
> активирован Flight v18 после отдельного согласования и не изменяет текущий
> runtime Flight v17.

## Операция и граница

```text
transformer.target-head-diagnostics.v1.report
```

Запрос принимает только `requestId`, точный `modelRef`, размер страницы и
непрозрачный `cursor`. Владелец выводится из аутентифицированного субъекта.
Неизвестный, чужой и удалённый `modelRef` возвращают одинаковый
`MODEL_NOT_FOUND`. Сохранённый артефакт диагностики не делает удалённую модель
видимой.

Проекция предназначена для диагностики процесса обучения. Она не является
повторной оценкой опубликованного checkpoint-а, проверкой его целостности,
оценкой качества вне обучающей выборки или заменой `predict`. Ответ не
раскрывает строки обучения, значения признаков и целей, тензоры параметров,
имена параметров, содержимое checkpoint-а, пути файлов, устройства или
топологию хранения.

Worker заканчивает обучение до создания `modelRef`. Поэтому его внутренний
артефакт v16 связывается с job, attempt, committed input manifest, job
configuration и model definition. При публикации сервис выдаёт `modelRef` и
связывает этот уже проверенный артефакт с generation в registry. Только эта
service-side связь попадает в публичный report вместе с `producingRunId`.

## Явная настройка

Наблюдения существуют только для fit-задания с runtime-настройкой:

```json
{"targetHead":"fullCommittedArtifact"}
```

Она является значением `diagnostics.targetHead` в Flight v18 и сохраняется в
metadata checkpoint/recovery v10. Значение означает последовательный проход
по полному закрытому committed input artifact после каждой эпохи. Настройка
входит в `jobConfigSha256` и recovery fence, но не входит в Semantic v4,
`modelDefinitionSha256`, D1 digests или совместимость warm start.

Для модели без такой настройки результат `report` — обычный
`notConfigured / TARGET_HEAD_DIAGNOSTICS_NOT_CONFIGURED`. Transformer не
создаёт исторические наблюдения из финального checkpoint-а.

## Наблюдение эпохи

`layout` один раз связывает каждую запись с точными `targetIdentity`,
`targetIndex` и `directComponentIdentity`. Все `targetHeads` в наблюдении
эпохи идут строго в этом порядке. Вспомогательные компоненты не входят в
layout.

Перед публикацией `available` Transformer проверяет, что layout соответствует
прямым компонентам и ordered target slots checkpoint-owned `ModelContract`.
Каждый `targetIndex` уникален, а в `targetHeads` ровно столько элементов,
сколько в layout. Epoch образуют плотный диапазон `1..completedEpochs`; их
`globalStep` строго возрастает. В каждой distribution `rowCount` равен
`sampleRowCount`, `minimum <= mean <= maximum`, а все опубликованные числа
конечны. При ненулевом
`gradientBatchCount` выполняется `gradientL2Mean <= gradientL2Maximum`.

Для каждого target slot `t` финальная аффинная головка состоит из строки
`W[t]` и смещения `b[t]`. В состоянии после последнего шага оптимизатора
эпохи для всех строк полного артефакта публикуются агрегаты `rawLogit` и
`publicPrediction`:

```text
rowCount = N
minimum = min(xᵢ)
maximum = max(xᵢ)
mean = (1 / N) × Σxᵢ
standardDeviation = sqrt((1 / N) × Σ(xᵢ - mean)²)
```

`rawLogit` — значение до любого публичного преобразования.
`publicPrediction` вычисляется ровно через checkpoint-owned определение
предсказания. Для
`PositiveClassWeightedBinaryCrossEntropyWithLogits` это:

```text
publicPrediction = sigmoid(rawLogit - log(positiveClassWeight))
```

Взвешенный score `sigmoid(rawLogit)` не выдаётся вместо публичной вероятности.

Перед каждым шагом оптимизатора для пакета `b` измеряется градиент только
именованного прямого компонента:

```text
c(t, b) = directComponent.weight × GlobalRowMean(operator(rawLogit, target))
gradientL2(t, b) = sqrt(||∂c/∂W[t]||₂² + (∂c/∂b[t])²)
```

Величина не включает вспомогательные компоненты, другие прямые компоненты или
суммарную функцию потерь. Она берётся из немасштабированного компонента до
глобального ограничения нормы градиента и до шага оптимизатора. Поэтому
масштабирование AMP не меняет её значение. `gradientBatchCount` — число
конечных пакетов, вошедших в `gradientL2Mean` и `gradientL2Maximum`. При нём,
равном нулю, оба агрегата равны `null`; общий счётчик нечисленных и
пропущенных пакетов остаётся в Training Telemetry v4.

`weightL2AfterEpoch` и `biasAfterEpoch` принадлежат той же головке после
последнего шага оптимизатора.

`headInput.rowCenteredL2Mean` измеряется на том же наборе после обновления и
является общим для всех target slots:

```text
h̄ = (1 / N) × Σhᵢ
rowCenteredL2Mean = (1 / N) × Σ||hᵢ - h̄||₂
```

Здесь `hᵢ` — вход финальной аффинной проекции. Вектор `h` и его ширина наружу
не выдаются.

`sampleIdentity` — provider-issued opaque identity полного committed artifact
и revision этой диагностической процедуры. Он стабилен для всех epochs одного
report и меняется при изменении artifact либо процедуры. Это не D1 digest и
не требует локального вычисления вызывающей системой.

## Наблюдательный проход

После последнего шага оптимизатора каждой эпохи Worker:

1. временно переводит модель в `eval()`;
2. выполняет последовательный проход по полному committed artifact в
   `torch.no_grad()` без `PayloadBatcher`, перестановки и случайной выборки;
3. собирает post-update агрегаты и восстанавливает прежний режим модели.

Проход не создаёт gradient, не меняет веса, состояние optimizer/scaler,
состояние RNG, порядок пакетов обучения, input receipts или публичные значения
`predict`. При одинаковых входных данных, seed и детерминированной
configuration включение диагностики должно сохранять итоговые веса и
предсказания; отличаться могут только `jobConfigSha256`, recovery metadata и
артефакт диагностики.

Если selection публикует weights лучшей, а не последней epoch, observation
`epoch = bestEpoch` описывает состояние опубликованных weights. Последняя
observation по-прежнему описывает фактически выполненный последний update и
не является неявной переоценкой published checkpoint-а.

Ошибка best-effort сбора не отменяет fit и не маскирует его терминальный
исход. Однако публичный `available` report выдаётся только после проверки
непрерывного диапазона `1..completedEpochs`, layout, identities и конечности
всех чисел; частичные наблюдения не выдаются как доступный отчёт.

Полный committed artifact не может молча заменяться выборкой. Его число строк
не превышает `maxCommittedArtifactRows` из capabilities. Если этот предел
превышен, fit завершается по своей обычной семантике без complete diagnostics
artifact, а последующий query возвращает `unavailable / NO_COMPLETE_REPORT`.

## Доступность, курсоры и ошибки

Порядок outcomes:

1. owner-scoped lookup и проверка metadata модели;
2. `notConfigured`, если `diagnostics.targetHead` отсутствует;
3. `pending`, только пока materialization полного артефакта доказуемо
   продолжается;
4. `unavailable`, когда полный отчёт не появится;
5. structured error целостности при противоречивом completion marker или
   observations;
6. `available` только для полной проверенной проекции.

Страницы содержат не более 100 эпох. Курсор связан с владельцем, моделью,
запуском и размером страницы; continuation повторяет точные `modelRef` и
`pageSize`; TTL составляет 900 секунд. Страницы образуют последовательные
неперекрывающиеся срезы epoch ASC. Не terminal страницы
удерживают проверенную неизменяемую проекцию в ограниченном локальном cache:
64 snapshots, 16 MiB на snapshot и 64 MiB суммарно. Неистёкшие snapshots не
вытесняются; terminal response не требует admission. После перезапуска
сервиса действующий cursor предыдущего процесса возвращает
`TARGET_HEAD_DIAGNOSTICS_CURSOR_INVALIDATED`; удаление модели имеет приоритет
над cursor.

Размер snapshot — длина UTF-8 представления удерживаемой проекции после RFC
8785 JCS canonicalization. Admission выполняется атомарно до выдачи cursor;
повторное использование идентичной удерживаемой проекции не расходует
capacity повторно.

Cursor содержит случайную identity успешного запуска только в подписанном
payload и не использует deployment или commit identity. При проверке сначала
проверяется owner-scoped model, затем подпись и binding, затем expiry, и лишь
после этого identity запуска. Поэтому уже истёкший cursor предыдущего запуска
даёт `TARGET_HEAD_DIAGNOSTICS_CURSOR_EXPIRED`, а не invalidated outcome.

Максимальный сериализованный ответ — 8 MiB. Точные structured outcomes
определены `schemas/error-detail.schema.json`; вызывающая система ветвится по
`code` и `reason`, а не по тексту.

| Ситуация | Outcome |
| --- | --- |
| Diagnostics не запрашивалась для generation | `notConfigured / TARGET_HEAD_DIAGNOSTICS_NOT_CONFIGURED` |
| Завершение materialization доказуемо ожидается | `pending / MATERIALIZATION_PENDING` |
| Полный report не появится | `unavailable / NO_COMPLETE_REPORT` |
| Retention больше не содержит artifact | `unavailable / RETENTION_EXPIRED` |
| Формат artifact не поддержан | `unavailable / FORMAT_UNSUPPORTED` |
| Неполные или противоречивые observations | `TARGET_HEAD_DIAGNOSTICS_CORRUPT / TARGET_HEAD_DIAGNOSTICS_INTEGRITY_FAILED` |
| Временная недоступность backend-а | `UNAVAILABLE / TARGET_HEAD_DIAGNOSTICS_BACKEND_UNAVAILABLE` |

## Связанные staged-версии

Пакет предполагает следующие новые, пока неактивные версии:

| Область | Версия | Назначение |
| --- | --- | --- |
| Flight | v18 | Новый action и capability diagnostics. |
| Worker | v16 | Сбор и публикация внутреннего артефакта наблюдений. |
| Checkpoint/recovery | v10 | Сохранение расширенной runtime-настройки diagnostics. |
| Model Catalog | v5 | Раскрытие этой настройки в detail. |

Semantic v4, Training Telemetry v4, Metrics v8 и Model Topology v2 остаются
без изменения. Existing published generations не требуют destructive migration:
их результат новой поверхности равен `notConfigured`.

## Fixtures

`fixtures/manifest.json` хеширует только офлайн-набор fixtures. Он не является
runtime fence, не входит в semantic identity и не участвует в `predict` или
warm-start compatibility. Fixtures `w1` и `w28` содержат по 200 непрерывных
observations; отдельный `w28-calibrated-probability` фиксирует
`sigmoid(rawLogit - log(28))` на трёх симметричных raw logit values.
