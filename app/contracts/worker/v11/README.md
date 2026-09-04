# Контракт процесса Transformer worker v11

Этот каталог содержит нормативный внутренний контракт между сервисом
Transformer и одной короткоживущей попыткой ML worker-а. Контракт не зависит от
публичного Flight-контракта, хотя эта версия использует идентификаторы Arrow
schema из Flight v10.

## Запуск и идентификация

Перед запуском процесса без shell-интерпретации сервис создаёт и надёжно
закрывает один неизменяемый стартовый manifest:

```text
transformer-worker run
  --contract-version=11
  --job-id=<uuid>
  --attempt=<positive-integer>
  --attempt-id=<uuid>
  --manifest=<service-controlled-path>
```

Worker проверяет argv, весь manifest и каждый указанный в нём artifact.
`attemptId` служит внутренним equality fence и не передаётся наружу. При
восстановлении создаются новая attempt и новый `attemptId`; второго внутреннего
fence не существует.

Возможности проверяются через тот же executable:

```text
transformer-worker inspect --contract-version=11
```

## Каналы

| Канал | Контракт |
| --- | --- |
| запуск | неизменяемый `command-manifest.schema.json` |
| новый input и EOF | ограниченный NDJSON `control-message.schema.json` через stdin |
| progress и lifecycle | ограниченный NDJSON `event.schema.json` через stdout |
| объёмные данные | неизменяемые Arrow IPC artifacts, указанные в manifests |
| диагностика | stderr |
| завершение transport-а | код завершения процесса |

Worker никогда не обращается к PostgreSQL и не следит за каталогом. Стартовый
manifest содержит текущий непрерывный snapshot входных данных и
`inputClosed`. Следующие непрерывные inputs и явный EOF сервис передаёт через
control channel. Источником истины остаётся PostgreSQL; новая attempt получает
свежий snapshot, поэтому потеря notification не приводит к потере committed
данных.

Control- и event-сообщения содержат `jobId`, числовой `attempt`, `attemptId` и
строго последовательный номер отдельно для каждого направления. Worker хранит
следующий ordinal и после принятия точного input отправляет `input.ack`.
Повторные сообщения проверяются как точные дубликаты и не приводят к повторному
использованию данных при обучении или прогнозировании. На текущей открытой
границе worker отправляет `input.waiting`; только это событие включает
input-idle timer сервиса.

## Compact input replay

Command manifest содержит закрытый `sourceEncoding` вида
`indexedFeatureBlocks`. Каждый input receipt соответствует
`arrow-manifest.schema.json` и фиксирует `chunks`, `logicalRows`, физические
`nativeRows` по блокам, range/example boundaries, число Arrow batches и
неизменяемый artifact. Эти поля вместе с artifact identity проверяются при
повторном предъявлении receipt в пределах attempt.

Worker memory-map-ит каждый compact Arrow artifact и для каждой top-level
range-chunk строки восстанавливает blocks по локальным
`observationOffsets`. Для каждого sequence position он берёт ровно
`windowRows` native rows, flatten-ит их и соединяет блоки в порядке и позициях
`sourceEncoding`. Dense `[rows, seqLen, featureDim]` создаётся срезами примерно
до 8 MiB; весь развёрнутый dataset в памяти или на диске не материализуется.

Физические Arrow batch, range chunk и payload boundaries не меняют порядок
logical examples, optimizer batches или bounded shuffle windows. После полного
чтения artifact worker сверяет фактические logical/chunk/native counters с
immutable receipt. Input использует только schema IDs
`transformer.indexed-feature-blocks.fit.v1` и
`transformer.indexed-feature-blocks.predict.v1`. Prediction artifact сохраняет
прежний target-aligned format и описывается отдельным
`prediction-manifest.schema.json`; input manifest не может быть принят как
output manifest.

## Семантика потоковой обработки

При fit нулевая epoch читает стартовые inputs и последующие control-сообщения
как единый упорядоченный поток. Границы RecordBatch и payload не являются
границами optimizer batch, shuffle window или epoch. EOF сбрасывает последнее
неполное shuffle window и завершает нулевую epoch. Последующие epochs повторно
читают полный неизменяемый набор входных данных.

До EOF worker не публикует recovery checkpoint нулевой epoch. Если worker
падает при открытом input, новая attempt повторяет эту незавершённую epoch с
начала. После EOF checkpoints остаются snapshots полных global epochs.

Событие `checkpoint` обязательно содержит recovery identity, artifact и
компактный core progress `epoch`, `step`, `loss`. Сервис атомарно
фиксирует checkpoint и этот progress в одной транзакции PostgreSQL. Строка
`training-metrics.schema.json` и checkpoint timings являются необязательной
telemetry, которую сервис сохраняет отдельной best-effort операцией. Сбой
telemetry не отменяет recovery generation и не завершает обучение ошибкой.
Envelope намеренно не отклоняет core event из-за содержимого optional полей;
сервис отдельно проверяет их строгими telemetry schemas перед сохранением.

Epoch telemetry отдельно фиксирует завершённые training batches, фактически
выполненные и пропущенные optimizer updates, AMP overflow и количество
finite/non-finite gradient norms. Mean, max и nearest-rank P95 считаются только
по конечным pre-clip norms; non-finite batch не отравляет статистику остальных
batch-ей. Число `step` сохраняет семантику завершённых training batches.

В worker implementation core результат epoch находится в `training/epoch.py`,
а optional observations — в `telemetry/`. Recovery state не восстанавливает
telemetry и не использует её для optimizer, selection или terminal outcome.

При успешном сборе checkpoint-событие дополнительно содержит monotonic
`checkpointSerializationMs`; сервис измеряет durable copy как
`checkpointPublicationMs` и сохраняет оба значения рядом с epoch interval.
Terminal fit result содержит такое же время сериализации итогового checkpoint.
Эти измерения не меняют training state и используются только при построении
одного итогового run summary после успешного fit.

Command manifest содержит точный `mlContract` с выбранными `targets`,
declarative objective и canonical digest. Для fit worker повторно валидирует
его и проверяет совпадение target width с model config до создания модели.
`training` и `diagnostics` передаются отдельно и не входят в objective identity.
`dataContract` содержит принадлежащий Consumer-у непрозрачный `profile` и
сохраняется целиком в model checkpoint и recovery. Predict и recovery требуют
точного совпадения всего документа; Transformer не нормализует и не
интерпретирует `profile`. Recovery также содержит тот же
`objectiveConfigSha256`; state другого objective не восстанавливается.

Fit manifest всегда содержит закрытый `initialization`. Для `random` это
только `{"kind":"random"}`. Для `publishedModel` сервис передаёт
`parentModelRef`, зафиксированные checkpoint и parent/current data-contract
digests и проверенный service-owned checkpoint artifact. Worker повторно
проверяет artifact digest и checkpoint format, требует точного совпадения model
и ML contracts и равенства parent/current `dataContractSha256`. Различающиеся
digests отклоняются как `MODEL_SCHEMA_MISMATCH`. Границы временного периода не
пересекают worker contract и не участвуют в этой проверке.

Published model является только источником полного `state_dict`. Worker
сначала создаёт новый trainer с optimizer, AMP scaler, RNG, selection и
progress нового job, затем загружает все parent weights. Training state
parent-а не читается. Recovery checkpoint, если он есть, относится уже к
текущему job и восстанавливается после parent weights. Terminal checkpoint и
`checkpointMetadata` сохраняют разрешённый lineage с parent/current
data-contract digests без внутреннего artifact path.

Predict может формировать локальные для attempt outputs по мере поступления
inputs, включая typed-empty outputs. Result manifest публикуется только после
EOF. Сервис публикует все outputs в одной terminal transaction.

Модель имеет только выбранные public target-aligned heads. Их непустое
подмножество сохраняет следующий канонический порядок:

```text
MeanReturn, SigmaReturn, ProbTP, ProbSL, VolatilityNext, HittingProbTP
```

Private `returnScale` создаётся только при зависимости objective и в result не
входит. Probability logits преобразуются через sigmoid до записи Arrow. Worker
проверяет ширину, finite values и диапазоны target-space до terminal result.

Все объявленные direct и auxiliary losses активны с первого optimizer step;
stage schedule отсутствует. Optional gradient-interaction diagnostics
вычисляет sampled gradients относительно общей representation model head и
попадает только в best-effort epoch telemetry.

## Жизненный цикл artifacts и семантика завершения

Worker пишет только внутри workspace своей attempt. Перед упоминанием artifact
в событии или result manifest он закрывает файл и выполняет fsync. Worker не
может публиковать публичный output, recovery generation, model generation или
`modelRef`.

Сервис проверяет каждый принадлежащий ему artifact, надёжно публикует его и
только затем создаёт запись. После сбоя может остаться неуказанный staged file,
но запись PostgreSQL никогда не ссылается на частичный файл.

- для публикации одновременно требуются `completed`, корректный result
  manifest и код завершения `0`;
- код `0` без ровно одного события `completed` является нарушением протокола;
- `completed` с последующим ненулевым кодом является нарушением протокола;
- ненулевой код без корректной безопасной ошибки преобразуется в
  `SUBPROCESS_FAILED`;
- невозможность остановить процесс до установленного сервисом срока
  преобразуется в `SUBPROCESS_HUNG`;
- поздние сообщения отклоняются по `attemptId` и допустимому execution state.
