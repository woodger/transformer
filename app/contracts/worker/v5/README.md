# Контракт процесса Transformer worker v5

Этот каталог содержит нормативный внутренний контракт между сервисом
Transformer и одной короткоживущей попыткой ML worker-а. Контракт не зависит от
публичного Flight-контракта, хотя эта версия использует идентификаторы Arrow
schema из Flight v4.

## Запуск и идентификация

Перед запуском процесса без shell-интерпретации сервис создаёт и надёжно
закрывает один неизменяемый стартовый manifest:

```text
transformer-worker run
  --contract-version=5
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
transformer-worker inspect --contract-version=5
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

## Семантика потоковой обработки

При fit нулевая epoch читает стартовые inputs и последующие control-сообщения
как единый упорядоченный поток. Границы RecordBatch и payload не являются
границами optimizer batch, shuffle window или epoch. EOF сбрасывает последнее
неполное shuffle window и завершает нулевую epoch. Последующие epochs повторно
читают полный неизменяемый набор входных данных.

До EOF worker не публикует recovery checkpoint нулевой epoch. Если worker
падает при открытом input, новая attempt повторяет эту незавершённую epoch с
начала. После EOF checkpoints остаются snapshots полных global epochs.

Событие `checkpoint` атомарно связывает recovery generation с полной строкой
`training-metrics.schema.json`. Сервис сохраняет checkpoint и метрику одной
транзакцией PostgreSQL. Поэтому восстановленная attempt не теряет метрики уже
зафиксированных эпох и не может заменить их другими значениями.

Каждое checkpoint-событие дополнительно содержит monotonic
`checkpointSerializationMs`; сервис измеряет durable copy как
`checkpointPublicationMs` и сохраняет оба значения рядом с epoch interval.
Terminal fit result содержит такое же время сериализации итогового checkpoint.
Эти измерения не меняют training state и используются только при построении
одного итогового run summary после успешного fit.

Command manifest содержит точный `mlContract`. Для fit worker повторно
вычисляет его из `training` и отклоняет несовпадение до создания модели.
Recovery содержит тот же `objectiveConfigSha256`; state другого objective не
восстанавливается.

Predict может формировать локальные для attempt outputs по мере поступления
inputs, включая typed-empty outputs. Result manifest публикуется только после
EOF. Сервис публикует все outputs в одной terminal transaction.

Модель имеет шесть public target-aligned heads и одну private uncertainty head.
Result Arrow artifact содержит только:

```text
meanReturn, sigmaReturn, probTP, probSL, volatilityNext, hittingProbTP
```

Probability logits преобразуются через sigmoid до записи Arrow. Worker
проверяет finite values и диапазоны target-space до terminal result.

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
