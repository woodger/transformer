# ADR 0001: граница job-сервиса Arrow Flight

- Статус: принято
- Дата: 2026-07-18

## Контекст

Inventory и Transformer должны работать на разных физических серверах.
Существующий интерфейс Transformer состоит из двух локальных framed Arrow
протоколов subprocess: `fit-stream` и `predict-stream`. Обучение изменяет
состояние optimizer/model, поэтому сетевой retry после начала execution
небезопасен.

## Решение

Transformer владеет single-instance сервисом Arrow Flight v1, ledger jobs в
PostgreSQL, временным spool filesystem и scheduler worker-ов. Flight RPC
handlers только выполняют authentication и validation, размещают inputs и
изменяют состояние job. Они не выполняют Torch. Worker запускает существующий
CLI с `shell=False` в отдельной группе процессов.

Один успешный DoPut соответствует одному логическому payload Inventory и
сохраняется как один файл Arrow IPC. Границы RecordBatch внутри этого DoPut
являются только transport chunking. Для prediction worker оборачивает каждый
сохранённый файл ровно в один прежний frame с 8-байтовым префиксом длины,
упорядочивая frames по `ordinal`. Для fit worker передаёт CLI долговечный
каталог inputs, чтобы тот мог повторно открывать каждый ordinal на каждой общей
для job epoch.

PostgreSQL является авторитетным источником состояния control plane, API
access tokens и metadata опубликованных моделей. Runtime Arrow payload-ы,
output attempts и logs находятся в `/tmp/transformer`. Storage epoch связывает
эти строки БД с одной generation runtime filesystem. При потере generation все
jobs и их runtime metadata удаляются вместо попытки восстановления без данных.
Прерванные `RUNNING` jobs завершаются с `EXECUTION_INTERRUPTED` и никогда не
запускаются повторно автоматически.

Transformer владеет файлами checkpoint. Только успешный fit атомарно публикует
неизменяемые checkpoint и metadata в постоянном каталоге project `models/`,
записывает его opaque `modelRef` в PostgreSQL и при необходимости продвигает
owner-scoped logical alias. Сетевые запросы никогда не содержат paths
filesystem или произвольные CLI arguments.

API access tokens выпускаются и отзываются через CLI Transformer. Flight-
процесс загружает digests активных tokens в RAM и обновляет cache через
PostgreSQL `LISTEN/NOTIFY`; authentication запроса не обращается к PostgreSQL.
После запуска worker queue также хранится в памяти процесса, а не реализуется
периодическим опросом БД.

TLS и выбор device для job независимы. Явный `cuda` проверяется при create и
start и никогда не заменяется CPU. Plaintext необходимо включать явно.

V1 намеренно является single-instance: нет scheduler нескольких replicas и
общего storage. `DoExchange` и `PollFlightInfo` не используются.

## Последствия

- Retry upload/start безопасны благодаря canonical idempotency records.
- Потерянный response можно повторить без повторного Torch execution.
- Prediction использует один subprocess и однократно загружает модель для всех
  sealed inputs.
- Fit использует долговечный spool как dataset, повторно читаемый между
  epochs: каждая epoch job посещает все непустые inputs по ordinal в рамках
  единых optimizer, loss schedule, checkpoint selector и early-stopping.
- CUDA jobs используют FIFO lane с capacity один; CPU capacity настраивается.
- PostgreSQL является единственным долговечным источником истины; второй
  локальной БД для координации или резервного копирования нет.
- Потеря `/tmp/transformer` делает недействительными все jobs, inputs, output
  tickets и idempotency records, связанные с этой generation runtime. Работа,
  включая долгий fit, начинается заново как новая job.
- Опубликованные модели и API tokens сохраняются при потере runtime. `modelRef`
  появляется только после записи checkpoint в постоянный каталог моделей и
  commit соответствующей transaction PostgreSQL.
- Один процесс владеет каталогом runtime через process-level lock. V1 остаётся
  single-instance, хотя PostgreSQL находится удалённо.
- Python binding server-а PyArrow 24 не может выразить все требуемые gRPC
  statuses или настроить лимит принимаемого server message. Заметка о
  зависимости v1 фиксирует точный mapping и fallback на application quotas.
