# Сервис Transformer Arrow Flight: операционное руководство v9

> Тип: операционное руководство. Запуск, recovery, shutdown и диагностика
> текущего Flight service.

Это руководство описывает единственный экземпляр сервиса Transformer Flight.
Детали wire-контракта для Consumer находятся в
[`руководстве по интеграции Consumer-ов`](../consumer-flight-integration.md), а
нормативные schemas и fixtures — в
[`app/contracts/flight/v9`](../../app/contracts/flight/v9/README.md). Текущие
process и data ownership boundaries описывает
[`архитектурный справочник`](../architecture.md), training и recovery —
[`training reference`](../training-runtime.md), а credential model, cache
consistency и channel security —
[`справочник аутентификации`](../authentication.md).
Rationale durable recovery и streaming lifecycle сохранён в
[ADR 0003](../adr/0003-durable-resumable-training-and-device-aware-execution.md)
и [ADR 0005](../adr/0005-durable-streaming-flight-v3.md); текущую операционную
семантику определяет это руководство.

## Требования к runtime

- Linux с `/usr/bin/python3`; подходящую версию system Python обеспечивает
  владелец deployment-среды, а зависимости приложения находятся только в
  project `.venv`.
- PyTorch, NumPy и PyArrow для обучения и Flight.
- SQLAlchemy 2, Psycopg 3, Alembic и python-dotenv для доступа к PostgreSQL.
- PostgreSQL, доступный в частной сети.
- Постоянный каталог project `models/` для успешно опубликованных моделей.
- Постоянный каталог project `recovery/` для fit inputs и внутренних
  checkpoints global epochs.
- Каталог `/tmp` в RAM, достаточный для prediction inputs и artifacts активных
  attempts.
- Доступный Linux `/proc`, поддержка libc `prctl(PR_SET_PDEATHSIG)` и право
  отправлять сигналы группам процессов, принадлежащим worker-у.
- Для планирования CUDA — `nvidia-smi` со стабильным выводом UUID GPU. Каждый
  видимый GPU должен быть доступен пользователю сервиса.

Production-версии Python packages зафиксированы только в
[`requirements.txt`](../../requirements.txt). Окружение создаётся на целевом
хосте по инструкции
[`deployment/systemd.md`](../deployment/systemd.md). Команды этого руководства
выполняются из `/home/nerv/transformer` через `./.venv/bin/python`.

```bash
cd /home/nerv/transformer
```

Каждый subprocess обучения или прогнозирования запускается в отдельной группе
процессов. При старте recovery сравнивает сохранённые PID, process group, boot
ID и start ticks процесса до отправки сигнала прерванной группе. Если identity
нельзя доказать безопасно, запуск завершается ошибкой вместо риска отправить
сигнал процессу с повторно использованным PID.

## Границы хранения и источники истины

PostgreSQL является единственным долговечным источником истины для:

- jobs, attempts и переходов состояний;
- metadata inputs/outputs, idempotency records и output tickets;
- зарегистрированных generations training recovery и истории retry;
- metadata опубликованных моделей и owner-scoped aliases моделей;
- committed epoch metrics, metadata run-owned metrics artifacts и состояние
  OpenSearch outbox;
- API access tokens;
- текущей storage epoch runtime.

Filesystem runtime намеренно является временным:

```text
/tmp/transformer/
  service.lock
  storage-epoch
  cuda-quarantine.json
  spool/
    jobs/{jobId}/
      inputs/{ordinal}-{payloadId}-{uploadToken}.arrow  # только prediction
      attempts/{attempt}/
        stdout.log
        stderr.log
        outputs/{ordinal}.arrow
        checkpoint.pth
```

Fit inputs и восстанавливаемое состояние обучения постоянны, но остаются
внутренними:

```text
<project-root>/recovery/
  jobs/{jobId}/
    inputs/{ordinal}-{payloadId}-{uploadToken}.arrow
    checkpoints/{completedEpoch}.pth
```

Постоянную публичную identity получают только успешно обученные модели:

```text
<project-root>/models/
  {modelRef}/
    checkpoint.pth
    metadata.json
```

Успешный fit может независимо получить best-effort telemetry run:

```text
<project-root>/telemetry/
  {jobId}/
    metrics.jsonl
    run-summary.json
```

Файлы сначала записываются рядом с конечным расположением, синхронизируются
через fsync, атомарно переименовываются, после чего выполняется fsync каталога.
Recovery checkpoint становится видимым только после надёжной записи файла и
регистрации его generation. Epoch telemetry фиксируется отдельной best-effort
транзакцией PostgreSQL. При публикации модели checkpoint successful attempt и
metadata атомарно публикуются в `models/`; только после прикладного commit
service может собрать файлы в `telemetry/` и зарегистрировать отдельный
OpenSearch outbox. Ошибка telemetry не меняет model generation или terminal
state. Неуспешные и прерванные attempts не создают generation модели.

Один процесс владеет каталогами runtime и recovery через неблокирующие файлы
`service.lock`. V9 остаётся single-instance: PostgreSQL не превращает
in-memory worker queue или локальные хранилища в scheduler нескольких replicas.

### Потеря `/tmp`

`storage-epoch` идентифицирует текущее поколение filesystem runtime. После
потери `/tmp/transformer` следующий процесс создаёт новую epoch. Transformer
удаляет prediction jobs, их inputs/outputs/tickets и связанные idempotency
records, поскольку эти artifacts невозможно восстановить. Незавершённый fit с
inputs в `recovery/` остаётся авторитетным: прерванная attempt переходит в
`RETRYING` и возобновляется с последней зарегистрированной завершённой epoch.

Опубликованные модели, постоянные fit inputs/checkpoints и API access tokens не
связаны с runtime epoch. Потеря `recovery/` обрабатывается иначе: отсутствие или
повреждение зарегистрированного input либо checkpoint приводит к явной ошибке
recovery; Transformer не начинает fit незаметно с нулевой epoch. PostgreSQL
хранит только metadata, поэтому ни один filesystem нельзя восстановить из БД.

## Настройка PostgreSQL

Параметры БД читаются из `<project-root>/.env`. Значения, уже присутствующие в
окружении процесса, имеют приоритет. Обязательные параметры:

```dotenv
POSTGRES_HOST=10.20.30.10
POSTGRES_PORT=5432
POSTGRES_DB=transformer
POSTGRES_USER=transformer
POSTGRES_PASSWORD=replace-with-a-secret
```

Если `POSTGRES_PORT` не указан, используется `5432`. БД должна оставаться в
частной сети. Credentials нельзя фиксировать в репозитории или включать в
логи.

Transformer использует schema PostgreSQL `transformer` и никогда не применяет
migrations при запуске. Service и административные команды требуют schema на
текущем Alembic head. Проверку revisions, upgrade, compatibility с baseline и
rollback boundary описывает
[`руководство по управлению схемой PostgreSQL`](database-migrations.md).

PostgreSQL хранит состояние control plane, а не Arrow payload-ы и не локальный
cache. Transactions короткие. In-process FIFO получает быстрые notifications
после commit, а единый maintenance cycle периодически сверяет с PostgreSQL
`QUEUED` и `RETRYING` строки, чтобы восстановить потерянное уведомление. Idle
worker lanes БД не опрашивают.

## Токены доступа API

Bearer authentication обязательна для каждого Flight RPC, включая actions,
DoPut, GetFlightInfo и DoGet. До предоставления сервиса Consumer-у оператор
должен выпустить и безопасно передать credential. Выдачу, просмотр, ротацию,
отзыв и проверку полного lifecycle описывает
[`руководство по управлению API access tokens`](api-access-tokens.md).
Credential model, cache consistency и channel security описаны в
[`справочнике аутентификации`](../authentication.md).

## Настройка Flight service

Приоритет конфигурации сервиса от низшего к высшему:

1. встроенные defaults в `app/config.py`;
2. поддерживаемые переменные окружения `TRANSFORMER_*`;
3. явно переданные options `flight serve`.

CLI предоставляет только overrides endpoint и transport:

```text
--host
--port
--tls-cert-file
--tls-key-file
--tls-ca-file
--tls-require-client-cert
```

Следующие параметры сервиса задаются в `app/config.py`, а не через окружение:

| Параметр Python | По умолчанию | Назначение |
| --- | --- | --- |
| `HOST_DEFAULT` | `127.0.0.1` | Адрес прослушивания Flight |
| `PORT_DEFAULT` | `8815` | Порт Flight; значение `0` разрешено в тестах |
| `ACCESS_TOKEN_CACHE_MAX_ENTRIES` | `1024` | Максимум положительных token cache entries |
| `ACCESS_TOKEN_CACHE_TTL_SECONDS` | `60.0` | Окно revalidation и верхняя граница revoke latency |
| `CPU_WORKERS` | `2` | Число одновременных CPU worker lanes |
| `RETENTION_SECONDS` | `604800` | Срок хранения terminal jobs |

Каталог runtime формируется через
`os.path.join(tempfile.gettempdir(), PROJECT_NAME)`. В целевом окружении
systemd это `/tmp/transformer`.

Соответствующие переменные окружения `TRANSFORMER_*` не читаются. У TLS и mTLS
нет постоянных значений по умолчанию: они включаются только явно переданными
certificate options команды `flight serve`. Flight v9 определяет
`gpuCapacity` по работоспособным физическим GPU, обнаруженным при запуске; это
не параметр приложения.

Certificate и key должны задаваться вместе. Их полная пара включает TLS, а
отсутствие обоих options выбирает plaintext. `tls-require-client-cert` также
требует CA file. Bearer authentication остаётся обязательной во всех transport
modes.

### Квоты и целевые параметры interoperability

| Переменная окружения | По умолчанию | Назначение |
| --- | --- | --- |
| `TRANSFORMER_MAX_MESSAGE_BYTES` | `16777216` | Целевой лимит interoperability клиента |
| `TRANSFORMER_TARGET_BATCH_BYTES` | `8388608` | Рекомендуемый размер RecordBatch producer-а |
| `TRANSFORMER_MAX_BATCH_BYTES` | `16777216` | Прикладной лимит RecordBatch |
| `TRANSFORMER_MAX_PAYLOAD_BYTES` | `536870912` | Лимит логического DoPut и сохранённого IPC-файла |
| `TRANSFORMER_MAX_ROWS_PER_PAYLOAD` | `2000000` | Число строк в одном DoPut |
| `TRANSFORMER_MAX_PAYLOADS_PER_JOB` | `100000` | Число логических payload-ов в job |
| `TRANSFORMER_MAX_JOB_BYTES` | `68719476736` | Общий объём committed inputs одной job |
| `TRANSFORMER_MAX_ACTIVE_JOBS_PER_SUBJECT` | `32` | Число non-terminal jobs на subject |

Проверяемый порядок:
`targetBatchBytes <= maxBatchBytes <= maxMessageBytes <= maxPayloadBytes`.
Consumer должен получать фактические значения через capabilities, а не
копировать defaults.

### Политика lifecycle

| Переменная окружения | По умолчанию | Назначение |
| --- | --- | --- |
| `TRANSFORMER_TICKET_TTL_SECONDS` | `600` | Срок действия opaque DoGet ticket |
| `TRANSFORMER_CANCEL_GRACE_SECONDS` | `10.0` | Пауза между SIGTERM и SIGKILL |
| `TRANSFORMER_SHUTDOWN_DRAIN_SECONDS` | `30.0` | Ожидание workers до принудительной отмены |
| `TRANSFORMER_SUBPROCESS_TIMEOUT_SECONDS` | `86400.0` | Жёсткий срок выполнения CLI |
| `TRANSFORMER_MAINTENANCE_INTERVAL_SECONDS` | `60` | Интервал maintenance |
| `TRANSFORMER_INPUT_IDLE_TIMEOUT_SECONDS` | `900.0` | Подтверждённое ожидание contiguous input до `INPUT_TIMEOUT` |
| `TRANSFORMER_ACQUIRE_IDLE_GRACE_SECONDS` | `30.0` | Пауза после успешного takeover владения |

Все квоты, capacities и интервалы должны быть положительными. Нулевым может
быть только порт сервиса.

## Ручной запуск на переднем плане

Production-запуск определён только в
[`deployment/systemd.md`](../deployment/systemd.md). Для foreground diagnostics
локальный plaintext-процесс можно запустить так:

```bash
./.venv/bin/python ./app/main.py flight serve \
  --host=127.0.0.1 \
  --port=8815
```

Для TLS endpoint:

```bash
./.venv/bin/python ./app/main.py flight serve \
  --host=0.0.0.0 \
  --port=8815 \
  --tls-cert-file=/run/secrets/transformer/tls.crt \
  --tls-key-file=/run/secrets/transformer/tls.key
```

Если требуются client certificates, добавьте `--tls-ca-file` и
`--tls-require-client-cert`. Убедитесь, что SAN server certificate совпадает с
адресом, который использует Consumer.

Процесс пишет структурированные JSON logs в stderr. Supervisor deployment-а
должен передавать SIGTERM, ждать не меньше `shutdownDrainSeconds +
cancelGraceSeconds` до внешнего SIGKILL и никогда не запускать два процесса с
одним каталогом runtime. Целевой Fedora systemd unit, политика каталога runtime
и действия operator описаны в
[`deployment/systemd.md`](../deployment/systemd.md).

## Запуск и восстановление

При запуске сервис:

1. блокирует каталоги runtime и recovery;
2. проверяет, что schema PostgreSQL находится на текущем Alembic head;
3. идентифицирует и безопасно завершает точные surviving worker process groups;
4. после завершения всех surviving workers удаляет temporary files, оставшиеся
   после сбоя;
5. обрабатывает изменение runtime epoch, удаляя jobs без необходимых runtime
   artifacts;
6. переводит прерванные постоянные fits в `RETRYING`, помечает прерванные
   predictions как `FAILED / EXECUTION_INTERRUPTED`, а прерванные
   `CANCELLING` jobs завершает как `CANCELLED`;
7. удаляет незавершённые upload reservations и unpublished/orphan artifacts;
8. сверяет постоянные каталоги моделей с metadata PostgreSQL, сохраняя
   `AVAILABLE` и ожидающие удаления `DELETING` generations;
9. инвентаризирует доступные физические CUDA devices, не инициализируя CUDA в
   Flight-процессе;
10. создаёт пустой bounded cache-aside API tokens без preload и listener;
11. загружает `QUEUED` и `RETRYING` jobs в in-memory device queues, запускает
    workers и maintenance; дальнейшая сверка в maintenance cycle
    восстанавливает потерянные queue notifications;
12. начинает обслуживать Flight RPC.

Возобновляемый fit восстанавливает последний зарегистрированный в PostgreSQL
checkpoint global epoch. Если checkpoint отсутствует, обучение начинается с
нулевой epoch на тех же постоянных inputs и неизменяемой конфигурации job. Для
warm-start job исходные weights повторно проверяются по зафиксированным parent
reference и digest; training recovery при наличии затем восстанавливает уже
состояние нового job. До EOF незавершённая нулевая epoch намеренно повторяется
полностью.
`inputIdleTimeout` отсчитывается только после сообщения worker об ожидании
следующего contiguous ordinal; out-of-order commit не продлевает timeout.

## Отмена и остановка

Поведение при отмене job:

- `WAITING_INPUT`, `QUEUED` и `RETRYING` transactionally переходят в
  `CANCELLED`; открытый input переходит в `ABORTED`;
- `RUNNING` переходит в `CANCELLING`, затем всей группе процессов worker
  отправляется SIGTERM и, если требуется после заданной паузы, SIGKILL;
- после победы cancellation в финальной transaction prediction output или
  model не публикуются;
- terminal state и опубликованные artifacts побеждают, только если финальная
  publication была committed раньше.

При SIGINT/SIGTERM процесс закрывает границу claim очереди, помечает себя как
draining, прекращает приём RPC work, ожидает running work и отменяет оставшиеся
worker groups. Maintenance останавливается до закрытия pool соединений
PostgreSQL и освобождения runtime lock.

## Хранение и ошибки хранилища

Maintenance периодически удаляет истёкшие output tickets и подходящие terminal
jobs. Job сохраняется, пока с ней связан действующий ticket или недавний
idempotency record. Recovery inputs/checkpoints удаляются после перехода fit в
terminal state. Каталоги опубликованных моделей не удаляются вместе с job, а
необязательная ссылка на producing job очищается. Компактная owner-scoped
identity tombstone сохраняется, поэтому `jobId` нельзя использовать повторно,
а точный lost-create replay остаётся разрешимым.

Published model generation имеет независимый двухфазный hard-delete lifecycle;
во Flight v9 нет сетевого action для её удаления. Команды оператора,
наблюдение `DELETING`/`DELETED`, filesystem retry и archive boundary описывает
[`руководство по управлению опубликованными моделями`](published-models.md).
Запрос удаления блокируется не только незавершённым prediction, но и любым
незавершённым fit, использующим generation как strict или transfer parent.

Сервис не использует настроенный admission watermark свободного места. Health
возвращает текущий свободный объём runtime и recovery storage, но не выводит из
него readiness. Реальное заполнение filesystem возвращает стабильный
`DISK_FULL` и не публикует частичные artifacts.

## Работоспособность и наблюдаемость

Отдельного неаутентифицированного HTTP health endpoint нет. Используйте
аутентифицированный Flight action `transformer.v9.health`.

- `live=true` означает, что процесс отвечает на action.
- `ready=true` требует, чтобы сервис не находился в draining и health check
  PostgreSQL завершался успешно.
- Доступность GPU, число физических devices и число quarantined devices
  возвращаются независимо.

Логи сервиса — по одному JSON object на строку в stderr. Они охватывают
lifecycle сервиса, завершение RPC/actions, переходы jobs, commits inputs,
выполнение worker, publication, runtime resets, recovery и maintenance. Bearer
credentials, authorization headers, output tickets, paths filesystem и argv
worker не логируются.

Health response также содержит ограниченные агрегированные in-process metrics.
Metrics сбрасываются при перезапуске сервиса и не являются долговечным
источником учёта; timestamps PostgreSQL доступны для анализа инцидента, пока
соответствующая job принадлежит активной generation runtime.

Рекомендуемые alerts:

- `ready=false` или недоступность PostgreSQL;
- заполнение filesystem с `DISK_FULL`;
- рост постоянного recovery/model storage вне прогноза;
- ошибки worker lane, GPU OOM, quarantined devices или повторные ошибки/retry
  subprocess;
- recovery прерванного процесса или сброс storage epoch runtime;
- долгое ожидание в queue относительно настроенной CPU/GPU capacity.

## Стабильные ошибки и ограничения PyArrow

Сервис завершает RPC ошибкой, а не возвращает error result. Стабильные codes
включаются в безопасный текст ошибки и terminal status. Raw tracebacks, paths
filesystem, credentials и stderr subprocess не должны попадать клиентам.

В PyArrow 24 подтверждены два ограничения bindings:

1. Python `FlightServerBase` не может отправить точные gRPC
   `ALREADY_EXISTS`, `FAILED_PRECONDITION` или `RESOURCE_EXHAUSTED`; сервис
   сохраняет стабильный application code в безопасном тексте.
2. Python `FlightServerBase` не позволяет настроить жёсткий server limit для
   размера принимаемого сообщения. Лимиты batch, logical payload, rows и job
   по-прежнему контролируются приложением.

Подробности записаны в
[`flight-dependency-note.md`](../flight-dependency-note.md).

## Известные ограничения v9

- Один экземпляр сервиса Transformer с одним локальным runtime storage и одним
  постоянным recovery storage.
- Нет планирования replicas и автоматического failover.
- Нет `DoExchange` и `PollFlightInfo`.
- Не более 100000 logical payload-ов на job. Inputs возвращаются ограниченной
  revision pagination, а close передаёт только итоги постоянного размера и
  digest.
- Один DoPut соответствует одному semantic payload; chunking RecordBatch и
  разбиение payload-ов не определяют optimizer batches, shuffle windows или
  epochs.
- Одна predict job однократно загружает один checkpoint и формирует один output
  на каждый ordinal input.
- Fit recovery выполняется только на границе завершённой global epoch;
  незавершённая epoch повторяется.
- Одна fit attempt использует один GPU; одна job не распределяется между GPU.
- Подтверждённо потерянный GPU находится в quarantine до следующей загрузки
  Linux; обычный restart сервиса не возвращает его в pool.
- Потеря runtime storage делает prediction work недействительной, но не
  затрагивает fit с целыми постоянными recovery artifacts.
- Transformer владеет checkpoints; клиенты получают только opaque значения
  `modelRef`.
- Output tickets краткоживущие и не являются ссылками на модель.
- Наличие полной пары TLS certificate/key выбирает TLS; отсутствие обоих
  options выбирает plaintext.
- Interoperability Node → PyArrow и физическое поведение CUDA требуют отдельной
  проверки в целевом окружении.
