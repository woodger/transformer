# Changelog

Все заметные изменения в этом проекте документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
а проект следует [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Восстановлены административные команды
  `auth tokens issue`, `auth tokens list` и
  `auth tokens revoke <TOKEN_ID>` для API credentials формата
  `a.<base64url>` со сроком действия три календарных месяца. Все выпущенные
  tokens принадлежат owner subject `inventory`.
- Migration `0015` создаёт новую таблицу API tokens и `LISTEN/NOTIFY` trigger.
  PostgreSQL хранит только SHA-256 digest credential; исходный bearer
  показывается один раз при выпуске.
- Migration `0016` нормализует database, уже отмеченные revision `0015`, но
  сохранившие историческую raw-колонку `token`: существующие credentials
  переводятся в digests без изменения ID, subject, timestamps или revoke
  status, после чего raw bearer удаляется.
- Migration `0017` удаляет все прежние бессрочные API tokens и добавляет
  обязательный `expires_at`; удалённые credentials не восстанавливаются.
- Migration `0018` физически удаляет прежние revoked token rows, удаляет
  `revoked_at` и оставляет cache notification только для issue/revoke через
  `INSERT/DELETE`.

### Changed

- Встроенные local, Flight service, PostgreSQL и OpenSearch defaults снова
  собраны в едином `app/config.py`; runtime-specific parsing, validation,
  environment precedence и worker v7 contract не изменены.
- Flight authentication снова использует PostgreSQL-backed digest cache.
  Точный subject credential является `owner_subject`; выпуск обновляет cache,
  revoke физически удаляет token и также обновляет cache без перезапуска
  сервиса, а срок действия проверяется локально при каждом новом RPC.

### Removed

- Удалены Ory Hydra introspection, `HYDRA_ENDPOINT`, OAuth client
  administration и команды `auth clients create|list|delete`.

## [0.1.14] - 2026-08-21

### Added

- Добавлена команда `models list --deleted`, которая показывает ожидающие
  очистки модели и минимальный audit archive с точным `deleted_at`.

### Changed

- Flight transport authentication переведена на introspection opaque OAuth
  access tokens Ory Hydra. Каждый новый RPC требует `active=true`,
  `token_type=Bearer`, audience `transformer` и scope
  `transformer:invoke`; точный `client_id` является owner identity. Flight v5
  и persisted job/model contracts не изменены. Runtime принимает базовый
  Hydra Admin API URL через `HYDRA_ENDPOINT`; introspection path зафиксирован
  в adapter. Решения авторизации кэшируются в ограниченном локальном
  для процесса LRU: успех до 15 секунд и `exp`, окончательный отказ на
  2 секунды; одновременные промахи объединяются в одну introspection.
- Штатное удаление модели сохраняет безопасный промежуточный state
  `DELETING`, но после удаления каталога физически удаляет строку модели из
  основной таблицы. Identity и timestamps переносятся в отдельный
  `deleted_models`; checkpoint metadata и ML contracts не сохраняются.
- Allocation generation учитывает рабочие модели и audit archive, поэтому
  номер остаётся монотонным после hard delete.
- Обычный `models list` показывает только `AVAILABLE`; модели в `DELETING`
  доступны через явный фильтр `--deleted`.

### Removed

- Удалены локальная выдача/отзыв API tokens, PostgreSQL token cache и fallback
  на credentials формата `a.<base64url>`. Необратимая migration `0014` удаляет
  таблицу `api_access_tokens` и её `LISTEN/NOTIFY` trigger.
- Удалены полные tombstones из основной таблицы `models` и её поле
  `deleted_at`. Необратимая migration `0012` уже очистила прежние строки
  `DELETED`; migration `0013` создаёт минимальный audit archive только для
  последующих удалений. Run-owned telemetry и terminal jobs сохраняют
  собственную retention policy.

## [0.1.13] - 2026-08-20

### Changed

- Flight v5 стал единственным публичным remote contract без v4 compatibility
  surface. Target identity использует только PascalCase-имена `MeanReturn`,
  `SigmaReturn`, `ProbTP`, `ProbSL`, `VolatilityNext` и `HittingProbTP`;
  повышены semantic IDs, checkpoint/recovery formats и worker contract v7.
- `dataContract` повышен до `inventory.learning-dataset` version 2 и получил
  обязательный Inventory-owned `profile`. Transformer сохраняет, сравнивает и
  возвращает весь документ без интерпретации; `model.describe` не дублирует
  нормативный порядок target отдельным списком.
- Epoch telemetry использует структурную пару `target.index`/`target.name` и
  generic metric names. Текущие artifact/point contracts — v3, outbox
  projection — `inventory.metrics.v4`, OpenSearch epoch index —
  `metrics-points-v3`.
- Training telemetry приведена к политике best effort: recovery checkpoint и
  model generation фиксируются независимо от epoch metrics, metrics artifacts
  и OpenSearch outbox. Сбой или отсутствие telemetry больше не меняет результат
  fit и не блокирует удаление модели.
- OpenSearch publisher отключается при ошибочной конфигурации, использует
  конечный retry budget и bounded admission для outbox; terminal delivery
  entries очищаются по retention policy.
- Поля telemetry во внутреннем worker v7 стали необязательными. Core checkpoint
  и fit result остаются строгими.
- Training state, compact job progress и telemetry разделены на независимых
  владельцев. Worker observations перенесены в `app/worker/telemetry`, а
  service records, artifacts, PostgreSQL repository и publisher — в явный
  telemetry slice; domain и общий ledger больше не содержат metrics API.
- Durable telemetry теперь принадлежит fit run (`jobId`) и хранится в
  отдельном `telemetry/<jobId>/`; удаление и список моделей не читают outbox.
  Миграция 0010 удаляет прежние model-owned metadata/outbox и снимает
  зависимость от model lifecycle; committed epoch intervals сохраняются.
- Удалены Consumer-owned statistics исходных training targets, неиспользуемый
  attempt-local metrics-файл и OpenSearch projection `metrics-artifacts-v2`.
  Текущая централизованная проекция содержит только epoch points и terminal run
  summary.
- Pytest сведён к двум явным режимам: самодостаточный основной набор и отдельно
  `gpu`. Временный PostgreSQL profile, скрытая fixture-маркировка и зависимые от
  внешней инфраструктуры тесты удалены до проектирования полноценного
  integration environment.
- Архитектурная проверка layout фиксирует только наличие текущих canonical
  locations и больше не проверяет исторические legacy paths.
- Основной pytest suite ускорен за счёт единого immutable import snapshot,
  устранения повторного worker inspect и несвязанной ML-инициализации. Worker
  проверяет command manifest до импорта executor и Torch.
- Необратимая migration `0011` требует предварительно удалить все модели и
  очищает несовместимые v4 jobs, recovery, idempotency, aliases и telemetry,
  сохраняя API tokens и tombstones удалённых model generations.

### Removed

- Удалены Flight v4, worker v6 и metrics v2 contracts; старые actions,
  descriptor paths, semantic aliases и checkpoint fallback отсутствуют.

### Fixed

- В `job.status` восстановлен checkpoint-aligned live progress fit:
  `epoch`, `step`, `loss_stage`, `loss`. Он фиксируется атомарно с recovery
  checkpoint, а расширенная training telemetry остаётся best effort и не
  попадает в прикладное состояние job.
- GPU test допускает штатные AMP scale backoffs на первых batch-ах, но требует
  восстановления `GradScaler`, применённого optimizer update и фактического
  изменения параметров модели.

## [0.1.12] - 2026-08-16

### Added

- Успешный fit публикует model-owned `run-summary.json` v2 с lifecycle
  durations, количеством attempts/recoveries, input counters и статистикой
  всех шести target-ов по принятому immutable dataset.
- Epoch telemetry различает завершённые training batches, применённые и
  пропущенные optimizer updates, AMP overflow и finite/non-finite gradients;
  для конечных pre-clip gradient norms публикуются mean, max и P95.
- Внутренний worker process contract повышен до v6. Новые metrics contracts
  используют проекцию `inventory.metrics.v3` и обычные индексы
  `metrics-points-v2`, `metrics-artifacts-v2` и `metrics-runs-v2`.

### Changed

- OpenSearch publisher поддерживает trusted-LAN HTTP как без authentication,
  так и с полной парой Basic Auth credentials; CA для HTTP запрещён. Оба
  metrics index template задают `number_of_replicas: 0` для текущего
  single-node deployment. Строгий HTTPS-профиль сохранён.
- PostgreSQL сохраняет lifecycle/recovery state и checkpoint-aligned epoch
  metrics как источник истины; OpenSearch остаётся post-commit аналитической
  проекцией и не участвует в результате fit.
- Нефинитная gradient norm одного batch больше не уничтожает статистику
  остальных batch-ей epoch. `globalTrainingStep` по-прежнему означает число
  завершённых training batches.
- Механизмы чтения экспериментальных metrics v1 и маршрутизация прежних
  записей outbox удалены; текущий runtime поддерживает только артефакты v2 и
  проекцию `inventory.metrics.v3`.

### Fixed

- Штатное удаление модели теперь очищает metadata model-owned
  `run-summary.json` вместе с checkpoint, epoch metrics и outbox.
- Conflict verification использует поддерживаемую OpenSearch форму `_mget`
  `docs` с per-document `_source`; повторная доставка идентичных metrics
  documents больше не блокируется ответом HTTP 400.

## [0.1.11] - 2026-08-16

### Added

- Добавлены штатные `models list` и `models delete`: удаление опубликованной
  generation проходит через durable `DELETING`/`DELETED` lifecycle, блокируется
  активными prediction jobs и сохраняет identity tombstone без alias fallback.
- Добавлена централизованная training telemetry: committed epoch metrics
  сохраняются вместе с recovery checkpoint, успешная модель получает
  immutable `metrics.jsonl`, а PostgreSQL outbox доставляет детерминированные
  `inventory.metrics.v1` points и artifact metadata в OpenSearch вне
  критического пути fit.
- Добавлены strict JSON Schemas и OpenSearch index templates для metrics v1,
  cross-language golden `eventId`, HTTPS publisher с bounded Bulk create,
  conflict integrity check, retry/backoff и health gauges backlog.
- Добавлены политика Python types и tensor runtime contracts и поэтапный
  Pyright strict baseline для admin/CLI, contracts, service domain/application,
  worker и типизированных outbound process/artifact boundaries.
- Pytest tests размечаются по внешним ресурсам `gpu` и `postgres`; быстрый
  CPU-набор не требует CUDA или PostgreSQL.

### Changed

- Внутренний worker process contract повышен до v4: checkpoint event атомарно
  связывает recovery generation с полной метрикой завершённой global epoch.
  Публичный Flight v4 не изменён.
- Структурная валидация Flight action requests и DoPut metadata переведена на
  нормативные JSON Schema Draft 2020-12 через `jsonschema`; Python ingress
  оставляет только семантические инварианты и mapping в типизированные DTO.
- Структура Python-кода приведена к одному каноническому пути на
  ответственность: local CLI вынесен в `app/local`, service outbound adapters
  названы по capabilities, PostgreSQL ledger и application DTO сгруппированы,
  worker checkpoints и batching получили собственных владельцев.
- CLI parser разделён на command-group parsers, formatting и options; общий
  `app/config.py` удалён, а defaults размещены у local/service/worker owners.
- Pytest suite разделён на `unit`, `contract`, `integration` и `architecture`
  с общими fixtures в `tests/support`; архитектурные тесты запрещают возврат
  прежних compatibility paths.
- Model input явно проверяет форму `[batch, sequence, features]`, immutable
  `seq_len`/`feature_dim` и `float32`; mask semantics и семь внутренних heads
  закреплены в типизированных сигнатурах и docstrings.
- В orchestration и boundary code безымянные `X`/`Y`/`x`/`pred` заменены на
  `features`, `targets` и `predictions`; wire fields и CLI options сохранены.
- Pyright strict scope расширен на Flight ingress, PostgreSQL adapters и
  service bootstrap; dynamic PyArrow/ORM boundaries локализованы и сразу
  переводятся в типизированные records, mappings и JSON-документы.
- Ruff `ANN` закрепляет явные типы параметров и результатов production-кода;
  исключения ограничены tests и двумя dynamic boundaries.
- `Trainer` принимает единый immutable `TrainConfig`; дублирующие параметры
  optimizer, schedule, AMP, selection и seed удалены из его конструктора.
- Удалены compatibility facades `app.flight`, `app.database` и прежние
  ML-пакеты: service и worker теперь имеют по одному каноническому import path.
- Команды проверки изменений собраны в одной политике тестирования; удалены
  ненормативные ML-заметки с устаревшими конфигурационными рекомендациями.

### Fixed

- ORM-модели сохраняют recursive `JsonValue` в runtime namespace, поэтому
  SQLAlchemy корректно разрешает postponed `Mapped[JsonObject]` annotations
  при запуске Alembic на Python 3.14.
- Metrics v1 использует два обычных versioned OpenSearch index вместо data
  streams: повторный Bulk `create` сохраняет глобальный конфликт `_id`, а
  проверка `documentSha256` выполняется пакетным `_mget` по concrete index.

## [0.1.10] - 2026-08-12

### Added

- Добавлен target-aligned ML-контракт Flight v4: шесть public predictions
  совпадают с target Inventory по индексу, имеют прямой supervision и
  проверяются по finite/range-инвариантам до публикации.
- Добавлены каноническая objective configuration, cross-language JSON fixture
  и `objectiveConfigSha256`; checkpoint selection учитывает только глобально
  агрегированные `L0…L5` полной epoch максимального stage.

### Changed

- Flight v4, worker process v3, `transformer-checkpoint-v3` и
  `transformer-training-recovery-v3` образуют одну breaking-границу. Private
  Gaussian scale отделён от public `sigmaReturn`; probability logits не
  пересекают prediction boundary.
- Миграция `0006` удаляет jobs, idempotency и recovery state прежнего
  objective, сохраняя access tokens, model identities и aliases. Прежние
  модели требуют полного переобучения и нового `modelRef`.
- Service boundary доведена до полноценной Clean Architecture: application
  commands/queries используют нейтральные DTO и capability ports, Flight
  presentation находится во inbound adapter, а PostgreSQL transactions,
  idempotency и projection mapping — в outbound adapters. Composition roots
  отдельно собирают job control и data plane; worker остаётся осознанным
  изолированным runtime-исключением без дополнительного слоения.

### Fixed

- `objectiveConfigSha256` теперь вычисляется по RFC 8785/JCS, поэтому Python и
  Node.js одинаково канонизируют JSON numbers, включая пары `1.0`/`1` и
  `0.0`/`0`. Golden digest и cross-language contract test обновлены.
- Flight v4 использует единое canonical определение Arrow physical
  schema для ingress, fingerprint, fixtures, durable replay и worker output.
  Неканоническая nested nullability отклоняется с `INVALID_ARGUMENT` до
  reservation, durable commit и запуска worker.
- Потеря in-process уведомления после durable commit больше не оставляет
  `QUEUED`/`RETRYING` job без исполнения: единый maintenance cycle периодически
  сверяет локальные очереди с PostgreSQL, не опрашивая БД из idle worker lanes.
- Ожидание следующего contiguous input корректно регистрируется при уже
  committed out-of-order payload; idle timeout не теряется из-за устаревшего
  `inputRevision` worker-а.
- Model lifecycle errors сохраняют стабильные коды `NOT_FOUND`,
  `MODEL_UNAVAILABLE` и `MODEL_CORRUPT` на service и worker boundaries.
- Пустой fit возвращает `EMPTY_INPUT` до повторной проверки доступности CUDA.
- Миграция `0005` приводит длину `models.model_ref` и
  `model_aliases.model_ref` к ORM-контракту `VARCHAR(128)`.

### Removed

- Удалены Flight v3 actions/descriptors/fixtures и worker v2 contract; runtime
  не содержит v3 compatibility surface или fallback.
- Удалено чтение прежних checkpoint formats и raw `state_dict` локальным CLI.

## [0.1.9] - 2026-08-11

### Added

- Добавлен durable streaming Transformer Flight v3: client-generated `jobId`,
  durable identity/tombstone, cross-system fencing через `job.acquire`,
  revision pagination inputs/outputs, `input.close` и `model.describe`.
- Добавлены worker process contract v2 с bounded control channel для committed
  inputs и EOF, а также нормативный ADR атомарного перехода без v2
  compatibility surface.
- Добавлена локальная CLI-команда `gmark` для CUDA stress test на синтетических
  production training steps, включая AMP, integrity checks, метрики
  `nvidia-smi`, опциональный VRAM ballast и температурный cutoff.
- В руководство начала работы добавлены выпуск, просмотр и отзыв API-токенов
  без перезапуска Flight service.

### Changed

- Breaking migration `0004` удаляет v2 jobs, inputs, attempts, tickets,
  idempotency и recovery state, сохраняя API access tokens и legacy model rows
  как uncertified; автоматического downgrade нет.
- Fit начинает epoch 0 после первого непустого durable payload до EOF;
  `input.state` и `execution.state` разделены, последующие эпохи перечитывают
  закрытый immutable dataset, а model/output публикуются только после EOF.
- Public Arrow schemas закреплены как `FixedSizeList<Float32>`, ML identity —
  через `dataContractSha256`; legacy models требуют явной сертификации или
  переобучения для v3.
- Training runtime вычисляет missing mask по одному разу для NaN/token
  diagnostics и model context и не синхронизирует CUDA ради all-missing branch;
  progress/JSONL публикует host-side durations input pipeline, missing
  statistics, CPU-to-device transfer и training step без добавления CUDA
  synchronization.
- Committed Flight inputs проходят полную value validation один раз до durable
  commit; worker проверяет immutable receipt и digest один раз за attempt, а
  закрытые эпохи используют physical-schema fast replay с bounded prefetch
  одного следующего CPU batch. CUDA scalar metrics одного training step
  материализуются одной компактной передачей на CPU.

### Fixed

- CUDA inventory subprocess запускается из project root и больше не зависит от
  рабочего каталога systemd service.
- `--version` сделан статическим: control-plane CLI и основной Flight service
  process больше не импортируют Torch/CUDA и не открывают NVIDIA device handles;
  runtime probing остаётся в изолированном worker `inspect` subprocess.
- `gmark --use-amp` выполняет bounded scale backoff при warm-up overflow и не
  публикует статическое значение занятой VRAM как результат stress test.

### Removed

- Удалены Flight v2 dispatcher/actions/descriptors и worker v1 contract;
  `job.seal` и `job.start` не имеют aliases или runtime fallback.

## [0.1.8] - 2026-08-09

### Added

- Добавлены реальные конкурентные PostgreSQL regression tests для
  exact/conflicting idempotency, `cancel` против result publication и
  согласованного status snapshot во время publication.
- Добавлен единый lock-файл всех зависимостей project `.venv`.
- Добавлена единая политика Python runtime: `.venv` создаётся от системного
  `/usr/bin/python3`, а application dependencies устанавливаются только из
  `requirements.txt` project interpreter.
- Добавлен независимый worker process contract v1 с immutable manifests,
  bounded NDJSON events, capability inspection и equality fence `attemptId`.
- Добавлена migration `0003`, создающая UUID execution identity для каждой
  PostgreSQL attempt.

### Changed

- Конфигурация Ruff, pytest и Alembic объединена в `pyproject.toml`; локальные
  tool caches складываются в единую игнорируемую директорию `.cache/`.
- Проект разделён на process-specific Clean Architecture boundaries:
  `app/service`, `app/worker` и `app/admin` имеют собственные composition
  roots, а публичный Flight v2 и внутренний worker v1 contracts находятся в
  `app/contracts/`.
- PostgreSQL adapter, ORM и Alembic перенесены в
  `app/service/adapters/outbound/postgres/`; прежние `app/database` и
  `app/flight` import paths временно сохранены совместимыми фасадами.
- ML model, training, Arrow tensor path и checkpoint runtime перенесены под
  `app/worker/`. Service обнаруживает Torch/CUDA только через
  `transformer-worker inspect` и не импортирует worker implementation.
- Каждый claimed attempt выполняется отдельным worker v1 subprocess через
  immutable manifest; progress, recovery checkpoints и terminal result
  принимаются как identity-bound NDJSON events, а публикация остаётся за
  service-процессом.
- Job lifecycle разделён на application commands/query, persistence boundary
  возвращает immutable records и согласованный `StatusSnapshot`.
- README сокращён до quick start и навигации; подробные CLI, Arrow stream,
  training runtime и deployment contracts вынесены в профильные документы.

### Fixed

- Инструкция запуска через systemd приведена к проверенной конфигурации Fedora:
  project запускается из `/home/nerv/transformer`, unit напрямую использует
  interpreter из `.venv`, а SELinux назначает тип `bin_t` только Python-ссылкам
  виртуального окружения.
- Seal manifest теперь проверяется под той же PostgreSQL row lock, что и
  переход job в `SEALED`; конкурентный `DoPut` больше не может оставить
  committed input за пределами sealed manifest.
- Prediction выполняет model forward ограниченными `batch_size` порциями,
  поэтому размер transport payload больше не определяет пиковый объём CUDA
  activations.
- Arrow input и prediction output преобразуются через векторные Arrow/NumPy
  buffers без Python list/scalar materialization.
- Все worker mutations, включая cancel и повтор `RETRYING`, атомарно проверяют
  текущий `attemptId`; запоздалый executor не может изменить новый attempt.
- Worker v1 сохраняет прежний формат fit-метрик и диагностические сообщения
  fit/predict в `stderr`, не смешивая их с NDJSON event stream в `stdout`.

## [0.1.7] - 2026-07-25

### Added

- Добавлен Ruff с единым минимальным baseline для Python lint checks.
- Добавлен `jsonschema` для полноценной Draft 2020-12 проверки нормативных
  Flight JSON Schemas и golden fixtures в contract tests.
- Добавлены persistent `recovery/` store и PostgreSQL metadata для fit inputs,
  completed-global-epoch checkpoints, попыток возобновления и состояния
  `RETRYING`.
- Добавлен boot-scoped physical CUDA inventory: отдельная lane на каждый
  доступный GPU, обязательная привязка subprocess через
  `CUDA_VISIBLE_DEVICES` и quarantine потерянного устройства до следующего
  Linux boot.

### Changed

- Ruff baseline расширен проверками потенциальных ошибок, безопасной
  модернизации Python-кода, порядка `__all__` и регулярных выражений в
  `pytest.raises`.
- Ruff также проверяет единый порядок Python import-блоков.
- Runtime storage вычисляется из системной temporary directory и технического
  имени проекта. Listen endpoint, plaintext policy, worker capacity и retention
  перенесены из `TRANSFORMER_*` в `app/config.py`;
  TLS/mTLS включается только явными параметрами `flight serve`.
- Flight contract переведён на breaking v2: добавлены recovery status,
  dynamic CUDA capacity/quarantine и раздельное состояние runtime/recovery
  storage.
- CUDA capacity удалена из application config и вычисляется по фактически
  доступным physical GPU.
- Fit checkpoints теперь сохраняют model, optimizer, AMP scaler,
  early-stopping/best-selection progress и RNG state. После service/host
  interruption обучение продолжается с последней полностью завершённой
  глобальной эпохи; подтверждённая потеря GPU переносит новую попытку на
  здоровое устройство.
- Prediction job после подтверждённой потери GPU также создаёт новую попытку
  на здоровом устройстве; ordinary subprocess failures и CUDA OOM остаются
  terminal.
- Публичное название проекта изменено на `Transformer Arrow Flight service`;
  технические имя CLI и `transformer-flight` не менялись.

### Fixed

- Worker scheduler возвращает job в локальную FIFO после временного
  PostgreSQL `SKIP LOCKED`, если durable state остаётся `QUEUED` или
  `RETRYING`; job больше не зависает без повторного claim.

### Removed

- Удалена runtime-совместимость Flight v1: прежние actions, descriptors,
  jobs, tickets и idempotency responses не переносятся через migration `0002`.
  Published models, aliases и API access tokens сохраняются.
- Удалён преждевременный `DISK_MIN_FREE_BYTES` и связанный proactive admission
  watermark. Фактические `ENOSPC`/`EDQUOT` по-прежнему возвращают стабильный
  `DISK_FULL`, а health сохраняет наблюдаемое свободное место хранилищ.

## [0.1.6] - 2026-07-23

### Added

- Добавлены адаптированные для Python/Transformer политики разработки:
  архитектурные границы, тестирование, документация, зависимости, именование,
  lifecycle кода и runtime entrypoints.
- Добавлены ADR внутренних границ Flight control plane, AST-проверки import
  graph и immutable records для execution/persistence boundaries.

### Changed

- Flight control plane разделён на durable upload session, trusted execution
  plan, subprocess runner, artifact publisher, attempt executor и
  Coordinator/Ledger use-case slices. Публичные фасады, Flight v1 contract,
  PostgreSQL schema и spool layout сохранены.
- Тесты приведены к проверке наблюдаемого поведения; PostgreSQL integration
  tests теперь требуют отдельную базу с именем `transformer_test*`.
- Руководство по systemd сокращено до последовательного ручного
  production-развёртывания без deployment automation.

### Fixed

- Flight fit формирует job-wide shuffle windows и optimizer batches независимо
  от границ transport payload, поэтому изменение размера упаковки больше не
  меняет training trajectory.

## [0.1.5] - 2026-07-23

### Added

- Добавлено руководство по ручному production-развёртыванию на Fedora/systemd
  259: готовые unit и tmpfiles configuration, migrations, CUDA validation,
  restart и recovery semantics.
- Добавлен корневой `.env.example` с документированными настройками PostgreSQL,
  Flight transport, runtime storage, worker capacity и retention.

## [0.1.4] - 2026-07-23

### Added

- Добавлены PostgreSQL control plane на SQLAlchemy 2, Alembic migrations и
  команды `db migrations status|apply|rollback`. Параметры подключения читаются
  из `.env`/окружения через `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`,
  `POSTGRES_USER` и `POSTGRES_PASSWORD`.
- Добавлены команды `auth tokens issue|list|revoke`. API tokens формата
  `a.<base64url>` хранятся в PostgreSQL, а Flight middleware проверяет их digest
  по in-memory cache, обновляемому через `LISTEN/NOTIFY`.

### Changed

- PostgreSQL стал единственным durable source of truth для jobs, attempts,
  idempotency, output tickets, model metadata и access tokens. Worker lanes
  получают задачи через in-memory FIFO и не опрашивают базу данных в idle loop.
- Незавершённые Arrow payloads и attempt artifacts перенесены в
  `/tmp/transformer`. Storage epoch не позволяет продолжить job после потери
  runtime filesystem: все связанные jobs отклоняются и удаляются, включая
  ранее завершённые runtime outputs.
- Только успешно обученные модели атомарно публикуются в persistent `models/`;
  их `modelRef` и access tokens переживают потерю `/tmp`.
- Command-specific help больше не показывает технический CLI default `None`:
  optional outputs, checkpoint-derived model parameters и Flight overrides
  описывают реальное fallback-поведение, а `--host` и `--port` показывают
  встроенные значения `127.0.0.1` и `8815`.
- `flight serve --help` теперь описывает назначение команды и явно фиксирует
  зависимости transport, TLS, mTLS и authentication parameters.
- Leaf command help больше не повторяет строку `-h, --help`; сам help-флаг
  остаётся доступным и документируется в глобальном `Usage`.
- Environment namespace конфигурации сервиса сокращён с
  `TRANSFORMER_FLIGHT_*` до `TRANSFORMER_*`; прежний namespace больше не
  поддерживается.

### Removed

- `flight serve` больше не принимает `--config` и `--state-dir`; загрузка
  service configuration из JSON-файла удалена.
- Удалены `--profile`, `TRANSFORMER_PROFILE` и поле `profile` service config.
  Plaintext transport теперь включается только через `--allow-plaintext` или
  `TRANSFORMER_ALLOW_PLAINTEXT=true` без дополнительных host restrictions.
- Удалены token JSON file и его CLI/environment configuration. Access tokens
  управляются только через PostgreSQL-backed команды `auth tokens`.

## [0.1.3] - 2026-07-22

### Changed

- CLI-параметр адреса Flight service переименован из `--bind-host` в `--host`,
  JSON-поле — из `bindHost` в `host`, а environment variable — из
  `TRANSFORMER_FLIGHT_BIND_HOST` в `TRANSFORMER_FLIGHT_HOST`; прежние имена
  больше не поддерживаются. Внутренняя конфигурация также использует единую
  пару `host` / `port`.
- Глобальный CLI help сокращён до global options и списка команд; примеры
  перенесены в применимые command-specific help, а для `--version` добавлена
  короткая форма `-v`.
- Команда запуска Flight service переименована из `serve-flight` в составную
  `flight serve`; прежнее имя больше не поддерживается.

### Fixed

- Flight fit теперь выполняет настроенное число эпох над всем sealed input job:
  внутри каждой эпохи durable payloads читаются по ordinal, а loss schedule,
  optimizer, checkpoint selection и early stopping больше не перезапускаются
  на границах транспортных payloads.

## [0.1.2] - 2026-07-19

### Changed

- Упрощены инструкции по установке PyTorch, NumPy, PyArrow и pytest на
  хосте; виртуальное окружение больше не представлено как обязательное.
- Главный README сфокусирован на настройке и запуске Transformer без
  consumer-specific сценариев.

### Removed

- Удалён временный `requirements.txt`; базовая установка зависимостей
  описана непосредственно в README.

### Fixed

- Flight v1 status для `CANCELLED` job теперь возвращает `error: null`, как
  требует нормативная JSON Schema. Старые durable-записи с cancellation error
  также сериализуются в wire-valid status без удаления state directory.

## [0.1.1] - 2026-07-19

### Added

- Добавлен single-instance Arrow Flight v1 job service для remote fit и predict:
  durable database ledger, filesystem spool, очереди, subprocess workers,
  cancellation, recovery, retention и operational observability.
- Добавлен нормативный Flight contract v1 с JSON Schemas, golden JSON/Arrow fixtures,
  ADR и production runbook.
- CLI переведён на настоящие mode-specific subcommands `fit`, `predict`,
  `fit-stream`, `predict-stream` и `plot-metrics` с ранней валидацией аргументов.
- Checkpoint v2 сохраняет model, training и data-schema configuration; чтение
  wrapped v1 и raw legacy checkpoints сохранено.
- Добавлены reproducibility controls `--seed` и `--deterministic`, атомарная
  публикация артефактов и ограничение размера stream frame.
- Добавлен monitor `ret_mae_skill` относительно zero-return baseline, выбор
  best checkpoint и настраиваемый minimum improvement.

### Changed

- Arrow contract уточняет типы, ширину, null/NaN/Infinity semantics, диапазоны
  target values и typed-empty input/output.
- Device policy унифицирована до `auto`, `cpu` и `cuda`; явно запрошенная CUDA
  больше не подменяется CPU.
- Streaming protocol фиксирует terminator/EOF semantics, binary-only stdout для
  `predict-stream` и один semantic frame на один logical payload.
- Поля metrics JSONL получили однозначные имена (`loss_ret`, `loss_prob`,
  `loss_ev`, `loss_vol`, `grad_norm` и расширенные timing/ratio names), а non-finite
  значения сериализуются как JSON `null`.
- Training console output разделён на одну строку конфигурации и компактную
  epoch summary.

### Removed

- Удалены legacy aliases `--amp` и `--per-week`; поддерживаемые имена —
  `--use-amp`, `--loss-schedule` и `--stage-size`.

### Fixed

- Gaussian NLL переведён на численно устойчивую формулу через
  `var = sigma² + 1e-6`.
- Loss schedule для step mode рассчитывает стадию по global optimizer step.
- Transformer использует последний valid timestep и безопасно обрабатывает
  полностью masked sequences.
- Early stopping корректно отслеживает улучшение monitor до прохождения
  baseline threshold.

### Security

- Flight RPC защищены bearer authentication, subject ownership, TLS/mTLS policy,
  expiring opaque tickets, quotas и запретом arbitrary paths/CLI arguments.

## [0.1.0] - 2026-05-20

### Added

- Первая версия PyTorch Transformer с file-based `fit` и `predict` для Apache Arrow
  datasets.
- Framed Arrow IPC режимы `fit-stream` и `predict-stream` с 8-byte big-endian
  length prefix и обработкой empty frames.
- Staged Bayesian trading loss, early stopping и mixed-precision training на CUDA.
- Training metrics в JSONL и построение SVG-графиков через `plot-metrics`.
- CLI help с описанием data/streaming contracts и команда `--version`.

[Unreleased]: https://github.com/woodger/transformer/compare/v0.1.14...HEAD
[0.1.14]: https://github.com/woodger/transformer/compare/v0.1.13...v0.1.14
[0.1.13]: https://github.com/woodger/transformer/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/woodger/transformer/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/woodger/transformer/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/woodger/transformer/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/woodger/transformer/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/woodger/transformer/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/woodger/transformer/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/woodger/transformer/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/woodger/transformer/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/woodger/transformer/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/woodger/transformer/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/woodger/transformer/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/woodger/transformer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/woodger/transformer/releases/tag/v0.1.0
