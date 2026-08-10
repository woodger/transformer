# Changelog

Все заметные изменения в этом проекте документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
а проект следует [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Добавлена локальная CLI-команда `gmark` для CUDA compute/VRAM stress test с
  проверкой целостности матричных вычислений, метриками `nvidia-smi` и
  температурным cutoff.

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

[Unreleased]: https://github.com/woodger/transformer/compare/v0.1.8...HEAD
[0.1.8]: https://github.com/woodger/transformer/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/woodger/transformer/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/woodger/transformer/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/woodger/transformer/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/woodger/transformer/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/woodger/transformer/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/woodger/transformer/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/woodger/transformer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/woodger/transformer/releases/tag/v0.1.0
